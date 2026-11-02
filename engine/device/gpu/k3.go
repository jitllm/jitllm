package gpu

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
)

// Kimi-K3 on the device (nn.LayerPlan.K3; engine/model/k3.go is the graph and
// kernels/k3.go the kernels it needed).
//
// The residual is the running residual and its checkpoints, stream-major in
// bs.x as Gemma 3n's and DeepSeek V4's streams are, and bs.x2 is every stream
// too. A block's attention reads the mix of the checkpoints banked so far and
// the running residual (K3ResScores, K3ResMix into x.mix), normed as any
// block's input is; K3Post then writes every stream of bs.x2 -- the running
// residual plus the attention's output (or, on a checkpoint block, the output
// alone) and the block's raw input banked at its checkpoint stream. The FFN
// reads the mix over bs.x2 the same way, and K3Post writes bs.x back with the
// FFN's output added. With no checkpoint banked a mix is the running residual
// itself, read in place. The head's mix is the host's, as for every model with
// streams.
//
// The latent mixture is the generic one with its input and output moved: the
// router reads the normed row at n_embd, the experts the routed_down
// projection of it at the latent width (x.lat, quantized into bs.a/bs.ax), and
// their weighted sum lands in x.latAcc, which is normed and routed_up into
// bs.mvOut before the shared experts add theirs. The MLA output gate is one
// more projection of the block's normed input (x.mgate), taken before the
// query's quantization replaces it, and sigma(gate) times the attention's
// output goes into o_proj.

// k3Block is one Kimi-K3 block's mixes and vectors.
type k3Block struct {
	bankA, bankF, push, latent int
	fault                      nn.K3Fault
	// resA and resF are the two mixes' score vectors, rnorm the routed sum's
	// norm (nil where the block has none).
	resA, resF, rnorm backend.Buf
}

func (b *k3Block) bufs() []backend.Buf {
	if b == nil {
		return nil
	}
	return []backend.Buf{b.resA, b.resF, b.rnorm}
}

// prepK3 uploads block li's Kimi-K3 vectors. The MLA output gate and the
// latent projections go through prepLayer's weight list (slots 36..38).
func (g *devTier) prepK3(l *layer, li int, p *nn.LayerPlan, w *nn.LayerWeights,
	up func(*backend.Buf, []float32)) string {
	k := p.K3
	if k == nil {
		return ""
	}
	b := &k3Block{bankA: k.BankAttn, bankF: k.BankFFN, push: k.Push, latent: k.Latent, fault: k.Fault}
	if k.Streams > 1 {
		if len(w.ResAttn) != p.NEmbd || len(w.ResFFN) != p.NEmbd {
			return fmt.Sprintf("block %d: the residual scores are %d and %d wide, want %d", li,
				len(w.ResAttn), len(w.ResFFN), p.NEmbd)
		}
		up(&b.resA, w.ResAttn)
		up(&b.resF, w.ResFFN)
		l.auxBytes += uint64(2 * 4 * p.NEmbd)
	}
	if k.Latent != 0 && w.RoutedNorm != nil {
		if len(w.RoutedNorm) != k.Latent {
			return fmt.Sprintf("block %d: the routed sum's norm is %d wide, want %d", li, len(w.RoutedNorm),
				k.Latent)
		}
		up(&b.rnorm, w.RoutedNorm)
		l.auxBytes += uint64(4 * k.Latent)
	}
	l.k3 = b
	return ""
}
