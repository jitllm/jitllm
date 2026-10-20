package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
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

// k3WhyNot names a Kimi-K3 shape this tier declines, or "".
func k3WhyNot(p *nn.LayerPlan) string {
	k := p.K3
	if k == nil {
		return ""
	}
	switch {
	case k.Streams < 1 || k.Push < 0 || k.Push >= max(k.Streams, 1) || k.BankAttn >= k.Streams ||
		k.BankFFN >= k.Streams:
		return fmt.Sprintf("Kimi-K3 residual attention over %d streams (banks %d and %d, checkpoint %d)",
			k.Streams, k.BankAttn, k.BankFFN, k.Push)
	case k.Latent != 0 && (p.NExpert == 0 || p.DenseMoE || p.ExpertWeightIn || p.MoEBias):
		return fmt.Sprintf("a latent mixture at %d beside a dense mixture, expert input weights or "+
			"mixture biases", k.Latent)
	case p.Parallel || p.AltUp != 0 || p.DS4 != nil:
		return "Kimi-K3's residual attention beside another residual structure"
	}
	return ""
}

// latentEps is the MLA latent norms' epsilon: the plan's own, or RMSEps.
func latentEps(p *nn.LayerPlan) float32 {
	if p.LatentNormEps != 0 {
		return float32(p.LatentNormEps)
	}
	return float32(p.RMSEps)
}

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

// k3Scratch is a scratch's Kimi-K3 buffers and kernels (bs.k3), built for its
// rows and every block the model has: a score and a mix kernel per bank
// count, and a placement per checkpoint stream.
type k3Scratch struct {
	streams        int
	score, mixK    map[int]backend.Kernel
	post           map[int]backend.Kernel
	scores, mix    backend.Buf
	lat, latAcc    backend.Buf
	latN           backend.Buf
	normL, quantL  backend.Kernel
	mgate, mgated  backend.Buf
	sigMG          backend.Kernel
	latent, mgateW int
}

// free releases the set's kernels and buffers.
func (x *k3Scratch) free(g *devShared) {
	if x == nil {
		return
	}
	for _, ks := range []map[int]backend.Kernel{x.score, x.mixK, x.post} {
		for _, k := range ks {
			if k != nil {
				g.closeKern(k)
			}
		}
	}
	for _, k := range []backend.Kernel{x.normL, x.quantL, x.sigMG} {
		if k != nil {
			g.closeKern(k)
		}
	}
	for _, b := range []backend.Buf{x.scores, x.mix, x.lat, x.latAcc, x.latN, x.mgate, x.mgated} {
		if b != nil {
			b.Free()
		}
	}
}

// k3Mix is the mix of nb checkpoints and the running residual over the R rows
// of src (every stream), scored by w: x.mix, or src itself (its stream 0 is
// the first R*n_embd floats) when nothing is banked.
func (g *devTier) k3Mix(lc *launcher, bs *blockScratch, l *layer, R int, src, w backend.Buf, nb int) backend.Buf {
	x := bs.k3
	if nb == 0 || l.k3.fault == nn.K3FaultNoBank {
		return src
	}
	lc.la(x.score[nb], R*(nb+1), src, w, x.scores)
	lc.la(x.mixK[nb], R*bs.p.NEmbd, src, x.scores, x.mix)
	g.K3Mixes++
	return x.mix
}

// k3AttnIn is block l's attention input before its norm: the mix over bs.x.
func (g *devTier) k3AttnIn(lc *launcher, bs *blockScratch, l *layer, R int) backend.Buf {
	return g.k3Mix(lc, bs, l, R, bs.x, l.k3.resA, l.k3.bankA)
}

// k3PostAttn writes every stream of bs.x2 from bs.x and the attention's output
// y: the running residual advanced (restarted on a checkpoint block) and the
// block's raw input banked at its checkpoint.
func (g *devTier) k3PostAttn(lc *launcher, bs *blockScratch, l *layer, R int, y backend.Buf) {
	lc.la(bs.k3.post[l.k3.push], bs.k3.streams*R*bs.p.NEmbd, bs.x, y, bs.x2)
}

// k3FFNIn is block l's FFN input before its norm: the mix over bs.x2.
func (g *devTier) k3FFNIn(lc *launcher, bs *blockScratch, l *layer, R int) backend.Buf {
	return g.k3Mix(lc, bs, l, R, bs.x2, l.k3.resF, l.k3.bankF)
}

// k3PostFFN writes every stream of bs.x from bs.x2 with the FFN's output y
// added to the running residual.
func (g *devTier) k3PostFFN(lc *launcher, bs *blockScratch, R int, y backend.Buf) {
	lc.la(bs.k3.post[0], bs.k3.streams*R*bs.p.NEmbd, bs.x2, y, bs.x)
}

// k3GateMLA takes the MLA output gate's projection of the block's normed
// input, quantized in bs.a/bs.ax, into x.mgate.
func (g *devTier) k3GateMLA(bs *blockScratch, l *layer, mvrun func(mv, *resident, backend.Buf)) {
	if l.mlaGate != nil {
		mvrun(l.mvMG, l.mlaGate, bs.k3.mgate)
	}
}

// k3GatedMLA is sigma(gate) times the R rows' attention output out, or out
// itself on a block with no gate.
func (g *devTier) k3GatedMLA(lc *launcher, bs *blockScratch, l *layer, R int, out backend.Buf) backend.Buf {
	if l.mlaGate == nil || l.k3 != nil && l.k3.fault == nn.K3FaultNoMLAGate {
		return out
	}
	x := bs.k3
	lc.la(x.sigMG, R*x.mgateW, x.mgate, out, x.mgated)
	return x.mgated
}
