package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
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

// initK3 builds bs's Kimi-K3 set. Callers hold g.mu.
func (g *devTier) initK3(bs *blockScratch) error {
	p := &bs.p
	k := p.K3
	if k == nil {
		return nil
	}
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	R, d, s := bs.rows, p.NEmbd, k.Streams
	actWin := p.ActWin
	if actWin <= 0 {
		actWin = 32
	}
	x := &k3Scratch{streams: s, score: map[int]backend.Kernel{}, mixK: map[int]backend.Kernel{},
		post: map[int]backend.Kernel{}}
	var err error
	comp := func(dst *backend.Kernel, mk func() (*ir.Kernel, error)) {
		if err != nil {
			return
		}
		var kk *ir.Kernel
		if kk, err = mk(); err != nil {
			return
		}
		*dst, err = g.compile(kk)
	}
	al := func(dst *backend.Buf, n int) {
		if err == nil && n > 0 {
			*dst, err = g.dev.Alloc(n * 4)
		}
	}
	if s > 1 {
		for nb := 1; nb < s; nb++ {
			var sk, mk backend.Kernel
			comp(&sk, func() (*ir.Kernel, error) {
				return kernels.K3ResScores(d, R, nb, float32(p.RMSEps), k.Fault == nn.K3FaultRawScores)
			})
			comp(&mk, func() (*ir.Kernel, error) { return kernels.K3ResMix(d, R, nb) })
			x.score[nb], x.mixK[nb] = sk, mk
		}
		for push := 0; push < s; push++ {
			var pk backend.Kernel
			restart := push > 0 && k.Fault != nn.K3FaultNoRestart
			comp(&pk, func() (*ir.Kernel, error) { return kernels.K3Post(d, R, s, push, restart) })
			x.post[push] = pk
		}
		al(&x.scores, R*s)
		al(&x.mix, R*d)
	}
	if L := k.Latent; L != 0 {
		x.latent = L
		comp(&x.normL, func() (*ir.Kernel, error) { return kernels.RMSNormRows(L, R, float32(p.RMSEps), false, false) })
		comp(&x.quantL, func() (*ir.Kernel, error) { return quantizeRows(R, L, actWin) })
		al(&x.lat, R*L)
		al(&x.latAcc, R*L)
		al(&x.latN, R*L)
	}
	if p.MLA() {
		n := p.NHead * p.HeadDimV
		x.mgateW = n
		comp(&x.sigMG, func() (*ir.Kernel, error) { return kernels.SigmoidMul(R * n) })
		al(&x.mgate, R*n)
		al(&x.mgated, R*n)
	}
	return err
}

// free releases the set's kernels and buffers.
func (x *k3Scratch) free() {
	if x == nil {
		return
	}
	for _, ks := range []map[int]backend.Kernel{x.score, x.mixK, x.post} {
		for _, k := range ks {
			if k != nil {
				closeKern(k)
			}
		}
	}
	for _, k := range []backend.Kernel{x.normL, x.quantL, x.sigMG} {
		if k != nil {
			closeKern(k)
		}
	}
	for _, b := range []backend.Buf{x.scores, x.mix, x.lat, x.latAcc, x.latN, x.mgate, x.mgated} {
		if b != nil {
			b.Free()
		}
	}
}
