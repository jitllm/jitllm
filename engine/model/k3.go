package model

import (
	"fmt"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// Kimi-K3's text model (jlm.ArchKimiK3), transcribed from Moonshot's
// modeling_kimi_linear.py, which llama.cpp's kimi-k3.cpp matches: Kimi-Linear's
// hybrid (KDA in delta.go, MLA with no rotary in mla.go) plus residual
// attention over a bank of block inputs, a latent mixture (k3MoE), situ
// (nn.ActSitu), an MLA output gate (mlaOutProject) and a full-rank KDA gate
// with an optional decay lower bound (nn.DeltaDecayBound32JIT).
//
// The bank crosses a device seam inside a token, so it is carried as residual
// STREAMS (Config.ResidW), stream-major: stream 0 is the running residual and
// stream 1+j checkpoint j. Every n_embd kernel that reads the residual reads
// stream 0 untouched; a device is handed every stream and hands every stream
// back, and the head's mix runs on the host (streamHead).
//
// Each piece's arithmetic: docs/engineering-history/model-correctness.md,
// "engine/model/k3.go".

// k3Fault breaks one piece of Kimi-K3's graph, for a gate's violation only.
type k3Fault int

// halfFault drops one half of a block that attends beside its mixer: the
// half computed and its output weighed by zero, so every tensor it reads
// stays bound as the container stores it.
type halfFault int

const (
	halfFaultNone halfFault = iota
	// halfFaultNoAttn leaves the block its mixer alone.
	halfFaultNoAttn
	// halfFaultNoMixer leaves the block its attention alone.
	halfFaultNoMixer
)

const (
	k3FaultNone k3Fault = iota
	// k3FaultNoBank mixes nothing: every sublayer reads the running residual,
	// as Kimi-Linear does, and the bank is never read.
	k3FaultNoBank
	// k3FaultNoRestart adds the attention output on a checkpoint block
	// instead of restarting the residual from it.
	k3FaultNoRestart
	// k3FaultRawScores scores the bank on the raw values instead of the
	// RMSNormed ones.
	k3FaultRawScores
	// k3FaultNoHeadMix reads the head from the running residual alone.
	k3FaultNoHeadMix
	// k3FaultNoLatentNorm skips the routed sum's RMSNorm in the latent.
	k3FaultNoLatentNorm
	// k3FaultNoMLAGate leaves the MLA output ungated.
	k3FaultNoMLAGate
	// k3FaultNoKDAGate leaves the linear blocks' output ungated.
	k3FaultNoKDAGate
)

// ResAttn reports Kimi-K3's residual attention: the bank of checkpoints and
// the mixes over it.
func (c *Config) ResAttn() bool { return c.AttnResBlock != 0 }

// resStreams is how many streams the residual attention carries: the
// running residual and one per checkpoint block, or zero without it.
func (c *Config) resStreams() int {
	if !c.ResAttn() {
		return 0
	}
	return 1 + c.resBankFinal()
}

// resBankFinal is how many checkpoints the bank holds after the last block:
// one per block whose index is a multiple of AttnResBlock.
func (c *Config) resBankFinal() int { return (c.NLayer + c.AttnResBlock - 1) / c.AttnResBlock }

// resBank is how many checkpoints block li's attention (ffn false) or FFN
// (ffn true) mixes: those banked before it, and for the FFN its own.
func (c *Config) resBank(li int, ffn bool) int {
	if ffn {
		return li/c.AttnResBlock + 1
	}
	return (li + c.AttnResBlock - 1) / c.AttnResBlock
}

// resCheckpoint reports whether block li banks its input, and the stream it
// banks into.
func (c *Config) resCheckpoint(li int) (stream int, ok bool) {
	if !c.ResAttn() || li%c.AttnResBlock != 0 {
		return 0, false
	}
	return 1 + li/c.AttnResBlock, true
}

// ExpWidth is the routed experts' input and output width: ExpertLatent on a
// latent mixture, n_embd everywhere else.
func (c *Config) ExpWidth() int {
	if c.ExpertLatent != 0 {
		return c.ExpertLatent
	}
	return c.NEmbd
}

// k3Config carries the container's Kimi-K3 fields into c and checks they
// describe a graph the engine runs. nact is how many activation flags the
// container set; situ is the only activation then.
func k3Config(out *Config, c *jlm.Config, nact int) error {
	out.AttnResBlock = int(c.AttnResBlock)
	out.ExpertLatent = int(c.ExpertLatent)
	out.KDALowerBound = c.KDALowerBound
	out.LatentNormEps = float64(c.LatentNormEps)
	if out.LatentNormEps == 0 {
		out.LatentNormEps = out.RMSEps
	}
	if c.SituBeta != 0 || c.SituLinearBeta != 0 {
		if nact != 0 || out.Act != nn.ActSiLU {
			return fmt.Errorf("model: %s: the container claims two activations", out.Arch)
		}
		out.Act = nn.ActSitu
	}
	switch {
	case out.KDALowerBound > 0:
		return fmt.Errorf("model: %s: a decay bound of %g; a bound is negative", out.Arch, out.KDALowerBound)
	case out.KDALowerBound != 0 && !out.ChanDecay():
		return fmt.Errorf("model: %s: a decay bound on a model with no Kimi Delta Attention", out.Arch)
	case out.ExpertLatent != 0 && (out.NExpert == 0 || out.ExpertLatent >= out.NEmbd || out.DenseMoE ||
		out.ExpertWeightIn):
		return fmt.Errorf("model: %s: a latent mixture at %d on %d experts at n_embd %d", out.Arch,
			out.ExpertLatent, out.NExpert, out.NEmbd)
	case out.AttnResBlock < 0 || out.ResAttn() && (out.PredStreams != 0 || out.DSV4() || out.Parallel ||
		out.NMTP != 0):
		return fmt.Errorf("model: %s: residual attention every %d blocks beside another residual "+
			"structure", out.Arch, out.AttnResBlock)
	}
	return nil
}

// f32Row is a vector as a one-row F32 matrix, which the generated matvec
// reads: a score vector dotted with every normed stream in one call.
func f32Row(v []float32) tensor {
	return tensor{typ: quant.F32, data: f32Bytes(v), rows: 1, k: len(v)}
}

// loadK3 reads block bi's Kimi-K3 weights beyond the common ones and checks
// their shapes: the residual attention's two score vectors, the MLA output
// gate, and the latent mixture's projections and norm.
func loadK3(l *layer, bi int32, cfg *Config, c *jlm.File, get func(jlm.Role, int32, int32) (tensor, error),
	vec func(jlm.Role, int32) ([]float32, error)) error {
	var err error
	if cfg.ResAttn() {
		if l.resAttn, err = vec(jlm.RoleAttnResScore, bi); err != nil {
			return err
		}
		if l.resFFN, err = vec(jlm.RoleFFNResScore, bi); err != nil {
			return err
		}
		if len(l.resAttn) != cfg.NEmbd || len(l.resFFN) != cfg.NEmbd {
			return fmt.Errorf("model: block %d's residual scores are %d and %d wide, want %d", bi,
				len(l.resAttn), len(l.resFFN), cfg.NEmbd)
		}
		l.resAttnT, l.resFFNT = f32Row(l.resAttn), f32Row(l.resFFN)
	}
	// The MLA output gate: RoleAttnGate in a full block of an MLA model,
	// NHead*HeadDimV rows over n_embd. (A linear block's RoleAttnGate is its
	// z projection, bound in build.)
	if cfg.MLA() && !cfg.LayerKind(int(bi)).Recurrent() && c.Has(jlm.RoleAttnGate, bi, -1) {
		if l.mlaGate, err = get(jlm.RoleAttnGate, bi, -1); err != nil {
			return err
		}
		if l.mlaGate.k != cfg.NEmbd || l.mlaGate.rows != cfg.NHead*cfg.HeadDimV {
			return fmt.Errorf("model: block %d's MLA output gate is %dx%d, want %dx%d", bi, l.mlaGate.k,
				l.mlaGate.rows, cfg.NEmbd, cfg.NHead*cfg.HeadDimV)
		}
	}
	if cfg.ExpertLatent != 0 && cfg.MoEAt(int(bi)) {
		if l.routedDown, err = get(jlm.RoleFFNRoutedDown, bi, -1); err != nil {
			return err
		}
		if l.routedUp, err = get(jlm.RoleFFNRoutedUp, bi, -1); err != nil {
			return err
		}
		if c.Has(jlm.RoleFFNRoutedNorm, bi, -1) {
			if l.routedNorm, err = vec(jlm.RoleFFNRoutedNorm, bi); err != nil {
				return err
			}
		}
		L := cfg.ExpertLatent
		if l.routedDown.k != cfg.NEmbd || l.routedDown.rows != L || l.routedUp.k != L ||
			l.routedUp.rows != cfg.NEmbd || (l.routedNorm != nil && len(l.routedNorm) != L) {
			return fmt.Errorf("model: block %d's latent mixture is %dx%d down, %dx%d up and a norm of %d; "+
				"want %d to %d and back", bi, l.routedDown.k, l.routedDown.rows, l.routedUp.k,
				l.routedUp.rows, len(l.routedNorm), cfg.NEmbd, L)
		}
	}
	return nil
}

// loadK3Head reads the head's residual-attention score vector.
func loadK3Head(m *Model, cfg *Config, vec func(jlm.Role, int32) ([]float32, error)) error {
	if !cfg.ResAttn() {
		return nil
	}
	var err error
	if m.resOut, err = vec(jlm.RoleOutputResScore, jlm.DenseBlock); err != nil {
		return err
	}
	if len(m.resOut) != cfg.NEmbd {
		return fmt.Errorf("model: the head's residual score is %d wide, want %d", len(m.resOut), cfg.NEmbd)
	}
	m.resOutT = f32Row(m.resOut)
	return nil
}

// k3Scratch is a instance's Kimi-K3 buffers, grown to the widest call so a warm
// decode token and a warm step allocate nothing.
type k3Scratch struct {
	rows int
	// per row: every stream a mix reads, normed (normed), their scores,
	// and the mix itself, the sublayer's input.
	normed, scores, mix []float32
	// ones is the unweighted norm's weight.
	ones []float32
	// per row of a latent mixture: the down-projected input, the routed sum
	// and the projection back.
	lat, latAcc, latUp []float32
}

// allocK3 sizes a instance's Kimi-K3 buffers for one row, and its streams.
func (s *Instance) allocK3() {
	c := s.m.Cfg
	if !c.ResAttn() && c.ExpertLatent == 0 {
		return
	}
	s.k3 = &k3Scratch{}
	if c.ResAttn() {
		s.xa = make([]float32, c.ResidW())
		s.x = s.xa[:c.NEmbd]
		s.k3.ones = make([]float32, c.NEmbd)
		for i := range s.k3.ones {
			s.k3.ones[i] = 1
		}
	}
	s.growK3(1)
}

// growK3 sizes the per-row buffers for n rows.
func (s *Instance) growK3(n int) {
	c, g := s.m.Cfg, s.k3
	if g == nil || g.rows >= n {
		return
	}
	g.rows = n
	d := c.NEmbd
	if c.ResAttn() {
		ns := c.resStreams()
		g.normed, g.scores, g.mix = make([]float32, n*ns*d), make([]float32, n*ns), make([]float32, n*d)
	}
	if L := c.ExpertLatent; L != 0 {
		g.lat, g.latAcc, g.latUp = make([]float32, n*L), make([]float32, n*L), make([]float32, n*d)
	}
}

// k3Expand empties the bank of n rows of x: a token's checkpoints are its
// own, and none exists before block 0.
func (s *Instance) k3Expand(x []float32, rows int) {
	c := s.m.Cfg
	for r := 0; r < rows; r++ {
		for k := 1; k < c.Streams(); k++ {
			clear(c.stream(x, k, rows, r))
		}
	}
}

// k3Mix is the residual attention's mix for n rows of x over the first nb
// checkpoints and the running residual (stream 0), scored by w: each row's
// sum_j p_j v_j with p the softmax of sum(w * rmsnorm(v_j)). It returns the
// rows' mixes, n_embd each, contiguous -- stream 0 itself when the bank is
// empty, where the softmax over one value is exactly one.
func (s *Instance) k3Mix(x []float32, n, nb int, w tensor) ([]float32, error) {
	c, g := s.m.Cfg, s.k3
	d := c.NEmbd
	if nb == 0 || c.k3Fault == k3FaultNoBank {
		return x[:n*d], nil
	}
	s.growK3(n)
	ns := nb + 1
	// Each row's streams normed side by side, the running residual last, as
	// the reference concatenates them; one matvec scores them all.
	for r := 0; r < n; r++ {
		for j := 0; j < ns; j++ {
			v := c.stream(x, 0, n, r)
			if j < nb {
				v = c.stream(x, j+1, n, r)
			}
			dst := g.normed[(r*ns+j)*d : (r*ns+j+1)*d]
			if c.k3Fault == k3FaultRawScores {
				copy(dst, v)
				continue
			}
			s.rmsnorm(dst, v, g.ones, c.RMSEps)
		}
	}
	s.jit.NewInput()
	if err := s.mm(g.scores[:n*ns], w, g.normed[:n*ns*d], n*ns); err != nil {
		return nil, err
	}
	for r := 0; r < n; r++ {
		p := g.scores[r*ns : (r+1)*ns]
		s.softmax(p, ns)
		mix := g.mix[r*d : (r+1)*d]
		clear(mix)
		for j := 0; j < ns; j++ {
			v := c.stream(x, 0, n, r)
			if j < nb {
				v = c.stream(x, j+1, n, r)
			}
			s.axpy(mix, v, p[j])
		}
	}
	s.jit.NewInput()
	return g.mix[:n*d], nil
}

// k3Collapse is the head's mix for n rows of x over the whole bank, written
// into stream 0, where the output norm reads it.
func (s *Instance) k3Collapse(x []float32, n int) error {
	c := s.m.Cfg
	if c.k3Fault == k3FaultNoHeadMix {
		return nil
	}
	if s.cpuCollapse(x, n) == composer.DeclineServes {
		return nil
	}
	in, err := s.k3Mix(x, n, c.resBankFinal(), s.m.resOutT)
	if err != nil {
		return err
	}
	copy(x[:n*c.NEmbd], in)
	return nil
}

// k3Plan adjusts block li's plan for Kimi-K3: the residual attention's mixes
// and checkpoint, the latent mixture's width, KDA's decay bound and the MLA
// latent norms' epsilon (Kimi-Linear's too).
func (s *Instance) k3Plan(li int, plan *nn.LayerPlan) {
	c := s.m.Cfg
	plan.LatentNormEps = c.LatentNormEps
	if c.LayerKind(li).Recurrent() {
		plan.Recurrent.DecayBound = c.KDALowerBound
	}
	if !c.ResAttn() && c.ExpertLatent == 0 {
		return
	}
	k := &nn.K3Plan{Streams: c.Streams()}
	if c.MoEAt(li) {
		k.Latent = c.ExpertLatent
	}
	if c.ResAttn() {
		k.BankAttn, k.BankFFN = c.resBank(li, false), c.resBank(li, true)
		k.Push, _ = c.resCheckpoint(li)
	}
	switch c.k3Fault {
	case k3FaultNoBank:
		k.Fault = nn.K3FaultNoBank
	case k3FaultNoRestart:
		k.Fault = nn.K3FaultNoRestart
	case k3FaultRawScores:
		k.Fault = nn.K3FaultRawScores
	case k3FaultNoLatentNorm:
		k.Fault = nn.K3FaultNoLatentNorm
	case k3FaultNoMLAGate:
		k.Fault = nn.K3FaultNoMLAGate
	case k3FaultNoKDAGate:
		k.Fault = nn.K3FaultNoKDAGate
	}
	plan.K3 = k
}
