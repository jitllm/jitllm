package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// Gemma 3n's AltUp and LAuReL (Config.PredStreams), transcribed from transformers'
// Gemma3nTextModel/Gemma3nTextDecoderLayer, which llama.cpp's gemma3n.cpp
// matches.
//
// The residual is AltUp streams of n_embd. At the embedding, stream 0 is the
// scaled embedding and stream k is altup_proj_k of it, matched to its
// magnitude:
//
//	x_k = mag(P_k x_0, x_0)      mag(p, r) = p * rms(r) / sqrt(max(mean(p^2), 1e-5))
//
// Each block predicts every stream from all of them, runs on stream 0 and
// corrects the rest by what the block did to it:
//
//	m      = tanh(R · rmsnorm(x_0) / n_embd)                 the modalities
//	pred_k = x_k + sum_j c_{kj} x_j,  c = C_pred m           AltUp^2 coefficients
//	a      = pred_0, through gemma3's block, the attention's residual add
//	         joined by LAuReL: (a + attn + laurel(norm(a))) / sqrt(2)
//	x_k    = pred_k + (C_corr m' + 1)_k (out - pred_0)       m' from the block's out
//	x_k   += post_norm(W_out (gelu(W_gate (x_0 * s)) * pl))   for k >= 1
//
// and the head reads the mean of x_0 and each mag(U_k x_k, x_0).
//
// The streams are carried STREAM-MAJOR: for `rows` rows, stream k of row r is
// at (k*rows + r)*n_embd. State.x is stream 0 of the one-row residual State.xa,
// and a chunk's s.bx holds its n rows' streams the same way, so every n_embd
// kernel that reads the residual reads stream 0 untouched and a device is
// handed the whole residual as one slice (Config.ResidW).

// magEps is the floor transformers puts under a stream's mean square before
// the magnitude match divides by its root.
const magEps = 1e-5

// Streams is how many residual streams a token carries: AltUp's, DeepSeek
// V4's hyper-connections, Kimi-K3's running residual and its checkpoints, or
// one.
func (c *Config) Streams() int { return max(c.PredStreams, c.HCMult, c.resStreams(), 1) }

// ResidW is one row's residual width, every stream of it.
func (c *Config) ResidW() int { return c.NEmbd * c.Streams() }

// stream is stream k of row r of a stream-major residual of `rows` rows.
func (c *Config) stream(x []float32, k, rows, r int) []float32 {
	d := c.NEmbd
	o := (k*rows + r) * d
	return x[o : o+d]
}

// resid is the one-row residual a device is handed: every stream.
func (s *State) resid() []float32 {
	if s.xa != nil {
		return s.xa
	}
	return s.x
}

// altRowsStep is rowsStep's device call for an AltUp model, whose rows are
// embedded: the streams expanded, the blocks on the device without the head,
// and the head on the host over the streams' mean, finished as the device's
// is (the final softcap; the caller adds the rest). With no logits wanted the
// rows' greedy tokens go to the head's Tokens, as the device's argmax would.
func (s *State) altRowsStep(rd nn.RowsDevice, pos, slot []int, tokens []int32, n int, logits []float32,
	nlogit int) error {
	c := s.m.Cfg
	x := s.bx[:n*c.ResidW()]
	if err := s.expandStreams(x, n); err != nil {
		return err
	}
	// Qwen3.8-Flash-Next's n-gram rows, each sequence's in position order.
	if c.NGramBlock != 0 {
		s.q4.setStep(s, pos, slot, tokens)
		if err := s.ngramToDevice(s.lo, s.hi, n, s.causalOrder(pos), s.q4.stepOwner, s.q4.callTok); err != nil {
			return err
		}
	}
	s.rowAxes(pos[0], 0, n)
	if !rd.LayersRows(s.lo, s.hi, pos, slot, s.maxSeq, x, s.bcs[:n*c.NRot], s.bswaTable(n, c.NRotSWA), nil) {
		why := "no reason given"
		if e, ok := s.ld.(nn.ErrReporter); ok {
			why = e.Err()
		}
		return fmt.Errorf("model: the device refused a %d-row step: %s", n, why)
	}
	greedy := logits == nil
	if greedy {
		nlogit = n
		if len(s.blogits) < n*c.NVocab {
			s.blogits = make([]float32, n*c.NVocab)
		}
		logits = s.blogits[:n*c.NVocab]
	}
	if nlogit == 0 {
		return nil
	}
	if err := s.collapseStreams(x, n); err != nil {
		return err
	}
	s.growBatch(n)
	if s.cpuHeadRows(s.bx, logits, nil, nlogit) != jit.DeclineServes {
		s.altHeadNorm(nlogit)
		if err := s.mm(logits, *s.outW, s.bh, nlogit); err != nil {
			return err
		}
		for i := 0; i < nlogit; i++ {
			softcap(logits[i*c.NVocab:(i+1)*c.NVocab], c.FinalSoftcap)
		}
	}
	if greedy {
		if len(s.head.Tokens) < n {
			s.head.Tokens = make([]int32, n)
		}
		for i := 0; i < n; i++ {
			s.head.Tokens[i] = Greedy(logits[i*c.NVocab : (i+1)*c.NVocab])
		}
	}
	return nil
}

// altHeadNorm is the output norm of a chunk's first n rows into s.bh, row by
// row: a step's handful of rows, where batchNormB's pool closure would cost
// a warm step an allocation.
func (s *State) altHeadNorm(n int) {
	d := s.m.Cfg.NEmbd
	for i := 0; i < n; i++ {
		s.norm(s.bh[i*d:(i+1)*d], s.bx[i*d:(i+1)*d], s.outNorm, s.m.outNormB)
	}
	s.jit.NewInput()
}

// loadAltUp reads the model's AltUp projections.
func loadAltUp(m *Model, cfg *Config, get func(jlm.Role, int32, int32) (tensor, error)) error {
	var err error
	d, s := cfg.NEmbd, cfg.Streams()
	if m.altProj, err = get(jlm.RoleStreamExpand, jlm.DenseBlock, -1); err != nil {
		return err
	}
	if m.altProj.k != d || m.altProj.rows != (s-1)*d {
		return fmt.Errorf("model: altup_proj is %dx%d, want %d rows of %d", m.altProj.rows, m.altProj.k, (s-1)*d, d)
	}
	m.altUnembd = make([]tensor, s-1)
	for k := range m.altUnembd {
		if m.altUnembd[k], err = get(jlm.RoleStreamCollapse1+jlm.Role(k), jlm.DenseBlock, -1); err != nil {
			return err
		}
		if t := m.altUnembd[k]; t.k != d || t.rows != d {
			return fmt.Errorf("model: altup_unembd_proj %d is %dx%d, want %dx%d", k+1, t.rows, t.k, d, d)
		}
	}
	return nil
}

// loadAltUpLayer reads block bi's AltUp and LAuReL tensors.
func loadAltUpLayer(l *layer, bi int32, cfg *Config,
	get func(jlm.Role, int32, int32) (tensor, error), vec func(jlm.Role, int32) ([]float32, error)) error {
	var err error
	d, s := cfg.NEmbd, cfg.Streams()
	for _, v := range []struct {
		dst  *[]float32
		role jlm.Role
		n    int
	}{
		{&l.altRouterNorm, jlm.RoleStreamRouterNorm, d},
		{&l.altPredT, jlm.RoleStreamPredictCoef, s * s * s},
		{&l.altCorrT, jlm.RoleStreamCorrectCoef, s * s},
		{&l.altCorrScale, jlm.RoleStreamCorrectScale, d},
		{&l.laurelPost, jlm.RoleLowRankNorm, d},
	} {
		if *v.dst, err = vec(v.role, bi); err != nil {
			return err
		}
		if len(*v.dst) != v.n {
			return fmt.Errorf("model: block %d's %v holds %d values, want %d", bi, v.role, len(*v.dst), v.n)
		}
	}
	if l.altRouter, err = get(jlm.RoleStreamRouter, bi, -1); err != nil {
		return err
	}
	if l.laurelL, err = get(jlm.RoleLowRankIn, bi, -1); err != nil {
		return err
	}
	if l.laurelR, err = get(jlm.RoleLowRankOut, bi, -1); err != nil {
		return err
	}
	if l.altRouter.k != d || l.altRouter.rows != s || l.laurelL.k != d || l.laurelR.k != l.laurelL.rows ||
		l.laurelR.rows != d {
		return fmt.Errorf("model: block %d's AltUp router is %dx%d and LAuReL %dx%d then %dx%d, want "+
			"%d rows of %d and a rank between", bi, l.altRouter.rows, l.altRouter.k, l.laurelL.rows,
			l.laurelL.k, l.laurelR.rows, l.laurelR.k, s, d)
	}
	return nil
}

// allocAltUp sizes the decode token's AltUp scratch.
func (s *State) allocAltUp() {
	c := s.m.Cfg
	if c.PredStreams == 0 {
		return
	}
	d, n := c.NEmbd, c.Streams()
	s.xa = make([]float32, n*d)
	s.x = s.xa[:d]
	s.altPred = make([]float32, n*d)
	s.altH = make([]float32, d)
	s.altM = make([]float32, n)
	s.altC = make([]float32, n*n)
	s.altOnes = make([]float32, n)
	for i := range s.altOnes {
		s.altOnes[i] = 1
	}
	s.altInn = make([]float32, d)
	s.altLaur = make([]float32, d)
	s.altLT = make([]float32, s.m.layers[0].laurelL.rows)
	s.altExp = make([]float32, (n-1)*d)
	s.altXC = make([]float32, d)
}

// altExpand fills streams 1.. of each of `rows` rows from stream 0, the
// scaled embedding.
func (s *State) altExpand(x []float32, rows int) error {
	if s.cpuExpand(x, rows) == jit.DeclineServes {
		return nil
	}
	m, c := s.m, s.m.Cfg
	d := c.NEmbd
	for r := 0; r < rows; r++ {
		x0 := c.stream(x, 0, rows, r)
		s.jit.NewInput()
		if err := s.mv(s.altExp, m.altProj, x0); err != nil {
			return err
		}
		for k := 1; k < c.Streams(); k++ {
			nn.MagMatch32JIT(c.stream(x, k, rows, r), s.altExp[(k-1)*d:k*d], x0, magEps)
		}
	}
	s.jit.NewInput()
	return nil
}

// altCollapse writes each row's head input into stream 0: the mean of
// stream 0 and each unembedded stream matched to stream 0's magnitude.
func (s *State) altCollapse(x []float32, rows int) error {
	if s.cpuCollapse(x, rows) == jit.DeclineServes {
		return nil
	}
	m, c := s.m, s.m.Cfg
	n := c.Streams()
	for r := 0; r < rows; r++ {
		x0 := c.stream(x, 0, rows, r)
		copy(s.altXC, x0)
		for k := 1; k < n; k++ {
			s.jit.NewInput()
			if err := s.mv(s.altH, m.altUnembd[k-1], c.stream(x, k, rows, r)); err != nil {
				return err
			}
			nn.MagMatch32JIT(s.altH, s.altH, x0, magEps)
			s.addInto(s.altXC, s.altH)
		}
		s.scale(s.altXC, float32(1/float64(n)))
		copy(x0, s.altXC)
	}
	s.jit.NewInput()
	return nil
}
