package model

import (
	"fmt"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
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
// at (k*rows + r)*n_embd. Instance.x is stream 0 of the one-row residual Instance.xa,
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
func (s *Instance) resid() []float32 {
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
func (s *Instance) altRowsStep(rd nn.RowsDevice, pos, slot []int, tokens []int32, n int, logits []float32,
	nlogit int) error {
	c := s.m.Cfg
	x := s.bx[:n*c.ResidW()]
	if err := s.expandStreams(x, n, 0); err != nil {
		return err
	}
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
	if why := s.cpuHeadRows(s.bx, logits, nil, nlogit); why != composer.DeclineServes {
		return s.headDeclined(why)
	}
	if greedy {
		if len(s.head.Tokens) < n {
			s.head.Tokens = make([]int32, n)
		}
		for i := 0; i < n; i++ {
			tok, err := s.Sample(nil, logits[i*c.NVocab:(i+1)*c.NVocab])
			if err != nil {
				return err
			}
			s.head.Tokens[i] = tok
		}
	}
	return nil
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

// loadAltUpLayer reads block bi's AltUp and LAuReL matrices; their vectors
// are the block's table's (layer.vecs).
func loadAltUpLayer(l *layer, bi int32, cfg *Config, getTo func(*tensor, jlm.Role, int32, int32) error) error {
	var err error
	d, s := cfg.NEmbd, cfg.Streams()
	if err = getTo(&l.altRouter, jlm.RoleStreamRouter, bi, -1); err != nil {
		return err
	}
	if err = getTo(&l.laurelL, jlm.RoleLowRankIn, bi, -1); err != nil {
		return err
	}
	if err = getTo(&l.laurelR, jlm.RoleLowRankOut, bi, -1); err != nil {
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
