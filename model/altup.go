package model

import (
	"fmt"

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
// V4's hyper-connections, the streams the converter states, or one.
func (c *Config) Streams() int { return max(c.PredStreams, c.HCMult, c.statedStreams(), 1) }

// statedStreams is how many streams the converter states, 0 for none.
func (c *Config) statedStreams() int {
	if c.stated == nil {
		return 0
	}
	return int(c.stated.N)
}

// ResidW is one row's residual width, every stream of it.
func (c *Config) ResidW() int { return c.NEmbd * c.Streams() }

// stream is stream k of row r of a stream-major residual of `rows` rows.
func (c *Config) Stream(x []float32, k, rows, r int) []float32 {
	d := c.NEmbd
	o := (k*rows + r) * d
	return x[o : o+d]
}

// loadAltUp reads the model's AltUp projections.
func loadAltUp(m *Model, cfg *Config, get func(jlm.Role, int32, int32) (Tensor, error)) error {
	var err error
	d, s := cfg.NEmbd, cfg.Streams()
	if m.AltProj, err = get(jlm.RoleStreamExpand, jlm.DenseBlock, -1); err != nil {
		return err
	}
	if m.AltProj.K != d || m.AltProj.Rows != (s-1)*d {
		return fmt.Errorf("model: altup_proj is %dx%d, want %d rows of %d", m.AltProj.Rows, m.AltProj.K, (s-1)*d, d)
	}
	m.AltUnembd = make([]Tensor, s-1)
	for k := range m.AltUnembd {
		if m.AltUnembd[k], err = get(jlm.RoleStreamCollapse1+jlm.Role(k), jlm.DenseBlock, -1); err != nil {
			return err
		}
		if t := m.AltUnembd[k]; t.K != d || t.Rows != d {
			return fmt.Errorf("model: altup_unembd_proj %d is %dx%d, want %dx%d", k+1, t.Rows, t.K, d, d)
		}
	}
	return nil
}

// loadAltUpLayer reads block bi's AltUp and LAuReL matrices; their vectors
// are the block's table's (layer.vecs).
func loadAltUpLayer(l *Layer, bi int32, cfg *Config, getTo func(*Tensor, jlm.Role, int32, int32) error) error {
	var err error
	d, s := cfg.NEmbd, cfg.Streams()
	if err = getTo(&l.AltRouter, jlm.RoleStreamRouter, bi, -1); err != nil {
		return err
	}
	if err = getTo(&l.LaurelL, jlm.RoleLowRankIn, bi, -1); err != nil {
		return err
	}
	if err = getTo(&l.LaurelR, jlm.RoleLowRankOut, bi, -1); err != nil {
		return err
	}
	if l.AltRouter.K != d || l.AltRouter.Rows != s || l.LaurelL.K != d || l.LaurelR.K != l.LaurelL.Rows ||
		l.LaurelR.Rows != d {
		return fmt.Errorf("model: block %d's AltUp router is %dx%d and LAuReL %dx%d then %dx%d, want "+
			"%d rows of %d and a rank between", bi, l.AltRouter.Rows, l.AltRouter.K, l.LaurelL.Rows,
			l.LaurelL.K, l.LaurelR.Rows, l.LaurelR.K, s, d)
	}
	return nil
}
