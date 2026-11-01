package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/nn"
)

// Gemma 4's per-layer embeddings (E2B/E4B, Config.PLEDim), transcribed from
// transformers' Gemma4TextModel.project_per_layer_inputs and
// Gemma4TextDecoderLayer, which llama.cpp's gemma4.cpp matches:
//
//	tok = table[token] * sqrt(P)                      NLayer*P wide
//	prj = rmsnorm_P(W_proj · x / sqrt(n_embd))       x is the scaled embedding,
//	                                                  normed per P-wide slice
//	pl  = (prj + tok) / sqrt(2)
//
// and in block li, after the FFN's residual add and before layer_scalar:
//
//	x += post_norm(W_out · (gelu(W_gate · x) * pl[li]))
//
// A row with no token (an embedding supplied through ForwardEmbd or a mixed
// prefill) takes token 0's table row, as llama.cpp does for an image's rows.

// pleWidth is one row's per-layer inputs, NLayer*PLEDim.
func (c *Config) pleWidth() int { return c.NLayer * c.PLEDim }

// pleInputs writes one row's per-layer inputs into dst from its token and its
// scaled embedding x.
func (s *State) pleInputs(dst []float32, token int32, x []float32) error {
	m, c := s.m, s.c
	p, w := c.PLEDim, c.pleWidth()
	t := s.pleT[:w]
	// An id past the table -- Gemma 3n's vision and audio tokens, whose
	// padding rows the converter cuts -- reads row 0, as transformers'
	// per_layer_inputs_mask maps it.
	if int(token) >= m.pleTok.rows {
		token = 0
	}
	if err := m.rowOf(&m.pleTok, t, int(token)); err != nil {
		return err
	}
	s.scale(t, float32(math.Sqrt(float64(p))))
	s.jit.NewInput()
	if err := s.mv(dst[:w], m.pleProj, x); err != nil {
		return err
	}
	s.scale(dst[:w], float32(1/math.Sqrt(float64(c.NEmbd))))
	for li := 0; li < c.NLayer; li++ {
		v := dst[li*p : (li+1)*p]
		s.rmsnorm(v, v, m.pleNorm, c.RMSEps)
	}
	s.axpy(dst[:w], t, 1)
	s.scale(dst[:w], float32(1/math.Sqrt2))
	return nil
}

// growPLE sizes the per-layer embedding scratch. A model without them gets
// none.
func (s *State) growPLE() {
	c := s.c
	if c.PLEDim == 0 || s.ple != nil {
		return
	}
	w := c.pleWidth()
	s.ple = make([]float32, w)
	s.pleT = make([]float32, w)
	s.pleG = make([]float32, c.PLEDim, ffnPad(c.PLEDim))
	s.pleO = make([]float32, c.NEmbd)
}

// kvGroupOffered reports whether block li may be offered by offerRange(lo,
// hi): any block outside a KV-sharing group, or one whose every group member
// is on a device already or offered by the same call.
func (s *State) kvGroupOffered(li, lo, hi int) bool {
	c := s.c
	if c.NKVShared == 0 {
		return true
	}
	src := c.KVSource(li)
	for b := 0; b < c.NLayer; b++ {
		if b != src && c.KVSource(b) != src {
			continue
		}
		if !s.devAt(b) && (b < lo || b >= hi) {
			return false
		}
	}
	return true
}

// kvSpan is the blocks that must be offered with block li: li alone, or the
// span of its KV-sharing group, source to last reader.
func (s *State) kvSpan(li int) (lo, hi int) {
	c := s.c
	lo, hi = li, li+1
	if c.NKVShared == 0 {
		return lo, hi
	}
	src := c.KVSource(li)
	for b := 0; b < c.NLayer; b++ {
		if b == src || c.KVSource(b) == src {
			lo, hi = min(lo, b), max(hi, b+1)
		}
	}
	return lo, hi
}

// fixKVGroups keeps every KV-sharing group whole (Gemma 4's E2B/E4B): a block
// attends to its source's history, which lives on exactly one side, so the
// source and every block reading it must be placed together. A group split by
// the placement -- a device that took the source and declined a reader, or a
// relocation that took one member to the CPU -- moves to the CPU whole, with the source's
// history. The device declines a reader whose source it does not hold (tier)
// and offerRange offers a group only whole, so this is the third half.
func (s *State) fixKVGroups() {
	c := s.c
	if c.NKVShared == 0 || s.ld == nil {
		return
	}
	for li := c.NLayer - c.NKVShared; li < c.NLayer; li++ {
		src := c.KVSource(li)
		if s.devAt(li) == s.devAt(src) {
			continue
		}
		// Every block of src's group moves to the CPU.
		for b := 0; b < c.NLayer; b++ {
			if b != src && c.KVSource(b) != src {
				continue
			}
			if !s.devAt(b) {
				continue
			}
			if b == src && !s.migrateKV(b, s.pos, false) {
				continue // the history cannot move; the device keeps the block
			}
			s.ld.ReleaseLayers(b, b+1)
			s.unmarkOnDev(b)
		}
		s.gpuTarget = min(s.gpuTarget, s.gpuLayers)
	}
}

// growBPLE sizes a batched chunk's per-layer inputs for n rows. Grows, never
// shrinks.
func (s *State) growBPLE(n int) {
	if w := n * s.c.pleWidth(); len(s.bple) < w {
		s.bple = make([]float32, w)
	}
}

// rowOf dequantizes row token of a table whose rows are tokens: the embedding
// or the per-layer embedding table.
func (m *Model) rowOf(e *tensor, dst []float32, token int) error {
	if token < 0 || token >= e.rows {
		return fmt.Errorf("model: token %d is outside a table of %d rows", token, e.rows)
	}
	if e.packed != nil && len(e.packed.QS) != 0 {
		nn.RowPacked32JIT(dst, e.typ, e.packed, token, e.rows, e.k)
		return nil
	}
	nn.Row32JIT(dst, e.typ, e.data, token, e.k)
	return nil
}
