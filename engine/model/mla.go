package model

import (
	"github.com/jitllm/jitllm/engine/nn"
)

// Multi-head Latent Attention, the DeepSeek-V2/V3 attention and the one place
// in this engine where a key and a value are not the same width.
//
// The cache holds one row per position for the whole layer:
//
//	[ latent (KVLoraRank) | rotary key (NRot) ]
//
// and every head attends to that same row. The up-projections that would turn
// the latent back into per-head keys and values are absorbed into the operands
// either side of the softmax:
//
//	score_h  = q_nope_h . (W_k_h . c)  =  (W_k_hᵀ . q_nope_h) . c
//	out_h    = Σ a_j (W_v_h . c_j)     =  W_v_h . (Σ a_j c_j)
//
// so W_k_hᵀ folds into the query and W_v_h into the output, and neither ever
// materialises a key or a value. That is why l.wkb is stored already
// transposed (the converter does it) and l.wvb is not.
//
// The value is the key's prefix rather than a second tensor: the scores read
// all KVLoraRank+NRot floats of the row and the accumulation only the leading
// KVLoraRank. kvPages.latent aliases the two caches and nn.AddAttnKV emits the
// kernels at the two widths.

// allocMLA sizes the per-State buffers one MLA token needs, so a decode token
// allocates nothing here.
func (s *State) allocMLA() {
	c := s.c
	if !c.MLA() {
		return
	}
	row := c.KVLoraRank + c.NRot
	s.mlaAbs = make([]float32, c.NHead*row)
	s.mlaAcc = make([]float32, c.NHead*c.KVLoraRank)
	s.mlaPE = make([]float32, c.NHead*c.NRot)
	s.mlaOut = make([]float32, c.NHead*c.HeadDimV)
	if c.QLoraRank != 0 {
		s.mlaQA = make([]float32, c.QLoraRank)
	}
	// One output slice per head for each gather, built once so the per-token
	// path appends nothing.
	s.mlaAbsO = make([][]float32, c.NHead)
	s.mlaOutO = make([][]float32, c.NHead)
	s.mlaKB = make([]*nn.Packed, 0, c.NHead)
	s.mlaVB = make([]*nn.Packed, 0, c.NHead)
	for h := 0; h < c.NHead; h++ {
		s.mlaAbsO[h] = s.mlaAbs[h*row : h*row+c.KVLoraRank]
		s.mlaOutO[h] = s.mlaOut[h*c.HeadDimV : (h+1)*c.HeadDimV]
	}
	if s.m.mlaGated() {
		s.mlaG = make([]float32, c.NHead*c.HeadDimV)
	}
	s.allocIndexer()
}

// ── the batched paths ──────────────────────────────────────────────────────────────
//
// The projections batch and the rest does not, the same split the dense path
// makes: q, the latent and their norms are row-independent; the rotary, the
// absorb and the cache write depend on the position and run per row. The
// operations and their order are mlaProject's.

// growMLABatch sizes the per-chunk MLA buffers.
func (s *State) growMLABatch(chunk int) {
	c := s.c
	if !c.MLA() || len(s.bmlaAbs) >= chunk*c.NHead*(c.KVLoraRank+c.NRot) {
		return
	}
	row := c.KVLoraRank + c.NRot
	s.bmlaAbs = make([]float32, chunk*c.NHead*row)
	s.bmlaAcc = make([]float32, chunk*c.NHead*c.KVLoraRank)
	s.bmlaKV = make([]float32, chunk*row)
	s.bmlaOut = make([]float32, chunk*c.NHead*c.HeadDimV)
	if s.m.mlaGated() {
		s.bmlaG = make([]float32, chunk*c.NHead*c.HeadDimV)
	}
	if c.QLoraRank != 0 {
		s.bmlaQA = make([]float32, chunk*c.QLoraRank)
	}
	// One output slice per (row, head) for the two gathers, built once.
	s.bmlaAbsO = make([][]float32, c.NHead)
	s.bmlaOutO = make([][]float32, c.NHead)
	s.growIndexerRows(chunk)
}

// attnWidths is the key and value widths the attention kernels are emitted
// for. They differ only under MLA, where a score covers the whole cached row
// and the accumulation only its latent prefix. NewState and stepDownKVToF32
// must both derive them here.
func (c *Config) attnWidths() (hdK, hdV int) {
	if c.KVLoraRank != 0 {
		return c.KVLoraRank + c.NRot, c.KVLoraRank
	}
	return c.HeadDim, c.HeadDim
}

// mlaGated reports whether any full block carries Kimi-K3's MLA output gate.
func (m *Model) mlaGated() bool {
	for i := range m.layers {
		if m.layers[i].mlaGate.rows != 0 {
			return true
		}
	}
	return false
}
