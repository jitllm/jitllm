package model

import ()

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

// attnWidths is the key and value widths the attention kernels are emitted
// for. They differ only under MLA, where a score covers the whole cached row
// and the accumulation only its latent prefix. NewInstance and stepDownKVToF32
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
