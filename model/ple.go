package model

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
func (c *Config) PleWidth() int { return c.NLayer * c.PLEDim }

// TapsWidth is the tap rows the first DeepTaps blocks add, DeepTaps*NEmbd:
// the head of a row's side input, its per-layer inputs after them.
func (c *Config) TapsWidth() int { return c.DeepTaps * c.NEmbd }

// SideWidth is one row's side input to the blocks (their graphs' last
// input): its tap rows, then its per-layer inputs.
func (c *Config) SideWidth() int { return c.TapsWidth() + c.PleWidth() }

// InputSideWidth is the side row the input rows write: the whole row, zeros
// where the taps go, for a model of per-layer embeddings; none for one of
// taps alone, whose side rows the taps' fill writes.
func (c *Config) InputSideWidth() int {
	if c.PLEDim == 0 {
		return 0
	}
	return c.SideWidth()
}
