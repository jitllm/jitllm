package model

import (
	"image"
)

// Span is one run of a mixed prompt: token ids, a picture, or precomputed
// embeddings -- one of them. Embd is n*NEmbd floats laid out row-major, and is
// the residual stream rather than a vocabulary row -- EmbdScale is not applied
// to it, for the reason ForwardEmbd's comment gives.
//
// An Embd span may carry Tokens too, one per row: the ids its rows embed,
// which are not looked up but read where a model reads the token past the
// embedding -- DeepSeek V4's hash-routed blocks, which refuse a row without one
// as its references cannot run it, and per-layer embeddings, which take token
// 0's for such a row (ple.go).
type Span struct {
	Tokens []int32
	Embd   []float32
	// Bidir says the span's rows attend to each other in both directions
	// rather than causally (Gemma 3's image tokens, which its reference masks
	// so). The rows before and after it stay causal.
	Bidir bool
	// Grid is an image span's shape, which is where its rows turn on a
	// multi-axis rotary (mrope.go). It is required there and ignored by a
	// model whose rotary is a plain index.
	Grid ImageGrid
	// Key names what an embedding span's rows came from (RowsKey), so the
	// prefix cache can name the pages that hold them and every page after.
	// nil leaves them unnamed, and nothing from them on is cached.
	Key *RowsKey
	// Picture is a picture whose rows the prefill encodes through the model's
	// tower, PicRows of them, when Embd is nil (Model.PictureSpan): a picture
	// is a span of the prompt, not a stage before it, and one the prefix
	// cache restores past runs no tower block.
	Picture image.Image
	PicRows int
	// Deep is an Embd span's tap rows (jlm.TowerTap, Qwen3-VL's deepstack),
	// [taps][rows][NEmbd]: text block i adds tap i's rows into the span's
	// after it runs. A picture span's are its tower's.
	Deep []float32
}

// RowsKey names the content an embedding span's rows came from: a hash over
// everything that decides them, known before they are computed, so the prefix
// cache names their pages as it names tokens.
type RowsKey [32]byte

func (sp Span) N(nEmbd int) int {
	if sp.Embd != nil {
		return len(sp.Embd) / nEmbd
	}
	if sp.Picture != nil {
		return sp.PicRows
	}
	return len(sp.Tokens)
}

// Zero reports whether the key names nothing.
func (k RowsKey) Zero() bool { return k == RowsKey{} }

// Rows reports whether the span is rows rather than tokens.
func (sp Span) Rows() bool { return sp.Embd != nil || sp.Picture != nil }
