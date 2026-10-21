package cpu

import (
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// The d plane's geometry. It is not the payload's, and on a narrow format not
// even the same row stride. It lives in an untagged file because both
// architectures' emitters and nn's callers derive from it.

// NarrowD reports whether this format's d plane packs more than one row into
// one word (kernels.DSlots > 1), without the caller needing a Quant.
func NarrowD(t quant.Type) bool { return DSlotsOf(t) > 1 }

// E8M0D reports whether the super-scale is MXFP4's E8M0 exponent: one byte,
// four rows to a word, and the stored byte shifted left 23 is the f32 scale.
// It is a third shape, not a narrower second: a reader widens a byte and
// shifts rather than converting an f16, and treating it as narrow reads two
// rows' scales as one f16 without failing.
func E8M0D(t quant.Type) bool {
	q, ok := kernels.QuantOf(t)
	return ok && kernels.DSlots(q) == 4
}

// E8M0Quant is E8M0D for a caller that already holds a kernels.Quant -- the
// emitters, which are parameterised by it rather than by a quant.Type.
func E8M0Quant(q kernels.Quant) bool { return kernels.DSlots(q) == 4 }

// DSlotsOf is how many rows share one d word: 1, 2 or 4.
func DSlotsOf(t quant.Type) int {
	q, ok := kernels.QuantOf(t)
	if !ok {
		return 1
	}
	return kernels.DSlots(q)
}

// DRowBytes is the d plane's bytes per row, within one super-block: four for
// the formats that carry a minimum beside the scale, two for a narrow f16
// scale, one for MXFP4's E8M0 byte.
func DRowBytes(t quant.Type) int { return 4 / DSlotsOf(t) }

// DSuperBytes is the d plane's bytes between consecutive super-blocks at a
// given row stride -- what a kernel adds to its d cursor when it finishes one.
//
// It rounds the row count up to a whole number of words: DSlots rows share a
// word, so a ragged count would start the next super-block part way through
// one. kernels.PackedWords reserves the same padding, from the same expression.
func DSuperBytes(t quant.Type, nrows int) int {
	n := DSlotsOf(t)
	return 4 * ((nrows + n - 1) / n)
}
