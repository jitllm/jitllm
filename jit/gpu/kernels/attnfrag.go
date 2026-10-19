package kernels

import "github.com/jitllm/jitllm/jit/gpu/ir"

// The paged decode attention's fragments, for the attention former
// (compose.Attend and the staged formers) that owns the loop nests: the paged
// history's addressing (paged.go's ABI: a row's descriptor words, a
// position's page through the table, its slot in the page), the binary16 K
// pair and half, a warp's butterfly, a row's key partitions, and the shape
// arithmetic the plan's schedule is made of. They are the arithmetic
// FlashDecodeKV, the staged kernels and the merges run, so a formed attention
// and the menu's compute the same bits.

// PagedDesc is word w of row row's descriptor.
func PagedDesc(b *ir.Builder, pRow, row ir.Value, w int) ir.Value { return pagedDesc(b, pRow, row, w) }

// PageOf is the page id holding position t of the row whose table starts at
// tabOff.
func PageOf(b *ir.Builder, pTab, tabOff, t ir.Value, page int) ir.Value {
	return pageOf(b, pTab, tabOff, t, page)
}

// PageRem is t's slot in its page of page positions.
func PageRem(b *ir.Builder, t ir.Value, page int) ir.Value { return pageRem(b, t, page) }

// KHalf is the binary16 half of K word w holding dimension e.
func KHalf(b *ir.Builder, w, e ir.Value) ir.Value { return kHalf(b, w, e) }

// KPair is the two dimensions a binary16 K word holds, low first.
func KPair(b *ir.Builder, w ir.Value) (lo, hi ir.Value) { return kPair(b, w) }

// Butterfly is v combined across a warp's 32 lanes by comb, every lane
// holding the result.
func Butterfly(b *ir.Builder, v ir.Value, comb func(x, y ir.Value) ir.Value) ir.Value {
	return butterfly(b, v, comb)
}

// PagedSplit is split sp's keys [St, En) of a row with keys [Ks, keyEnd), the
// row's table Tab, the last key it may address and whether it has any.
type PagedSplit struct{ s stagedSplit }

// PagedSplitOf is row row's split sp of splits, from its descriptor.
func PagedSplitOf(b *ir.Builder, pRow, row, sp ir.Value, splits int) PagedSplit {
	return PagedSplit{pagedSplitOf(b, pRow, row, sp, splits)}
}

// Tab, Ks, St, En, Last and Live are the split's table offset, key start,
// first and end key, last addressable key, and whether the row has keys.
func (p PagedSplit) Tab() ir.Value  { return p.s.tab }
func (p PagedSplit) Ks() ir.Value   { return p.s.ks }
func (p PagedSplit) St() ir.Value   { return p.s.st }
func (p PagedSplit) En() ir.Value   { return p.s.en }
func (p PagedSplit) Last() ir.Value { return p.s.last }
func (p PagedSplit) Live() ir.Value { return p.s.live }

// Key is where key St+slot of the split is read.
func (p PagedSplit) Key(b *ir.Builder, slot ir.Value) ir.Value { return p.s.key(b, slot) }

// Valid is whether slot holds one of the row's keys.
func (p PagedSplit) Valid(b *ir.Builder, slot ir.Value) ir.Value { return p.s.valid(b, slot) }

// FlashKVGeom is FlashDecodeKV's warps for s: W warps, Wd slices of the
// dimensions of ds each, times Wk groups of keys.
func FlashKVGeom(s FlashShape) (w, wd, wk, ds int) { return flashKVGeom(s) }

// FlashKVGroup is how many query heads one FlashDecodeKV workgroup serves.
func FlashKVGroup(s FlashShape) int { return flashKVGroup(s) }

// FlashKVVecV is how many consecutive V elements a lane loads at once.
func FlashKVVecV(s FlashShape) int { return flashKVVecV(s) }

// FlashKVCheck is FlashDecodeKV's refusal of a shape, nil where it builds.
func FlashKVCheck(s FlashShape) error { return flashKVCheck(s) }

// StagedGroup is how many query heads of one KV head a staged thread serves.
func StagedGroup(s FlashShape) int { return stagedGroup(s) }

// StagedValue is the width the staged accumulate and the merge run at.
func StagedValue(s FlashShape) int { return stagedValue(s) }

// StagedCheck is the staged kernels' refusal of a shape, nil where they
// build.
func StagedCheck(what string, s FlashShape) error { return stagedCheck(what, s) }
