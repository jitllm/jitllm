package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// unrollBlocks states the loop the library states (Config.Unroll): block i
// runs source layer Unroll[i], with a history of its own. A layer's first
// block takes its weights' page, in the order the blocks first read them,
// and a later block naming the layer again reads that page
// (jlm.Config.WeightsOf), so a pass stores nothing. Every statement of one
// entry per layer follows its block's layer. A source layer no block runs is
// refused: its weights would be a page nothing reads.
func unrollBlocks(cfg *Config, s *jlm.Source) error {
	order := cfg.Unroll
	if order == nil {
		return nil
	}
	if n := len(cfg.SwiGLULimits); n != 0 && n != int(cfg.NLayer) {
		return fmt.Errorf("convert: unroll: SwiGLU limits has %d entries for %d layers", n, cfg.NLayer)
	}
	limits := cfg.SwiGLULimits
	if err := UnrollSource(s, order); err != nil {
		return err
	}
	if len(limits) != 0 {
		cfg.SwiGLULimits = byLayer(limits, order)
	}
	return nil
}

// UnrollSource is unrollBlocks over a source's config and tensors: block i
// runs source layer order[i].
func UnrollSource(s *jlm.Source, order []int) error {
	c := s.Config
	n := int(c.NLayer)
	page := make([]int, n)
	for i := range page {
		page[i] = -1
	}
	pages := 0
	weightsOf := make([]uint32, 0, len(order)+int(c.HeadBlocks()+c.PredictionBlocks()))
	for b, l := range order {
		if l < 0 || l >= n {
			return fmt.Errorf("convert: unroll: block %d runs layer %d of %d", b, l, n)
		}
		if page[l] < 0 {
			page[l] = pages
			pages++
		}
		weightsOf = append(weightsOf, uint32(page[l]))
	}
	if pages != n {
		return fmt.Errorf("convert: unroll runs %d of the source's %d layers", pages, n)
	}
	if err := byIndexFollows(c, order); err != nil {
		return err
	}
	for k := range c.HeadBlocks() + c.PredictionBlocks() {
		weightsOf = append(weightsOf, uint32(pages)+k)
	}
	var err error
	follow := func(name string, have int) bool {
		if err == nil && have != 0 && have != n {
			err = fmt.Errorf("convert: unroll: %s has %d entries for %d layers", name, have, n)
		}
		return have == n
	}
	if follow("layer kinds", len(c.LayerKinds)) {
		c.LayerKinds = byLayer(c.LayerKinds, order)
	}
	if follow("geometries", len(c.Geoms)) {
		c.Geoms = byLayer(c.Geoms, order)
	}
	if follow("windows", len(c.Windows)) {
		c.Windows = byLayer(c.Windows, order)
	}
	if follow("compression kinds", len(c.CompKinds)) {
		c.CompKinds = byLayer(c.CompKinds, order)
	}
	if follow("activation layers", len(c.ActLayers)) {
		c.ActLayers = byLayer(c.ActLayers, order)
	}
	if follow("bidirectional layers", len(c.BidirLayers)) {
		c.BidirLayers = byLayer(c.BidirLayers, order)
	}
	if err != nil {
		return err
	}
	for i := range s.Tensors {
		t := &s.Tensors[i]
		switch {
		case t.Block < 0 || t.Role.Vision():
		case int(t.Block) < n:
			t.Block = int32(page[t.Block])
		default:
			t.Block += int32(pages - n)
		}
	}
	c.NLayer = uint32(len(order))
	c.WeightsOf = weightsOf
	return nil
}

// byIndexFollows refuses a loop over a config whose scalars say by a
// block's index what the block does, where a block's index says otherwise
// than the index of the layer it runs: a sliding window's period (and the
// NoPE layers it names; each block's own window, Windows, follows the
// unroll), the leading dense blocks and the mixture's step,
// the hash-routed blocks. A scalar whose meaning is the block count itself,
// a residual checkpoint's period or the KV-sharing tail, is refused
// whenever it is set: re-indexed, it would name other blocks.
func byIndexFollows(c *jlm.Config, order []int) error {
	switch {
	case c.NKVShared != 0:
		return fmt.Errorf("convert: unroll: the last %d blocks reading others' history are named by the block count: %w", c.NKVShared, ErrNotImplemented)
	case c.AttnResBlock != 0:
		return fmt.Errorf("convert: unroll: a residual checkpoint every %d blocks is named by the block index: %w", c.AttnResBlock, ErrNotImplemented)
	}
	for _, p := range []struct {
		what string
		at   func(int) bool
	}{
		{"the sliding window's period", func(li int) bool { return len(c.Windows) == 0 && c.Sliding(li) }},
		{"the leading dense blocks and the mixture's step", func(li int) bool {
			return c.NExpert != 0 && li >= int(c.NDenseLead) && (c.MoEStep <= 1 || (li+1)%int(c.MoEStep) == 0)
		}},
		{"the hash-routed blocks", func(li int) bool { return li < int(c.NHashLayers) }},
	} {
		for b, l := range order {
			if p.at(b) != p.at(l) {
				return fmt.Errorf("convert: unroll: %s says block %d does otherwise than layer %d, which it runs: %w", p.what, b, l, ErrNotImplemented)
			}
		}
	}
	return nil
}

// byLayer is each block's entry of a list of one a layer: the entry of the
// layer the block runs.
func byLayer[T any](of []T, order []int) []T {
	out := make([]T, len(order))
	for b, l := range order {
		out[b] = of[l]
	}
	return out
}

// exprInts is a list of whole numbers an expression evaluated to.
func exprInts(what string, v any) ([]int, error) {
	if v == nil {
		return nil, nil
	}
	list, ok := v.([]any)
	if !ok {
		return nil, fmt.Errorf("convert: %s is %v, not a list", what, v)
	}
	out := make([]int, len(list))
	for i, x := range list {
		u, err := exprUint(fmt.Sprintf("%s[%d]", what, i), x)
		if err != nil {
			return nil, err
		}
		out[i] = int(u)
	}
	return out, nil
}
