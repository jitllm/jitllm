package graph

import (
	"slices"
	"testing"
)

// TestHistoriesBindByBlockAndIndex: a block binds its own histories, or
// the first other block's it names where it keeps none, at their indices,
// and every other block's history it reads by (block, index) after them,
// each once.
func TestHistoriesBindByBlockAndIndex(t *testing.T) {
	appendTo := func(h uint8) Op {
		return Op{Kind: OpKVAppend, In: []ValueID{0, 0}, Writes: []StateRef{{State: StateKV, Version: 1, Hist: h}}}
	}
	attend := func(rs ...StateRef) Op { return Op{Kind: OpAttend, In: []ValueID{0}, Reads: rs} }
	past := StateRef{State: StateKV, From: 3, Hist: 1}
	for _, c := range []struct {
		name string
		ops  []Op
		src  int
		more []HistoryRead
	}{
		{"own", []Op{appendTo(0), attend(StateRef{State: StateKV, Version: 1})}, -1, nil},
		{"another's", []Op{attend(SharedHistory(2))}, 2, nil},
		{"another's past its rows'", []Op{attend(past)}, 2, nil},
		{"two blocks'", []Op{attend(SharedHistory(2)), attend(SharedHistory(1)), attend(SharedHistory(1))}, 2, []HistoryRead{{1, 0}}},
		{"three blocks', one past its rows'", []Op{attend(SharedHistory(0)), attend(SharedHistory(2)), attend(past)}, 0, []HistoryRead{{2, 0}, {2, 1}}},
		{"another's beside its own rows'", []Op{appendTo(0), attend(SharedHistory(2))}, -1, []HistoryRead{{2, 0}}},
		{"another's beside a history of its own", []Op{appendTo(1), attend(SharedHistory(2))}, -1, []HistoryRead{{2, 0}}},
	} {
		g := &Graph{Block: 5, Values: []Value{{Width: 8}}, Ops: c.ops}
		if n, more := g.ReadsHistoryOf(), g.ReadsMore(); n != c.src || !slices.Equal(more, c.more) {
			t.Errorf("%s: block %d and %v, want block %d and %v", c.name, n, more, c.src, c.more)
		}
		for j, h := range c.more {
			if i := g.MoreIndex(StateRef{State: StateKV, From: h.Block + 1, Hist: uint8(h.Hist)}); i != j {
				t.Errorf("%s: block %d's history %d at %d, want %d", c.name, h.Block, h.Hist, i, j)
			}
		}
		if c.src >= 0 && g.MoreIndex(SharedHistory(c.src)) != -1 {
			t.Errorf("%s: block %d's rows bound past its histories", c.name, c.src)
		}
	}
}

// TestKeptHistoriesFollowTheAppends: each history past the rows' a graph
// appends to, by index, with its period, width and whether its values are
// its keys, as the append states them.
func TestKeptHistoriesFollowTheAppends(t *testing.T) {
	g := &Graph{Values: []Value{{Width: 8}, {Width: 12}, {Width: 12}}, Ops: []Op{
		{Kind: OpKVAppend, In: []ValueID{0, 0}, Writes: []StateRef{{State: StateKV, Version: 1}}},
		{Kind: OpKVAppend, In: []ValueID{1, 2}, Writes: []StateRef{{State: StateKV, Version: 1, Hist: 3}}, Attr: Attrs{Span: 4}},
		{Kind: OpKVAppend, In: []ValueID{0, 0}, Writes: []StateRef{{State: StateKV, Version: 1, Hist: 1}}},
	}}
	want := []KeptHistory{{Hist: 1, Period: 1, Width: 8, ValuesInKeys: true}, {Hist: 3, Period: 4, Width: 12}}
	if got := g.KeptHistories(); !slices.Equal(got, want) {
		t.Fatalf("kept %v, want %v", got, want)
	}
}
