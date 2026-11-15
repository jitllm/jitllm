package graph

import (
	"slices"
	"testing"
)

// TestReadsHistoryOfRefusesWhatItCannotName: a block reads its own
// histories or one other block's, any of them, in their place; reading
// two blocks' or another's beside one of its own is refused.
func TestReadsHistoryOfRefusesWhatItCannotName(t *testing.T) {
	appendTo := func(h uint8) Op {
		return Op{Kind: OpKVAppend, In: []ValueID{0, 0}, Writes: []StateRef{{State: StateKV, Version: 1, Hist: h}}}
	}
	attend := func(rs ...StateRef) Op { return Op{Kind: OpAttend, In: []ValueID{0}, Reads: rs} }
	for _, c := range []struct {
		name string
		ops  []Op
		want int
		bad  bool
	}{
		{"own", []Op{appendTo(0), attend(StateRef{State: StateKV, Version: 1})}, -1, false},
		{"another's", []Op{attend(SharedHistory(2))}, 2, false},
		{"another's past its rows'", []Op{attend(StateRef{State: StateKV, From: 3, Hist: 1})}, 2, false},
		{"two blocks'", []Op{attend(SharedHistory(2)), attend(SharedHistory(1))}, -1, true},
		{"another's beside its own rows'", []Op{appendTo(0), attend(SharedHistory(2))}, -1, true},
		{"another's beside a history of its own", []Op{appendTo(1), attend(SharedHistory(2))}, -1, true},
	} {
		g := &Graph{Block: 5, Values: []Value{{Width: 8}}, Ops: c.ops}
		n, err := g.ReadsHistoryOf()
		if (err != nil) != c.bad || n != c.want {
			t.Errorf("%s: block %d, error %v; want block %d, refused %v", c.name, n, err, c.want, c.bad)
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
