package jlm

import (
	"slices"
	"strings"
	"testing"
)

// TestStatedActivationsRoundTrip writes stated activations -- a map of every
// operand kind, a reduction in tree order, a per-layer index -- into the
// config record and reads them back term for term, beside a model that
// states none; and refuses a record whose terms read a later term, a
// reduction's value as the expression's, or an index past the list.
func TestStatedActivationsRoundTrip(t *testing.T) {
	in := func(i uint32) Arg { return Arg{Kind: ArgIn, Index: i} }
	at := func(i uint32) Arg { return Arg{Kind: ArgTerm, Index: i} }
	acts := []Expr{
		{Terms: []Term{
			{Fn: FnAdd, Reduce: true, Tree: true, Args: [3]Arg{in(0)}},
			{Fn: FnDiv, Args: [3]Arg{at(0), {Kind: ArgWidth}}},
			{Fn: FnWhere, Args: [3]Arg{in(0), {Kind: ArgStored, Role: RoleXIELU, Index: 3}, {Kind: ArgConst, Value: -0.75}}},
			{Fn: FnFma, Args: [3]Arg{at(2), at(1), in(1)}},
		}},
		{Terms: []Term{{Fn: FnTanh, Args: [3]Arg{in(0)}}, {Fn: FnMul, Args: [3]Arg{at(0), in(1)}}}},
	}
	c := &Config{Arch: ArchLlama, NLayer: 3, Acts: acts, ActLayers: []uint8{1, 0, 1}}
	got, err := decodeConfig(encodeConfig(c))
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Acts) != len(acts) || !slices.Equal(got.ActLayers, c.ActLayers) {
		t.Fatalf("%d activations for layers %v read back", len(got.Acts), got.ActLayers)
	}
	for i := range acts {
		if !slices.Equal(got.Acts[i].Terms, acts[i].Terms) {
			t.Fatalf("activation %d reads back as %v", i, got.Acts[i].Terms)
		}
	}
	if e := got.ActAt(0); e == nil || len(e.Terms) != 2 {
		t.Fatalf("layer 0 reads %v, want the second activation", e)
	}
	if none, err := decodeConfig(encodeConfig(&Config{Arch: ArchLlama, NLayer: 3})); err != nil || none.ActAt(0) != nil {
		t.Fatalf("a model that states no activation reads back %v (%v)", none.ActAt(0), err)
	}
	for _, bad := range []struct {
		name string
		c    Config
		want string
	}{
		{"a later term", Config{Arch: ArchLlama, NLayer: 1, Acts: []Expr{{Terms: []Term{{Fn: FnExp, Args: [3]Arg{at(0)}}}}}}, "earlier"},
		{"a reduction's value", Config{Arch: ArchLlama, NLayer: 1, Acts: []Expr{{Terms: []Term{{Fn: FnAdd, Reduce: true, Args: [3]Arg{in(0)}}}}}}, "reduction"},
		{"an index past the list", Config{Arch: ArchLlama, NLayer: 1, Acts: acts[1:], ActLayers: []uint8{1}}, "activation 1 of 1"},
	} {
		if _, err := decodeConfig(encodeConfig(&bad.c)); err == nil || !strings.Contains(err.Error(), bad.want) {
			t.Errorf("%s: read back with %v, want a refusal naming %q", bad.name, err, bad.want)
		}
	}
}
