package jlm

import (
	"reflect"
	"strings"
	"testing"
)

// TestSelectionRoundTrips holds the blocks' stated selection to the config's
// tails' contract: it round-trips, a config without one writes no tail of
// it, and a selection naming blocks, programs or a rank it does not have is
// refused rather than read.
func TestSelectionRoundTrips(t *testing.T) {
	base := func() *Config {
		return &Config{Arch: ArchLlama, NLayer: 2, NEmbd: 64, NHead: 4, NKVHead: 2,
			HeadDim: 16, NRot: 16, NFFN: 128, NVocab: 1000, RMSEps: 1e-5, RopeBase: 1e4}
	}
	stated := func() *Config {
		c := base()
		c.Selection = &Selection{Blocks: []bool{false, true},
			Key: &Program{NIn: 1, Ops: []TowerOp{{Code: TOpLinear, In: []uint16{0}, Bind: []Role{RoleIdxK}},
				{Code: TOpRMSNorm, In: []uint16{1}, Bind: []Role{RoleIdxKNorm}, Eps: 1e-5},
				{Code: TOpRope, In: []uint16{2}, Span: 16}}, Out: 3},
			Query: &Program{NIn: 2, Ops: []TowerOp{{Code: TOpLinear, In: []uint16{0}, Bind: []Role{RoleIdxQ}},
				{Code: TOpRope, In: []uint16{2}, Span: 16, Chain: RopeNeox}}, Out: 3},
			KeyWidth: 16, Heads: 2, HeadDim: 16, Groups: 2, Rank: RankBest, Block: 4, Keep: 2, Local: 1}
		return c
	}
	got, err := decodeConfig(encodeConfig(stated()))
	if err != nil {
		t.Fatal(err)
	}
	if want := stated().Selection; !reflect.DeepEqual(got.Selection, want) {
		t.Fatalf("round trip:\n got %+v\nwant %+v", got.Selection, want)
	}
	if plain, _ := decodeConfig(encodeConfig(base())); plain.Selection != nil {
		t.Fatalf("a config without a selection read as %+v", plain.Selection)
	}
	for _, v := range []struct {
		name, want string
		bad        func(c *Config)
	}{
		{"a block too few", "states 1 blocks of 2", func(c *Config) { c.Selection.Blocks = c.Selection.Blocks[:1] }},
		{"no key", "no key", func(c *Config) { c.Selection.Key = nil }},
		{"a query of one input", "programs read", func(c *Config) { c.Selection.Query.NIn = 1 }},
		{"an unknown rank", "rank 3", func(c *Config) { c.Selection.Rank = 3 }},
		{"more local blocks than it keeps", "keeping 2 blocks of 4, 3 local", func(c *Config) { c.Selection.Local = 3 }},
		{"a key narrower than a head", "keys 8 wide", func(c *Config) { c.Selection.KeyWidth = 8 }},
		{"a rotary of no head", "turns heads of 0", func(c *Config) { c.Selection.Key.Ops[2].Span = 0 }},
	} {
		c := stated()
		v.bad(c)
		if _, err := decodeConfig(encodeConfig(c)); err == nil || !strings.Contains(err.Error(), v.want) {
			t.Errorf("%s: %v, want an error naming %q", v.name, err, v.want)
		}
	}
}
