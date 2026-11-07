package convert

import (
	"errors"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestASelectionOffTheHeadIsRefused: outside an MLA model the indexer's key
// is cached as one more head of the block's history, so the converter
// states a selection only where the key is one head -- its width the
// head's, one group or one a kv head, every block at one geometry, no KV
// sharing -- and refuses any other with ErrNotImplemented, where there is a
// file in front of a person. An MLA model's key is held apart and is never
// refused for it.
func TestASelectionOffTheHeadIsRefused(t *testing.T) {
	base := func() *jlm.Config {
		return &jlm.Config{NLayer: 4, NHead: 4, NKVHead: 2, HeadDim: 64, NRot: 64,
			IdxHeads: 2, IdxHeadDim: 64, IdxTopK: 8, IdxBlock: 4, IdxLocal: 1, RMSEps: 1e-6}
	}
	stores := func(jlm.Role, int32) bool { return true }
	for _, tc := range []struct {
		name   string
		change func(c *jlm.Config)
		refuse bool
	}{
		{"one head", func(*jlm.Config) {}, false},
		{"one group", func(c *jlm.Config) { c.IdxHeads = 1 }, false},
		{"a narrower key", func(c *jlm.Config) { c.IdxHeadDim = 32 }, true},
		{"three groups over two kv heads", func(c *jlm.Config) { c.IdxHeads = 3 }, true},
		{"KV sharing", func(c *jlm.Config) { c.NKVShared = 1 }, true},
		{"blocks of two geometries", func(c *jlm.Config) {
			c.Geoms = []jlm.Geom{{NKVHead: 2, HeadDim: 64, NRot: 64}, {NKVHead: 2, HeadDim: 128, NRot: 128},
				{NKVHead: 2, HeadDim: 64, NRot: 64}, {NKVHead: 2, HeadDim: 64, NRot: 64}}
		}, true},
		{"an MLA model's narrower key", func(c *jlm.Config) {
			c.IdxBlock, c.IdxLocal, c.KVLoraRank, c.IdxHeadDim = 0, 0, 16, 32
		}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := base()
			tc.change(c)
			err := StateSelection(c, stores)
			if !tc.refuse {
				if err != nil {
					t.Fatal(err)
				}
				if c.Selection == nil {
					t.Fatal("no selection stated")
				}
				return
			}
			if !errors.Is(err, ErrNotImplemented) {
				t.Fatalf("refused with %v, want ErrNotImplemented", err)
			}
			if c.Selection != nil {
				t.Fatal("a refused selection was stated all the same")
			}
		})
	}
}
