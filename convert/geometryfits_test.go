package convert

import (
	"errors"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestABlockOfAnotherGeometryIsRefused: a block of its own rotary width that
// is not a sliding block turned by the sliding layers' own rotary, and a
// KV-sharing block whose source holds another shape, are refused at
// conversion with ErrNotImplemented; sliding blocks of their own width and
// a KV-sharing block over a source of its shape convert.
func TestABlockOfAnotherGeometryIsRefused(t *testing.T) {
	g := func(hd, rot uint32) jlm.Geom { return jlm.Geom{NKVHead: 2, HeadDim: hd, NRot: rot, NFFN: 256} }
	base := func() *jlm.Config {
		return &jlm.Config{NLayer: 4, NHead: 4, NKVHead: 2, HeadDim: 64, NRot: 64, NFFN: 256}
	}
	for _, tc := range []struct {
		name   string
		change func(c *jlm.Config)
		refuse bool
	}{
		{"one geometry", func(*jlm.Config) {}, false},
		{"sliding blocks of their own width", func(c *jlm.Config) {
			c.SWAWindow, c.SWAPeriod = 128, 4
			c.Geoms = []jlm.Geom{g(64, 32), g(64, 32), g(64, 32), g(64, 64)}
		}, false},
		{"a global block of its own width", func(c *jlm.Config) {
			c.SWAWindow, c.SWAPeriod = 128, 4
			c.Geoms = []jlm.Geom{g(64, 32), g(64, 32), g(64, 32), g(64, 48)}
		}, true},
		{"a block of its own width and no sliding layers", func(c *jlm.Config) {
			c.Geoms = []jlm.Geom{g(64, 64), g(64, 32), g(64, 64), g(64, 64)}
		}, true},
		{"KV sharing over its own shape", func(c *jlm.Config) { c.NKVShared = 1 }, false},
		{"KV sharing over another head width", func(c *jlm.Config) {
			c.NKVShared = 1
			c.Geoms = []jlm.Geom{g(64, 64), g(64, 64), g(128, 64), g(64, 64)}
		}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := base()
			tc.change(c)
			err := geometryFits(&Config{Config: c})
			switch {
			case tc.refuse && !errors.Is(err, ErrNotImplemented):
				t.Fatalf("refused with %v, want ErrNotImplemented", err)
			case !tc.refuse && err != nil:
				t.Fatal(err)
			}
		})
	}
}
