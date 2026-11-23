package gpu

import (
	"strings"
	"testing"
)

// TestABadKVPageIsRefusedAtOpen: a page size off the contract -- not a power
// of two, under 64, or negative -- is refused when the tier is made, naming
// it, rather than faulting at the first step; one on it opens.
func TestABadKVPageIsRefusedAtOpen(t *testing.T) {
	for _, p := range []int{0, 64, 128, 256, 512, 32, 96, 100, -64} {
		g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 1 << 40}}, WithDeviceTune(TuneOff),
			WithConfig(func(c *Config) { c.KVPage = p }))
		good := p == 0 || p >= 64 && p&(p-1) == 0
		switch {
		case good && err != nil:
			t.Errorf("KVPage %d was refused: %v", p, err)
		case !good && err == nil:
			t.Errorf("KVPage %d opened", p)
		case !good && !strings.Contains(err.Error(), "KVPage"):
			t.Errorf("KVPage %d was refused without naming it: %v", p, err)
		}
		if g != nil {
			g.Close()
		}
	}
}
