package composer

import (
	"fmt"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/graph"
)

// TestSplitChunksHoldWholeAlignments forms split attentions for a card at
// subgroups of 32, 16, 8 and 1 lanes and stops after SplitRange: every
// partition's chunk is a whole number of its positions' alignment, and its
// partitions reach every key. A chunk short of a whole alignment leaves a
// partition's share past the positions its threads compute, which at one
// lane put the split attention's output at 4e30.
func TestSplitChunksHoldWholeAlignments(t *testing.T) {
	r := Default()
	split := 0
	for _, lanes := range []int{32, 16, 8, 1} {
		dev := Device{Kind: DeviceGPU, Slots: 80 * 2048, Lanes: lanes, Units: 80}
		prof := r.Profiles[DeviceGPU].on(dev)
		i := slices.IndexFunc(prof.Passes, func(p Pass) bool { return p.Name() == SplitRange{}.Name() })
		if i < 0 {
			t.Fatal("the card's profile runs no SplitRange")
		}
		prof.Passes = prof.Passes[:i+1]
		for _, sh := range [][3]int{{6, 6, 48}, {16, 8, 128}, {8, 1, 256}, {32, 8, 64}, {14, 2, 64}} {
			for _, keys := range []int{256, 4096, 32768} {
				name := fmt.Sprintf("%d lanes %d:%dx%d at %d keys", lanes, sh[0], sh[1], sh[2], keys)
				b := graph.New()
				n := b.KVAppend(b.In("k", sh[1]*sh[2]), b.In("v", sh[1]*sh[2]), 0)
				b.Out(b.Attend(b.In("q", sh[0]*sh[2]), n, graph.Attrs{NHead: sh[0], NKVHead: sh[1], HeadDim: sh[2], Scale: 0.1}))
				ks, err := r.formWith(Request{Graph: b.G, Step: Step{Rows: 1, Page: 128, Keys: keys, ActWindow: 32}, Device: dev,
					History: History{Keys: Pool{ByDim: true}, Values: Pool{Half: true}}}, prof, nil)
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				for _, k := range ks {
					p := k.Positions
					if !slices.Contains(k.Passes, SplitRange{}.Name()) || p == nil || p.Align <= 1 {
						continue
					}
					var S, J int
					switch {
					case p.Axis > 0:
						S, J = k.Space[p.Axis-1], k.Space[p.Axis]
					case p.Loop >= 0:
						S, J = k.Space[len(k.Space)-1], k.Loops[p.Loop].Trip
					default:
						continue
					}
					split++
					if J%p.Align != 0 {
						t.Errorf("%s: %s takes a chunk of %d positions, not a whole number of their alignment %d", name, k.Name, J, p.Align)
					}
					if S*J < keys {
						t.Errorf("%s: %s takes %d partitions of %d positions for %d keys", name, k.Name, S, J, keys)
					}
				}
			}
		}
	}
	if split == 0 {
		t.Fatal("no kernel was split: the forms never reach SplitRange's partitions")
	}
	t.Logf("%d split kernels", split)
}
