package composer

import (
	"testing"

	"github.com/jitllm/jitllm/composer/ir"
)

// TestFlashSplitFillsTheWaves holds flashSplit to the V100's measured
// choices at a 512-row chunk: Llama's 128-wide heads (256 workgroups of 128
// threads, two a unit) and 256-wide (128 of 256 threads holding most of a
// unit's memory, one a unit) take 3, which fills 768 of 800 and 384 of 400
// slots; 64-wide (128 of 256, two a unit in one wave) takes none; a grid of
// 32, under the units, takes 5, one whole wave of 160; a device stating no
// units takes none.
func TestFlashSplitFillsTheWaves(t *testing.T) {
	v100 := Device{Kind: DeviceGPU, Units: 80, Slots: 80 * 2048, Regs: 32, Shared: 96 << 10}
	form := func(group, shared int, space ...int) *Kernel {
		return &Kernel{Work: true, Group: group, Space: space,
			Bufs:  []Buf{{Kind: BufShared, Elem: ir.U32, Size: shared / 4}},
			Nodes: []Node{{Op: OpLoad, Buf: 0}}}
	}
	for _, c := range []struct {
		name string
		k    *Kernel
		d    Device
		want int
	}{
		{"128-wide", form(128, 47212, 16, 16), v100, 3},
		{"256-wide", form(256, 61440, 8, 16), v100, 3},
		{"64-wide", form(256, 49004, 8, 16), v100, 1},
		{"under the units", form(128, 47212, 4, 8), v100, 5},
		{"no units", form(128, 47212, 16, 16), Device{Kind: DeviceGPU, Slots: 80 * 2048}, 1},
	} {
		if got := flashSplit(c.k, c.d); got != c.want {
			t.Errorf("%s: %d workgroups a tile, want %d", c.name, got, c.want)
		}
	}
}

// TestFlashCrossingFollowsTheMerge holds the profile's model to the V100's
// forms at a 512-row chunk, split 3: 128-wide heads (256 workgroups of 64
// rows, 32 positions a block, two a unit) price the split less past 928
// positions and 256-wide (128 of 64 rows, 16 a block, one a unit) past 896,
// where the bench measured both losing at 512 and winning at 1536. A merge
// twice as dear moves the crossing out, and a split whose waves grow no
// slower than the whole's never prices less.
func TestFlashCrossingFollowsTheMerge(t *testing.T) {
	var p Flash
	for _, ps := range gpuProfile().Passes {
		if f, ok := ps.(Flash); ok {
			p = f
		}
	}
	v100 := Device{Kind: DeviceGPU, Units: 80, Slots: 80 * 2048, Regs: 32, Shared: 96 << 10}
	whole := &Kernel{Work: true, Group: 128, Space: []int{16, 16},
		Bufs:  []Buf{{Kind: BufShared, Elem: ir.U32, Size: 47212 / 4}},
		Nodes: []Node{{Op: OpLoad, Buf: 0}}}
	g := flashGeom{TR: 64, KB: 32}
	out := 512 * 32 * 128
	if got := p.crossing(whole, g, 3, out, 256, v100); got != 928 {
		t.Errorf("128-wide: the split prices less past %d positions, want 928", got)
	}
	wide := &Kernel{Work: true, Group: 256, Space: []int{8, 16},
		Bufs:  []Buf{{Kind: BufShared, Elem: ir.U32, Size: 61440 / 4}},
		Nodes: []Node{{Op: OpLoad, Buf: 0}}}
	if got := p.crossing(wide, flashGeom{TR: 64, KB: 16}, 3, 512*16*256, 512, v100); got != 896 {
		t.Errorf("256-wide: the split prices less past %d positions, want 896", got)
	}
	dear := p
	dear.Launch, dear.Plane = 2*p.Launch, 2*p.Plane
	if got, at := dear.crossing(whole, g, 3, out, 256, v100), p.crossing(whole, g, 3, out, 256, v100); got <= at {
		t.Errorf("a merge twice as dear crosses at %d positions, the profile's at %d", got, at)
	}
	if got := p.crossing(whole, g, 2, out, 256, v100); got != -1 {
		t.Errorf("a split of 2, four waves against two, crosses at %d positions: want never", got)
	}
}
