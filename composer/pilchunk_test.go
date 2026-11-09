package composer

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/graph"
)

// TestPILChunksAddToTheWholeWindow holds each band's taps (PILChunk) to
// the one pass's (PILTapsAt): over bands of every width from 1 to past
// the window, an output's weights at each source sample are the whole
// window's, met in exactly one band, and a band past the window holds none.
func TestPILChunksAddToTheWholeWindow(t *testing.T) {
	for _, c := range []struct {
		in, out int
		filter  uint8
		torch   bool
	}{
		{97, 32, graph.FilterBilinear, false}, {97, 32, graph.FilterCubic, false}, {300, 7, graph.FilterCubic, false},
		{1000, 3, graph.FilterCubic, true}, {40, 90, graph.FilterCubic, false}, {61, 61, graph.FilterBilinear, false},
	} {
		K := PILTapCount(c.filter, c.in, c.out)
		E := PILEntryWords(K)
		whole := make([]uint32, c.out*E)
		PILTapsAt(c.filter, c.torch, c.in, c.out, 0, c.out, K, whole)
		for _, band := range []int{1, 2, 3, 7, K - 1, K, 2 * K} {
			if band < 1 {
				continue
			}
			got := make([][]int32, c.out)
			for o := range got {
				got[o] = make([]int32, c.in)
			}
			chunk := make([]uint32, c.out*PILEntryWords(band))
			for lo := 0; lo < c.in; lo += band {
				hi := min(lo+band, c.in)
				PILChunk(c.filter, PILPrec(c.filter, c.torch, c.in, c.out), c.in, c.out, 0, c.out, band, lo, hi, chunk)
				for o := range c.out {
					e := chunk[o*PILEntryWords(band):]
					first, count := int(e[0]), int(e[1])
					if count > 0 && (first < lo || first+count > hi) {
						t.Fatalf("%d to %d, bands of %d: output %d's taps %d+%d leave the band %d-%d", c.in, c.out, band, o, first, count, lo, hi)
					}
					for x := range count {
						got[o][first+x] += int32(e[2+x])
					}
				}
			}
			for o := range c.out {
				e := whole[o*E:]
				first, count := int(e[0]), int(e[1])
				for x := range c.in {
					var want int32
					if x >= first && x < first+count {
						want = int32(e[2+x-first])
					}
					if got[o][x] != want {
						t.Fatalf("%d to %d (filter %d, torch %v), bands of %d: output %d at sample %d reads %d in bands, %d whole",
							c.in, c.out, c.filter, c.torch, band, o, x, got[o][x], want)
					}
				}
			}
		}
		t.Log(fmt.Sprintf("%d to %d filter %d torch %v: every band width's weights add to the whole window's", c.in, c.out, c.filter, c.torch))
	}
}
