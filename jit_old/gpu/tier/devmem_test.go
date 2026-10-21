//go:build linux

package tier

import (
	"sort"
	"testing"
)

// TestPagingReturnsEveryDeviceBuffer: eviction counters say nothing about
// frees, and a tier that leaked every evicted byte would otherwise pass the
// suite until the card ran out much later. So it runs a real paging workload
// against a budget too small to hold it, closes the tier, and requires the
// ledger to balance in count and in bytes.
func TestPagingReturnsEveryDeviceBuffer(t *testing.T) {
	p := pagePlan()
	const fit = 3
	d := &fakeDev{}
	g, err := New([]Slot{{Dev: d, Bytes: cardFor(t, p, fit)}},
		WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	const blocks = 6
	streamBlocks(g, blocks)
	ws := pageBlocks(p, blocks)
	for li := 0; li < blocks; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("block %d declined: %s", li, g.Err())
		}
	}
	// Drive enough traffic that eviction and re-admission actually happen.
	for round := 0; round < 3; round++ {
		for li := 0; li < blocks; li++ {
			g.PrepLayer(li, p, ws[li])
		}
	}
	g.Close()

	if d.doubleFree != 0 {
		t.Errorf("%d buffer(s) freed twice: the ledger below cannot be trusted", d.doubleFree)
	}
	if d.freed != d.bufs || d.freedBytes != d.allocBytes {
		by := map[string][2]int{}
		for _, b := range d.liveBufs {
			e := by[b.site]
			by[b.site] = [2]int{e[0] + 1, e[1] + b.n}
		}
		sites := make([]string, 0, len(by))
		for k := range by {
			sites = append(sites, k)
		}
		sort.Slice(sites, func(i, j int) bool { return by[sites[i]][1] > by[sites[j]][1] })
		msg := ""
		for _, k := range sites {
			msg += "\n    " + k + ": " + itoaT(by[k][0]) + " buffers, " + itoaT(by[k][1]) + " bytes"
		}
		t.Fatalf("device memory did not come back after Close:\n"+
			"  allocated %d buffers / %d bytes\n"+
			"  freed     %d buffers / %d bytes\n"+
			"  LEAKED    %d buffers / %d bytes, by allocation site:%s",
			d.bufs, d.allocBytes, d.freed, d.freedBytes,
			d.bufs-d.freed, d.allocBytes-d.freedBytes, msg)
	}
	t.Logf("%d buffers / %d bytes allocated and all returned across %d blocks in %d slots",
		d.bufs, d.allocBytes, blocks, fit)
}

func itoaT(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
