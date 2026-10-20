package backend_test

import (
	"math/rand"
	"runtime"
	"testing"
	"time"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// memReporter is a device that reports its free memory.
type memReporter interface {
	Mem() (free, total uint64, err error)
}

// TestPagedGatesLeakNothing runs every paged gate's harness repeatedly on
// each device and demands that the process holds the same bytes of device
// buffers after the first pass as after the last (the backend's own count, on
// every device, an iGPU included), that a discrete card's free memory does not
// fall steadily (every kernel given back too), and that once the devices are
// closed the process has no more goroutines than it started with.
//
// The free-memory reading is the card's, so it is a quiet card's gate:
// another process testing on it moves the reading by tens of MiB between
// passes, which read as a leak to a check that only asked for a fall.
func TestPagedGatesLeakNothing(t *testing.T) {
	gpuLock(t)
	base := runtime.NumGoroutine()
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	const page = 64
	rows := pagedCases(page)["rows"]
	volta := map[backend.Device]bool{}
	for _, d := range mmaDevices(t, devs, ir.MMAVolta()) {
		volta[d] = true
	}
	pass := func(d backend.Device) {
		s := oldkernels.FlashShape{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: page, Splits: 3, Sink: true}
		p := newPagedPool(rand.New(rand.NewSource(3)), page, s.KVHeads*s.Dim, false, rows)
		if k, err := oldkernels.FlashDecodeKV(s); err == nil && laneOK(t, d, k) {
			if _, err := pagedAttn(t, d, s, true, p, p.tab, rows); err != nil {
				t.Fatal(err)
			}
		}
		st := s
		st.Splits, st.Chunk = 4, 128
		if _, err := pagedStaged(t, d, st, 1, 1, p, p.tab, rows); err != nil {
			t.Fatal(err)
		}
		if err := pagedWriteCheck(t, d, page, 2, 64, 64, true, true, false, []pagedRow{{0, 0, page - 1}, {0, 0, 2*page + 3}, {5, 5, 5}}); err != nil {
			t.Fatal(err)
		}
		// binary16 K through both decode paths and its writers, with vector V
		// loads where the backend lowers them.
		h := s
		h.F16K, h.VecV = true, d.API() != "spirv"
		ph := newPagedPool(rand.New(rand.NewSource(3)), page, h.KVHeads*h.Dim, false, rows).withF16K()
		if k, err := oldkernels.FlashDecodeKV(h); err == nil && laneOK(t, d, k) {
			if _, err := pagedAttn(t, d, h, true, ph, ph.tab, rows); err != nil {
				t.Fatal(err)
			}
		}
		hs := h
		hs.Splits, hs.Chunk = 4, 128
		if _, err := pagedStaged(t, d, hs, 1, 1, ph, ph.tab, rows); err != nil {
			t.Fatal(err)
		}
		if err := pagedWriteCheck(t, d, page, 2, 64, 64, true, false, true, []pagedRow{{0, 0, page - 1}, {0, 0, 2*page + 3}, {5, 5, 5}}); err != nil {
			t.Fatal(err)
		}
		// The prefill paths: the staged one on every device, the flash ones
		// where they lower.
		prows := prefillRows(prefillCases(page, 64)[1])
		ps := oldkernels.FlashShape{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: page, Chunk: 64, Sink: true}
		ps.Splits = prefillSplits(prows, ps.Chunk, 1)
		pp := newPrefillPool(rand.New(rand.NewSource(3)), page, ps.KVHeads*ps.Dim, false, false, false, prows)
		if _, _, err := pagedPrefillStaged(t, d, ps, prefillForm{scoresQT: 2, scoresKT: 2, accQT: 4, smLanes: 1}, pp, pp.tab, pp.desc, prows); err != nil {
			t.Fatal(err)
		}
		fs := oldkernels.FlashPrefill70Shape{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: page, Splits: 2}
		if d.API() == "msl" || volta[d] {
			if _, _, err := pagedFlashPrefill(t, d, fs, true, formOf(d.API() == "msl"), pp, pp.tab, pp.desc, prows); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, d := range devs {
		// A unified-memory device (an iGPU, a software rasteriser) reports the
		// host's memory, which the rest of the machine moves.
		m, ok := d.(memReporter)
		ok = ok && !unifiedDev(d)
		// The bytes of device buffers this process holds after passes 1, 11
		// and 21, by the backend's own count: a pass frees every buffer it
		// allocates, so the count comes back to the same bytes whatever any
		// other process does with the card.
		cnt, counted := d.(backend.AllocCounter)
		var held [3]uint64
		// Free memory after the same passes, which also sees what the count
		// does not (compiled modules, driver objects). The card is shared with
		// the desktop and with any other process testing on it, whose own
		// allocations move the reading. A leak moves it in both halves, and by
		// about the same amount (1.3 MiB a pass here); another process moves
		// it by whatever it allocated.
		var free [3]uint64
		for i := 0; i < 21; i++ {
			pass(d)
			if i%10 == 0 {
				if ok {
					free[i/10], _, _ = m.Mem()
				}
				if counted {
					held[i/10] = cnt.Allocated()
				}
			}
		}
		if counted && (held[1] != held[0] || held[2] != held[1]) {
			t.Errorf("%s/%s: the process holds %d -> %d -> %d bytes of device buffers after passes 1, 11 and 21",
				d.API(), d.Name(), held[0], held[1], held[2])
		}
		if ok {
			a, b := int64(free[0])-int64(free[1]), int64(free[1])-int64(free[2])
			switch {
			case a <= 4<<20 || b <= 4<<20:
			case a < 3*b && b < 3*a:
				t.Errorf("%s/%s: free device memory fell %d then %d KiB over ten passes each", d.API(), d.Name(),
					a>>10, b>>10)
			default:
				t.Logf("%s/%s: free device memory fell %d then %d KiB, not a leak's steady fall: another "+
					"process allocated on the card, so this reading proved nothing", d.API(), d.Name(), a>>10, b>>10)
			}
			t.Logf("%s/%s: free %d -> %d -> %d KiB", d.API(), d.Name(), free[0]>>10, free[1]>>10, free[2]>>10)
		}
		d.Close()
	}
	// Closing a device ends its goroutines; give them a moment to exit.
	deadline := time.Now().Add(5 * time.Second)
	for runtime.NumGoroutine() > base && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if n := runtime.NumGoroutine(); n > base {
		t.Errorf("%d goroutines after closing every device, %d before opening", n, base)
	}
}
