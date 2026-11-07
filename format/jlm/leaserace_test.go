package jlm

import (
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

// newReal builds a File over a real temp file, because a lease's whole job is
// to protect a frame across an actual read and newFake has no handle to read
// through. Block i's page is filled with byte(i+1), so a frame that came back
// off the free list holding someone else's bytes is visible by inspection.
func newReal(t *testing.T, nblocks int, page, chunk uint64) *Frames {
	t.Helper()
	const dataOff = uint64(1) << 20
	p := filepath.Join(t.TempDir(), "fake.jlm")
	buf := make([]byte, dataOff+uint64(nblocks)*page)
	for i := 0; i < nblocks; i++ {
		lo := dataOff + uint64(i)*page
		for j := uint64(0); j < page; j++ {
			buf[lo+j] = byte(i + 1)
		}
	}
	if err := os.WriteFile(p, buf, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	h, err := os.Open(p)
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	t.Cleanup(func() { h.Close() })
	f := &Frames{frameCore: &frameCore{file: &File{
		Path: p,
		H: &Header{
			PageSize: page, NBlocks: uint32(nblocks),
			VisPageSize: Align, NVisBlocks: 0,
			DataOff: dataOff, VisDataOff: dataOff + uint64(nblocks)*page,
		},
		f: h,
	}, chunk: chunk}, bit: 1}
	f.views = []*Frames{f}
	f.pages = make([][]byte, nblocks)
	f.holders = make([]uint64, nblocks)
	f.free = map[uint64][][]byte{}
	f.recent = -1
	return f
}

// look reports, under the lock, whether chunk c of block b is marked filled and
// what byte actually sits in it. Both halves matter: a recycled frame can carry
// the right mark with the wrong bytes, and a fresh frame the right bytes with
// no mark.
func (f *Frames) look(b, c int) (filled bool, got byte, resident bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.pages[b] == nil {
		return false, 0, false
	}
	if f.filled != nil && f.filled[b] != nil && c < len(f.filled[b]) {
		filled = f.filled[b][c]
	}
	return filled, f.pages[b][uint64(c)*f.chunkSize()], true
}

// TestHoldPinsBeforeItReads is the violation gate for the order inside Hold.
// EnsureRanges drops f.mu for its I/O, so reading then pinning lets an
// evictor take the frame in between; claim() then hands back a recycled
// buffer with an all-false bitmap and Hold pins another block's memory. It
// has to be concurrent to open the window, but the assertion is not
// timing-dependent: every Hold that returns nil must leave its ranges filled
// with this block's own bytes.
func TestHoldPinsBeforeItReads(t *testing.T) {
	const (
		nblocks = 8
		chunk   = uint64(4096)
		page    = 8 * chunk
		rounds  = 4000
	)
	f := newReal(t, nblocks, page, chunk)
	f.SetBudget(page) // ONE frame for eight blocks: every fault evicts

	base, _ := f.file.pageAt(0)
	const c = 3
	rs := []Range{{Off: base + uint64(c)*chunk, N: chunk}}

	var stop atomic.Bool
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { // the evictor: block 0 is the only other resident page
		defer wg.Done()
		for i := 0; !stop.Load(); i++ {
			b := 1 + i%(nblocks-1)
			off, _ := f.file.pageAt(b)
			f.EnsureRanges(b, []Range{{Off: off, N: chunk}})
			f.DropPage(b)
		}
	}()

	bad := 0
	for i := 0; i < rounds; i++ {
		l, err := f.Hold(0, rs)
		if err != nil {
			stop.Store(true)
			wg.Wait()
			t.Fatalf("round %d: Hold: %v", i, err)
		}
		filled, got, res := f.look(0, c)
		if !res || !filled || got != 1 {
			if bad < 3 {
				t.Errorf("round %d: Hold returned a lease whose range is not "+
					"readable: resident=%v filled=%v byte=%d (want resident, "+
					"filled, byte 1) -- the frame was evicted between the read "+
					"and the pin", i, res, filled, got)
			}
			bad++
		}
		l.Release()
	}
	stop.Store(true)
	wg.Wait()
	if bad > 0 {
		t.Errorf("%d of %d leases were handed out over a frame that had been "+
			"taken", bad, rounds)
	}
}
