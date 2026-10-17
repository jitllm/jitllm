package jlm_test

import (
	"bytes"
	"errors"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jitllm/jitllm/format/jlm"
)

// svcFixture opens the two-block mixture of expertSource: eight expert pages,
// four per block. pages(e) are bank e's expert pages in expert order.
type svcFixture struct {
	c    *jlm.File
	bank *jlm.Entry
	// pages[x] is expert x's page in block 1's gate bank, then block 0's
	// (indices 0..3 and 4..7 of the eight).
	pages [8]int
	sel   []uint32
}

func openSvc(t *testing.T) *svcFixture {
	t.Helper()
	src, _ := expertSource(t)
	dst := filepath.Join(t.TempDir(), "m.jlm")
	if _, err := jlm.Write(dst, src, jlm.Fingerprint{Host: "t"}); err != nil {
		t.Fatal(err)
	}
	c, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	e, ok := c.Find(jlm.RoleExpGateBank, 1, -1)
	if !ok || !c.ExpertPaged(e) {
		t.Fatal("no expert-paged bank in block 1")
	}
	fx := &svcFixture{c: c, bank: e}
	for x := 0; x < 4; x++ {
		fx.pages[x] = c.ExpertPage(e, x)
	}
	return fx
}

// offsets are the file offsets of each of the bank's expert pages.
func (fx *svcFixture) page(x int) (lo, hi int64) {
	off, n := fx.c.PageOffset(fx.pages[x])
	return off, off + int64(n)
}

func (fx *svcFixture) in(off int64, x int) bool {
	lo, hi := fx.page(x)
	return off >= lo && off < hi
}

// gate blocks every read whose offset lies in a page listed in held until
// release is closed, and says when the first of them started.
type gate struct {
	fx      *svcFixture
	held    map[int]bool
	release chan struct{}
	started chan struct{}
	once    sync.Once
	counts  [4]atomic.Int64
}

func newGate(fx *svcFixture, held ...int) *gate {
	g := &gate{fx: fx, held: map[int]bool{}, release: make(chan struct{}), started: make(chan struct{}, 16)}
	for _, x := range held {
		g.held[x] = true
	}
	fx.c.SetReadHook(g.hook)
	return g
}

func (g *gate) hook(off int64) error {
	for x := 0; x < 4; x++ {
		if g.fx.in(off, x) {
			g.counts[x].Add(1)
			if g.held[x] {
				g.started <- struct{}{}
				<-g.release
			}
		}
	}
	return nil
}

func (g *gate) open() { g.once.Do(func() { close(g.release) }) }

func within(t *testing.T, d time.Duration, what string, fn func()) {
	t.Helper()
	done := make(chan struct{})
	go func() { fn(); close(done) }()
	select {
	case <-done:
	case <-time.After(d):
		t.Fatalf("%s did not complete in %v", what, d)
	}
}

func waitStarted(t *testing.T, g *gate, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		select {
		case <-g.started:
		case <-time.After(5 * time.Second):
			t.Fatalf("only %d of %d held reads started", i, n)
		}
	}
}

// TestDemandDoesNotWaitForASlowGuess holds a speculative read of one page
// open and demands a different, missing page: the demand must complete while
// the guess is still in flight.
func TestDemandDoesNotWaitForASlowGuess(t *testing.T) {
	fx := openSvc(t)
	g := newGate(fx, 0)
	defer g.open()
	fx.c.Speculate(fx.bank, []uint32{0})
	waitStarted(t, g, 1)
	within(t, 3*time.Second, "a demanded read beside a slow unused guess", func() {
		if err := fx.c.EnsureExperts(fx.bank, []uint32{2}); err != nil {
			t.Error(err)
		}
	})
	if g.counts[2].Load() == 0 {
		t.Fatal("the demanded page was never read")
	}
}

// TestDemandPromotesAQueuedGuess fills every speculative worker with a held
// read of block 0's experts, queues block 1's guesses behind them, and demands
// one of the queued ones: the demand reads it at once, and the queued record
// then does nothing (no page is read twice).
func TestDemandPromotesAQueuedGuess(t *testing.T) {
	fx := openSvc(t)
	e0, _ := fx.c.Find(jlm.RoleExpGateBank, 0, -1)
	promoted := fx.pages[3]
	release := make(chan struct{})
	var blocked, reads atomic.Int64
	fx.c.SetReadHook(func(off int64) error {
		lo, n := fx.c.PageOffset(promoted)
		if off >= lo && off < lo+int64(n) {
			reads.Add(1)
			return nil
		}
		blocked.Add(1)
		<-release
		return nil
	})
	var once sync.Once
	open := func() { once.Do(func() { close(release) }) }
	defer open()
	fx.c.Speculate(e0, []uint32{0, 1, 2, 3})
	waitBlocked(t, &blocked, 4)
	fx.c.Speculate(fx.bank, []uint32{0, 1, 3})
	within(t, 3*time.Second, "a demand for a queued guess", func() {
		if err := fx.c.EnsureExperts(fx.bank, []uint32{3}); err != nil {
			t.Error(err)
		}
	})
	open()
	time.Sleep(100 * time.Millisecond)
	if n := reads.Load(); n != 1 {
		t.Fatalf("the promoted page was read %d times, want 1", n)
	}
}

func waitBlocked(t *testing.T, n *atomic.Int64, want int64) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for n.Load() < want {
		if time.Now().After(deadline) {
			t.Fatalf("only %d of %d reads reached the hook", n.Load(), want)
		}
		time.Sleep(time.Millisecond)
	}
}

// TestAJoinedPageIsReadOnce has many demanders and a guess ask for the same
// missing page at once: it is read exactly once.
func TestAJoinedPageIsReadOnce(t *testing.T) {
	fx := openSvc(t)
	g := newGate(fx, 1)
	defer g.open()
	fx.c.Speculate(fx.bank, []uint32{1})
	waitStarted(t, g, 1)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := fx.c.EnsureExperts(fx.bank, []uint32{1}); err != nil {
				t.Error(err)
			}
		}()
	}
	time.Sleep(50 * time.Millisecond)
	g.open()
	within(t, 3*time.Second, "the joiners", wg.Wait)
	if n := g.counts[1].Load(); n != 1 {
		t.Fatalf("the page was read %d times, want 1", n)
	}
}

// TestAFrameBeingReadIsNotEvicted holds a guess's read open with a budget of
// two expert pages and demands three others: the pinned frame is never
// handed to another page, and every page ends with its own bytes.
func TestAFrameBeingReadIsNotEvicted(t *testing.T) {
	fx := openSvc(t)
	_, size := fx.c.PageOffset(fx.pages[0])
	fx.c.SetBudget(2 * size)
	g := newGate(fx, 0)
	defer g.open()
	fx.c.Speculate(fx.bank, []uint32{0})
	waitStarted(t, g, 1)
	for _, x := range []uint32{1, 2, 3, 1, 2} {
		if err := fx.c.EnsureExperts(fx.bank, []uint32{x}); err != nil {
			t.Fatal(err)
		}
	}
	g.open()
	time.Sleep(100 * time.Millisecond)
	// Whatever is resident now holds its own bytes: a frame handed to
	// another page mid-read is filled with the first page's.
	resident := 0
	for x := 0; x < 4; x++ {
		if qs, _, _ := fx.c.Sheet(fx.bank, x); qs != nil {
			resident++
			if !bytes.Equal(qs, referenceSheet(t, fx, x)) {
				t.Errorf("expert %d: the frame holds another page's bytes", x)
			}
		}
	}
	if resident == 0 {
		t.Fatal("no page is resident to check")
	}
	for x := 0; x < 4; x++ {
		if err := fx.c.EnsureExperts(fx.bank, []uint32{uint32(x)}); err != nil {
			t.Fatal(err)
		}
		lease, err := fx.c.Hold(fx.pages[x], nil)
		if err != nil {
			t.Fatal(err)
		}
		qs, _, _ := fx.c.Sheet(fx.bank, x)
		want := referenceSheet(t, fx, x)
		if !bytes.Equal(qs, want) {
			t.Errorf("expert %d: the frame holds another page's bytes", x)
		}
		lease.Release()
	}
}

func referenceSheet(t *testing.T, fx *svcFixture, x int) []byte {
	t.Helper()
	ref, err := jlm.Open(fx.c.Path)
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Close()
	e, _ := ref.Find(jlm.RoleExpGateBank, 1, -1)
	if err := ref.EnsurePage(ref.ExpertPage(e, x)); err != nil {
		t.Fatal(err)
	}
	qs, _, _ := ref.Sheet(e, x)
	return append([]byte(nil), qs...)
}

// TestCloseWaitsForReadsInFlight closes the container while a guess is being
// read: Close returns only after the read ends.
func TestCloseWaitsForReadsInFlight(t *testing.T) {
	fx := openSvc(t)
	g := newGate(fx, 0)
	defer g.open()
	fx.c.Speculate(fx.bank, []uint32{0})
	waitStarted(t, g, 1)
	closed := make(chan error, 1)
	go func() { closed <- fx.c.Close() }()
	select {
	case <-closed:
		t.Fatal("Close returned while a read was in flight")
	case <-time.After(100 * time.Millisecond):
	}
	g.open()
	select {
	case err := <-closed:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Close did not return after the read ended")
	}
}

// TestSpeculativeErrorsDoNotFailATokenButDemandedOnesDo fails the reads of one
// page: a guess of it fails quietly, a demand for it returns the error, and
// a demand that joined the failing guess reports its own read's error rather
// than hanging or inheriting a silent success.
func TestSpeculativeErrorsDoNotFailATokenButDemandedOnesDo(t *testing.T) {
	fx := openSvc(t)
	boom := errors.New("boom")
	var fail atomic.Bool
	fail.Store(true)
	var started = make(chan struct{}, 4)
	release := make(chan struct{})
	fx.c.SetReadHook(func(off int64) error {
		if fx.in(off, 2) {
			started <- struct{}{}
			if fail.Load() {
				<-release
				return boom
			}
		}
		return nil
	})
	fx.c.Speculate(fx.bank, []uint32{2})
	<-started
	res := make(chan error, 1)
	go func() { res <- fx.c.EnsureExperts(fx.bank, []uint32{2}) }()
	time.Sleep(50 * time.Millisecond)
	close(release)
	select {
	case err := <-res:
		if !errors.Is(err, boom) {
			t.Fatalf("a demand joined to a failed guess returned %v, want its own read's error", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("a demand joined to a failed guess hung")
	}
	if _, failed := fx.c.SpeculationStats(); failed != 1 {
		t.Fatalf("%d failed guesses counted, want 1", failed)
	}
	// The failure did not poison the page or the service.
	fail.Store(false)
	if err := fx.c.EnsureExperts(fx.bank, []uint32{2, 3}); err != nil {
		t.Fatal(err)
	}
	fx.c.Speculate(fx.bank, []uint32{0})
	if err := fx.c.EnsureExperts(fx.bank, []uint32{0}); err != nil {
		t.Fatal(err)
	}
}

// TestSpeculativeOccupancyIsBounded caps the frames guesses may hold at two
// and speculates over four pages: no more than two are held.
func TestSpeculativeOccupancyIsBounded(t *testing.T) {
	fx := openSvc(t)
	fx.c.SetSpeculation(2)
	fx.c.Speculate(fx.bank, []uint32{0, 1, 2, 3})
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if skipped, _ := fx.c.SpeculationStats(); skipped >= 2 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if got := fx.c.SpeculationHeld(); got != 2 {
		t.Fatalf("%d frames held by guesses, want the cap of 2", got)
	}
	// A demand takes a page off the charge.
	if err := fx.c.EnsureExperts(fx.bank, []uint32{0, 1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if got := fx.c.SpeculationHeld(); got != 0 {
		t.Fatalf("%d frames still charged after every page was demanded", got)
	}
}

// TestWarmSelectionAllocatesNothing is the warm path of the decode token:
// demanding resident pages and guessing resident pages costs no allocation.
func TestWarmSelectionAllocatesNothing(t *testing.T) {
	fx := openSvc(t)
	sel := []uint32{0, 1, 2, 3}
	if err := fx.c.EnsureExperts(fx.bank, sel); err != nil {
		t.Fatal(err)
	}
	fx.c.Speculate(fx.bank, sel)
	if n := testing.AllocsPerRun(50, func() {
		if err := fx.c.EnsureExperts(fx.bank, sel); err != nil {
			t.Fatal(err)
		}
		fx.c.Speculate(fx.bank, sel)
		fx.c.Abandon()
	}); n != 0 {
		t.Fatalf("a warm selection allocated %v times", n)
	}
}
