package tier

import (
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/ir"
)

// The graph bookkeeping, on a fake device. A graph re-captured every token
// computes the right answer and only costs more, so the invariants pinned here
// are counts and sequences: capture once, replay after, and replay the
// launches the direct path would have issued. The fake records launches
// rather than running them, which is the only way to compare the sequences.

type fakeLaunch struct {
	kern          int
	groups, width int
	bufs          []int
}

type fakeDev struct {
	kerns, bufs int
	// freed, allocBytes and freedBytes make "did the memory come back"
	// checkable. doubleFree is separate so a second free cannot balance the
	// ledger it corrupts.
	freed      int
	allocBytes int
	freedBytes int
	doubleFree int
	liveBufs   map[int]*fakeBuf
	// allocMax refuses any Alloc above this many bytes when non-zero, as a
	// full card does. See TestGrowFailureDoesNotPoisonTheBuffer.
	allocMax int
	// capBytes refuses an Alloc that would take the live bytes past it when
	// non-zero: a card of that size, which a budget alone cannot model (the
	// budget is the tier's arithmetic; this is the driver's answer).
	capBytes int
	// page is the allocation granularity Footprint rounds to; 0 is none.
	page      int
	log       []fakeLaunch // what the session issued directly
	graphs    map[int][]fakeLaunch
	nextGraph int
	live      int // recordings made and not freed
	// closed counts Close calls, so a device closed twice and one never
	// closed are both visible.
	closed int
	// unified makes this device report UnifiedMemory like an integrated GPU.
	// See poolsFor.
	unified bool
	// launchCost is how long a launch takes, so order() has something to
	// measure.
	launchCost time.Duration
	// irs is every kernel compiled, in order, so a gate can read what the
	// tier emitted.
	irs []*ir.Kernel
	// inSession is how deep this device is inside Session callbacks, and
	// failCapture makes every launch inside a Record fail. Together they hold
	// the fake to CUDA's rule that freeing a Recording (or any buffer) inside
	// a session deadlocks the owner goroutine; the fake panics instead.
	inSession   int
	failCapture bool
	// failCompile makes every Compile fail, as a driver that rejects the
	// emitted code does. See TestCompileFailureIsCounted.
	failCompile bool
	// name distinguishes two fakes in a failure message; "fake" when unset.
	name string
	// api is what API() reports; "fake" when unset. Two fakes with different
	// APIs and the same name are one card reached twice (see planSlots).
	api string
	// free and total are what Mem() reports. total 0 makes Mem fail, as a
	// backend that cannot say.
	free, total uint64
	// uuid is the physical device behind this fake, as backend.Identity
	// means it. Empty makes the device unidentifiable, and it must stay
	// distinct from every other device. It is separate from name so tests can
	// fabricate two different cards with the same name.
	uuid string
	// reads and readBytes count readbacks to the host, writeBytes uploads,
	// and copies and copyBytes device-to-device copies, so a path that must
	// stay on the device is checkable (TestRecurrentPoolReturnsEveryByte).
	reads, readBytes  int
	writeBytes        int
	copies, copyBytes int
}

// Identity satisfies backend.Identified. An empty uuid reports unknown rather
// than an empty key that would compare equal to another.
func (d *fakeDev) Identity() backend.Identity {
	if d.uuid == "" {
		return backend.Identity{Source: "fake: this device reports no identity"}
	}
	return backend.Identity{Key: "uuid:" + d.uuid, Source: "fake"}
}

// Mem reports the fake's memory, as every backend.Device does.
func (d *fakeDev) Mem() (free, total uint64, err error) {
	if d.total == 0 {
		return 0, 0, fmt.Errorf("fake: this device does not report its memory")
	}
	return d.free, d.total, nil
}

// fakeBuf counts its own free, in count and bytes (a free of the wrong buffer
// balances the count but not the bytes), so paging gates can assert that
// memory came back and not only that a block was paged out.
type fakeBuf struct {
	id   int
	n    int
	d    *fakeDev
	gone bool
	// site is where this buffer was allocated, so a leak names the line.
	site string
}
type fakeKern struct{ id int }
type fakeRec struct {
	d  *fakeDev
	id int
}

func (b *fakeBuf) Write(p []byte) error {
	b.d.writeBytes += len(p)
	return nil
}

func (b *fakeBuf) WriteAt(_ int, p []byte) error {
	b.d.writeBytes += len(p)
	return nil
}
func (b *fakeBuf) Read(p []byte) error {
	b.d.reads++
	b.d.readBytes += len(p)
	for i := range p {
		p[i] = 0
	}
	return nil
}
func (b *fakeBuf) Free() {
	if b.d.inSession > 0 {
		panic("fake: a buffer freed inside a Session -- on CUDA this deadlocks the owner goroutine")
	}
	if b.gone {
		b.d.doubleFree++ // freeing twice is its own bug, and it must not net out
		return
	}
	b.gone = true
	b.d.freed++
	b.d.freedBytes += b.n
	delete(b.d.liveBufs, b.id)
}
func (k *fakeKern) Close()                                              {}
func (k *fakeKern) Launch(groups, width int, bufs ...backend.Buf) error { return nil }

func (d *fakeDev) Name() string {
	if d.name != "" {
		return d.name
	}
	return "fake"
}
func (d *fakeDev) API() string {
	if d.api != "" {
		return d.api
	}
	return "fake"
}

// Slots 0: the fake does not know its capacity, so the split falls back to
// accSplitFor's default, as on Vulkan and Metal.
func (d *fakeDev) Slots() int          { return 0 }
func (d *fakeDev) Close()              { d.closed++ }
func (d *fakeDev) UnifiedMemory() bool { return d.unified }

// Allocated implements backend.AllocCounter: the tier charges what its
// scratch builds allocate by this count's difference.
func (d *fakeDev) Allocated() uint64 { return uint64(d.allocBytes - d.freedBytes) }

// FootprintOf implements backend.AllocCounter.
func (d *fakeDev) FootprintOf(n int) uint64 {
	if d.page <= 0 {
		return uint64(n)
	}
	return uint64((n + d.page - 1) / d.page * d.page)
}

// Footprint implements backend.AllocCounter: each live buffer rounded up to
// page bytes, the way a driver hands memory out; a zero page rounds nothing.
func (d *fakeDev) Footprint() uint64 {
	if d.page <= 0 {
		return d.Allocated()
	}
	var n uint64
	for _, b := range d.liveBufs {
		n += uint64((b.n + d.page - 1) / d.page * d.page)
	}
	return n
}

func (d *fakeDev) Alloc(n int) (backend.Buf, error) {
	if d.inSession > 0 {
		panic("fake: Alloc inside a Session -- on CUDA this deadlocks the owner goroutine")
	}
	if d.allocMax > 0 && n > d.allocMax {
		return nil, fmt.Errorf("fake: out of memory: %d bytes", n)
	}
	if d.capBytes > 0 && d.allocBytes-d.freedBytes+n > d.capBytes {
		return nil, fmt.Errorf("fake: out of memory: %d bytes with %d of %d live", n, d.allocBytes-d.freedBytes, d.capBytes)
	}
	d.bufs++
	d.allocBytes += n
	b := &fakeBuf{id: d.bufs, n: n, d: d, site: allocSite()}
	if d.liveBufs == nil {
		d.liveBufs = map[int]*fakeBuf{}
	}
	d.liveBufs[b.id] = b
	return b, nil
}

// Copy holds the tier to what a real backend refuses: a buffer of another
// device or one already freed, a range past either end, and a copy inside a
// Session, which on CUDA deadlocks the owner goroutine.
func (d *fakeDev) Copy(dst backend.Buf, dstOff int, src backend.Buf, srcOff, n int) error {
	if d.inSession > 0 {
		panic("fake: Copy inside a Session -- on CUDA this deadlocks the owner goroutine")
	}
	db, sb := dst.(*fakeBuf), src.(*fakeBuf)
	switch {
	case db.d != d || sb.d != d:
		return fmt.Errorf("fake: a copy between buffers of another device")
	case db.gone || sb.gone:
		return fmt.Errorf("fake: a copy from or to a freed buffer (%s, %s)", sb.site, db.site)
	case n < 0 || srcOff < 0 || dstOff < 0 || srcOff+n > sb.n || dstOff+n > db.n:
		return fmt.Errorf("fake: copy of %d bytes from %d of %d to %d of %d", n, srcOff, sb.n, dstOff, db.n)
	}
	d.copies++
	d.copyBytes += n
	return nil
}

func (d *fakeDev) Compile(k *ir.Kernel) (backend.Kernel, error) {
	if d.failCompile {
		return nil, fmt.Errorf("fake: %s does not compile", k.Name)
	}
	d.kerns++
	d.irs = append(d.irs, k)
	return &fakeKern{id: d.kerns}, nil
}

func (d *fakeDev) Session(f func(backend.Session)) {
	d.inSession++
	defer func() { d.inSession-- }()
	f(&fakeSession{d: d})
}

type fakeSession struct {
	d    *fakeDev
	rec  *[]fakeLaunch // non-nil while recording
	into []fakeLaunch  // what rec points at
}

func (s *fakeSession) Write(b backend.Buf, p []byte) error            { return nil }
func (s *fakeSession) WriteAt(b backend.Buf, off int, p []byte) error { return nil }
func (s *fakeSession) Read(b backend.Buf, p []byte) error             { return b.Read(p) }
func (s *fakeSession) Sync() error                                    { return nil }

func (s *fakeSession) Launch(k backend.Kernel, groups, width int, bufs ...backend.Buf) error {
	if s.d.launchCost > 0 {
		time.Sleep(s.d.launchCost)
	}
	l := fakeLaunch{kern: k.(*fakeKern).id, groups: groups, width: width}
	for _, b := range bufs {
		l.bufs = append(l.bufs, b.(*fakeBuf).id)
	}
	if s.rec != nil {
		if s.d.failCapture {
			return fmt.Errorf("fake: launch failed during capture")
		}
		*s.rec = append(*s.rec, l)
		return nil
	}
	s.d.log = append(s.d.log, l)
	return nil
}

func (s *fakeSession) BeginRecord() error {
	s.into = nil
	s.rec = &s.into
	return nil
}

func (s *fakeSession) EndRecord() (backend.Recording, error) {
	into := s.into
	s.rec, s.into = nil, nil
	s.d.nextGraph++
	if s.d.graphs == nil {
		s.d.graphs = map[int][]fakeLaunch{}
	}
	s.d.graphs[s.d.nextGraph] = into
	s.d.live++
	return &fakeRec{d: s.d, id: s.d.nextGraph}, nil
}

func (s *fakeSession) Replay(r backend.Recording) error {
	s.d.log = append(s.d.log, s.d.graphs[r.(*fakeRec).id]...)
	return nil
}

func (r *fakeRec) Free() {
	if r.d.inSession > 0 {
		panic("fake: a Recording freed inside a Session -- on CUDA this deadlocks the owner goroutine")
	}
	delete(r.d.graphs, r.id)
	r.d.live--
}

// fakePlan is the one small block every fake-device test runs.
func fakePlan() *nn.LayerPlan {
	return &nn.LayerPlan{
		NEmbd: 128, NHead: 4, NKVHead: 2, HeadDim: 32, NRot: 32, NFFN: 256,
		MaxSeq: 512, RMSEps: 1e-5, RopeBase: 10000,
	}
}

// fakeTier builds a one-block tier over the fake device. It returns the
// per-device tier, a complete nn.LayerDevice over one card; multi_test.go
// covers the router.
func fakeTier(t *testing.T) (*devTier, *fakeDev, *nn.LayerPlan) {
	t.Helper()
	// No split tuning: timing a fake device would pick widths from noise.
	d := &fakeDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(g.Close)
	one := g.devs[0]
	p := fakePlan()
	q4 := func(rows, k int) nn.Weight {
		return nn.Weight{T: quant.Q4_0, Data: make([]byte, rows*(k/32)*18), Rows: rows, K: k}
	}
	kv := p.NKVHead * p.HeadDim
	w := &nn.LayerWeights{
		AttnNorm: make([]float32, p.NEmbd), FFNNorm: make([]float32, p.NEmbd),
		Wq:   q4(p.NHead*p.HeadDim, p.NEmbd),
		Wk:   q4(kv, p.NEmbd),
		Wv:   q4(kv, p.NEmbd),
		Wo:   q4(p.NEmbd, p.NHead*p.HeadDim),
		Gate: q4(p.NFFN, p.NEmbd),
		Up:   q4(p.NFFN, p.NEmbd),
		Down: q4(p.NEmbd, p.NFFN),
	}
	if !one.PrepLayer(0, p, w) {
		t.Fatalf("PrepLayer declined: %s", one.LastErr)
	}
	return one, d, p
}

func step(t *testing.T, g *devTier, p *nn.LayerPlan, pos int) {
	t.Helper()
	x := make([]float32, p.NEmbd)
	cs := make([]float32, p.NRot)
	if !g.Layers(0, 1, pos, 1, x, cs, nil, nil) {
		t.Fatalf("Layers(pos=%d) failed: %s", pos, g.LastErr)
	}
}

// TestGraphCapturedOnceAndReplayed is the whole design in one assertion: the
// launch sequence is recorded on the first token and replayed by every token
// after it, until the rounded position count rolls over.
func TestGraphCapturedOnceAndReplayed(t *testing.T) {
	g, d, p := fakeTier(t)
	defer g.dropGraph()
	warmProbe(t, g, d, p)

	step(t, g, p, 0)
	if g.Captures != 1 {
		t.Fatalf("first token: %d captures, want 1", g.Captures)
	}
	perToken := len(d.log)
	if perToken == 0 {
		t.Fatal("the replay issued no launches")
	}
	// Every token inside one grain replays what the first one recorded.
	for pos := 1; pos < scoreGrain; pos++ {
		step(t, g, p, pos)
	}
	if g.Captures != 1 {
		t.Fatalf("%d captures within one grain, want 1", g.Captures)
	}
	if len(d.log) != perToken*scoreGrain {
		t.Fatalf("%d launches over %d tokens, want %d", len(d.log), scoreGrain, perToken*scoreGrain)
	}
	// Crossing the grain re-captures, exactly once, and retires the old one.
	step(t, g, p, scoreGrain)
	if g.Captures != 2 {
		t.Fatalf("%d captures after the grain rolled over, want 2", g.Captures)
	}
	if d.live != 1 {
		t.Fatalf("%d recordings live, want 1: the retired one was not freed", d.live)
	}
	g.dropGraph()
	if d.live != 0 {
		t.Fatalf("%d recordings live after dropGraph", d.live)
	}
}

// TestGraphReplaysWhatItWouldHaveIssued is the invariant a wrong graph breaks
// silently: the recorded sequence must be the sequence the direct path issues,
// launch for launch, buffer for buffer.
func TestGraphReplaysWhatItWouldHaveIssued(t *testing.T) {
	g, d, p := fakeTier(t)
	defer g.dropGraph()

	warmProbe(t, g, d, p)

	g.NoGraph = true
	step(t, g, p, 3)
	direct := d.log
	d.log = nil

	g.NoGraph = false
	step(t, g, p, 3)
	replayed := d.log
	if g.Captures != 1 {
		t.Fatalf("%d captures, want 1", g.Captures)
	}
	if len(direct) != len(replayed) {
		for i := 0; i < len(direct) || i < len(replayed); i++ {
			var a, b string
			if i < len(direct) {
				a = fmt.Sprint(direct[i])
			}
			if i < len(replayed) {
				b = fmt.Sprint(replayed[i])
			}
			if a != b {
				t.Logf("first difference at %d:\n  direct %s\n  graph  %s", i, a, b)
				break
			}
		}
		t.Fatalf("direct issued %d launches, the graph %d", len(direct), len(replayed))
	}
	for i := range direct {
		if fmt.Sprint(direct[i]) != fmt.Sprint(replayed[i]) {
			t.Fatalf("launch %d: direct %v, graph %v", i, direct[i], replayed[i])
		}
	}
}

// TestGraphDroppedWhenPartBufMoves: a captured graph holds device addresses, so
// a buffer freed and reallocated under it must drop the graph. A model split
// across the seam does exactly that when MatVec grows partBuf.
func TestGraphDroppedWhenPartBufMoves(t *testing.T) {
	g, d, p := fakeTier(t)
	defer g.dropGraph()

	step(t, g, p, 0)
	if len(g.recs) == 0 {
		t.Fatal("nothing was captured")
	}
	if !g.sizePart(g.partCap + 1) {
		t.Fatal("sizePart failed")
	}
	if len(g.recs) != 0 {
		t.Fatal("a recording survived the buffer it names being reallocated")
	}
	if d.live != 0 {
		t.Fatalf("%d recordings live after the drop", d.live)
	}
	step(t, g, p, 1)
	if g.Captures != 2 {
		t.Fatalf("%d captures, want 2: the sequence had to be recorded again", g.Captures)
	}
}

// warmProbe runs one step with graphs off so pickSoftmax's on-device check of
// the warp reduction (a launch on first use) has happened; otherwise the first
// counted token carries one launch no replay repeats.
func warmProbe(t *testing.T, g *devTier, d *fakeDev, p *nn.LayerPlan) {
	t.Helper()
	was := g.NoGraph
	g.NoGraph = true
	step(t, g, p, 0)
	g.NoGraph = was
	d.log = nil
}

// allocSite names the tier frame that asked for a buffer, skipping this file.
func allocSite() string {
	var pcs [12]uintptr
	n := runtime.Callers(3, pcs[:])
	fr := runtime.CallersFrames(pcs[:n])
	for {
		f, more := fr.Next()
		if !strings.Contains(f.File, "_test.go") && strings.Contains(f.File, "jit/gpu/tier") {
			return fmt.Sprintf("%s:%d", filepath.Base(f.File), f.Line)
		}
		if !more {
			return "?"
		}
	}
}
