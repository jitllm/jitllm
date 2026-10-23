package tier

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// The packed arena, at the tier's level: a block uploaded, released and
// uploaded again must get the same bytes without being packed twice.
//
// A repack is deterministic, so it produces the right bytes and tokens and
// shows only as a page-in that costs about 3x the transfer. So the counter is
// asserted first and the bytes second, and each test first checks that the
// second upload landed in different device buffers, i.e. that it really paged.

// recDev is a device that keeps what was written to it, so two uploads can be
// compared byte for byte without a card.
type recDev struct {
	bufs  int
	kerns int
}

type recBuf struct {
	id   int
	data []byte
}

func (b *recBuf) Write(p []byte) error {
	b.data = append(b.data[:0], p...)
	return nil
}

// WriteAt models a real allocation rather than recording the call, so an
// offset bug cannot slip through the gates that read back what was written.
func (b *recBuf) WriteAt(off int, p []byte) error {
	if off < 0 {
		return fmt.Errorf("rec: negative offset %d", off)
	}
	if n := off + len(p); n > len(b.data) {
		b.data = append(b.data, make([]byte, n-len(b.data))...)
	}
	copy(b.data[off:], p)
	return nil
}

func (b *recBuf) Read(p []byte) error {
	copy(p, b.data)
	return nil
}

func (b *recBuf) Free() { b.data = nil }

type recKern struct{ id int }

func (k *recKern) Close()                                              {}
func (k *recKern) Launch(groups, width int, bufs ...backend.Buf) error { return nil }

func (d *recDev) Name() string { return "rec" }
func (d *recDev) API() string  { return "rec" }
func (d *recDev) Slots() int   { return 0 }
func (d *recDev) Close()       {}

// Mem reports nothing: the recording device has no memory to speak of.
func (d *recDev) Mem() (free, total uint64, err error) {
	return 0, 0, fmt.Errorf("rec: this device does not report its memory")
}

func (d *recDev) Alloc(n int) (backend.Buf, error) {
	d.bufs++
	return &recBuf{id: d.bufs, data: make([]byte, 0, n)}, nil
}

func (d *recDev) Compile(k *ir.Kernel) (backend.Kernel, error) {
	d.kerns++
	return &recKern{id: d.kerns}, nil
}

// Copy moves the recorded bytes, as a device would, growing the destination
// the way WriteAt does.
func (d *recDev) Copy(dst backend.Buf, dstOff int, src backend.Buf, srcOff, n int) error {
	sb := src.(*recBuf)
	if srcOff < 0 || srcOff+n > len(sb.data) {
		return fmt.Errorf("rec: copy of %d bytes from offset %d of %d", n, srcOff, len(sb.data))
	}
	return dst.WriteAt(dstOff, append([]byte(nil), sb.data[srcOff:srcOff+n]...))
}

func (d *recDev) Session(f func(backend.Session)) { f(&recSession{}) }

type recSession struct{}

func (recSession) Write(b backend.Buf, p []byte) error { return b.Write(p) }
func (recSession) WriteAt(b backend.Buf, off int, p []byte) error {
	return b.WriteAt(off, p)
}
func (recSession) Read(b backend.Buf, p []byte) error { return b.Read(p) }
func (recSession) Sync() error                        { return nil }
func (recSession) Launch(k backend.Kernel, groups, width int, bufs ...backend.Buf) error {
	return nil
}

// arenaPlan is small enough to page eight times in a test and wide enough that
// every format field of the pack is exercised.
func arenaPlan() *nn.LayerPlan {
	return &nn.LayerPlan{
		NEmbd: 256, NHead: 8, NKVHead: 4, HeadDim: 32, NRot: 32, NFFN: 512,
		MaxSeq: 64, RMSEps: 1e-5, RopeBase: 10000,
	}
}

// snapshot is every weight buffer of a resident block, by buffer identity and by
// content.
type snapshot struct {
	ids   []int
	bytes [][]byte
}

func snapWeights(t *testing.T, g *devTier, li int) snapshot {
	t.Helper()
	g.mu.Lock()
	defer g.mu.Unlock()
	l := g.layers[li]
	if l == nil {
		t.Fatalf("block %d is not resident, so there is nothing to compare", li)
	}
	var s snapshot
	for _, r := range []*resident{l.wq, l.wk, l.wv, l.wo, l.gate, l.up, l.down} {
		if r == nil || !r.ok {
			t.Fatalf("block %d has a tensor that is not resident", li)
		}
		for _, b := range []backend.Buf{r.qs, r.d, r.sc} {
			rb, ok := b.(*recBuf)
			if !ok {
				t.Fatalf("expected a recording buffer, got %T", b)
			}
			s.ids = append(s.ids, rb.id)
			s.bytes = append(s.bytes, append([]byte(nil), rb.data...))
		}
	}
	return s
}

// TestArenaPacksOncePerPageIn pages one block in eight times and asserts, in
// order:
//
//  1. the pack ran once, counting both the arena's packs and any the tier
//     made outside it;
//  2. something actually paged (the buffers were freed and reallocated);
//  3. the device bytes are identical across every page-in.
func TestArenaPacksOncePerPageIn(t *testing.T) {
	d := &recDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	g.SetArena(1 << 30)

	p := arenaPlan()
	w := blockWeights(p)
	fill(w, 20260913)
	one := g.devs[0]

	const pageIns = 8
	var first snapshot
	for i := 0; i < pageIns; i++ {
		if !one.PrepLayer(0, p, w) {
			t.Fatalf("page-in %d: PrepLayer declined: %s", i, one.Err())
		}
		got := snapWeights(t, one, 0)
		if i == 0 {
			first = got
		} else {
			same := 0
			for j := range got.ids {
				if got.ids[j] == first.ids[j] {
					same++
				}
			}
			if same != 0 {
				t.Fatalf("page-in %d reused %d of %d device buffers: nothing was freed, so "+
					"this test is not measuring a page-in at all", i, same, len(got.ids))
			}
			for j := range got.bytes {
				if len(got.bytes[j]) != len(first.bytes[j]) {
					t.Fatalf("page-in %d buffer %d is %d bytes, first upload was %d",
						i, j, len(got.bytes[j]), len(first.bytes[j]))
				}
				for b := range got.bytes[j] {
					if got.bytes[j][b] != first.bytes[j][b] {
						t.Fatalf("page-in %d buffer %d byte %d: %#x, first upload %#x -- "+
							"a paged block is not the same weights",
							i, j, b, got.bytes[j][b], first.bytes[j][b])
					}
				}
			}
		}
		one.ReleaseLayers(0, 1)
	}

	as := g.ArenaStats()
	packs := as.Packs + int64(g.Stats().Packs)
	const tensors = 7 // q, k, v, o, gate, up, down
	if packs != tensors {
		t.Fatalf("%d packs (%d in the arena, %d outside it) for %d tensors over %d page-ins, "+
			"want %d: packing is 1.01 GB/s against an upload's 3.19, so a repack per page-in "+
			"costs 3.1x the transfer it replaces",
			packs, as.Packs, g.Stats().Packs, tensors, pageIns, tensors)
	}
	if as.Hits < tensors*(pageIns-1) {
		t.Fatalf("%d arena hits over %d page-ins of %d tensors, want at least %d",
			as.Hits, pageIns, tensors, tensors*(pageIns-1))
	}
	t.Logf("%d page-ins, %d packs, %d arena hits, %.2f MiB retained",
		pageIns, packs, as.Hits, float64(as.Bytes)/(1<<20))
}

// TestArenaOffRepacksEveryPageIn is the violation arm: the same block with a
// zero arena budget must repack on every page-in while producing identical
// bytes, which is why the gate above is a counter. Zero is the default budget,
// so this also pins a model that fits: no arena entries, nothing retained.
func TestArenaOffRepacksEveryPageIn(t *testing.T) {
	d := &recDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	// No SetArena: zero is the default.

	p := arenaPlan()
	w := blockWeights(p)
	fill(w, 20260913)
	one := g.devs[0]

	const pageIns = 4
	const tensors = 7
	var first snapshot
	for i := 0; i < pageIns; i++ {
		if !one.PrepLayer(0, p, w) {
			t.Fatalf("page-in %d: PrepLayer declined: %s", i, one.Err())
		}
		got := snapWeights(t, one, 0)
		if i == 0 {
			first = got
		} else {
			for j := range got.bytes {
				if string(got.bytes[j]) != string(first.bytes[j]) {
					t.Fatalf("page-in %d buffer %d differs: the PACKER is not deterministic, "+
						"which breaks the premise of both arms", i, j)
				}
			}
		}
		one.ReleaseLayers(0, 1)
	}

	as := g.ArenaStats()
	if as.Packs != 0 || as.Bytes != 0 || as.Entries != 0 {
		t.Fatalf("a model that FITS grew an arena: %d packs, %d bytes, %d entries,  evicted",
			as.Packs, as.Bytes, as.Entries)
	}
	if want := tensors * pageIns; g.Stats().Packs != want {
		t.Fatalf("%d packs over %d page-ins with the arena off, want %d -- if this is %d the "+
			"counter is not counting the repacks the other arm proves are absent",
			g.Stats().Packs, pageIns, want, tensors)
	}
	t.Logf("arena off: %d packs for %d page-ins of %d tensors, and the bytes are identical "+
		"either way -- which is why the gate is a counter", g.Stats().Packs, pageIns, tensors)
}

// TestArenaPagesInOnRealHardware checks the round trip against a driver: the
// arena is an anonymous mapping rather than Go heap memory, and only a real
// device upload can show a pointer the driver cannot take.
func TestArenaPagesInOnRealHardware(t *testing.T) {
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, dev := range devs[1:] {
		dev.Close()
	}
	g, err := New([]Slot{{Dev: devs[0], Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	g.SetArena(1 << 28)

	p := arenaPlan()
	w := blockWeights(p)
	fill(w, 4242)
	one := g.devs[0]

	read := func() []byte {
		one.mu.Lock()
		defer one.mu.Unlock()
		l := one.layers[0]
		if l == nil {
			t.Fatalf("block 0 is not resident")
		}
		if l.wq == nil || !l.wq.ok {
			t.Fatalf("block 0's q projection is not resident")
		}
		nq, _, _, err := kernels.PackedWords(l.wq.t, l.wq.nrows, l.wq.k)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]byte, nq*4)
		if err := l.wq.qs.Read(out); err != nil {
			t.Fatalf("read back: %v", err)
		}
		return out
	}

	if !one.PrepLayer(0, p, w) {
		t.Fatalf("PrepLayer declined: %s", one.Err())
	}
	first := read()
	one.ReleaseLayers(0, 1)
	if !one.PrepLayer(0, p, w) {
		t.Fatalf("page-in: PrepLayer declined: %s", one.Err())
	}
	second := read()
	one.ReleaseLayers(0, 1)

	if len(first) != len(second) {
		t.Fatalf("%d bytes then %d", len(first), len(second))
	}
	diff := 0
	for i := range first {
		if first[i] != second[i] {
			diff++
		}
	}
	if diff != 0 {
		t.Fatalf("%d of %d bytes differ after a page-in from the arena on %s",
			diff, len(first), devs[0].Name())
	}
	as := g.ArenaStats()
	if packs := as.Packs + int64(g.Stats().Packs); packs != 7 {
		t.Fatalf("%d packs for 7 tensors and 2 uploads, want 7", packs)
	}
	t.Logf("%s: %d bytes identical across a page-in, %d packs for 2 uploads",
		devs[0].Name(), len(first), as.Packs+int64(g.Stats().Packs))
}
