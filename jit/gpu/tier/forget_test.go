package tier

import (
	"strings"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/quant"
)

// TestForgetFreesTheCopiesInItsRange is the device's half of placement.md 15v:
// a lone matvec's copy is keyed on its address, and Forget is the host saying
// the bytes there changed. The copy inside the range goes -- its buffers freed
// and its charge refunded -- and the next offer uploads afresh; a copy outside
// it stays. Asked while the device's lock is held, which is where a page-in
// from inside an upload asks from, it is queued and applied by the next lookup.
func TestForgetFreesTheCopiesInItsRange(t *testing.T) {
	d := &fakeDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff), WithMinMatVecBytes(0))
	if err != nil {
		t.Fatal(err)
	}
	const rows, k = 64, 256
	// One allocation, two weights: the second starts where the first ends, so
	// a range that overruns by a byte would take it too.
	mem := make([]byte, 2*rows*k*4)
	w1, w2 := mem[:rows*k*4], mem[rows*k*4:]
	x, out := make([]float32, k), make([]float32, rows)
	serve := func(w []byte) {
		t.Helper()
		if !g.MatVec(out, quant.F32, w, x, rows, k) {
			t.Fatalf("a lone F32 matvec was declined: %s", g.Err())
		}
	}
	// The weights' charge, without the staging the first matvec grew: the
	// device's own scratch, charged beside them (scratch.go).
	weights := func() uint64 { return g.Bytes() - g.Stats().ScratchBytes }
	serve(w1)
	serve(w2)
	both := weights()
	one := both / 2
	if both == 0 || g.Stats().Forgotten != 0 {
		t.Fatalf("%d bytes resident, %d forgotten before any Forget", both, g.Stats().Forgotten)
	}

	freed := d.freedBytes
	g.Forget(unsafe.Pointer(&w1[0]), uintptr(len(w1)))
	if got := g.Stats().Forgotten; got != 1 {
		t.Fatalf("Forget over the first weight dropped %d copies, want 1", got)
	}
	if weights() != both-one || d.freedBytes == freed {
		t.Fatalf("after Forget: %d bytes charged (want %d), %d device bytes freed",
			weights(), both-one, d.freedBytes-freed)
	}
	// The second is still the device's: offering it uploads nothing.
	alloc := d.allocBytes
	serve(w2)
	if d.allocBytes != alloc {
		t.Fatalf("the weight outside the range was uploaded again (%d bytes)", d.allocBytes-alloc)
	}
	// The first comes back as a fresh upload.
	serve(w1)
	if d.allocBytes == alloc || weights() != both {
		t.Fatalf("the forgotten weight was served without an upload (%d bytes, %d charged)",
			d.allocBytes-alloc, weights())
	}

	// From inside the device's lock: queued, not applied, and not a deadlock.
	dt := g.devs[0]
	dt.mu.Lock()
	g.Forget(unsafe.Pointer(&w1[0]), uintptr(len(w1)))
	dt.mu.Unlock()
	if !dt.forgetPend.Load() || g.Stats().Forgotten != 1 {
		t.Fatalf("a Forget under the device's lock was applied (%d forgotten) or lost (pending %v)",
			g.Stats().Forgotten, dt.forgetPend.Load())
	}
	serve(w2) // any lookup applies the queue first
	if g.Stats().Forgotten != 2 || weights() != both-one {
		t.Fatalf("the queued Forget was not applied by the next lookup: %d forgotten, %d charged",
			g.Stats().Forgotten, weights())
	}

	g.Close()
	if len(d.liveBufs) != 0 || d.doubleFree != 0 || d.allocBytes != d.freedBytes {
		t.Fatalf("after Close: %d buffer(s) live, %d freed twice, %d of %d bytes freed",
			len(d.liveBufs), d.doubleFree, d.freedBytes, d.allocBytes)
	}
}

// TestATierRefusesASecondModel is the router's half of model.TestATierServesOneModel
// on the fake: a block of model 2 is refused by name while model 1 holds
// anything, ReleaseModel of a model that does not hold the tier changes
// nothing, and ReleaseModel of the one that does gives back every buffer it
// placed -- blocks, head, the scratch built from its plan -- so model 2 is
// admitted onto a device that has seen nothing.
func TestATierRefusesASecondModel(t *testing.T) {
	d := &fakeDev{}
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	plan := func(model uint64) *nn.LayerPlan {
		p := fakePlan()
		p.Model = model
		return p
	}
	q4 := func(rows, k int) nn.Weight {
		return nn.Weight{T: quant.Q4_0, Data: make([]byte, rows*(k/32)*18), Rows: rows, K: k}
	}
	block := func(p *nn.LayerPlan) *nn.LayerWeights {
		kv := p.NKVHead * p.HeadDim
		return &nn.LayerWeights{
			AttnNorm: make([]float32, p.NEmbd), FFNNorm: make([]float32, p.NEmbd),
			Wq: q4(p.NHead*p.HeadDim, p.NEmbd), Wk: q4(kv, p.NEmbd), Wv: q4(kv, p.NEmbd),
			Wo: q4(p.NEmbd, p.NHead*p.HeadDim), Gate: q4(p.NFFN, p.NEmbd),
			Up: q4(p.NFFN, p.NEmbd), Down: q4(p.NEmbd, p.NFFN),
		}
	}
	head := func(model uint64) *nn.Head {
		p := fakePlan()
		return &nn.Head{Model: model, Norm: make([]float32, p.NEmbd), W: q4(512, p.NEmbd),
			Logits: make([]float32, 512)}
	}
	live0 := len(d.liveBufs)

	p1 := plan(1)
	if !g.PrepLayer(0, p1, block(p1)) || !g.PrepLayer(1, p1, block(p1)) {
		t.Fatalf("model 1 was refused an empty tier: %s", g.Err())
	}
	if !g.PrepHead(head(1)) {
		t.Fatalf("model 1's head was refused: %s", g.Err())
	}
	p2 := plan(2)
	if g.PrepLayer(0, p2, block(p2)) || g.PrepLayer(2, p2, block(p2)) || g.PrepHead(head(2)) {
		t.Fatal("model 2's offers were taken while model 1 holds the tier")
	}
	if e := g.Err(); !strings.Contains(e, "serves one model") {
		t.Fatalf("the refusal does not say why: %q", e)
	}
	if g.PrewarmLayer(3, p2, block(p2)) {
		t.Fatal("model 2's block was prewarmed while model 1 holds the tier")
	}
	g.ReleaseModel(2)
	if got := g.Placed()[0]; got != 2 {
		t.Fatalf("ReleaseModel of a model that holds nothing left %d of model 1's 2 blocks", got)
	}

	g.ReleaseModel(1)
	if got := g.Placed()[0]; got != 0 || g.Bytes() != 0 {
		t.Fatalf("after ReleaseModel: %d block(s) placed, %d bytes charged", got, g.Bytes())
	}
	if g.HeadResident() {
		t.Fatal("model 1's head survived ReleaseModel")
	}
	// The per-call staging is the device's, grown to the widest matvec any
	// model asked for, and serves the next model as it is.
	dt := g.devs[0]
	staging := map[backend.Buf]bool{}
	for _, b := range []backend.Buf{dt.aBuf, dt.axBuf, dt.outBuf, dt.partBuf, dt.xfBuf, dt.f16Buf} {
		if b != nil {
			staging[b] = true
		}
	}
	left := 0
	for _, b := range d.liveBufs {
		if !staging[b] {
			t.Logf("  live: %d bytes from %s", b.n, b.site)
			left++
		}
	}
	if left > live0 {
		t.Fatalf("%d buffer(s) model 1 placed are still live", left-live0)
	}
	if !g.PrepLayer(0, p2, block(p2)) {
		t.Fatalf("model 2 was refused after model 1 released the tier: %s", g.Err())
	}
}
