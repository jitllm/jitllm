package tier

import (
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"github.com/jitllm/jitllm/format/quant"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
)

// The multi-device tier, on fake devices. Every assertion is about which
// device holds and runs a block: a placement that put everything on device 0
// gives identical tokens, so only per-device counters can see it.

// blocks builds n blocks with distinct weights. resident() keys on the weight
// bytes' address, so shared weights would make blocks 2..n free and a
// two-block budget would admit five.
func blocks(p *nn.LayerPlan, n int) []*nn.LayerWeights {
	out := make([]*nn.LayerWeights, n)
	for i := range out {
		out[i] = blockWeights(p)
	}
	return out
}

// twoBlockBytes measures what two blocks of this plan cost on a device (weights,
// norms and the history of a first token: the page a run takes), so a budget
// that admits exactly two can be set.
func twoBlockBytes(t *testing.T, p *nn.LayerPlan, opt ...Option) uint64 {
	return twoBlockBytesOn(t, &fakeDev{}, p, opt...)
}

// twoBlockBytesOn is twoBlockBytes measured on d.
func twoBlockBytesOn(t *testing.T, d *fakeDev, p *nn.LayerPlan, opt ...Option) uint64 {
	t.Helper()
	g, err := New([]Slot{{Dev: d, Bytes: 1 << 40}}, opt...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	ws := blocks(p, 2)
	for li := 0; li < 2; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("sizing run: PrepLayer(%d) declined: %s", li, g.Err())
		}
	}
	if !g.ReserveKV(1) {
		t.Fatalf("sizing run: no room for a first page: %s", g.Err())
	}
	return g.devs[0].used
}

// placement returns the device index that holds each block, which is the one
// thing a multi-device tier has to get right.
func placement(g *GPU, n int) []int {
	out := make([]int, n)
	g.mu.Lock()
	defer g.mu.Unlock()
	for li := 0; li < n; li++ {
		out[li] = -1
		if d := g.own[li]; d != nil {
			out[li] = d.ord
		}
	}
	return out
}

// TestEveryDeviceTakesBlocksAndRunsThem: with a budget that cannot hold the
// model on one device, both devices must hold blocks and both must run them.
// The halves fail separately: a router that records a block on device 1 and
// submits it to device 0 passes placement and computes with the wrong buffers.
func TestEveryDeviceTakesBlocksAndRunsThem(t *testing.T) {
	// The order is given, not probed: a fake device would be ranked by noise.
	p := fakePlan()
	const nblocks = 5
	ws := blocks(p, nblocks)
	two := twoBlockBytes(t, p)

	a, b := &fakeDev{}, &fakeDev{}
	g, err := New([]Slot{{Dev: a, Bytes: two}, {Dev: b, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	for li := 0; li < nblocks; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("PrepLayer(%d) declined on every device: %s", li, g.Err())
		}
	}

	per := g.Placed()
	t.Logf("placement %v, blocks per device %v, %d crossing(s)", placement(g, nblocks), per, g.Crossings())
	for i, n := range per {
		if n == 0 {
			t.Fatalf("device %d took NO blocks: placement %v, blocks per device %v -- "+
				"a tier that puts every block on one device is the failure this test exists for",
				i, placement(g, nblocks), per)
		}
	}
	if per[0]+per[1] != nblocks {
		t.Fatalf("%d blocks placed, offered %d", per[0]+per[1], nblocks)
	}
	// The budget was sized for two, so the split is known exactly.
	if per[0] != 2 {
		t.Fatalf("the first device took %d blocks under a two-block budget", per[0])
	}

	// And now run one token through the whole range.
	x := make([]float32, p.NEmbd)
	cs := make([]float32, p.NRot)
	for i := range x {
		x[i] = float32(i%7) * 0.01
	}
	if !g.Layers(0, nblocks, 0, 1, x, cs, nil, nil) {
		t.Fatalf("Layers: %s", g.Err())
	}
	for i, st := range g.DevStats() {
		if st.Blocks == 0 {
			t.Fatalf("device %d RAN no blocks (%s): the placement said %d",
				i, st.Device, per[i])
		}
	}
	if len(a.log) == 0 || len(b.log) == 0 {
		t.Fatalf("launches issued: device 0 %d, device 1 %d -- a device that was handed "+
			"blocks and no launches computed nothing", len(a.log), len(b.log))
	}
}

// TestHeadOnlySubmissionReachesItsDevice: an empty block range with a head is
// not an empty call. Under a partial seam the blocks ran on the host and the
// projection is on a device; a router that did nothing and returned true would
// leave the previous token's logits in place.
func TestHeadOnlySubmissionReachesItsDevice(t *testing.T) {
	p := fakePlan()
	ws := blocks(p, 3)
	a, b := &fakeDev{}, &fakeDev{}
	g, err := New([]Slot{{Dev: a, Bytes: twoBlockBytes(t, p)}, {Dev: b, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	// Three blocks over a two-block budget put the last on the second device,
	// which is where the projection must go.
	for li := 0; li < 3; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("PrepLayer(%d): %s", li, g.Err())
		}
	}
	const nvocab = 64
	h := &nn.Head{
		Norm:   make([]float32, p.NEmbd),
		W:      nn.Weight{T: quant.Q4_0, Data: make([]byte, nvocab*(p.NEmbd/32)*18), Rows: nvocab, K: p.NEmbd},
		Logits: make([]float32, nvocab),
	}
	if !g.PrepHead(h) {
		t.Fatalf("PrepHead: %s", g.Err())
	}
	if !g.HeadResident() {
		t.Fatal("PrepHead succeeded and HeadResident says no")
	}
	if g.head != g.devs[1] {
		t.Fatalf("the projection went to device %d, and the last block is on device %d",
			g.head.ord, g.devs[1].ord)
	}
	a.log, b.log = nil, nil
	x := make([]float32, p.NEmbd)
	cs := make([]float32, p.NRot)
	if !g.Layers(3, 3, 0, 1, x, cs, nil, h) {
		t.Fatalf("head-only Layers: %s", g.Err())
	}
	if len(b.log) == 0 {
		t.Fatalf("a head-only call issued %d launches on the head's device: it returned "+
			"true without filling the logits", len(b.log))
	}
}

// TestBlocksAreContiguousPerDevice: each device's blocks are one run and there
// are (devices - 1) crossings. A scattered placement computes the same token
// with a residual-stream crossing per device change, invisible to correctness
// gates.
func TestBlocksAreContiguousPerDevice(t *testing.T) {
	p := fakePlan()
	const nblocks = 5
	ws := blocks(p, nblocks)
	two := twoBlockBytes(t, p)

	g, err := New([]Slot{
		{Dev: &fakeDev{}, Bytes: two},
		{Dev: &fakeDev{}, Bytes: two},
		{Dev: &fakeDev{}, Bytes: 1 << 40},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	for li := 0; li < nblocks; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("PrepLayer(%d) declined on every device: %s", li, g.Err())
		}
	}
	at := placement(g, nblocks)
	t.Logf("placement %v, %d crossing(s)", at, g.Crossings())
	// One run per device: the index a block sits on never decreases, and a
	// device that has been left is never returned to.
	seen := map[int]bool{}
	last := -1
	for li, d := range at {
		if d < 0 {
			t.Fatalf("block %d is on no device: %v", li, at)
		}
		if d != last {
			if seen[d] {
				t.Fatalf("device %d holds a second, separate run: placement %v -- "+
					"every change of device costs one crossing of the residual stream per token",
					d, at)
			}
			seen[d], last = true, d
		}
	}
	if c := g.Crossings(); c != len(seen)-1 {
		t.Fatalf("%d crossings over %d device(s): placement %v", c, len(seen), at)
	}
}

// TestFasterDeviceGoesFirst: the order is the timed matvec choose() uses, not
// the enumeration order, since a block on a slow device while the fast one had
// room is never undone.
func TestFasterDeviceGoesFirst(t *testing.T) {
	// Not TuneOff: this test is about the probe.
	slow := &fakeDev{name: "slow", launchCost: 500 * time.Microsecond}
	fast := &fakeDev{name: "fast"}
	got := order([]backend.Device{slow, fast}, TuneAuto)
	names := []string{got[0].Name(), got[1].Name()}
	if got[0] != backend.Device(fast) {
		t.Fatalf("order ranked %v, and the first one is where the first blocks go", names)
	}
	g, err := New(poolsFor(got, 1<<30))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	if g.devs[0].dev != backend.Device(fast) {
		t.Fatalf("the tier's first device is %q, not the fastest", g.devs[0].dev.Name())
	}
}

// TestUnifiedDevicesShareOnePool: an integrated GPU's memory is the host's, so
// unified devices share one budget. Separate budgets would place more than the
// machine has.
func TestUnifiedDevicesShareOnePool(t *testing.T) {
	p := fakePlan()
	ws := blocks(p, 4)
	two := twoBlockBytes(t, p)

	slots := poolsFor([]backend.Device{
		&fakeDev{unified: true}, &fakeDev{}, &fakeDev{unified: true},
	}, two)
	if slots[0].Pool == nil || slots[0].Pool != slots[2].Pool {
		t.Fatalf("the two unified devices are on pools %v and %v, and their memory is the same memory",
			slots[0].Pool, slots[2].Pool)
	}
	// A discrete device has its own card's pool from the planner, which must
	// not be the shared host pool.
	if slots[1].Pool == slots[0].Pool || slots[1].Pool.Host() {
		t.Fatal("a discrete device was put on the shared host pool")
	}

	// Two unified devices, one pool with room for two blocks. Four blocks
	// offered: two land, two do not, because the bytes are the same bytes.
	g, err := New(poolsFor([]backend.Device{&fakeDev{unified: true}, &fakeDev{unified: true}}, two))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	took := 0
	for li := 0; li < 4; li++ {
		if g.PrepLayer(li, p, ws[li]) {
			took++
		}
	}
	if took != 2 {
		t.Fatalf("%d blocks placed on two devices sharing a two-block pool (%d of %d bytes used): "+
			"the second device spent the first one's memory again",
			took, g.Pools()[0].Used(), g.Pools()[0].Limit())
	}
	if n := len(g.Pools()); n != 1 {
		t.Fatalf("%d pools for two unified devices, want 1", n)
	}
}

// TestOpenFailureClosesEveryDevice: a device that is already open costs a
// context, a locked OS thread and an owner goroutine (see backend.OpenCUDA), so
// failing to build the tier must not strand the ones that did open.
func TestOpenFailureClosesEveryDevice(t *testing.T) {
	a, b := &fakeDev{}, &fakeDev{}
	if _, err := New([]Slot{{Dev: a}, {Dev: nil}, {Dev: b}}); err == nil {
		t.Fatal("New accepted a nil device")
	}
	if a.closed != 1 || b.closed != 1 {
		t.Fatalf("after a failed New: device 0 closed %d times, device 2 closed %d times -- "+
			"the ones that opened were leaked", a.closed, b.closed)
	}
}

func TestCloseClosesEveryDevice(t *testing.T) {
	a, b := &fakeDev{}, &fakeDev{}
	g, err := New([]Slot{{Dev: a, Bytes: 1 << 30}, {Dev: b, Bytes: 1 << 30}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	g.Close()
	if a.closed != 1 || b.closed != 1 {
		t.Fatalf("Close closed device 0 %d times and device 1 %d times", a.closed, b.closed)
	}
}

// TestReleaseMovesTheCursorBack: the cursor only goes forward while a placement
// grows, and a release is the one thing that genuinely frees the room that made
// a device decline.
func TestReleaseMovesTheCursorBack(t *testing.T) {
	p := fakePlan()
	ws := blocks(p, 5)
	two := twoBlockBytes(t, p)
	g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: two}, {Dev: &fakeDev{}, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	for li := 0; li < 5; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("PrepLayer(%d): %s", li, g.Err())
		}
	}
	if g.cur != 1 {
		t.Fatalf("cursor at %d after the first device filled up, want 1", g.cur)
	}
	g.ReleaseLayers(2, 5) // everything the second device took
	if g.cur != 0 {
		t.Fatalf("cursor at %d after releasing the second device's blocks, want 0", g.cur)
	}
	// And the first device can grow again into what it still has.
	if b := g.Placed(); b[0] != 2 || b[1] != 0 {
		t.Fatalf("blocks per device %v after the release", b)
	}
}

// TestSingleDeviceTierHasOnePoolOfItsBudget: a one-device tier's budget is one
// pool of exactly the size it was given.
func TestSingleDeviceTierHasOnePoolOfItsBudget(t *testing.T) {
	g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 123 << 20}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	if len(g.pools) != 1 || g.pools[0].Limit() != 123<<20 {
		t.Fatalf("pools %v", g.pools)
	}
	if g.devs[0].limit != 123<<20 {
		t.Fatalf("device limit %d", g.devs[0].limit)
	}
	if g.Crossings() != 0 {
		t.Fatal("a single-device tier crosses the seam more than once a token")
	}
}

// TestPrewarmOverlapsLayers is the data race gate and only means anything
// under -race: nn.LayerDevice promises PrewarmLayer is safe while Layers runs,
// and a field wrongly shared between devices is a race the tokens never show.
//
//	./scripts/cap 8G -- go test -race ./jit_old/gpu/tier -run Prewarm
func TestPrewarmOverlapsLayers(t *testing.T) {
	p := fakePlan()
	ws := blocks(p, 8)
	g, err := New([]Slot{
		{Dev: &fakeDev{}, Bytes: twoBlockBytes(t, p)},
		{Dev: &fakeDev{}, Bytes: 1 << 40},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	for li := 0; li < 4; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("PrepLayer(%d): %s", li, g.Err())
		}
	}
	// The prewarm runs until the decode loop is done, not for a fixed count:
	// against a fake device a fixed count finishes before the loops overlap.
	stop, done := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		for {
			select {
			case <-stop:
				return
			default:
			}
			for li := 4; li < 8; li++ {
				g.PrewarmLayer(li, p, ws[li])
			}
		}
	}()
	// The main loop decodes and abandons staged packs (as `jitllm verify
	// -migrate` does), the one operation that reaches every device.
	x := make([]float32, p.NEmbd)
	cs := make([]float32, p.NRot)
	for pos := 0; pos < 200; pos++ {
		if !g.Layers(0, 4, pos, 1, x, cs, nil, nil) {
			t.Fatalf("Layers(pos=%d): %s", pos, g.Err())
		}
		g.DropStage()
		g.Stats()
		g.Placed()
		g.Bytes()
	}
	close(stop)
	<-done
}

// ---------------------------------------------------------------------------
// Real hardware.

// TestRealDevicesBothTakeBlocks runs on real cards: two devices hold blocks of
// one model at once, and the split run must agree with the one-device run. The
// agreement is the point; a split that passed the wrong residual would still
// fill every counter.
//
// The devices are allDevices' -- every CUDA ordinal and every Vulkan GPU, one
// API per card -- not backend.Open's, which is one device per backend and saw
// one of a 2xT4 box's two cards. The first two after ordering run the gate.
func TestRealDevicesBothTakeBlocks(t *testing.T) {
	devs, err := opener{}.allDevices(backend.Opts{})
	if len(devs) < 2 {
		for _, d := range devs {
			d.Close()
		}
		t.Skipf("this box enumerates %d distinct GPU device(s) across every backend; two are needed (%v)", len(devs), err)
	}
	devs = order(devs, TuneAuto)
	for _, d := range devs[2:] {
		d.Close()
	}
	devs = devs[:2]
	for i, d := range devs {
		t.Logf("device %d: %-40s [%s]", i, d.Name(), d.API())
	}
	p := &nn.LayerPlan{
		NEmbd: 128, NHead: 4, NKVHead: 2, HeadDim: 32, NRot: 32, NFFN: 256,
		MaxSeq: 64, RMSEps: 1e-5, RopeBase: 10000,
	}
	const nblocks = 4
	ws := blocks(p, nblocks)
	// Real weights: a zero block computes zero on any device.
	for i, w := range ws {
		fill(w, uint32(12345+i*7919))
	}

	g, err := New(poolsFor(devs, 1<<30))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()

	x0 := make([]float32, p.NEmbd)
	for i := range x0 {
		x0[i] = float32(math.Sin(float64(i)*0.37)) * 0.1
	}
	cs := make([]float32, p.NRot)
	for i := 0; i < p.NRot; i += 2 {
		cs[i], cs[i+1] = 1, 0
	}

	// Arm 1: every block on the fastest device.
	for li := 0; li < nblocks; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("PrepLayer(%d): %s", li, g.Err())
		}
	}
	if b := g.Placed(); b[0] != nblocks {
		t.Fatalf("the first device took %v of %d blocks with the whole card available", b, nblocks)
	}
	one := append([]float32(nil), x0...)
	if !g.Layers(0, nblocks, 0, 1, one, cs, nil, nil) {
		t.Fatalf("Layers on one device: %s", g.Err())
	}

	// Arm 2: cut the first device's limit (policy), not its pool (memory):
	// two APIs onto one card share a pool, and cutting it would starve the
	// second device too.
	half := g.devs[0].used / 2
	g.ReleaseLayers(0, nblocks)
	g.devs[0].limit = half
	for li := 0; li < nblocks; li++ {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("split placement: PrepLayer(%d) declined on every device: %s", li, g.Err())
		}
	}
	per := g.Placed()
	t.Logf("split placement %v, blocks per device %v, %d crossing(s)",
		placement(g, nblocks), per, g.Crossings())
	for i, n := range per {
		if n == 0 {
			t.Fatalf("device %d (%s) took NO blocks: %v", i, g.DevStats()[i].Device, per)
		}
	}
	split := append([]float32(nil), x0...)
	if !g.Layers(0, nblocks, 0, 1, split, cs, nil, nil) {
		t.Fatalf("Layers across two devices: %s", g.Err())
	}
	for i, st := range g.DevStats() {
		t.Logf("device %d %-40s %d block(s) placed, %d run, %.3f MiB resident",
			i, st.Device, per[i], st.Blocks, float64(g.devs[i].used)/(1<<20))
		if st.Blocks == 0 {
			t.Fatalf("device %d ran no blocks", i)
		}
	}

	// NMSE, not equality: the arms run on different hardware and backends.
	var num, den float64
	for i := range one {
		d := float64(one[i] - split[i])
		num += d * d
		den += float64(one[i]) * float64(one[i])
	}
	nmse := num / (den + 1e-30)
	t.Logf("one device vs split: NMSE %.3e", nmse)
	for i, v := range split {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("the split run produced %v at %d: a non-finite reference scores every "+
				"comparison below at zero (RULE 10)", v, i)
		}
	}
	if !(nmse < 1e-8) || math.IsNaN(nmse) {
		t.Fatalf("splitting the model across two devices changed the residual stream: NMSE %g\n"+
			"  one   %v\n  split %v", nmse, one[:8], split[:8])
	}
}

// fill puts real numbers in a block's weights. The Q4_0 scale is fixed, since a
// random binary16 is sometimes Inf or NaN, and the seed is per block so the
// blocks are different functions whose order matters.
func fill(w *nn.LayerWeights, seed uint32) {
	next := func() byte {
		seed = seed*1664525 + 1013904223
		return byte(seed >> 24)
	}
	const q40Block = 18          // 2 bytes of binary16 scale + 16 bytes of nibbles
	const scale = uint16(0x2C00) // 0.0625, so a weight lands in [-0.5, 0.44]
	for _, t := range [][]byte{w.Wq.Data, w.Wk.Data, w.Wv.Data, w.Wo.Data,
		w.Gate.Data, w.Up.Data, w.Down.Data} {
		for i := range t {
			t[i] = next()
		}
		for b := 0; b+q40Block <= len(t); b += q40Block {
			t[b], t[b+1] = byte(scale&0xff), byte(scale>>8)
		}
	}
	for _, f := range [][]float32{w.AttnNorm, w.FFNNorm} {
		for i := range f {
			f[i] = 1 + float32(i%5)*0.01
		}
	}
}

// init makes the backend verbose under JITLLM_GPU_VERBOSE.
func init() {
	if os.Getenv("JITLLM_GPU_VERBOSE") != "" {
		fmt.Fprintln(os.Stderr, "tier: JITLLM_VK_DEVICE picks which Vulkan device the real two-device gate uses")
		backend.SetVerbose(true)
	}
}

// TestASecondSessionFindsTheBlocksWhereTheFirstLeftThem: a second State
// re-offers every block the first placed, and each must land on the device
// already holding it rather than being uploaded again from the cursor.
func TestASecondSessionFindsTheBlocksWhereTheFirstLeftThem(t *testing.T) {
	p := fakePlan()
	const nblocks = 4
	ws := blocks(p, nblocks)
	two := twoBlockBytes(t, p)
	g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: two}, {Dev: &fakeDev{}, Bytes: two}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	// The first session closes before the second attaches: its KV cache goes
	// back, its weights stay.
	for i := 0; i < 2; i++ {
		s := g.Attach()
		for li := 0; li < nblocks; li++ {
			if !s.PrepLayer(li, p, ws[li]) {
				t.Fatalf("session %d: PrepLayer(%d) declined: %s (placement %v)", i, li, g.Err(), placement(g, nblocks))
			}
		}
		s.(nn.Session).Detach()
	}
	if at := placement(g, nblocks); fmt.Sprint(at) != "[0 0 1 1]" {
		t.Fatalf("placement %v after two sessions, want [0 0 1 1]", at)
	}
	if u0, u1 := g.devs[0].used, g.devs[1].used; u0 > two || u1 > two {
		t.Fatalf("device bytes %d / %d over a budget of %d: a block was uploaded twice", u0, u1, two)
	}
}

// TestABlockWhoseCacheDoesNotFitGoesToTheNextDevice: admission prices a block's
// whole cost, weights plus KV cache and norms. Pricing only the weights
// admitted a block whose cache did not fit, and trim() then paged blocks out
// with paging off.
func TestABlockWhoseCacheDoesNotFitGoesToTheNextDevice(t *testing.T) {
	p := fakePlan()
	ws := blocks(p, 3)
	sz, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 1 << 40}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for li := 0; li < 2; li++ {
		if !sz.PrepLayer(li, p, blocks(p, 1)[0]) {
			t.Fatalf("sizing run: PrepLayer(%d) declined: %s", li, sz.Err())
		}
	}
	// The history: a contiguous cache's bytes, or the paged pool's.
	two, kv := sz.devs[0].used, (sz.devs[0].KVBytes+sz.devs[0].kvPoolBytes())/2
	sz.Close()
	if kv == 0 {
		t.Fatalf("the sizing run charged no KV cache, so this test cannot separate weights from history")
	}
	// Room for two blocks' weights and one and a half caches: the second
	// block's weights fit, its cache does not.
	a, b := &fakeDev{}, &fakeDev{}
	g, err := New([]Slot{{Dev: a, Bytes: two - kv/2}, {Dev: b, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	for li := range ws {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("PrepLayer(%d) declined on every device: %s", li, g.Err())
		}
	}
	d0 := g.devs[0]
	t.Logf("placement %v, device 0 used %d of %d, %d page-out(s)", placement(g, len(ws)), d0.used, d0.limit, d0.PageOuts)
	if d0.PageOuts != 0 || d0.used > d0.limit {
		t.Fatalf("device 0 paged %d block(s) out at placement and holds %d of %d: it admitted a "+
			"block whose KV cache did not fit, and paging is off", d0.PageOuts, d0.used, d0.limit)
	}
	if got := placement(g, len(ws)); got[0] != 0 || got[1] != 1 || got[2] != 1 {
		t.Fatalf("placement %v, want [0 1 1]: the block that does not fit belongs on the next device", got)
	}
}

// TestEverySessionPlacesWhatTheFirstDid is Config.Sessions: a device that
// reserves n histories per block gives the n-th concurrent session the same
// blocks the first got. Without the reservation the second session silently
// ran on the host.
func TestEverySessionPlacesWhatTheFirstDid(t *testing.T) {
	p := fakePlan()
	p.MaxSeq = 64 // a history well under a block's weights, as in a real model
	const nblocks = 3
	ws := blocks(p, nblocks)
	sz, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 1 << 40}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for li := 0; li < 2; li++ {
		if !sz.PrepLayer(li, p, blocks(p, 1)[0]) {
			t.Fatalf("sizing run: PrepLayer(%d) declined: %s", li, sz.Err())
		}
	}
	blk, kv := sz.devs[0].used/2, sz.devs[0].KVBytes/2
	sz.Close()
	// Every block fits with one history, and half a second history spare.
	budget := nblocks*blk + kv/2
	for _, sessions := range []int{2, 3} {
		g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: budget}}, WithDeviceTune(TuneOff),
			WithConfig(func(c *Config) { c.Sessions = sessions }))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		var got []int
		for i := 0; i < sessions; i++ {
			s, n := g.Attach(), 0
			for li := 0; li < nblocks; li++ {
				if s.PrepLayer(li, p, ws[li]) {
					n++
				}
			}
			got = append(got, n)
		}
		d := g.devs[0]
		t.Logf("sessions %d: blocks per session %v, used %d + reserved %d of %d, %d session decline(s)",
			sessions, got, d.used, d.reserved(), d.limit, d.SessionDeclines)
		for i, n := range got {
			if n == 0 || n != got[0] {
				g.Close()
				t.Fatalf("sessions %d: blocks per session %v -- session %d did not get what the first did",
					sessions, got, i)
			}
		}
		if d.SessionDeclines != 0 || d.used > d.limit {
			g.Close()
			t.Fatalf("sessions %d: %d session decline(s), %d of %d used", sessions, d.SessionDeclines, d.used, d.limit)
		}
		g.Close()
	}
}

// TestAFullTailSendsTheHeadToACardWithRoom: when the device holding the last
// block has no room for the output projection, another device takes it and
// Layers runs it as a head-only submission there, never the host while a card
// has room.
func TestAFullTailSendsTheHeadToACardWithRoom(t *testing.T) {
	p := fakePlan()
	ws := blocks(p, 2)
	two := twoBlockBytes(t, p)
	a, b := &fakeDev{}, &fakeDev{}
	// The first device holds exactly the two blocks and nothing more; the
	// second has room and no blocks.
	g, err := New([]Slot{{Dev: a, Bytes: two}, {Dev: b, Bytes: 1 << 40}}, WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	for li := range ws {
		if !g.PrepLayer(li, p, ws[li]) {
			t.Fatalf("PrepLayer(%d): %s", li, g.Err())
		}
	}
	if per := g.Placed(); per[0] != 2 || per[1] != 0 {
		t.Fatalf("blocks per device %v, want [2 0]", per)
	}
	const nvocab = 64
	h := &nn.Head{
		Norm:   make([]float32, p.NEmbd),
		W:      nn.Weight{T: quant.Q4_0, Data: make([]byte, nvocab*(p.NEmbd/32)*18), Rows: nvocab, K: p.NEmbd},
		Logits: make([]float32, nvocab),
	}
	if !g.PrepHead(h) {
		t.Fatalf("PrepHead refused with an empty card beside the full tail: %s", g.Err())
	}
	if hd := g.HeadDevice(); hd != g.devs[1].ord {
		t.Fatalf("the projection is on device %d, want the one with room (%d)", hd, g.devs[1].ord)
	}
	b.log = nil
	x := make([]float32, p.NEmbd)
	cs := make([]float32, p.NRot)
	if !g.Layers(0, 2, 0, 1, x, cs, nil, h) {
		t.Fatalf("Layers with the head on another device: %s", g.Err())
	}
	if len(b.log) == 0 {
		t.Fatal("the head's device issued no launches: the logits were never computed")
	}
}

// TestAPlannedModelThatFitsIsSpreadOverTheDevices is GPU.PlanBlocks: blocks
// that fit are split in proportion to the budgets over the fewest devices that
// hold them, rather than filling the first (which left no room for prefill
// scratch). A model that does not fit keeps fill-first placement.
func TestAPlannedModelThatFitsIsSpreadOverTheDevices(t *testing.T) {
	p := fakePlan()
	sz, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 1 << 40}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !sz.PrepLayer(0, p, blocks(p, 1)[0]) {
		t.Fatalf("sizing run: %s", sz.Err())
	}
	// A device's own scratch is charged once beside its blocks (scratch.go).
	sc := sz.devs[0].scratch
	blk := sz.devs[0].used - sc
	sz.Close()
	for _, tc := range []struct {
		n     int
		plan  bool
		extra uint64
		want  string
	}{
		{4, true, 0, "[0 0 1 1]"},  // fits in two: spread over TWO, not three
		{4, false, 0, "[0 0 0 1]"}, // no plan: fill-first, as before
		{2, true, 0, "[0 0]"},      // one device holds it with room: stays on one
		// One device holds the weights to its last byte, which leaves none
		// for the prompt's scratch: the plan takes a second.
		{3, true, 0, "[0 0 1]"},
		{7, true, 0, "[0 0 1 1 1 2 2]"},         // needs all three: spread over three
		{10, true, 0, "[0 0 0 1 1 1 2 2 2 -1]"}, // does not fit: fill-first, one left over
		// Two devices hold the weights (the first case) and not the history
		// and head beside them: the plan takes a third. The first two blocks
		// price the plan, so they are on device 0 already.
		{4, true, 2 * blk, "[0 0 1 2]"},
	} {
		// Each device a plan touches adds a crossing, so it spreads over the
		// fewest that hold the model.
		g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: sc + 3*blk}, {Dev: &fakeDev{}, Bytes: sc + 3*blk},
			{Dev: &fakeDev{}, Bytes: sc + 3*blk}}, WithDeviceTune(TuneOff))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if tc.plan {
			g.PlanBlocks(tc.n, tc.extra)
		}
		ws := blocks(p, tc.n)
		for li := range ws {
			g.PrepLayer(li, p, ws[li])
		}
		got := fmt.Sprint(placement(g, tc.n))
		g.Close()
		if got != tc.want {
			t.Errorf("%d blocks, plan %v: placement %s, want %s", tc.n, tc.plan, got, tc.want)
		}
	}
}

// TestASpillThatFindsNoRoomStaysASpill is GPU.SpillAfter when no later device
// has room: every offer of the block in that attempt goes to the later
// devices only, and refuses. The model offers a held block twice (the
// plan-only offer, then the full one); a mark the first refusal cleared sent
// the second back onto the card it was leaving, the model read that as a
// spill, and relocateUntil spun on one block until the test timed out.
func TestASpillThatFindsNoRoomStaysASpill(t *testing.T) {
	p := fakePlan()
	sz, err := New([]Slot{{Dev: &fakeDev{}, Bytes: 1 << 40}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !sz.PrepLayer(0, p, blocks(p, 1)[0]) {
		t.Fatalf("sizing run: %s", sz.Err())
	}
	sc := sz.devs[0].scratch
	blk := sz.devs[0].used - sc
	sz.Close()
	// Room for two blocks each, and a little: both devices full at four.
	g, err := New([]Slot{{Dev: &fakeDev{}, Bytes: sc + 2*blk + blk/2}, {Dev: &fakeDev{}, Bytes: sc + 2*blk + blk/2}},
		WithDeviceTune(TuneOff))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	ws := blocks(p, 4)
	for li := range ws {
		g.PrepLayer(li, p, ws[li])
	}
	if got := fmt.Sprint(placement(g, 4)); got != "[0 0 1 1]" {
		t.Fatalf("placement %s, want [0 0 1 1]", got)
	}
	if !g.SpillAfter(1) {
		t.Fatal("block 1 is on the first of two devices and SpillAfter refused")
	}
	for i := range 2 {
		if g.PrepLayer(1, p, ws[1]) {
			t.Fatalf("offer %d of a spilling block was taken with no room on any later device: placement %s",
				i+1, fmt.Sprint(placement(g, 4)))
		}
	}
	if got := fmt.Sprint(placement(g, 4)); got != "[0 0 1 1]" {
		t.Fatalf("placement %s after a spill that found no room, want [0 0 1 1]", got)
	}
}
