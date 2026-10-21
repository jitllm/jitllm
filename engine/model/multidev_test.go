//go:build linux

package model

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestTwoDevicesAgreeWithTheHost splits ONE model's blocks across TWO distinct
// accelerators and compares its logits against the host: two KV caches, two
// scratch sets, and a residual crossing between devices through the host are
// wiring the per-device gates cannot see.
//
// It asserts both devices took blocks (a run where one declines everything is
// single-device and skips). The bound is the f32 reduction band, not equality.
// It fails against GPU.runs returning its per-device runs in reverse order.
func TestTwoDevicesAgreeWithTheHost(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := Open(path, noTune, WithKVF16(false))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	ids := []int32{5, 11, 23, 41, 67, 89}
	want := teacherForce(t, m, ids)

	// A budget of a third of the model per device forces the split; otherwise
	// the first card takes everything. It is derived from the model so another
	// fixture still splits.
	cap3 := m.PageSize()*uint64(m.Cfg.NLayer)/3 + m.PageSize()
	g, err := openTier(t, tier.WithDevices("all"), tier.WithDeviceTune(tier.TuneOff),
		tier.WithBudget(cap3))
	if err != nil || g == nil {
		noDevice(t, "devices", err)
	}
	defer g.Close()
	if n := len(g.Placed()); n < 2 {
		t.Skipf("this host enumerates %d distinct device(s); the gate needs two", n)
	}
	t.Logf("%d device(s), %.1f MiB budget each, %d block(s) of %.1f MiB",
		len(g.Placed()), float64(cap3)/(1<<20), m.Cfg.NLayer, float64(m.PageSize())/(1<<20))

	st := m.NewState(len(ids) + 1)
	defer st.Close()
	st.SetDeviceLayers(g, -1)

	placed := g.Placed()
	busy := 0
	total := 0
	for _, n := range placed {
		total += n
		if n > 0 {
			busy++
		}
	}
	t.Logf("blocks per device: %v (%d of %d placed), %d crossing(s)",
		placed, total, m.Cfg.NLayer, g.Crossings())
	if busy < 2 {
		// Not a pass: a single-device placement is covered elsewhere. The usual
		// cause is the second device declining on budget.
		t.Skipf("only %d device took blocks (%v) -- nothing here is a two-device "+
			"run, so this gate would prove nothing", busy, placed)
	}
	if total == 0 {
		t.Fatalf("no block was placed at all: %v", declineList(st))
	}

	got := runIDs(t, st, ids)
	if p2 := g.Placed(); !sameInts(p2, placed) {
		t.Fatalf("the placement moved during the run: %v -> %v", placed, p2)
	}
	worst, worstD := worstOf(want, got)
	t.Logf("two devices vs host: worst logit NMSE %.3e, worst |dlogit| %.4f", worst, worstD)
	if !(worst < mlaDeviceNMSE) {
		t.Fatalf("a model split across %d devices disagrees with the host: "+
			"NMSE %.3e, |dlogit| %.3f", busy, worst, worstD)
	}
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestTheSeamMovesAcrossTwoDevices shrinks and regrows a placement that spans
// TWO accelerators, mid-sequence, with live history behind it. ReleaseLayers
// recomputes g.cur from the tail, so a partial release on a two-device
// placement leaves cur on the second device, a path one device cannot reach.
// Each moved block's KV cache must follow it home and back out.
//
// The reference is the same run with the seam held still, so the move is the
// only variable. It fails against migrateKV as a no-op in the to-device
// direction.
func TestTheSeamMovesAcrossTwoDevices(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := Open(path, noTune, WithKVF16(false))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	ids := []int32{5, 11, 23, 41, 67, 89, 101, 127, 151}
	cap3 := m.PageSize()*uint64(m.Cfg.NLayer)/3 + m.PageSize()

	// The reference: the same tokens, the same two devices, seam held still.
	// Not the host: the device/host reduction-order band is as wide as a
	// migration bug on a short prompt.
	ref := func() ([][]float32, []int) {
		g, err := openTier(t, tier.WithDevices("all"), tier.WithDeviceTune(tier.TuneOff),
			tier.WithBudget(cap3))
		if err != nil || g == nil {
			noDevice(t, "devices", err)
		}
		defer g.Close()
		st := m.NewState(len(ids) + 1)
		defer st.Close()
		st.SetDeviceLayers(g, -1)
		return runIDs(t, st, ids), g.Placed()
	}
	want, placed := ref()
	busy := 0
	for _, n := range placed {
		if n > 0 {
			busy++
		}
	}
	if busy < 2 {
		t.Skipf("only %d device took blocks (%v) -- no seam spans two devices here",
			busy, placed)
	}

	g, err := openTier(t, tier.WithDevices("all"), tier.WithDeviceTune(tier.TuneOff),
		tier.WithBudget(cap3))
	if err != nil || g == nil {
		noDevice(t, "devices", err)
	}
	defer g.Close()
	st := m.NewState(len(ids) + 1)
	defer st.Close()
	st.SetDeviceLayers(g, -1)
	full := st.GPULayers()
	t.Logf("seam held still: %v (%d block(s))", placed, full)

	got := make([][]float32, len(ids))
	step := func(lo, hi int) {
		for i := lo; i < hi; i++ {
			lg, err := st.Forward(ids[i])
			if err != nil {
				t.Fatalf("token %d: %v", i, err)
			}
			got[i] = append([]float32(nil), lg...)
		}
	}
	a, b := len(ids)/3, 2*len(ids)/3
	step(0, a)
	// A partial shrink, which leaves cur on the second device. Half the placed
	// blocks go home; the ones that stay are the first device's.
	half := full / 2
	if n := st.SetGPULayers(half); n != half {
		t.Fatalf("shrinking to %d left %d placed: %v", half, n, g.Err())
	}
	t.Logf("after shrink to %d: %v, %d crossing(s)", half, g.Placed(), g.Crossings())
	step(a, b)
	if n := st.SetGPULayers(full); n != full {
		t.Fatalf("regrowing to %d took %d: %v", full, n, g.Err())
	}
	t.Logf("after regrow to %d: %v, %d crossing(s)", full, g.Placed(), g.Crossings())
	step(b, len(ids))

	worst, worstD := worstOf(want, got)
	t.Logf("seam moved across two devices vs seam held still: NMSE %.3e, |dlogit| %.4f",
		worst, worstD)
	if !(worst < mlaDeviceNMSE) {
		t.Fatalf("a seam that moved across two devices lost its history: "+
			"NMSE %.3e, |dlogit| %.3f", worst, worstD)
	}
}

// TestTheHeadRidesTheRightDeviceOfTwo places EVERY block across two devices, so
// that the output projection lands on the second one and rides its submission
// after a crossing through the host.
//
// The head is offered only when every block is placed, so this uses half the
// model per device (all blocks, two devices busy) rather than the split gate's
// third. The head-on-another-device arm is covered (as unreachable) by
// TestTheHeadGetsItsOwnSubmissionWhenItIsNotOnTheLastDevice.
func TestTheHeadRidesTheRightDeviceOfTwo(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := Open(path, noTune, WithKVF16(false))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	ids := []int32{5, 11, 23, 41, 67, 89}
	want := teacherForce(t, m, ids)

	// Half the model per device, plus the dense region the head lives in.
	cap2 := m.PageSize()*uint64(m.Cfg.NLayer)/2 + m.PageSize() + m.HostBytes()/4
	g, err := openTier(t, tier.WithDevices("all"), tier.WithDeviceTune(tier.TuneOff),
		tier.WithBudget(cap2))
	if err != nil || g == nil {
		noDevice(t, "devices", err)
	}
	defer g.Close()
	if len(g.Placed()) < 2 {
		t.Skip("this host has one device")
	}

	st := m.NewState(len(ids) + 1)
	defer st.Close()
	st.SetDeviceLayers(g, -1)
	placed := g.Placed()
	busy := 0
	for _, n := range placed {
		if n > 0 {
			busy++
		}
	}
	t.Logf("%v placed (%d of %d), head on device: %v, %d crossing(s)",
		placed, st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), g.Crossings())
	if busy < 2 || st.GPULayers() != m.Cfg.NLayer {
		t.Skipf("need every block placed across two devices; got %v (%d of %d)",
			placed, st.GPULayers(), m.Cfg.NLayer)
	}
	if !st.HeadOnDevice() {
		// Not a pass: a head left on the host is the sibling gate's case.
		t.Skip("every block is placed and the head stayed on the host -- " +
			"this gate would prove nothing")
	}

	got := runIDs(t, st, ids)
	worst, worstD := worstOf(want, got)
	t.Logf("head on the second device vs host: NMSE %.3e, |dlogit| %.4f", worst, worstD)
	if !(worst < mlaDeviceNMSE) {
		t.Fatalf("the output projection on a two-device placement disagrees with "+
			"the host: NMSE %.3e, |dlogit| %.3f", worst, worstD)
	}
}

// TestTheHeadGetsItsOwnSubmissionWhenItIsNotOnTheLastDevice reaches the arm of
// GPU.Layers that a static placement cannot: the output projection sitting on a
// device that holds no block of the range being run.
//
// That arm is unreachable from model/, so this skips with the reason:
// PrepHead targets g.tail(), the seam shrink clears s.head and s.headReady, and
// an earlier device holding the projection runs with a partial seam that never
// hands GPU.Layers a head. See docs/engineering-history/placement.md. The gate
// is here to run the arm if a placement policy ever makes it live.
func TestTheHeadGetsItsOwnSubmissionWhenItIsNotOnTheLastDevice(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := Open(path, noTune, WithKVF16(false))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	ids := []int32{5, 11, 23, 41, 67, 89}
	want := teacherForce(t, m, ids)

	cap2 := m.PageSize()*uint64(m.Cfg.NLayer)/2 + m.PageSize() + m.HostBytes()/4
	g, err := openTier(t, tier.WithDevices("all"), tier.WithDeviceTune(tier.TuneOff),
		tier.WithBudget(cap2))
	if err != nil || g == nil {
		noDevice(t, "devices", err)
	}
	defer g.Close()
	if len(g.Placed()) < 2 {
		t.Skip("this host has one device")
	}
	st := m.NewState(len(ids) + 1)
	defer st.Close()
	st.SetDeviceLayers(g, -1)
	placed := g.Placed()
	if st.GPULayers() != m.Cfg.NLayer || placed[len(placed)-1] == 0 || !st.HeadOnDevice() {
		t.Skipf("need every block across two devices with the head placed; got %v (%d of %d), head %v",
			placed, st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice())
	}

	// Shrink past the LAST device's blocks: the head stays where PrepHead put
	// it, on a device that now holds none of the range.
	keep := m.Cfg.NLayer - placed[len(placed)-1]
	if n := st.SetGPULayers(keep); n != keep {
		t.Fatalf("shrinking to %d left %d placed: %v", keep, n, g.Err())
	}
	after := g.Placed()
	t.Logf("shrank %v -> %v, head still on a device: %v, %d crossing(s)",
		placed, after, st.HeadOnDevice(), g.Crossings())
	if after[len(after)-1] != 0 {
		t.Skipf("the last device still holds %d block(s); the arm needs it empty", after[len(after)-1])
	}
	if !st.HeadOnDevice() {
		// The shrink drops the model's head while the tier's g.head survives;
		// SetHeadOnDevice(true) is the public way to re-arm it.
		st.SetHeadOnDevice(true)
		if !st.HeadOnDevice() {
			t.Skip("the shrink cleared headReady as well as head, so the projection " +
				"cannot be put back on a device that holds no block -- GPU.Layers' " +
				"own-submission arm is unreachable from model/ and stays insurance")
		}
		t.Log("the shrink dropped the head; SetHeadOnDevice(true) put it back on " +
			"a device that now holds no block of the range -- the arm under test")
	}

	got := runIDs(t, st, ids)
	worst, worstD := worstOf(want, got)
	t.Logf("head alone on its own submission vs host: NMSE %.3e, |dlogit| %.4f", worst, worstD)
	if !(worst < mlaDeviceNMSE) {
		t.Fatalf("the output projection on a device holding no block of the range "+
			"disagrees with the host: NMSE %.3e, |dlogit| %.3f", worst, worstD)
	}
}

// TestAWidePromptPipelinesAcrossTwoDevices prefills a prompt of three device
// chunks' worth on a model split over two accelerators, where the chunk is
// pipelined (GPU.pipeline): the second device runs piece k while the first
// runs piece k+1. The reference is the host on the same prompt, the prompt's
// logits and four decode steps after it, which read the history every piece
// wrote. The selection check is Stats.PipelinePieces on both devices.
func TestAWidePromptPipelinesAcrossTwoDevices(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := Open(path, noTune, WithKVF16(false))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode(strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 60), true)[:1300]
	run := func(st *State) [][]float32 {
		lg, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		out := [][]float32{append([]float32(nil), lg...)}
		for i := 0; i < 4; i++ {
			if lg, err = st.Forward(Greedy(lg)); err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	host := m.NewState(len(ids) + 8)
	want := run(host)
	host.Close()

	cap3 := m.PageSize()*uint64(m.Cfg.NLayer)/2 + 2*m.PageSize()
	g, err := openTier(t, tier.WithDevices("all"), tier.WithDeviceTune(tier.TuneOff), tier.WithBudget(cap3))
	if err != nil || g == nil {
		noDevice(t, "devices", err)
	}
	defer g.Close()
	if n := len(g.Placed()); n < 2 {
		t.Skipf("this host enumerates %d distinct device(s); the gate needs two", n)
	}
	st := m.NewState(len(ids) + 8)
	defer st.Close()
	st.SetDevice(g)
	if st.GPULayers() != m.Cfg.NLayer {
		t.Skipf("%d of %d blocks placed (%v): a pipelined chunk needs every block on a device -- this gate proved nothing",
			st.GPULayers(), m.Cfg.NLayer, g.Placed())
	}
	got := run(st)
	// Every device that holds blocks runs pieces; a host may enumerate more
	// devices than the placement used (a multi-GPU host lists each card under CUDA
	// and Vulkan).
	busy := 0
	var pieces []int
	for i, n := range g.Placed() {
		if n == 0 {
			continue
		}
		busy++
		pc := g.DevStats()[i].PipelinePieces
		pieces = append(pieces, pc)
		if pc == 0 {
			t.Fatalf("device %d holds %d blocks and ran no pipelined piece (%v): the chunk went one device after another",
				i, n, g.Placed())
		}
	}
	if busy < 2 {
		t.Skipf("the blocks went on %d device (%v): nothing here is pipelined -- this gate proved nothing", busy, g.Placed())
	}
	worst, worstD := worstOf(want, got)
	t.Logf("%v blocks; pieces %v; worst logit NMSE %.3e, |dlogit| %.4f", g.Placed(), pieces, worst, worstD)
	if !(worst < mlaDeviceNMSE) {
		t.Fatalf("a pipelined prompt over two devices disagrees with the host: NMSE %.3e, |dlogit| %.3f", worst, worstD)
	}
}
