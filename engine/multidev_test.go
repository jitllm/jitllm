//go:build linux

package engine

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
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
	instOpts := []Option{WithKVF16(false)}
	m, err := model.Open(path)
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	ids := []int32{5, 11, 23, 41, 67, 89}
	want := teacherForce(t, m, ids, instOpts...)

	// A budget of a third of the model per device forces the split; otherwise
	// the first card takes everything. It is derived from the model so another
	// fixture still splits.
	cap3 := m.PageSize()*uint64(m.Cfg.NLayer)/3 + m.PageSize()
	g, err := openGPU(t, gpu.WithDevices("all"), gpu.WithDeviceTune(gpu.TuneOff),
		gpu.WithBudget(cap3))
	if err != nil || g == nil {
		noDevice(t, "devices", err)
	}
	defer g.Close()
	if n := len(g.Placed()); n < 2 {
		t.Skipf("this host enumerates %d distinct device(s); the gate needs two", n)
	}
	t.Logf("%d device(s), %.1f MiB budget each, %d block(s) of %.1f MiB",
		len(g.Placed()), float64(cap3)/(1<<20), m.Cfg.NLayer, float64(m.PageSize())/(1<<20))

	st := testInstance(t, m, len(ids)+1, instOpts...)
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
	instOpts := []Option{WithKVF16(false)}
	m, err := model.Open(path)
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
		g, err := openGPU(t, gpu.WithDevices("all"), gpu.WithDeviceTune(gpu.TuneOff),
			gpu.WithBudget(cap3))
		if err != nil || g == nil {
			noDevice(t, "devices", err)
		}
		defer g.Close()
		st := testInstance(t, m, len(ids)+1, instOpts...)
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

	g, err := openGPU(t, gpu.WithDevices("all"), gpu.WithDeviceTune(gpu.TuneOff),
		gpu.WithBudget(cap3))
	if err != nil || g == nil {
		noDevice(t, "devices", err)
	}
	defer g.Close()
	st := testInstance(t, m, len(ids)+1, instOpts...)
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

// headBesideHalf is a budget a device of two takes half of m's blocks under
// with the output projection beside them, from what one tier charges for the
// whole model: its blocks with the projection on the CPU, then with it placed
// -- the projection, its vectors and logits, and the reserved width's per-row
// logits it is admitted with. The budget is a block over that half, and below
// the whole model, so every block spreads over two devices and the tail takes
// the projection.
func headBesideHalf(t *testing.T, m *model.Model, devs string, opts []Option) uint64 {
	t.Helper()
	charge := func(head bool) (used, scratch uint64) {
		g, err := openGPU(t, gpu.WithDevices(devs), gpu.WithDeviceTune(gpu.TuneOff))
		if err != nil || g == nil {
			noDevice(t, devs, err)
		}
		defer g.Close()
		st := testInstance(t, m, 8, append(append([]Option(nil), opts...), WithHeadPlacement(head))...)
		defer st.Close()
		st.SetDeviceLayers(g, -1)
		if st.GPULayers() != m.Cfg.NLayer || st.HeadOnDevice() != head {
			t.Fatalf("unbudgeted, the devices took %d of %d blocks, the projection on a device %v, asked %v: %s",
				st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), head, g.Err())
		}
		for _, ds := range g.DevStats() {
			used += ds.BudgetUsed
			scratch += ds.ScratchBytes
		}
		return used, scratch
	}
	blocks, scratch := charge(false)
	whole, _ := charge(true)
	if whole <= blocks || blocks <= scratch {
		t.Fatalf("the blocks charge %d bytes (%d of scratch) and the projection %d more", blocks, scratch, int64(whole)-int64(blocks))
	}
	n := uint64(m.Cfg.NLayer)
	block := (blocks - scratch + n - 1) / n
	b := scratch + (n-n/2+1)*block + (whole - blocks)
	t.Logf("a block charges %d bytes, the projection %d, scratch %d: a budget of %d a device", block, whole-blocks, scratch, b)
	return b
}

// TestTheHeadRidesTheRightDeviceOfTwo places EVERY block across two devices, so
// that the output projection lands on the second one and rides its submission
// after a crossing through the host.
//
// The budget is measured (headBesideHalf): half the model a device with the
// projection beside the tail's half, so both devices are busy and the
// projection fits. A budget written as half the model plus a guessed margin
// left the projection on the host on a V100 pair -- it is admitted with the
// reserved width's per-row logits, 64 MiB beside its own 55 MB -- and the gate
// skipped.
func TestTheHeadRidesTheRightDeviceOfTwo(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	instOpts := []Option{WithKVF16(false)}
	m, err := model.Open(path)
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	ids := []int32{5, 11, 23, 41, 67, 89}
	want := teacherForce(t, m, ids, instOpts...)

	a, b := moveDevices(t)
	devs := a + "," + b
	cap2 := headBesideHalf(t, m, devs, instOpts)
	g, err := openGPU(t, gpu.WithDevices(devs), gpu.WithDeviceTune(gpu.TuneOff),
		gpu.WithBudget(cap2))
	if err != nil || g == nil {
		noDevice(t, devs, err)
	}
	defer g.Close()
	distinctCards(t, g)

	st := testInstance(t, m, len(ids)+1, instOpts...)
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
		t.Fatalf("every block across two devices under a budget of %d each: got %v (%d of %d): %s",
			cap2, placed, st.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	if !st.HeadOnDevice() {
		t.Fatalf("every block is placed and the projection stayed on the host under a budget of %d each: %s",
			cap2, g.Err())
	}
	hw := g.Hardware()
	tail := len(placed) - 1
	for placed[tail] == 0 {
		tail--
	}
	if at, _ := st.ld.(nn.NamedDevice).HeadDeviceName(); at == "" || at != hw[tail].Name {
		t.Fatalf("the projection is on %q, not on %s, the device holding the last block", at, hw[tail].Name)
	}
	before := g.DevStats()
	got := runIDs(t, st, ids)
	for d, ds := range g.DevStats() {
		w := 0
		if d == tail {
			w = len(ids)
		}
		if ran := ds.Heads - before[d].Heads; ran != w {
			t.Fatalf("%s ran the projection %d time(s) over %d tokens, want %d", hw[d].Name, ran, len(ids), w)
		}
	}
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
// Every block goes across two devices with the projection on the tail, then
// the seam shrinks past the tail's blocks, which gives the projection back
// (SetGPULayers releases it), and MoveHead names the tail for it again: a
// device holding none of the range, so the projection runs as a submission
// of its own there, counted per device (Stats.Heads).
func TestTheHeadGetsItsOwnSubmissionWhenItIsNotOnTheLastDevice(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	instOpts := []Option{WithKVF16(false)}
	m, err := model.Open(path)
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	ids := []int32{5, 11, 23, 41, 67, 89}
	want := teacherForce(t, m, ids, instOpts...)

	a, b := moveDevices(t)
	devs := a + "," + b
	cap2 := headBesideHalf(t, m, devs, instOpts)
	g, err := openGPU(t, gpu.WithDevices(devs), gpu.WithDeviceTune(gpu.TuneOff),
		gpu.WithBudget(cap2))
	if err != nil || g == nil {
		noDevice(t, devs, err)
	}
	defer g.Close()
	distinctCards(t, g)
	st := testInstance(t, m, len(ids)+1, instOpts...)
	defer st.Close()
	st.SetDeviceLayers(g, -1)
	placed := g.Placed()
	tail := len(placed) - 1
	if st.GPULayers() != m.Cfg.NLayer || placed[tail] == 0 || placed[0] == 0 || !st.HeadOnDevice() {
		t.Fatalf("every block across two devices with the projection placed under a budget of %d each: got %v (%d of %d), projection %v: %s",
			cap2, placed, st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), g.Err())
	}

	keep := m.Cfg.NLayer - placed[tail]
	if n := st.SetGPULayers(keep); n != keep {
		t.Fatalf("shrinking to %d left %d placed: %v", keep, n, g.Err())
	}
	after := g.Placed()
	if after[tail] != 0 || st.HeadOnDevice() {
		t.Fatalf("shrank %v -> %v: the tail holds %d block(s), the projection on a device %v",
			placed, after, after[tail], st.HeadOnDevice())
	}
	name := g.Hardware()[tail].Name
	if held := g.DevStats()[tail].HeadBytes; held != 0 {
		t.Fatalf("the shrink gave the projection back and %s still holds %d bytes of it", name, held)
	}
	mv, err := st.MoveHead(name)
	if err != nil || !mv.Head || !st.HeadOnDevice() {
		t.Fatalf("the projection to %s, which holds no block: moved %v, on a device %v, %v: %s", name, mv.Head, st.HeadOnDevice(), err, g.Err())
	}
	t.Logf("shrank %v -> %v and named %s for the projection, %d crossing(s)", placed, after, name, g.Crossings())

	before := g.DevStats()
	got := runIDs(t, st, ids)
	for d, ds := range g.DevStats() {
		w := 0
		if d == tail {
			w = len(ids)
		}
		if ran := ds.Heads - before[d].Heads; ran != w {
			t.Fatalf("%s ran the projection %d time(s) over %d tokens, want %d", g.Hardware()[d].Name, ran, len(ids), w)
		}
	}
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
	instOpts := []Option{WithKVF16(false)}
	m, err := model.Open(path)
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode(strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 60), true)[:1300]
	run := func(st *Instance) [][]float32 {
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
	host := testInstance(t, m, len(ids)+8, instOpts...)
	want := run(host)
	host.Close()

	// Each card is capped at what the whole model takes on one card, less a
	// little under half its blocks: measured, since the scratch and history
	// beside the blocks are no flat number of pages. Two cards then hold every
	// block between them, and neither holds them all.
	probe, err := openGPU(t, gpu.WithDevices("all"), gpu.WithDeviceTune(gpu.TuneOff),
		gpu.WithConfig(func(c *gpu.Config) { c.FillFirst = true }))
	if err != nil || probe == nil {
		noDevice(t, "devices", err)
	}
	pst := testInstance(t, m, len(ids)+8, instOpts...)
	pst.SetDevice(probe)
	if pst.GPULayers() != m.Cfg.NLayer {
		pst.Close()
		probe.Close()
		t.Skipf("one card holds %d of %d blocks: the cap is measured from a card holding them all", pst.GPULayers(), m.Cfg.NLayer)
	}
	whole := probe.Budgets()[0].Used
	pst.Close()
	probe.Close()
	cap3 := whole - uint64(m.Cfg.NLayer/2-2)*m.PageSize()
	g, err := openGPU(t, gpu.WithDevices("all"), gpu.WithDeviceTune(gpu.TuneOff), gpu.WithBudget(cap3))
	if err != nil || g == nil {
		noDevice(t, "devices", err)
	}
	defer g.Close()
	if n := len(g.Placed()); n < 2 {
		t.Skipf("this host enumerates %d distinct device(s); the gate needs two", n)
	}
	st := testInstance(t, m, len(ids)+8, instOpts...)
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
