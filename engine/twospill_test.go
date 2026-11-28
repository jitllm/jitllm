package engine

import (
	"os"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestTwoInstancesSpillOntoTheNextCard places two instances of one model on
// one fill-first tier whose first card is capped, so the blocks past its room
// go to the second, and prefills both: each prompt is three fifths of the
// history the first card has room for once both are placed (KVRoom), so the
// second's history has no room there beside the first's, and its blocks there
// spill onto the next card as copies of its own while the first's stay. Each
// instance keeps every block on the cards, composes none on the CPU while its
// tokens run, and decodes the host's tokens.
func TestTwoInstancesSpillOntoTheNextCard(t *testing.T) {
	devs := testdevices.Take(t, 2, "cuda:0,cuda:1")
	p := jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm"))
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := model.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const gen, maxSeq = 8, 4096
	spec := devs[0] + "=" + strconv.FormatUint(spillBudget(380_000_000), 10) + "," + devs[1]
	g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff),
		gpu.WithConfig(func(c *gpu.Config) { c.FillFirst = true }))
	if err != nil || g == nil {
		noDevice(t, spec, err)
	}
	defer g.Close()
	distinctCards(t, g)
	var insts [2]*Instance
	for i := range insts {
		insts[i] = testInstance(t, m, maxSeq)
		defer insts[i].Close()
		if err := insts[i].SetDevice(g); err != nil {
			t.Fatal(err)
		}
	}
	if pl := g.Placed(); len(pl) != 2 || pl[0] == 0 || pl[1] == 0 {
		t.Skipf("the capped card did not spill onto the next one: placed %v -- this gate proved nothing", pl)
	}
	room, page := g.KVRoom()
	n := max(room*3/5, 1)
	if n+gen >= maxSeq {
		t.Fatalf("the first card has room for %d positions of history, past the %d an instance holds: the cap leaves a block's room", room, maxSeq)
	}
	t.Logf("the first card has room for %d positions (pages of %d): prompts of %d", room, page, n)
	var prompts [2][]int32
	for i, s := range []string{"The capital of France is a city that ", "Once upon a time, in a small village by the sea, "} {
		prompts[i] = m.Vocab.Encode(s, true)
		for len(prompts[i]) < n {
			prompts[i] = append(prompts[i], m.Vocab.Encode(s, false)...)
		}
		prompts[i] = prompts[i][:n]
	}
	var want [2][]int32
	for i, pr := range prompts {
		st := testInstance(t, m, n+gen+1)
		lg, err := st.Prefill(pr)
		for err == nil && len(want[i]) < gen {
			want[i] = append(want[i], Greedy(lg))
			lg, err = st.Forward(want[i][len(want[i])-1])
		}
		st.Close()
		if err != nil {
			t.Fatal(err)
		}
	}
	var before [2][]int
	for i, st := range insts {
		before[i] = append([]int(nil), st.Account(nn.CPU).Forms...)
	}
	var got [2][]int32
	var lg [2][]float32
	for i, st := range insts {
		if lg[i], err = st.Prefill(prompts[i]); err != nil {
			t.Fatal(err)
		}
	}
	for len(got[1]) < gen {
		for i, st := range insts {
			got[i] = append(got[i], Greedy(lg[i]))
			if len(got[i]) < gen {
				if lg[i], err = st.Forward(got[i][len(got[i])-1]); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	spills := 0
	for i, st := range insts {
		formed := 0
		for li, n := range st.Account(nn.CPU).Forms {
			if li < len(before[i]) {
				n -= before[i][li]
			}
			formed += n
		}
		spills += st.Spills()
		t.Logf("instance %d: %v on %s, %v on %s, %d spill(s), %d relocation(s), %d demotion(s), %d CPU form(s) (%s)",
			i, st.placedOn(devs[0]), devs[0], st.placedOn(devs[1]), devs[1], st.Spills(), st.Relocations(), st.DeviceDemotions(), formed, g.Err())
		if n := len(st.placedOn(devs[0])) + len(st.placedOn(devs[1])); n != m.Cfg.NLayer || st.GPULayers() != m.Cfg.NLayer ||
			st.DeviceDemotions() != 0 || st.Relocations() != 0 || formed != 0 {
			t.Fatalf("instance %d: %d/%d blocks on the cards (%d counted), %d relocation(s), %d demotion(s), %d block(s) composed on the CPU while the tokens ran (%s)",
				i, n, m.Cfg.NLayer, st.GPULayers(), st.Relocations(), st.DeviceDemotions(), formed, g.Err())
		}
		for j := range want[i] {
			if got[i][j] == want[i][j] {
				continue
			}
			if gap := hostGap(t, m, prompts[i], want[i][:j], want[i][j], got[i][j]); gap > batchTie {
				t.Fatalf("instance %d token %d: device %d, host %d, %.4f apart\ngot  %v\nwant %v", i, j, got[i][j], want[i][j], gap, got[i], want[i])
			}
			t.Logf("instance %d token %d: a tie, not compared past it", i, j)
			break
		}
	}
	if spills == 0 {
		t.Fatal("no block spilled: the first card held both histories, and the gate proved nothing")
	}
	if s := g.Stats(); s.Blocks == 0 || s.PoolOver != 0 {
		t.Fatalf("the tier ran %d block(s), %d charge(s) past the pool", s.Blocks, s.PoolOver)
	}
}

// TestTwoInstancesShareACappedFirstDevice places two instances of one model
// on one fill-first tier whose first device is capped, the second instance
// sharing the blocks the first placed. The second instance arms a spill of
// the block at the first device's edge and the first offers that block
// before the spill's own offer, as two instances placing at once do; then
// each prefills past the positions a device reserves. Each instance keeps its blocks on the devices, composes none on
// the CPU, and decodes the host's tokens, whether the two devices are two
// cards or one card under two backends (one pool). The spill mark was the
// tier's one: the first instance's plan-only offer followed it to the second
// device as a block of nothing, which the composer refused ("its graph binds
// attn_q, which the block does not hold"), and its full offer moved the
// block off the device it was on.
func TestTwoInstancesShareACappedFirstDevice(t *testing.T) {
	devs := testdevices.Take(t, 2, "cuda:0,cuda:1")
	p := jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm"))
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := model.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	// Each prompt runs past the 2048 positions a device reserves (tier
	// initKVCap), so each instance's history grows on both devices.
	var prompts [2][]int32
	for i, s := range []string{"The capital of France is Paris. Water boils at 100 degrees. ",
		"Once upon a time, in a small village by the sea, a fisherman mended his nets. "} {
		for len(prompts[i]) < 2600 {
			prompts[i] = append(prompts[i], m.Vocab.Encode(s, false)...)
		}
	}
	const gen = 8
	maxSeq := max(len(prompts[0]), len(prompts[1])) + gen + 1
	var want [2][]int32
	for i, pr := range prompts {
		st := testInstance(t, m, maxSeq)
		lg, err := st.Prefill(pr)
		for err == nil && len(want[i]) < gen {
			want[i] = append(want[i], Greedy(lg))
			lg, err = st.Forward(want[i][len(want[i])-1])
		}
		st.Close()
		if err != nil {
			t.Fatal(err)
		}
	}
	spec := devs[0] + "=" + strconv.FormatUint(spillBudget(380_000_000), 10) + "," + devs[1]
	g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff),
		gpu.WithConfig(func(c *gpu.Config) { c.FillFirst = true }))
	if err != nil || g == nil {
		noDevice(t, spec, err)
	}
	defer g.Close()
	t.Logf("%s: %q", spec, cardClash(g.Hardware()))
	var insts [2]*Instance
	var before [2][]int
	for i := range insts {
		insts[i] = testInstance(t, m, maxSeq)
		defer insts[i].Close()
		if err := insts[i].SetDevice(g); err != nil {
			t.Fatal(err)
		}
		before[i] = append([]int(nil), insts[i].Account(nn.CPU).Forms...)
		t.Logf("instance %d placed: %v on %s, %v on %s (%s)", i, insts[i].placedOn(devs[0]), devs[0], insts[i].placedOn(devs[1]), devs[1], g.Err())
	}
	if pl := g.Placed(); len(pl) != 2 || pl[0] == 0 || pl[1] == 0 {
		t.Fatalf("the capped device did not spill onto the next one: placed %v", pl)
	}
	for i, st := range insts {
		if n := len(st.placedOn(devs[0])) + len(st.placedOn(devs[1])); n != m.Cfg.NLayer {
			t.Fatalf("instance %d: %d of %d blocks placed on the devices (%s)", i, n, m.Cfg.NLayer, g.Err())
		}
	}
	// The second instance arms a spill of the block at the first device's
	// edge, and the first instance offers that block before the spill's own
	// offer (two instances placing at once): the first's offer shares the
	// block where it is.
	on0 := insts[0].placedOn(devs[0])
	edge := on0[len(on0)-1]
	if insts[1].ldSpill == nil || !insts[1].ldSpill.SpillAfter(edge) {
		t.Fatalf("block %d is on the first device and the second instance's spill of it was refused", edge)
	}
	insts[0].unmarkOnDev(edge)
	insts[0].offerRange(edge, edge+1)
	if at, ok := insts[0].DeviceOf(edge); !ok || at != nn.DeviceName(devs[0]) {
		t.Fatalf("the first instance's block %d is on %q (placed %v), not %s, after its offer beside the other's spill (%s)",
			edge, at, ok, devs[0], g.Err())
	}
	// What the offer composed is the placement's, not the tokens'.
	for i, st := range insts {
		before[i] = append(before[i][:0], st.Account(nn.CPU).Forms...)
	}
	var got [2][]int32
	var lg [2][]float32
	for i, st := range insts {
		if lg[i], err = st.Prefill(prompts[i]); err != nil {
			t.Fatal(err)
		}
	}
	for len(got[1]) < gen {
		for i, st := range insts {
			got[i] = append(got[i], Greedy(lg[i]))
			if len(got[i]) < gen {
				if lg[i], err = st.Forward(got[i][len(got[i])-1]); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	for i, st := range insts {
		formed := 0
		for li, n := range st.Account(nn.CPU).Forms {
			if li < len(before[i]) {
				n -= before[i][li]
			}
			formed += n
		}
		t.Logf("instance %d: %v on %s, %v on %s, %d spill(s), %d relocation(s), %d demotion(s), %d CPU form(s) (%s)",
			i, st.placedOn(devs[0]), devs[0], st.placedOn(devs[1]), devs[1], st.Spills(), st.Relocations(), st.DeviceDemotions(), formed, g.Err())
		if n := len(st.placedOn(devs[0])) + len(st.placedOn(devs[1])); n != m.Cfg.NLayer || st.GPULayers() != m.Cfg.NLayer ||
			st.DeviceDemotions() != 0 || st.Relocations() != 0 || formed != 0 {
			t.Fatalf("instance %d: %d/%d blocks on the devices (%d counted), %d relocation(s), %d demotion(s), %d block(s) composed on the CPU while the tokens ran (%s)",
				i, n, m.Cfg.NLayer, st.GPULayers(), st.Relocations(), st.DeviceDemotions(), formed, g.Err())
		}
		for j := range want[i] {
			if got[i][j] == want[i][j] {
				continue
			}
			if gap := hostGap(t, m, prompts[i], want[i][:j], want[i][j], got[i][j]); gap > batchTie {
				t.Fatalf("instance %d token %d: device %d, host %d, %.4f apart\ngot  %v\nwant %v", i, j, got[i][j], want[i][j], gap, got[i], want[i])
			}
			t.Logf("instance %d token %d: a tie, not compared past it", i, j)
			break
		}
	}
	for _, s := range g.DevStats() {
		if s.PoolOver != 0 || s.ScratchOver != 0 {
			t.Fatalf("%s: %d charge(s) and %d scratch window(s) past the pool", s.Device, s.PoolOver, s.ScratchOver)
		}
	}
}
