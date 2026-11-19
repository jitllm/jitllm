package engine

import (
	"fmt"
	"os"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// cardClash names the first two of a tier's devices that are one physical card
// (equal Hardware keys, the UUID both CUDA and Vulkan report), or "" when the
// tier's devices are distinct. A key the backend could not supply is the Ref
// and matches nothing but itself.
func cardClash(hw []gpu.Hardware) string {
	for i := range hw {
		for j := i + 1; j < len(hw); j++ {
			if hw[i].Key == hw[j].Key {
				return fmt.Sprintf("%s and %s are one physical card (%s)", hw[i].Ref, hw[j].Ref, hw[i].Key)
			}
		}
	}
	return ""
}

// distinctCards fails a gate that moves or splits blocks between two devices
// when those devices are one card reached through two backends: they share one
// memory pool, so the gate would measure a card too small for both arms rather
// than a move between cards. It fails and does not skip, so a mis-selected
// JITLLM_TEST_DEVICES is seen.
func distinctCards(t testing.TB, g *gpu.GPU) {
	t.Helper()
	if c := cardClash(g.Hardware()); c != "" {
		t.Fatalf("a two-card gate was given one card: %s; name two cards in JITLLM_TEST_DEVICES (a CUDA ordinal and a Vulkan ordinal of another card)", c)
	}
}

// TestCardClashNamesBothDevices is the check's violation on stand-in hardware:
// two entries with one key are refused, naming both; distinct keys pass.
func TestCardClashNamesBothDevices(t *testing.T) {
	same := []gpu.Hardware{{Key: "u1", Ref: "cuda:0"}, {Key: "u1", Ref: "vulkan:4"}}
	if c := cardClash(same); c == "" {
		t.Fatal("cuda:0 and vulkan:4 with one key were not flagged")
	} else if want := "cuda:0 and vulkan:4"; len(c) < len(want) || c[:len(want)] != want {
		t.Fatalf("the message does not name both devices: %q", c)
	}
	if c := cardClash([]gpu.Hardware{{Key: "u1", Ref: "cuda:0"}, {Key: "u2", Ref: "vulkan:4"}}); c != "" {
		t.Fatalf("two cards were flagged: %s", c)
	}
}

// TestOneCardUnderTwoBackendsIsOneBudget opens a CUDA and a Vulkan device that
// are one card as a single tier and holds two things: the tier spends one pool
// for both (the card's bytes are counted once), and the two-card gates' check
// refuses the pair.
func TestOneCardUnderTwoBackendsIsOneBudget(t *testing.T) {
	c := testdevices.One("cuda:0")
	vk := []string{"vulkan:0", "vulkan:1", "vulkan:2", "vulkan:3", "vulkan:4", "vulkan:5", "vulkan:6", "vulkan:7"}
	for _, v := range testdevices.Within(vk) {
		g, err := openGPU(t, gpu.WithDevices(c+","+v), gpu.WithDeviceTune(gpu.TuneOff))
		if err != nil || g == nil {
			continue
		}
		clash := cardClash(g.Hardware())
		bs := g.Budgets()
		g.Close()
		if clash == "" {
			continue
		}
		t.Logf("%s + %s: %s", c, v, clash)
		if len(bs) != 2 || bs[0].Pool != bs[1].Pool || bs[0].PoolLimit != bs[1].PoolLimit {
			t.Fatalf("one card under two backends has budgets %+v: they must spend one pool", bs)
		}
		if bs[0].PoolLimit >= bs[0].Limit+bs[1].Limit {
			t.Fatalf("pool limit %d is the sum of the devices' limits (%d + %d): the card is counted twice",
				bs[0].PoolLimit, bs[0].Limit, bs[1].Limit)
		}
		return
	}
	noDevice(t, c+" and a Vulkan device of the same card", nil)
}

// TestOneCardUnderTwoBackendsSpillsAndRelocates is the spill gate on one card
// listed twice, a CUDA device and the Vulkan device of the same card: one
// pool, sized at the larger cap, both views spending it at once while a
// prompt's pieces are pipelined through them. A view that found room and
// uploaded after lost it to the other between the two, and the prompt was
// declined (4 of 9 runs on a V100).
func TestOneCardUnderTwoBackendsSpillsAndRelocates(t *testing.T) {
	c := testdevices.One("cuda:0")
	vk := []string{"vulkan:0", "vulkan:1", "vulkan:2", "vulkan:3", "vulkan:4", "vulkan:5", "vulkan:6", "vulkan:7"}
	for _, v := range testdevices.Within(vk) {
		g, err := openGPU(t, gpu.WithDevices(c+","+v), gpu.WithDeviceTune(gpu.TuneOff))
		if err != nil || g == nil {
			continue
		}
		clash := cardClash(g.Hardware())
		g.Close()
		if clash == "" {
			continue
		}
		t.Logf("%s + %s: %s", c, v, clash)
		fullCardSpills(t, []string{c, v}, func(t testing.TB, g *gpu.GPU) {
			t.Helper()
			if cardClash(g.Hardware()) == "" {
				t.Fatalf("%s and %s opened as two cards", c, v)
			}
		})
		return
	}
	noDevice(t, c+" and a Vulkan device of the same card", nil)
}

// sameCardPair is a CUDA device and the Vulkan device of the same card, or
// nothing when the run has none.
func sameCardPair(t *testing.T) []string {
	t.Helper()
	c := testdevices.One("cuda:0")
	vk := []string{"vulkan:0", "vulkan:1", "vulkan:2", "vulkan:3", "vulkan:4", "vulkan:5", "vulkan:6", "vulkan:7"}
	for _, v := range testdevices.Within(vk) {
		g, err := openGPU(t, gpu.WithDevices(c+","+v), gpu.WithDeviceTune(gpu.TuneOff))
		if err != nil || g == nil {
			continue
		}
		clash := cardClash(g.Hardware())
		g.Close()
		if clash != "" {
			t.Logf("%s + %s: %s", c, v, clash)
			return []string{c, v}
		}
	}
	noDevice(t, c+" and a Vulkan device of the same card", nil)
	return nil
}

// TestOneCardUnderTwoBackendsGrowsHistoryOnBoth is the history's half of the
// pool one card's two views spend: a long prompt's pieces are pipelined
// through both devices (GPU.pipeline), so its history grows on both devices
// of one pool at once, the pool sized to what the run holds and a little more
// (JITLLM_SQUEEZE_SLACK bytes, less where negative). Every growth holds its room before it
// allocates (stateHold); checked and charged after, the two devices spent the
// same bytes and the pool went past its limit (Stats.PoolOver).
func TestOneCardUnderTwoBackendsGrowsHistoryOnBoth(t *testing.T) {
	devs := sameCardPair(t)
	p := jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm"))
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := model.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	var prompt []int32
	for len(prompt) < 2600 { // past the 2048 positions a device reserves (tier initKVCap)
		prompt = append(prompt, m.Vocab.Encode("The capital of France is Paris. Water boils at 100 degrees. ", false)...)
	}
	const gen = 8
	run := func(st *Instance) ([]int32, error) {
		lg, err := st.Prefill(prompt)
		if err != nil {
			return nil, err
		}
		out := []int32{Greedy(lg)}
		for len(out) < gen {
			next, err := st.ForwardGreedy(out[len(out)-1])
			if err != nil {
				return out, err
			}
			out = append(out, next)
		}
		return out, nil
	}
	host := testInstance(t, m, len(prompt)+gen+1)
	want, err := run(host)
	host.Close()
	if err != nil {
		t.Fatal(err)
	}
	// The first device is capped so the second takes blocks too (as the
	// spill gate's); the squeeze is the pool's.
	first := spillBudget(380_000_000)
	open := func(caps map[string]uint64) *gpu.GPU {
		spec := ""
		for _, name := range devs {
			if spec != "" {
				spec += ","
			}
			spec += name
			if b := caps[name]; b > 0 {
				spec += "=" + strconv.FormatUint(b, 10)
			}
		}
		g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff),
			gpu.WithConfig(func(c *gpu.Config) { c.FillFirst = true }))
		if err != nil || g == nil {
			t.Fatalf("opening %s: %v", spec, err)
		}
		if cardClash(g.Hardware()) == "" {
			t.Fatalf("%s opened as two cards", spec)
		}
		return g
	}
	// once runs the prompt on g, and returns what g holds once it has run,
	// before the instance closes.
	// whole asks every block on the card: the squeeze may send one home
	// (a relocation, as the spill gate's), never demote the instance.
	once := func(t *testing.T, g *gpu.GPU, whole bool) (got []int32, held uint64, ds []gpu.Stats) {
		t.Helper()
		st := testInstance(t, m, len(prompt)+gen+1)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		got, err := run(st)
		if err != nil {
			t.Fatalf("%v (%s)", err, g.Err())
		}
		if pl := g.Placed(); len(pl) != 2 || pl[0] == 0 || pl[1] == 0 {
			t.Fatalf("placed %v: the gate wants blocks on both devices of the card", pl)
		}
		t.Logf("%d of %d blocks on the card, %d relocation(s), %d spill(s)", st.GPULayers(), m.Cfg.NLayer, st.Relocations(), st.Spills())
		if whole && st.GPULayers() != m.Cfg.NLayer || st.DeviceDemotions() != 0 {
			t.Fatalf("%d of %d blocks ran on the card, %d demotion(s): %s",
				st.GPULayers(), m.Cfg.NLayer, st.DeviceDemotions(), g.Err())
		}
		for _, b := range g.Budgets() {
			held += b.Used
		}
		return got, held, g.DevStats()
	}
	probe := open(map[string]uint64{devs[0]: first})
	_, held, _ := once(t, probe, true)
	probe.Close()
	slack := int64(16 << 20)
	if v, err := strconv.ParseInt(os.Getenv("JITLLM_SQUEEZE_SLACK"), 10, 64); err == nil {
		slack = v
	}
	g := open(map[string]uint64{devs[0]: first, devs[1]: uint64(int64(held) + slack)})
	defer g.Close()
	got, _, ds := once(t, g, false)
	for _, b := range g.Budgets() {
		t.Logf("%s: %d of %d used, pool %s %d", b.Name, b.Used, b.Limit, b.Pool, b.PoolLimit)
	}
	for i, s := range ds {
		t.Logf("%s: history %d bytes (pool %d), %d page-out(s), %d of the other device's, %d KV eviction(s), %d charge(s) and %d scratch window(s) past the pool",
			s.Device, s.KVBytes, s.KVPoolBytes, s.PageOuts, s.PeerPageOuts, s.KVEvictions, s.PoolOver, s.ScratchOver)
		if s.KVPoolBytes == 0 && s.KVBytes == 0 {
			t.Fatalf("device %d holds no history: the gate grew history on one device only", i)
		}
		if s.PoolOver != 0 {
			t.Fatalf("%s: %d charge(s) left the pool past its limit", s.Device, s.PoolOver)
		}
	}
	for j := range want {
		if got[j] == want[j] {
			continue
		}
		if gap := hostGap(t, m, prompt, want[:j], want[j], got[j]); gap > batchTie {
			t.Fatalf("token %d: device %d, host %d, %.4f apart\ngot  %v\nwant %v", j, got[j], want[j], gap, got, want)
		}
		t.Logf("token %d: a tie, not compared past it", j)
		break
	}
}
