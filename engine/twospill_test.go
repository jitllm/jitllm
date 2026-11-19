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
// go to the second, and prefills both: the second's history has no room on
// the first card beside the first's, so its blocks there spill onto the next
// card as copies of its own while the first's stay. Each instance keeps every
// block on the cards, composes none on the CPU while its tokens run, and
// decodes the host's tokens.
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
	prompts := [2][]int32{
		m.Vocab.Encode("The capital of France is a city that", true),
		m.Vocab.Encode("Once upon a time, in a small village by the sea", true),
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
