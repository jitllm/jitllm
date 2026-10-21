package model

import (
	"os"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestAFullCardSpillsABlockOntoTheNextOne runs a prompt whose history outgrows
// the first card of two, and demands the first card hand its highest block to
// the second (State.Spills) instead of demoting the model -- with the tokens
// the host produces. With only the first card full the second has room; with
// both full the block comes home (a relocation).
//
// "Full" is measured, not guessed: a first placement, fill-first, shows what
// each card holds, and the run caps each card at that plus a MiB, which the growth past
// the 2048 positions a device reserves (tier initKVCap) cannot fit in. A budget
// picked by hand left whatever room placement's rounding happened to leave,
// and on some budgets that held the growth and nothing spilled.
func TestAFullCardSpillsABlockOntoTheNextOne(t *testing.T) {
	testdevices.Require(t, "cuda:0")
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	var prompt []int32
	for len(prompt) < 2600 { // past the 2048 positions a device reserves (tier initKVCap)
		prompt = append(prompt, m.Vocab.Encode("The capital of France is Paris. Water boils at 100 degrees. ", false)...)
	}
	const gen = 8
	run := func(g *tier.GPU) (*State, []int32) {
		st := m.NewState(len(prompt) + gen + 1)
		if g != nil {
			st.SetDevice(g)
		}
		lg, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		out := []int32{Greedy(lg)}
		for len(out) < gen {
			next, err := st.ForwardGreedy(out[len(out)-1])
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, next)
		}
		return st, out
	}
	host, want := run(nil)
	host.Close()
	// The first card is capped so the second takes blocks too (chosen with
	// JITLLM_SPILL_BUDGET).
	first := spillBudget(380_000_000)
	// open tries cuda:0 beside each second device, capping the ones caps names.
	open := func(caps map[string]uint64) *tier.GPU {
		for _, second := range testdevices.Within([]string{"cuda:1", "vulkan:1"}) {
			spec := ""
			for _, name := range []string{"cuda:0", second} {
				if spec != "" {
					spec += ","
				}
				spec += name
				if b := caps[name]; b > 0 {
					spec += "=" + strconv.FormatUint(b, 10)
				}
			}
			// Fill-first: the proportional plan leaves every card room for the
			// batched prefill's scratch, far more than this history grows by.
			d, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff),
				tier.WithConfig(func(c *tier.Config) { c.FillFirst = true }))
			if err == nil && d != nil && len(d.Placed()) == 2 {
				return d
			}
			if d != nil {
				d.Close()
			}
		}
		return nil
	}
	// What each card holds once placed; the arms cap them there.
	probe := open(map[string]uint64{"cuda:0": first})
	if probe == nil {
		t.Skip("no two devices -- this gate proved nothing")
	}
	pst := m.NewState(len(prompt) + gen + 1)
	if err := pst.SetDevice(probe); err != nil {
		t.Fatal(err)
	}
	held := map[string]uint64{}
	for _, b := range probe.Budgets() {
		held[b.Name] = b.Used + 1<<20
		t.Logf("probe: %s (%s) holds %d of %d, placed %v", b.Name, b.Device, b.Used, b.Limit, probe.Placed())
	}
	pst.Close()
	probe.Close()
	for _, c := range []struct {
		name         string
		caps         map[string]uint64
		spill, reloc bool
	}{{"spill", map[string]uint64{"cuda:0": held["cuda:0"]}, true, false}, {"relocate", held, false, true}} {
		t.Run(c.name, func(t *testing.T) {
			g := open(c.caps)
			if g == nil {
				t.Skip("no two devices -- this gate proved nothing")
			}
			defer g.Close()
			st, got := run(g)
			defer st.Close()
			t.Logf("placed %v, %d spill(s), %d relocation(s), %d demotion(s), %d/%d blocks", g.Placed(),
				st.Spills(), st.Relocations(), st.DeviceDemotions(), st.GPULayers(), m.Cfg.NLayer)
			if st.DeviceDemotions() != 0 || (c.spill && st.Spills() == 0) || (c.reloc && st.Relocations() == 0) {
				t.Fatalf("want spill %v, relocation %v and no demotion: %d spill(s), %d relocation(s), %d demotion(s) (%s)",
					c.spill, c.reloc, st.Spills(), st.Relocations(), st.DeviceDemotions(), g.Err())
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
		})
	}
}

// spillBudget is def, or JITLLM_SPILL_BUDGET to re-derive the choice on other
// hardware.
func spillBudget(def uint64) uint64 {
	if v, err := strconv.ParseUint(os.Getenv("JITLLM_SPILL_BUDGET"), 10, 64); err == nil {
		return v
	}
	return def
}
