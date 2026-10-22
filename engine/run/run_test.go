package run_test

import (
	"flag"
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/run"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
)

var (
	gpu      = jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	runModel = flag.String("run-model", "synth-ernie45", "the model TestRunnerDecodesAsTheHost runs")
)

// TestRunnerDecodesAsTheHost runs a model's decode (synth-ernie45, or
// -run-model) through the runner
// on every device a backend opens, at a capacity inside the attention's
// first split count and one past it: a prompt, then the host's choices, each
// position's logits held to the engine's host decode in the coherence band
// and its token to the host's, no kernel writing past its storage, and a
// warm step allocating nothing.
func TestRunnerDecodesAsTheHost(t *testing.T) {
	path := testmodels.Path(*runModel + ".jlm")
	f, err := jlm.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := f.LoadAll(); err != nil {
		t.Fatal(err)
	}
	m, err := model.Open(path, model.WithJITOptions(nn.WithTune(nn.TuneOff)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	c := f.Config()
	rope := nn.Rope{NRot: int(c.NRot), Base: float64(c.RopeBase)}
	ids := m.Vocab.Encode("Hello there", true)
	const more = 6
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Skip("no device")
	}
	for _, dev := range devs {
		for _, capacity := range []int{16, 300} {
			t.Run(fmt.Sprintf("%s/%d", dev.API(), capacity), func(t *testing.T) {
				r, err := run.New(f, dev, gpu, capacity, rope, run.WithGuard(4096))
				if err != nil {
					t.Fatal(err)
				}
				defer r.Close()
				s, err := r.NewSession()
				if err != nil {
					t.Fatal(err)
				}
				defer s.Close()
				st := m.NewState(len(ids) + more + 8)
				defer st.Close()
				worst, tok, n := 0.0, ids[0], len(ids)+more
				for pos := range n {
					want, err := st.Forward(tok)
					if err != nil {
						t.Fatal(err)
					}
					got, token, err := s.Step(tok, pos)
					if err != nil {
						t.Fatal(err)
					}
					var num, den float64
					best := 0
					for i := range want {
						d := float64(got[i]) - float64(want[i])
						num += d * d
						den += float64(want[i]) * float64(want[i])
						if want[i] > want[best] {
							best = i
						}
					}
					// The coherence band (RULE 11c): an int8 activation rounds
					// differently on the host, which moves a quantized model's
					// logits by up to 7e-4 on main's own device path too.
					nmse := num / den
					if !(nmse < 1e-3) {
						t.Fatalf("position %d: logits NMSE %.3g against the host", pos, nmse)
					}
					worst = max(worst, nmse)
					// A different token is a tie only where the host itself
					// nearly chose it.
					if int(token) != best {
						if margin := want[best] - want[token]; margin >= 0.05 {
							t.Fatalf("position %d: token %d, the host's %d by a margin of %.3f", pos, token, best, margin)
						}
						t.Logf("position %d: token %d, the host's %d, a tie of %.4f", pos, token, best, want[best]-want[token])
					}
					tok = int32(best)
					if pos+1 < len(ids) {
						tok = ids[pos+1]
					}
				}
				if err := s.Overrun(); err != nil {
					t.Fatal(err)
				}
				allocs := testing.AllocsPerRun(16, func() {
					if _, _, err := s.Step(tok, n); err != nil {
						t.Fatal(err)
					}
				})
				// The race detector's instrumentation allocates: the count is
				// the build's without it.
				if allocs != 0 && !raceEnabled {
					t.Errorf("a warm step allocates %.1f objects", allocs)
				}
				t.Logf("%d positions on %s, one plan for %d keys: worst logits NMSE %.3g, %.0f allocations a step", n, dev.Name(), capacity, worst, allocs)
			})
		}
	}
}
