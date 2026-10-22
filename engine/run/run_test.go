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
// position's logits held to the engine's host decode, its KV cache f32 as
// the plan's is, and its token to the host's, no kernel writing past its storage, and a
// warm step allocating nothing. Workgroup memory is poisoned, so a kernel
// reading a slot no thread wrote fails every run.
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
	// The host's KV cache is f32 as the plan's pools are: its shipping f16
	// cache alone moves stories15M-q8_0's logits by 6.6e-4.
	m, err := model.Open(path, model.WithJITOptions(nn.WithTune(nn.TuneOff)), model.WithKVType(model.KVF32))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
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
				reg := jit.Default()
				reg.PoisonWorkgroupMemory()
				r, err := run.New(m.Composing(), dev, gpu, capacity, run.WithGuard(4096), run.WithComposer(reg))
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
					nmse := num / den
					if !(nmse < 1e-5) {
						t.Fatalf("position %d: logits NMSE %.3g against the host", pos, nmse)
					}
					worst = max(worst, nmse)
					if int(token) != best {
						t.Fatalf("position %d: token %d, the host's %d", pos, token, best)
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

// TestRunnerStepsRowsAsItDecodes runs a prompt a chunk of rows at a time,
// each row its own position and history, and holds each chunk's last
// row's logits to the plan's a row at a time after the same tokens.
func TestRunnerStepsRowsAsItDecodes(t *testing.T) {
	path := testmodels.Path(*runModel + ".jlm")
	m, err := model.Open(path, model.WithJITOptions(nn.WithTune(nn.TuneOff)), model.WithKVType(model.KVF32))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode("The capital of France is Paris. The capital of Germany is", true)
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
		for _, rows := range []int{4, 7} {
			t.Run(fmt.Sprintf("%s/%d", dev.API(), rows), func(t *testing.T) {
				reg := jit.Default()
				reg.PoisonWorkgroupMemory()
				r, err := run.New(m.Composing(), dev, gpu, 32, run.WithRows(rows), run.WithGuard(4096), run.WithComposer(reg))
				if err != nil {
					t.Fatal(err)
				}
				defer r.Close()
				s, err := r.NewSession()
				if err != nil {
					t.Fatal(err)
				}
				defer s.Close()
				// The reference is the plan a row at a time, which the decode
				// gates hold to the host: a chunk's rows run the same plan's
				// primitives, its own rows wider.
				one, err := run.New(m.Composing(), dev, gpu, 32)
				if err != nil {
					t.Fatal(err)
				}
				defer one.Close()
				os, err := one.NewSession()
				if err != nil {
					t.Fatal(err)
				}
				defer os.Close()
				worst, chunks := 0.0, 0
				for at := 0; at < len(ids); at += rows {
					chunk := ids[at:min(at+rows, len(ids))]
					var want []float32
					for i, tok := range chunk {
						l, _, err := os.Step(tok, at+i)
						if err != nil {
							t.Fatal(err)
						}
						want = append(want[:0], l...)
					}
					got, _, err := s.Rows(chunk, at)
					if err != nil {
						t.Fatal(err)
					}
					var num, den float64
					for i := range want {
						d := float64(got[i]) - float64(want[i])
						num += d * d
						den += float64(want[i]) * float64(want[i])
					}
					if nmse := num / den; !(nmse < 1e-5) {
						t.Fatalf("positions %d to %d: logits NMSE %.3g against a row at a time", at, at+len(chunk)-1, nmse)
					} else {
						worst = max(worst, nmse)
					}
					chunks++
				}
				if err := s.Overrun(); err != nil {
					t.Fatal(err)
				}
				t.Logf("%d tokens in %d steps of up to %d rows on %s: worst logits NMSE %.3g", len(ids), chunks, rows, dev.Name(), worst)
			})
		}
	}
}
