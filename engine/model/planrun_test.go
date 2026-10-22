package model

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/run"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
)

// TestPlanDecodesAsTheHost opens a model on the host and WithPlanDevice on
// every device a backend opens, and holds the planned model's Prefill of a
// prompt -- two chunks of rows and a part of one, or on a recurrent model a
// step a token -- and Forward of the host's choices after it to the host's
// logits in the coherence band and its tokens; a batch entry point on the
// planned model refuses, and a warm Forward allocates nothing.
func TestPlanDecodesAsTheHost(t *testing.T) {
	for _, c := range []struct {
		name   string
		chunks bool
	}{{"synth-ernie45", true}, {"synth-lfm2", false}} {
		t.Run(c.name, func(t *testing.T) { planDecodes(t, testmodels.Path(c.name+".jlm"), c.chunks) })
	}
}

// planDecodes is TestPlanDecodesAsTheHost on the model at path, its prompt
// in chunks of rows where chunks.
func planDecodes(t *testing.T, path string, chunks bool) {
	host, err := Open(path, noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	var ids []int32
	const chunk = 32
	for k := 1; len(ids) <= 2*chunk+chunk/2; k++ {
		ids = host.Vocab.Encode(strings.Repeat("The capital of France is Paris. ", k)+"Hello there", true)
	}
	if len(ids) >= 3*chunk {
		t.Fatalf("a prompt of %d tokens, want two chunks of %d and a part", len(ids), chunk)
	}
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Skip("no device")
	}
	gpu := jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	for _, dev := range devs {
		t.Run(dev.API(), func(t *testing.T) {
			m, err := Open(path, noTune, WithPlanDevice(dev, gpu), WithPlanChunk(chunk))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			const more = 6
			hs, ps := host.NewState(len(ids)+more+8), m.NewState(len(ids)+more+32)
			defer hs.Close()
			defer ps.Close()
			want, err := hs.Prefill(ids)
			if err != nil {
				t.Fatal(err)
			}
			got, err := ps.Prefill(ids)
			if err != nil {
				t.Fatal(err)
			}
			if (ps.planChunk != nil) != chunks {
				t.Fatalf("the prompt ran in chunks: %v, want %v", ps.planChunk != nil, chunks)
			}
			worst := 0.0
			for i := range more {
				nmse, _ := nmseOf(got, want)
				best := Greedy(want)
				if !(nmse < 1e-5) {
					t.Fatalf("position %d: logits NMSE %.3g against the host", len(ids)+i-1, nmse)
				}
				if g := Greedy(got); g != best {
					t.Fatalf("position %d: token %d, the host's %d", len(ids)+i-1, g, best)
				}
				worst = max(worst, nmse)
				if want, err = hs.Forward(best); err != nil {
					t.Fatal(err)
				}
				if got, err = ps.Forward(best); err != nil {
					t.Fatal(err)
				}
			}
			if ps.Pos() != hs.Pos() {
				t.Errorf("the planned State is at %d, the host's at %d", ps.Pos(), hs.Pos())
			}
			if _, err := ps.PrefillCached([]int32{1, 2}); err == nil || !strings.Contains(err.Error(), "composer's plan") {
				t.Errorf("PrefillCached on the plan: %v, want a refusal", err)
			}
			allocs := testing.AllocsPerRun(8, func() {
				if _, err := ps.Forward(3); err != nil {
					t.Fatal(err)
				}
			})
			// The race detector's instrumentation allocates: the count is the
			// build's without it.
			if allocs != 0 && !raceEnabled {
				t.Errorf("a warm Forward on the plan allocates %.1f objects", allocs)
			}
			t.Logf("%s: %d positions, worst logits NMSE %.3g, %.0f allocations a token", dev.Name(), len(ids)+more, worst, allocs)
		})
	}
}

// TestPlanDecidesAsTheHost composes a decision model with Laya's head --
// ModernBERT's trunk, or d1-omni's LFM2 encoder with its centred short
// convolutions, then the head's blocks after the prologue (the trunk's
// final norm and the question type's row) and the scorer at the request's
// markers -- and holds its scores for questions of each type to the host's
// encode and layaScores on every device a backend opens.
func TestPlanDecidesAsTheHost(t *testing.T) {
	for _, c := range []struct {
		name, gguf string
		kind       jlm.DecisionKind
		bound      float64
	}{{"fixture", "laya/fixture/laya-fixture.gguf", jlm.DecisionLaya, 1e-9}, {"laya", "laya/gguf/Laya-BF16.gguf", jlm.DecisionLaya, 1e-6},
		{"d1-omni", d1OmniModel, jlm.DecisionD1Omni, 1e-6}} {
		t.Run(c.name, func(t *testing.T) {
			m := openDecision(t, c.gguf, c.kind, noTune)
			defer m.Close()
			// Past the local layers' window, so the window bites.
			var ids []int32
			for k := 1; len(ids) <= m.Cfg.SWAWindow+8; k++ {
				ids = m.Vocab.Encode(strings.Repeat("Is the capital of France Paris? ", k)+"Answer with one of the options below.", true)
			}
			dc, err := m.NewDecider(len(ids) + 8)
			if err != nil {
				t.Fatal(err)
			}
			defer dc.Close()
			e := dc.emb
			markers := []int32{1, int32(len(ids) / 2), int32(len(ids) - 2)}
			rows := []int{int(markers[0]), int(markers[1]), int(markers[2])}
			devs := backend.Open()
			defer func() {
				for _, d := range devs {
					d.Close()
				}
			}()
			if len(devs) == 0 {
				t.Skip("no device")
			}
			gpu := jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}
			for _, dev := range devs {
				r, err := run.New(m.Composing(), dev, gpu, len(ids), run.WithRows(len(ids)))
				if err != nil {
					t.Fatal(err)
				}
				s, err := r.NewSession()
				if err != nil {
					t.Fatal(err)
				}
				for qt := range 3 {
					if err := hostEncode(e, ids); err != nil {
						t.Fatal(err)
					}
					want := make([]float32, len(markers))
					if err := e.layaScores(want, len(ids), jlm.QuestionType(qt), rows); err != nil {
						t.Fatal(err)
					}
					got, err := s.Decide(ids, qt, markers)
					if err != nil {
						t.Fatal(err)
					}
					nmse, _ := nmseOf(got, want)
					t.Logf("%s, question type %d: scores %v, the host's %v, NMSE %.3g", dev.API(), qt, got, want, nmse)
					if !(nmse <= c.bound) {
						t.Errorf("%s, question type %d: NMSE %.3g against the host", dev.API(), qt, nmse)
					}
				}
				s.Close()
				r.Close()
			}
		})
	}
}
