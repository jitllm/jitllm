package model

import (
	"slices"
	"strings"
	"testing"
	"unsafe"

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
			// A step the request fills, and one it leaves rows past: a row
			// past the live ones repeats the last, and an op that reads its
			// neighbours (d1-omni's convolution) must too.
			for _, dev := range devs {
				for _, step := range []int{len(ids), len(ids) + 40} {
					r, err := run.New(m.Composing(), dev, gpu, step, run.WithRows(step))
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
						t.Logf("%s, %d rows, question type %d: scores %v, the host's %v, NMSE %.3g", dev.API(), step, qt, got, want, nmse)
						if !(nmse <= c.bound) {
							t.Errorf("%s, %d rows, question type %d: NMSE %.3g against the host", dev.API(), step, qt, nmse)
						}
					}
					s.Close()
					r.Close()
				}
			}
		})
	}
}

// TestLayaPlanMatchesReference is TestLayaMatchesReference on the
// composer's plan: a model opened with WithPlanDevice answers the
// reference's requests through the Decider -- its template, its sequences,
// its markers -- at the host's tolerance, on every device a backend opens.
func TestLayaPlanMatchesReference(t *testing.T) {
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
	for _, c := range layaCases {
		t.Run(c.name, func(t *testing.T) {
			for _, dev := range devs {
				t.Run(dev.API(), func(t *testing.T) {
					m := openDecision(t, c.gguf, jlm.DecisionLaya, noTune, WithPlanDevice(dev, gpu))
					defer m.Close()
					layaMatchesReference(t, m, c.gold, c.tol, layaArm{name: "plan " + dev.API(), plan: true})
				})
			}
		})
	}
}

// TestD1OmniPlanMatchesReference is TestD1OmniMatchesReference with the
// readout on the composer's plan, on every device a backend opens.
func TestD1OmniPlanMatchesReference(t *testing.T) {
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
			m := openDecision(t, d1OmniModel, jlm.DecisionD1Omni, noTune, WithPlanDevice(dev, gpu))
			defer m.Close()
			d1OmniMatchesReference(t, m)
		})
	}
}

// TestPlanEncodesPicturesAsTheHost composes each tower -- SmolVLM's (the
// patches gathered from the picture through the patch embedding, its
// positions' rows, the blocks, the post-norm, the regroup and the
// projector) and llava-phi-3's (a class row before the patches, a pre-norm,
// quick-GELU blocks, the class row dropped, a two-matrix projector) -- and
// holds it to the host's Encode on every device a backend opens. Both sides
// quantize the Q8_0 matrices' activations to int8, whose roundings compound
// through the blocks (main's device gate stops after one block for it), so
// each part is held teacher-forced, the host's input in, to 5e-6 (a tenth of
// main's one-block bound; the worst clean part reads 1.6e-6, and Qwen2-VL's
// rotary with its axes swapped 3.1e-5 at block 0), and the free-running
// picture to 5e-2 (32 blocks of Qwen2-VL read 1.6e-2, the violations 0.12
// and past).
func TestPlanEncodesPicturesAsTheHost(t *testing.T) {
	for _, tc := range []struct {
		name string
		open func(testing.TB, ...Option) (*Model, *Tower)
		gh, gw int // a dynamic tower's grid; 0 for a fixed tower's square
	}{{"smolvlm", openTower, 0, 0}, {"llava", openLlava, 0, 0},
		{"qwen2vl", func(tb testing.TB, _ ...Option) (*Model, *Tower) { m := openQwen2VL(tb); return m, m.Tower() }, 16, 20}} {
		t.Run(tc.name, func(t *testing.T) {
			m, tw := tc.open(t, noTune)
			defer m.Close()
			planEncodesAsTheHost(t, m, tw, tc.gh, tc.gw)
		})
	}
}

// planEncodesAsTheHost is TestPlanEncodesPicturesAsTheHost on one tower.
func planEncodesAsTheHost(t *testing.T, m *Model, tw *Tower, gh, gw int) {
	c := tw.Cfg
	if gh == 0 {
		gh, gw = c.ImageSz/c.PatchSz, c.ImageSz/c.PatchSz
	}
	px := make([]float32, gh*gw*c.PatchSz*c.PatchSz*3)
	for i := range px {
		px[i] = float32((i*37)%255)/127.5 - 1
	}
	host := tw.testState()
	var want [][]float32
	host.vis.afterBlock = func(_ int, x []float32) { want = append(want, slices.Clone(x)) }
	out, err := host.EncodeGrid(px, gh, gw)
	host.Close()
	if err != nil {
		t.Fatal(err)
	}
	want = append(want, slices.Clone(out))
	n := gh * gw
	if c.CLS {
		n++
	}
	tokens := len(out) / c.ProjDim
	src := m.TowerComposing()
	if err := src.(towerComposing).towerRefusal(); err != nil {
		t.Fatal(err)
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
			r, err := run.New(src, dev, gpu, n, run.WithRows(n))
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			s, err := r.NewSession()
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			p := r.Plan()
			// Each part's output read once its last launch has run, held to
			// the host's, then replaced by it; a rotary table has no host
			// stop.
			last, stop := map[int]int{}, map[int]int{}
			k := 0
			for pi, pp := range p.Parts {
				k += len(pp.Codes)
				if pp.Part.Kind != jit.PartTable {
					last[k-1], stop[pi] = pi, len(stop)
				}
			}
			if len(stop) != len(want) {
				t.Fatalf("%d parts, the host stops at %d", len(stop), len(want))
			}
			checked := 0
			forced := func(i int) {
				pi, ok := last[i]
				if !ok {
					return
				}
				key := jit.Operand{Kind: jit.OperandValue, Of: pi, Value: p.Parts[pi].Part.Graph.Out[0]}
				at := slices.IndexFunc(p.Storages, func(st jit.Storage) bool { return st.Key == key })
				if at < 0 {
					t.Errorf("%v: its output has no storage", p.Parts[pi].Part)
					return
				}
				b, w := s.Storage(at), want[stop[pi]]
				got := unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)[:len(w)]
				nmse, worst := nmseOf(got, w)
				t.Logf("%v, the host's input in: NMSE %.3g, worst %.3g", p.Parts[pi].Part, nmse, worst)
				if !(nmse <= 5e-6) {
					t.Errorf("%v: NMSE %.3g against the host, past 5e-6", p.Parts[pi].Part, nmse)
				}
				checked++
				copy(got, w)
				if err := s.SetStorage(at, b); err != nil {
					t.Fatal(err)
				}
			}
			if err := s.PictureEach(px, gw, n, forced); err != nil {
				t.Fatal(err)
			}
			if checked != len(want) {
				t.Fatalf("%d of %d parts checked", checked, len(want))
			}
			got, err := s.Picture(px, gw, n, tokens)
			if err != nil {
				t.Fatal(err)
			}
			nmse, worst := nmseOf(got, out)
			t.Logf("free-running: %d rows of %d, NMSE %.3g against the host, worst %.3g", tokens, c.ProjDim, nmse, worst)
			if !(nmse <= 5e-2) {
				t.Errorf("free-running: NMSE %.3g against the host", nmse)
			}
		})
	}
}
