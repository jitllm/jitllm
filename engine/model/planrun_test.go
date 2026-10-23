package model

import (
	"fmt"
	"github.com/jitllm/jitllm/engine/graph"
	"path/filepath"
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
// towerCases are the towers the plan composes: how each opens, and a
// dynamic tower's grid (0 for a fixed tower's square).
var towerCases = []struct {
	name   string
	open   func(testing.TB, ...Option) (*Model, *Tower)
	gh, gw int
}{{"smolvlm", openTower, 0, 0}, {"llava", openLlava, 0, 0},
	{"qwen2vl", func(tb testing.TB, _ ...Option) (*Model, *Tower) { m := openQwen2VL(tb); return m, m.Tower() }, 16, 20},
	{"qwen25vl", openQwen25VL, 16, 20},
	{"qwen35", openPairTower("Qwen3.5-0.8B-f16.gguf", "mmproj-Qwen3.5-0.8B-f16.gguf"), 16, 20},
	{"qwen3vl", openPairTower("Qwen3-VL-2B-Instruct-f16.gguf", "mmproj-Qwen3-VL-2B-Instruct-f16.gguf"), 16, 20}}

func TestPlanEncodesPicturesAsTheHost(t *testing.T) {
	for _, tc := range towerCases {
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
	// The plan's rows are each projected row then each deepstack tap's;
	// the host's the projected rows, then each tap's.
	taps := len(c.Deep)
	tokens := len(out) / (c.ProjDim * (1 + taps))
	if taps > 0 {
		lay := make([]float32, len(out))
		for r := range tokens {
			for j := range 1 + taps {
				copy(lay[(r*(1+taps)+j)*c.ProjDim:], out[(j*tokens+r)*c.ProjDim:(j*tokens+r+1)*c.ProjDim])
			}
		}
		out = lay
	}
	want = append(want, slices.Clone(out))
	n := gh * gw
	if c.CLS {
		n++
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
			// The one model's plan: its text parts and its vision segment.
			r, err := run.New(m.Composing(), dev, gpu, n+64, run.WithPictureRows(n))
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
			// Each vision part's output read once its last launch has run,
			// held to the old engine path's, then replaced by it; a rotary
			// table has no stop, and a seam's residual is the block's.
			last, stop := map[int]int{}, map[int]int{}
			head, text := -1, 0
			k := 0
			for pi, pp := range p.Parts {
				k += len(pp.Codes)
				if pp.Part.Segment != jit.SegmentVision {
					text++
					continue
				}
				if pp.Part.Kind != jit.PartTable && pp.Part.Kind != jit.PartSeam {
					last[k-1], stop[pi] = pi, len(stop)
				}
				if pp.Part.Kind == jit.PartHead {
					head = pi
				}
			}
			if text == 0 || head < 0 {
				t.Fatalf("the plan holds %d text parts and no vision head (%d): not the one model's", text, head)
			}
			if len(stop) != len(want) {
				t.Fatalf("%d vision parts, the old path stops at %d", len(stop), len(want))
			}
			outOf := func(pi int) (int, []float32) {
				key := jit.Operand{Kind: jit.OperandValue, Of: pi, Value: p.Parts[pi].Part.Graph.Out[0]}
				at := slices.IndexFunc(p.Storages, func(st jit.Storage) bool { return st.Key == key })
				if at < 0 {
					t.Fatalf("%v: its output has no storage", p.Parts[pi].Part)
				}
				b := s.Storage(at)
				return at, unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
			}
			checked := 0
			forced := func(i int) {
				pi, ok := last[i]
				if !ok {
					return
				}
				at, all := outOf(pi)
				w := want[stop[pi]]
				if len(all) < len(w) {
					t.Fatalf("%v (part %d): its output's storage %d holds %d floats, the old path's %d (%+v)", p.Parts[pi].Part, pi, at, len(all), len(w), p.Storages[at])
				}
				got := all[:len(w)]
				nmse, worst := nmseOf(got, w)
				t.Logf("%v, the old path's input in: NMSE %.3g, worst %.3g", p.Parts[pi].Part, nmse, worst)
				if !(nmse <= 5e-6) {
					t.Errorf("%v: NMSE %.3g against the old path, past 5e-6", p.Parts[pi].Part, nmse)
				}
				checked++
				copy(got, w)
				if err := s.SetStorage(at, unsafe.Slice((*byte)(unsafe.Pointer(&all[0])), 4*len(all))); err != nil {
					t.Fatal(err)
				}
			}
			pic := tw.TowerPicture(px, gh, gw)
			if pic.Rows != n {
				t.Fatalf("the picture's step is %d rows, the old path's %d", pic.Rows, n)
			}
			if err := s.PictureEach(pic, forced); err != nil {
				t.Fatal(err)
			}
			if checked != len(want) {
				t.Fatalf("%d of %d parts checked", checked, len(want))
			}
			if err := s.Picture(pic); err != nil {
				t.Fatal(err)
			}
			_, all := outOf(head)
			got := all[:len(out)]
			nmse, worst := nmseOf(got, out)
			t.Logf("free-running: %d rows of %d, NMSE %.3g against the old path, worst %.3g", tokens, c.ProjDim, nmse, worst)
			if !(nmse <= 5e-2) {
				t.Errorf("free-running: NMSE %.3g against the old path", nmse)
			}
		})
	}
}

// TestPlanPrefillsPicturesAsTheOldPath prefills a prompt with a picture in
// it -- tokens, the picture's span, tokens -- through State.PrefillMixed on
// the model's one plan, against the old engine path's, for the towers whose
// text model turns by a plain rotary; a multi-axis text rotary is refused by
// name.
func TestPlanPrefillsPicturesAsTheOldPath(t *testing.T) {
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
	for _, tc := range towerCases {
		t.Run(tc.name, func(t *testing.T) {
			old, tw := tc.open(t, noTune)
			defer old.Close()
			c := tw.Cfg
			gh, gw := tc.gh, tc.gw
			if gh == 0 {
				gh, gw = c.ImageSz/c.PatchSz, c.ImageSz/c.PatchSz
			}
			px := make([]float32, gh*gw*c.PatchSz*c.PatchSz*3)
			for i := range px {
				px[i] = float32((i*37)%255)/127.5 - 1
			}
			rows := gh * gw / (max(c.Scale, 1) * max(c.Scale, 1))
			pic := &Picture{Pixels: px, GH: gh, GW: gw, Rows: rows}
			before, after := old.Vocab.Encode("Describe the picture:", true), old.Vocab.Encode(" It shows", false)
			spans := []Span{{Tokens: before}, {Picture: pic}, {Tokens: after}}
			n := len(before) + rows + len(after)
			// A row a step, and steps of several rows: a picture's rows
			// spliced into either as the text's own.
			for _, arm := range []struct {
				dev   backend.Device
				chunk int
			}{{devs[0], 0}, {devs[0], 16}, {devs[len(devs)-1], 16}} {
				dev := arm.dev
				t.Run(fmt.Sprintf("%s/chunk%d", dev.API(), arm.chunk), func(t *testing.T) {
					pm, err := Open(old.container.Path, noTune, WithPlanDevice(dev, gpu), WithPlanPictureRows(gh*gw+1), WithPlanChunk(arm.chunk))
					if err != nil {
						t.Fatal(err)
					}
					defer pm.Close()
					ps := pm.NewState(n + 8)
					defer ps.Close()
					got, err := ps.PrefillMixed(spans...)
					if old.Cfg.RopeSections != [4]int{} || old.Cfg.RopeInterleaved {
						if err == nil || !strings.Contains(err.Error(), "multi-axis text rotary") {
							t.Fatalf("a picture on a multi-axis text rotary: %v, want its refusal", err)
						}
						return
					}
					if err != nil {
						t.Fatal(err)
					}
					hs := old.NewState(n + 8)
					defer hs.Close()
					want, err := hs.PrefillMixed(spans...)
					if err != nil {
						t.Fatal(err)
					}
					nmse, worst := nmseOf(got, want)
					t.Logf("%d rows (%d of them the picture's): logits NMSE %.3g against the old path, worst %.3g, top-1 %d and %d",
						n, rows, nmse, worst, Greedy(got), Greedy(want))
					if !(nmse <= 5e-2) || Greedy(got) != Greedy(want) && want[Greedy(got)] < want[Greedy(want)]-0.25 {
						t.Errorf("the prompt's logits stand %.3g from the old path's", nmse)
					}
				})
			}
		})
	}
}

// openQwen25VL is Qwen2.5-VL-3B's pair, converted and opened.
var openQwen25VL = openPairTower("Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf")

// openPairTower opens a text model and its mmproj, converted into one
// container, and hands back the model and its tower.
func openPairTower(text, proj string) func(testing.TB, ...Option) (*Model, *Tower) {
	return func(t testing.TB, opts ...Option) (*Model, *Tower) {
		t.Helper()
		text, proj := testmodels.Path(text), testmodels.Path(proj)
		if !vlmAvailable(text, proj) {
			testmodels.Missing(t, "%s and its mmproj", text)
		}
		m, err := Open(jlmOfPair(t, text, proj), opts...)
		if err != nil {
			t.Fatal(err)
		}
		return m, m.Tower()
	}
}

// TestTowerIsStructural lowers each tower, then again with its projector
// renamed to one no table knows, and requires the same graphs, rotary and
// tensors: the tower's graph is read from what its container states and
// stores, never from its family's name.
func TestTowerIsStructural(t *testing.T) {
	for _, tc := range towerCases {
		t.Run(tc.name, func(t *testing.T) {
			m, tw := tc.open(t, noTune)
			defer m.Close()
			lowered := func() string {
				// A fresh lowering each time: the model's cached one would
				// read the first name twice.
				src, err := m.lowerTower()
				if err != nil {
					t.Fatal(err)
				}
				// Every field of every graph: its values, its ops' kinds,
				// operands, bindings and attributes.
				var b strings.Builder
				full := func(g *graph.Graph) {
					if g != nil {
						fmt.Fprintf(&b, "%+v\n", *g)
					}
				}
				full(src.EmbedGraph())
				full(src.HeadGraph())
				for li := range src.Blocks() + 1 {
					full(src.BlockGraph(li))
					full(src.SeamGraph(li))
				}
				fmt.Fprintf(&b, "%+v %+v", src.Rotaries(), src.Ropes())
				return b.String()
			}
			want := lowered()
			v := m.container.Vision()
			p, k, name := v.Projector, tw.Cfg.Kind, tw.Cfg.Projector
			v.Projector, tw.Cfg.Kind, tw.Cfg.Projector = 0x7fff, 0x7fff, "renamed"
			defer func() { v.Projector, tw.Cfg.Kind, tw.Cfg.Projector = p, k, name }()
			if got := lowered(); got != want {
				t.Errorf("the tower lowers otherwise under another name")
			}
		})
	}
}

// TestPlanRefusesTheOldEnginePath opens models on the composer's plan and
// requires every entry the plan does not compose to refuse by name rather
// than run the old engine path: a token step past Forward's (greedy,
// sampled, from an embedding), a speculative step, an embedding, a decision
// readout the plan does not compose, a picture.
func TestPlanRefusesTheOldEnginePath(t *testing.T) {
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Skip("no device")
	}
	dev, gpu := devs[0], jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	refused := func(t *testing.T, what string, err error) {
		t.Helper()
		if err == nil || !strings.Contains(err.Error(), "the composer's plan") {
			t.Errorf("%s on the plan: %v, want a refusal naming the plan", what, err)
		}
	}
	t.Run("token steps", func(t *testing.T) {
		m, err := Open(testmodels.Path("synth-ernie45.jlm"), noTune, WithPlanDevice(dev, gpu))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		s := m.NewState(64)
		defer s.Close()
		// A greedy token is the plan's own sample step's: the argmax of the
		// logits the plan returns, the lowest index on a tie.
		logits, err := s.Forward(1)
		if err != nil {
			t.Fatalf("Forward, which the plan runs: %v", err)
		}
		best := 0
		for i, v := range logits {
			if v > logits[best] {
				best = i
			}
		}
		if s.planTok != int32(best) {
			t.Errorf("the plan sampled %d, the argmax of its logits is %d", s.planTok, best)
		}
		if _, err := s.ForwardGreedy(int32(best)); err != nil {
			t.Fatalf("ForwardGreedy, whose token the plan samples: %v", err)
		}
		_, err = s.ForwardSample(1, &Sampler{Temp: 0.7})
		refused(t, "ForwardSample at a temperature", err)
		_, err = s.ForwardEmbd(make([]float32, m.Cfg.NEmbd))
		refused(t, "ForwardEmbd", err)
		_, err = s.Park()
		refused(t, "Park", err)
		_, err = s.Resume()
		refused(t, "Resume", err)
		refused(t, "ShareKV", s.ShareKV(nil))
		refused(t, "SetKVBudget", s.SetKVBudget(1<<20))
		b := m.NewBatch(2, 64)
		defer b.Close()
		_, err = b.Forward(1)
		refused(t, "Forward on a batch of two sequences", err)
	})
	t.Run("embedding", func(t *testing.T) {
		m, err := Open(jlmOf(t, testmodels.Path(filepath.Join("embed", "all-minilm-l6.gguf"))), noTune, WithPlanDevice(dev, gpu))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		_, err = m.NewEmbedder()
		refused(t, "NewEmbedder", err)
	})
	t.Run("decision", func(t *testing.T) {
		m := openDecision(t, "vllm-d3/vllm-d3-edge-Q8_0.gguf", jlm.DecisionVllmD3, noTune, WithPlanDevice(dev, gpu))
		defer m.Close()
		_, err := m.NewDecider(0)
		refused(t, "a vllm-d3 Decider", err)
	})
	t.Run("picture", func(t *testing.T) {
		m, _ := openTower(t, noTune, WithPlanDevice(dev, gpu))
		defer m.Close()
		s := m.NewState(64)
		defer s.Close()
		_, err := s.Vision()
		refused(t, "Vision", err)
	})
}
