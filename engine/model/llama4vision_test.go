package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// Llama 4's vision against transformers' own Llama4ForConditionalGeneration
// (scripts/llama4visiongold.py): the processor's tiles, the tower and
// adapter's rows, the prompt's layout and the whole model from the pixels.

type l4vGolden struct {
	Pre    []int32 `json:"pre"`
	Rows   []int32 `json:"rows"`
	Post   []int32 `json:"post"`
	Cont   []int32 `json:"cont"`
	Aspect [2]int  `json:"aspect"`
	Image  struct {
		H      int       `json:"h"`
		W      int       `json:"w"`
		RGB    []byte    `json:"rgb"`
		Tiles  int       `json:"tiles"`
		PX     []float32 `json:"px"`
		Embd   []float32 `json:"embd"`
		EmbdQ8 []float32 `json:"embd_q8"`
		EmbdA8 []float32 `json:"embd_a8"`
	} `json:"image"`
	Logits   []pixLogits `json:"logits"`
	LogitsA8 []pixLogits `json:"logits_a8"`
}

const l4vFixture = "synth-llama4v"

func openLlama4V(t testing.TB, opts ...Option) (*Model, *l4vGolden) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "llama4v", l4vFixture+".json"))
	if err != nil {
		t.Fatal(err)
	}
	g := &l4vGolden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	text, proj := testmodels.Path(l4vFixture+".gguf"), testmodels.Path("mmproj-"+l4vFixture+".gguf")
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- run scripts/llama4visiongold.py (RULE 11)", text)
	}
	m, err := Open(jlmOfPair(t, text, proj), append([]Option{noTune, WithKVF16(false)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return m, g
}

// l4vPieces is the golden's picture cut into the engine's tiles.
func l4vPieces(t *testing.T, tw *Tower, g *l4vGolden) (rows, cols int, ps []*Picture) {
	t.Helper()
	rows, cols, ps, err := tw.Llama4Pieces(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
	if err != nil {
		t.Fatal(err)
	}
	return rows, cols, ps
}

// l4vEncode runs every tile of ps through a fresh vision State and returns
// their rows back to back.
func l4vEncode(t *testing.T, tw *Tower, ps []*Picture) []float32 {
	t.Helper()
	s := tw.testState()
	defer s.Close()
	var out []float32
	for _, p := range ps {
		e, err := s.Encode(p.Pixels)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, e...)
	}
	return out
}

// TestLlama4PreprocessMatchesTheProcessor holds the tiles to transformers'
// Llama4ImageProcessor: the canvas, the resize onto it, the padding, the
// order, the global tile and the bfloat16 normalisation, to the bit. The
// violation is the normalisation in float32.
func TestLlama4PreprocessMatchesTheProcessor(t *testing.T) {
	m, g := openLlama4V(t)
	defer m.Close()
	tw := m.Tower()
	rows, cols, ps := l4vPieces(t, tw, g)
	if [2]int{rows, cols} != g.Aspect || len(ps) != g.Image.Tiles {
		t.Fatalf("a %dx%d canvas of %d tiles, the processor's %v of %d", rows, cols, len(ps), g.Aspect, g.Image.Tiles)
	}
	var got []float32
	for _, p := range ps {
		got = append(got, p.Pixels...)
	}
	n, worst := pixelDiff(got, g.Image.PX)
	t.Logf("%d tiles (%dx%d and a global one): %d of %d values differ, max|d| %.2e", len(ps), rows, cols,
		n, len(got), worst)
	// The golden is written to eight decimals; a bfloat16 value is exact in
	// fewer, so a value that differs at all is a different value.
	if worst > 1e-8 {
		t.Fatalf("the tiles are not the processor's: %d values differ, max|d| %.2e", n, worst)
	}
	// The violation: the normalisation in float32, which every other
	// processor here runs.
	p := toRGB8(goldenImage(g.Image.W, g.Image.H, g.Image.RGB))
	res := p.resize(tw.Cfg.ImageSz, tw.Cfg.ImageSz, aaBilinear)
	f32 := make([]float32, len(ps[len(ps)-1].Pixels))
	res.normalizeInto(f32, tw.Cfg.Mean, tw.Cfg.Std)
	nb, _ := pixelDiff(f32, ps[len(ps)-1].Pixels)
	if nb == 0 {
		t.Fatal("a float32 normalisation gives the same global tile: the gate cannot see bfloat16")
	}
	t.Logf("float32 normalisation (the violation): %d of %d values of the global tile differ", nb, len(f32))
}

// TestLlama4TowerMatchesTransformers holds every tile's rows to
// transformers' tower, adapter and projector at the engine's arithmetic, and
// each of the family's pieces taken out must move it past the bound: the
// class token last, the rotary's positions from one, the column before the
// row, the shuffle's order.
func TestLlama4TowerMatchesTransformers(t *testing.T) {
	m, g := openLlama4V(t)
	defer m.Close()
	tw := m.Tower()
	_, _, ps := l4vPieces(t, tw, g)
	got := l4vEncode(t, tw, ps)
	if len(got) != len(g.Image.EmbdA8) {
		t.Fatalf("%d values, transformers' %d", len(got), len(g.Image.EmbdA8))
	}
	clean, worst := nmse32(g.Image.EmbdA8, got)
	f32, _ := nmse32(g.Image.Embd, got)
	q8, _ := nmse32(g.Image.Embd, g.Image.EmbdA8)
	t.Logf("%d tiles, %d rows of %d: NMSE %.3e, max|d| %.4f at the engine's arithmetic (%.3e against "+
		"the f32 tower, where the arithmetic alone moves transformers by %.3e)",
		len(ps), len(got)/tw.Cfg.ProjDim, tw.Cfg.ProjDim, clean, worst, f32, q8)
	if math.IsNaN(clean) || clean > towerBound {
		t.Fatalf("the tower's rows are not transformers': NMSE %.3e (bound %.0e)", clean, towerBound)
	}
	// Positions from zero move every patch by one on both axes, which a
	// rotary's relative scores cannot see: only the class token, which does
	// not turn, tells the two apart, so that violation is held to ten times
	// the clean arm rather than to the bound.
	for _, v := range []struct {
		name string
		set  func() func()
		weak bool
	}{
		{"the class token first", func() func() {
			tw.Cfg.CLSLast = false
			return func() { tw.Cfg.CLSLast = true }
		}, false},
		{"rotary positions from zero", func() func() {
			tw.Cfg.fault = faultLlama4FromZero
			return func() { tw.Cfg.fault = towerFaultNone }
		}, true},
		{"the row before the column", func() func() {
			tw.Cfg.fault = faultLlama4RowFirst
			return func() { tw.Cfg.fault = towerFaultNone }
		}, false},
		{"the shuffle transposed", func() func() {
			tw.Cfg.fault = towerFaultShuffleSwap
			return func() { tw.Cfg.fault = towerFaultNone }
		}, false},
	} {
		undo := v.set()
		bad := l4vEncode(t, tw, ps)
		undo()
		nb, _ := nmse32(g.Image.EmbdA8, bad)
		t.Logf("violation %q: NMSE %.3e", v.name, nb)
		if !(nb > 10*towerBound) && !(v.weak && nb > 10*clean) {
			t.Errorf("%s reads %.3e against a clean %.3e: the gate cannot see it", v.name, nb, clean)
		}
	}
}

// l4vSpans is the golden's prompt with the picture laid out by Llama4Spans,
// after checking the layout is the processor's token for token.
func l4vSpans(t *testing.T, m *Model, g *l4vGolden, ps []*Picture) []Span {
	t.Helper()
	img, err := m.Llama4Spans(g.Aspect[0], g.Aspect[1], ps)
	if err != nil {
		t.Fatal(err)
	}
	const patch = 200092
	var ids []int32
	for _, sp := range img {
		if sp.Picture != nil {
			for range sp.Picture.Rows {
				ids = append(ids, patch)
			}
			continue
		}
		ids = append(ids, sp.Tokens...)
	}
	if len(ids) != len(g.Rows) {
		t.Fatalf("the picture lays out %d positions, the processor %d", len(ids), len(g.Rows))
	}
	for i := range ids {
		if ids[i] != g.Rows[i] {
			t.Fatalf("position %d of the picture is %d, the processor's %d", i, ids[i], g.Rows[i])
		}
	}
	return append(append([]Span{{Tokens: g.Pre}}, img...), Span{Tokens: g.Post})
}

// TestLlama4MatchesTransformersEndToEnd is the whole model from the pixels:
// the tiles spans of the prompt, encoded by the prefill on the State's own
// vision segment, against transformers' logits with its tower at the
// engine's arithmetic (the f32 reference is reported).
func TestLlama4MatchesTransformersEndToEnd(t *testing.T) {
	m, g := openLlama4V(t)
	defer m.Close()
	_, _, ps := l4vPieces(t, m.Tower(), g)
	run := func(spans []Span) [][]float32 {
		st := m.NewState(SpanPositions(spans, m.Cfg.NEmbd) + len(g.Cont) + 1)
		defer st.Close()
		lg, err := st.PrefillMixed(spans...)
		if err != nil {
			t.Fatal(err)
		}
		out := [][]float32{append([]float32(nil), lg...)}
		for _, id := range g.Cont[:len(g.Cont)-1] {
			if lg, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	spans := l4vSpans(t, m, g, ps)
	got := run(spans)
	w, flips := pixWorst(g.LogitsA8, got)
	wf, ff := pixWorst(g.Logits, got)
	t.Logf("from the pixels: worst logit NMSE %.3e, %d argmax flips (%.3e and %d against the f32 tower)",
		w, flips, wf, ff)
	if math.IsNaN(w) || w > 2e-2 || flips > 0 {
		t.Fatalf("end to end: worst NMSE %.3e, %d argmax flips", w, flips)
	}
	// The violation: the global tile first, as a processor that put the
	// overview ahead (MiniCPM-V's) would lay it out.
	sw := append([]*Picture{ps[len(ps)-1]}, ps[:len(ps)-1]...)
	bad := run(l4vSpans(t, m, g, sw))
	nb, _ := pixWorst(g.LogitsA8, bad)
	t.Logf("the global tile first (the violation): %.3e", nb)
	if !(nb > 10*w) {
		t.Fatalf("the tiles' order reads %.3e against %.3e: the gate cannot see it", nb, w)
	}
}

// TestLlama4OnDeviceMatchesHost places every tower block on each device
// backend present, holds the rows to the host's, and brings them home.
func TestLlama4OnDeviceMatchesHost(t *testing.T) {
	m, g := openLlama4V(t)
	defer m.Close()
	tw := m.Tower()
	_, _, ps := l4vPieces(t, tw, g)
	want := l4vEncode(t, tw, ps)
	ran := 0
	for _, spec := range stepDevices() {
		gpu, err := tier.OpenWith(tier.WithDevices(spec))
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			s := tw.testState()
			defer s.Close()
			s.SetDevice(gpu)
			if n := s.GPUBlocks(); n != tw.Cfg.NLayer {
				t.Fatalf("%s took %d of %d tower blocks (%s)", spec, n, tw.Cfg.NLayer, gpu.Err())
			}
			var got []float32
			for _, p := range ps {
				e, err := s.Encode(p.Pixels)
				if err != nil {
					t.Fatal(err)
				}
				got = append(got, e...)
			}
			nm, worst := nmseOf(got, want)
			t.Logf("%s, all %d blocks: NMSE %.3e, max|d| %.4f against the host", spec, tw.Cfg.NLayer, nm, worst)
			if math.IsNaN(nm) || math.IsInf(nm, 0) || nm > 1e-3 {
				t.Fatalf("%s: the tower on the device reads NMSE %.3e against the host", spec, nm)
			}
			if err := s.SetDevice(nil); err != nil {
				t.Fatal(err)
			}
			if nb, _ := nmseOf(l4vEncodeOn(t, s, ps), want); nb != 0 {
				t.Fatalf("%s: home again, the host reads NMSE %.3e against itself", spec, nb)
			}
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}

func l4vEncodeOn(t *testing.T, s *State, ps []*Picture) []float32 {
	t.Helper()
	var out []float32
	for _, p := range ps {
		e, err := s.Encode(p.Pixels)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, e...)
	}
	return out
}
