package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// LFM2-VL against transformers' own Lfm2VlForConditionalGeneration at toy
// size, every parameter and buffer randomised, both GGUFs written by
// llama.cpp's convert_hf_to_gguf.py (scripts/lfm2visiongold.py --fixture).
// Its projector carries the optional LayerNorm d1's lacks, and its picture
// is cut into 3x2 tiles and a thumbnail.

const lfm2Fixture = "synth-lfm2vl"

type lfm2FixGold struct {
	W, H, Rows, Cols int
	Pieces           [][2]int
	IDs              []int32 `json:"ids"`
	ImageToken       int32   `json:"image_token"`
	Cont             []int32 `json:"cont"`
	Logits           []pixLogits
	LogitsA8         []pixLogits `json:"logits_a8"`
}

func openLFM2Fixture(t testing.TB, opts ...Option) (*Model, *lfm2FixGold, string) {
	t.Helper()
	text, proj := testmodels.Path(lfm2Fixture+".gguf"), testmodels.Path("mmproj-"+lfm2Fixture+".gguf")
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj -- run scripts/lfm2visiongold.py --fixture (RULE 11)", text)
	}
	base := testmodels.Path(filepath.Join(lfm2Fixture+"-gold", "picture"))
	raw, err := os.ReadFile(base + ".json")
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v -- run scripts/lfm2visiongold.py --fixture (RULE 11)", err)
	}
	g := &lfm2FixGold{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	m, err := Open(jlmOfPair(t, text, proj), append([]Option{noTune, WithKVF16(false)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	if m.Tower() == nil || m.Tower().projNorm == nil {
		m.Close()
		t.Fatal("the fixture's container carries no tower or no projector LayerNorm, so this gate proves nothing")
	}
	return m, g, base
}

// TestLFM2FixtureMatchesTransformers holds the fixture's pieces, tower and
// projector to transformers at the engine's arithmetic, then the whole model
// from the pixels to the logits; each graph feature removed must fail it.
func TestLFM2FixtureMatchesTransformers(t *testing.T) {
	m, g, base := openLFM2Fixture(t)
	defer m.Close()
	tw := m.Tower()
	c := tw.Cfg
	path := base + ".png"
	s := tw.testState()
	px, vit, emb, n := lfm2Encode(t, tw, s, path)
	s.Close()
	if n != len(g.Pieces) || g.Cols*g.Rows < 2 {
		t.Fatalf("%d pieces for a %dx%d grid, the processor's %d", n, g.Cols, g.Rows, len(g.Pieces))
	}
	nd, _ := pixelDiff(px, readF32(t, base+".px"))
	if nd != 0 {
		t.Fatalf("%d pixel values differ from the processor's", nd)
	}
	wantVit8, wantEmb8 := readF32(t, base+".vit_a8"), readF32(t, base+".emb_a8")
	nv, _ := nmse32(wantVit8, vit)
	ne, _ := nmse32(wantEmb8, emb)
	nf, _ := nmse32(readF32(t, base+".emb"), emb)
	t.Logf("%dx%d tiles and a thumbnail: tower NMSE %.3e, projector NMSE %.3e at the engine's arithmetic "+
		"(%.3e against the f32 reference)", g.Cols, g.Rows, nv, ne, nf)
	if math.IsNaN(ne) || nv > towerBound || ne > towerBound {
		t.Fatalf("tower %.3e, projector %.3e (bound %.0e)", nv, ne, towerBound)
	}
	proj := func() []float32 {
		ps := tw.testState()
		defer ps.Close()
		var out []float32
		at := 0
		for _, p := range g.Pieces {
			rows := p[0] * p[1] * c.NEmbd
			ps.vis.gh, ps.vis.gw = p[0], p[1]
			pj, err := ps.project(append([]float32(nil), wantVit8[at:at+rows]...))
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, pj...)
			at += rows
		}
		return out
	}
	pn, _ := nmse32(wantEmb8, proj())
	t.Logf("the projector alone on transformers' tower rows: NMSE %.3e", pn)
	if math.IsNaN(pn) || pn > 1e-5 {
		t.Fatalf("the projector alone reads NMSE %.3e", pn)
	}
	for _, v := range []struct {
		name string
		set  func() func()
	}{
		{"no projector LayerNorm", func() func() { tw.Cfg.fault = faultLFM2NoProjNorm; return func() { tw.Cfg.fault = towerFaultNone } }},
		{"llama.cpp's tanh GELU", func() func() { tw.Cfg.fault = faultLFM2TanhGELU; return func() { tw.Cfg.fault = towerFaultNone } }},
		{"the unshuffle transposed", func() func() {
			was := tw.Cfg.fault
			tw.Cfg.fault = towerFaultShuffleSwap
			return func() { tw.Cfg.fault = was }
		}},
	} {
		undo := v.set()
		bad, _ := nmse32(wantEmb8, proj())
		undo()
		t.Logf("violation %q, the projector alone: NMSE %.3e", v.name, bad)
		if !(bad > 10*pn) {
			t.Errorf("%s reads %.3e against a clean %.3e: the gate cannot see it", v.name, bad, pn)
		}
	}
	tw.Cfg.fault = faultPhi4PosTransposed
	s = tw.testState()
	_, _, bad, _ := lfm2Encode(t, tw, s, path)
	s.Close()
	tw.Cfg.fault = towerFaultNone
	nb, _ := nmse32(wantEmb8, bad)
	t.Logf("violation, the table resized to the grid's transpose: NMSE %.3e", nb)
	if !(nb > 10*ne) {
		t.Errorf("a transposed position table reads %.3e against a clean %.3e", nb, ne)
	}

	got := lfm2FixRun(t, m, g, nil)
	w, flips := pixWorst(g.LogitsA8, got)
	wf, _ := pixWorst(g.Logits, got)
	t.Logf("from the pixels: worst logit NMSE %.3e, %d argmax flips (%.3e against the f32 tower)", w, flips, wf)
	for i := range got {
		if r := g.LogitsA8[i]; Greedy(got[i]) != r.Argmax && r.Margin >= tieMargin {
			t.Fatalf("row %d parts from transformers at a margin of %.3f", i, r.Margin)
		}
	}
	if math.IsNaN(w) || w > 2e-2 {
		t.Fatalf("end to end: worst NMSE %.3e", w)
	}
	for _, spec := range stepDevices() {
		gpu, err := openTier(t, tier.WithDevices(spec))
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			dev := lfm2FixRun(t, m, g, gpu)
			worst := 0.0
			for i := range dev {
				nm, _ := nmseOf(dev[i], got[i])
				worst = math.Max(worst, nm)
			}
			t.Logf("%s, every text and tower block: worst logit NMSE %.3e against the host", spec, worst)
			if math.IsNaN(worst) || worst > 1e-3 {
				t.Fatalf("%s: %.3e against the host", spec, worst)
			}
		})
	}
}

// lfm2FixRun lays the golden's prompt out as the processor did -- its text
// tokens and the picture's run (LFM2Spans) -- holds every token id to the
// processor's, prefills it and decodes the continuation; on gpu, every text
// and tower block placed.
func lfm2FixRun(t *testing.T, m *Model, g *lfm2FixGold, gpu *tier.GPU) [][]float32 {
	t.Helper()
	tw := m.Tower()
	cols, rows, pieces, err := tw.LFM2Pieces(decodeImage(t, testmodels.Path(filepath.Join(lfm2Fixture+"-gold", "picture.png"))))
	if err != nil {
		t.Fatal(err)
	}
	run, err := m.LFM2Spans(cols, rows, pieces)
	if err != nil {
		t.Fatal(err)
	}
	start, ok1 := m.Vocab.ID("<|image_start|>")
	end, ok2 := m.Vocab.ID("<|image_end|>")
	if !ok1 || !ok2 {
		t.Fatal("the vocabulary has no image markers")
	}
	first, last := -1, -1
	for i, id := range g.IDs {
		if id == start {
			first = i
		}
		if id == end {
			last = i
		}
	}
	if first < 0 || last < first {
		t.Fatal("the golden's prompt has no picture")
	}
	spans := append([]Span{{Tokens: g.IDs[:first]}}, run...)
	spans = append(spans, Span{Tokens: g.IDs[last+1:]})
	// The layout, id for id: every marker where the processor put it, and
	// each picture row where an image token stood.
	at := 0
	for _, sp := range spans {
		if sp.Picture != nil {
			for k := 0; k < sp.Picture.Rows; k++ {
				if g.IDs[at+k] != g.ImageToken {
					t.Fatalf("position %d is a picture row here and %d in the processor's prompt", at+k, g.IDs[at+k])
				}
			}
			at += sp.Picture.Rows
			continue
		}
		for _, id := range sp.Tokens {
			if g.IDs[at] != id {
				t.Fatalf("position %d is %d here and %d in the processor's prompt", at, id, g.IDs[at])
			}
			at++
		}
	}
	if at != len(g.IDs) {
		t.Fatalf("%d positions, the processor's %d", at, len(g.IDs))
	}
	s := m.NewState(len(g.IDs) + len(g.Cont) + 1)
	defer s.Close()
	if gpu != nil {
		if err := s.SetDevice(gpu); err != nil {
			t.Fatal(err)
		}
		if s.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("the device took %d of %d text blocks (%s)", s.GPULayers(), m.Cfg.NLayer, gpu.Err())
		}
		vs, err := s.Vision()
		if err != nil {
			t.Fatal(err)
		}
		if vs.GPUBlocks() != tw.Cfg.NLayer {
			t.Fatalf("the device took %d of %d tower blocks (%s)", vs.GPUBlocks(), tw.Cfg.NLayer, gpu.Err())
		}
	}
	lg, err := s.PrefillMixed(spans...)
	if err != nil {
		t.Fatal(err)
	}
	out := [][]float32{append([]float32(nil), lg...)}
	for _, id := range g.Cont[:len(g.Cont)-1] {
		if lg, err = s.Forward(id); err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out
}
