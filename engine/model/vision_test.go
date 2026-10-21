package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// testImage lives in testdata so the test runs on every checkout.
const testImage = "testdata/quad-and-disc.png"

// vlmTextPath is the TEXT half of the same VLM. A tower is not a model on its
// own any more -- one container carries both -- so every tower gate converts
// the pair and asks the model for its tower.
var vlmTextPath = testmodels.Path("SmolVLM-256M-Instruct-Q8_0.gguf")

// openTower opens the merged container and hands back the model and its tower.
// The model owns the container; closing IT closes both.
func openTower(t testing.TB, opts ...Option) (*Model, *Tower) {
	t.Helper()
	if !vlmAvailable(vlmTextPath, towerPath) {
		t.Skipf("MODEL MISSING: %s, and no merged container (set JITLLM_MODELS to the "+
			"model directory) -- this gate proved nothing", towerPath)
	}
	m, err := Open(jlmOfPair(t, vlmTextPath, towerPath), opts...)
	if err != nil {
		t.Fatal(err)
	}
	tw := m.Tower()
	if tw == nil {
		m.Close()
		t.Fatal("the merged container carries no tower, so this gate proves nothing")
	}
	return m, tw
}

// towerPath is the mmproj in the model directory (JITLLM_MODELS), resolved
// through testmodels so the tower gates run on every host.
var towerPath = testmodels.Path("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")

// TestOpenTower gates the tower loader against a real mmproj. Every figure was
// read off the file first and is asserted second.
func TestOpenTower(t *testing.T) {
	if !vlmAvailable(vlmTextPath, towerPath) {
		t.Skipf("MODEL MISSING: %s, and no merged container (set JITLLM_MODELS to the "+
			"model directory) -- this gate proved nothing", towerPath)
	}
	vlm, tw := openTower(t)
	defer vlm.Close()
	c := tw.Cfg

	for _, k := range []struct {
		name string
		got  int
		want int
	}{
		{"NLayer", c.NLayer, 12}, {"NEmbd", c.NEmbd, 768}, {"NFFN", c.NFFN, 3072},
		{"NHead", c.NHead, 12}, {"HeadDim", c.HeadDim, 64},
		{"ImageSz", c.ImageSz, 512}, {"PatchSz", c.PatchSz, 16},
		{"ProjDim", c.ProjDim, 576}, {"Scale", c.Scale, 4},
		{"Patches", c.Patches(), 1024}, {"Tokens", c.Tokens(), 64},
	} {
		if k.got != k.want {
			t.Errorf("%s = %d, want %d", k.name, k.got, k.want)
		}
	}
	if c.Act != nn.ActGELU {
		t.Errorf("activation %v; SmolVLM's mmproj sets clip.use_gelu", c.Act)
	}
	if c.CLS || c.PreLN {
		t.Errorf("CLS %v PreLN %v; an idefics3 tower has neither", c.CLS, c.PreLN)
	}
	if c.Seq() != c.Patches() {
		t.Errorf("Seq %d, Patches %d: with no class token they are the same", c.Seq(), c.Patches())
	}
	if c.Projector != "idefics3" {
		t.Errorf("projector %q", c.Projector)
	}
	if len(tw.layers()) != c.NLayer {
		t.Fatalf("%d blocks, want %d", len(tw.layers()), c.NLayer)
	}

	// The biases and layer norms must be loaded: without them the tower runs
	// and answers fluently wrong.
	b := &tw.layers()[0]
	for _, v := range []struct {
		name string
		got  []float32
		want int
	}{
		{"ln1.weight", b.attnNorm, c.NEmbd}, {"ln1.bias", b.attnNormB, c.NEmbd},
		{"ln2.weight", b.ffnNorm, c.NEmbd}, {"ln2.bias", b.ffnNormB, c.NEmbd},
		{"attn_q.bias", b.bq, c.NEmbd}, {"attn_out.bias", b.bo, c.NEmbd},
		{"fc1.bias", b.upB, c.NFFN}, {"fc2.bias", b.downB, c.NEmbd},
	} {
		if len(v.got) != v.want {
			t.Errorf("block 0 %s is %d floats, want %d", v.name, len(v.got), v.want)
		}
	}
	if b.wq.k != c.NEmbd || b.wq.rows != c.NEmbd {
		t.Errorf("wq is %dx%d, want %dx%d", b.wq.rows, b.wq.k, c.NEmbd, c.NEmbd)
	}
	// fc1 expands and fc2 contracts, whichever way the file names them:
	// SmolVLM's ffn_up/ffn_down are inverted relative to llama's convention.
	if b.up.k != c.NEmbd || b.up.rows != c.NFFN {
		t.Errorf("fc1 is %dx%d, want %dx%d", b.up.rows, b.up.k, c.NFFN, c.NEmbd)
	}
	if b.down.k != c.NFFN || b.down.rows != c.NEmbd {
		t.Errorf("fc2 is %dx%d, want %dx%d", b.down.rows, b.down.k, c.NEmbd, c.NFFN)
	}

	// The projector consumes Scale^2 patches at once: 768*16 = 12288 -> 576.
	if tw.projW.k != c.NEmbd*c.Scale*c.Scale || tw.projW.rows != c.ProjDim {
		t.Errorf("projector is %dx%d, want %dx%d", tw.projW.rows, tw.projW.k, c.ProjDim, c.NEmbd*c.Scale*c.Scale)
	}
	if n := c.Seq() * c.NEmbd; len(tw.posW) != n {
		t.Errorf("position embeddings %d, want %d", len(tw.posW), n)
	}

	// Every matmul's k must be a multiple of 32, the quantized block size.
	for _, w := range []struct {
		name string
		k    int
	}{{"wq", b.wq.k}, {"wo", b.wo.k}, {"fc1", b.up.k}, {"fc2", b.down.k}, {"proj", tw.projW.k}} {
		if w.k%32 != 0 {
			t.Errorf("%s has k=%d, not a multiple of 32", w.name, w.k)
		}
	}
	t.Logf("tower: %d blocks d=%d ffn=%d heads=%d  %d patches -> %d tokens of %d",
		c.NLayer, c.NEmbd, c.NFFN, c.NHead, c.Patches(), c.Tokens(), c.ProjDim)
}

// TestConvertRefusesAMismatchedPair checks the three refusals: Open refuses a
// GGUF, convert refuses a tower as the text model, and convert refuses a text
// model and tower that do not belong together (the width check is made once,
// at conversion).
func TestConvertRefusesAMismatchedPair(t *testing.T) {
	text := testmodels.Path("tinyllama-1.1b-q3_K_M.gguf")
	if _, err := os.Stat(text); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	if m, err := Open(text); err == nil {
		m.Close()
		t.Error("Open accepted a GGUF")
	} else if _, ok := err.(*NotConvertedError); !ok {
		t.Errorf("Open refused a GGUF with %v, want NotConvertedError", err)
	}
	if !vlmAvailable(vlmTextPath, towerPath) {
		t.Skipf("MODEL MISSING: %s, and no merged container (set JITLLM_MODELS to the "+
			"model directory) -- this gate proved nothing", towerPath)
	}
	// tinyllama's NEmbd is 2048 and SmolVLM's tower projects to 576, so this
	// pair cannot be converted together.
	dst := t.TempDir() + "/mismatch.jlm"
	_, err := convert.FromGGUFs(text, towerPath, dst, jlm.Fingerprint{Host: "test"})
	if err == nil {
		t.Fatal("convert accepted a tower that projects to the wrong width")
	}
	if !strings.Contains(err.Error(), "do not belong together") {
		t.Errorf("convert refused the pair with %v, want a message naming the mismatch", err)
	}
	// And the tower must not be offered as the TEXT model of the pair.
	if _, err := convert.FromGGUFs(towerPath, towerPath, dst, jlm.Fingerprint{Host: "test"}); err == nil {
		t.Error("convert accepted an mmproj as the text model")
	}
}

// TestTowerEncodes runs a real forward pass over an image-sized input. The
// numerical oracle is TestTowerMatchesLlamaCpp; this gates that the graph runs
// at the real shapes, is deterministic, and produces finite, non-constant
// output.
func TestTowerEncodes(t *testing.T) {
	if !vlmAvailable(vlmTextPath, towerPath) {
		t.Skipf("MODEL MISSING: %s, and no merged container (set JITLLM_MODELS to the "+
			"model directory) -- this gate proved nothing", towerPath)
	}
	// noTune: see the determinism assertion below, which needs the first
	// encode to be the settled one.
	vlm, tw := openTower(t, noTune)
	defer vlm.Close()
	c := tw.Cfg
	s := tw.testState()
	defer s.Close()

	px := make([]float32, c.ImageSz*c.ImageSz*3)
	for i := range px {
		// A deterministic gradient, not noise: a constant image would hide a
		// broken patch gather, since every patch would be identical.
		px[i] = float32((i*37)%255)/127.5 - 1
	}
	out, err := s.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != c.Tokens()*c.ProjDim {
		t.Fatalf("%d floats, want %d", len(out), c.Tokens()*c.ProjDim)
	}
	mn, mx := float64(out[0]), float64(out[0])
	for _, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite embedding")
		}
		mn, mx = math.Min(mn, float64(v)), math.Max(mx, float64(v))
	}
	if mx-mn < 1e-6 {
		t.Errorf("embeddings are constant (%g..%g) -- the graph is not doing anything", mn, mx)
	}
	// Determinism needs the tuner pinned: while it duels tiles, consecutive
	// encodes alternate between two values (int8 activations make a tile change
	// visible). Warming it instead costs ~22 encodes, too slow for the package
	// timeout, so noTune (at openTower) makes the first encode the settled one.
	s2 := tw.testState()
	defer s2.Close()
	first, err := s2.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	settled := append([]float32(nil), first...)
	out2, err := s2.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	var sse, sy2 float64
	for i := range settled {
		d := float64(settled[i] - out2[i])
		sse += d * d
		sy2 += float64(settled[i]) * float64(settled[i])
	}
	if nmse := sse / sy2; nmse > 1e-12 {
		t.Errorf("after the tuner settled, two runs disagree at NMSE %.3e", nmse)
	}
	t.Logf("%d tokens x %d, range %.3f..%.3f", c.Tokens(), c.ProjDim, mn, mx)
}

// TestPreprocessAndEncode preprocesses and encodes a real PNG. The image (four
// coloured quadrants and a disc) has structure so a broken patch gather or
// pixel shuffle shows up as embeddings that do not vary across the token grid.
func TestPreprocessAndEncode(t *testing.T) {
	if !vlmAvailable(vlmTextPath, towerPath) {
		t.Skipf("MODEL MISSING: %s, and no merged container (set JITLLM_MODELS to the "+
			"model directory) -- this gate proved nothing", towerPath)
	}
	f, err := os.Open(testImage)
	if err != nil {
		t.Fatalf("test image missing from testdata: %v", err)
	}
	defer f.Close()
	vlm, tw := openTower(t)
	defer vlm.Close()
	c := tw.Cfg

	px, err := preprocess(tw, f)
	if err != nil {
		t.Fatal(err)
	}
	if len(px) != c.ImageSz*c.ImageSz*3 {
		t.Fatalf("%d pixels, want %d", len(px), c.ImageSz*c.ImageSz*3)
	}
	lo, hi := float64(px[0]), float64(px[0])
	for _, v := range px {
		lo, hi = math.Min(lo, float64(v)), math.Max(hi, float64(v))
	}
	// mean 0.5 std 0.5 over [0,1] maps to [-1, 1].
	if lo < -1.001 || hi > 1.001 {
		t.Errorf("normalised range %.3f..%.3f, want within [-1,1]", lo, hi)
	}

	s := tw.testState()
	defer s.Close()
	out, err := s.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	// The tokens must differ: uniform embeddings over this picture mean the
	// gather or the shuffle collapsed the spatial layout.
	tok, d := c.Tokens(), c.ProjDim
	var maxPair float64
	for a := 0; a < tok; a++ {
		for b := a + 1; b < tok; b++ {
			var sse float64
			for i := 0; i < d; i++ {
				x := out[a*d+i] - out[b*d+i]
				sse += float64(x) * float64(x)
			}
			maxPair = math.Max(maxPair, sse)
		}
	}
	if maxPair < 1e-3 {
		t.Errorf("all %d image tokens are identical (max pair SSE %.3e) -- spatial layout collapsed", tok, maxPair)
	}
	t.Logf("preprocessed %.3f..%.3f; %d tokens, max pairwise SSE %.1f", lo, hi, tok, maxPair)
}

// TestTowerOpensWithoutFaultingAPage checks the tower is paged: Open faults
// nothing, Encode faults what it runs, and the six matrices stay in pages
// rather than the dense region. It fails against a LoadAll() at open.
func TestTowerOpensWithoutFaultingAPage(t *testing.T) {
	if !vlmAvailable(vlmTextPath, towerPath) {
		t.Skipf("MODEL MISSING: %s, and no merged container (set JITLLM_MODELS to the "+
			"model directory) -- this gate proved nothing", towerPath)
	}
	vlm, tw := openTower(t)
	defer vlm.Close()

	// Vision blocks start at NBlocks in the shared block space; counting from 0
	// would count the text model's.
	base, nb := int(tw.container.H.NBlocks), tw.container.VisionBlocks()
	if nb == 0 {
		t.Fatal("the container carries zero vision pages, so this gate proves nothing: " +
			"the six matrix roles per block must stay OUT of jlm.Role.Expanded")
	}
	resident := 0
	for i := base; i < base+nb; i++ {
		if tw.container.Resident(i) {
			resident++
		}
	}
	if resident != 0 {
		t.Fatalf("OpenTower faulted %d of %d pages; it must read shapes and the dense "+
			"region only, and leave every page to Encode", resident, nb)
	}

	// And the pages must actually hold the weights: a block whose six matrices
	// all landed in the dense region would pass the check above for the wrong
	// reason.
	for _, w := range []*tensor{&tw.layers()[0].wq, &tw.layers()[0].up} {
		if w.e == nil {
			t.Fatal("block 0 is missing a matrix entry")
		}
		if w.e.Block == jlm.DenseBlock || w.e.Role.Expanded() {
			t.Fatalf("%v is always-resident; the tower's matvec weights must stay "+
				"in pages so they can be relocated", w.e.Role)
		}
		if len(w.data) != 0 {
			t.Fatalf("%v is bound to %d bytes with its page out", w.e.Role, len(w.data))
		}
	}

	// Encoding faults the blocks it runs, and only those.
	st := tw.testState()
	defer st.Close()
	if _, err := st.Encode(make([]float32, tw.Cfg.ImageSz*tw.Cfg.ImageSz*3)); err != nil {
		t.Fatal(err)
	}
	resident = 0
	for i := base; i < base+nb; i++ {
		if tw.container.Resident(i) {
			resident++
		}
	}
	if resident != nb {
		t.Fatalf("after Encode %d of %d vision pages are resident; every block was run",
			resident, nb)
	}
}

// TestTowerEncodesUnderAPageBudget checks the tower is paged rather than merely
// lazy: a budget smaller than the tower must still give a bit-identical
// encoding, which needs eviction, re-read and re-bind all to be right.
func TestTowerEncodesUnderAPageBudget(t *testing.T) {
	px := make([]float32, 0)
	encode := func(budget uint64) []float32 {
		var opts []Option
		if budget > 0 {
			opts = append(opts, WithPageBudget(budget))
		}
		vlm, tw := openTower(t, opts...)
		defer vlm.Close()
		if budget > 0 {
			var all uint64
			for i := 0; i < vlm.container.NPages(); i++ {
				all += vlm.container.PageBytes(i)
			}
			if budget >= all {
				t.Fatalf("a %d-byte budget against %d of pages evicts nothing, so this "+
					"arm proves nothing", budget, all)
			}
		}
		if len(px) == 0 {
			px = make([]float32, tw.Cfg.ImageSz*tw.Cfg.ImageSz*3)
			for i := range px {
				px[i] = float32((i*37)%255)/127.5 - 1
			}
		}
		st := tw.testState()
		defer st.Close()
		out, err := st.Encode(px)
		if err != nil {
			t.Fatalf("budget=%d: %v", budget, err)
		}
		if _, ev := vlm.container.Faults(); budget > 0 && ev == 0 {
			t.Fatalf("budget=%d evicted nothing; it was not the binding constraint", budget)
		}
		return append([]float32(nil), out...)
	}

	full := encode(0)
	// Vision pages are the bigger of the two arrays here, so a budget of a few
	// of them forces eviction across both.
	vlm, _ := openTower(t)
	vpage := vlm.container.PageBytes(int(vlm.container.H.NBlocks))
	vlm.Close()
	for _, k := range []uint64{1, 2, 4} {
		got := encode(k * vpage)
		if len(got) != len(full) {
			t.Fatalf("budget=%d pages: %d floats, want %d", k, len(got), len(full))
		}
		for i := range full {
			if got[i] != full[i] {
				t.Fatalf("budget=%d pages: element %d is %v, want %v -- a page-in bound "+
					"a stale span", k, i, got[i], full[i])
			}
		}
	}
}

// TestOneContainerCarriesTextAndVision asserts the text and vision blocks are
// pages of the same container, in one block space, against one budget. A
// regression to two files fails at m.Tower() == nil or the vision page count.
func TestOneContainerCarriesTextAndVision(t *testing.T) {
	vlm, tw := openTower(t)
	defer vlm.Close()
	c := vlm.container

	nText, nVis := int(c.H.NBlocks), c.VisionBlocks()
	if nText == 0 || nVis == 0 {
		t.Fatalf("container has %d text and %d vision blocks; one file must carry both",
			nText, nVis)
	}
	if got := c.NPages(); got != nText+nVis {
		t.Fatalf("the block space is %d pages for %d text + %d vision", got, nText, nVis)
	}
	if nVis != tw.Cfg.NLayer {
		t.Fatalf("%d vision pages for a %d-layer tower", nVis, tw.Cfg.NLayer)
	}
	// The two arrays keep their own page sizes; that is why there are two.
	text, vis := c.PageBytes(0), c.PageBytes(nText)
	if text == 0 || vis == 0 {
		t.Fatalf("page sizes are %d and %d", text, vis)
	}
	// A vision block's bytes must resolve through the same Span every text
	// weight uses, with no second entry point.
	if e := tw.layers()[0].wq.e; e == nil {
		t.Fatal("vision block 0 has no q entry")
	} else if got := c.BlockOf(e); got != nText {
		t.Fatalf("vision block 0's q maps to block %d, want %d", got, nText)
	}
	// And one budget binds across both arrays.
	if b := c.Budget(); b != 0 && b < text+vis {
		t.Fatalf("budget %d cannot hold one page of each array (%d + %d)", b, text, vis)
	}
}

// TestTowerOffersItsBlocksToTheDevice checks the tower's blocks reach
// PrepLayer: a block the device cannot express must be declined by the device,
// not withheld by the model. A regression to withholding fails with 0 offers.
func TestTowerOffersItsBlocksToTheDevice(t *testing.T) {
	vlm, tw := openTower(t)
	defer vlm.Close()

	d := &countingDevice{}
	s := tw.testState()
	defer s.Close()
	s.SetDevice(d)

	// One offer, not twelve: placement is a contiguous prefix, so offering
	// stops at the first decline, as the text model's loop does.
	if d.offered < 1 {
		t.Fatalf("the device was offered %d tower blocks; a block the tier cannot "+
			"express must be DECLINED by it, not withheld here", d.offered)
	}
	if s.GPUBlocks() != 0 {
		t.Fatalf("a device that declined every block reports %d placed", s.GPUBlocks())
	}
	// The offer must carry the shape, or the tier is declining a block it was
	// never told about.
	if !d.sawVision {
		t.Error("the plan did not set Vision, so a tier would judge a ViT block " +
			"by a text block's rules")
	}
	if !d.sawNormBias {
		t.Error("the weights carried no LayerNorm bias: a tier that adds the " +
			"attention biases and skips these runs and is wrong")
	}
	if !d.sawMLPBias {
		t.Error("the weights carried no MLP biases")
	}
}

// countingDevice is an nn.LayerDevice that declines everything and remembers
// what it was shown. It is not a real tier because this gate is about the offer.
type countingDevice struct {
	// Embedded nil: this fake implements PrepLayer and nothing else, so a
	// change that starts calling another method here panics with the method
	// name instead of silently measuring something else.
	nn.LayerDevice
	offered                            int
	sawVision, sawNormBias, sawMLPBias bool
}

// Reserve is the scratch sizing every offer sequence opens with; this fake
// has none to size.
func (d *countingDevice) Reserve(maxRows, maxK int) bool { return true }

func (d *countingDevice) PrepLayer(li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	d.offered++
	if p.NonCausal {
		d.sawVision = true
	}
	if len(w.AttnNormB) > 0 && len(w.FFNNormB) > 0 {
		d.sawNormBias = true
	}
	if len(w.BUp) > 0 && len(w.BDown) > 0 {
		d.sawMLPBias = true
	}
	return false // every tier declines a ViT block, by name
}

// TestTowerOnDeviceMatchesHost compares a device-run tower block against the
// host (itself held to llama.cpp by TestTowerMatchesLlamaCpp). A ViT block's
// differences (LayerNorm biases, biased q/k/v/o, ungated GELU MLP,
// bidirectional attention, no rotary) all fail silently if wired wrong.
//
// The bound is taken after one block: both tiers quantize activations to int8
// and round differently, so error compounds over twelve blocks without any
// defect. End-to-end token ids are cmd/jitllm's VLM gate.
func TestTowerOnDeviceMatchesHost(t *testing.T) {
	vlm, tw := openTower(t)
	defer vlm.Close()
	c := tw.Cfg

	px := make([]float32, c.ImageSz*c.ImageSz*3)
	for i := range px {
		px[i] = float32((i*37)%255)/127.5 - 1
	}
	// after0 runs the tower and returns the residual after the FIRST block.
	after0 := func(s *State) []float32 {
		var got []float32
		s.vis.afterBlock = func(li int, x []float32) {
			if li == 0 && got == nil {
				got = append([]float32(nil), x...)
			}
		}
		if _, err := s.Encode(px); err != nil {
			t.Fatal(err)
		}
		if got == nil {
			t.Fatal("no residual captured after block 0")
		}
		return got
	}

	host := tw.testState()
	want := after0(host)
	host.Close()

	// Every backend on this host, not just the one tier.Open would pick:
	// lowertest proves the kernels assemble on all three, only this proves they
	// run. Absent backends are logged and skipped.
	ran := 0
	for _, spec := range stepDevices() {
		g, err := openTier(t, tier.WithDevices(spec))
		if err != nil || g == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer g.Close()

			// Every block must be accepted.
			all := tw.testState()
			all.SetDevice(g)
			n := all.GPUBlocks()
			all.Close()
			if n != c.NLayer {
				t.Fatalf("%s took %d of %d tower blocks (%s)", spec, n, c.NLayer, g.Err())
			}

			// Exactly one block is placed for the comparison: the device runs
			// its range in one submission, so the residual is only observable
			// where that range ends.
			defer func(was int) { tw.opt.towerGPULayers = was }(tw.opt.towerGPULayers)
			tw.opt.towerGPULayers = 1
			s := tw.testState()
			defer s.Close()
			s.SetDevice(g)
			if s.GPUBlocks() != 1 {
				t.Fatalf("asked for 1 device block, got %d", s.GPUBlocks())
			}
			got := after0(s)

			if len(got) != len(want) {
				t.Fatalf("device residual is %d floats, host %d", len(got), len(want))
			}
			var num, den, worst float64
			for i := range want {
				d := float64(got[i] - want[i])
				num += d * d
				den += float64(want[i]) * float64(want[i])
				if a := math.Abs(d); a > worst {
					worst = a
				}
			}
			nmse := num / den
			t.Logf("%s, after block 0: NMSE %.3e, max|d| %.4f over %d rows of %d",
				spec, nmse, worst, c.Patches(), c.NEmbd)
			// The bound is a decade above the clean reading and far below any
			// of the silent failures (a causal mask reads ~1e-1). Non-finite is
			// checked first because NaN > 5e-5 is false.
			if math.IsNaN(nmse) || math.IsInf(nmse, 0) || math.IsInf(float64(worst), 0) {
				t.Fatalf("NMSE %v, max|d| %v on %s: the comparison is degenerate, "+
					"which is a failure and not a pass", nmse, worst, spec)
			}
			// Both sides quantize activations to int8 where the device takes
			// the dp4a path, so they share that rounding and agree closely.
			// A tensor-core path that multiplies binary16 activations differs from
			// the host by the int8 rounding itself, and Metal's matvec rounds the
			// activation its own way too. The bound follows the arithmetic the arm
			// ran, far under the violations' 2.1e-1 either way.
			bound := 5e-5
			if st := g.Stats(); st.VoltaGemm+st.GemmF16 > 0 || spec == "metal" {
				bound = 5e-4
			}
			if nmse > bound {
				t.Fatalf("NMSE %.3e after ONE block on %s (bound %.0e): the device is not "+
					"running this block's graph", nmse, spec, bound)
			}
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}
