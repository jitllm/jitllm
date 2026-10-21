package model

import (
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// The llava-phi-3 gates: a CLIP tower with a class token, a pre-loop
// LayerNorm, a two-linear MLP projector and quick-GELU. Each produces fluent
// wrong output rather than an error if it is wrong, so each is asserted on its
// own rather than through one end-to-end number.

// llavaText and llavaProj are the pair, in the model directory (JITLLM_MODELS).
var llavaText = testmodels.Path("llava-phi-3-mini-int4.gguf")
var llavaProj = testmodels.Path("llava-phi-3-mini-mmproj-f16.gguf")

// openLlava converts the pair and opens the container. A missing model fails
// rather than skips, naming the files to fetch.
func openLlava(t testing.TB, opts ...Option) (*Model, *Tower) {
	t.Helper()
	if !vlmAvailable(llavaText, llavaProj) {
		testmodels.Missing(t, "%s, its mmproj and its merged container are all missing. RULE 11: a "+
			"missing artefact is a TASK, not a constraint -- fetch llava-phi-3-mini "+
			"(text + mmproj) into "+testmodels.Dir()+" "+
			"(set JITLLM_MODELS to the model directory) and re-run. Skipping here would make this "+
			"gate a green line that proves nothing.", llavaText)
	}
	m, err := Open(jlmOfPair(t, llavaText, llavaProj), opts...)
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

// llavaImage is an asymmetric picture (a bright region plus a per-channel
// ramp). The things this file gates are permutations, under which a periodic
// image such as a checkerboard is nearly invariant.
func llavaImage(n int) []float32 {
	px := make([]float32, n*n*3)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			for ch := 0; ch < 3; ch++ {
				v := float32(x) / float32(n)
				if y < n/3 && x > n/2 {
					v = 1
				}
				px[(y*n+x)*3+ch] = v * float32(ch+1) / 3
			}
		}
	}
	return px
}

// TestLlavaTowerLoads reads the geometry off the container and asserts every
// number that decides the graph.
func TestLlavaTowerLoads(t *testing.T) {
	vlm, tw := openLlava(t)
	defer vlm.Close()
	c := tw.Cfg
	for _, k := range []struct {
		name      string
		got, want int
	}{
		{"NLayer", c.NLayer, 23}, {"NEmbd", c.NEmbd, 1024}, {"NFFN", c.NFFN, 4096},
		{"NHead", c.NHead, 16}, {"HeadDim", c.HeadDim, 64},
		{"ImageSz", c.ImageSz, 336}, {"PatchSz", c.PatchSz, 14},
		{"Scale", c.Scale, 1},
		{"Patches", c.Patches(), 576},
		// Seq and Tokens differ: 577 rows (with the class token) go through
		// the blocks and 576 come out of the projector.
		{"Seq", c.Seq(), 577}, {"Tokens", c.Tokens(), 576},
		// Not clip.vision.projection_dim (768, CLIP's contrastive dim): mm.2
		// emits 3072, which is what phi-3 wants.
		{"ProjDim", c.ProjDim, 3072},
	} {
		if k.got != k.want {
			t.Errorf("%s = %d, want %d", k.name, k.got, k.want)
		}
	}
	if c.Projector != "mlp" {
		t.Errorf("projector %q, want mlp", c.Projector)
	}
	// clip.use_gelu is 0 and there is no clip.use_silu, which means quick-GELU,
	// not SiLU.
	if c.Act != nn.ActQuickGELU {
		t.Errorf("activation %v, want quick-gelu (clip.use_gelu=0 and no use_silu)", c.Act)
	}
	if !c.CLS {
		t.Error("no class token; CLIP prepends one and the position table has 577 rows for it")
	}
	if !c.PreLN {
		t.Error("no pre-loop LayerNorm; v.pre_ln.weight is in the file")
	}
	if n := c.Seq() * c.NEmbd; len(tw.posW) != n {
		t.Errorf("position table is %d floats, want %d", len(tw.posW), n)
	}
	if len(tw.clsW) != c.NEmbd {
		t.Errorf("class embedding is %d floats, want %d", len(tw.clsW), c.NEmbd)
	}
	t.Logf("llava-phi-3 tower: %d blocks, %d wide, %d patches + 1 class row, "+
		"projector %s %d->%d->%d, %v", c.NLayer, c.NEmbd, c.Patches(), c.Projector,
		c.NEmbd, tw.projW.rows, c.ProjDim, c.Act)
}

// TestLlavaTowerEncodes is the end-to-end one: a real image through 23 blocks
// and a two-matrix projector.
func TestLlavaTowerEncodes(t *testing.T) {
	vlm, tw := openLlava(t)
	defer vlm.Close()
	c := tw.Cfg
	s := tw.testState()
	defer s.Close()

	out, err := s.Encode(llavaImage(c.ImageSz))
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != c.Tokens()*c.ProjDim {
		t.Fatalf("%d floats, want %d", len(out), c.Tokens()*c.ProjDim)
	}
	mn, mx := math.Inf(1), math.Inf(-1)
	var sum float64
	for _, v := range out {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			t.Fatalf("non-finite embedding %v", v)
		}
		mn, mx, sum = math.Min(mn, f), math.Max(mx, f), sum+f
	}
	// A collapsed tower is finite; the spread is what says the graph carried
	// information.
	if mx-mn < 1e-3 {
		t.Fatalf("every embedding is within 1e-3 of every other (%.6f..%.6f): the tower "+
			"produced a constant, which is finite and useless", mn, mx)
	}
	// Rows must differ from each other too: a projector fed the same row 576
	// times passes the spread check above and is a picture of one patch.
	same := 0
	for r := 1; r < c.Tokens(); r++ {
		d := float64(0)
		for j := 0; j < 16; j++ {
			d += math.Abs(float64(out[r*c.ProjDim+j] - out[j]))
		}
		if d < 1e-6 {
			same++
		}
	}
	if same > 0 {
		t.Fatalf("%d of %d output rows are identical to row 0", same, c.Tokens()-1)
	}
	t.Logf("llava-phi-3 encode: %d embeddings x %d, range %.4f..%.4f, mean %.6f",
		c.Tokens(), c.ProjDim, mn, mx, sum/float64(len(out)))
}

// TestLlavaTowerTouchesNoRowMajorQuant asserts no quantized weight is read
// through the GGUF block layout during an encode (see engine/nn/rowmajor.go). The
// counter is package-wide, so it is reset right before the encode.
func TestLlavaTowerTouchesNoRowMajorQuant(t *testing.T) {
	vlm, tw := openLlava(t)
	defer vlm.Close()
	s := tw.testState()
	defer s.Close()

	// Every matrix packed is the precondition, checked first.
	for i, w := range tw.weights() {
		if w.packed == nil {
			t.Fatalf("tower matrix %d is %v and not packed: the row-major family is "+
				"reachable for it", i, w.typ)
		}
		if !jlm.Packed(w.e.Type) {
			t.Fatalf("tower matrix %d is container type %v, which is not packed", i, w.e.Type)
		}
	}

	nn.ResetRowMajorQuant()
	if _, err := s.Encode(llavaImage(tw.Cfg.ImageSz)); err != nil {
		t.Fatal(err)
	}
	if got := nn.RowMajorQuantCalls(); got != 0 {
		t.Fatalf("%d quantized matrices were read through the GGUF block layout during one "+
			"encode; the engine reads ONE weight layout and GGUF is the converter's input", got)
	}
	t.Logf("llava-phi-3 encode: %d tower matrices, all packed, 0 calls into the row-major "+
		"quantized family", len(tw.weights()))
}

// TestLlavaPatchEmbeddingIsTwoDimensional pins the converter's reshape: the 4-D
// conv kernel is written as a 2-D matmul with k padded to a whole block.
func TestLlavaPatchEmbeddingIsTwoDimensional(t *testing.T) {
	vlm, tw := openLlava(t)
	defer vlm.Close()
	c := tw.Cfg
	e := tw.patchW.e
	if e.NDim != 2 {
		t.Fatalf("patch_embd is %d-dimensional in the container, want 2", e.NDim)
	}
	want := c.PatchSz * c.PatchSz * 3
	if tw.patchW.rows != c.NEmbd {
		t.Errorf("patch_embd has %d rows, want %d", tw.patchW.rows, c.NEmbd)
	}
	if tw.patchW.k < want || tw.patchW.k%32 != 0 {
		t.Errorf("patch_embd k is %d; want >= %d (14x14x3) and a multiple of 32", tw.patchW.k, want)
	}
	if tw.patchW.k != 608 {
		t.Errorf("patch_embd k is %d, want 608 (588 padded to 19 blocks of 32)", tw.patchW.k)
	}
	if !jlm.Packed(e.Type) {
		t.Errorf("patch_embd is %v, not a packed type", e.Type)
	}
	t.Logf("patch_embd: %d x %d, %v -- a conv whose stride equals its kernel, written as "+
		"the matmul it is", tw.patchW.k, tw.patchW.rows, e.Type)
}

// TestLlavaTowerRunsOnTheDevice checks every quick-GELU tower block is accepted
// and one block on the device matches the host after block 0, as
// TestTowerOnDeviceMatchesHost does. The weights are F16, so this also covers
// the float matvec end to end.
func TestLlavaTowerRunsOnTheDevice(t *testing.T) {
	vlm, tw := openLlava(t)
	defer vlm.Close()
	c := tw.Cfg
	px := make([]float32, c.ImageSz*c.ImageSz*3)
	for i := range px {
		px[i] = float32((i*37)%255)/127.5 - 1
	}
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
			all := tw.testState()
			all.SetDevice(g)
			n := all.GPUBlocks()
			all.Close()
			if n != c.NLayer {
				t.Fatalf("%s took %d of %d quick-GELU tower blocks (%s)", spec, n, c.NLayer, g.Err())
			}
			defer func(was int) { tw.opt.towerGPULayers = was }(tw.opt.towerGPULayers)
			tw.opt.towerGPULayers = 1
			s := tw.testState()
			defer s.Close()
			s.SetDevice(g)
			if s.GPUBlocks() != 1 {
				t.Fatalf("asked for 1 device block, got %d", s.GPUBlocks())
			}
			got := after0(s)
			var num, den float64
			for i := range want {
				d := float64(got[i] - want[i])
				num, den = num+d*d, den+float64(want[i])*float64(want[i])
			}
			nmse := num / den
			if math.IsNaN(nmse) || math.IsInf(nmse, 0) || !(nmse < 5e-5) {
				t.Fatalf("NMSE %.3e after ONE block on %s", nmse, spec)
			}
			t.Logf("%s: %d of %d blocks accepted; after block 0, NMSE %.3e", spec, n, c.NLayer, nmse)
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}

// onePatch paints exactly one patch bright and everything else black.
func onePatch(n, patch, idx int) []float32 {
	px := make([]float32, n*n*3)
	side := n / patch
	py, pxi := idx/side, idx%side
	for dy := 0; dy < patch; dy++ {
		for dx := 0; dx < patch; dx++ {
			o := ((py*patch+dy)*n + pxi*patch + dx) * 3
			px[o], px[o+1], px[o+2] = 1, 1, 1
		}
	}
	return px
}

// TestLlavaOutputRowsAlignWithPatches gates which rows the projector consumes.
// With a class token at row 0 it must take rows 1..576; an off-by-one drops
// the first or last patch and still yields plausible embeddings.
//
// It paints one patch and diffs against an all-black encode, removing the
// common mode that bidirectional attention spreads to every row. Patch 0 and
// the last patch are included because an off-by-one moves them off the end;
// patch 0 has the weakest margin, hence the 2x bar.
func TestLlavaOutputRowsAlignWithPatches(t *testing.T) {
	vlm, tw := openLlava(t)
	defer vlm.Close()
	c := tw.Cfg
	s := tw.testState()
	defer s.Close()

	enc := func(px []float32) []float32 {
		out, err := s.Encode(px)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), out...)
	}
	base := enc(make([]float32, c.ImageSz*c.ImageSz*3))

	for _, idx := range []int{0, 137, c.Patches() - 1} {
		out := enc(onePatch(c.ImageSz, c.PatchSz, idx))
		best, bestD, second := -1, -1.0, -1.0
		for r := 0; r < c.Tokens(); r++ {
			d := 0.0
			for j := 0; j < c.ProjDim; j++ {
				e := float64(out[r*c.ProjDim+j]) - float64(base[r*c.ProjDim+j])
				d += e * e
			}
			if d > bestD {
				best, second, bestD = r, bestD, d
			} else if d > second {
				second = d
			}
		}
		if best != idx {
			t.Fatalf("patch %d was painted and output row %d moved most (%.3f against "+
				"the runner-up %.3f): the projector is reading the wrong rows",
				idx, best, bestD, second)
		}
		// The margin is asserted too: a narrow winner is a coin toss.
		if bestD < 2*second {
			t.Fatalf("patch %d -> row %d, but only %.2fx the runner-up; the signal is too "+
				"weak for this gate to discriminate a one-row shift", idx, best, bestD/second)
		}
		t.Logf("patch %d -> row %d, %.1fx the runner-up (%.1f against %.1f)",
			idx, best, bestD/second, bestD, second)
	}
}

// TestLlavaPositionsArePairedWithTheirRows gates the position table's PAIRING:
// row r of the residual must read row r of the table and no other. It perturbs
// one row at a time (rotating the whole table would pass a tower that reads
// row 0 everywhere); both ends are covered.
//
// It does not prove the offset (patch i reading row i+1); that is pinned by
// the table having Seq() rows and by TestLlavaOutputRowsAlignWithPatches.
func TestLlavaPositionsArePairedWithTheirRows(t *testing.T) {
	vlm, tw := openLlava(t)
	defer vlm.Close()
	c := tw.Cfg
	px := llavaImage(c.ImageSz)

	enc := func() []float32 {
		s := tw.testState()
		defer s.Close()
		out, err := s.Encode(px)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), out...)
	}
	rel := func(a, b []float32) float64 {
		var sse, sy2 float64
		for i := range a {
			d := float64(a[i]) - float64(b[i])
			sse, sy2 = sse+d*d, sy2+float64(b[i])*float64(b[i])
		}
		return math.Sqrt(sse / sy2)
	}

	want := enc()
	if d := rel(enc(), want); d != 0 {
		t.Fatalf("two identical encodes differ by %.3e; the control fails, so the "+
			"violations below would prove nothing", d)
	}

	saved := append([]float32(nil), tw.posW...)
	defer copy(tw.posW, saved)
	for _, row := range []int{0, 300, c.Seq() - 1} {
		copy(tw.posW, saved)
		for j := row * c.NEmbd; j < (row+1)*c.NEmbd; j++ {
			tw.posW[j] += 1
		}
		d := rel(enc(), want)
		if d < 1e-6 {
			t.Fatalf("perturbing position row %d of %d moved the embeddings by %.3e: that "+
				"row is not read by the row it belongs to", row, c.Seq(), d)
		}
		t.Logf("perturbing position row %d moves the output by %.5f relative RMS", row, d)
	}
}

// TestLlavaPreLoopNormIsApplied checks pre_ln is applied, not just loaded: a
// skipped pre_ln does not fault, since later LayerNorms renormalise. It
// perturbs the weight and asserts the output moves, after a same-encode-twice
// control.
func TestLlavaPreLoopNormIsApplied(t *testing.T) {
	vlm, tw := openLlava(t)
	defer vlm.Close()
	c := tw.Cfg
	if !c.PreLN || tw.preLnW == nil {
		t.Fatal("this container carries no pre_ln, so the gate proves nothing")
	}
	px := llavaImage(c.ImageSz)
	enc := func() []float32 {
		s := tw.testState()
		defer s.Close()
		out, err := s.Encode(px)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), out...)
	}
	rel := func(a, b []float32) float64 {
		var sse, sy2 float64
		for i := range a {
			d := float64(a[i]) - float64(b[i])
			sse, sy2 = sse+d*d, sy2+float64(b[i])*float64(b[i])
		}
		return math.Sqrt(sse / sy2)
	}
	want := enc()
	if d := rel(enc(), want); d != 0 {
		t.Fatalf("two identical encodes differ by %.3e; the control fails", d)
	}
	saved := append([]float32(nil), tw.preLnW...)
	defer copy(tw.preLnW, saved)
	for i := range tw.preLnW {
		tw.preLnW[i] *= 2
	}
	if d := rel(enc(), want); d < 0.01 {
		t.Fatalf("doubling v.pre_ln.weight moved the embeddings by %.3e relative RMS: "+
			"the pre-loop norm is loaded and not applied", d)
	} else {
		t.Logf("doubling v.pre_ln.weight moves the output by %.4f relative RMS", d)
	}
}

// TestLlavaEncodesARealImage runs the shipping path: a PNG off disk,
// Preprocess (decode, resize, mean/std), Encode. It covers this tower's own
// geometry (336x336, patch 14) and CLIP's mean/std, which the synthetic-input
// tests skip.
func TestLlavaEncodesARealImage(t *testing.T) {
	vlm, tw := openLlava(t)
	defer vlm.Close()
	c := tw.Cfg
	f, err := os.Open(testImage)
	if err != nil {
		t.Fatalf("%s: %v -- RULE 11: a missing artefact is a task", testImage, err)
	}
	defer f.Close()
	px, err := preprocess(tw, f)
	if err != nil {
		t.Fatal(err)
	}
	if len(px) != c.ImageSz*c.ImageSz*3 {
		t.Fatalf("Preprocess gave %d floats, want %d", len(px), c.ImageSz*c.ImageSz*3)
	}
	s := tw.testState()
	defer s.Close()
	out, err := s.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	mn, mx := math.Inf(1), math.Inf(-1)
	for _, v := range out {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			t.Fatalf("non-finite embedding %v", v)
		}
		mn, mx = math.Min(mn, f), math.Max(mx, f)
	}
	if mx-mn < 1e-3 {
		t.Fatalf("every embedding is within 1e-3 of every other (%.6f..%.6f)", mn, mx)
	}
	t.Logf("%s -> %d embeddings x %d, range %.4f..%.4f; first row starts %.4f %.4f %.4f",
		testImage, c.Tokens(), c.ProjDim, mn, mx, out[0], out[1], out[2])
}
