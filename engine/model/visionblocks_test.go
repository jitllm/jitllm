package model

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// The gates of docs/design/vision-as-blocks.md: a picture runs through the
// model's own runner, kernel cache, placement and scheduler, and its rows are
// named, so the image cache and the prefix cache can find it again.

// TestVisionRunsAsBlocksOfTheModel is the structural gate: the tower's blocks
// are the model's layers, an encode runs every one of them through the
// runner's block body on the text State's JIT, and nothing in the package
// offers a block to a device but the one placement path.
func TestVisionRunsAsBlocksOfTheModel(t *testing.T) {
	m, tw := openTower(t)
	defer m.Close()
	c := tw.Cfg
	if tw.base != int(m.container.H.NBlocks) || len(m.layers) != tw.base+c.NLayer {
		t.Fatalf("the vision blocks are not the model's blocks %d..%d (layers %d)",
			tw.base, tw.base+c.NLayer, len(m.layers))
	}
	st := m.NewState(64)
	defer st.Close()
	vs, err := st.Vision()
	if err != nil {
		t.Fatal(err)
	}
	if vs.jit != st.jit {
		t.Fatal("the vision segment runs on a JIT of its own: a second kernel cache")
	}
	if again, _ := st.Vision(); again != vs {
		t.Fatal("a second Vision call made a second vision State")
	}
	if _, err := vs.Encode(rainbow(c.ImageSz)); err != nil {
		t.Fatal(err)
	}
	if got := int(vs.vis.hostBlocks); got != c.NLayer {
		t.Fatalf("%d of %d tower blocks ran through the runner's block body", got, c.NLayer)
	}
	t.Logf("%d tower blocks at %d..%d of the block space, %d run by hostRows on the text State's JIT",
		c.NLayer, tw.base, tw.base+c.NLayer, vs.vis.hostBlocks)
}

// TestPrepLayerHasOneCaller is the one offer path read off the source:
// PrepLayer is called from offerRange and nowhere else in the package's
// production code, so a tower (or anything) placed through a path of its own
// fails here.
func TestPrepLayerHasOneCaller(t *testing.T) {
	if os.Getenv("JITLLM_SHIPPED_BINARY") != "" {
		t.Skip("LOUD SKIP -- NOT A PASS: this reads the package's source, and this is a shipped test " +
			"binary running away from the tree it was built from. That tree runs this check.")
	}
	callers := prepLayerCallers(t)
	if !slices.Equal(callers, []string{"offerRange"}) {
		t.Fatalf("PrepLayer is called from %v: the vision tower (or anything) has a placement path of its own",
			callers)
	}
}

// prepLayerCallers is every production function in engine/model that calls
// PrepLayer, by name.
func prepLayerCallers(t *testing.T) []string {
	t.Helper()
	fset := token.NewFileSet()
	ents, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, e := range ents {
		n := e.Name()
		if !strings.HasSuffix(n, ".go") || strings.HasSuffix(n, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, filepath.Join(".", n), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			ast.Inspect(fd.Body, func(x ast.Node) bool {
				if call, ok := x.(*ast.CallExpr); ok {
					if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "PrepLayer" {
						if !slices.Contains(out, fd.Name.Name) {
							out = append(out, fd.Name.Name)
						}
					}
				}
				return true
			})
		}
	}
	slices.Sort(out)
	return out
}

// TestImageCacheSkipsTheTower: a picture encoded twice runs the tower once --
// the second encode is the image cache's, no block runs and the rows are the
// same bits -- and a different picture runs it again.
func TestImageCacheSkipsTheTower(t *testing.T) {
	m, tw := openTower(t)
	defer m.Close()
	st := m.NewState(64)
	defer st.Close()
	vs, err := st.Vision()
	if err != nil {
		t.Fatal(err)
	}
	px := rainbow(tw.Cfg.ImageSz)
	first, err := vs.EncodeImage(px)
	if err != nil {
		t.Fatal(err)
	}
	ran := vs.vis.hostBlocks + vs.vis.devBlocks
	second, err := vs.EncodeImage(px)
	if err != nil {
		t.Fatal(err)
	}
	if again := vs.vis.hostBlocks + vs.vis.devBlocks; again != ran {
		t.Fatalf("the second encode of one picture ran %d tower blocks", again-ran)
	}
	if !slices.Equal(first.Embd, second.Embd) || *first.Key != *second.Key {
		t.Fatal("the cached picture's rows or name differ from the encode's")
	}
	other := append([]float32(nil), px...)
	other[len(other)/2] += 0.25
	third, err := vs.EncodeImage(other)
	if err != nil {
		t.Fatal(err)
	}
	if vs.vis.hostBlocks+vs.vis.devBlocks == ran || *third.Key == *first.Key {
		t.Fatal("a different picture was answered from the cache")
	}
	hits, misses := m.ImageCacheStats()
	t.Logf("%d hit(s), %d miss(es); a cached picture ran 0 blocks, a different one %d",
		hits, misses, vs.vis.hostBlocks+vs.vis.devBlocks-ran)
}

// TestImageKeyCoversTheLayout: the name covers what the tower reads -- the
// pixels, the grid they are patched on and the preprocessing -- so one pixel
// buffer on two grids is two pictures. Against the violation (faultKeyAliases: the
// pixels alone) the two alias.
func TestImageKeyCoversTheLayout(t *testing.T) {
	c := TowerConfig{Projector: "qwen2vl_merger", ImageSz: 56, PatchSz: 14, Scale: 2,
		Mean: [3]float64{0.5, 0.5, 0.5}, Std: [3]float64{0.5, 0.5, 0.5}}
	px := rainbow(56)
	for _, fault := range []bool{false, true} {
		if fault {
			c.fault = faultKeyAliases
		}
		a, b := c.imageKey(px, 2, 8), c.imageKey(px, 8, 2)
		other := c
		other.Mean[0] = 0.48
		d := other.imageKey(px, 2, 8)
		c.fault = towerFaultNone
		if !fault && (a == b || a == d) {
			t.Fatal("one pixel buffer under another grid or another normalisation keys the same")
		}
		if fault && a != b {
			t.Fatal("the violation (pixels alone) still tells the grids apart, so this gate proves nothing")
		}
	}
}

// TestTwoTurnImageChatRestoresPastThePicture: a second turn about the same
// picture restores every position the first turn's prompt covered -- the
// picture's rows and the text after them -- and decodes the same tokens a
// cold run does. Against the violation (the picture's rows unnamed, the gap
// the prefix cache stops at) nothing past the picture restores, and a
// different picture restores nothing past its start.
func TestTwoTurnImageChatRestoresPastThePicture(t *testing.T) {
	m, tw := openTower(t)
	defer m.Close()
	px := rainbow(tw.Cfg.ImageSz)
	other := append([]float32(nil), px...)
	other[7] += 0.5
	turn1 := []ChatMessage{{Role: "user", Content: "What is in this picture?", Images: 1}}
	turn2 := append(append([]ChatMessage(nil), turn1...),
		ChatMessage{Role: "assistant", Content: "A rainbow."},
		ChatMessage{Role: "user", Content: "What colour is at the top?"})
	type result struct {
		restored, imgAt, imgEnd, n int
		// lookups is the image cache's hits and misses the prompt made, and
		// blocks the tower blocks it ran: zero both when the prefix cache
		// restored the picture whole.
		lookups, blocks int64
		toks            []int32
	}
	side := tw.Cfg.ImageSz / tw.Cfg.PatchSz
	run := func(store KVStore, msgs []ChatMessage, pix []float32, named bool) result {
		st := m.NewState(512)
		defer st.Close()
		st.SetKVStore(store)
		mustKey(t, st, "vision-as-blocks/two-turn")
		var img Image
		if named {
			// A picture span: the prefill encodes it, unless the restore
			// covered it.
			p, err := tw.Cfg.picture(pix, side, side)
			if err != nil {
				t.Fatal(err)
			}
			img = Image{Picture: p}
		} else {
			// Rows with no name: the gap the prefix cache stops at.
			vs, err := st.Vision()
			if err != nil {
				t.Fatal(err)
			}
			if img, err = vs.EncodeImage(pix); err != nil {
				t.Fatal(err)
			}
			img.Key = nil
		}
		spans, err := m.ChatSpansImages(msgs, []Image{img}, true)
		if err != nil {
			t.Fatal(err)
		}
		var r result
		for _, sp := range spans {
			if sp.rows() {
				r.imgEnd = r.imgAt + sp.n(m.Cfg.NEmbd)
				break
			}
			r.imgAt += sp.n(m.Cfg.NEmbd)
		}
		h0, m0 := m.ImageCacheStats()
		lg, err := st.PrefillCachedMixed(spans...)
		if err != nil {
			t.Fatal(err)
		}
		h1, m1 := m.ImageCacheStats()
		r.lookups = h1 - h0 + m1 - m0
		if st.visState != nil {
			r.blocks = st.visState.vis.hostBlocks + st.visState.vis.devBlocks
		}
		r.restored, r.n = st.KVRestored(), SpanPositions(spans, m.Cfg.NEmbd)
		for range 8 {
			id := Greedy(lg)
			r.toks = append(r.toks, id)
			if lg, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return r
	}
	cold := run(NewMemStore(), turn2, px, true)
	store := NewMemStore()
	t1 := run(store, turn1, px, true)
	t2 := run(store, turn2, px, true)
	// Every whole chunk of turn 1's prompt is turn 2's too, and is found: a
	// partial last chunk is sealed by its fill and found only by a prompt that
	// ends there, as for text.
	if want := t1.n / kvChunk * kvChunk; t2.restored < want || t2.restored <= t1.imgEnd {
		t.Fatalf("turn 2 restored %d positions of turn 1's %d (want %d, past the picture at %d..%d)",
			t2.restored, t1.n, want, t1.imgAt, t1.imgEnd)
	}
	if !slices.Equal(t2.toks, cold.toks) {
		t.Fatalf("the restored turn decodes %v, a cold one %v", t2.toks, cold.toks)
	}
	if t2.lookups != 0 || t2.blocks != 0 {
		t.Fatalf("turn 2 restored the picture and still asked the image cache %d time(s) and ran %d tower "+
			"block(s)", t2.lookups, t2.blocks)
	}
	if cold.blocks != int64(tw.Cfg.NLayer) && cold.lookups == 0 {
		t.Fatalf("the cold turn neither encoded the picture nor asked the image cache")
	}
	diff := run(store, turn2, other, true)
	if diff.restored > diff.imgAt {
		t.Fatalf("a different picture restored %d positions, past its start at %d", diff.restored, diff.imgAt)
	}
	unnamed := run(store, turn2, px, false)
	if unnamed.restored > unnamed.imgAt {
		t.Fatalf("unnamed rows restored %d positions, past the picture's start at %d", unnamed.restored, unnamed.imgAt)
	}
	t.Logf("turn 1 %d positions (picture %d..%d); turn 2 restored %d of %d with no tower block and no image-cache "+
		"lookup, tokens %v = cold; another picture restored %d, unnamed rows %d", t1.n, t1.imgAt, t1.imgEnd,
		t2.restored, t2.n, t2.toks, diff.restored, unnamed.restored)
}

// TestImageEncodeStepsBesideDecode: a picture encoded in StepRuns -- in one
// step, and spread over several by blocks -- beside two sessions decoding
// gives the rows its encode alone gives, and leaves each session's logits
// those it has alone, bit for bit on the host and on each device.
//
// Untuned: on arm64 a timed shape picks its matvec over its first calls, so a
// session alone and the same session beside a picture would run two kernels.
func TestImageEncodeStepsBesideDecode(t *testing.T) {
	m, tw := openTower(t, noTune)
	defer m.Close()
	px := rainbow(tw.Cfg.ImageSz)
	prompt := m.Vocab.Encode("The capital of France is", true)
	const steps = 4
	type arm struct {
		logits [2][][]float32
		rows   []float32
		ran    int
	}
	run := func(t *testing.T, g nn.Device, joint bool, blocks int) arm {
		var a arm
		var sts [2]*State
		for i := range sts {
			sts[i] = m.NewState(64)
			defer sts[i].Close()
			if g != nil {
				if err := sts[i].SetDevice(g); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := sts[i].Prefill(prompt); err != nil {
				t.Fatal(err)
			}
		}
		pic := m.NewState(64)
		defer pic.Close()
		if g != nil {
			if err := pic.SetDevice(g); err != nil {
				t.Fatal(err)
			}
		}
		vs, err := pic.Vision()
		if err != nil {
			t.Fatal(err)
		}
		tok := [2]int32{prompt[len(prompt)-1], prompt[len(prompt)-2]}
		if !joint {
			out, err := vs.Encode(px)
			if err != nil {
				t.Fatal(err)
			}
			a.rows = append([]float32(nil), out...)
		}
		side := tw.Cfg.ImageSz / tw.Cfg.PatchSz
		picture, err := tw.Cfg.picture(px, side, side)
		if err != nil {
			t.Fatal(err)
		}
		enc := &Encode{Picture: picture, Blocks: blocks}
		for range steps {
			runs := []Run{{State: sts[0], Tokens: tok[:1], Logits: true}, {State: sts[1], Tokens: tok[1:], Logits: true}}
			if joint && !enc.Done {
				runs = append(runs, Run{State: pic, Encode: enc})
			}
			out, err := StepRuns(runs)
			if err != nil {
				t.Fatal(err)
			}
			for i := range 2 {
				a.logits[i] = append(a.logits[i], append([]float32(nil), out[i]...))
				tok[i] = Greedy(out[i])
			}
		}
		for joint && !enc.Done {
			if _, err := StepRuns([]Run{{State: pic, Encode: enc}}); err != nil {
				t.Fatal(err)
			}
		}
		if joint {
			a.rows, a.ran = enc.Image.Embd, enc.Ran
		}
		return a
	}
	check := func(t *testing.T, g nn.Device) {
		alone := run(t, g, false, 0)
		for _, blocks := range []int{0, 5} {
			// Each arm encodes the picture: the image cache would answer the
			// second from the first.
			m.imgCache.clearForTest()
			got := run(t, g, true, blocks)
			if got.ran != tw.Cfg.NLayer {
				t.Fatalf("blocks %d: the stepped encode ran %d of %d tower blocks", blocks, got.ran, tw.Cfg.NLayer)
			}
			if !slices.Equal(got.rows, alone.rows) {
				nm, _ := nmseOf(got.rows, alone.rows)
				t.Fatalf("blocks %d: the picture's rows encoded in the steps differ from its encode alone (NMSE %.3e)",
					blocks, nm)
			}
			for i := range 2 {
				for k := range got.logits[i] {
					if !slices.Equal(got.logits[i][k], alone.logits[i][k]) {
						t.Fatalf("blocks %d: session %d's logits at step %d moved with a picture encoded beside it",
							blocks, i, k)
					}
				}
			}
			t.Logf("blocks a step %d: the picture's %d rows and both sessions' %d steps bit-identical to alone",
				blocks, len(got.rows)/tw.Cfg.ProjDim, steps)
		}
	}
	t.Run("host", func(t *testing.T) { check(t, nil) })
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			g, err := tier.OpenWith(tier.WithDevices(spec))
			if err != nil || g == nil {
				t.Skipf("%s: not present (%v)", spec, err)
			}
			defer g.Close()
			check(t, g)
		})
	}
}

// TestTowerEncodeResumesAcrossARelocation: an encode stopped after k blocks
// on a device, its blocks brought home, and finished on the host in a later
// step gives the bits of an encode placed that way from the start
// (WithTowerGPULayers k) -- under a page budget below the tower, so every
// host block is also evicted and read again. A transformer tower (SmolVLM's)
// and a convolutional one (Gemma 3n's MobileNet-V5, whose activation changes
// shape from block to block) both.
func TestTowerEncodeResumesAcrossARelocation(t *testing.T) {
	for _, f := range []struct {
		name string
		open func(t testing.TB, opts ...Option) (*Model, *Tower)
	}{
		{"smolvlm", openTower},
		{"gemma3n", func(t testing.TB, opts ...Option) (*Model, *Tower) {
			m, _ := openGemma3nV(t, opts...)
			return m, m.Tower()
		}},
		{"lfm2", func(t testing.TB, opts ...Option) (*Model, *Tower) {
			m := openD1VL(t, opts...)
			return m, m.Tower()
		}},
	} {
		t.Run(f.name, func(t *testing.T) { towerResumesAcrossARelocation(t, f.open) })
	}
}

func towerResumesAcrossARelocation(t *testing.T, openTower func(t testing.TB, opts ...Option) (*Model, *Tower)) {
	const k = 4
	m0, tw0 := openTower(t)
	base := int(m0.container.H.NBlocks)
	budget := 2 * m0.container.PageBytes(base)
	px := rainbow(tw0.Cfg.ImageSz)
	n := tw0.Cfg.NLayer
	m0.Close()
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			g, err := tier.OpenWith(tier.WithDevices(spec))
			if err != nil || g == nil {
				t.Skipf("%s: not present (%v)", spec, err)
			}
			defer g.Close()
			// From the start: the first k blocks on the device, the rest home.
			ma, twa := openTower(t, WithTowerGPULayers(k), WithPageBudget(budget))
			sa := twa.testState()
			if err := sa.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if sa.GPUBlocks() != k {
				t.Fatalf("WithTowerGPULayers(%d) placed %d blocks", k, sa.GPUBlocks())
			}
			want, err := sa.Encode(px)
			if err != nil {
				t.Fatal(err)
			}
			want = append([]float32(nil), want...)
			sa.Close()
			ma.Close()

			// Relocated in the middle: every block on the device, k of them
			// run, the device detached, the rest run at home.
			mb, twb := openTower(t, WithPageBudget(budget))
			defer mb.Close()
			sb := twb.testState()
			defer sb.Close()
			if err := sb.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if sb.GPUBlocks() <= k {
				t.Skipf("%s placed %d tower blocks: the relocation needs more than %d", spec, sb.GPUBlocks(), k)
			}
			side := twb.Cfg.ImageSz / twb.Cfg.PatchSz
			if err := sb.encodeBegin(px, side, side); err != nil {
				t.Fatal(err)
			}
			if done, err := sb.encodeBlocks(k); err != nil || done {
				t.Fatalf("the first step: done %v, %v", done, err)
			}
			if err := sb.SetDevice(nil); err != nil {
				t.Fatal(err)
			}
			if sb.GPUBlocks() != 0 {
				t.Fatalf("%d blocks stayed on the device after it was detached", sb.GPUBlocks())
			}
			if done, err := sb.encodeBlocks(0); err != nil || !done {
				t.Fatalf("the second step: done %v, %v", done, err)
			}
			got, err := sb.encodeFinish()
			if err != nil {
				t.Fatal(err)
			}
			if _, ev := mb.container.Faults(); ev == 0 {
				t.Fatal("a budget of two vision pages evicted nothing: paging was not exercised")
			}
			if !slices.Equal(got, want) {
				nm, _ := nmseOf(got, want)
				t.Fatalf("relocated mid-encode the rows differ from the same placement from the start (NMSE %.3e)", nm)
			}
			t.Logf("%s: %d of %d blocks on the device, then home mid-encode, under a %d-byte budget: bit-identical",
				spec, k, n, budget)
		})
	}
}

// TestWarmImageEncodeDoesNotAllocate: a second encode of a picture on a
// warm vision State makes no engine heap allocation, for every tower family:
// the runner's block body, the entry and the projector programs alike.
func TestWarmImageEncodeDoesNotAllocate(t *testing.T) {
	for _, f := range towerFamilies() {
		t.Run(f.name, func(t *testing.T) {
			m, tw := f.open(t)
			defer m.Close()
			st := m.NewState(64)
			defer st.Close()
			vs, err := st.Vision()
			if err != nil {
				t.Fatal(err)
			}
			px := rainbow(tw.Cfg.ImageSz)
			if _, err := vs.Encode(px); err != nil {
				t.Fatal(err)
			}
			// One measured encode (AllocsPerRun warms with one more): a large
			// tower's encode is seconds on the host.
			allocs := testing.AllocsPerRun(1, func() {
				if _, err := vs.Encode(px); err != nil {
					t.Fatal(err)
				}
			})
			t.Logf("%s: a warm encode makes %.1f allocations", f.name, allocs)
			if allocs != 0 {
				t.Fatalf("%s: a warm encode of a picture made %.1f heap allocations", f.name, allocs)
			}
		})
	}
}

// towerFamily is one vision family's fixture for the gates that run every
// family.
type towerFamily struct {
	name string
	open func(t *testing.T) (*Model, *Tower)
}

// towerFamilies is every projector family on its smallest fixture.
func towerFamilies() []towerFamily {
	return []towerFamily{
		{"smolvlm", func(t *testing.T) (*Model, *Tower) { return openTower(t) }},
		{"llava", func(t *testing.T) (*Model, *Tower) { return openLlava(t) }},
		{"gemma3", func(t *testing.T) (*Model, *Tower) { return openGemma3(t) }},
		{"internvl", func(t *testing.T) (*Model, *Tower) { return openInternVL(t) }},
		{"qwen2vl", func(t *testing.T) (*Model, *Tower) { m, _ := openQVL(t, "synth-qwen2vl"); return m, m.Tower() }},
		{"qwen25vl", func(t *testing.T) (*Model, *Tower) { m, _ := openQVL(t, "synth-qwen25vl"); return m, m.Tower() }},
		{"qwen3vl", func(t *testing.T) (*Model, *Tower) { m, _ := openQVL(t, "synth-qwen3vl"); return m, m.Tower() }},
		{"qwen35vl", func(t *testing.T) (*Model, *Tower) { m, _ := openQVL(t, "synth-qwen35vl"); return m, m.Tower() }},
		{"glm4v", func(t *testing.T) (*Model, *Tower) { m, _ := openQVL(t, "synth-glm4v"); return m, m.Tower() }},
		{"kimivl", func(t *testing.T) (*Model, *Tower) { m, _ := openQVL(t, "synth-kimivl"); return m, m.Tower() }},
		{"minicpmv", func(t *testing.T) (*Model, *Tower) { m := openMiniCPMV(t); return m, m.Tower() }},
		{"janus", func(t *testing.T) (*Model, *Tower) { m := openJanus(t); return m, m.Tower() }},
		{"pixtral", func(t *testing.T) (*Model, *Tower) { m, _ := openPixtral(t); return m, m.Tower() }},
		{"llama4", func(t *testing.T) (*Model, *Tower) { m, _ := openLlama4V(t); return m, m.Tower() }},
		{"gemma4", func(t *testing.T) (*Model, *Tower) { m, _ := openGemma4V(t); return m, m.Tower() }},
		{"phi4", func(t *testing.T) (*Model, *Tower) { m, _ := openPhi4VFixture(t); return m, m.Tower() }},
		{"gemma3n", func(t *testing.T) (*Model, *Tower) { m, _ := openGemma3nV(t); return m, m.Tower() }},
		{"lfm2", func(t *testing.T) (*Model, *Tower) { m := openD1VL(t); return m, m.Tower() }},
	}
}

// clearForTest empties the image cache, so the next encode of a picture runs
// the tower.
func (c *imageCache) clearForTest() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries, c.bytes = nil, 0
	c.lru.Init()
}
