package engine

import (
	"math"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// unrolled is the edit that states src's n blocks run twice over: blocks
// n..2n-1 a second pass of blocks 0..n-1, each block with its own history.
// pages nil copies the weights into pages of the second pass's own; else
// the second pass reads pages (jlm.Config.WeightsOf) and its blocks store
// nothing.
func unrolled(pages func(n int) []uint32) func(*jlm.Source) {
	return func(s *jlm.Source) {
		c := s.Config
		n := int(c.NLayer)
		order := make([]int, 2*n)
		for i := range order {
			order[i] = i % n
		}
		// Every statement of one entry per layer follows the unrolling.
		v := reflect.ValueOf(c).Elem()
		for i := 0; i < v.NumField(); i++ {
			f := v.Field(i)
			if !v.Type().Field(i).IsExported() || f.Kind() != reflect.Slice || f.Len() != n {
				continue
			}
			out := reflect.MakeSlice(f.Type(), 2*n, 2*n)
			for j, o := range order {
				out.Index(j).Set(f.Index(o))
			}
			f.Set(out)
		}
		c.NLayer = uint32(2 * n)
		if pages != nil {
			c.WeightsOf = pages(n)
			return
		}
		for _, tn := range s.Tensors {
			if tn.Block >= 0 && int(tn.Block) < n && !tn.Role.Vision() {
				tn.Block += int32(n)
				s.Tensors = append(s.Tensors, tn)
			}
		}
	}
}

// TestBlocksShareTheirWeights holds a block reading another block's page
// (jlm.Config.WeightsOf), the unrolled loop's later passes, to the same
// model with the weights copied into pages of their own, on a dense fixture
// and on a mixture whose second pass reads the first pass's expert bank: the
// answer is the copy's bit for bit on the CPU device, and within the device
// band with every block on a card, across a seam that puts the owners (the
// first pass) on the card and the sharing pass on the CPU, with the sharing
// pass placed on the card and its owners on the CPU, and with the sharing
// pass moved there mid-prompt. The sharers going to the card leave the pages
// (and the expert pages) their owners read from host memory where they are:
// the CPU's lanes hold a page once for every block they run from it
// (Instance.holdFrame, by PageOf), so after the move the owners read no
// page again. The shared container holds half the pages, and a run reads each page
// once. A second pass reading its pages in another order is the violation,
// and differs.
func TestBlocksShareTheirWeights(t *testing.T) {
	for _, name := range []string{"synth-olmo2.gguf", "synth-qwen2moe.gguf"} {
		t.Run(name, func(t *testing.T) { blocksShareTheirWeights(t, name) })
	}
}

func blocksShareTheirWeights(t *testing.T, name string) {
	src, ok := existingModel(testmodels.Path(name))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (see docs/testing.md) -- this gate proved nothing", testmodels.Path(name))
	}
	own := func(n int) []uint32 {
		p := make([]uint32, 2*n)
		for i := range p {
			p[i] = uint32(i % n)
		}
		return p
	}
	swapped := func(n int) []uint32 {
		p := own(n)
		for i := n; i < 2*n; i++ {
			p[i] = uint32(2*n - 1 - i)
		}
		return p
	}
	open := func(name string, edit func(*jlm.Source)) *model.Model {
		path := filepath.Join(t.TempDir(), name+model.Ext)
		if err := convertStated(src, path, edit); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		m, err := model.Open(path)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}
	copied := open("copied", unrolled(nil))
	shared := open("shared", unrolled(own))
	wrong := open("swapped", unrolled(swapped))
	// The converter's own statement of the loop (convert.UnrollSource, what a
	// library file's unroll states) reads the pages the hand-made one does.
	stated := open("stated", loopedTwice)

	n := copied.Cfg.NLayer / 2
	mixture := copied.Cfg.NExpert > 0
	if n < 2 {
		t.Fatalf("the fixture has %d blocks; the violation needs two", n)
	}
	if shared.Cfg.NLayer != 2*n || int(shared.Container.H.NBlocks) != n || int(copied.Container.H.NBlocks) != 2*n {
		t.Fatalf("shared: %d blocks over %d pages; copied: %d pages; want %d over %d and %d",
			shared.Cfg.NLayer, shared.Container.H.NBlocks, copied.Container.H.NBlocks, 2*n, n, 2*n)
	}
	if mixture && shared.Container.H.NExpPages*2 != copied.Container.H.NExpPages {
		t.Fatalf("shared: %d expert pages, copied %d: the second pass does not read the first's bank",
			shared.Container.H.NExpPages, copied.Container.H.NExpPages)
	}
	in := shared.Vocab.Encode("The capital of France is", true)
	if len(in) < 3 {
		t.Fatalf("a prompt of %d tokens reads too few positions", len(in))
	}

	cpu := func(m *model.Model) ([]float32, int64, int) {
		st := testInstance(t, m, len(in)+8)
		var out []float32
		var err error
		for _, id := range in {
			if out, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		faults, _ := st.fr.Faults()
		return append([]float32(nil), out...), faults, st.fr.ResidentPages()
	}
	want, cFaults, cPages := cpu(copied)
	got, sFaults, sPages := cpu(shared)
	bad, _, _ := cpu(wrong)
	byStatement, _, _ := cpu(stated)
	read := make([]uint32, 2*n)
	for i := range read {
		read[i] = uint32(stated.Container.PageOf(i))
	}
	if int(stated.Container.H.NBlocks) != int(n) || !slices.Equal(read, own(int(n))) {
		t.Fatalf("the stated loop: %d pages, pages %v; want %d and %v",
			stated.Container.H.NBlocks, read, n, own(int(n)))
	}
	if !slices.Equal(byStatement, got) {
		t.Fatalf("the stated loop's logits differ from the hand-made sharing's by %.3g", maxAbsDiff(byStatement, got))
	}
	if !allFinite(want) {
		t.Fatal("the copied model's logits are not finite")
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("logit %d: shared %v, copied %v", i, got[i], want[i])
		}
	}
	if maxAbsDiff(bad, want) == 0 {
		t.Fatal("the second pass reading its pages swapped answers as the copy does: the gate cannot see which page a block reads")
	}
	// A mixture's expert pages are read as tokens select them, so only its
	// block pages count once each.
	if !mixture && (sPages != int(n) || cPages != 2*int(n) || sFaults != int64(n) || cFaults != 2*int64(n)) {
		t.Fatalf("pages resident %d (copied %d), faulted in %d (copied %d); want %d, %d, %d, %d",
			sPages, cPages, sFaults, cFaults, n, 2*n, n, 2*n)
	}
	if mixture && sFaults >= cFaults {
		t.Fatalf("the shared mixture faulted %d pages in, the copy %d", sFaults, cFaults)
	}
	t.Logf("CPU: equal over %d logits; the violation %.3g away; %d pages resident, %d faults (copied %d, %d)",
		len(want), maxAbsDiff(bad, want), sPages, sFaults, cPages, cFaults)

	dev := testdevices.Take(t, 1, "cuda:0")[0]
	g, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	// card runs m with seam blocks on the card (-1 every block), with the
	// second pass on it and the first on the CPU (sharers), or with every
	// block on the CPU and the second pass moved to the card mid-prompt
	// (moved), its faults then those after the move.
	const sharers, moved = -2, -3
	card := func(m *model.Model, seam int) ([]float32, int64) {
		var opts []Option
		if seam == moved {
			opts = append(opts, WithPlacement(Placement{Rest: &Place{On: nn.CPU}, Strict: true}))
		}
		if seam == sharers {
			up := map[int]Place{}
			for li := int(n); li < 2*int(n); li++ {
				up[li] = Place{On: dev}
			}
			opts = append(opts, WithPlacement(Placement{Blocks: up, Rest: &Place{On: nn.CPU}, Strict: true}))
		}
		st := testInstance(t, m, len(in)+8, opts...)
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if seam >= 0 {
			st.SetGPULayers(seam)
		}
		second := make([]int, n)
		for i := range second {
			second[i] = int(n) + i
		}
		switch placed := st.placedOn(dev); {
		case seam == moved && len(placed) != 0:
			t.Fatalf("blocks %v on the card before the move", placed)
		case seam == moved:
		case seam == sharers:
			if !slices.Equal(placed, second) {
				t.Fatalf("blocks %v on the card, want the second pass's %v", placed, second)
			}
		case seam < 0 && len(placed) != m.Blocks(), seam >= 0 && len(placed) != seam:
			t.Fatalf("%d blocks on the card, want %d", len(placed), max(seam, m.Blocks()))
		}
		var out []float32
		var before int64
		for i, id := range in {
			if seam == moved && i == len(in)/2 {
				mv, err := st.MoveBlocks(int(n), 2*int(n), dev)
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(mv.Moved, second) || !slices.Equal(st.placedOn(dev), second) {
					t.Fatalf("moved %v (declined %v), on the card %v; want the second pass's %v", mv.Moved, mv.Declined, st.placedOn(dev), second)
				}
				// The copy's second pass gives its own pages back; the
				// shared one's pages are the first pass's, which stay.
				for li := 0; li < 2*int(n); li++ {
					held := m.BlockResident(st.fr, li)
					switch {
					case li < int(n) && !held:
						t.Fatalf("after the move block %d, on the CPU, has lost its page %d", li, m.Container.PageOf(li))
					case li >= int(n) && held && m.Container.PageOf(li) == li:
						t.Fatalf("after the move block %d's own page %d is still in host memory: the move released nothing, and the arm proves nothing", li, li)
					}
				}
				before, _ = st.fr.Faults()
			}
			if out, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		faults, _ := st.fr.Faults()
		return append([]float32(nil), out...), faults - before
	}
	band := 0.0
	for _, seam := range []int{-1, int(n), sharers, moved} {
		cw, cf := card(copied, seam)
		cs, sf := card(shared, seam)
		cb, _ := card(wrong, seam)
		if !allFinite(cs) {
			t.Fatalf("seam %d: the shared model's card logits are not finite", seam)
		}
		d, v := maxAbsDiff(cs, cw), maxAbsDiff(cb, cw)
		band = max(band, maxAbsDiff(cw, want))
		if d > 0.05 || v < 10*max(d, 1e-3) {
			t.Fatalf("seam %d: shared %.3g from the copy, the violation %.3g", seam, d, v)
		}
		// The second pass on the card leaves its pages to the first pass,
		// which reads them from host memory: read once. Moved
		// there mid-prompt, the first pass reads nothing more than the copy
		// does: none of a dense model's pages, only the experts a token
		// selects for the first time.
		switch {
		case seam == sharers && !mixture && sf != int64(n):
			t.Fatalf("the second pass on the card: the shared model faulted %d pages in, over %d pages", sf, n)
		case seam == moved && (sf > cf || !mixture && sf != 0):
			t.Fatalf("the second pass moved to the card: the shared model faulted %d pages in after it, the copy %d", sf, cf)
		}
		t.Logf("seam %d: shared %.3g from the copy, the violation %.3g; %d faults (copied %d)", seam, d, v, sf, cf)
	}
	t.Logf("the copy's card band against the CPU: %.3g", band)
}

func allFinite(a []float32) bool {
	for _, v := range a {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return len(a) > 0
}

// TestAReaderWithoutTheLoopRefusesIt holds the pages blocks read
// (jlm.Config.WeightsOf, a config tail a reader before it drops) to what
// such a reader makes of a loop: the container with the tail dropped has
// blocks past its pages, whose graphs name weights no page holds, and Open
// refuses it rather than running them on nothing.
func TestAReaderWithoutTheLoopRefusesIt(t *testing.T) {
	src := jlmOf(t, testmodels.Path("synth-olmo2.gguf"+loopSuffix))
	m, err := model.Open(src)
	if err != nil {
		t.Fatal(err)
	}
	blocks, pages := m.Cfg.NLayer, int(m.Container.H.NBlocks)
	m.Close()
	if blocks <= pages {
		t.Fatalf("the loop runs %d blocks over %d pages: no block shares a page", blocks, pages)
	}
	old := filepath.Join(t.TempDir(), "older"+model.Ext)
	if err := copyContainer(src, old); err != nil {
		t.Fatal(err)
	}
	if _, err := jlm.Restate(old, func(f *jlm.File, c *jlm.Config) (bool, error) {
		c.WeightsOf = nil
		return true, nil
	}); err != nil {
		t.Fatalf("the container with its tail dropped cannot be written, so no reader's open was asked: %v", err)
	}
	if m, err := model.Open(old); err == nil {
		m.Close()
		t.Fatalf("%d blocks over %d pages, the pages they read dropped, opened", blocks, pages)
	} else {
		t.Logf("refused: %v", err)
	}
}
