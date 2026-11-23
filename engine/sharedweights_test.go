package engine

import (
	"math"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
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
// model with the weights copied into pages of their own: the answer is the
// copy's bit for bit on the CPU device, and within the device band with
// every block on a card and across a seam that puts the sharing pass on the
// card and its owners on the CPU. The shared container holds half the
// pages, and a run reads each page once. A second pass reading its pages
// in another order is the violation, and differs.
func TestBlocksShareTheirWeights(t *testing.T) {
	src, ok := existingModel(testmodels.Path("synth-olmo2.gguf"))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (see docs/testing.md) -- this gate proved nothing", testmodels.Path("synth-olmo2.gguf"))
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
	stated := open("stated", func(s *jlm.Source) {
		order := make([]int, 2*s.Config.NLayer)
		for i := range order {
			order[i] = i % int(s.Config.NLayer)
		}
		if err := convert.UnrollSource(s, order); err != nil {
			t.Fatal(err)
		}
	})

	n := copied.Cfg.NLayer / 2
	if n < 2 {
		t.Fatalf("the fixture has %d blocks; the violation needs two", n)
	}
	if shared.Cfg.NLayer != 2*n || int(shared.Container.H.NBlocks) != n || int(copied.Container.H.NBlocks) != 2*n {
		t.Fatalf("shared: %d blocks over %d pages; copied: %d pages; want %d over %d and %d",
			shared.Cfg.NLayer, shared.Container.H.NBlocks, copied.Container.H.NBlocks, 2*n, n, 2*n)
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
	byStatement, _, stPages := cpu(stated)
	read := make([]uint32, 2*n)
	for i := range read {
		read[i] = uint32(stated.Container.PageOf(i))
	}
	if int(stated.Container.H.NBlocks) != int(n) || stPages != int(n) || !slices.Equal(read, own(int(n))) {
		t.Fatalf("the stated loop: %d pages, %d resident, pages %v; want %d and %v",
			stated.Container.H.NBlocks, stPages, read, n, own(int(n)))
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
	if sPages != int(n) || cPages != 2*int(n) || sFaults != int64(n) || cFaults != 2*int64(n) {
		t.Fatalf("pages resident %d (copied %d), faulted in %d (copied %d); want %d, %d, %d, %d",
			sPages, cPages, sFaults, cFaults, n, 2*n, n, 2*n)
	}
	t.Logf("CPU: equal over %d logits; the violation %.3g away; %d pages read once", len(want), maxAbsDiff(bad, want), sPages)

	g, err := openGPU(t, gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	card := func(m *model.Model, seam int) []float32 {
		st := testInstance(t, m, len(in)+8)
		st.SetDevice(g)
		if seam >= 0 {
			st.SetGPULayers(seam)
		}
		placed := st.GPULayers()
		if seam < 0 && placed != m.Blocks() || seam >= 0 && placed != seam {
			t.Fatalf("%d blocks on the card, want %d", placed, max(seam, m.Blocks()))
		}
		var out []float32
		for _, id := range in {
			if out, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return append([]float32(nil), out...)
	}
	band := 0.0
	for _, seam := range []int{-1, int(n)} {
		cw, cs, cb := card(copied, seam), card(shared, seam), card(wrong, seam)
		if !allFinite(cs) {
			t.Fatalf("seam %d: the shared model's card logits are not finite", seam)
		}
		d, v := maxAbsDiff(cs, cw), maxAbsDiff(cb, cw)
		band = max(band, maxAbsDiff(cw, want))
		if d > 0.05 || v < 10*max(d, 1e-3) {
			t.Fatalf("seam %d: shared %.3g from the copy, the violation %.3g", seam, d, v)
		}
		t.Logf("seam %d: shared %.3g from the copy, the violation %.3g", seam, d, v)
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
