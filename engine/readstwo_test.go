package engine

import (
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// readsTwoSuffix names a fixture whose last block is restated by readsTwo:
// beside its own history it attends to blocks 0's and 1's, each by (block,
// index), and adds the three attentions. "synth-seedoss.gguf@readstwo" is
// synth-seedoss so restated.
const readsTwoSuffix = "@readstwo"

// readsTwoFixture is the restated fixture the principle gates run.
const readsTwoFixture = "synth-seedoss.gguf" + readsTwoSuffix

// readingTwo is the container of the fixture at path with its last block
// restated by readsTwo, beside the fixture's own container.
func readingTwo(path string) (string, error) {
	return restatedLast(path, "-readstwo", []int{0, 1})
}

// restatedLast is the container of the fixture at path, named with tag,
// whose last block also attends to each history of rows reads names.
func restatedLast(path, tag string, reads []int) (string, error) {
	src, err := convertPair(path, "")
	if err != nil {
		return "", err
	}
	dst := strings.TrimSuffix(src, jlm.Ext) + tag + jlm.Ext
	si, err := os.Stat(src)
	if err != nil {
		return "", err
	}
	if writtenByThisBuild(dst, si.ModTime()) {
		return dst, nil
	}
	m, err := model.Open(src)
	if err != nil {
		return "", err
	}
	li := m.Cfg.NLayer - 1
	g, err := readsTwo(m.Composing().BlockGraph(li), reads)
	m.Close()
	if err != nil {
		return "", err
	}
	b, err := graph.Encode(g)
	if err != nil {
		return "", err
	}
	tmp, err := servedName(dst)
	if err != nil {
		return "", err
	}
	if err := copyContainer(src, tmp); err != nil {
		os.Remove(tmp)
		return "", err
	}
	if _, err := jlm.Restate(tmp, func(f *jlm.File, c *jlm.Config) (bool, error) {
		if li >= len(c.Blocks.Of) {
			return false, fmt.Errorf("%s states %d block graphs, not block %d's", src, len(c.Blocks.Of), li)
		}
		c.Blocks.Graphs = append(slices.Clone(c.Blocks.Graphs), string(b))
		c.Blocks.Of = slices.Clone(c.Blocks.Of)
		c.Blocks.Of[li] = uint16(len(c.Blocks.Graphs) - 1)
		return true, nil
	}); err != nil {
		os.Remove(tmp)
		return "", err
	}
	return place(tmp, dst)
}

// copyContainer copies the container at src to dst.
func copyContainer(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

// readsTwo is g with its attention over its own rows joined by one over
// the rows' history of each block of reads, the same queries, heads and
// scale, the attentions added before anything reads the first.
func readsTwo(g *graph.Graph, reads []int) (*graph.Graph, error) {
	hs := make([]graph.HistoryRead, len(reads))
	for i, n := range reads {
		hs[i] = graph.HistoryRead{Block: n}
	}
	return readsHists(g, hs)
}

// readsHists is readsTwo over each history of reads, by (block, index).
func readsHists(g *graph.Graph, reads []graph.HistoryRead) (*graph.Graph, error) {
	if g == nil {
		return nil, fmt.Errorf("the block states no graph")
	}
	out := cloned(g)
	for i, o := range out.Ops {
		if o.Kind != graph.OpAttend || len(o.Reads) != 1 || o.Reads[0].State != graph.StateKV || o.Reads[0].From != 0 || o.Reads[0].Hist != 0 {
			continue
		}
		y := o.Out[0]
		sum := y
		var ops []graph.Op
		for _, r := range reads {
			n := r.Block
			if n >= g.Block {
				return nil, fmt.Errorf("block %d reads block %d's history, which is not below it", g.Block, n)
			}
			v := out.Values[y]
			v.Name = fmt.Sprintf("%s.of%d.%d", v.Name, n, r.Hist)
			a := graph.ValueID(len(out.Values))
			out.Values = append(out.Values, v)
			ref := graph.SharedHistory(n)
			ref.Hist = uint8(r.Hist)
			ops = append(ops, graph.Op{Kind: graph.OpAttend, In: slices.Clone(o.In), Out: []graph.ValueID{a},
				Weight: slices.Clone(o.Weight), Reads: []graph.StateRef{ref}, Attr: o.Attr})
			v.Name += ".sum"
			s := graph.ValueID(len(out.Values))
			out.Values = append(out.Values, v)
			ops = append(ops, graph.Op{Kind: graph.OpAdd, In: []graph.ValueID{sum, a}, Out: []graph.ValueID{s}})
			sum = s
		}
		redirect(out, y, sum, i+1)
		out.Ops = slices.Insert(out.Ops, i+1, ops...)
		return out, nil
	}
	return nil, fmt.Errorf("block %d states no attention over its own rows", g.Block)
}

// keepsTwice is g with its rows appended a second time as its history h,
// kept beside the rows: the same keys and values, read by no op of g.
func keepsTwice(g *graph.Graph, h uint8) (*graph.Graph, error) {
	out := cloned(g)
	for i, o := range out.Ops {
		if o.Kind != graph.OpKVAppend || len(o.Writes) != 1 || o.Writes[0].State != graph.StateKV || o.Writes[0].From != 0 || o.Writes[0].Hist != 0 {
			continue
		}
		c := o
		c.In = slices.Clone(o.In)
		c.Reads = []graph.StateRef{{State: graph.StateKV, Hist: h}}
		c.Writes = []graph.StateRef{{State: graph.StateKV, Version: 1, Hist: h}}
		out.Ops = slices.Insert(out.Ops, i+1, c)
		if out.Routed != (graph.Region{}) && out.Routed.Lo > i {
			out.Routed.Lo, out.Routed.Hi = out.Routed.Lo+1, out.Routed.Hi+1
		}
		return out, nil
	}
	return nil, fmt.Errorf("block %d appends to no rows of its own", g.Block)
}

// openKeepsTwo opens synth-seedoss with block 0 keeping its rows a second
// time as its history 1 (keepsTwice), block n-2 reading that history and
// block n-1 its rows (readsHists): two readers of one block reading two of
// its histories, the lower one the history past the rows alone.
func openKeepsTwo(t *testing.T, _ string) (*model.Model, *c6Golden, []Option) {
	t.Helper()
	m, err := model.Open(jlmOf(t, testmodels.Path("synth-seedoss.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	nl := m.Cfg.NLayer
	f := m.FrontGraphs()
	k, err := keepsTwice(m.Composing().BlockGraph(0), 1)
	if err != nil {
		m.Close()
		t.Fatal(err)
	}
	lo, err := readsHists(m.Composing().BlockGraph(nl-2), []graph.HistoryRead{{Block: 0, Hist: 1}})
	if err != nil {
		m.Close()
		t.Fatal(err)
	}
	hi, err := readsHists(m.Composing().BlockGraph(nl-1), []graph.HistoryRead{{Block: 0}})
	if err != nil {
		m.Close()
		t.Fatal(err)
	}
	f.Blocks[0], f.Blocks[nl-2], f.Blocks[nl-1] = k, lo, hi
	if kept := m.Composing().BlockGraph(0).KeptHistories(); len(kept) != 1 || kept[0].Hist != 1 {
		m.Close()
		t.Fatalf("block 0 keeps %v past its rows: the restatement selected nothing", kept)
	}
	if got := m.Composing().BlockGraph(nl - 2).ReadsMore(); !slices.Equal(got, []graph.HistoryRead{{Block: 0, Hist: 1}}) {
		m.Close()
		t.Fatalf("block %d reads %v past its own: the restatement selected nothing", nl-2, got)
	}
	return m, nil, []Option{WithKVF16(false)}
}

// readsAnother says a block of m reads another block's history, as its
// graph states it.
func readsAnother(m *model.Model) bool {
	for li := range m.Cfg.NLayer {
		if g := m.Composing().BlockGraph(li); g.ReadsHistoryOf() >= 0 || len(g.ReadsMore()) > 0 {
			return true
		}
	}
	return false
}

// readsOutside is a block of [lo, hi) whose graph reads a history of a
// block outside it, -1 for none.
func readsOutside(m *model.Model, lo, hi int) int {
	out := func(b int) bool { return b < lo || b >= hi }
	for li := lo; li < hi; li++ {
		g := m.Composing().BlockGraph(li)
		if n := g.ReadsHistoryOf(); n >= 0 && out(n) {
			return li
		}
		for _, r := range g.ReadsMore() {
			if out(r.Block) {
				return li
			}
		}
	}
	return -1
}

// TestABlockReadsTwoBlocksHistories holds a block reading the histories of
// two other blocks beside its own, each bound by (block, index): its graph
// selects both (graph.Graph.ReadsMore), each read is load-bearing (the
// fixture reading block 0's twice answers otherwise, and so does the
// fixture unrestated), every block on each card answers as every block on
// the CPU, a binary16 history answers as an f32 one, and each history is
// held once, by the block that keeps it.
func TestABlockReadsTwoBlocksHistories(t *testing.T) {
	base := testmodels.Path("synth-seedoss.gguf")
	two := jlmOf(t, base+readsTwoSuffix)
	twice, err := restatedLast(base, "-readstwice", []int{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	open := func(t *testing.T, path string) *model.Model {
		m, err := model.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}
	runAs := func(t *testing.T, m *model.Model, g *gpu.GPU, f16 bool) ([]float32, *Instance) {
		prompt := make([]int32, 40)
		for i := range prompt {
			prompt[i] = int32((7 + 13*i) % m.Cfg.NVocab)
		}
		s := testInstance(t, m, 96, WithKVF16(f16), withCPUFloat())
		if g != nil {
			if err := s.SetDeviceLayers(g, m.Cfg.NLayer); err != nil {
				t.Fatal(err)
			}
			if s.GPULayers() != m.Cfg.NLayer {
				t.Fatalf("the card took %d of %d blocks: %s", s.GPULayers(), m.Cfg.NLayer, g.Err())
			}
		}
		lg, err := s.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		for range 6 {
			if lg, err = s.Forward(int32(argmax(lg))); err != nil {
				t.Fatal(err)
			}
		}
		return slices.Clone(lg), s
	}
	run := func(t *testing.T, m *model.Model, g *gpu.GPU) ([]float32, *Instance) { return runAs(t, m, g, false) }
	m := open(t, two)
	li := m.Cfg.NLayer - 1
	g := m.Composing().BlockGraph(li)
	if got := g.ReadsMore(); g.ReadsHistoryOf() != -1 || !slices.Equal(got, []graph.HistoryRead{{Block: 0}, {Block: 1}}) {
		t.Fatalf("block %d reads %v past its own: the restatement selected nothing", li, got)
	}
	cpu, s := run(t, m, nil)
	for _, v := range cpu {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("the CPU's logits are not finite")
		}
	}
	for b := range li + 1 {
		if s.kv.layers[b].mem.Len() == 0 {
			t.Errorf("block %d keeps its history and holds no page", b)
		}
	}
	// A binary16 history reads the blocks' histories as their blocks hold
	// them: history 0 in halves, as the reading block binds it.
	half, _ := runAs(t, m, nil, true)
	if nm, _ := nmseOf(half, cpu); math.IsNaN(nm) || nm > 1e-3 {
		t.Errorf("block %d over a binary16 history reads NMSE %.3e from its f32 history's logits", li, nm)
	} else {
		t.Logf("over a binary16 history: NMSE %.3e from the f32 history's logits", nm)
	}
	own, _ := run(t, open(t, jlmOf(t, base)), nil)
	same, _ := run(t, open(t, twice), nil)
	for _, c := range []struct {
		name string
		lg   []float32
	}{{"the fixture unrestated", own}, {"the fixture reading block 0's history twice", same}} {
		d := maxAbsDiff(cpu, c.lg)
		if d < 1e-3 {
			t.Errorf("block %d reading blocks 0's and 1's histories answers as %s (%.2e apart): a read is not load-bearing", li, c.name, d)
		}
		t.Logf("%.3f from %s", d, c.name)
	}
	for _, d := range testdevices.Take(t, 2, "cuda:0,vulkan:1") {
		t.Run(d, func(t *testing.T) {
			gd, err := openGPU(t, gpu.WithDevices(d), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil {
				noDevice(t, d, err)
			}
			defer gd.Close()
			card, _ := run(t, m, gd)
			diff := maxAbsDiff(cpu, card)
			if diff > 0.5 {
				t.Errorf("every block on %s answers %.3f from the CPU", d, diff)
			}
			t.Logf("every block on %s: %.2e from the CPU", d, diff)
			if argmax(card) != argmax(cpu) {
				t.Errorf("every block on %s draws %d, the CPU %d", d, argmax(card), argmax(cpu))
			}
		})
	}
}
