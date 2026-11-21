package engine

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// twoStateSuffix names a hybrid fixture whose first recurrent block is
// restated to keep a longer state slot than the others: the scan's slot
// holds twoStatePad floats past the rule's (graph.OpStateScan's Span), which
// nothing reads or writes, so the model computes what its source does over
// recurrent state of two sizes a sequence.
const twoStateSuffix = "@twostate"

// twoWindowSuffix names one whose first block keeping a window is restated
// to convolve every second position (graph.OpStateConv's Dil): its window is
// twice the others', and every float of it is read, so a block's window
// staged at another block's size is a different answer.
const twoWindowSuffix = "@twowindow"

const twoStatePad = 48

// restatedFixture is the source a name with twoStateSuffix or
// twoWindowSuffix restates and the suffix, false for any other name.
func restatedFixture(path string) (src, suffix string, ok bool) {
	for _, sf := range []string{twoStateSuffix, twoWindowSuffix} {
		if s, ok := strings.CutSuffix(path, sf); ok {
			return s, sf, true
		}
	}
	return "", "", false
}

// convertTwoState is convertPair for a name restatedFixture names: the
// source's container, its first recurrent block's graph restated
// (twoStateGraph), at <name>-twostate.jlm or <name>-twowindow.jlm.
func convertTwoState(src, suffix string) (string, error) {
	from, err := convertPair(src, "")
	if err != nil {
		return "", err
	}
	dst := strings.TrimSuffix(src, filepath.Ext(src)) + "-" + strings.TrimPrefix(suffix, "@") + jlm.Ext
	fi, err := os.Stat(from)
	if err != nil {
		return "", err
	}
	if writtenByThisBuild(dst, fi.ModTime()) {
		return dst, nil
	}
	enc, li, err := twoStateGraph(from, suffix == twoWindowSuffix)
	if err != nil {
		return "", err
	}
	tmp, err := servedName(dst)
	if err != nil {
		return "", err
	}
	b, err := os.ReadFile(from)
	if err == nil {
		err = os.WriteFile(tmp, b, 0o644)
	}
	if err == nil {
		_, err = jlm.Restate(tmp, func(f *jlm.File, c *jlm.Config) (bool, error) {
			c.Blocks.Graphs = append(slices.Clone(c.Blocks.Graphs), string(enc))
			c.Blocks.Of = slices.Clone(c.Blocks.Of)
			c.Blocks.Of[li] = uint16(len(c.Blocks.Graphs) - 1)
			return true, nil
		})
	}
	if err != nil {
		os.Remove(tmp)
		return "", err
	}
	return place(tmp, dst)
}

// twoStateGraph is the first block of the container at path that keeps a
// recurrent state, its scan's slot lengthened by twoStatePad floats or, with
// window, its convolution dilated by two, encoded, and the block's index.
func twoStateGraph(path string, window bool) ([]byte, int, error) {
	m, err := model.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer m.Close()
	g := m.Composing()
	for li := range m.Cfg.NLayer {
		bg := g.BlockGraph(li)
		conv, state := bg.RecurrentState()
		if window && conv == 0 || !window && state == 0 {
			continue
		}
		out := cloned(bg)
		for i := range out.Ops {
			switch o := &out.Ops[i]; {
			case window && o.Kind == graph.OpStateConv && !(len(o.Reads) == 1 && o.Reads[0].State == graph.StateRecState):
				o.Attr.Dil = 2
			case !window && o.Kind == graph.OpStateScan:
				o.Attr.Span = state + twoStatePad
			}
		}
		if c, s := out.RecurrentState(); c == conv && s == state {
			return nil, 0, fmt.Errorf("%s: block %d's restatement keeps its state's size", path, li)
		}
		enc, err := graph.Encode(out)
		return enc, li, err
	}
	return nil, 0, fmt.Errorf("%s keeps no recurrent state to restate", path)
}

// TestBlocksOfTwoStateSizesRun holds a model whose recurrent blocks keep
// state of two sizes (twoStateSuffix) to the model it restates: nothing
// reads the first block's longer slot past its rule's, so every greedy token
// and every margin is the source's, on the CPU device and with every block
// on each card. The restated container is checked to keep two sizes first,
// so the gate cannot pass on a restatement that reached nothing.
func TestBlocksOfTwoStateSizesRun(t *testing.T) {
	name := "synth-granitehybrid.gguf"
	base := jlmOf(t, testmodels.Path(name))
	two := jlmOf(t, testmodels.Path(name+twoStateSuffix))
	m, err := model.Open(two)
	if err != nil {
		t.Fatal(err)
	}
	sizes := map[int]bool{}
	for li := range m.Cfg.NLayer {
		if c, s := m.Composing().BlockGraph(li).RecurrentState(); c+s > 0 {
			sizes[c+s] = true
		}
	}
	m.Close()
	if len(sizes) < 2 {
		t.Fatalf("%s keeps recurrent state of %d size(s): the restatement reached nothing", two, len(sizes))
	}
	const prompt = "The capital of France is"
	alike := func(t *testing.T, want, got []int32, wm, gm []float32) {
		t.Helper()
		if !slices.Equal(got, want) {
			t.Fatalf("two state sizes generate %v, the source %v", got, want)
		}
		for i := range wm {
			if gm[i] != wm[i] {
				t.Fatalf("step %d's margin is %g with two state sizes, %g in the source", i, gm[i], wm[i])
			}
		}
		t.Logf("%d tokens alike over %d recurrent state sizes", len(got), len(sizes))
	}
	t.Run("cpu", func(t *testing.T) {
		want, wm, _ := greedy(t, base, prompt, 12)
		got, gm, _ := greedy(t, two, prompt, 12)
		alike(t, want, got, wm, gm)
	})
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			want, wm := greedyOnCard(t, base, spec, prompt, 12)
			got, gm := greedyOnCard(t, two, spec, prompt, 12)
			alike(t, want, got, wm, gm)
		})
	}
}

// greedyOnCard is greedy with every block of the container at path on the
// card spec names, its tuning off so both arms reduce in one order.
func greedyOnCard(t *testing.T, path, spec, prompt string, n int) (ids []int32, margins []float32) {
	t.Helper()
	g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, spec, err)
	}
	defer g.Close()
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	in := m.Vocab.Encode(prompt, true)
	st := testInstance(t, m, len(in)+n+1)
	defer st.Close()
	st.SetDeviceLayers(g, m.Cfg.NLayer)
	if st.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("%s placed %d of %d blocks: %s", spec, st.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	logits, err := st.Prefill(in)
	if err != nil {
		t.Fatal(err)
	}
	for range n {
		best, bv, second := int32(0), float32(-1e30), float32(-1e30)
		for j, v := range logits {
			if v > bv {
				best, bv, second = int32(j), v, bv
			} else if v > second {
				second = v
			}
		}
		ids = append(ids, best)
		margins = append(margins, bv-second)
		if logits, err = st.Forward(best); err != nil {
			t.Fatal(err)
		}
	}
	return ids, margins
}
