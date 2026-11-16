package engine

import (
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// pastTheRows is g with its rows' history (history 0) kept as history 1
// instead: one entry a position, its values its own, read and written by
// every op that read or wrote the rows' -- another block's rows' read as
// that block's history 1.
func pastTheRows(g *graph.Graph) *graph.Graph {
	c := *g
	c.Ops = slices.Clone(g.Ops)
	move := func(rs []graph.StateRef) []graph.StateRef {
		rs = slices.Clone(rs)
		for i := range rs {
			if rs[i].State == graph.StateKV && rs[i].Hist == 0 {
				rs[i].Hist = 1
			}
		}
		return rs
	}
	for i := range c.Ops {
		c.Ops[i].Reads, c.Ops[i].Writes = move(c.Ops[i].Reads), move(c.Ops[i].Writes)
	}
	return &c
}

// TestAHistoryPastTheRowsHoldsItsValues holds a history past a block's
// rows' whose values are its own, an entry a position, to the answer of the
// rows' history it restates: a model whose every block keeps its attention
// history as history 1 -- and whose blocks reading another's read that
// block's history 1 -- answers as the model does, on the CPU, on a card with
// every block placed, and after the blocks come home mid-decode with that
// history's keys and values.
func TestAHistoryPastTheRowsHoldsItsValues(t *testing.T) {
	for _, c := range []struct {
		name    string
		readers bool
	}{{"synth-olmo2-gqa", false}, {"synth-gemma3n", true}} {
		t.Run(c.name, func(t *testing.T) { historyPastTheRowsHoldsItsValues(t, c.name, c.readers) })
	}
}

func historyPastTheRowsHoldsItsValues(t *testing.T, name string, readers bool) {
	p := testmodels.Path(name + ".gguf")
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	open := func(moved bool) *model.Model {
		m, err := model.Open(jlmOf(t, p))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		if moved {
			f := m.FrontGraphs()
			for li := range f.Blocks {
				f.Blocks[li] = pastTheRows(f.Blocks[li])
			}
		}
		return m
	}
	base, mv := open(false), open(true)
	kept, read := 0, 0
	for li := 0; li < mv.Cfg.NLayer; li++ {
		g := mv.Composing().BlockGraph(li)
		for _, k := range g.KeptHistories() {
			if k.ValuesInKeys || k.Period != 1 || k.Hist != 1 {
				t.Fatalf("block %d keeps %+v: the fixture states history 1, values of their own, an entry a position", li, k)
			}
			kept++
		}
		if n, err := g.ReadsHistoryOf(); err != nil {
			t.Fatal(err)
		} else if n >= 0 {
			read++
		}
	}
	if kept+read != mv.Cfg.NLayer || readers != (read > 0) {
		t.Fatalf("%d of %d blocks keep a history past their rows', %d read another's: the gate selected nothing", kept, mv.Cfg.NLayer, read)
	}
	prompt := base.Vocab.Encode(strings.Repeat("The clerk counted barrels of salt on the upper floor. ", 8), true)[:90]
	const steps = 40
	maxSeq := len(prompt) + steps + 1
	// Every arm is fed the host's greedy tokens.
	var ids []int32
	run := func(t *testing.T, m *model.Model, g *gpu.GPU, home int) [][]float32 {
		st := testInstance(t, m, maxSeq, WithKVType(model.KVF32))
		t.Cleanup(func() { st.Close() })
		if g != nil {
			if err := st.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if n := st.devCount(); n != m.Cfg.NLayer {
				t.Skipf("CARD TOO SMALL: %d of %d blocks placed (%s) -- this gate proved nothing here", n, m.Cfg.NLayer, g.Err())
			}
		}
		lg, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		out := [][]float32{slices.Clone(lg)}
		for i := range steps {
			if i == home {
				if n := st.SetGPULayers(0); n != 0 || st.devCount() != 0 {
					t.Fatalf("step %d: %d block(s) stayed on the card (%s)", i, st.devCount(), g.Err())
				}
			}
			if len(ids) < steps {
				ids = append(ids, Greedy(lg))
			}
			if lg, err = st.Forward(ids[i]); err != nil {
				t.Fatalf("step %d: %v", i, err)
			}
			out = append(out, slices.Clone(lg))
		}
		if g == nil && m == mv {
			wroteHistory1(t, st)
		}
		return out
	}
	near := func(t *testing.T, what string, got, want [][]float32, bound float64) {
		t.Helper()
		worst, at := 0.0, 0
		for i := range want {
			if e := logitNMSE(got[i], want[i]); !(e <= worst) {
				worst, at = e, i
			}
		}
		t.Logf("%s: worst logit NMSE %.3e at step %d", what, worst, at)
		if !(worst <= bound) {
			t.Errorf("%s: logit NMSE %.3e at step %d, past %.0e", what, worst, at, bound)
		}
	}
	host := run(t, base, nil, -1)
	cpu := run(t, mv, nil, -1)
	near(t, "the CPU, history 1 against the rows'", cpu, host, 1e-8)
	t.Run("card", func(t *testing.T) {
		card := func() *gpu.GPU {
			g, err := openGPU(t, gpu.WithDevices(testdevices.Like(t, "cuda:0")), gpu.WithDeviceTune(gpu.TuneOff),
				gpu.WithConfig(func(c *gpu.Config) { c.KVPage = 64 }))
			if err != nil || g == nil {
				noDevice(t, "device", err)
			}
			t.Cleanup(g.Close)
			return g
		}
		rows := run(t, base, card(), -1)
		on := run(t, mv, card(), -1)
		near(t, "the card, history 1 against the rows'", on, rows, 1e-4)
		near(t, "the card against the CPU", on, cpu, evictNMSE)
		home := run(t, mv, card(), steps/2)
		near(t, "home mid-decode against the card", home[:steps/2+1], on[:steps/2+1], 1e-4)
		near(t, "home mid-decode against the CPU", home, cpu, evictNMSE)
	})
}

// wroteHistory1 says each block of st keeping a history writes its keys
// and values into history 1 and nothing into its rows': the arm ran the
// history past the rows', not the rows' it restates.
func wroteHistory1(t *testing.T, st *Instance) {
	t.Helper()
	nonzero := func(v []float32) bool { return slices.ContainsFunc(v, func(x float32) bool { return x != 0 }) }
	for li := st.lo; li < st.hi; li++ {
		if st.readsOther(li) {
			continue
		}
		e := st.kv.histSlot(li, 1)
		if e == 0 {
			t.Fatalf("block %d holds no history 1", li)
		}
		pl, rows := st.kv.layers[e].mem, st.kv.layers[li].mem
		if pl.Len() == 0 || !nonzero(pl.K(0)) || !nonzero(pl.V(0)) {
			t.Fatalf("block %d's history 1 holds no keys or no values", li)
		}
		for n := range rows.Len() {
			if nonzero(rows.K(n)) || nonzero(rows.V(n)) {
				t.Fatalf("block %d wrote its rows' history page %d", li, n)
			}
		}
	}
}
