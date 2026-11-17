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

// pastTheRows is g with its rows' history (history 0) kept as history h
// instead: one entry a position, its values its own, read and written by
// every op that read or wrote the rows' -- another block's rows' read as
// that block's history h.
func pastTheRows(g *graph.Graph, h uint8) *graph.Graph {
	c := *g
	c.Ops = slices.Clone(g.Ops)
	move := func(rs []graph.StateRef) []graph.StateRef {
		rs = slices.Clone(rs)
		for i := range rs {
			if rs[i].State == graph.StateKV && rs[i].Hist == 0 {
				rs[i].Hist = h
			}
		}
		return rs
	}
	for i := range c.Ops {
		c.Ops[i].Reads, c.Ops[i].Writes = move(c.Ops[i].Reads), move(c.Ops[i].Writes)
	}
	return &c
}

// valuesOfTheirOwn is g with each history past its rows whose values are
// its keys appending its values as a copy of them (a whole-row OpSlice):
// the same history, held keys and values.
func valuesOfTheirOwn(g *graph.Graph) *graph.Graph {
	c := *g
	c.Values = slices.Clone(g.Values)
	c.Ops = nil
	for i, o := range g.Ops {
		if o.Kind == graph.OpKVAppend && len(o.Writes) == 1 && o.Writes[0].Hist > 0 && len(o.In) == 2 && o.In[0] == o.In[1] {
			v := c.Values[o.In[0]]
			v.Name += ".values"
			c.Values = append(c.Values, v)
			id := graph.ValueID(len(c.Values) - 1)
			c.Ops = append(c.Ops, graph.Op{Kind: graph.OpSlice, In: []graph.ValueID{o.In[0]}, Out: []graph.ValueID{id}, Attr: graph.Attrs{Width: v.Width}})
			o.In = []graph.ValueID{o.In[0], id}
			if c.Routed != (graph.Region{}) && c.Routed.Lo > i {
				c.Routed.Lo, c.Routed.Hi = c.Routed.Lo+1, c.Routed.Hi+1
			}
		}
		c.Ops = append(c.Ops, o)
	}
	return &c
}

// valuesCase is one model of TestAHistoryPastTheRowsHoldsItsValues: the
// fixture, how its blocks are restated, the history its rows' moved to (0
// where they stay) and whether some of its blocks read another's.
type valuesCase struct {
	model   string
	restate func(*graph.Graph) *graph.Graph
	hist    int
	readers bool
}

// TestAHistoryPastTheRowsHoldsItsValues holds a history past a block's
// rows' whose values are its own to the answer of the history it restates:
// a model whose every block keeps its attention history as history 1 (or
// 2, history 1 kept by none) -- and whose blocks reading another's read
// that block's -- answers as the model does, and a model whose histories of
// an entry each several positions hold their values beside their keys
// answers as the one holding them in their keys. Each on the CPU, with its
// history paged out and faulted back, with a warm token allocating
// nothing; on a card with every block placed and on a partial seam, after
// the blocks come home mid-decode, with the card's pages evicted and
// brought home, and moved between two cards.
func TestAHistoryPastTheRowsHoldsItsValues(t *testing.T) {
	rows := func(h uint8) func(*graph.Graph) *graph.Graph {
		return func(g *graph.Graph) *graph.Graph { return pastTheRows(g, h) }
	}
	for _, c := range []struct {
		name string
		valuesCase
	}{
		{"synth-olmo2-gqa", valuesCase{"synth-olmo2-gqa", rows(1), 1, false}},
		{"synth-gemma3n", valuesCase{"synth-gemma3n", rows(1), 1, true}},
		{"synth-gemma3n-hist2", valuesCase{"synth-gemma3n", rows(2), 2, true}},
		{"synth-deepseek4", valuesCase{"synth-deepseek4", valuesOfTheirOwn, 0, false}},
		{"synth-deepseek4-hca", valuesCase{"synth-deepseek4-hca", valuesOfTheirOwn, 0, false}},
	} {
		t.Run(c.name, func(t *testing.T) { historyPastTheRowsHoldsItsValues(t, c.valuesCase) })
	}
}

func historyPastTheRowsHoldsItsValues(t *testing.T, c valuesCase) {
	p := testmodels.Path(c.model + ".gguf")
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	path := jlmOf(t, p)
	openAs := func(moved bool) (*model.Model, error) {
		m, err := model.Open(path)
		if err != nil || !moved {
			return m, err
		}
		f := m.FrontGraphs()
		for li := range f.Blocks {
			f.Blocks[li] = c.restate(f.Blocks[li])
		}
		return m, nil
	}
	open := func(t *testing.T, moved bool) *model.Model {
		m, err := openAs(moved)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}
	base, mv := open(t, false), open(t, true)
	keptValues(t, mv, c)
	prompt := base.Vocab.Encode(strings.Repeat("The clerk counted barrels of salt on the upper floor. ", 8), true)[:90]
	const steps = 40
	maxSeq := len(prompt) + steps + 1
	// Every arm is fed the host's greedy tokens.
	var ids []int32
	run := func(t *testing.T, m *model.Model, g *gpu.GPU, seam, home int) [][]float32 {
		st := testInstance(t, m, maxSeq, WithKVType(model.KVF32))
		t.Cleanup(func() { st.Close() })
		if g != nil {
			if err := st.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if n := st.devCount(); n != m.Cfg.NLayer {
				t.Fatalf("%d of %d blocks placed (%s): a fixture this small fits any card", n, m.Cfg.NLayer, g.Err())
			}
			if seam > 0 {
				if n := st.SetGPULayers(seam); n != seam || st.devCount() != seam {
					t.Fatalf("a seam at %d placed %d block(s) (%s)", seam, st.devCount(), g.Err())
				}
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
			wroteValues(t, st, m, c.hist)
		}
		return out
	}
	host := run(t, base, nil, 0, -1)
	cpu := run(t, mv, nil, 0, -1)
	valuesNear(t, "the CPU, the restated history against the model's", cpu, host, 1e-8)
	t.Run("paged", func(t *testing.T) { valuesPaged(t, base, mv) })
	t.Run("allocs", func(t *testing.T) {
		m := open(t, true)
		decodeAllocs(t, m, nil, WithPageBudget(1<<40))
	})
	card := func(t *testing.T) *gpu.GPU {
		g, err := openGPU(t, gpu.WithDevices(testdevices.Like(t, "cuda:0")), gpu.WithDeviceTune(gpu.TuneOff),
			gpu.WithConfig(func(c *gpu.Config) { c.KVPage = 64 }))
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		t.Cleanup(g.Close)
		return g
	}
	t.Run("card", func(t *testing.T) {
		rows := run(t, base, card(t), 0, -1)
		on := run(t, mv, card(t), 0, -1)
		valuesNear(t, "the card, the restated history against the model's", on, rows, 1e-4)
		valuesNear(t, "the card against the CPU", on, cpu, evictNMSE)
		home := run(t, mv, card(t), 0, steps/2)
		valuesNear(t, "home mid-decode against the card", home[:steps/2+1], on[:steps/2+1], 1e-4)
		valuesNear(t, "home mid-decode against the CPU", home, cpu, evictNMSE)
		if n := mv.Cfg.NLayer / 2; n > 0 {
			seam := run(t, mv, card(t), n, -1)
			valuesNear(t, "a partial seam against the CPU", seam, cpu, evictNMSE)
		}
	})
	t.Run("card/allocs", func(t *testing.T) {
		g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(testdevices.Like(t, "cuda:0"))}, testTierOpts(t)...)...)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		t.Cleanup(g.Close)
		decodeAllocs(t, open(t, true), g, WithStreamTrial(false), WithPageBudget(1<<40))
	})
	t.Run("card/evicted", func(t *testing.T) {
		eg := evictGateOn(t, open(t, true))
		t.Run("released", func(t *testing.T) { evictedHistoryComesHome(t, eg, nil, nil) })
		t.Run("evicted", func(t *testing.T) {
			free := eg.run(t, nil, nil)
			evictedHistoryComesHome(t, eg, &free.squeeze, free.logits)
		})
	})
	mvOpen := func() (*model.Model, error) { return openAs(true) }
	t.Run("two-cards", func(t *testing.T) { blocksMoveBetweenDevicesOf(t, mvOpen, moveWhole, composedParts, nil) })
	t.Run("two-cards/upper", func(t *testing.T) { blocksMoveBetweenDevicesOf(t, mvOpen, moveUpper, composedParts, nil) })
}

// keptValues says the restated model mv states what c restates: every
// history past the rows its blocks keep holds its values beside its keys,
// the rows' moved to c.hist an entry a position, or, where the rows' stay,
// at least one history of an entry each several positions.
func keptValues(t *testing.T, mv *model.Model, c valuesCase) {
	t.Helper()
	kept, read, periodic := 0, 0, 0
	for li := 0; li < mv.Cfg.NLayer; li++ {
		g := mv.Composing().BlockGraph(li)
		for _, k := range g.KeptHistories() {
			if k.ValuesInKeys || c.hist > 0 && (k.Period != 1 || k.Hist != c.hist) {
				t.Fatalf("block %d keeps %+v: the fixture states history %d, values of their own", li, k, c.hist)
			}
			if k.Period > 1 {
				periodic++
			}
			kept++
		}
		if g.ReadsHistoryOf() >= 0 {
			read++
		}
	}
	selected := periodic > 0
	if c.hist > 0 {
		selected = kept+read == mv.Cfg.NLayer
	}
	if !selected || c.readers != (read > 0) {
		t.Fatalf("%d of %d blocks keep a history past their rows' (%d of an entry each several positions), %d read another's: the gate selected nothing",
			kept, mv.Cfg.NLayer, periodic, read)
	}
}

func valuesNear(t *testing.T, what string, got, want [][]float32, bound float64) {
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

// valuesPaged runs a long prompt on mv with every sealed page of its
// history evicted into a store and faulted back as the steps read it,
// against the same prompt unconstrained on base: a history past the rows
// sends its keys and values home and reads them back.
func valuesPaged(t *testing.T, base, mv *model.Model) {
	full := base.Vocab.Encode(strings.Repeat("The river rose every spring until the bridges were islands. ", 90), true)
	const n, decode = 720, 8
	if len(full) < n+decode {
		t.Fatalf("the prompt is %d tokens, short of %d", len(full), n+decode)
	}
	arm := func(m *model.Model, paged bool) [][]float32 {
		s := testInstance(t, m, n+decode+1, WithKVType(model.KVF32))
		defer s.Close()
		var planes []int
		if paged {
			s.SetKVStore(NewMemStore())
			mustBudget(t, s, 1)
			for li := s.lo; li < s.hi; li++ {
				if s.readsOther(li) {
					continue
				}
				for _, k := range m.Composing().BlockGraph(li).KeptHistories() {
					if e := s.kv.histSlot(li, k.Hist); e != 0 {
						planes = append(planes, e)
					}
				}
			}
		}
		out := 0
		sent := func() {
			for _, e := range planes {
				if slices.Contains(s.kv.layers[e].out, true) {
					out++
				}
			}
		}
		lg, err := s.Prefill(full[:n])
		if err != nil {
			t.Fatal(err)
		}
		sent()
		logits := [][]float32{slices.Clone(lg)}
		for i := range decode {
			if lg, err = s.Forward(full[n+i]); err != nil {
				t.Fatal(err)
			}
			sent()
			logits = append(logits, slices.Clone(lg))
		}
		if paged {
			if out == 0 || s.kv.fetched == 0 {
				t.Fatalf("no page of the %d histories past the rows went home (%d pages evicted, %d faulted back): the arm paged nothing it gates",
					len(planes), s.kv.evicted, s.kv.fetched)
			}
			t.Logf("%d pages evicted, %d faulted back; %d histories past the rows held pages at home, summed over %d reads", s.kv.evicted, s.kv.fetched, out, 1+decode)
		}
		return logits
	}
	valuesNear(t, "paged against the model unconstrained", arm(mv, true), arm(base, false), 1e-8)
}

// wroteValues says each block of st keeping a history past its rows writes
// its keys and values there, and, where its rows' moved to history h,
// nothing into its rows': the arm ran the history it restates.
func wroteValues(t *testing.T, st *Instance, m *model.Model, h int) {
	t.Helper()
	nonzero := func(v []float32) bool { return slices.ContainsFunc(v, func(x float32) bool { return x != 0 }) }
	for li := st.lo; li < st.hi; li++ {
		if st.readsOther(li) {
			continue
		}
		for _, k := range m.Composing().BlockGraph(li).KeptHistories() {
			e := st.kv.histSlot(li, k.Hist)
			if e == 0 {
				t.Fatalf("block %d holds no history %d", li, k.Hist)
			}
			if pl := st.kv.layers[e].mem; pl.Len() == 0 || !nonzero(pl.K(0)) || !nonzero(pl.V(0)) {
				t.Fatalf("block %d's history %d holds no keys or no values", li, k.Hist)
			}
		}
		if h == 0 {
			continue
		}
		rows := st.kv.layers[li].mem
		for n := range rows.Len() {
			if nonzero(rows.K(n)) || nonzero(rows.V(n)) {
				t.Fatalf("block %d wrote its rows' history page %d", li, n)
			}
		}
	}
}
