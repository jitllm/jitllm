package engine

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/model"
)

// hfWorst is m's worst NMSE over the golden's ids against the logits
// transformers recorded (scripts/hfgold.py), +Inf where a logit is not finite.
func hfWorst(t *testing.T, m *model.Model, golden string) float64 {
	t.Helper()
	return hfWorstOn(t, m, golden, nil, WithKVF16(false))
}

// hfWorstOn is hfWorst with the instance placed by place before the first
// token, under opts.
func hfWorstOn(t *testing.T, m *model.Model, golden string, place func(*Instance), opts ...Option) float64 {
	t.Helper()
	ids, heads := hfGold(t, golden)
	st := testInstance(t, m, len(ids)+1, opts...)
	defer st.Close()
	if place != nil {
		place(st)
	}
	worst := 0.0
	for p, id := range ids {
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		worst = max(worst, hfNMSE64(lg, heads[p]))
	}
	return worst
}

// hfGold is the golden's ids and the logits transformers recorded at each.
func hfGold(t *testing.T, golden string) ([]int32, [][]float64) {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "testdata", "golden", "hf", golden+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var gold struct {
		IDs []int32 `json:"ids"`
		Pos []struct {
			Head []float64 `json:"head"`
		} `json:"pos"`
	}
	if err := json.Unmarshal(b, &gold); err != nil {
		t.Fatal(err)
	}
	heads := make([][]float64, len(gold.Pos))
	for p, x := range gold.Pos {
		heads[p] = x.Head
	}
	return gold.IDs, heads
}

// hfNMSE64 is lg's NMSE against want, +Inf where it is not finite.
func hfNMSE64(lg []float32, want []float64) float64 {
	var num, den float64
	for i, w := range want {
		d := float64(lg[i]) - w
		num, den = num+d*d, den+w*w
	}
	if e := num / den; !math.IsNaN(e) && !math.IsInf(e, 0) {
		return e
	}
	return math.Inf(1)
}

// TestGPTJFeaturesAreLoadBearing holds what GPT-J's library file states
// (convert/models/gptj.json) against transformers' own class: the stated graph
// within TestSafetensorsMatchTransformers' band, and each feature, removed,
// outside it -- the MLP reading the attention's norm, the rotary over
// rotary_dim of each head on adjacent pairs, the tanh GELU and the head's bias.
func TestGPTJFeaturesAreLoadBearing(t *testing.T) {
	const name = "synth-hf-gptj"
	m, err := model.Open(hfContainer(t, name))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if w := hfWorst(t, m, name); w > hfNMSE {
		t.Fatalf("the stated graph: worst NMSE %.3e against transformers (bound %.0e)", w, hfNMSE)
	}
	if m.Vector(jlm.RoleOutputBias) == nil || m.Cfg.NRot >= m.Cfg.HeadDim || m.Cfg.RopeNeox {
		t.Fatalf("the fixture lost a feature: head bias %v, rotary %d of %d, NEOX %v",
			m.Vector(jlm.RoleOutputBias) != nil, m.Cfg.NRot, m.Cfg.HeadDim, m.Cfg.RopeNeox)
	}
	unnormed := func(m *model.Model) func() {
		return restated(m, func(g *graph.Graph) *graph.Graph {
			out := cloned(g)
			for i := range out.Ops {
				if o := &out.Ops[i]; binding(jlm.RoleFFNUp)(i, o) {
					for _, n := range out.Ops {
						if n.Kind == graph.OpLayerNorm {
							o.In[0] = n.In[0]
						}
					}
				}
			}
			return out
		})
	}
	ropeEdit := func(edit func(a *graph.Attrs)) func(m *model.Model) func() {
		return func(m *model.Model) func() {
			return restated(m, editing(opsOfKind(graph.OpRope), func(o *graph.Op) { edit(&o.Attr) }))
		}
	}
	erfGELU := func(m *model.Model) func() {
		return restated(m, editing(opsOfKind(graph.OpExpr), func(o *graph.Op) {
			terms := append([]graph.Term(nil), o.Attr.Expr...)
			for i := range terms {
				if terms[i].Fn == graph.MapTanh {
					terms[i].Fn = graph.MapErf
				}
			}
			o.Attr.Expr = terms
		}))
	}
	noHeadBias := func(m *model.Model) func() {
		v := m.Vector(jlm.RoleOutputBias)
		was := append([]float32(nil), v...)
		clear(v)
		return func() { copy(v, was) }
	}
	for _, c := range []struct {
		name  string
		apply func(*model.Model) func()
	}{
		{"the MLP reading the block's input unnormed", unnormed},
		{"rotary over the whole head", ropeEdit(func(a *graph.Attrs) { a.NRot = a.HeadDim })},
		{"NEOX rotary for adjacent pairs", ropeEdit(func(a *graph.Attrs) { a.Layout = graph.RopeNeox })},
		{"the erf GELU for the tanh", erfGELU},
		{"no head bias", noHeadBias},
	} {
		undo := c.apply(m)
		w := hfWorst(t, m, name)
		undo()
		if !(w > hfNMSE) {
			t.Errorf("%s: worst NMSE %.3e, inside the band (%.0e): the feature is not load-bearing", c.name, w, hfNMSE)
			continue
		}
		t.Logf("%s: worst NMSE %.3e", c.name, w)
	}
}
