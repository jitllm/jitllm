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
	st := testInstance(t, m, len(gold.IDs)+1, WithKVF16(false))
	defer st.Close()
	worst := 0.0
	for p, id := range gold.IDs {
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		var num, den float64
		for i, w := range gold.Pos[p].Head {
			d := float64(lg[i]) - w
			num, den = num+d*d, den+w*w
		}
		nmse := num / den
		if math.IsNaN(nmse) || math.IsInf(nmse, 0) {
			return math.Inf(1)
		}
		worst = max(worst, nmse)
	}
	return worst
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
