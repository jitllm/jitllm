package model

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/tok/jinja"
)

// TestDecisionAnswerMatchesTheOracle: a decision's answer arithmetic -- the
// scaled softmax, the variants' mean, the expected level, the mode, TypeSafe's
// confidences and Laya's entropy -- runs on generated kernels, held to the
// oracle's float64 at every option count 1..255 (ragged against every vector
// width), with a guard past the end of every scratch buffer.
func TestDecisionAnswerMatchesTheOracle(t *testing.T) {
	answerGate(t, "")
}

// answerGate runs the sweep with one feature of the arithmetic removed, and
// returns the number of mismatches (RULE 10: the violation runs below).
func answerGate(t *testing.T, violation string) int {
	t.Helper()
	j := nn.NewJIT(256, 256, []quant.Type{quant.F32})
	if j == nil {
		t.Fatal("no code generator on this host")
	}
	defer j.Close()
	// An instance for the composed argmax the answer picks with.
	path, ok := existingModel(models["stories260K"])
	if !ok {
		t.Fatalf("model not present: %s (nor its container)", models["stories260K"])
	}
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	st := m.NewInstance(8)
	defer st.Close()
	const guard = 64
	sentinel := float32(math.NaN())
	bad := 0
	fail := func(format string, args ...any) {
		bad++
		if violation == "" {
			t.Errorf(format, args...)
		}
	}
	rng := rand.New(rand.NewSource(7))
	for _, c := range []struct {
		kind     jlm.DecisionKind
		typ      jlm.QuestionType
		variants int
	}{
		{jlm.DecisionLev, jlm.QuestionChoice, 2},
		{jlm.DecisionLev, jlm.QuestionNoul, 1},
		{jlm.DecisionLFM2D1, jlm.QuestionScore, 1},
		{jlm.DecisionLFM2D1, jlm.QuestionChoice, 1},
		{jlm.DecisionLaya, jlm.QuestionChoice, 1},
		{jlm.DecisionLaya, jlm.QuestionScore, 1},
	} {
		for n := 1; n <= 255; n++ {
			if c.typ == jlm.QuestionNoul && n < 2 {
				continue
			}
			temp := float32(0.5 + rng.Float64()*2)
			d := &Decider{kind: c.kind, cfg: &jlm.Config{DecisionTemps: []jlm.DecisionTemp{{Type: c.typ, T: temp}}},
				m: m, st: st, emb: &Embedder{jit: j}, violation: violation}
			d.acc, d.prob, d.row = make([]float32, n+guard), make([]float32, n+guard), make([]float32, n+guard)
			d.ramp = make([]float32, n+guard)
			for i := range d.ramp {
				d.ramp[i] = float32(i)
			}
			for _, b := range [][]float32{d.acc, d.prob, d.row} {
				for i := n; i < len(b); i++ {
					b[i] = sentinel
				}
			}
			q := DecisionQuestion{ID: "q", Type: c.typ}
			for i := 0; i < n; i++ {
				q.Options = append(q.Options, DecisionOption{Key: fmt.Sprint(i), Description: jinja.None()})
			}
			if c.typ == jlm.QuestionNoul {
				q.Options = []DecisionOption{{Key: "false"}, {Key: "true"}}
			}
			vs := make([][]float32, c.variants)
			for v := range vs {
				vs[v] = make([]float32, n)
				for i := range vs[v] {
					vs[v][i] = float32(rng.Float64()*16 - 8)
				}
			}
			a, err := d.answer(&q, vs)
			if err != nil {
				t.Fatal(err)
			}
			for _, b := range [][]float32{d.acc, d.prob, d.row} {
				for i := n; i < len(b); i++ {
					if b[i] == b[i] {
						fail("%v %v n=%d: scratch written past n at %d", c.kind, c.typ, n, i)
						break
					}
				}
			}
			ref := oracle.Decide(vs, float64(temp))
			near := func(what string, got, want, tol float64) {
				if math.IsNaN(got) || math.Abs(got-want) > tol {
					fail("%v %v n=%d %s: %v, the oracle %v", c.kind, c.typ, n, what, got, want)
				}
			}
			switch c.typ {
			case jlm.QuestionNoul:
				near("noul", a.Noul, ref.Expected/float64(n-1), 1e-5)
				continue
			case jlm.QuestionChoice:
				if a.Choice != q.Options[ref.Mode].Key {
					fail("%v n=%d: chose %s, the oracle %s", c.kind, n, a.Choice, q.Options[ref.Mode].Key)
				}
			case jlm.QuestionScore:
				near("score", a.Score, ref.Expected, 1e-5*float64(n))
			}
			for i := range a.Probs {
				near("p", a.Probs[i], ref.Probs[i], 2e-6)
			}
			switch {
			case c.kind == jlm.DecisionLaya:
				near("laya confidence", a.Confidence, ref.Laya, 1e-4)
			case c.typ == jlm.QuestionChoice:
				near("confidence", a.Confidence, ref.ConfChoice, 1e-5)
			default:
				near("confidence", a.Confidence, ref.ConfScore, 1e-4)
			}
		}
	}
	return bad
}

// TestDecisionAnswerGateDiscriminates removes a step of the arithmetic at a
// time; the sweep must fail on each.
func TestDecisionAnswerGateDiscriminates(t *testing.T) {
	for _, v := range []string{"uncalibrated", "unreversed", "summed", "from-zero"} {
		t.Run(v, func(t *testing.T) {
			if n := answerGate(t, v); n == 0 {
				t.Errorf("removing %s failed nothing: the gate does not see it", v)
			} else {
				t.Logf("%s: %d mismatches", v, n)
			}
		})
	}
}
