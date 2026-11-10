package convert

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
)

// evalExpr is e over one row of each operand, in float64: the test's own
// reading of the op language, nothing the engine runs.
func evalExpr(t *testing.T, e *jlm.Expr, in [][]float64, stored map[jlm.Role][]float64) []float64 {
	t.Helper()
	w := len(in[0])
	vals := make([][]float64, len(e.Terms))
	arg := func(a jlm.Arg, i int) float64 {
		switch a.Kind {
		case jlm.ArgIn:
			return in[a.Index][i]
		case jlm.ArgTerm:
			v := vals[a.Index]
			if len(v) == 1 {
				return v[0]
			}
			return v[i]
		case jlm.ArgConst:
			return float64(a.Value)
		case jlm.ArgStored:
			return stored[a.Role][a.Index]
		case jlm.ArgWidth:
			return float64(w)
		}
		t.Fatalf("an operand of kind %d", a.Kind)
		return 0
	}
	for k, tm := range e.Terms {
		if tm.Reduce {
			s := arg(tm.Args[0], 0)
			for i := 1; i < w; i++ {
				switch x := arg(tm.Args[0], i); tm.Fn {
				case jlm.FnMax:
					s = math.Max(s, x)
				case jlm.FnMin:
					s = math.Min(s, x)
				default:
					s += x
				}
			}
			vals[k] = []float64{s}
			continue
		}
		out := make([]float64, w)
		for i := range out {
			a, b, c := arg(tm.Args[0], i), 0.0, 0.0
			if n := tm.Fn.Arity(); n > 1 {
				b = arg(tm.Args[1], i)
				if n > 2 {
					c = arg(tm.Args[2], i)
				}
			}
			switch tm.Fn {
			case jlm.FnAdd:
				out[i] = a + b
			case jlm.FnSub:
				out[i] = a - b
			case jlm.FnMul:
				out[i] = a * b
			case jlm.FnDiv:
				out[i] = a / b
			case jlm.FnMax:
				out[i] = math.Max(a, b)
			case jlm.FnMin:
				out[i] = math.Min(a, b)
			case jlm.FnExp:
				out[i] = math.Exp(a)
			case jlm.FnExpm1:
				out[i] = math.Expm1(a)
			case jlm.FnSigmoid:
				out[i] = 1 / (1 + math.Exp(-a))
			case jlm.FnSoftplus:
				out[i] = math.Log1p(math.Exp(a))
			case jlm.FnSqrt:
				out[i] = math.Sqrt(a)
			case jlm.FnTanh:
				out[i] = math.Tanh(a)
			case jlm.FnErf:
				out[i] = math.Erf(a)
			case jlm.FnFma:
				out[i] = a*b + c
			case jlm.FnWhere:
				out[i] = c
				if a > 0 {
					out[i] = b
				}
			case jlm.FnAbs:
				out[i] = math.Abs(a)
			case jlm.FnLess:
				out[i] = 0
				if a < b {
					out[i] = 1
				}
			default:
				t.Fatalf("no reading of %v", tm.Fn)
			}
		}
		vals[k] = out
	}
	return vals[len(vals)-1]
}

func sig(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

func geluTanhRef(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x)))
}

// actCase is a converter's config and what its reference computes.
type actCase struct {
	name   string
	cfg    Config
	gated  bool
	stored map[jlm.Role][]float64
	ref    func(g, u []float64) []float64
}

func each(f func(g, u float64) float64) func(g, u []float64) []float64 {
	return func(g, u []float64) []float64 {
		out := make([]float64, len(g))
		for i := range g {
			out[i] = f(g[i], u[i])
		}
		return out
	}
}

func xieluNumbers() []float64 {
	return []float64{0.8, 0.6, 0.5, -1e-6}
}

func actCases() []actCase {
	return []actCase{
		{"silu", Config{Config: &jlm.Config{}}, true, nil, each(func(g, u float64) float64 { return g * sig(g) * u })},
		{"gelu tanh, ungated", Config{Config: &jlm.Config{Flags: jlm.FlagGELU}}, false, nil, each(func(g, _ float64) float64 { return geluTanhRef(g) })},
		{"squared relu", Config{Config: &jlm.Config{Flags: jlm.FlagReLU2}}, true, nil, each(func(g, u float64) float64 { r := math.Max(g, 0); return r * r * u })},
		{"gelu erf", Config{Arch: ArchModernBERT, Config: &jlm.Config{Flags: jlm.FlagGELU}}, true, nil,
			each(func(g, u float64) float64 { return 0.5 * g * (1 + math.Erf(g/math.Sqrt2)) * u })},
		{"xielu", Config{Config: &jlm.Config{}}, false, map[jlm.Role][]float64{jlm.RoleXIELU: xieluNumbers()},
			each(func(x, _ float64) float64 {
				ap, an, b, eps := xieluNumbers()[0], xieluNumbers()[1], xieluNumbers()[2], xieluNumbers()[3]
				if x > 0 {
					return ap*x*x + b*x
				}
				return (math.Expm1(math.Min(x, eps))-x)*an + b*x
			})},
		{"gaussian top-k", Config{Config: &jlm.Config{Flags: jlm.FlagGELU, NSparse: 1, SparseStd: 1.6448536}}, true, nil,
			func(g, u []float64) []float64 {
				mean, v := 0.0, 0.0
				for _, x := range g {
					mean += x
				}
				mean /= float64(len(g))
				for _, x := range g {
					v += (x - mean) * (x - mean)
				}
				cut := mean + 1.6448536*math.Sqrt(v/float64(len(g)))
				out := make([]float64, len(g))
				for i := range g {
					out[i] = geluTanhRef(math.Max(g[i]-cut, 0)) * u[i]
				}
				return out
			}},
		{"clamped swiglu", Config{Config: &jlm.Config{Flags2: jlm.Flag2SwiGLUClamp}}, true, nil, each(func(g, u float64) float64 {
			x := math.Min(g, 10)
			return x * sig(x) * math.Max(math.Min(u, 10), -10)
		})},
		{"gpt-oss swiglu", Config{Config: &jlm.Config{Flags: jlm.FlagSwiGLUOAI}}, true, nil, each(func(g, u float64) float64 {
			x := math.Min(g, 7)
			return x * sig(1.702*x) * (math.Max(math.Min(u, 7), -7) + 1)
		})},
		{"situ", Config{Config: &jlm.Config{SituBeta: 4, SituLinearBeta: 25}}, true, nil, each(func(g, u float64) float64 {
			return 4 * math.Tanh(g/4) * sig(g) * 25 * math.Tanh(u/25)
		})},
	}
}

// actRows are operands that reach every branch: both signs, the clamps'
// bounds and past them, and a tail above the gaussian's cutoff.
func actRows() (g, u []float64) {
	for i := range 32 {
		x := float64(i-14) * 0.83
		g, u = append(g, x), append(u, 12-float64(i%9)*3)
	}
	return g, u
}

func agrees(a, b []float64) (float64, bool) {
	worst := 0.0
	for i := range a {
		worst = max(worst, math.Abs(a[i]-b[i])/(1+math.Abs(b[i])))
	}
	return worst, worst < 1e-5
}

// TestStatedActivationsAreTheReferences reads every activation the converter
// states, as the op language defines its terms, against the formula of the
// model's reference, gated and ungated as the block stores; and each
// statement taken apart -- every constant moved, every select's branches
// swapped -- is told from it. The gaussian top-k is stated for the sparse
// layers alone.
func TestStatedActivationsAreTheReferences(t *testing.T) {
	for _, c := range actCases() {
		t.Run(c.name, func(t *testing.T) {
			g, u := actRows()
			if c.cfg.NSparse != 0 {
				// A gate whose tail stands a little past the cutoff, where
				// the GELU after it is not yet linear.
				for i := range g {
					g[i] = float64(i%11-5) * 0.15
				}
				copy(g[len(g)-4:], []float64{1.5, 2.5, 3.5, 4.5})
			}
			inner := *c.cfg.Config
			cfg := &Config{Arch: c.cfg.Arch, Config: &inner}
			cfg.NLayer = 2
			s := &jlm.Source{Config: cfg.Config}
			for li := range int32(2) {
				s.Tensors = append(s.Tensors, jlm.Tensor{Role: jlm.RoleFFNUp, Block: li})
				if c.gated {
					s.Tensors = append(s.Tensors, jlm.Tensor{Role: jlm.RoleFFNGate, Block: li})
				}
				for r := range c.stored {
					s.Tensors = append(s.Tensors, jlm.Tensor{Role: r, Block: li})
				}
			}
			if err := StateActivations(cfg, s); err != nil {
				t.Fatal(err)
			}
			e := cfg.ActAt(0)
			if e == nil {
				t.Fatal("no activation stated")
			}
			in := [][]float64{g}
			if c.gated {
				in = append(in, u)
			}
			if n := e.Reads(); n != len(in) {
				t.Fatalf("the statement reads %d operands, the block has %d", n, len(in))
			}
			want := c.ref(g, u)
			if worst, ok := agrees(evalExpr(t, e, in, c.stored), want); !ok {
				t.Fatalf("the stated activation is %.3e from the reference", worst)
			}
			if sparse := cfg.NSparse != 0; sparse != (len(cfg.Acts) == 2) || sparse && slices.Equal(cfg.ActAt(1).Terms, e.Terms) {
				t.Fatalf("%d activations stated for layers %v", len(cfg.Acts), cfg.ActLayers)
			}
			moved := 0
			for k, tm := range e.Terms {
				for j := range tm.Args {
					bad := jlm.Expr{Terms: slices.Clone(e.Terms)}
					switch a := &bad.Terms[k].Args[j]; {
					case a.Kind == jlm.ArgConst:
						a.Value += 0.25
					case j == 1 && tm.Fn == jlm.FnWhere:
						bad.Terms[k].Args[1], bad.Terms[k].Args[2] = tm.Args[2], tm.Args[1]
					default:
						continue
					}
					moved++
					if worst, ok := agrees(evalExpr(t, &bad, in, c.stored), want); ok {
						t.Errorf("term %d's operand %d taken apart still agrees (%.3e)", k, j, worst)
					}
				}
			}
			t.Logf("%d terms, %d violations told apart", len(e.Terms), moved)
		})
	}
}

// TestRouterScoreIsTheReferences reads the router's score each family's
// configuration states over a row of logits, against the formula its
// reference computes: softmax over the experts, sigmoid, sqrt(softplus); and
// none where the routed weights are not the selected scores. A score that
// drops the softmax's maximum still agrees in value, so a second row of large
// logits holds it to the shifted form.
func TestRouterScoreIsTheReferences(t *testing.T) {
	softmax := func(x []float64) []float64 {
		m := slices.Max(x)
		out, s := make([]float64, len(x)), 0.0
		for i, v := range x {
			out[i] = math.Exp(v - m)
			s += out[i]
		}
		for i := range out {
			out[i] /= s
		}
		return out
	}
	for _, c := range []struct {
		name string
		cfg  Config
		ref  func([]float64) []float64
	}{
		{"softmax", Config{Config: &jlm.Config{NExpert: 6}}, softmax},
		{"sigmoid", Config{Config: &jlm.Config{NExpert: 6, Flags: jlm.FlagExpertSigmoid}}, func(x []float64) []float64 {
			return each(func(g, _ float64) float64 { return sig(g) })(x, x)
		}},
		{"sqrt softplus", Config{Config: &jlm.Config{NExpert: 6, Flags2: jlm.Flag2ExpertSqrtSoftplus}}, func(x []float64) []float64 {
			return each(func(g, _ float64) float64 { return math.Sqrt(math.Log1p(math.Exp(g))) })(x, x)
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			e := RouterScore(&c.cfg)
			if err := e.Valid(); err != nil || e.Reads() != 1 {
				t.Fatalf("the stated score is not an expression of the logits: %v, %d operands", err, e.Reads())
			}
			for _, x := range [][]float64{{-1.5, 0.25, 2, -0.75, 1, 0}, {300, 310, 305, 299, 301, 309}} {
				got, want := evalExpr(t, &e, [][]float64{x}, nil), c.ref(x)
				for i := range want {
					if d := math.Abs(got[i] - want[i]); !(d <= 1e-12*math.Max(1, math.Abs(want[i]))) {
						t.Fatalf("logits %v: expert %d scores %v, the reference %v", x, i, got[i], want[i])
					}
				}
			}
		})
	}
	if e := RouterScore(&Config{Config: &jlm.Config{}}); len(e.Terms) != 0 {
		t.Fatalf("a model with no experts states a score of %d terms, want none", len(e.Terms))
	}
	// Sparsemixer picks by the raw logits.
	e := RouterScore(&Config{Arch: ArchPhiMoE, Config: &jlm.Config{NExpert: 16}})
	x := []float64{-1.5, 0.25, 2, -0.75, 1, 0}
	if got := evalExpr(t, &e, [][]float64{x}, nil); !slices.Equal(got, x) {
		t.Fatalf("sparsemixer scores %v by the logits %v", got, x)
	}
}

// TestRouterWeightIsSparsemixers evaluates the routed weight Phi-3.5-MoE's
// configuration states, pick by pick as the front end reads it (the logits
// with the earlier picks at -inf, and the pick's logit), against the
// oracle's sparsemixer on rows that keep one, several and every logit in
// each softmax; and each of the oracle's violations lands away from it.
// Every other mixture states none.
func TestRouterWeightIsSparsemixers(t *testing.T) {
	if e := RouterWeight(&Config{Config: &jlm.Config{NExpert: 16}}); len(e.Terms) != 0 {
		t.Fatalf("a softmax mixture states a weight of %d terms, want none", len(e.Terms))
	}
	e := RouterWeight(&Config{Arch: ArchPhiMoE, Config: &jlm.Config{NExpert: 16}})
	if err := e.Valid(); err != nil || e.Reads() != 2 {
		t.Fatalf("the stated weight is not an expression of the logits and the pick's: %v, %d operands", err, e.Reads())
	}
	weights := func(s []float32) []float32 {
		x := make([]float64, len(s))
		for i, v := range s {
			x[i] = float64(v)
		}
		var out []float32
		for k := range 2 {
			pick := 0
			for i := range x {
				if x[i] > x[pick] {
					pick = i
				}
			}
			m := make([]float64, len(x))
			for i := range m {
				m[i] = x[pick]
			}
			out = append(out, float32(evalExpr(t, &e, [][]float64{x, m}, nil)[pick]))
			if k == 0 {
				x = slices.Clone(x)
				x[pick] = math.Inf(-1)
			}
		}
		return out
	}
	rows := [][]float32{{-1.5, 0.25, 2, -0.75, 1, 0}, {1, 0.99, 0.985, -3, 0.2, 0.989}, {0, 0, 0, 0}, {-2, -2.01, -5, -1.99},
		{300, 299.9, 10, 299.95}, {1, 0.9801, -1, 0.5}}
	far := map[oracle.SparseMixerFault]float64{}
	for _, s := range rows {
		got := weights(s)
		want := oracle.SparseMixer(s, PhiMoEJitter, oracle.SparseMixerFaultNone).Wt
		for k := range want {
			if d := math.Abs(float64(got[k] - want[k])); !(d <= 1e-6) {
				t.Fatalf("logits %v: pick %d weighs %v, the oracle %v", s, k, got[k], want[k])
			}
		}
		for _, f := range []oracle.SparseMixerFault{oracle.SparseMixerFaultNoMask, oracle.SparseMixerFaultKeepFirst,
			oracle.SparseMixerFaultRenormalised, oracle.SparseMixerFaultAbsFactor} {
			bad := oracle.SparseMixer(s, PhiMoEJitter, f).Wt
			for k := range bad {
				far[f] = max(far[f], math.Abs(float64(got[k]-bad[k])))
			}
		}
	}
	for f, d := range far {
		if !(d > 1e-3) {
			t.Errorf("violation %d: the stated weight is %.3g from it at the farthest", f, d)
		}
	}
	t.Logf("the violations land %v from the stated weight", far)
}
