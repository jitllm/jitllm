package convert

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
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
			s := 0.0
			for i := range w {
				s += arg(tm.Args[0], i)
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
	cfg    jlm.Config
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

var xieluNumbers = []float64{0.8, 0.6, 0.5, -1e-6}

var actCases = []actCase{
	{"silu", jlm.Config{}, true, nil, each(func(g, u float64) float64 { return g * sig(g) * u })},
	{"gelu tanh, ungated", jlm.Config{Flags: jlm.FlagGELU}, false, nil, each(func(g, _ float64) float64 { return geluTanhRef(g) })},
	{"squared relu", jlm.Config{Flags: jlm.FlagReLU2}, true, nil, each(func(g, u float64) float64 { r := math.Max(g, 0); return r * r * u })},
	{"gelu erf", jlm.Config{Arch: jlm.ArchModernBERT, Flags: jlm.FlagGELU}, true, nil,
		each(func(g, u float64) float64 { return 0.5 * g * (1 + math.Erf(g/math.Sqrt2)) * u })},
	{"xielu", jlm.Config{Arch: jlm.ArchApertus}, false, map[jlm.Role][]float64{jlm.RoleXIELU: xieluNumbers},
		each(func(x, _ float64) float64 {
			ap, an, b, eps := xieluNumbers[0], xieluNumbers[1], xieluNumbers[2], xieluNumbers[3]
			if x > 0 {
				return ap*x*x + b*x
			}
			return (math.Expm1(math.Min(x, eps))-x)*an + b*x
		})},
	{"gaussian top-k", jlm.Config{Flags: jlm.FlagGELU, NSparse: 1, SparseStd: 1.6448536}, true, nil,
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
	{"clamped swiglu", jlm.Config{Flags2: jlm.Flag2SwiGLUClamp}, true, nil, each(func(g, u float64) float64 {
		x := math.Min(g, 10)
		return x * sig(x) * math.Max(math.Min(u, 10), -10)
	})},
	{"gpt-oss swiglu", jlm.Config{Flags: jlm.FlagSwiGLUOAI}, true, nil, each(func(g, u float64) float64 {
		x := math.Min(g, 7)
		return x * sig(1.702*x) * (math.Max(math.Min(u, 7), -7) + 1)
	})},
	{"situ", jlm.Config{SituBeta: 4, SituLinearBeta: 25}, true, nil, each(func(g, u float64) float64 {
		return 4 * math.Tanh(g/4) * sig(g) * 25 * math.Tanh(u/25)
	})},
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
	for _, c := range actCases {
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
			cfg := c.cfg
			cfg.NLayer = 2
			s := &jlm.Source{Config: &cfg}
			for li := range int32(2) {
				s.Tensors = append(s.Tensors, jlm.Tensor{Role: jlm.RoleFFNUp, Block: li})
				if c.gated {
					s.Tensors = append(s.Tensors, jlm.Tensor{Role: jlm.RoleFFNGate, Block: li})
				}
			}
			if err := StateActivations(s); err != nil {
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
