package convert

import (
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
)

//go:embed rotary.json
var rotaryJSON []byte

// rotaryMaxPairs bounds the pairs a stated rotary is evaluated over: a
// width read from a file sizes nothing unchecked (RULE 9).
const rotaryMaxPairs = 1 << 14

// rotaryFile is rotary.json: the rotary scaling methods by their published
// names, each a per-pair divisor of the rotary's angles stated in the
// model.json expression language.
type rotaryFile struct {
	Methods map[string]*rotaryMethod `json:"methods"`
}

// rotaryMethod is one scaling method: the parameters it reads, named values
// each evaluated in order over the ones before it, and the divisor of pair
// pair's angle. Every expression reads the parameters, rotated (the width
// the rotary turns), base (its base as the container states it), pair and
// pairs; it calls log, floor, ceil, min, max, sqrt and f32 (a float32
// rounding, for a reference that computes in float32).
type rotaryMethod struct {
	Params  []string    `json:"params"`
	Values  [][2]string `json:"values"`
	Divisor string      `json:"divisor"`
}

func readRotary() (*rotaryFile, error) {
	var f rotaryFile
	if err := json.Unmarshal(rotaryJSON, &f); err != nil {
		return nil, fmt.Errorf("convert: rotary.json: %w", err)
	}
	return &f, nil
}

// rotaryDivisors is method's divisor of each of pairs pairs, a rotary of
// width rotated and base base, its parameters params. A parameter the method
// does not read, or one it reads and is not given, is refused, so a
// statement and its caller cannot disagree on what the method takes.
func rotaryDivisors(method string, pairs int, rotated, base float64, params map[string]any) ([]float32, error) {
	lib, err := readRotary()
	if err != nil {
		return nil, err
	}
	m := lib.Methods[method]
	if m == nil {
		return nil, fmt.Errorf("convert: rotary.json states no method %q: %w", method, ErrNotImplemented)
	}
	for k := range params {
		if !slices.Contains(m.Params, k) {
			return nil, fmt.Errorf("convert: rotary method %s reads no parameter %s", method, k)
		}
	}
	for _, k := range m.Params {
		if _, ok := params[k]; !ok {
			return nil, fmt.Errorf("convert: rotary method %s reads %s, which is not given", method, k)
		}
	}
	if pairs <= 0 || pairs > rotaryMaxPairs {
		return nil, fmt.Errorf("convert: a rotary of %d pairs", pairs)
	}
	values := make([]*exprNode, len(m.Values))
	for i, v := range m.Values {
		if values[i], err = parseExpr(v[1]); err != nil {
			return nil, fmt.Errorf("convert: rotary method %s: %s: %w", method, v[0], err)
		}
	}
	div, err := parseExpr(m.Divisor)
	if err != nil {
		return nil, fmt.Errorf("convert: rotary method %s: divisor: %w", method, err)
	}
	out := make([]float32, pairs)
	for p := range out {
		bound := map[string]any{"rotated": rotated, "base": base, "pair": float64(p), "pairs": float64(pairs)}
		for k, v := range params {
			bound[k] = v
		}
		env := func(path string) any { return bound[path] }
		for i, v := range m.Values {
			x, err := evalRotary(values[i], env)
			if err != nil {
				return nil, fmt.Errorf("convert: rotary method %s: %s: %w", method, v[0], err)
			}
			bound[v[0]] = x
		}
		d, err := evalRotary(div, env)
		if err != nil {
			return nil, fmt.Errorf("convert: rotary method %s: divisor: %w", method, err)
		}
		f, ok := d.(float64)
		if !ok || !(f > 0) || math.IsInf(f, 0) {
			return nil, fmt.Errorf("convert: rotary method %s: pair %d's divisor is %v", method, p, d)
		}
		out[p] = float32(f)
	}
	return out, nil
}

// rotaryFns are the functions a rotary statement calls.
func rotaryFns() map[string]func([]float64) float64 {
	return map[string]func([]float64) float64{
		"log":   func(a []float64) float64 { return math.Log(a[0]) },
		"sqrt":  func(a []float64) float64 { return math.Sqrt(a[0]) },
		"floor": func(a []float64) float64 { return math.Floor(a[0]) },
		"ceil":  func(a []float64) float64 { return math.Ceil(a[0]) },
		"f32":   func(a []float64) float64 { return float64(float32(a[0])) },
		"min":   func(a []float64) float64 { return math.Min(a[0], a[1]) },
		"max":   func(a []float64) float64 { return math.Max(a[0], a[1]) },
	}
}

// rotaryArity is each rotary function's argument count.
func rotaryArity(name string) int {
	if name == "min" || name == "max" {
		return 2
	}
	return 1
}

// evalRotary is evalMJ with the rotary's functions: a call's arguments are
// rotary expressions too, a condition evaluates only the branch it takes, and
// every other node is the config language's over its calls' values.
func evalRotary(n *exprNode, env exprEnv) (any, error) {
	switch n.op {
	case "call":
		fn, ok := rotaryFns()[n.str]
		if !ok || len(n.args) != rotaryArity(n.str) {
			return nil, fmt.Errorf("%s() with %d arguments is not a rotary function", n.str, len(n.args))
		}
		args := make([]float64, len(n.args))
		for i, a := range n.args {
			v, err := evalRotary(a, env)
			if err != nil {
				return nil, err
			}
			f, ok := v.(float64)
			if !ok {
				return nil, fmt.Errorf("%s: %s is %s, not a number", exprText(n), exprText(a), exprKind(v))
			}
			args[i] = f
		}
		v := fn(args)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("%s is %g", exprText(n), v)
		}
		return v, nil
	case "?:":
		c, err := evalRotary(n.args[0], env)
		if err != nil {
			return nil, err
		}
		b, ok := c.(bool)
		if !ok {
			return nil, fmt.Errorf("%s is %s, not a condition", exprText(n.args[0]), exprKind(c))
		}
		if b {
			return evalRotary(n.args[1], env)
		}
		return evalRotary(n.args[2], env)
	}
	if !slices.ContainsFunc(n.args, exprCalls) {
		return evalMJ(n, env)
	}
	m := *n
	m.args = make([]*exprNode, len(n.args))
	for i, a := range n.args {
		v, err := evalRotary(a, env)
		if err != nil {
			return nil, err
		}
		m.args[i] = exprValue(v)
	}
	return evalMJ(&m, env)
}

// exprCalls reports whether n calls a function.
func exprCalls(n *exprNode) bool {
	return n.op == "call" || slices.ContainsFunc(n.args, exprCalls)
}

// exprValue is v as a literal node.
func exprValue(v any) *exprNode {
	switch x := v.(type) {
	case float64:
		return &exprNode{op: "num", num: x}
	case bool:
		return &exprNode{op: "bool", b: x}
	case string:
		return &exprNode{op: "str", str: x}
	}
	return &exprNode{op: "null"}
}

// StateRotary states a scaled rotary's angles per pair: a container whose
// config states its scaling by method (YaRN's factor and correction range,
// a linear factor) carries instead the method's divisor of each pair
// (rotary.json) times the per-pair factors the file ships, as the
// rope_freqs tensor (or the short factors it ships in its place) and the
// long factors past the original context, and states no method. The
// scaling's magnitude stays where the config states it (AttnFactor, and
// YarnFactor with YarnLogMul for the score), since it is not a frequency.
func StateRotary(c *jlm.Config, s *jlm.Source) error {
	method, params := "", map[string]any{}
	switch {
	case c.YarnFactor > 0 && (c.YarnBetaFast != 0 || c.YarnBetaSlow != 0):
		if !(c.YarnBetaFast > 0) || !(c.YarnBetaSlow > 0) || c.YarnOrigCtx == 0 {
			return fmt.Errorf("convert: a YaRN of correction range %g to %g over %d positions", c.YarnBetaFast,
				c.YarnBetaSlow, c.YarnOrigCtx)
		}
		method = "yarn"
		params["factor"], params["original_context"] = float64(c.YarnFactor), float64(c.YarnOrigCtx)
		params["beta_fast"], params["beta_slow"] = float64(c.YarnBetaFast), float64(c.YarnBetaSlow)
		params["truncate"] = !c.Flags.Has(jlm.FlagYarnExact)
	case c.RopeLinear > 0:
		method = "linear"
		params["factor"] = float64(c.RopeLinear)
	default:
		return nil
	}
	div, err := rotaryDivisors(method, int(c.NRot)/2, float64(c.NRot), float64(c.RopeBase), params)
	if err != nil {
		return err
	}
	short := -1
	for _, r := range []jlm.Role{jlm.RoleRopeFreqs, jlm.RoleRopeFactorsShort} {
		if short = slices.IndexFunc(s.Tensors, func(t jlm.Tensor) bool { return t.Role == r && t.Block == jlm.DenseBlock }); short >= 0 {
			break
		}
	}
	if short < 0 {
		s.Tensors = append(s.Tensors, ropeFreqsTensor(div))
	} else if err := scaleFactors(&s.Tensors[short], div); err != nil {
		return err
	}
	if long := slices.IndexFunc(s.Tensors, func(t jlm.Tensor) bool {
		return t.Role == jlm.RoleRopeFactorsLong && t.Block == jlm.DenseBlock
	}); long >= 0 {
		if err := scaleFactors(&s.Tensors[long], div); err != nil {
			return err
		}
	}
	c.RopeLinear, c.YarnBetaFast, c.YarnBetaSlow = 0, 0, 0
	c.Flags &^= jlm.FlagYarnExact
	return nil
}

// scaleFactors multiplies t's per-pair factors by div, a pair each, in
// float32, and stores them as float32.
func scaleFactors(t *jlm.Tensor, div []float32) error {
	if t.NDim != 1 || t.Dims[0] != uint64(len(div)) {
		return fmt.Errorf("convert: %s holds %d rotary factors for %d pairs", t.Name, t.Dims[0], len(div))
	}
	b, err := t.Bytes()
	if err != nil {
		return err
	}
	f, err := floatsOf(t.Type, b, len(div))
	if err != nil {
		return fmt.Errorf("convert: %s: %w", t.Name, err)
	}
	out := make([]byte, 4*len(f))
	for i := range f {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(f[i]*div[i]))
	}
	t.Type, t.Data, t.Load = jlm.TypeF32, out, nil
	return nil
}
