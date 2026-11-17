package convert

import (
	_ "embed"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
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
//
// A method a source states by name is read by that source's statement: HF a
// config.json's, GGUF a GGUF's keys (rotarySource).
type rotaryMethod struct {
	Describe string        `json:"describe"`
	Aliases  []string      `json:"aliases"`
	Params   []string      `json:"params"`
	HF       *rotarySource `json:"hf"`
	GGUF     *rotarySource `json:"gguf"`
	Values   [][2]string   `json:"values"`
	Divisor  string        `json:"divisor"`
}

// rotarySource is how one kind of source states a method. Read is each
// parameter's expression over the source -- a config's scaling.key the
// scaling object's key and any other path the config's, a GGUF's path a key
// past its architecture -- and context, attn_factor, rotated and base what
// the container states so far; Conventions are reads a class's statement
// names in their place (its yarn_magnitude). Require refuses a reading whose
// condition fails, by its text; where When is false the method states
// nothing; Check refuses as Require does, past When; Implements refuses one
// this engine does not run. Magnitude is the cos and sin magnitude the
// container states (over attn_factor, the one before it), Keeps each
// container field the method states (rotaryKeeps; a null value keeps
// nothing), Factors the parameters that are per-pair factor lists, short
// and long, and Clears the container's flags its divisors consume (a
// library file's flag names), cleared once they are stated.
type rotarySource struct {
	Read        map[string]string            `json:"read"`
	Conventions map[string]map[string]string `json:"conventions"`
	Require     [][2]string                  `json:"require"`
	When        string                       `json:"when"`
	Check       [][2]string                  `json:"check"`
	Magnitude   string                       `json:"magnitude"`
	Implements  [][2]string                  `json:"implements"`
	Keeps       map[string]string            `json:"keeps"`
	Factors     map[string]string            `json:"factors"`
	Clears      []string                     `json:"clears"`
}

// method is the method rotary.json states under name or an alias of it, and
// its name; nil where none is.
func (f *rotaryFile) method(name string) (*rotaryMethod, string) {
	if m := f.Methods[name]; m != nil {
		return m, name
	}
	for k, m := range f.Methods {
		if slices.Contains(m.Aliases, name) {
			return m, k
		}
	}
	return nil, ""
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
// does not read is refused, so a statement and its caller cannot disagree on
// what the method takes; one it reads and is not given is null, which an
// expression computing with it refuses by name.
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

// rotaryKeeps are the container fields a method keeps (rotarySource.Keeps),
// each set from its expression's value.
func rotaryKeeps() map[string]func(c *Config, v float64) error {
	return map[string]func(*Config, float64) error{
		"yarn_factor": func(c *Config, v float64) error {
			c.YarnFactor = float32(v)
			return nil
		},
		"original_context": func(c *Config, v float64) error {
			if v < 0 || v > math.MaxUint32 || v != math.Trunc(v) {
				return fmt.Errorf("an original context of %g positions", v)
			}
			c.YarnOrigCtx = uint32(v)
			return nil
		},
		"yarn_log_multiplier": func(c *Config, v float64) error {
			c.YarnLogMul = float32(v)
			return nil
		},
	}
}

// rotaryReading is one source's statement of a rotary scaling: the method,
// its name, the source's statement of it and the parameters read.
type rotaryReading struct {
	m      *rotaryMethod
	name   string
	src    *rotarySource
	params map[string]any
}

// rotaryScaling is obj, a config's rope scaling object, read by its method's
// HF statement, the config's paths through env; nil for none or the default.
// A method rotary.json does not state, one it states no HF reading of, or
// one methods does not list is refused by name.
func rotaryScaling(cfg *Config, obj map[string]any, env exprEnv, methods []string) (*rotaryReading, error) {
	t := obj["rope_type"]
	if t == nil {
		t = obj["type"]
	}
	if t == nil || t == "default" {
		return nil, nil
	}
	lib, err := readRotary()
	if err != nil {
		return nil, err
	}
	s, _ := t.(string)
	m, name := lib.method(s)
	if m == nil || m.HF == nil || len(methods) > 0 && !slices.Contains(methods, s) && !slices.Contains(methods, name) {
		return nil, fmt.Errorf("rope scaling %v is not implemented: %w", t, ErrNotImplemented)
	}
	bound := cfg.rotaryBound()
	return readRotary1(m, name, m.HF, m.HF.Read, func(p string) any {
		if k, ok := strings.CutPrefix(p, "scaling."); ok {
			return obj[k]
		}
		if v, ok := bound[p]; ok {
			return v
		}
		return env(p)
	})
}

// ggufRotary is f's rope.scaling.type read by its method's GGUF statement,
// the reads of convention (a class's yarn_magnitude) in place of the
// method's where it states that convention; nil for none. A method
// rotary.json states no GGUF reading of is refused by name.
func ggufRotary(f *meta.File, c *Config, convention string) (*rotaryReading, error) {
	kind := ""
	if kv, ok := f.Key("rope.scaling.type"); ok {
		s, ok := kv.String()
		if !ok {
			return nil, fmt.Errorf("rope.scaling.type is not a string")
		}
		kind = s
	}
	if kind == "" || kind == "none" {
		return nil, nil
	}
	lib, err := readRotary()
	if err != nil {
		return nil, err
	}
	m, name := lib.method(kind)
	if m == nil || m.GGUF == nil {
		return nil, fmt.Errorf("rope.scaling.type %q is not implemented: %w", kind, ErrNotImplemented)
	}
	reads := m.GGUF.Read
	if over, ok := m.GGUF.Conventions[convention]; ok {
		reads = maps.Clone(reads)
		maps.Copy(reads, over)
	} else if convention != "" && len(m.GGUF.Conventions) > 0 {
		return nil, fmt.Errorf("rope scaling %s states no convention %q", name, convention)
	}
	bound, env := c.rotaryBound(), ggufEnv(f)
	return readRotary1(m, name, m.GGUF, reads, func(p string) any {
		if v, ok := bound[p]; ok {
			return v
		}
		return env(p)
	})
}

// readRotary1 is the reading of method m, name, by src: each of reads
// evaluated over at.
func readRotary1(m *rotaryMethod, name string, src *rotarySource, reads map[string]string, at exprEnv) (*rotaryReading, error) {
	params := map[string]any{}
	for _, k := range slices.Sorted(maps.Keys(reads)) {
		v, err := evalRotaryText(reads[k], at)
		if err != nil {
			return nil, fmt.Errorf("rope scaling %s: %s: %w", name, k, err)
		}
		params[k] = v
	}
	return &rotaryReading{m: m, name: name, src: src, params: params}, nil
}

// rotaryBound is what a method's expressions read of the container stated so
// far: its context, its cos and sin magnitude, its rotary's width and base,
// and whether it takes YaRN's correction range as computed (yarn_exact).
func (c *Config) rotaryBound() map[string]any {
	rot := c.NRot
	if rot == 0 {
		rot = c.HeadDim
	}
	return map[string]any{"context": float64(c.NCtx), "attn_factor": float64(c.AttnFactor),
		"rotated": float64(rot), "base": float64(c.RopeBase), "yarn_exact": c.Flags.Has(jlm.FlagYarnExact)}
}

// clear clears the flags r's divisors consume on c, once they are stated.
func (r *rotaryReading) clear(c *Config) error {
	for _, name := range r.src.Clears {
		f, ok := mjFlags()[name]
		if !ok {
			return fmt.Errorf("rope scaling %s clears %s, which is no flag", r.name, name)
		}
		c.Flags &^= f
	}
	return nil
}

// env is what r's conditions and values read: c's bound values and r's
// parameters.
func (r *rotaryReading) env(c *Config) map[string]any {
	bound := c.rotaryBound()
	maps.Copy(bound, r.params)
	return bound
}

// conds refuses where one of conds is not true over env, by its text, wrap
// wrapping the refusal where it is not nil.
func (r *rotaryReading) conds(conds [][2]string, env map[string]any, wrap error) error {
	for _, k := range conds {
		v, err := evalRotaryText(k[0], func(p string) any { return env[p] })
		switch {
		case err != nil:
			return fmt.Errorf("rope scaling %s: %s: %w", r.name, k[1], err)
		case v == true:
		case wrap != nil:
			return fmt.Errorf("rope scaling %s: %s: %w", r.name, k[1], wrap)
		default:
			return fmt.Errorf("rope scaling %s: %s", r.name, k[1])
		}
	}
	return nil
}

// holds reports whether r states anything on c: refused where a Require or,
// past When, a Check fails, false where When is.
func (r *rotaryReading) holds(c *Config) (bool, error) {
	env := r.env(c)
	if err := r.conds(r.src.Require, env, nil); err != nil {
		return false, err
	}
	if r.src.When != "" {
		v, err := evalRotaryText(r.src.When, func(p string) any { return env[p] })
		if err != nil {
			return false, fmt.Errorf("rope scaling %s: %w", r.name, err)
		}
		if v != true {
			return false, nil
		}
	}
	return true, r.conds(r.src.Check, env, nil)
}

// settle states r on c -- the magnitude, refused where an Implements fails,
// and the fields kept -- and returns its factor lists as rope_factors_short
// and rope_factors_long.
func (r *rotaryReading) settle(c *Config) ([]jlm.Tensor, error) {
	env := r.env(c)
	value := func(what, src string) (any, error) {
		v, err := evalRotaryText(src, func(p string) any { return env[p] })
		if err != nil {
			return nil, fmt.Errorf("rope scaling %s: %s: %w", r.name, what, err)
		}
		return v, nil
	}
	if r.src.Magnitude != "" {
		v, err := value("magnitude", r.src.Magnitude)
		if err != nil {
			return nil, err
		}
		f, ok := v.(float64)
		if !ok {
			return nil, fmt.Errorf("rope scaling %s: the magnitude is %s, not a number", r.name, exprKind(v))
		}
		env["magnitude"] = f
		c.AttnFactor = float32(f)
	}
	if err := r.conds(r.src.Implements, env, ErrNotImplemented); err != nil {
		return nil, err
	}
	for _, k := range slices.Sorted(maps.Keys(r.src.Keeps)) {
		set := rotaryKeeps()[k]
		if set == nil {
			return nil, fmt.Errorf("rope scaling %s keeps %s, which a container does not hold", r.name, k)
		}
		v, err := value(k, r.src.Keeps[k])
		if err != nil {
			return nil, err
		}
		if v == nil {
			continue
		}
		f, ok := v.(float64)
		if !ok {
			return nil, fmt.Errorf("rope scaling %s: %s is %s, not a number", r.name, k, exprKind(v))
		}
		if err := set(c, f); err != nil {
			return nil, fmt.Errorf("rope scaling %s: %w", r.name, err)
		}
	}
	pairs := int(env["rotated"].(float64)) / 2
	var out []jlm.Tensor
	for _, k := range []string{"short", "long"} {
		p, ok := r.src.Factors[k]
		if !ok {
			continue
		}
		l, _ := r.params[p].([]any)
		f := make([]float32, len(l))
		for i, x := range l {
			v, ok := x.(float64)
			if !ok {
				return nil, fmt.Errorf("rope scaling %s: %s %d is %s, not a number", r.name, p, i, exprKind(x))
			}
			f[i] = float32(v)
		}
		if len(f) != pairs {
			return nil, fmt.Errorf("rope scaling %s: %s factors: %s holds %d for %d rotary pairs", r.name, r.m.Describe, p, len(f), pairs)
		}
		t := ropeFreqsTensor(f)
		t.Role, t.Name = jlm.RoleRopeFactorsShort, "rope_factors_short.weight"
		if k == "long" {
			t.Role, t.Name = jlm.RoleRopeFactorsLong, "rope_factors_long.weight"
		}
		out = append(out, t)
	}
	return out, nil
}

// divisors is r's divisor of each of c's rotary pairs, nil for a method that
// states none.
func (r *rotaryReading) divisors(c *Config) ([]float32, error) {
	if r.m.Divisor == "" {
		return nil, nil
	}
	rot := c.rotaryBound()["rotated"].(float64)
	return rotaryDivisors(r.name, int(rot)/2, rot, float64(c.RopeBase), r.params)
}

// stateGGUFRotary states f's rotary scaling, read by its method's GGUF
// statement, on s: its divisors times the per-pair factors s ships, as
// StateRotary states a container's (scaleRotary). The magnitude and the
// fields kept were stated with the config (ropeScalingOf).
func stateGGUFRotary(f *meta.File, s *jlm.Source) error {
	st, _, err := ggufStates(f.Arch())
	if err != nil {
		return err
	}
	convention := ""
	if st != nil {
		convention = st.YarnMagnitude
	}
	c := &Config{Config: s.Config}
	r, err := ggufRotary(f, c, convention)
	if err != nil || r == nil {
		return err
	}
	if ok, err := r.holds(c); err != nil || !ok {
		return err
	}
	div, err := r.divisors(c)
	if err != nil || div == nil {
		return err
	}
	if err := scaleRotary(s, div); err != nil {
		return err
	}
	return r.clear(c)
}

// evalRotaryText is src, a rotary expression, over env.
func evalRotaryText(src string, env exprEnv) (any, error) {
	n, err := parseExpr(src)
	if err != nil {
		return nil, err
	}
	return evalRotary(n, env)
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
	if err := stateMethodFields(c, s); err != nil {
		return err
	}
	return foldYarnScore(c)
}

// foldYarnScore states YaRN's score magnitude in c's attention scale: the
// stated scale (or 1/sqrt of the head width) times (YarnLogMul ln YarnFactor
// + 1)^2, the correction a class's attention squares (DeepSeek's
// softmax_scale), with YarnFactor and YarnLogMul cleared. The angles already
// turn by the factors the container states per pair, so a container states
// no YaRN that a reader could apply again. Blocks of several head widths with
// no stated scale have no one scale to state it in, and are refused.
func foldYarnScore(c *jlm.Config) error {
	if c.YarnLogMul != 0 && c.YarnFactor > 1 {
		base := float64(c.AttnScale)
		if base == 0 {
			for b := range int(c.NLayer) {
				if c.HeadDimAt(b) != c.HeadDim {
					return fmt.Errorf("convert: a YaRN score magnitude over blocks of %d and %d wide heads: %w",
						c.HeadDim, c.HeadDimAt(b), ErrNotImplemented)
				}
			}
			base = 1 / math.Sqrt(float64(c.HeadDim))
		}
		m := float64(c.YarnLogMul)*math.Log(float64(c.YarnFactor)) + 1
		c.AttnScale = float32(base * (m * m))
	}
	c.YarnFactor, c.YarnLogMul = 0, 0
	return nil
}

// stateMethodFields is StateRotary's reading of a container that states its
// scaling by method fields (YaRN's factor and correction range, a linear
// factor): the method's divisors times the factors s ships, the fields
// cleared.
func stateMethodFields(c *jlm.Config, s *jlm.Source) error {
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
	if err := scaleRotary(s, div); err != nil {
		return err
	}
	c.RopeLinear, c.YarnBetaFast, c.YarnBetaSlow = 0, 0, 0
	c.Flags &^= jlm.FlagYarnExact
	return nil
}

// scaleRotary multiplies the per-pair factors s ships by div, a pair each:
// its rope_freqs (or the short factors it ships in their place), stated as
// div where it ships none, and its long factors.
func scaleRotary(s *jlm.Source, div []float32) error {
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
		return scaleFactors(&s.Tensors[long], div)
	}
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
