package convert

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/tok/jinja"
)

// A model.json's graph states its blocks' ops: each a step of the graph
// language, its operands values named by earlier ops, its weights tensors by
// their container names, its numbers expressions over the block's facts
// (blockFacts). The converter evaluates it for every block and the container
// carries each block's graph (jlm.BlockGraphs), which the engine decodes
// rather than builds.
type mjGraph struct {
	Block []mjBlockOp `json:"block"`
	// exprs are its expressions compiled, by their text (check).
	exprs map[string]*jinja.Expr
}

// blockExprSteps is the most steps one of a graph's expressions takes.
const blockExprSteps = 1 << 12

// mjBlockOp is one op of a block. Op names the step: in, norm, layer_norm,
// head_norm, matvec, add, rope, kv_append, attend, act, stated, emit,
// split_sum, mark, routed or out.
type mjBlockOp struct {
	Op string `json:"op"`
	// If, where set, is an expression: the op is the block's where it holds.
	If string `json:"if"`
	// In are the op's operands, values named by earlier ops' Out.
	In mjNames `json:"in"`
	// W are its weights by tensor name (attn_q, attn_q.bias); a name ending
	// in ? is bound where the block stores it.
	W mjNames `json:"w"`
	// Out names the op's value, or a kv_append's history; a later op may
	// name another so.
	Out string `json:"out"`
	// Name is the value's name in the graph, where the step does not name it.
	Name string `json:"name"`
	// Rows is a matvec's output width, Width an input's or an activation's,
	// Alpha an add's scale of its sum (0 and 1 are none), all expressions.
	Rows  string `json:"rows"`
	Width string `json:"width"`
	Alpha string `json:"alpha"`
	// Hist is the history an attend reads, a kv_append's Out.
	Hist string `json:"hist"`
	// Kind is an emit's op, by the graph language's name of it; Outs its
	// results, Attr its attributes by graph.Attrs field name, each an
	// expression (a function by its name), and Bind its weights with their
	// shapes.
	Kind string            `json:"kind"`
	Outs []mjResult        `json:"outs"`
	Attr map[string]string `json:"attr"`
	Bind []mjBinding       `json:"bind"`
	// Of is the statement a stated op runs: router_score, the mixture's
	// score of each expert from its router's logits.
	Of string `json:"of"`
	// Parts is how many equal slices a split_sum adds, an expression.
	Parts string `json:"parts"`
	// From is the mark a routed op's region starts at.
	From string `json:"from"`
}

// mjResult is one of an emit's results: its name in the graph, its width (an
// expression), its element (f32, u32 or i8) and the name later ops read it by.
type mjResult struct {
	Name  string `json:"name"`
	Width string `json:"width"`
	Elem  string `json:"elem"`
	As    string `json:"as"`
}

// mjBinding is one of an emit's weights: a tensor name (a trailing ? binds
// it where stored) and its shape as expressions, or the shape it is stored
// at where K is absent.
type mjBinding struct {
	W      string `json:"w"`
	K      string `json:"k"`
	Rows   string `json:"rows"`
	Sheets string `json:"sheets"`
}

// elems are the elements a result may hold.
func elems() map[string]graph.Elem {
	return map[string]graph.Elem{"": graph.ElemF32, "f32": graph.ElemF32, "u32": graph.ElemU32, "i8": graph.ElemI8}
}

// mjNames is one name or a list of them.
type mjNames []string

func (n *mjNames) UnmarshalJSON(b []byte) error {
	var one string
	if json.Unmarshal(b, &one) == nil {
		*n = mjNames{one}
		return nil
	}
	var many []string
	if err := json.Unmarshal(b, &many); err != nil {
		return fmt.Errorf("want a name or a list of names: %w", err)
	}
	*n = many
	return nil
}

// graphSteps are each step's operand and weight counts: the least, and the
// most (-1 for any).
func graphSteps() map[string][4]int {
	return map[string][4]int{
		"in":          {0, 0, 0, 0},
		"norm":        {1, 1, 1, 1},
		"layer_norm":  {1, 1, 1, -1},
		"head_norm":   {1, 1, 1, 1},
		"matvec":      {1, 1, 1, -1},
		"add":         {2, 2, 0, 0},
		"rope":        {1, 1, 0, 0},
		"kv_append":   {2, 2, 0, 0},
		"attend":      {1, 1, 0, -1},
		"act":         {1, -1, 0, 0},
		"stated":      {1, -1, 0, 0},
		"emit":        {0, -1, 0, 0},
		"split_sum":   {1, 1, 0, 0},
		"mark":        {0, 0, 0, 0},
		"routed":      {0, 0, 0, 0},
		"let":         {1, 1, 0, 0},
		"sigmoid_mul": {2, 2, 0, 0},
		"out":         {1, -1, 0, 0},
	}
}

// statements are what a stated op may run.
func statements() map[string]bool { return map[string]bool{"router_score": true} }

// check refuses a graph naming a step this converter does not take, an op
// of the wrong arity, a value read before an op names it, an expression
// that does not parse or an op with no output it needs.
func (g *mjGraph) check() error {
	if len(g.Block) == 0 {
		return fmt.Errorf("a graph states no block")
	}
	named := map[string]bool{}
	for i, o := range g.Block {
		n, ok := graphSteps()[o.Op]
		if !ok {
			return fmt.Errorf("op %d: %q is not a step of the graph language", i, o.Op)
		}
		if len(o.In) < n[0] || n[1] >= 0 && len(o.In) > n[1] || len(o.W) < n[2] || n[3] >= 0 && len(o.W) > n[3] {
			return fmt.Errorf("op %d (%s): %d operands and %d weights", i, o.Op, len(o.In), len(o.W))
		}
		for _, v := range o.In {
			if !named[v] {
				return fmt.Errorf("op %d (%s) reads %q before an op names it", i, o.Op, v)
			}
		}
		if o.Op == "attend" && !named[o.Hist] {
			return fmt.Errorf("op %d attends history %q before an op names it", i, o.Hist)
		}
		if o.Op == "routed" && !named[o.From] {
			return fmt.Errorf("op %d's routed region starts at %q, which no op marks before it", i, o.From)
		}
		xs := []string{o.If, o.Rows, o.Width, o.Alpha, o.Parts}
		for _, r := range o.Outs {
			xs = append(xs, r.Width)
		}
		for _, b := range o.Bind {
			xs = append(xs, b.K, b.Rows, b.Sheets)
		}
		for f, x := range o.Attr {
			if fl, ok := reflect.TypeOf(graph.Attrs{}).FieldByName(f); !ok || !attrSettable(fl.Type) {
				return fmt.Errorf("op %d (%s): %q is not an attribute a stated op sets", i, o.Op, f)
			}
			xs = append(xs, x)
		}
		for _, x := range xs {
			if err := g.compile(x); err != nil {
				return fmt.Errorf("op %d (%s): %w", i, o.Op, err)
			}
		}
		ws := slices.Clone(o.W)
		for _, b := range o.Bind {
			ws = append(ws, b.W)
		}
		for _, w := range ws {
			if _, ok := blockRole(strings.TrimSuffix(w, "?")); !ok {
				return fmt.Errorf("op %d (%s): %q is not a tensor name the container knows", i, o.Op, w)
			}
		}
		switch {
		case o.Op == "out", o.Op == "routed":
		case o.Op == "emit":
			if graph.KindNamed(o.Kind) == graph.OpInvalid {
				return fmt.Errorf("op %d emits %q, which is not an op of the graph language", i, o.Kind)
			}
			if len(o.Outs) == 0 {
				return fmt.Errorf("op %d (%s) states no result", i, o.Kind)
			}
			for k, r := range o.Outs {
				if _, ok := elems()[r.Elem]; !ok || r.Width == "" || r.Name == "" {
					return fmt.Errorf("op %d (%s): result %d is %q of %q wide, element %q", i, o.Kind, k, r.Name, r.Width, r.Elem)
				}
				if as := r.As; as != "" {
					named[as] = true
				}
			}
			if o.Out != "" {
				named[o.Out] = true
			}
			continue
		case o.Out == "":
			return fmt.Errorf("op %d (%s) names no output", i, o.Op)
		case o.Op == "in" && o.Width == "", o.Op == "matvec" && o.Rows == "", o.Op == "act" && o.Width == "",
			o.Op == "stated" && o.Width == "":
			return fmt.Errorf("op %d (%s) states no width", i, o.Op)
		case o.Op == "stated" && !statements()[o.Of]:
			return fmt.Errorf("op %d states %q, which is not a statement a block runs", i, o.Of)
		case o.Op == "split_sum" && (o.Parts == "" || o.Name == ""):
			return fmt.Errorf("op %d's split_sum states no parts or no name", i)
		}
		named[o.Out] = true
	}
	if g.Block[len(g.Block)-1].Op != "out" {
		return fmt.Errorf("a block's graph ends in an out")
	}
	return nil
}

// compile compiles one of the graph's expressions, refusing one that reads
// a name that is not a block's fact.
func (g *mjGraph) compile(src string) error {
	if src == "" || g.exprs[src] != nil {
		return nil
	}
	x, err := jinja.CompileExpr(src)
	if err != nil {
		return err
	}
	known := (&blockFacts{c: &jlm.Config{}}).vars()
	for _, n := range x.Names() {
		if _, ok := known[n]; !ok {
			return fmt.Errorf("expression %q reads %q, which is not a fact of a block", src, n)
		}
	}
	if g.exprs == nil {
		g.exprs = map[string]*jinja.Expr{}
	}
	g.exprs[src] = x
	return nil
}

// value is expression src's value over a block's facts.
func (g *mjGraph) value(src string, vars map[string]jinja.Value) (jinja.Value, error) {
	x, ok := g.exprs[src]
	if !ok {
		return jinja.Undefined(), fmt.Errorf("expression %q was never checked", src)
	}
	v, err := x.Eval(vars, blockExprSteps)
	if err == nil && v.IsUndefined() {
		err = fmt.Errorf("expression %q is undefined", src)
	}
	return v, err
}

// num is expression src's value as a number.
func (g *mjGraph) num(src string, vars map[string]jinja.Value) (float64, error) {
	v, err := g.value(src, vars)
	switch {
	case err != nil:
		return 0, err
	case v.IsInt():
		return float64(v.AsInt()), nil
	case v.IsFloat():
		return v.AsFloat(), nil
	}
	return 0, fmt.Errorf("expression %q is %v, not a number", src, v)
}

// flag is expression src's value as true or false.
func (g *mjGraph) flag(src string, vars map[string]jinja.Value) (bool, error) {
	v, err := g.value(src, vars)
	if err != nil {
		return false, err
	}
	if !v.IsBool() {
		return false, fmt.Errorf("expression %q is %v, not true or false", src, v)
	}
	return v.AsBool(), nil
}

// blockRole is the role a graph's weight names, in the tensor names a
// model.json's tensors map onto: attn_q for attn_q.weight, attn_q.bias for
// itself.
func blockRole(name string) (jlm.Role, bool) {
	if r, ok := roleOf(name + ".weight"); ok {
		return r, true
	}
	return roleOf(name)
}

// StateBlockGraphs states every block's graph in c from g, held to the
// tensors s stores for it.
func StateBlockGraphs(c *Config, s *jlm.Source, g *mjGraph) error {
	n := int(c.NLayer + c.NMTP)
	if c.NMTP != 0 {
		return fmt.Errorf("convert: a model.json's graph for prediction blocks: %w", ErrNotImplemented)
	}
	var out jlm.BlockGraphs
	at := map[string]uint16{}
	for li := range n {
		bg, err := blockGraphOf(c.Config, s, li, g)
		if err != nil {
			return fmt.Errorf("convert: block %d's stated graph: %w", li, err)
		}
		b, err := graph.Encode(bg)
		if err != nil {
			return err
		}
		i, ok := at[string(b)]
		if !ok {
			if len(out.Graphs) > math.MaxUint16 {
				return fmt.Errorf("convert: more than %d distinct block graphs", math.MaxUint16)
			}
			i = uint16(len(out.Graphs))
			at[string(b)] = i
			out.Graphs = append(out.Graphs, string(b))
		}
		out.Of = append(out.Of, i)
	}
	c.Blocks = out
	return nil
}

// blockGraphOf is block li's graph as g states it.
func blockGraphOf(c *jlm.Config, s *jlm.Source, li int, g *mjGraph) (*graph.Graph, error) {
	bf := blockFactsOf(c, s, li)
	b := graph.NewBlock(li)
	vals := map[string]graph.ValueID{}
	hists := map[string]graph.StateRef{}
	marks := map[string]int{}
	vars := bf.vars()
	num := func(x string) (float64, error) { return g.num(x, vars) }
	for i, o := range g.Block {
		if o.If != "" {
			on, err := g.flag(o.If, vars)
			if err != nil {
				return nil, fmt.Errorf("op %d (%s): %w", i, o.Op, err)
			}
			if !on {
				continue
			}
		}
		in := make([]graph.ValueID, len(o.In))
		for k, v := range o.In {
			id, ok := vals[v]
			if !ok {
				return nil, fmt.Errorf("op %d (%s) reads %q, which no op this block runs names", i, o.Op, v)
			}
			in[k] = id
		}
		var w []jlm.Role
		for _, name := range o.W {
			r, _ := blockRole(strings.TrimSuffix(name, "?"))
			if _, ok := bf.facts.Inventory[r]; !ok && strings.HasSuffix(name, "?") {
				continue
			}
			w = append(w, r)
		}
		var y graph.ValueID
		switch o.Op {
		case "in":
			width, err := num(o.Width)
			if err != nil {
				return nil, err
			}
			y = b.In(o.Out, int(width))
		case "norm":
			y = b.Norm(in[0], w[0], bf.eps)
		case "layer_norm":
			y = b.LayerNorm(in[0], w[0], bf.eps, w[1:]...)
		case "head_norm":
			y = b.HeadNorm(in[0], w[0], bf.headDim, bf.eps)
		case "matvec":
			rows, err := num(o.Rows)
			if err != nil {
				return nil, err
			}
			y = b.MatVec(in[0], w[0], int(rows), w[1:]...)
		case "add":
			alpha := 0.0
			if o.Alpha != "" {
				var err error
				if alpha, err = num(o.Alpha); err != nil {
					return nil, err
				}
			}
			if alpha != 0 && alpha != 1 {
				y = b.Emit(graph.OpAdd, in, []graph.Result{{Name: "sum", Width: b.G.Values[in[0]].Width}}, nil,
					graph.Attrs{Alpha: alpha}, nil, nil)[0]
			} else {
				y = b.Add(in[0], in[1])
			}
		case "rope":
			a, err := bf.rope()
			if err != nil {
				return nil, err
			}
			name := o.Name
			if name == "" {
				name = "rope"
			}
			y = b.Emit(graph.OpRope, in, []graph.Result{{Name: name, Width: b.G.Values[in[0]].Width}}, nil, a, nil, nil)[0]
		case "kv_append":
			hists[o.Out] = graph.StateRef{State: graph.StateKV, Version: b.KVAppend(in[0], in[1], 0)}
			continue
		case "attend":
			h, ok := hists[o.Hist]
			if !ok {
				return nil, fmt.Errorf("op %d attends history %q, which no op this block runs appends", i, o.Hist)
			}
			y = b.AttendOf(in[0], h, bf.attend(), w...)
		case "act":
			width, err := num(o.Width)
			if err != nil {
				return nil, err
			}
			name := o.Name
			if name == "" {
				name = "act"
			}
			if y, err = b.Stated(name, c.ActAt(li), in, int(width), bf.facts); err != nil {
				return nil, fmt.Errorf("op %d's stated activation: %w", i, err)
			}
		case "stated":
			width, err := num(o.Width)
			if err != nil {
				return nil, err
			}
			name := o.Name
			if name == "" {
				name = o.Of
			}
			if y, err = b.Stated(name, &c.RouterScore, in, int(width), bf.facts); err != nil {
				return nil, fmt.Errorf("op %d's stated %s: %w", i, o.Of, err)
			}
		case "emit":
			outs, err := emitted(b, g, o, in, bf, vars)
			if err != nil {
				return nil, fmt.Errorf("op %d (%s): %w", i, o.Kind, err)
			}
			for k, r := range o.Outs {
				if r.As != "" {
					vals[r.As] = outs[k]
				}
			}
			if o.Out != "" {
				vals[o.Out] = outs[0]
			}
			continue
		case "split_sum":
			parts, err := num(o.Parts)
			if err != nil {
				return nil, err
			}
			n, wd := int(parts), b.G.Values[in[0]].Width
			if n < 1 || wd%n != 0 {
				return nil, fmt.Errorf("op %d splits %d into %v parts", i, wd, parts)
			}
			for j := range n {
				s := b.Emit(graph.OpSlice, in, []graph.Result{{Name: o.Name, Width: wd / n}}, nil,
					graph.Attrs{Offset: j * wd / n, Width: wd / n}, nil, nil)[0]
				if j == 0 {
					y = s
				} else {
					y = b.Add(y, s)
				}
			}
		case "let":
			y = in[0]
		case "sigmoid_mul":
			name := o.Name
			if name == "" {
				name = "gated"
			}
			y = b.Expr(name, b.G.Values[in[1]].Width, []graph.Term{
				{Fn: graph.MapSigmoid, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}}},
				{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgTerm, Index: 0}, {Kind: graph.ArgIn, Index: 1}}},
			}, nil, in...)
		case "mark":
			marks[o.Out] = len(b.G.Ops)
			continue
		case "routed":
			b.G.Routed = graph.Region{Lo: marks[o.From], Hi: len(b.G.Ops)}
			continue
		case "out":
			b.Out(in...)
			continue
		}
		vals[o.Out] = y
	}
	if err := graph.Verify(b.G, bf.facts); err != nil {
		return nil, err
	}
	if u := graph.Unbound(b.G, bf.facts); len(u) != 0 {
		return nil, fmt.Errorf("the block stores %v, which its graph does not bind", u)
	}
	return b.G, nil
}

// emitted is an emit's op on b over in: its results, weights and
// attributes evaluated over the block's facts.
func emitted(b *graph.Builder, g *mjGraph, o mjBlockOp, in []graph.ValueID, bf *blockFacts, vars map[string]jinja.Value) ([]graph.ValueID, error) {
	num := func(x string) (float64, error) { return g.num(x, vars) }
	var outs []graph.Result
	for _, r := range o.Outs {
		w, err := num(r.Width)
		if err != nil {
			return nil, err
		}
		outs = append(outs, graph.Result{Name: r.Name, Width: int(w), Elem: elems()[r.Elem]})
	}
	var ws []graph.Binding
	for _, bd := range o.Bind {
		r, _ := blockRole(strings.TrimSuffix(bd.W, "?"))
		sh, held := bf.facts.Inventory[r]
		if !held && strings.HasSuffix(bd.W, "?") {
			continue
		}
		if bd.K == "" {
			ws = append(ws, graph.Binding{Role: r, K: sh.K, Rows: sh.Rows})
			continue
		}
		bn := graph.Binding{Role: r}
		for _, f := range []struct {
			x  string
			to *int
		}{{bd.K, &bn.K}, {bd.Rows, &bn.Rows}, {bd.Sheets, &bn.Sheets}} {
			if f.x == "" {
				continue
			}
			v, err := num(f.x)
			if err != nil {
				return nil, err
			}
			*f.to = int(v)
		}
		ws = append(ws, bn)
	}
	var a graph.Attrs
	for f, x := range o.Attr {
		v, err := g.value(x, vars)
		if err != nil {
			return nil, fmt.Errorf("attribute %s: %w", f, err)
		}
		if err := setAttr(&a, f, v); err != nil {
			return nil, err
		}
	}
	return b.Emit(graph.KindNamed(o.Kind), in, outs, ws, a, nil, nil), nil
}

// attrSettable says a stated op may set an attribute of type t: a number, a
// flag, or a function by its name.
func attrSettable(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Int, reflect.Float64, reflect.Bool, reflect.Uint8:
		return true
	}
	return false
}

// setAttr sets a's field f to v, an expression's value: an integer field
// an integral number, a function its name.
func setAttr(a *graph.Attrs, f string, v jinja.Value) error {
	fv := reflect.ValueOf(a).Elem().FieldByName(f)
	if !fv.IsValid() || !attrSettable(fv.Type()) {
		return fmt.Errorf("%q is not an attribute a stated op sets", f)
	}
	if fv.Type() == reflect.TypeOf(jlm.Fn(0)) {
		var fn jlm.Fn
		if v.IsString() {
			fn = fnNamed(v.AsString())
		}
		if fn == 0 {
			return fmt.Errorf("attribute %s is %v, not a function's name", f, v)
		}
		fv.SetUint(uint64(fn))
		return nil
	}
	if fv.Kind() == reflect.Bool {
		if !v.IsBool() {
			return fmt.Errorf("attribute %s is %v, not a flag", f, v)
		}
		fv.SetBool(v.AsBool())
		return nil
	}
	var n float64
	switch {
	case v.IsInt():
		n = float64(v.AsInt())
	case v.IsFloat():
		n = v.AsFloat()
	default:
		return fmt.Errorf("attribute %s is %v, not a number", f, v)
	}
	switch fv.Kind() {
	case reflect.Float64:
		fv.SetFloat(n)
	case reflect.Int:
		if n != math.Trunc(n) || math.IsInf(n, 0) {
			return fmt.Errorf("attribute %s is %v, not an integer", f, n)
		}
		fv.SetInt(int64(n))
	case reflect.Uint8:
		if n != math.Trunc(n) || n < 0 || n > math.MaxUint8 {
			return fmt.Errorf("attribute %s is %v, not a small integer", f, n)
		}
		fv.SetUint(uint64(n))
	}
	return nil
}

// fnNamed is the function named name, 0 for none.
func fnNamed(name string) jlm.Fn {
	for f := jlm.Fn(1); f < math.MaxUint8; f++ {
		if s := f.String(); s == name {
			return f
		} else if strings.HasPrefix(s, "fn") {
			return 0
		}
	}
	return 0
}

// blockFacts are what a block's graph is stated over: the block's tensors
// and its geometry, attention and rotary as the container's config states
// them, as the engine read them before the converter stated its graphs
// (model.Config).
type blockFacts struct {
	c       *jlm.Config
	li      int
	facts   graph.Facts
	dense   map[jlm.Role]bool
	headDim int
	eps     float64
}

func blockFactsOf(c *jlm.Config, s *jlm.Source, li int) *blockFacts {
	bf := &blockFacts{c: c, li: li, facts: graph.Facts{Inventory: map[jlm.Role]graph.Shape{}}, dense: map[jlm.Role]bool{},
		headDim: int(c.HeadDimAt(li)), eps: float64(c.RMSEps)}
	for _, t := range s.Tensors {
		if t.Block == jlm.DenseBlock {
			bf.dense[t.Role] = true
			continue
		}
		if int(t.Block) != li || t.Role.Vision() || nextnRole(t.Role) || t.Index >= 0 {
			continue
		}
		sh := graph.Shape{K: int(t.Dims[0])}
		if t.NDim >= 2 {
			sh.Rows = int(t.Dims[1])
		}
		if t.NDim >= 3 {
			sh.Sheets = int(t.Dims[2])
		}
		bf.facts.Inventory[t.Role] = sh
	}
	return bf
}

// nextnRole is a prediction block's entry's tensor, which its entry's graph
// binds, not its body's.
func nextnRole(r jlm.Role) bool {
	switch r {
	case jlm.RoleNextnEHProj, jlm.RoleNextnENorm, jlm.RoleNextnHNorm, jlm.RoleNextnHeadNorm, jlm.RoleNextnHead, jlm.RoleNextnEmbd:
		return true
	}
	return false
}

// vars are the facts as a graph's expressions read them: the block's index,
// its geometry, attention, mixture and numbers, stored(name) whether it
// stores a tensor and rows(name) a stored one's rows.
func (bf *blockFacts) vars() map[string]jinja.Value {
	c, li := bf.c, bf.li
	i := func(n uint32) jinja.Value { return jinja.NewInt(int64(n)) }
	tensor := func(name string, args []jinja.Value) (jlm.Role, graph.Shape, bool, error) {
		if len(args) != 1 || !args[0].IsString() {
			return 0, graph.Shape{}, false, fmt.Errorf("%s takes one tensor name", name)
		}
		r, ok := blockRole(args[0].AsString())
		if !ok {
			return 0, graph.Shape{}, false, fmt.Errorf("%s(%q): not a tensor name the container knows", name, args[0].AsString())
		}
		sh, held := bf.facts.Inventory[r]
		return r, sh, held, nil
	}
	return map[string]jinja.Value{
		"layer":              jinja.NewInt(int64(li)),
		"embd":               i(c.NEmbd),
		"heads":              i(c.NHead),
		"kv_heads":           i(c.NKVHeadAt(li)),
		"head_dim":           jinja.NewInt(int64(bf.headDim)),
		"ffn":                i(c.NFFNAt(li)),
		"eps":                jinja.NewFloat(bf.eps),
		"residual_scale":     jinja.NewFloat(float64(c.ResidualScale)),
		"rope_at":            jinja.NewBool(bf.ropeAt()),
		"sliding":            jinja.NewBool(bf.sliding()),
		"inf":                jinja.NewFloat(math.Inf(1)),
		"experts":            i(c.NExpert),
		"experts_used":       i(c.NExpertUsed),
		"expert_groups":      i(c.NExpertGroup),
		"expert_groups_used": i(c.NExpertGroupUsed),
		"expert_scale":       jinja.NewFloat(float64(c.ExpertScale)),
		"expert_norm":        jinja.NewBool(!c.Flags.Has(jlm.FlagNoExpertNorm)),
		"expert_weight_in":   jinja.NewBool(c.Flags.Has(jlm.FlagExpertWeightIn)),
		"moe":                jinja.NewBool(bf.moe()),
		"stored": jinja.NewCallable("stored", func(args []jinja.Value, _ map[string]jinja.Value) (jinja.Value, error) {
			_, _, held, err := tensor("stored", args)
			return jinja.NewBool(held), err
		}),
		"rows": jinja.NewCallable("rows", func(args []jinja.Value, _ map[string]jinja.Value) (jinja.Value, error) {
			_, sh, held, err := tensor("rows", args)
			if err == nil && !held {
				err = fmt.Errorf("rows(%q): the block stores no such tensor", args[0].AsString())
			}
			return jinja.NewInt(int64(sh.Rows)), err
		}),
	}
}

// sliding says the block attends within the window: every layer but each
// period's last, or where the period's global layer is its first, every
// layer but each period's first.
func (bf *blockFacts) sliding() bool {
	c, li := bf.c, bf.li
	w, p := int(c.SWAWindow), int(c.SWAPeriod)
	if c.Flags2.Has(jlm.Flag2SWAGlobalFirst) {
		return w > 0 && p > 1 && li%p != 0
	}
	return w > 0 && p > 0 && li%p < p-1
}

// moe says the block's feed-forward is the mixture: the model's is, the
// block is past the leading dense run, on the mixture's step, and stores a
// router.
func (bf *blockFacts) moe() bool {
	c, li := bf.c, bf.li
	if c.NExpert == 0 || li < int(c.NDenseLead) {
		return false
	}
	if _, ok := bf.facts.Inventory[jlm.RoleRouter]; !ok {
		return false
	}
	return c.MoEStep <= 1 || (li+1)%int(c.MoEStep) == 0
}

// ropeAt says the block turns q and k: not where the model encodes no
// position, nor on a period's global layer where those skip the rotary.
func (bf *blockFacts) ropeAt() bool {
	c, li := bf.c, bf.li
	if c.Flags.Has(jlm.FlagNoPosEnc) {
		return false
	}
	p := int(c.SWAPeriod)
	return !(c.Flags.Has(jlm.FlagNoPEGlobal) && !(p > 0 && li%p < p-1))
}

// attend is the block's attention: its heads, the score's scale (YaRN's
// correction squared where the config states one), softcap and window, and
// both ways where the model attends so.
func (bf *blockFacts) attend() graph.Attrs {
	c := bf.c
	scale := 1 / math.Sqrt(float64(c.HeadDim))
	if c.AttnScale != 0 {
		scale = float64(c.AttnScale)
	}
	if c.YarnLogMul != 0 && c.YarnFactor > 1 {
		m := float64(c.YarnLogMul)*math.Log(float64(c.YarnFactor)) + 1
		scale *= m * m
	}
	bidir := c.Flags.Has(jlm.FlagNonCausal)
	a := graph.Attrs{NHead: int(c.NHead), NKVHead: int(c.NKVHeadAt(bf.li)), HeadDim: bf.headDim, Scale: scale,
		Softcap: float64(c.AttnSoftcap), Bidir: bidir, RowWindows: bidir && c.Flags2.Has(jlm.Flag2MediaPrefix)}
	if bf.sliding() {
		a.Window, a.Chunked = int(c.SWAWindow), c.Flags.Has(jlm.FlagSWAChunked)
	}
	return a
}

// rope is the rotary the block turns by: the sliding layers' where the
// model trains them their own base or width, the model's otherwise, with
// its per-pair factors where it stores them or scales its rotary.
func (bf *blockFacts) rope() (graph.Attrs, error) {
	c := bf.c
	if c.RopeSections != [4]uint32{} || c.Flags.Has(jlm.FlagRopeXD) {
		return graph.Attrs{}, fmt.Errorf("a stated graph's rotary over sections: %w", ErrNotImplemented)
	}
	if len(c.CompKinds) != 0 {
		return graph.Attrs{}, fmt.Errorf("a stated graph's rotary in a compressing model: %w", ErrNotImplemented)
	}
	nrot, base, scale := int(c.NRot), float64(c.RopeBase), float64(c.AttnFactor)
	if scale == 0 {
		scale = 1
	}
	freqs := bf.dense[jlm.RoleRopeFreqs] || bf.dense[jlm.RoleRopeFactorsShort] || c.YarnFactor > 0 || c.RopeLinear > 0
	local := 0
	for i := range c.Geoms {
		if (&blockFacts{c: c, li: i}).sliding() {
			local = max(local, int(c.NRotAt(i)))
		}
	}
	if local == 0 {
		local = nrot
	}
	if (c.RopeBaseSWA != 0 || local != nrot) && bf.sliding() {
		if c.RopeBaseSWA != 0 {
			base = float64(c.RopeBaseSWA)
		}
		nrot, freqs = local, false
		if c.RopeScaleSWA != 0 {
			scale = float64(c.RopeScaleSWA)
		}
	}
	layout := graph.RopeNorm
	if c.Flags.Has(jlm.FlagRopeNeox) {
		layout = graph.RopeNeox
	}
	a := graph.Attrs{HeadDim: bf.headDim, NRot: nrot, Base: base, Layout: layout, Freqs: freqs, Mscale: scale}
	if c.Flags2.Has(jlm.Flag2RopeTail) {
		a.Offset = bf.headDim - nrot
	}
	return a, nil
}
