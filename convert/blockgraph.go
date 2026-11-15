package convert

import (
	"cmp"
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
	// shapes. On an attend or a rope, Attr sets attributes over the block's.
	Kind string            `json:"kind"`
	Outs []mjResult        `json:"outs"`
	Attr map[string]string `json:"attr"`
	Bind []mjBinding       `json:"bind"`
	// Of is the statement a stated op runs (statements).
	Of string `json:"of"`
	// Advances are the block states an emit reads at their version in force
	// and writes the next of (states).
	Advances []string `json:"advances"`
	// Terms are an expr's elementwise function, each term a function of
	// the operands, constants, the bound vectors' elements and earlier
	// terms (graph.Term).
	Terms []mjTerm `json:"terms"`
	// Parts is how many equal slices a split_sum adds, an expression.
	Parts string `json:"parts"`
	// From is the mark a routed op's region starts at.
	From string `json:"from"`
	// Eps is a norm's epsilon where it is not the block's, an expression.
	Eps string `json:"eps"`
	// Int8, where it holds (an expression), has a matvec read its input's
	// int8 codes per row, the product scaled by the row's scale and the
	// biases added after; an input is quantized once for every matvec.
	Int8 string `json:"int8"`
}

// mjTerm is one term of an expr: a function by name and up to three
// arguments.
type mjTerm struct {
	Fn   string  `json:"fn"`
	Args []mjArg `json:"args"`
}

// mjArg is an expr term's argument: operand In, earlier term Term, element
// Elem of bound vector W, or the constant Const (an expression); exactly one.
type mjArg struct {
	In    *int   `json:"in"`
	Term  *int   `json:"term"`
	W     *int   `json:"w"`
	Elem  int    `json:"elem"`
	Const string `json:"const"`
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
		"silu":        {1, 1, 0, 0},
		"silu_gated":  {2, 2, 0, 0},
		"expr":        {0, -1, 0, 0},
		"out":         {1, -1, 0, 0},
	}
}

// statements are what a stated op may run: the router's score of each
// expert and its picks' weights (expressions), and the delta rule's decay
// and gate (programs).
func statements() map[string]bool {
	return map[string]bool{"router_score": true, "router_weight": true, "qkv_act": true, "delta_decay": true, "delta_gate": true}
}

// states are the block states an emit may advance, by name.
func states() map[string]graph.State {
	return map[string]graph.State{"kv": graph.StateKV, "rec.conv": graph.StateRecConv, "rec.state": graph.StateRecState}
}

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
		xs := []string{o.If, o.Rows, o.Width, o.Alpha, o.Parts, o.Int8, o.Eps}
		for k, tm := range o.Terms {
			if fnNamed(tm.Fn) == 0 || len(tm.Args) == 0 || len(tm.Args) > 3 {
				return fmt.Errorf("op %d (%s): term %d is %q of %d arguments", i, o.Op, k, tm.Fn, len(tm.Args))
			}
			for _, a := range tm.Args {
				set := 0
				for _, p := range []bool{a.In != nil, a.Term != nil, a.W != nil, a.Const != ""} {
					if p {
						set++
					}
				}
				if set != 1 || a.Term != nil && *a.Term >= k {
					return fmt.Errorf("op %d (%s): term %d's argument is not one operand, earlier term, vector or constant", i, o.Op, k)
				}
				xs = append(xs, a.Const)
			}
		}
		if o.Op == "expr" && (len(o.Terms) == 0 || o.Width == "") {
			return fmt.Errorf("op %d's expr states no terms or no width", i)
		}
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
		for _, st := range o.Advances {
			if _, ok := states()[st]; !ok || o.Op != "emit" {
				return fmt.Errorf("op %d (%s) advances %q, which is not a block state an emit advances", i, o.Op, st)
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
			o.Op == "stated" && (o.Of == "router_score" || o.Of == "router_weight") && o.Width == "":
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
	if r, ok := roleOf(name); ok {
		return r, true
	}
	// A tensor the converter makes, by splitting or folding, has the
	// container's name alone (ssm_x.dt of ssm_x); the block roles are the
	// low codes.
	for r := jlm.Role(1); r < 2048; r++ {
		if r.Valid() && r.String() == name {
			return r, true
		}
	}
	return 0, false
}

// StateBlockGraphs states every block's graph in c from g, held to the
// tensors s stores for it.
func StateBlockGraphs(c *Config, s *jlm.Source, g *mjGraph) error {
	n := int(c.NLayer + c.NMTP)
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
	codes := map[graph.ValueID][2]graph.ValueID{}
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
			eps := bf.eps
			if o.Eps != "" {
				var err error
				if eps, err = num(o.Eps); err != nil {
					return nil, err
				}
			}
			y = b.Norm(in[0], w[0], eps)
		case "layer_norm":
			y = b.LayerNorm(in[0], w[0], bf.eps, w[1:]...)
		case "head_norm":
			hd := bf.headDim
			if o.Width != "" {
				n, err := num(o.Width)
				if err != nil {
					return nil, err
				}
				hd = int(n)
			}
			y = b.HeadNorm(in[0], w[0], hd, bf.eps)
		case "matvec":
			rows, err := num(o.Rows)
			if err != nil {
				return nil, err
			}
			q8 := false
			if o.Int8 != "" {
				if q8, err = g.flag(o.Int8, vars); err != nil {
					return nil, fmt.Errorf("op %d (matvec): %w", i, err)
				}
			}
			if !q8 {
				y = b.MatVec(in[0], w[0], int(rows), w[1:]...)
				break
			}
			q, ok := codes[in[0]]
			if !ok {
				q[0], q[1] = b.Int8Codes(in[0])
				codes[in[0]] = q
			}
			y = b.Map("scaled", graph.Attrs{Fn: graph.MapMul}, nil, int(rows), b.MatVec(q[0], w[0], int(rows)), q[1])
			for _, br := range w[1:] {
				y = b.Map("biased", graph.Attrs{Fn: graph.MapAdd}, []graph.Binding{{Role: br, K: int(rows)}}, int(rows), y)
			}
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
			if a, err = overridden(a, g, o, vars); err != nil {
				return nil, fmt.Errorf("op %d (rope): %w", i, err)
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
			a, err := overridden(bf.attend(), g, o, vars)
			if err != nil {
				return nil, fmt.Errorf("op %d (attend): %w", i, err)
			}
			y = b.AttendOf(in[0], h, a, w...)
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
			var err error
			if y, err = stated(b, c, o, in, bf, num); err != nil {
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
		case "expr":
			var err error
			if y, err = exprOp(b, g, o, in, bf, vars); err != nil {
				return nil, fmt.Errorf("op %d (expr): %w", i, err)
			}
		case "silu_gated":
			name := o.Name
			if name == "" {
				name = "gated"
			}
			y = b.Expr(name, b.G.Values[in[1]].Width, []graph.Term{
				{Fn: graph.MapSigmoid, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}}},
				{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}, {Kind: graph.ArgTerm, Index: 0}}},
				{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgTerm, Index: 1}, {Kind: graph.ArgIn, Index: 1}}},
			}, nil, in...)
		case "silu":
			name := o.Name
			if name == "" {
				name = "silu"
			}
			y = b.Expr(name, b.G.Values[in[0]].Width, []graph.Term{
				{Fn: graph.MapSigmoid, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}}},
				{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}, {Kind: graph.ArgTerm, Index: 0}}},
			}, nil, in[0])
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
	ws, err := bindings(o, bf, num)
	if err != nil {
		return nil, err
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
	var reads, writes []graph.StateRef
	for _, name := range o.Advances {
		st := states()[name]
		v := b.G.StateVersion(st)
		reads, writes = append(reads, graph.StateRef{State: st, Version: v}), append(writes, graph.StateRef{State: st, Version: v + 1})
	}
	return b.Emit(graph.KindNamed(o.Kind), in, outs, ws, a, reads, writes), nil
}

// routerWeight is each of k picks' weight: the stated row rw over the logits
// with the earlier picks removed and the pick's logit, read at the pick.
func routerWeight(b *graph.Builder, rw *jlm.Expr, logits, choose, sel graph.ValueID, k int, f graph.Facts) (graph.ValueID, error) {
	e := b.G.Values[logits].Width
	take := func(name string, v graph.ValueID) graph.ValueID {
		return b.Emit(graph.OpTake, []graph.ValueID{v, sel}, []graph.Result{{Name: name, Width: k}}, nil, graph.Attrs{}, nil, nil)[0]
	}
	slice1 := func(name string, v graph.ValueID, j int) graph.ValueID {
		return b.Emit(graph.OpSlice, []graph.ValueID{v}, []graph.Result{{Name: name, Width: 1}}, nil, graph.Attrs{Offset: j, Width: 1}, nil, nil)[0]
	}
	rank := b.Emit(graph.OpRank, []graph.ValueID{choose}, []graph.Result{{Name: "router.rank", Width: e, Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)[0]
	picked := take("router.picked", logits)
	var ws []graph.ValueID
	for j := range k {
		rest := logits
		if j > 0 {
			earlier := b.Map("router.earlier", graph.Attrs{Fn: graph.MapLess, Const: float64(j)}, nil, e, rank)
			rest = b.Expr("router.rest", e, []graph.Term{{Fn: graph.MapWhere, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0},
				{Kind: graph.ArgConst, Const: math.Inf(-1)}, {Kind: graph.ArgIn, Index: 1}}}}, nil, earlier, logits)
		}
		row, err := b.Stated("router.weight", rw, []graph.ValueID{rest, slice1("router.pick", picked, j)}, e, f)
		if err != nil {
			return 0, err
		}
		ws = append(ws, slice1("router.w", take("router.row", row), j))
	}
	if k == 1 {
		return ws[0], nil
	}
	return b.Emit(graph.OpConcat, ws, []graph.Result{{Name: "router.w", Width: k}}, nil, graph.Attrs{}, nil, nil)[0], nil
}

// overridden is a with each attribute o states set over it.
func overridden(a graph.Attrs, g *mjGraph, o mjBlockOp, vars map[string]jinja.Value) (graph.Attrs, error) {
	for f, x := range o.Attr {
		v, err := g.value(x, vars)
		if err != nil {
			return a, fmt.Errorf("attribute %s: %w", f, err)
		}
		if err := setAttr(&a, f, v); err != nil {
			return a, err
		}
	}
	return a, nil
}

// exprOp is an expr's op on b over in: its terms with their constants
// evaluated over the block's facts, and its bound vectors.
func exprOp(b *graph.Builder, g *mjGraph, o mjBlockOp, in []graph.ValueID, bf *blockFacts, vars map[string]jinja.Value) (graph.ValueID, error) {
	num := func(x string) (float64, error) { return g.num(x, vars) }
	width, err := num(o.Width)
	if err != nil {
		return 0, err
	}
	ws, err := bindings(o, bf, num)
	if err != nil {
		return 0, err
	}
	var terms []graph.Term
	for _, tm := range o.Terms {
		t := graph.Term{Fn: fnNamed(tm.Fn)}
		for k, a := range tm.Args {
			switch {
			case a.In != nil:
				if *a.In >= len(in) {
					return 0, fmt.Errorf("a term reads operand %d of %d", *a.In, len(in))
				}
				t.Args[k] = graph.Arg{Kind: graph.ArgIn, Index: *a.In}
			case a.Term != nil:
				t.Args[k] = graph.Arg{Kind: graph.ArgTerm, Index: *a.Term}
			case a.W != nil:
				if *a.W >= len(ws) {
					return 0, fmt.Errorf("a term reads vector %d of %d bound", *a.W, len(ws))
				}
				t.Args[k] = graph.Arg{Kind: graph.ArgWeight, Index: *a.W, Elem: a.Elem}
			default:
				c, err := num(a.Const)
				if err != nil {
					return 0, err
				}
				t.Args[k] = graph.Arg{Kind: graph.ArgConst, Const: c}
			}
		}
		terms = append(terms, t)
	}
	name := o.Name
	if name == "" {
		name = "expr"
	}
	return b.Expr(name, int(width), terms, ws, in...), nil
}

// bindings are an op's stated weights: each tensor at the shape stated, or
// at the shape it is stored at where none is; a ? name where stored.
func bindings(o mjBlockOp, bf *blockFacts, num func(string) (float64, error)) ([]graph.Binding, error) {
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
	return ws, nil
}

// stated is a stated op's value: the router's score, an expression over
// the logits, or the delta rule's decay or gate, programs lowered as the
// graph's ops (one of no op is the input it names).
func stated(b *graph.Builder, c *jlm.Config, o mjBlockOp, in []graph.ValueID, bf *blockFacts,
	num func(string) (float64, error)) (graph.ValueID, error) {
	name := o.Name
	if name == "" {
		name = o.Of
	}
	var p *jlm.Program
	switch o.Of {
	case "router_score":
		width, err := num(o.Width)
		if err != nil {
			return 0, err
		}
		return b.Stated(name, &c.RouterScore, in, int(width), bf.facts)
	case "router_weight":
		k, err := num(o.Width)
		if err != nil {
			return 0, err
		}
		if len(in) != 3 {
			return 0, fmt.Errorf("the router's weight reads the logits, the scores it chose by and the picks, not %d operands", len(in))
		}
		return routerWeight(b, &c.RouterWeight, in[0], in[1], in[2], int(k), bf.facts)
	case "qkv_act":
		return b.Stated(name, c.QKVAct(), in, b.G.Values[in[0]].Width, bf.facts)
	case "delta_decay":
		p = c.DeltaDecay
	case "delta_gate":
		p = c.DeltaGate
	}
	if p == nil {
		return 0, fmt.Errorf("the container states no %s", o.Of)
	}
	if len(p.Ops) == 0 {
		if int(p.Out) >= len(in) {
			return 0, fmt.Errorf("a program of no op naming input %d of %d", p.Out, len(in))
		}
		return in[p.Out], nil
	}
	return graph.LowerProgram(b, nil, p, in, 1, bf.facts)
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
	// taps is how many blocks add a picture's tap rows: the tower's taps.
	taps int
	// chanDecay says the model's delta rule decays per channel: it stores
	// the forget gate's low-rank pair.
	chanDecay bool
}

func blockFactsOf(c *jlm.Config, s *jlm.Source, li int) *blockFacts {
	bf := &blockFacts{c: c, li: li, facts: graph.Facts{Inventory: map[jlm.Role]graph.Shape{}}, dense: map[jlm.Role]bool{},
		headDim: int(c.HeadDimAt(li)), eps: float64(c.RMSEps)}
	if s.Vision != nil && s.Vision.Tower != nil {
		bf.taps = len(s.Vision.Tower.Taps)
	}
	for _, t := range s.Tensors {
		bf.chanDecay = bf.chanDecay || t.Role == jlm.RoleSSMFB
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
	// A prediction block's own head mix is its collapse graph's, not its
	// body's.
	if li >= int(c.NLayer) {
		for _, r := range []jlm.Role{jlm.RoleHCHeadNorm, jlm.RoleHCHeadDown, jlm.RoleHCHeadUp} {
			delete(bf.facts.Inventory, r)
		}
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
		"linear":             jinja.NewBool(li < len(c.LayerKinds) && c.LayerKinds[li] == jlm.LayerLinearAttn),
		"parallel":           jinja.NewBool(c.Flags.Has(jlm.FlagParallel)),
		"qk_l2":              jinja.NewBool(c.Flags.Has(jlm.FlagQKL2Norm)),
		"qk_norm":            jinja.NewBool(c.QKNormAt(li)),
		"router_weight":      jinja.NewBool(len(c.RouterWeight.Terms) != 0),
		"token_act":          jinja.NewBool(c.ActQuant == jlm.ActQuantRowInt8),
		"kv_lora":            i(c.KVLoraRank),
		"rot":                i(c.NRot),
		"v_head_dim":         jinja.NewInt(int64(cmp.Or(c.HeadDimV, c.HeadDim))),
		"latent_eps":         jinja.NewFloat(cmp.Or(float64(c.LatentNormEps), bf.eps)),
		"qk_norm_wide":       jinja.NewBool(c.Flags.Has(jlm.FlagQKNormWide)),
		"qk_norm_post":       jinja.NewBool(c.Flags.Has(jlm.FlagQKNormPostRope)),
		"nope_global":        jinja.NewBool(c.Flags.Has(jlm.FlagNoPEGlobal)),
		"temp_scale":         jinja.NewFloat(float64(c.AttnTempScale)),
		"temp_offset":        jinja.NewFloat(float64(c.AttnTempOffset)),
		"temp_floor":         i(c.AttnTempFloor),
		"layer_norm":         jinja.NewBool(c.Flags.Has(jlm.FlagLayerNorm)),
		"qkv_act":            jinja.NewBool(c.QKVAct() != nil),
		"v_heads":            i(c.SSM.NHeadV),
		"k_heads":            i(c.SSM.Groups),
		"k_dim":              i(c.SSM.StateSize),
		"v_dim":              jinja.NewInt(int64(bf.vDim())),
		"conv":               i(c.SSM.ConvKernel),
		"chans":              jinja.NewInt(int64(c.SSM.Inner) + 2*int64(c.SSM.Groups)*int64(c.SSM.StateSize)),
		"delta_scale":        jinja.NewFloat(bf.deltaScale()),
		"chan_decay":         jinja.NewBool(bf.chanDecay),
		"tiled":              jinja.NewBool(c.Flags.Has(jlm.FlagDeltaKeyTiled)),
		"l2_eps":             jinja.NewFloat(bf.l2Eps()),
		"state_span":         jinja.NewInt(int64(bf.stateSpan())),
		"norm_groups":        jinja.NewInt(int64(max(c.SSM.Groups, 1))),
		"kind":               jinja.NewString(bf.kind()),
		"deep_taps":          jinja.NewInt(int64(bf.taps)),
		"side_width":         jinja.NewInt(int64(bf.taps)*int64(c.NEmbd) + int64(c.NLayer)*int64(c.PLEDim)),
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

// kind is the block's kind as the container states it (jlm.LayerKind's
// name): attention where it states none.
func (bf *blockFacts) kind() string {
	if bf.li < len(bf.c.LayerKinds) {
		return bf.c.LayerKinds[bf.li].String()
	}
	return jlm.LayerFullAttn.String()
}

// vDim is a delta rule's value head width.
func (bf *blockFacts) vDim() int {
	if bf.c.SSM.NHeadV == 0 {
		return 0
	}
	return int(bf.c.SSM.Inner / bf.c.SSM.NHeadV)
}

// deltaScale is the delta rule's 1/sqrt of its key width, in binary32.
func (bf *blockFacts) deltaScale() float64 {
	if bf.c.SSM.StateSize == 0 {
		return 0
	}
	return float64(float32(1 / math.Sqrt(float64(bf.c.SSM.StateSize))))
}

// l2Eps is the epsilon the delta rule's unit norms of q and k take: the
// model's, or Kimi's 1e-6 where the decay is per channel.
func (bf *blockFacts) l2Eps() float64 {
	if bf.chanDecay {
		return 1e-6
	}
	return bf.eps
}

// stateSpan is a delta rule's state floats a sequence where the block's
// state carries a stated stream's tail past the rule's, 0 where it is the
// rule's alone.
func (bf *blockFacts) stateSpan() int {
	t := 0
	for i := range bf.c.Streams.Blocks {
		t = max(t, bf.c.Streams.Blocks[i].Pre.Read.Tail())
	}
	if t == 0 {
		return 0
	}
	return int(bf.c.SSM.NHeadV)*bf.vDim()*int(bf.c.SSM.StateSize) + t
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
	if c.Flags.Has(jlm.FlagRopeXD) {
		return graph.Attrs{}, fmt.Errorf("a stated graph's XD rotary: %w", ErrNotImplemented)
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
	if sec := c.RopeSections; sec != [4]uint32{} {
		if nrot != int(c.NRot) || base != float64(c.RopeBase) {
			return graph.Attrs{}, fmt.Errorf("rotary sections on the sliding layers' own rotary")
		}
		for i, n := range sec {
			a.Sections[i] = int(n)
		}
		// The sections lie one after another or interleaved; where the two
		// lay the pairs out alike, they read as interleaved.
		a.Interleaved = c.Flags.Has(jlm.FlagRopeInterleaved) || slices.Equal(mropeRuns(sec), imropeRuns(sec, nrot/2))
	}
	return a, nil
}

// ropeRun is a run of a multimodal rotary's pairs turned by one coordinate
// (axis) from frequency freq.
type ropeRun struct{ pairs, axis, freq int }

// mropeRuns is the sections one after another: each coordinate's pairs,
// at the frequencies that follow the last's.
func mropeRuns(sec [4]uint32) []ropeRun {
	var runs []ropeRun
	off := 0
	for ax, n := range sec {
		if n == 0 {
			continue
		}
		runs = append(runs, ropeRun{int(n), ax, off})
		off += int(n)
	}
	return runs
}

// imropeRuns is the sections interleaved over pairs pairs: pair i turns by
// coordinate 1 where i%3 is 1 and i is below three times its section, by 2
// where i%3 is 2 and below three times its, and by 0 otherwise, at
// frequency i.
func imropeRuns(sec [4]uint32, pairs int) []ropeRun {
	var runs []ropeRun
	for i := range pairs {
		ax := 0
		switch {
		case i%3 == 1 && i < 3*int(sec[1]):
			ax = 1
		case i%3 == 2 && i < 3*int(sec[2]):
			ax = 2
		}
		if n := len(runs); n > 0 && runs[n-1].axis == ax {
			runs[n-1].pairs++
			continue
		}
		runs = append(runs, ropeRun{1, ax, i})
	}
	return runs
}
