package convert

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// A model.json's graph states its blocks' ops: each a step of the graph
// language, its operands values named by earlier ops, its weights tensors by
// their container names, its numbers expressions over the block's facts
// (blockFacts). The converter evaluates it for every block and the container
// carries each block's graph (jlm.BlockGraphs), which the engine decodes
// rather than builds.
type mjGraph struct {
	Block []mjBlockOp `json:"block"`
}

// mjBlockOp is one op of a block. Op names the step: in, norm, layer_norm,
// head_norm, matvec, add, rope, kv_append, attend, act or out.
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
		"in":         {0, 0, 0, 0},
		"norm":       {1, 1, 1, 1},
		"layer_norm": {1, 1, 1, -1},
		"head_norm":  {1, 1, 1, 1},
		"matvec":     {1, 1, 1, -1},
		"add":        {2, 2, 0, 0},
		"rope":       {1, 1, 0, 0},
		"kv_append":  {2, 2, 0, 0},
		"attend":     {1, 1, 0, -1},
		"act":        {1, -1, 0, 0},
		"out":        {1, -1, 0, 0},
	}
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
		for _, x := range []string{o.If, o.Rows, o.Width, o.Alpha} {
			if x == "" {
				continue
			}
			if _, err := parseExpr(x); err != nil {
				return fmt.Errorf("op %d (%s): %w", i, o.Op, err)
			}
		}
		for _, w := range o.W {
			if _, ok := blockRole(strings.TrimSuffix(w, "?")); !ok {
				return fmt.Errorf("op %d (%s): %q is not a tensor name the container knows", i, o.Op, w)
			}
		}
		switch {
		case o.Op == "out":
		case o.Out == "":
			return fmt.Errorf("op %d (%s) names no output", i, o.Op)
		case o.Op == "in" && o.Width == "", o.Op == "matvec" && o.Rows == "", o.Op == "act" && o.Width == "":
			return fmt.Errorf("op %d (%s) states no width", i, o.Op)
		}
		named[o.Out] = true
	}
	if g.Block[len(g.Block)-1].Op != "out" {
		return fmt.Errorf("a block's graph ends in an out")
	}
	return nil
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
	num := func(x string) (float64, error) {
		n, err := parseExpr(x)
		if err != nil {
			return 0, err
		}
		return evalNum(n, bf.env)
	}
	for i, o := range g.Block {
		if o.If != "" {
			n, err := parseExpr(o.If)
			if err != nil {
				return nil, err
			}
			on, err := evalBool(n, bf.env)
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

// env is the facts as an expression reads them: the block's index, its
// geometry, its numbers, stored.<name> whether it stores a tensor and
// rows.<name> a stored one's rows.
func (bf *blockFacts) env(path string) any {
	c := bf.c
	switch path {
	case "layer":
		return float64(bf.li)
	case "embd":
		return float64(c.NEmbd)
	case "heads":
		return float64(c.NHead)
	case "kv_heads":
		return float64(c.NKVHeadAt(bf.li))
	case "head_dim":
		return float64(bf.headDim)
	case "ffn":
		return float64(c.NFFNAt(bf.li))
	case "eps":
		return bf.eps
	case "residual_scale":
		return float64(c.ResidualScale)
	case "rope_at":
		return bf.ropeAt()
	case "sliding":
		return bf.sliding()
	}
	if n, ok := strings.CutPrefix(path, "stored."); ok {
		r, ok := blockRole(n)
		_, held := bf.facts.Inventory[r]
		return ok && held
	}
	if n, ok := strings.CutPrefix(path, "rows."); ok {
		if r, ok := blockRole(n); ok {
			if sh, held := bf.facts.Inventory[r]; held {
				return float64(sh.Rows)
			}
		}
	}
	return nil
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
