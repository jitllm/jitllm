package convert

import (
	"fmt"
	"math"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// mjStreams is the residual's streams a graph states (jlm.Streams), each
// program ops of the tower op language built by stateProg, its expressions
// over a block's facts (blockFacts.vars) and its weights by the graph's
// tensor names: Count the streams; Expand stream k's input row from the
// embedding, Var the stream, from 1; Block each block's reads and writes,
// over its own facts; Head the row the head reads. Expand and Head read the
// first block's facts. Defs are op lists an op calls by name. Where If, over
// the first block's facts, does not hold, the model has no streams.
type mjStreams struct {
	If     string                 `json:"if"`
	Count  string                 `json:"count"`
	Defs   map[string][]mjTowerOp `json:"defs"`
	Expand *mjStreamProgs         `json:"expand"`
	Block  mjStreamBlock          `json:"block"`
	Head   *mjSubStreams          `json:"head"`
}

// mjStreamProgs is a program for each stream, Var its index; Copy, true,
// states none, each stream a copy of what it is made from (jlm.Streams).
type mjStreamProgs struct {
	Var  string      `json:"var"`
	Ops  []mjTowerOp `json:"ops"`
	Copy bool        `json:"copy"`
}

// mjStreamBlock is a block's use of the streams (jlm.BlockStreams): Bank
// the stream its input is banked into (0 for none), and its sublayers'.
type mjStreamBlock struct {
	Bank string        `json:"bank"`
	Pre  *mjSubStreams `json:"pre"`
	Attn *mjSubStreams `json:"attn"`
	FFN  *mjSubStreams `json:"ffn"`
}

// mjSubStreams is a sublayer's use of the streams (jlm.SubStreams) where If
// holds: From its read's inputs, each run, in, side or an @ and an
// expression of a stream, of one of those names, or of a list of them; Read
// the read over them (absent, the running residual), where ReadIf holds;
// Exports the read's values the writes take, each names its ops give (keys)
// or an @ and an expression of a value or a list of them; Join, where JoinIf
// holds, Post and Writes the container's.
type mjSubStreams struct {
	If      string         `json:"if"`
	ReadIf  string         `json:"read_if"`
	JoinIf  string         `json:"join_if"`
	From    []string       `json:"from"`
	Read    []mjTowerOp    `json:"read"`
	Exports []string       `json:"exports"`
	Join    []mjTowerOp    `json:"join"`
	Post    []mjTowerOp    `json:"post"`
	Writes  *mjStreamProgs `json:"writes"`
}

// subs are the block's sublayer statements and the head's.
func (st *mjStreams) subs() []*mjSubStreams {
	return []*mjSubStreams{st.Block.Pre, st.Block.Attn, st.Block.FFN, st.Head}
}

// check refuses a statement whose ops, calls or weights the container does
// not know, and compiles its expressions into g's.
func (st *mjStreams) check(g *mjGraph, names tensorNames) error {
	if st.Count == "" {
		return fmt.Errorf("the streams state no count")
	}
	role := func(n string) error {
		if _, ok := names.blockRole(strings.TrimSuffix(n, "?")); !ok {
			return fmt.Errorf("the streams bind %q, which is no tensor name the container knows", n)
		}
		return nil
	}
	var progs [][]mjTowerOp
	for _, d := range st.Defs {
		progs = append(progs, d)
	}
	srcs := []string{st.If, st.Count, st.Block.Bank}
	vars := []string{"inputs"}
	each := func(ps *mjStreamProgs) error {
		if ps == nil {
			return nil
		}
		if ps.Copy != (ps.Var == "" && len(ps.Ops) == 0) {
			return fmt.Errorf("a stream's programs are a copy, or state their variable and their ops")
		}
		progs, vars = append(progs, ps.Ops), append(vars, ps.Var)
		return nil
	}
	if err := each(st.Expand); err != nil {
		return err
	}
	for _, u := range st.subs() {
		if u == nil {
			continue
		}
		if len(u.From) == 0 && u.Read != nil {
			return fmt.Errorf("a stream read states no inputs")
		}
		srcs = append(srcs, u.If, u.ReadIf, u.JoinIf)
		for _, xs := range [][]string{u.From, u.Exports} {
			for _, x := range xs {
				if at, ok := strings.CutPrefix(x, "@"); ok {
					srcs = append(srcs, at)
				} else if i := strings.IndexByte(x, '['); i > 0 && strings.HasSuffix(x, "]") {
					srcs = append(srcs, x[i+1:len(x)-1])
				}
			}
		}
		progs = append(progs, u.Read, u.Join, u.Post)
		if err := each(u.Writes); err != nil {
			return err
		}
	}
	read, eaches, err := progsCheck(progs, []string{"let", "call"}, st.Defs, role)
	if err != nil {
		return err
	}
	vars = append(vars, eaches...)
	for _, src := range append(srcs, read...) {
		if err := g.compile(src, vars...); err != nil {
			return fmt.Errorf("the streams: %w", err)
		}
	}
	return nil
}

// LibraryStreams states c's streams over f's tensors as the library states
// them for GGUF architecture arch, reading no source keys: what a test that
// changes a config states the changed streams by.
func LibraryStreams(f *jlm.File, c *jlm.Config, arch string) error {
	g, err := ggufGraph(arch)
	if err != nil {
		return err
	}
	s := &jlm.Source{}
	for _, e := range f.Entries() {
		s.Tensors = append(s.Tensors, jlm.Tensor{Role: e.Role, Block: e.Block, Index: e.Index, Type: e.Type, NDim: e.NDim,
			Dims: e.Dims, Name: e.Name})
	}
	return stateGraphStreams(c, s, g, jinja.NewDict())
}

// stateGraphStreams states c's streams as g states them, src the source's
// keys. Stated streams are the container's, and the count the converter
// laid the tensors out by (PredStreams) is not. A config of several streams,
// or of an n-gram block, that no statement states is refused.
func stateGraphStreams(c *jlm.Config, s *jlm.Source, g *mjGraph, src jinja.Value) error {
	if c.NLayer == 0 {
		return nil
	}
	if g == nil || g.Streams == nil {
		if c.HCMult > 1 || c.AttnResBlock > 0 || c.PredStreams > 1 || c.NGramBlock > 0 {
			return fmt.Errorf("convert: the config carries residual streams (%d hyper-connected, a bank every %d blocks, "+
				"%d predicted, an n-gram block at %d) and the library states none: %w", c.HCMult, c.AttnResBlock,
				c.PredStreams, c.NGramBlock, ErrNotImplemented)
		}
		return nil
	}
	st := g.Streams
	names, err := readTensorNames()
	if err != nil {
		return err
	}
	var ferr error
	facts := func(li int) (*blockFacts, map[string]jinja.Value) {
		bf := blockFactsOf(c, s, li, names)
		bf.src = src
		vars, err := g.withFacts(bf.vars())
		if err != nil && ferr == nil {
			ferr = fmt.Errorf("convert: block %d's streams: %w", li, err)
		}
		return bf, vars
	}
	_, f0 := facts(0)
	if ferr != nil {
		return ferr
	}
	if st.If != "" {
		on, err := g.flag(st.If, f0)
		if err != nil {
			return fmt.Errorf("convert: the streams' condition: %w", err)
		}
		if !on {
			return nil
		}
	}
	n, err := g.num(st.Count, f0)
	if err != nil {
		return fmt.Errorf("convert: the streams' count: %w", err)
	}
	if n < 2 || n > math.MaxUint8 || n != math.Trunc(n) {
		return fmt.Errorf("convert: %v residual streams: a statement states 2 to %d", n, math.MaxUint8)
	}
	c.Streams, c.PredStreams = jlm.Streams{N: uint8(n)}, 0
	builder := func(bf *blockFacts, vars map[string]jinja.Value, where string) (*stateProg, func(string) bool) {
		held := func(name string) bool {
			r, ok := names.blockRole(name)
			_, in := bf.facts.Inventory[r]
			return ok && (in || bf.dense[r])
		}
		return &stateProg{where: where, facts: vars, defs: st.Defs, role: names.blockRole,
			value: func(x string) (jinja.Value, error) { return g.value(x, vars) },
			num: func(x string) (float64, error) {
				if x == "" {
					return 0, nil
				}
				return g.num(x, vars)
			}}, held
	}
	bf, vars := facts(0)
	sp, held := builder(bf, vars, "the streams")
	if e := st.Expand; e != nil {
		c.Streams.Expand = make([]*jlm.Program, int(n)-1)
		for k := 1; k < int(n) && !e.Copy; k++ {
			vars[e.Var] = jinja.NewInt(int64(k))
			p, err := sp.prog(e.Ops, 1, held)
			if err != nil {
				return err
			}
			c.Streams.Expand[k-1] = &p
		}
		delete(vars, e.Var)
	}
	if c.Streams.Head, err = streamSub(st.Head, sp, held, int(n)); err != nil {
		return err
	}
	c.Streams.Blocks = make([]jlm.BlockStreams, c.NLayer+predictionBlocks(c, s.Tensors))
	for li := range c.Streams.Blocks {
		bf, vars := facts(li)
		if ferr != nil {
			return ferr
		}
		sp, held := builder(bf, vars, fmt.Sprintf("block %d's streams", li))
		b := &c.Streams.Blocks[li]
		if st.Block.Bank != "" {
			k, err := sp.num(st.Block.Bank)
			if err != nil {
				return err
			}
			if k < 0 || k >= n {
				return fmt.Errorf("convert: block %d banks into stream %v of %v", li, k, n)
			}
			b.Bank = uint8(k)
		}
		for _, u := range []struct {
			st *mjSubStreams
			to *jlm.SubStreams
		}{{st.Block.Pre, &b.Pre}, {st.Block.Attn, &b.Attn}, {st.Block.FFN, &b.FFN}} {
			if *u.to, err = streamSub(u.st, sp, held, int(n)); err != nil {
				return err
			}
		}
	}
	return nil
}

// streamSub is u's sublayer over n streams, built by sp: zero where u is
// absent or its If does not hold.
func streamSub(u *mjSubStreams, sp *stateProg, held func(string) bool, n int) (jlm.SubStreams, error) {
	var out jlm.SubStreams
	if u == nil {
		return out, nil
	}
	flag := func(x string) (bool, error) {
		if x == "" {
			return true, nil
		}
		on, err := sp.value(x)
		if err == nil && !on.IsBool() {
			err = fmt.Errorf("convert: %s: %q is %v, not true or false", sp.where, x, on)
		}
		return err == nil && on.AsBool(), err
	}
	on, err := flag(u.If)
	if err != nil || !on {
		return out, err
	}
	reads, err := flag(u.ReadIf)
	if err != nil {
		return out, err
	}
	joins, err := flag(u.JoinIf)
	if err != nil {
		return out, err
	}
	list := func(x string) ([]jinja.Value, error) {
		v, err := sp.value(x)
		if err != nil {
			return nil, err
		}
		if v.IsList() {
			return v.AsList().Items, nil
		}
		return []jinja.Value{v}, nil
	}
	streams := map[string]int8{"run": jlm.StreamRun, "in": jlm.StreamIn, "side": jlm.StreamSide}
	for _, f := range u.From {
		if !reads {
			break
		}
		if k, ok := streams[f]; ok {
			out.From = append(out.From, k)
			continue
		}
		at, ok := strings.CutPrefix(f, "@")
		if !ok {
			return out, fmt.Errorf("convert: %s: a read from %q, not run, in, side or an @ and a stream", sp.where, f)
		}
		xs, err := list(at)
		if err != nil {
			return out, err
		}
		for _, x := range xs {
			switch {
			case x.IsString() && streams[x.AsString()] != 0:
				out.From = append(out.From, streams[x.AsString()])
			case x.IsInt() && x.AsInt() >= 0 && x.AsInt() < int64(n):
				out.From = append(out.From, int8(x.AsInt()))
			default:
				return out, fmt.Errorf("convert: %s: a read from %v of %d streams", sp.where, x, n)
			}
		}
	}
	var named map[string]uint16
	if u.Read != nil && reads {
		p, nm, err := sp.progNamed(u.Read, len(out.From), held)
		if err != nil {
			return out, err
		}
		out.Read, named = &p, nm
	}
	for _, x := range u.Exports {
		if out.Read == nil {
			break
		}
		if at, ok := strings.CutPrefix(x, "@"); ok {
			vs, err := list(at)
			if err != nil {
				return out, err
			}
			for _, v := range vs {
				if !v.IsInt() || v.AsInt() < 0 || v.AsInt() >= int64(int(out.Read.NIn)+len(out.Read.Ops)) {
					return out, fmt.Errorf("convert: %s: an export of value %v the read has not", sp.where, v)
				}
				out.Exports = append(out.Exports, uint16(v.AsInt()))
			}
			continue
		}
		ks, err := sp.keys(x)
		if err != nil {
			return out, err
		}
		for _, k := range ks {
			v, ok := named[k]
			if !ok {
				return out, fmt.Errorf("convert: %s: an export of %q, which the read names not", sp.where, k)
			}
			out.Exports = append(out.Exports, v)
		}
	}
	prog := func(ops []mjTowerOp, nin int) (*jlm.Program, error) {
		if ops == nil {
			return nil, nil
		}
		p, err := sp.prog(ops, nin, held)
		return &p, err
	}
	if joins {
		if out.Join, err = prog(u.Join, 2); err != nil {
			return out, err
		}
	}
	if out.Post, err = prog(u.Post, 1+len(out.Exports)); err != nil {
		return out, err
	}
	if w := u.Writes; w != nil {
		nin := 2 + len(out.Exports)
		if out.Post != nil {
			nin++
		}
		out.Writes = make([]*jlm.Program, n)
		for k := range n {
			if w.Copy {
				break
			}
			sp.facts[w.Var] = jinja.NewInt(int64(k))
			if out.Writes[k], err = prog(w.Ops, nin); err != nil {
				return out, err
			}
		}
		delete(sp.facts, w.Var)
	}
	return out, nil
}
