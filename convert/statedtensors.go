package convert

import (
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// statedCut is one target a stated tensor is cut into: its role and the
// primitives that cut it (none, the tensor whole).
type statedCut struct {
	role jlm.Role
	do   mjPipe
	when *exprNode
}

// statedDrop is a block tensor a GGUF's table drops, by its name past the
// blk.N. prefix, where when holds.
type statedDrop struct {
	name string
	when *exprNode
}

// statedCuts is source tensor n's targets: each a role -- by the
// vocabulary's name for it, or the container's -- and its primitives,
// several targets each cut by its own.
func statedCuts(names tensorNames, n string, ts mjTargets) ([]statedCut, error) {
	out := make([]statedCut, len(ts))
	for i, t := range ts {
		r, ok := names.roleOf(t.Role)
		if !ok {
			r, ok = rolesByName()[t.Role]
		}
		if !ok {
			return nil, fmt.Errorf("%s's target %q is neither a tensor name nor a role the container knows", n, t.Role)
		}
		if len(ts) > 1 && len(t.Do) == 0 {
			return nil, fmt.Errorf("%s feeds %d roles, and %s takes no primitives to cut it by", n, len(ts), t.Role)
		}
		if err := t.Do.check(); err != nil {
			return nil, fmt.Errorf("%s: %s: %w", n, t.Role, err)
		}
		out[i] = statedCut{role: r, do: t.Do}
		if t.If != "" {
			x, err := parseExpr(t.If)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", n, err)
			}
			out[i].when = x
		}
	}
	return out, nil
}

// statedSizes is a table's sizes, each parsed, by a name of letters, digits
// and underscores.
func statedSizes(sizes map[string]string) (map[string]*exprNode, error) {
	out := make(map[string]*exprNode, len(sizes))
	for _, n := range slices.Sorted(maps.Keys(sizes)) {
		if n == "" || strings.Trim(n, "abcdefghijklmnopqrstuvwxyz0123456789_") != "" {
			return nil, fmt.Errorf("size %q is not a name of letters, digits and underscores", n)
		}
		x, err := parseExpr(sizes[n])
		if err != nil {
			return nil, fmt.Errorf("size %s: %w", n, err)
		}
		out[n] = x
	}
	return out, nil
}

// keyValue is a GGUF key a table reads as a block's source: its value, by
// its bare name (llama.cpp writes some without the architecture's) or past
// the architecture's prefix, for every block where it is a number, its
// block's entry where it is a list of one a block.
func keyValue(f *meta.File, k string, block, blocks int) (float32, error) {
	v, ok := f.KV[k]
	if !ok {
		if v, ok = f.Key(k); !ok {
			return 0, fmt.Errorf("convert: no key %s", k)
		}
	}
	if v.Len() == 0 {
		x, ok := v.Float()
		if !ok {
			return 0, fmt.Errorf("convert: %s is %v, not a number", k, v.Type)
		}
		return float32(x), nil
	}
	xs, err := v.Float32s()
	if err != nil {
		return 0, fmt.Errorf("convert: %s: %w", k, err)
	}
	if len(xs) != blocks {
		return 0, fmt.Errorf("convert: %s has %d values for %d blocks", k, len(xs), blocks)
	}
	return xs[block], nil
}

// statedGathers runs c's statement's gathers (gatherBlocks) over s's block
// tensors and the keys it names, every block's result after s's tensors;
// a gathered tensor is not stored.
func statedGathers(c *Config, s *jlm.Source, f *meta.File) error {
	if c == nil || c.Stated == nil || len(c.Stated.gathers) == 0 {
		return nil
	}
	st := c.Stated
	n := int(c.NLayer)
	var names []string
	byName := map[string]*jlm.Tensor{}
	for i := range s.Tensors {
		names = append(names, s.Tensors[i].Name)
		byName[s.Tensors[i].Name] = &s.Tensors[i]
	}
	keyed := map[string]int{}
	for b := range n {
		for _, k := range st.keys {
			kn := "blk." + itoa(b) + "." + k
			names, keyed[kn] = append(names, kn), b
		}
	}
	read := func(name string) ([]float32, error) {
		if b, ok := keyed[name]; ok {
			_, k, _ := strings.Cut(strings.TrimPrefix(name, "blk."), ".")
			x, err := keyValue(f, k, b, n)
			return []float32{x}, err
		}
		t := byName[name]
		src, ok := sourceType(t.Type)
		if !ok {
			return nil, fmt.Errorf("convert: %s is %v, which has no source type", t.Name, t.Type)
		}
		b, err := t.Bytes()
		if err != nil {
			return nil, err
		}
		vals := make([]float32, mjProd(mjTorch(t)))
		return vals, quant.Dequant32(src, b, vals)
	}
	out, used, err := gatherBlocks(ggufEnv(f), "blk.", st.gathers, names, read, n)
	if err != nil {
		return err
	}
	kept := s.Tensors[:0]
	for _, t := range s.Tensors {
		if !used[t.Name] {
			kept = append(kept, t)
		}
	}
	s.Tensors = append(kept, out...)
	return nil
}

// statedTensors cuts each of s's tensors c's statement names -- a
// model-level one by its name, a block's by its name past the blk.N. prefix,
// a linear-attention block's table first -- into its targets, laid out in its
// place, env the source's keys; a target of a joined role is held, and each
// block's joins are laid out after every other tensor, by block and role.
func statedTensors(c *Config, s *jlm.Source, env exprEnv) error {
	if c == nil || c.Stated == nil {
		return nil
	}
	st := c.Stated
	if len(st.modelTensors)+len(st.blockTensors)+len(st.linearTensors)+len(st.drops) == 0 {
		return nil
	}
	// A tensor's conditions and primitives read its block, whether it is
	// packed, the table's sizes and the source's keys; a size reads all but
	// the sizes, and fail is where one that cannot be read says why.
	tensorEnv := func(t *jlm.Tensor, fail *error) exprEnv {
		own := func(p string) any {
			switch p {
			case "block":
				return float64(t.Block)
			case "packed":
				return jlm.Packed(t.Type)
			}
			return env(p)
		}
		return func(p string) any {
			n, ok := strings.CutPrefix(p, "size.")
			if !ok {
				return own(p)
			}
			x, ok := st.sizes[n]
			if !ok {
				return nil
			}
			v, err := evalMJ(x, own)
			if err != nil && *fail == nil {
				*fail = fmt.Errorf("size.%s: %w", n, err)
			}
			return v
		}
	}
	holds := func(x *exprNode, t *jlm.Tensor) (bool, error) {
		var fail error
		v, err := evalMJ(x, tensorEnv(t, &fail))
		if fail != nil {
			return false, fail
		}
		b, ok := v.(bool)
		if err == nil && !ok {
			err = fmt.Errorf("a condition is %v, not true or false", v)
		}
		return b, err
	}
	type slot struct {
		block int32
		role  jlm.Role
	}
	fused := map[slot][]*jlm.Tensor{}
	out := s.Tensors[:0:0]
	for _, t := range s.Tensors {
		cuts, ok := st.modelTensors[t.Name]
		rest := t.Name
		if t.Block >= 0 {
			r, in := ggufBlockRest(t.Name)
			rest = r
			cuts, ok = st.blockTensors[rest]
			if lc, lin := st.linearTensors[rest]; lin && int(t.Block) < len(c.LayerKinds) && c.LayerKinds[t.Block] == jlm.LayerLinearAttn {
				cuts, ok = lc, true
			}
			ok = ok && in
		}
		dropped := false
		for _, d := range st.drops {
			if t.Block < 0 || d.name != rest || dropped {
				continue
			}
			var err error
			if dropped, err = holds(d.when, &t); err != nil {
				return fmt.Errorf("convert: %s: drop: %w", t.Name, err)
			}
		}
		if dropped {
			continue
		}
		// A target is cut where its if holds; a tensor none of whose targets
		// is cut is kept as it is.
		if ok {
			var on []statedCut
			for _, x := range cuts {
				h := x.when == nil
				if !h {
					var err error
					if h, err = holds(x.when, &t); err != nil {
						return fmt.Errorf("convert: %s: %w", t.Name, err)
					}
				}
				if h {
					on = append(on, x)
				}
			}
			cuts, ok = on, len(on) > 0
		}
		if !ok {
			out = append(out, t)
			continue
		}
		in := mjVal{shape: mjTorch(&t), typ: t.Type}
		if slices.ContainsFunc(cuts, func(x statedCut) bool { return len(x.do) > 0 && !x.do.reshapes() }) {
			b, err := t.Bytes()
			if err != nil {
				return err
			}
			in.data = b
		}
		for _, x := range cuts {
			p := t
			p.Role = x.role
			if len(cuts) > 1 {
				p.Name = fmt.Sprintf("%s[%v]", t.Name, x.role)
			}
			if len(x.do) > 0 {
				var fail error
				v, err := x.do.run(tensorEnv(&t, &fail), []mjVal{in})
				if fail != nil {
					err = fail
				}
				if err != nil {
					return fmt.Errorf("convert: %s: %v: %w", t.Name, x.role, err)
				}
				if err := mjSetShape(&p, v); err != nil {
					return err
				}
				if !x.do.reshapes() {
					p.Data, p.Load = v.data, nil
				}
			}
			srcs, joins := st.fuse[p.Role]
			if !joins || p.Block < 0 {
				out = append(out, p)
				continue
			}
			i := slices.Index(srcs, rest)
			if i < 0 {
				return fmt.Errorf("convert: %s feeds the joined %v and is none of its sources %v", t.Name, p.Role, srcs)
			}
			k := slot{p.Block, p.Role}
			if fused[k] == nil {
				fused[k] = make([]*jlm.Tensor, len(srcs))
			}
			if fused[k][i] != nil {
				return fmt.Errorf("convert: block %d's %v is joined from %s twice", p.Block, p.Role, rest)
			}
			fused[k][i] = &p
		}
	}
	keys := make([]slot, 0, len(fused))
	for k := range fused {
		keys = append(keys, k)
	}
	slices.SortFunc(keys, func(a, b slot) int {
		if a.block != b.block {
			return int(a.block) - int(b.block)
		}
		return int(a.role) - int(b.role)
	})
	for _, k := range keys {
		for i, p := range fused[k] {
			if p == nil {
				return fmt.Errorf("convert: block %d's %v lacks its source %s", k.block, k.role, st.fuse[k.role][i])
			}
		}
		e, err := joinRows(fused[k], k.role)
		if err != nil {
			return err
		}
		if p := st.joined[k.role]; len(p) > 0 {
			var fail error
			v, err := p.run(tensorEnv(&e, &fail), []mjVal{{shape: mjTorch(&e), typ: e.Type, data: e.Data}})
			if fail != nil {
				err = fail
			}
			if err != nil {
				return fmt.Errorf("convert: block %d's joined %v: %w", k.block, k.role, err)
			}
			if err := mjSetShape(&e, v); err != nil {
				return err
			}
			e.Data, e.Load = v.data, nil
		}
		out = append(out, e)
	}
	s.Tensors = out
	return nil
}

// ggufBlockRest is a block tensor's name past its blk.N. prefix, false for a
// name of none.
func ggufBlockRest(name string) (string, bool) {
	r, ok := strings.CutPrefix(name, "blk.")
	i := strings.IndexByte(r, '.')
	if !ok || i <= 0 || strings.Trim(r[:i], "0123456789") != "" {
		return name, false
	}
	return r[i+1:], true
}

// joinRows stacks matrices of one k by rows, in order. Of one type the
// bytes go back to back (row-major blocks). Of several -- a mixed
// quantization such as Kimi-K3-0.40B's Q4_K_M, attn_q at Q4_K beside attn_k
// and attn_v at Q8_0 -- every part is dequantized and the whole re-stored as
// Q8_0 (F32 where k is not whole Q8_0 blocks): a Q8_0 part comes back bit
// for bit (its block's largest code is 127, so the requantization finds the
// stored scale again), and a narrower part pays Q8_0's bytes and rounding.
func joinRows(ts []*jlm.Tensor, role jlm.Role) (jlm.Tensor, error) {
	f := ts[0]
	e := jlm.Tensor{Role: role, Block: f.Block, Index: -1, Type: f.Type, NDim: 2, Name: f.Name}
	k := f.Dims[0]
	var rows uint64
	mixed := false
	for _, t := range ts {
		if t.NDim != 2 || t.Dims[0] != k {
			return jlm.Tensor{}, fmt.Errorf("convert: %s is %v, and %s's rows are %d wide", t.Name, t.Dims[:t.NDim], f.Name, k)
		}
		rows += t.Dims[1]
		mixed = mixed || t.Type != f.Type
	}
	e.Dims[0], e.Dims[1] = k, rows
	if !mixed {
		for _, t := range ts {
			b, err := t.Bytes()
			if err != nil {
				return jlm.Tensor{}, err
			}
			e.Data = append(e.Data, b...)
		}
		return e, nil
	}
	vals := make([]float32, k*rows)
	off := uint64(0)
	for _, t := range ts {
		src, ok := sourceType(t.Type)
		if !ok {
			return jlm.Tensor{}, fmt.Errorf("convert: %s is %v, which has no source type", t.Name, t.Type)
		}
		b, err := t.Bytes()
		if err != nil {
			return jlm.Tensor{}, err
		}
		n := k * t.Dims[1]
		if err := quant.Dequant32(src, b, vals[off:off+n]); err != nil {
			return jlm.Tensor{}, fmt.Errorf("convert: %s: %w", t.Name, err)
		}
		off += n
	}
	if k%q8Elems == 0 {
		e.Type, e.Data = jlm.TypeQ8, quantizeQ8Rows(vals, int(k), int(rows))
		return e, nil
	}
	e.Type, e.Data = jlm.TypeF32, f32AsBytes(vals)
	return e, nil
}
