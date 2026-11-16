package convert

import (
	"fmt"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// statedCut is one target a stated tensor is cut into: its role and the
// primitives that cut it (none, the tensor whole).
type statedCut struct {
	role jlm.Role
	do   mjPipe
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
	}
	return out, nil
}

// statedTensors cuts each of s's tensors c's statement names -- a
// model-level one by its name, a block's by its name past the blk.N. prefix,
// a recurrent block's table first -- into its targets, laid out in its
// place, env the source's keys; a target of a joined role is held, and each
// block's joins are laid out after every other tensor, by block and role.
func statedTensors(c *Config, s *jlm.Source, env exprEnv) error {
	if c == nil || c.Stated == nil {
		return nil
	}
	st := c.Stated
	if len(st.modelTensors)+len(st.blockTensors)+len(st.linearTensors) == 0 {
		return nil
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
			r, in := strings.CutPrefix(t.Name, "blk."+itoa(int(t.Block))+".")
			rest = r
			cuts, ok = st.blockTensors[rest]
			if lc, lin := st.linearTensors[rest]; lin && int(t.Block) < len(c.LayerKinds) && c.LayerKinds[t.Block].Recurrent() {
				cuts, ok = lc, true
			}
			ok = ok && in
		}
		if !ok {
			out = append(out, t)
			continue
		}
		in := mjVal{shape: mjTorch(&t), typ: t.Type}
		if slices.ContainsFunc(cuts, func(x statedCut) bool { return len(x.do) > 0 }) {
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
				v, err := x.do.run(env, []mjVal{in})
				if err != nil {
					return fmt.Errorf("convert: %s: %v: %w", t.Name, x.role, err)
				}
				if err := mjSetShape(&p, v); err != nil {
					return err
				}
				p.Data, p.Load = v.data, nil
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
		out = append(out, e)
	}
	s.Tensors = out
	return nil
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
