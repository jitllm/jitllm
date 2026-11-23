package convert

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
)

// hfPlaced is a model-level source name whose targets are placed in
// blocks: the name, or where it holds {i} the text before and after the
// index, the first target's role and the cut.
type hfPlaced struct {
	pre, post string
	indexed   bool
	role      jlm.Role
	split     hfSplit
}

// index is i where name is p's, -1 where it is not.
func (p hfPlaced) index(name string) int {
	if !p.indexed {
		if name == p.pre {
			return 0
		}
		return -1
	}
	mid, ok := strings.CutPrefix(name, p.pre)
	if !ok {
		return -1
	}
	if mid, ok = strings.CutSuffix(mid, p.post); !ok || mid == "" {
		return -1
	}
	i, err := strconv.Atoi(mid)
	if err != nil || i < 0 || strconv.Itoa(i) != mid {
		return -1
	}
	return i
}

// placedOf is the placed source name is, if any.
func (a *hfArch) placedOf(name string) (hfPlaced, int, bool) {
	for _, p := range a.placed {
		if i := p.index(name); i >= 0 {
			return p, i, true
		}
	}
	return hfPlaced{}, 0, false
}

// addPlaced states model-level source n, whose targets go to blocks: each
// target cut by its primitives over the config and i, Each times where it
// states a count, where its Where holds, into the block its Block names.
func (mj *modelJSON) addPlaced(a *hfArch, st *mjState, n string, ts mjTargets, retarget func(jlm.Role) jlm.Role) error {
	p := hfPlaced{pre: n}
	if pre, post, ok := strings.Cut(n, "{i}"); ok {
		if strings.Contains(post, "{i}") {
			return fmt.Errorf("convert: %s: %s names {i} twice", mj.name, n)
		}
		p = hfPlaced{pre: pre, post: post, indexed: true}
	}
	type target struct {
		t                  mjTensor
		role               jlm.Role
		each, where, block *exprNode
	}
	cuts := make([]target, len(ts))
	for k, t := range ts {
		if t.If != "" {
			return fmt.Errorf("convert: %s: %s: a class's target states no if, which is a GGUF table's", mj.name, n)
		}
		r, ok := mj.names.roleOf(t.Role)
		if !ok {
			return fmt.Errorf("convert: %s: %s's role %q is not a tensor name the container knows", mj.name, n, t.Role)
		}
		if err := t.Do.check(); err != nil {
			return fmt.Errorf("convert: %s: %s: %w", mj.name, n, err)
		}
		if p.indexed && t.Each != "" {
			return fmt.Errorf("convert: %s: %s takes i from its name and counts its targets too", mj.name, n)
		}
		c := target{t: t, role: retarget(r)}
		for _, x := range []struct {
			src string
			to  **exprNode
		}{{t.Each, &c.each}, {t.Where, &c.where}, {t.Block, &c.block}} {
			if x.src == "" {
				continue
			}
			e, err := parseExpr(x.src)
			if err != nil {
				return fmt.Errorf("convert: %s: %s: %w", mj.name, n, err)
			}
			*x.to = e
		}
		cuts[k] = c
	}
	p.role = cuts[0].role
	p.split = func(e jlm.Tensor, data []byte, c *Config) ([]jlm.Tensor, error) {
		st.mu.Lock()
		env := st.env
		st.mu.Unlock()
		at := 0
		if p.indexed {
			at = p.index(e.Name)
		}
		in := mjVal{shape: mjTorch(&e), typ: e.Type, data: data}
		var out []jlm.Tensor
		for _, cut := range cuts {
			runs := 1
			if cut.each != nil {
				v, err := evalMJ(cut.each, env)
				if err != nil {
					return nil, fmt.Errorf("convert: %s: %v: %w", e.Name, cut.role, err)
				}
				m, err := exprUint("each", v)
				if err != nil || m > 1<<16 {
					return nil, fmt.Errorf("convert: %s: %v: %d cuts (%v)", e.Name, cut.role, m, err)
				}
				runs = int(m)
			}
			for j := range runs {
				i := at
				if cut.each != nil {
					i = j
				}
				ienv := func(path string) any {
					if path == "i" {
						return float64(i)
					}
					return env(path)
				}
				if cut.where != nil {
					on, err := evalBool(cut.where, ienv)
					if err != nil {
						return nil, fmt.Errorf("convert: %s: %v: %w", e.Name, cut.role, err)
					}
					if !on {
						continue
					}
				}
				part := e
				if cut.block != nil {
					v, err := evalMJ(cut.block, ienv)
					if err != nil {
						return nil, fmt.Errorf("convert: %s: %v: %w", e.Name, cut.role, err)
					}
					b, err := exprUint("block", v)
					if err != nil || b >= uint64(c.NLayer) {
						return nil, fmt.Errorf("convert: %s: %v goes to block %d of %d (%v)", e.Name, cut.role, b, c.NLayer, err)
					}
					part.Block = int32(b)
				}
				v, err := cut.t.Do.run(ienv, []mjVal{in})
				if err != nil {
					return nil, fmt.Errorf("convert: %s: %v: %w", e.Name, cut.role, err)
				}
				part.Role, part.Name = cut.role, fmt.Sprintf("%s[%v %d]", e.Name, cut.role, i)
				if err := mjSetShape(&part, v); err != nil {
					return nil, err
				}
				part.Data = v.data
				out = append(out, part)
			}
		}
		if len(out) == 0 {
			return nil, fmt.Errorf("convert: %s: no target of %s is cut from it", e.Name, n)
		}
		return out, nil
	}
	a.placed = append(a.placed, p)
	return nil
}
