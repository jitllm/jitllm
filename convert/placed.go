package convert

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/convert/meta"
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

// placedName is the pattern source name n states: n itself, or where it
// holds {i} the text before and after the index.
func placedName(n string) (hfPlaced, error) {
	pre, post, ok := strings.Cut(n, "{i}")
	if !ok {
		return hfPlaced{pre: n}, nil
	}
	if strings.Contains(post, "{i}") {
		return hfPlaced{}, fmt.Errorf("%s names {i} twice", n)
	}
	return hfPlaced{pre: pre, post: post, indexed: true}, nil
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

// placedCut is a placed target, its expressions parsed.
type placedCut struct {
	t                  mjTensor
	each, where, block *exprNode
}

func parsePlaced(t mjTensor) (placedCut, error) {
	c := placedCut{t: t}
	for _, x := range []struct {
		src string
		to  **exprNode
	}{{t.Each, &c.each}, {t.Where, &c.where}, {t.Block, &c.block}} {
		if x.src == "" {
			continue
		}
		e, err := parseExpr(x.src)
		if err != nil {
			return c, err
		}
		*x.to = e
	}
	return c, nil
}

// cuts calls fn for each target c cuts from a source whose name's index is
// at: its i, its block (-1 where c states none) and the config with i.
func (c placedCut) cuts(env exprEnv, at int, fn func(i int, block int64, ienv exprEnv) error) error {
	runs := 1
	if c.each != nil {
		v, err := evalMJ(c.each, env)
		if err != nil {
			return err
		}
		m, err := exprUint("each", v)
		if err != nil || m > 1<<16 {
			return fmt.Errorf("%d cuts (%v)", m, err)
		}
		runs = int(m)
	}
	for j := range runs {
		i := at
		if c.each != nil {
			i = j
		}
		ienv := func(path string) any {
			if path == "i" {
				return float64(i)
			}
			return env(path)
		}
		if c.where != nil {
			on, err := evalBool(c.where, ienv)
			if err != nil {
				return err
			}
			if !on {
				continue
			}
		}
		b := int64(-1)
		if c.block != nil {
			v, err := evalMJ(c.block, ienv)
			if err != nil {
				return err
			}
			u, err := exprUint("block", v)
			if err != nil || u > 1<<20 {
				return fmt.Errorf("block %d (%v)", u, err)
			}
			b = int64(u)
		}
		if err := fn(i, b, ienv); err != nil {
			return err
		}
	}
	return nil
}

// placedTargets is the targets the model table places from source name,
// and the index its name matches, where the table places any.
func (mj *modelJSON) placedTargets(name string) (mjTargets, int, bool) {
	for n, ts := range mj.Tensors.Model {
		p, err := placedName(n)
		if err != nil || (!p.indexed && !anyPlaced(ts)) {
			continue
		}
		if i := p.index(name); i >= 0 {
			return ts, i, true
		}
	}
	return nil, 0, false
}

func anyPlaced(ts mjTargets) bool {
	for _, t := range ts {
		if t.placed() {
			return true
		}
	}
	return false
}

// addPlaced states model-level source n, whose targets go to blocks: each
// target cut by its primitives over the config and i, Each times where it
// states a count, where its Where holds, into the block its Block names.
func (mj *modelJSON) addPlaced(a *hfArch, st *mjState, n string, ts mjTargets, retarget func(jlm.Role) jlm.Role) error {
	p, err := placedName(n)
	if err != nil {
		return fmt.Errorf("convert: %s: %w", mj.name, err)
	}
	type target struct {
		placedCut
		role jlm.Role
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
		c, err := parsePlaced(t)
		if err != nil {
			return fmt.Errorf("convert: %s: %s: %w", mj.name, n, err)
		}
		cuts[k] = target{placedCut: c, role: retarget(r)}
	}
	p.role = cuts[0].role
	p.split = func(e jlm.Tensor, data []byte, c *Config) ([]jlm.Tensor, error) {
		st.mu.Lock()
		env := st.env
		st.mu.Unlock()
		in := mjVal{shape: mjTorch(&e), typ: e.Type, data: data}
		var out []jlm.Tensor
		for _, cut := range cuts {
			err := cut.cuts(env, p.index(e.Name), func(i int, b int64, ienv exprEnv) error {
				part := e
				if b >= 0 {
					if b >= int64(c.NLayer) {
						return fmt.Errorf("goes to block %d of %d", b, c.NLayer)
					}
					part.Block = int32(b)
				}
				v, err := cut.t.Do.run(ienv, []mjVal{in})
				if err != nil {
					return err
				}
				part.Role, part.Name = cut.role, fmt.Sprintf("%s[%v %d]", e.Name, cut.role, i)
				if err := mjSetShape(&part, v); err != nil {
					return err
				}
				part.Data = v.data
				out = append(out, part)
				return nil
			})
			if err != nil {
				return nil, fmt.Errorf("convert: %s: %v: %w", e.Name, cut.role, err)
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

// placedStandIns are source n's placed targets standing in, each under the
// name its block gives its role, at the shape its primitives cut it to.
func placedStandIns(env exprEnv, n string, shape []uint64, ts mjTargets, at int, have map[string]bool) ([]meta.Tensor, error) {
	var out []meta.Tensor
	for _, t := range ts {
		c, err := parsePlaced(t)
		if err != nil {
			return nil, fmt.Errorf("%s as %s: %w", n, t.Role, err)
		}
		err = c.cuts(env, at, func(i int, b int64, ienv exprEnv) error {
			gn := t.Role
			if b >= 0 {
				gn = "blk." + itoa(int(b)) + "." + t.Role
			}
			if have[gn] {
				return nil
			}
			have[gn] = true
			sh := shape
			if len(t.Do) > 0 {
				v, err := t.Do.run(ienv, []mjVal{{shape: shape, typ: jlm.TypeF32}})
				if err != nil {
					return err
				}
				sh = v.shape
			}
			dims := make([]uint64, len(sh))
			for k, d := range sh {
				dims[len(sh)-1-k] = d
			}
			out = append(out, meta.Tensor{Name: gn, Dims: dims})
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("%s as %s: %w", n, t.Role, err)
		}
	}
	return out, nil
}
