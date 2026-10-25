package graph

import (
	"errors"
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
)

// Shape is a stored tensor's shape: K its input width (a vector's length),
// Rows its output rows (0 for a vector), Sheets a bank's experts (0 for one
// matrix).
type Shape struct{ K, Rows, Sheets int }

// Facts is what the container stores for a block, independently of any graph
// built for it: its tensors by role. The verifier holds a graph to them.
type Facts struct {
	Inventory map[jlm.Role]Shape
}

// Verify checks g's structure before anything is formed, and returns every
// violation found, joined: the op language's version; that every value has
// one producer and is defined before it is read; that every state read sees
// the version in force and every write advances it by one; that every
// binding names a tensor the container stores at the shape the op binds; that
// a matrix product's widths are its weight's and a norm's scale its row's; and
// that the outputs are produced. What a graph computes is held elsewhere, by
// running it against a reference; this checks what can be checked without.
func Verify(g *Graph, f Facts) error {
	if g == nil {
		return errors.New("graph: no graph")
	}
	if g.Version != Version {
		return fmt.Errorf("graph: block %d is op language v%d; this build reads v%d", g.Block, g.Version, Version)
	}
	var errs []error
	bad := func(format string, a ...any) { errs = append(errs, fmt.Errorf(format, a...)) }
	n := len(g.Values)
	defined := make([]bool, n)
	valid := func(id ValueID) bool { return id >= 0 && int(id) < n }
	width := func(id ValueID) int {
		if valid(id) {
			return g.Values[id].Width
		}
		return -1
	}
	// A block reads the residual; a model-wide graph (the input rows) may
	// read nothing but its step.
	if len(g.In) == 0 && g.Block >= 0 {
		bad("graph: block %d has no input", g.Block)
	}
	for _, id := range g.In {
		if !valid(id) {
			bad("graph: block %d's input %d is not a value", g.Block, id)
			continue
		}
		defined[id] = true
	}
	state := map[string]int{}
	for i, o := range g.Ops {
		at := fmt.Sprintf("graph: block %d op %d (%s)", g.Block, i, o.Kind)
		if o.Kind <= OpInvalid || o.Kind >= numOps {
			bad("%s: unknown op", at)
			continue
		}
		for _, id := range o.In {
			switch {
			case !valid(id):
				bad("%s: input %d is not a value", at, id)
			case !defined[id]:
				bad("%s: reads %%%s before anything writes it", at, g.Values[id].Name)
			}
		}
		for _, id := range o.Out {
			switch {
			case !valid(id):
				bad("%s: output %d is not a value", at, id)
			case defined[id]:
				bad("%s: writes %%%s, which already has a producer", at, g.Values[id].Name)
			default:
				defined[id] = true
			}
		}
		for _, r := range o.Reads {
			if state[r.Name] != r.Version {
				bad("%s: reads %s@%d where %s@%d is in force", at, r.Name, r.Version, r.Name, state[r.Name])
			}
		}
		for _, w := range o.Writes {
			if w.Version != state[w.Name]+1 {
				bad("%s: writes %s@%d over %s@%d", at, w.Name, w.Version, w.Name, state[w.Name])
			}
			state[w.Name] = w.Version
		}
		for _, w := range o.Weight {
			s, ok := f.Inventory[w.Role]
			switch {
			case !ok:
				bad("%s: binds %v, which the container does not store for the block", at, w.Role)
			case s != (Shape{w.K, w.Rows, w.Sheets}):
				bad("%s: binds %v as %dx%dx%d, stored %dx%dx%d", at, w.Role, w.K, w.Rows, w.Sheets, s.K, s.Rows, s.Sheets)
			}
		}
		switch o.Kind {
		case OpMatVec:
			if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) == 0 || len(o.Weight) > 2 {
				bad("%s: wants one input, one output, a matrix and at most a bias", at)
				break
			}
			if w := o.Weight[0]; w.K != width(o.In[0]) || w.Rows != width(o.Out[0]) {
				bad("%s: %v is %dx%d between widths %d and %d", at, w.Role, w.K, w.Rows, width(o.In[0]), width(o.Out[0]))
			}
			if len(o.Weight) == 2 && (o.Weight[1].Rows != 0 || o.Weight[1].K != width(o.Out[0])) {
				bad("%s: bias %v of %d on a row of %d", at, o.Weight[1].Role, o.Weight[1].K, width(o.Out[0]))
			}
		case OpNorm, OpLayerNorm:
			if len(o.In) != 1 || len(o.Out) != 1 || len(o.Weight) == 0 {
				bad("%s: wants one input, one output and a scale", at)
				break
			}
			if o.Weight[0].K != width(o.In[0]) || width(o.Out[0]) != width(o.In[0]) {
				bad("%s: a scale of %d over a row of %d into %d", at, o.Weight[0].K, width(o.In[0]), width(o.Out[0]))
			}
			if o.Kind == OpLayerNorm && (len(o.Weight) > 2 || len(o.Weight) == 2 && (o.Weight[1].Rows != 0 || o.Weight[1].K != width(o.In[0]))) {
				bad("%s: %d weights where a scale and a bias of the row's %d are", at, len(o.Weight), width(o.In[0]))
			}
		case OpHeadNorm:
			if len(o.In) != 1 || len(o.Out) != 1 || o.Attr.HeadDim <= 0 || width(o.In[0])%o.Attr.HeadDim != 0 {
				bad("%s: heads of %d over a row of %d", at, o.Attr.HeadDim, width(o.In[0]))
				break
			}
			if len(o.Weight) > 0 && o.Weight[0].K != o.Attr.HeadDim {
				bad("%s: a scale of %d over heads of %d", at, o.Weight[0].K, o.Attr.HeadDim)
			}
		case OpAdd, OpActMul:
			if len(o.In) != 2 || len(o.Out) != 1 || width(o.In[0]) != width(o.In[1]) || width(o.Out[0]) != width(o.In[0]) {
				bad("%s: widths %d and %d into %d", at, width(o.In[0]), width(o.In[1]), width(o.Out[0]))
			}
		case OpMap:
			if a := o.Attr; a.Width != 0 {
				switch {
				case len(o.Weight) != 1:
					bad("%s: a window of %d on %d bound vectors", at, a.Width, len(o.Weight))
				case a.Width < 0 || a.Offset < 0 || a.Offset+a.Width > o.Weight[0].K:
					bad("%s: a window of %d from %d on a vector of %d", at, a.Width, a.Offset, o.Weight[0].K)
				}
			}
		}
	}
	for _, id := range g.Out {
		if !valid(id) || !defined[id] {
			bad("graph: block %d's output %d is never written", g.Block, id)
		}
	}
	return errors.Join(errs...)
}

// Unbound is every tensor role f's inventory stores that no op of g binds,
// in role order: a stored tensor no op reads is a graph that dropped part of
// the block.
func Unbound(g *Graph, f Facts) []jlm.Role {
	bound := map[jlm.Role]bool{}
	for _, o := range g.Ops {
		for _, w := range o.Weight {
			bound[w.Role] = true
		}
	}
	var out []jlm.Role
	for r := range f.Inventory {
		if !bound[r] {
			out = append(out, r)
		}
	}
	slices.Sort(out)
	return out
}
