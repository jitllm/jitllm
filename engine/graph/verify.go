package graph

import (
	"errors"
	"fmt"
	"sort"

	"github.com/jitllm/jitllm/format/jlm"
)

// Shape is a stored tensor's shape: K its input width (a vector's length),
// Rows its output rows (0 for a vector).
type Shape struct{ K, Rows int }

// Facts is what the container states about a block, independently of any
// graph built for it: the configuration (the rotary, the epsilon, the head
// counts) and the tensors stored for the block by role. The verifier holds a
// graph to them.
type Facts struct {
	Config    Dense
	Inventory map[jlm.Role]Shape
}

// Verify checks g before anything is allocated or emitted and returns every
// violation found, joined: the version; that every value has one producer
// and is defined before it is read; that widths agree with the weights each
// op binds; that every state read has its producing write ordered before it
// and every attention reads a version this block wrote; that every binding
// names a stored tensor of the shape the op needs and every stored tensor is
// bound by some op; that the rotary, norms and attention carry what the
// configuration states; and that the outputs are produced. It sizes nothing
// from the graph's counts.
func Verify(g *Graph, f Facts) error {
	var errs []error
	bad := func(format string, a ...any) { errs = append(errs, fmt.Errorf(format, a...)) }
	if g == nil {
		return errors.New("graph: no graph")
	}
	if g.Version != Version {
		return fmt.Errorf("graph: block %d is op language v%d; this build reads v%d", g.Block, g.Version, Version)
	}
	n := len(g.Values)
	defined := make([]bool, n)
	valid := func(id ValueID) bool { return id >= 0 && int(id) < n }
	for _, id := range g.In {
		if !valid(id) {
			bad("graph: block input %d is not a value", id)
			continue
		}
		defined[id] = true
	}
	if len(g.In) == 0 {
		bad("graph: block %d has no input", g.Block)
	}
	width := func(id ValueID) int {
		if valid(id) {
			return g.Values[id].Width
		}
		return -1
	}
	state := map[string]int{}
	bound := map[jlm.Role]bool{}
	c := &f.Config
	for i, o := range g.Ops {
		at := fmt.Sprintf("graph: block %d op %d (%s)", g.Block, i, o.Kind)
		if o.Kind <= OpInvalid || o.Kind >= numOps {
			bad("%s: unknown op", at)
			continue
		}
		for _, id := range o.In {
			if !valid(id) {
				bad("%s: input %d is not a value", at, id)
			} else if !defined[id] {
				bad("%s: reads %%%s before anything writes it", at, g.Values[id].Name)
			}
		}
		for _, id := range o.Out {
			if !valid(id) {
				bad("%s: output %d is not a value", at, id)
			} else if defined[id] {
				bad("%s: writes %%%s, which already has a producer", at, g.Values[id].Name)
			} else {
				defined[id] = true
			}
		}
		for _, r := range o.Reads {
			if state[r.Name] != r.Version {
				bad("%s: reads %s@%d, but %s@%d is the version in force here", at, r.Name, r.Version, r.Name, state[r.Name])
			}
		}
		for _, w := range o.Writes {
			if w.Version != state[w.Name]+1 {
				bad("%s: writes %s@%d over %s@%d", at, w.Name, w.Version, w.Name, state[w.Name])
			}
			state[w.Name] = w.Version
		}
		for _, w := range o.Weight {
			bound[w.Role] = true
			s, ok := f.Inventory[w.Role]
			switch {
			case !ok:
				bad("%s: binds %v, which the container does not store for the block", at, w.Role)
			case s.K != w.K || s.Rows != w.Rows:
				bad("%s: binds %v as %dx%d, stored %dx%d", at, w.Role, w.K, w.Rows, s.K, s.Rows)
			}
		}
		ins := func(k int) bool {
			if len(o.In) != k {
				bad("%s: %d inputs, wants %d", at, len(o.In), k)
				return false
			}
			return true
		}
		outs := func(k int) bool {
			if len(o.Out) != k {
				bad("%s: %d outputs, wants %d", at, len(o.Out), k)
				return false
			}
			return true
		}
		switch o.Kind {
		case OpNorm, OpHeadNorm:
			if !ins(1) || !outs(1) || len(o.Weight) != 1 {
				bad("%s: wants one input, one output and one scale", at)
				break
			}
			if o.Attr.Eps != c.Eps {
				bad("%s: epsilon %g, the configuration states %g", at, o.Attr.Eps, c.Eps)
			}
			if width(o.Out[0]) != width(o.In[0]) {
				bad("%s: width %d into %d", at, width(o.In[0]), width(o.Out[0]))
			}
			if o.Kind == OpNorm && o.Weight[0].K != width(o.In[0]) {
				bad("%s: scale of %d over a row of %d", at, o.Weight[0].K, width(o.In[0]))
			}
			if o.Kind == OpHeadNorm && (o.Attr.HeadDim != c.HeadDim || o.Weight[0].K != c.HeadDim || width(o.In[0])%c.HeadDim != 0) {
				bad("%s: heads of %d with a scale of %d over a row of %d; the configuration states heads of %d",
					at, o.Attr.HeadDim, o.Weight[0].K, width(o.In[0]), c.HeadDim)
			}
		case OpMatVec:
			if !ins(1) || !outs(1) || len(o.Weight) == 0 || len(o.Weight) > 2 {
				bad("%s: wants one input, one output, a matrix and at most a bias", at)
				break
			}
			if w := o.Weight[0]; w.K != width(o.In[0]) || w.Rows != width(o.Out[0]) {
				bad("%s: %v is %dx%d between widths %d and %d", at, w.Role, w.K, w.Rows, width(o.In[0]), width(o.Out[0]))
			}
			if len(o.Weight) == 2 && (o.Weight[1].Rows != 0 || o.Weight[1].K != width(o.Out[0])) {
				bad("%s: bias %v of %d on a row of %d", at, o.Weight[1].Role, o.Weight[1].K, width(o.Out[0]))
			}
		case OpRope:
			if !ins(1) || !outs(1) {
				break
			}
			a := o.Attr
			if a.Layout != c.Layout || a.Freqs != c.RopeFreqs || a.NRot != c.NRot || a.Base != c.Base || a.HeadDim != c.HeadDim {
				bad("%s: %v pairs over %d of %d at base %g; the configuration states %v over %d of %d at %g",
					at, a.Layout, a.NRot, a.HeadDim, a.Base, c.Layout, c.NRot, c.HeadDim, c.Base)
			}
			if width(o.In[0])%c.HeadDim != 0 || width(o.Out[0]) != width(o.In[0]) {
				bad("%s: a row of %d is not whole heads of %d", at, width(o.In[0]), c.HeadDim)
			}
		case OpKVAppend:
			if !ins(2) || !outs(0) {
				break
			}
			if len(o.Writes) != 1 || o.Writes[0].Name != "kv" {
				bad("%s: writes no kv version", at)
			}
			if kvd := c.NKVHead * c.HeadDim; width(o.In[0]) != kvd || width(o.In[1]) != kvd {
				bad("%s: k of %d and v of %d into rows of %d", at, width(o.In[0]), width(o.In[1]), kvd)
			}
		case OpAttend:
			if !ins(1) || !outs(1) {
				break
			}
			a := o.Attr
			if len(o.Reads) != 1 || o.Reads[0].Name != "kv" || o.Reads[0].Version == 0 {
				bad("%s: reads no kv version this block wrote: this position's row is never in the history", at)
			}
			if a.NHead != c.NHead || a.NKVHead != c.NKVHead || a.HeadDim != c.HeadDim || a.Scale != c.Scale ||
				a.Window != c.Window || a.Softcap != c.Softcap {
				bad("%s: %d/%d heads of %d, scale %g, window %d, softcap %g; the configuration states %d/%d of %d, %g, %d, %g",
					at, a.NHead, a.NKVHead, a.HeadDim, a.Scale, a.Window, a.Softcap,
					c.NHead, c.NKVHead, c.HeadDim, c.Scale, c.Window, c.Softcap)
			}
			if width(o.In[0]) != c.NHead*c.HeadDim || width(o.Out[0]) != c.NHead*c.HeadDim {
				bad("%s: q of %d, out of %d", at, width(o.In[0]), width(o.Out[0]))
			}
		case OpAdd, OpActMul:
			if !ins(2) || !outs(1) {
				break
			}
			if width(o.In[0]) != width(o.In[1]) || width(o.Out[0]) != width(o.In[0]) {
				bad("%s: widths %d and %d into %d", at, width(o.In[0]), width(o.In[1]), width(o.Out[0]))
			}
			if o.Kind == OpActMul && o.Attr.Act != c.Act {
				bad("%s: activation %d, the configuration states %d", at, o.Attr.Act, c.Act)
			}
		}
	}
	for _, id := range g.Out {
		if !valid(id) || !defined[id] {
			bad("graph: block %d output %d is never produced", g.Block, id)
		}
	}
	if len(g.Out) == 0 {
		bad("graph: block %d has no output", g.Block)
	}
	var unbound []string
	for r := range f.Inventory {
		if !bound[r] {
			unbound = append(unbound, r.String())
		}
	}
	sort.Strings(unbound)
	for _, r := range unbound {
		bad("graph: block %d stores %s, which no op binds: the graph does not compute what the container holds", g.Block, r)
	}
	return errors.Join(errs...)
}
