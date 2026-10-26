package jit

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// Request is what a runtime call site asks the composer to form: a slice of
// a block's graph, its weights' storage formats, the rotaries its tables are
// named from, the step's shape and the device.
type Request struct {
	Graph    *graph.Graph
	Part     PartKind
	Formats  map[jlm.Role]jlm.Type
	Rotaries []graph.Rotary
	Step     Step
	Device   Device
	// History is the layout of the pools the slice's history ops write and
	// read: the call site's, addressed as it is.
	History History
}

// Form is req's kernels in launch order, and the Key they are cached under:
// the slice decomposed, generated, passed and lowered as a model's part is
// (Kernels, Lower), once per Key. What it cannot form is an error naming the
// op and the slice's shape, remembered as the kernels are. The codes are
// shared by every request of the Key and are not the caller's to change.
func (r *Registry) Form(req Request) (Key, []Code, error) {
	if req.Graph == nil {
		return 0, nil, fmt.Errorf("jit: a request with no graph")
	}
	key, words, err := r.keyOf(req)
	if err != nil {
		return 0, nil, err
	}
	c := r.cache
	if c == nil {
		codes, err := r.form(req)
		return key, codes, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if f, ok := c.m[key]; ok {
		if !slices.Equal(f.words, words) {
			return 0, nil, fmt.Errorf("jit: two requests share the key %#x", key)
		}
		return key, f.codes, f.err
	}
	codes, err := r.form(req)
	c.m[key] = formed{words: words, codes: codes, err: err}
	return key, codes, err
}

func (r *Registry) form(req Request) ([]Code, error) {
	prof, ok := r.Profiles[req.Device.Kind]
	if !ok {
		return nil, fmt.Errorf("jit: no profile for %s", req.Device.Kind)
	}
	sp := newExprSplit()
	var ks []*Kernel
	for {
		c := r.context(req.Graph, req.Part, req.Formats, req.Rotaries, req.Step, req.Device)
		c.History, c.split = req.History, sp
		prims, err := r.decomposeOps(c, 0)
		if err != nil {
			return nil, fmt.Errorf("jit: cannot form %s at %d rows on %s: %w", slice(req.Graph), req.Step.Rows, req.Device.Kind, err)
		}
		ks = nil
		for _, p := range prims {
			k, err := generate(p)
			if err != nil {
				return nil, fmt.Errorf("jit: cannot form %s: %w", slice(req.Graph), err)
			}
			k.Part, k.Unfused, k.Unjammed, k.Unpaired, k.Unwide, k.Uncursored = req.Part, sp.unfused[k.Of], sp.unjammed[k.Of], sp.unpaired[k.Of], sp.unwide[k.Of], sp.uncursored[k.Of]
			k.Capacity = liveOf(p, req.Step, req.Device)
			ks = append(ks, k)
		}
		ks, err = prof.run(ks, req.Device)
		if sp.splitRefused(err) {
			continue
		}
		if err != nil {
			return nil, err
		}
		break
	}
	codes, err := r.Lower(ks, req.Device)
	if err != nil {
		return nil, err
	}
	out := make([]Code, len(codes))
	for i, c := range codes {
		out[i] = c.Code
	}
	return out, nil
}

// slice names a request's graph by its ops, for an error.
func slice(g *graph.Graph) string {
	s := ""
	for i, o := range g.Ops {
		if i > 0 {
			s += "+"
		}
		s += o.Kind.String()
	}
	return s
}

// liveOf is the capacity in rows a primitive of a step of several rows is
// formed for, its leading axis the step's rows: a call runs the rows its
// table states (tableCall), at most these (Kernel.Capacity) -- a GPU's
// launch held to them (CallRows), a CPU's calls (RowCalls). 0 for a one-row
// step's, whose form is decode's, and for a scan's, whose launch is its
// state rows and whose trips are the step's rows.
func liveOf(p Prim, s Step, d Device) int {
	if s.Rows < 2 || len(p.Shape) == 0 || p.Shape[0] != s.Rows || p.Kind == PrimScan {
		return 0
	}
	return s.Rows
}
