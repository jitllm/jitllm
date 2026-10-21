// Package jit is the composer: a block's graph ops decomposed into
// primitives, each lowered naively to a loop nest, transformed by registered
// passes, and emitted for the device in front of it.
package jit

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/ir"
)

// Block is what the composer forms kernels for: one block's graph at a step
// of Rows rows, its weights' storage formats by role.
type Block struct {
	Graph   *graph.Graph
	Rows    int
	Formats map[jlm.Role]jlm.Type
	// Page is the history's positions a page.
	Page int
}

// Kernel is one formed kernel and the passes that formed it.
type Kernel struct {
	IR     *ir.Kernel
	Passes []string
}

// Compose forms every kernel of block b on target t under profile p. An op
// it cannot form is an error, beside the kernels of those it formed.
func Compose(b Block, t Target, p Profile) ([]Kernel, error) {
	prims, derr := decompose(b, p)
	var prog []*Nest
	for _, pr := range prims {
		nests, err := lower(pr)
		if err != nil {
			return nil, err
		}
		for _, n := range nests {
			n.Kind = pr.Kind
			n.Prim = &pr
		}
		prog = append(prog, nests...)
	}
	prog, err := p.run(prog, t)
	if err != nil {
		return nil, err
	}
	var out []Kernel
	for _, n := range prog {
		k, err := emit(n, t)
		if err != nil {
			return nil, fmt.Errorf("jit: %s: %w\n%s", n.Name, err, n)
		}
		out = append(out, Kernel{IR: k, Passes: n.Passes})
	}
	return out, derr
}
