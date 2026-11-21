package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
)

// SpanThreads forms a kernel's sums over runs (OpSpanSum) on a device whose
// elements are threads, in order: a run of at most a subgroup, on a device
// that guarantees a subgroup's lanes (Device.Lanes), a butterfly of
// exchanges across its lanes (OpShuffle) in a workgroup of whole runs; a
// longer run, or any on a device that guarantees no lanes, a sum over a
// workgroup of the run's width (OpGroupSum). A thread past the last
// element takes its place in the last run (Kernel.Spans).
type SpanThreads struct{}

func (SpanThreads) Name() string { return "spanthreads" }

// spanMost is the longest run a workgroup sums: the most threads a
// workgroup of every backend here holds.
const spanMost = 1024

func (p SpanThreads) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		S, err := k.spanOf()
		if S == 0 || err != nil {
			return nil, false, err
		}
		if k.Work || k.Group > 0 {
			return nil, false, fmt.Errorf("jit: %s sums runs over a workgroup a pass bound before", k.Name)
		}
		if most := min(spanMost, d.groupMost()); S > most {
			return nil, false, fmt.Errorf("jit: %s sums runs of %d, past the %d threads a workgroup holds", k.Name, S, most)
		}
		m := k.clone()
		m.Spans = S
		butterfly := S <= ir.SubgroupLanes && d.Lanes >= ir.SubgroupLanes
		if !butterfly {
			m.Group = S
		}
		for x := range m.Nodes {
			nd := m.Nodes[x]
			if nd.Op != OpSpanSum {
				continue
			}
			if !butterfly {
				m.Nodes[x] = Node{Op: OpGroupSum, Type: ir.F32, A: nd.A}
				continue
			}
			m.replaceUses(x, m.laneSum(nd.A, S))
		}
		return []*Kernel{m}, true, nil
	})
}

// spanOf is the run k's sums over runs take, 0 for a kernel with none.
func (k *Kernel) spanOf() (int, error) {
	S := 0
	for _, nd := range k.Nodes {
		if nd.Op != OpSpanSum {
			continue
		}
		if s := int(nd.Imm); s < 1 || s&(s-1) != 0 || S != 0 && s != S {
			return 0, fmt.Errorf("jit: %s sums runs of %d beside runs of %d", k.Name, s, S)
		}
		S = int(nd.Imm)
	}
	return S, nil
}

// laneSum is float v summed over each run of w lanes it is in, a butterfly
// of exchanges across them.
func (k *Kernel) laneSum(v, w int) int {
	for m := 1; m < w; m *= 2 {
		v = k.bin(OpAdd, ir.F32, v, k.add(Node{Op: OpShuffle, Type: ir.F32, A: v, Imm: int64(m)}))
	}
	return v
}

// replaceUses makes every read of node x a read of node v.
func (k *Kernel) replaceUses(x, v int) {
	for i := range k.Nodes {
		if i == v {
			continue
		}
		for _, a := range k.Operands(&k.Nodes[i]) {
			if *a == x {
				*a = v
			}
		}
	}
	for i := range k.Stores {
		s := &k.Stores[i]
		if s.Barrier || s.Run {
			continue
		}
		for _, a := range k.storeArgs(s) {
			if *a == x {
				*a = v
			}
		}
	}
}
