package composer

import "github.com/jitllm/jitllm/composer/ir"

// Fit gives a kernel of one workgroup whose threads are its elements no
// more threads than its elements, rounded up to a subgroup. A surplus
// thread is clamped to the last element and computes it again beside the
// thread that owns it: on the M4 an iteration of 32 rows in a workgroup of
// 128 ran four subgroups of the same chains through one core's pipes. A
// kernel whose threads work together through their workgroup (shares),
// read their place in it or deal a loop's trips over it keeps its width.
// It runs last, after every pass that reads a kernel's width or merges
// kernels of one width (Parts, Phases), so what launched together still
// does.
type Fit struct{}

func (Fit) Name() string { return "fit" }

func (p Fit) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		w := (max(k.Elements(), 1) + ir.SubgroupLanes - 1) / ir.SubgroupLanes * ir.SubgroupLanes
		if w >= k.Group || !fits(k) {
			return nil, false, nil
		}
		m := k.clone()
		m.Group = w
		return []*Kernel{m}, true, nil
	})
}

// fits says k's threads are its elements and nothing else: one workgroup's
// elements, each thread's its own, no thread reading its place in the
// workgroup or another's value, and no loop dealt over the workgroup.
func fits(k *Kernel) bool {
	if k.Group <= 0 || k.Work || k.groups() != 1 || k.Kind == PrimMatrix || k.Kind == PrimScan || shares(k) ||
		k.Heads != nil || k.Resident != 0 || k.Tile != 0 || k.TileRows != 0 || k.Matrix != [2]int{} {
		return false
	}
	for _, nd := range k.Nodes {
		if nd.Tile != nil || nd.Op == OpThread || nd.Op == OpLocal || nd.Op == OpMMA {
			return false
		}
	}
	for _, l := range k.Loops {
		if l.Spread || l.Bound != Unbound || l.Unit != 0 {
			return false
		}
	}
	return true
}
