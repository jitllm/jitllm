package tier

import (
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// kernelSharedMem is the static threadgroup memory a generated kernel may
// declare: 48 KiB, CUDA's static per-block limit without the dynamic opt-in
// (CUDA C Programming Guide, "Technical Specifications per Compute
// Capability"), and the bound every kernel builder here checks against
// (kernels.ActIn, GemmVolta, GemmInt8). It is this code's choice on Vulkan
// and Metal, not a figure read from the device.
const kernelSharedMem = 48 << 10

// target is the graph.Target a rewrite's guards and the verdict keys ask on
// this device at row class rows: its backend, the code generator's ISA tier,
// the matrix instruction it lowers, the subgroup the IR's kernels take, and
// the threadgroup memory a kernel may declare. Every probe it reads is
// cached per device. Callers hold g.mu.
func (g *devTier) target(rows graph.RowClass) graph.Target {
	return graph.Target{
		API:       g.dev.API(),
		ISA:       backend.ISAOf(g.dev),
		MMA:       g.mmaShape(),
		Subgroup:  ir.SubgroupLanes,
		SharedMem: kernelSharedMem,
		Rows:      rows,
	}
}

// mmaShape is the widest matrix instruction the composer lowers on this
// device, "" for none.
func (g *devTier) mmaShape() string {
	if g.composeTarget().IntMMA() {
		return "m16n8k32"
	}
	return ""
}

// offer is the decision of the rewrite named name on block l at row class
// rows on this device: nil to offer its recipes, else the reason they are
// not offered. A block without a graph decides from the tier's own facts,
// which the caller checks; ok reports whether the graph decided.
func (g *devTier) offer(l *layer, name string, rows graph.RowClass) (ok bool, key string, err error) {
	o, found := graph.Find(l.rewrites, name)
	if !found && !l.graphed {
		// No graph, or a sublayer's that does not name this site.
		return false, "", nil
	}
	if !found {
		return true, "", errNoMatch(name)
	}
	t := g.target(rows)
	return true, o.Key(t), o.Offer(t)
}

// errNoMatch is the refusal of a rewrite that matched nowhere on the block.
type errNoMatch string

func (e errNoMatch) Error() string { return "the block's graph has no " + string(e) + " region" }
