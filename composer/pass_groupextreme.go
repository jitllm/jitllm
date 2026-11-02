package composer

import (
	"math"

	"github.com/jitllm/jitllm/composer/ir"
)

// GroupExtreme runs an argmax whose elements leave the device's threads idle
// as a workgroup an element: thread t takes the loop's terms t, t+G, ...,
// its first largest and the value carried with it, and the workgroup keeps
// the largest of the threads' and, of equal ones, the value of the earliest
// term. G is the fewest threads, a power of two from a subgroup to 1024,
// that take at most Trips terms each.
type GroupExtreme struct {
	Trips int
	Kinds []PrimKind
}

func (GroupExtreme) Name() string { return "groupextreme" }

func (s GroupExtreme) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(s.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) { return s.apply(k, d) })
}

func (s GroupExtreme) apply(n *Kernel, d Device) ([]*Kernel, bool, error) {
	if d.Kind != DeviceGPU || n.Work || !covers(s.Kinds, n.Kind) || n.Few > 0 || n.Many > 0 || n.Capacity > 0 {
		return nil, false, nil
	}
	c, ext, car, vs, cs, ok := extremeSplittable(n)
	if !ok || n.Nodes[ext].Type != ir.F32 || n.Nodes[n.Nodes[ext].A].Op != OpMax {
		return nil, false, nil
	}
	T, E := n.Loops[c].Trip, n.Elements()
	if T <= s.Trips || T >= 1<<24 || d.Slots > 0 && E >= d.Slots {
		return nil, false, nil
	}
	G := ir.SubgroupLanes
	for G < 1024 && G*s.Trips < T {
		G *= 2
	}
	term := n.extremeTerm(ext)
	won := n.Nodes[n.Nodes[car].A].A
	it := n.iterOf(c)
	if term < 0 || it < 0 || n.Nodes[won].Op != OpLess {
		return nil, false, nil
	}
	U := ir.U32
	g := n.clone()
	g.Name = n.Name + ".group"
	g.Work, g.Group = true, G
	g.Loops[c].Trip = (T + G - 1) / G
	tid := g.add(Node{Op: OpThread, Type: U})
	step := g.bin(OpMul, U, it, g.u32(G))
	k := g.bin(OpAdd, U, tid, step)
	at := g.bin(OpMin, U, k, g.u32(T-1))
	g.redirect(it, at, step)
	if T%G != 0 {
		ninf := g.f32(math.Float32bits(float32(math.Inf(-1))))
		guarded := g.add(Node{Op: OpSelect, Type: ir.F32, A: g.bin(OpLess, ir.Pred, k, g.u32(T)), B: term, C: ninf})
		g.redirect(term, guarded, guarded)
	}
	pos := g.add(Node{Op: OpFold, At: c, Type: U, B: g.u32(0)})
	g.Nodes[pos].A = g.add(Node{Op: OpSelect, Type: U, A: won, B: k, C: pos})

	best := g.un(OpGroupMax, ir.F32, ext)
	ninf := g.f32(math.Float32bits(float32(math.Inf(-1))))
	negPos := g.bin(OpSub, ir.F32, g.f32(0), g.un(OpCvtF32, ir.F32, pos))
	key := g.add(Node{Op: OpSelect, Type: ir.F32, A: g.bin(OpLess, ir.Pred, ext, best), B: ninf, C: negPos})
	first := g.un(OpCvtI32, ir.I32, g.bin(OpSub, ir.F32, g.f32(0), g.un(OpGroupMax, ir.F32, key)))
	from := g.bin(OpRem, U, g.un(OpBitcast, U, first), g.u32(G))
	ct := n.Nodes[car].Type
	bits := car
	if ct != ir.F32 {
		bits = g.un(OpBitcast, ir.F32, car)
	}
	got := g.add(Node{Op: OpExchange, Type: ir.F32, A: bits, B: from, C: -1, Imm: 1, At: int(OpMax)})
	if ct != ir.F32 {
		got = g.un(OpBitcast, ct, got)
	}
	g.Stores[cs].Val = got
	if vs >= 0 {
		g.Stores[vs].Val = best
	}
	return []*Kernel{g}, true, nil
}
