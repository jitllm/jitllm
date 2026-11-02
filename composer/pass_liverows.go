package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
)

// LiveRows holds a launch's threads to the elements of its live rows, where
// the step's tables say how many of its leading axis's rows are live: the
// live rows' elements, at least a row's and at most the space's.
type LiveRows struct{}

func (LiveRows) Name() string { return "liverows" }

func (p LiveRows) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Rows == 0 || k.Clamp != 0 || k.Work || len(k.Space) == 0 || k.Space[0] == 0 {
			return nil, false, nil
		}
		m := k.clone()
		E := m.Elements()
		inner := E / m.Space[0]
		live := m.bin(OpMin, ir.U32, m.bin(OpMul, ir.U32, m.Rows-1, m.u32(inner)), m.u32(E))
		m.Clamp = m.bin(OpMax, ir.U32, m.u32(inner), live) + 1
		m.RowsHeld = m.Capacity > 0
		return []*Kernel{m}, true, nil
	})
}

// CallRows holds a kernel of a form's capacity (Kernel.Capacity) to the
// call's rows, its table's word (tableCall), and sizes its launch to
// them (Code.LiveGroups), so one form serves every count of rows. A kernel
// whose threads each compute their own elements, or a workgroup kernel of
// one axis, has its threads (workgroups) clamped to the last element those
// rows reach, a surplus one recomputing it: no thread reads or writes a row
// the call does not state, a history store at a stale row's descriptor
// included. A kernel whose threads combine their values -- workgroup
// memory, an exchange, a shuffle, a combine over the group -- is held by its
// launch alone, its last workgroup's threads past the call's rows computing
// rows of the capacity no live row reads, as clamping them would hand a
// combine copies of another element; such a kernel reading or writing the
// history, which a stale row would address, is an error naming it -- unless
// its workgroups are a row each (a staged attention's), whose launch holds
// the call's rows exactly.
type CallRows struct{}

func (CallRows) Name() string { return "callrows" }

func (p CallRows) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Capacity == 0 || k.RowsHeld || len(k.Space) == 0 || k.Space[0] == 0 {
			return nil, false, nil
		}
		m := k.clone()
		m.RowsHeld = true
		if m.Work && len(m.Space) > 1 || !m.Work && combines(m) {
			if m.boundsSurplusHistory() {
				o, _ := m.history()
				return nil, false, fmt.Errorf("jit: %s combines across threads and binds %s, which a row past the call's would address",
					m.Name, o.Name)
			}
			return []*Kernel{m}, true, nil
		}
		lim := m.liveElements(m.Space, m.Capacity)
		if m.Clamp != 0 {
			lim = m.bin(OpMin, ir.U32, m.Clamp-1, lim)
		}
		m.Clamp = lim + 1
		return []*Kernel{m}, true, nil
	})
}

// boundsSurplusHistory reports whether k, held to a call's rows by its
// launch alone, has a surplus row reach the history: it combines across
// threads and binds the history, its workgroups not a row each (a staged
// attention's, whose launch holds the call's rows exactly).
func (k *Kernel) boundsSurplusHistory() bool {
	if !(k.Work && len(k.Space) > 1 || !k.Work && combines(k)) {
		return false
	}
	_, ok := k.history()
	return ok && !(k.Work && k.Space[0] == k.Capacity)
}

// combines reports whether k's threads combine their values: through
// workgroup memory, an exchange or a barrier (shares), a shuffle, or a
// combine over the workgroup.
func combines(k *Kernel) bool {
	if shares(k) {
		return true
	}
	for _, nd := range k.Nodes {
		switch nd.Op {
		case OpShuffle, OpGroupSum, OpGroupMax, OpLocal:
			return true
		}
	}
	return false
}

// history is a table of the step's history or its rows' descriptors k
// binds, which a row a call does not state would address.
func (k *Kernel) history() (Operand, bool) {
	for _, b := range k.Bufs {
		switch o := b.Operand; {
		case o.Kind != OperandTable:
		case o.Table == TableRows, o.Table == TableWindows, o.Table == TablePages, o.Table == TableKeys, o.Table == TableValues:
			return o, true
		}
	}
	return Operand{}, false
}

// liveRows is the node of the call's rows: the word of the call's table, at
// least one and at most capacity live.
func (k *Kernel) liveRows(live int) int {
	b := -1
	for i, bf := range k.Bufs {
		if bf.Kind == BufIn && bf.Operand == tableCall {
			b = i
			break
		}
	}
	if b < 0 {
		b = k.buf(Buf{Kind: BufIn, Elem: ir.U32, Operand: tableCall, Extent: 1})
	}
	U := ir.U32
	return k.bin(OpMin, U, k.bin(OpMax, U, k.load(b, k.u32(0)), k.u32(1)), k.u32(live))
}

// liveElements is the node of the elements of a space of capacity live the
// call's rows reach: ceil(rows * space[0] / live) coordinates of its
// leading axis, each the product of the others. The leading axis's whole
// rows, rows * (space[0] / live), come apart from its remainder's share, so
// no product passes 32 bits where the space's does not: a flattened leading
// axis of 4096 rows of 1152 times 4096 rows does.
func (k *Kernel) liveElements(space []int, live int) int {
	U := ir.U32
	rows := k.liveRows(live)
	units := k.bin(OpMul, U, rows, k.u32(space[0]/live))
	if r := space[0] % live; r != 0 {
		n := k.bin(OpAdd, U, k.bin(OpMul, U, rows, k.u32(r)), k.u32(live-1))
		units = k.bin(OpAdd, U, units, k.bin(OpDiv, U, n, k.u32(live)))
	}
	inner := 1
	for _, x := range space[1:] {
		inner *= x
	}
	return k.bin(OpMul, U, units, k.u32(inner))
}

// LiveUnitsOf is the coordinates of the code's leading axis a call of rows
// rows reaches: on a GPU the units its launch runs (LiveGroups), on a CPU
// the leading axis of the calls the caller runs (RowCalls); the whole axis
// where the code holds no rows of a step.
func (c *Code) LiveUnitsOf(rows int) int {
	if c.Live == 0 || rows >= c.Live {
		return c.LiveUnits
	}
	return (max(rows, 1)*c.LiveUnits + c.Live - 1) / c.Live
}

// LiveGroups is the workgroups a launch of the code runs at a call of rows
// rows: its capacity's where it holds no rows of a step.
func (c *Code) LiveGroups(rows int) int {
	if c.Live == 0 || rows >= c.Live {
		return c.Groups
	}
	units := c.LiveUnitsOf(rows)
	if c.Work {
		return units * c.LiveInner
	}
	g := max(c.LiveGroup, 1)
	return (units*c.LiveInner + g - 1) / g
}

// RunsAt reports whether a call of rows rows launches the code: a form for
// few rows (Few) only at a call of at most Few, its tiled form (Many) only at
// a call of more; a call of 0 rows runs the form's capacity.
func (c *Code) RunsAt(rows int) bool {
	switch {
	case c.Few > 0:
		return rows > 0 && rows <= c.Few
	case c.Many > 0:
		return rows <= 0 || rows > c.Many
	}
	return true
}
