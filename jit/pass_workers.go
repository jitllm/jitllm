package jit

import "github.com/jitllm/jitllm/jit/ir"

// Workers deals the rows of a kernel over a weight's rows on a device
// whose code runs a call at a time on each of its workers (Device.Slots),
// the kernels whose work is the weight's bytes: the last axis of
// the element space is cut into calls, a new leading axis of the calls'
// space, each call a run of consecutive rows. The cut divides the rows
// exactly, each call a whole number of Rows rows, and is the one that
// leaves the busiest worker the fewest rows, the fewest calls among equals.
type Workers struct {
	Rows int
}

func (Workers) Name() string { return "workers" }

func (p Workers) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	if d.Slots <= 1 {
		return ks, nil
	}
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Work || k.Group > 0 || k.Heads != nil || k.Calls > 0 || len(k.Space) == 0 || len(k.Loops) == 0 || !k.readsWeight() {
			return nil, false, nil
		}
		a := len(k.Space) - 1
		parts := workerParts(k.Space[a], d.Slots, max(p.Rows, 1))
		if parts <= 1 {
			return nil, false, nil
		}
		m := k.clone()
		chunk := m.Space[a] / parts
		var at []int
		for i := range m.Nodes {
			if nd := &m.Nodes[i]; nd.Op == OpCoord {
				if nd.A == a {
					at = append(at, i)
				}
				nd.A++
			}
		}
		part := m.add(Node{Op: OpCoord, A: 0, Type: ir.U32})
		base := m.bin(OpMul, ir.U32, part, m.u32(chunk))
		for _, i := range at {
			inner := m.add(Node{Op: OpCoord, A: a + 1, Type: ir.U32})
			m.Nodes[i] = Node{Op: OpAdd, Type: ir.U32, A: inner, B: base}
		}
		if pos := m.Positions; pos != nil && pos.Axis >= 0 {
			q := *pos
			q.Axis++
			m.Positions = &q
		}
		m.Space = append([]int{parts}, m.Space...)
		m.Space[a+1] = chunk
		m.Calls = 1
		return []*Kernel{m}, true, nil
	})
}

// workerParts is the calls rows rows are cut into over workers: a divisor
// of rows leaving each call a multiple of unit rows, the one whose busiest
// worker holds the fewest rows, the fewest calls among equals; 1 where none
// does better than one call.
func workerParts(rows, workers, unit int) int {
	best, cost := 1, rows
	for parts := 2; parts*unit <= rows; parts++ {
		if rows%parts != 0 || (rows/parts)%unit != 0 {
			continue
		}
		if c := (parts + workers - 1) / workers * (rows / parts); c < cost {
			best, cost = parts, c
		}
	}
	return best
}

// readsWeight reports whether the kernel reads a matrix weight.
func (k *Kernel) readsWeight() bool {
	for _, b := range k.Bufs {
		if b.Operand.Kind == OperandWeight {
			return true
		}
	}
	return false
}
