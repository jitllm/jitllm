package jit

import "github.com/jitllm/jitllm/jit/ir"

// Workers deals the rows of a kernel over a weight's rows on a device
// whose code runs a call at a time on each of its workers (Device.Slots),
// the kernels whose work is the weight's bytes: the last axis of
// the element space is cut into calls, a new leading axis of the calls'
// space, each call a run of consecutive rows. The cut divides the rows
// exactly, each call a whole number of Rows rows, and is the one that
// leaves the busiest worker the fewest rows, the fewest calls among equals.
//
// A kernel held to a call's rows (RowCalls) is cut into tiles of at most
// Tile of the weight's rows instead, a calls' axis after the step's rows,
// whatever the workers: the caller runs a tile's calls for every row of the
// step before the next tile's (Code.TileMajor), so a tile of the weight is
// read from memory once for all the step's rows.
type Workers struct {
	Rows, Tile int
}

func (Workers) Name() string { return "workers" }

func (p Workers) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		rows := k.RowsHeld && k.Calls == 1 && k.Heads == nil && len(k.Space) >= 2
		if k.Work || k.Group > 0 || k.Heads != nil || k.Calls > 0 && !rows || len(k.Space) == 0 || len(k.Loops) == 0 || !k.readsWeight() && k.Kind != PrimScan {
			return nil, false, nil
		}
		a := len(k.Space) - 1
		var parts int
		switch {
		case rows:
			parts = tileParts(k.Space[a], max(p.Rows, 1), p.Tile)
		case k.Kind == PrimScan:
			// A scan's state row a call: its rows' steps are the call's
			// loops, with no loop over state rows around them.
			parts = k.Space[a]
		case d.Slots > 1:
			parts = workerParts(k.Space[a], d.Slots, max(p.Rows, 1))
		}
		chunk, share := 0, 0
		if d.Slots > 1 && !rows && k.Kind != PrimScan {
			if c, n := shareParts(k.Space[a], d.Slots, max(p.Rows, 1)); n > 1 && c <= workerCost(k.Space[a], parts, d.Slots) {
				chunk, parts = c, n
				if k.Space[a]%c != 0 {
					share = k.Space[a]
				}
			}
		}
		if parts <= 1 {
			return nil, false, nil
		}
		at := k.Calls
		m := k.clone()
		if chunk == 0 {
			chunk = m.Space[a] / parts
		}
		m.Share = share
		var cut []int
		for i := range m.Nodes {
			if nd := &m.Nodes[i]; nd.Op == OpCoord && nd.A >= at {
				if nd.A == a {
					cut = append(cut, i)
				}
				nd.A++
			}
		}
		part := m.add(Node{Op: OpCoord, A: at, Type: ir.U32})
		base := m.bin(OpMul, ir.U32, part, m.u32(chunk))
		for _, i := range cut {
			inner := m.add(Node{Op: OpCoord, A: a + 1, Type: ir.U32})
			m.Nodes[i] = Node{Op: OpAdd, Type: ir.U32, A: inner, B: base}
		}
		if pos := m.Positions; pos != nil && pos.Axis >= at {
			q := *pos
			q.Axis++
			m.Positions = &q
		}
		m.Space = append(append(append([]int(nil), m.Space[:at]...), parts), m.Space[at:]...)
		m.Space[a+1] = chunk
		m.Calls = at + 1
		m.TileMajor = rows
		return []*Kernel{m}, true, nil
	})
}

// tileParts is the tiles rows rows are cut into for a kernel held to a
// call's rows: the fewest dividing rows that leave each a multiple of unit
// rows and at most tile; 1 where tile is unset or no cut does.
func tileParts(rows, unit, tile int) int {
	if tile <= 0 {
		return 1
	}
	for parts := 2; parts*unit <= rows; parts++ {
		if rows%parts == 0 && (rows/parts)%unit == 0 && rows/parts <= tile {
			return parts
		}
	}
	return 1
}

// workerParts is the calls rows rows are cut into over workers: a divisor
// of rows leaving each call a multiple of unit rows, the one whose busiest
// worker holds the fewest rows; among equals the most, up to callsAWorker a
// worker, so the workers the pool finds free take up the calls of one the
// OS holds off its core. 1 where none does better than one call.
func workerParts(rows, workers, unit int) int {
	best, cost := 1, rows
	for parts := 2; parts*unit <= rows; parts++ {
		if rows%parts != 0 || (rows/parts)%unit != 0 {
			continue
		}
		c := (parts + workers - 1) / workers * (rows / parts)
		if c < cost || c == cost && parts <= callsAWorker*workers {
			best, cost = parts, c
		}
	}
	return best
}

// workerCost is the rows the busiest worker holds with rows cut into parts
// equal calls.
func workerCost(rows, parts, workers int) int {
	if parts <= 1 {
		return rows
	}
	return (parts + workers - 1) / workers * (rows / parts)
}

// shareParts is a worker's whole share of rows as one call: the call's rows,
// a multiple of unit, and the calls, the last holding what is left.
func shareParts(rows, workers, unit int) (int, int) {
	c := ((rows+workers-1)/workers + unit - 1) / unit * unit
	if rows%unit != 0 || c >= rows {
		return 0, 1
	}
	return c, (rows + c - 1) / c
}

// callsAWorker is the most calls a worker's share of a weight is cut into
// where the cut costs the busiest worker no rows.
const callsAWorker = 4

// readsWeight reports whether the kernel reads a matrix weight.
func (k *Kernel) readsWeight() bool {
	for _, b := range k.Bufs {
		if b.Operand.Kind == OperandWeight {
			return true
		}
	}
	return false
}
