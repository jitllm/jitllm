package tier

import "fmt"

// Values coloured by liveness (docs/design/composer.md, "Slots assigned
// after rewriting"): a value the builder gives no slot of its own
// (slotAuto) takes one of slotVal0.. once the program's rewrites are done,
// so a new composition of primitives needs no new scratch name. Two values
// share a slot only where their lives do not meet: a value lives from the op
// that writes it to the last op that reads it, so an op's input and output
// never share one (RULE 13).

// colourVals assigns every slotAuto value of p a value slot, the lowest free
// one at its first write, and returns each slot's need: the widest value it
// holds, in elements a row.
func colourVals(p *devProgram) ([numVals]int, error) {
	var need [numVals]int
	first := make([]int, len(p.vals))
	last := make([]int, len(p.vals))
	for v := range first {
		first[v], last[v] = -1, -1
	}
	for i, o := range p.ops {
		for _, v := range o.Out {
			if first[v] < 0 {
				first[v] = i
			}
			last[v] = max(last[v], i)
		}
		for _, v := range o.In {
			last[v] = max(last[v], i)
		}
	}
	var busyTo [numVals]int // the last op of the value a slot holds
	for v := range busyTo {
		busyTo[v] = -1
	}
	for i := range p.ops {
		for _, v := range p.ops[i].Out {
			if p.slots[v] != slotAuto || first[v] != i {
				continue
			}
			c := -1
			for s := range busyTo {
				if busyTo[s] < i {
					c = s
					break
				}
			}
			if c < 0 {
				return need, fmt.Errorf("device program: more than %d values live at op %d (%s)", numVals, i, p.ops[i].Kind)
			}
			p.slots[v] = slotVal0 + devSlot(c)
			busyTo[c] = last[v]
			need[c] = max(need[c], p.stride(v))
		}
	}
	for v, s := range p.slots {
		// A value no op reads or writes any more (a rewrite folded it away)
		// has nothing to colour.
		if s == slotAuto && (first[v] >= 0 || last[v] >= 0) {
			return need, fmt.Errorf("device program: value %s is read and never written", p.names[v])
		}
	}
	// RULE 13 over the coloured slots: no op reads a buffer it writes.
	for i, o := range p.ops {
		for _, w := range o.Out {
			for _, r := range o.In {
				if s := p.slots[w]; s >= slotVal0 && s < slotVal0+numVals && p.slots[r] == s {
					return need, fmt.Errorf("device program: op %d (%s) reads and writes %s", i, o.Kind, s)
				}
			}
		}
	}
	return need, nil
}

// operandNeed is the bytes of int8 values and of their scales the widest
// operand p quantizes takes over rows rows (bs.a's and bs.ax's layout).
func operandNeed(p *devProgram, rows int) (int, int) {
	w := 0
	for _, o := range p.ops {
		for _, v := range o.Out {
			if p.vals[v].Elem == elemQ8 {
				w = max(w, pad32(p.vals[v].Width))
			}
		}
	}
	return rows * w, 3 * rows * (w / 32) * 4
}

// valsFit reports whether the lane's value buffers already hold need over
// rows rows.
func (g *devTier) valsFit(need [numVals]int, rows int) bool {
	for c, n := range need {
		if n*max(rows, 1) > g.valCaps[c] {
			return false
		}
	}
	return true
}

// sizeVals grows the lane's value buffers to need elements a row over rows
// rows each. Callers hold g.mu, outside a submission.
func (g *devTier) sizeVals(need [numVals]int, rows int) bool {
	for c, n := range need {
		if n > 0 && !g.regrow(&g.valBufs[c], &g.valCaps[c], n*max(rows, 1), 4, true, true) {
			return false
		}
	}
	return true
}
