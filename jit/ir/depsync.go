package ir

import "fmt"

const DepTile = int64(1) << 32

func (b *Builder) DepTask(id int) { b.emit(Op{Kind: OpDepTask, Type: Void, Imm: int64(id)}) }

func (b *Builder) DepEpoch(buf Value, word int) Value {
	return b.emit(Op{Kind: OpDepEpoch, Type: U32, Args: [3]Value{buf}, Imm: int64(word)})
}

func (b *Builder) DepWait(buf, word, target Value, producer int, tile bool) {
	imm := int64(producer)
	if tile {
		imm |= DepTile
	}
	b.emit(Op{Kind: OpDepWait, Type: Void, Args: [3]Value{buf, word, target}, Imm: imm})
}

func (b *Builder) DepSignal(buf, word Value, task int) {
	b.emit(Op{Kind: OpDepSignal, Type: Void, Args: [3]Value{buf, word}, Imm: int64(task)})
}

func (b *Builder) DepExit(buf Value, base int) {
	b.emit(Op{Kind: OpDepExit, Type: Void, Args: [3]Value{buf}, Imm: int64(base)})
}

func isDep(k Kind) bool {
	switch k {
	case OpDepTask, OpDepEpoch, OpDepWait, OpDepSignal, OpDepExit:
		return true
	}
	return false
}

func (k *Kernel) validateDeps() error {
	buf, exits, last := -1, 0, -1
	for i, o := range k.Ops {
		if !isDep(o.Kind) {
			continue
		}
		last = i
		if o.Kind == OpDepTask {
			continue
		}
		p := k.paramOf(o.Args[0])
		if p < 0 {
			return fmt.Errorf("ir: op %d (%s): the dep buffer must be a parameter", i, o.Kind)
		}
		if k.Params[p].Elem != U32 {
			return fmt.Errorf("ir: op %d: dep buffer %q is %s, not u32", i, k.Params[p].Name, k.Params[p].Elem)
		}
		if buf >= 0 && buf != p {
			return fmt.Errorf("ir: op %d: a second dep buffer (param %d after %d)", i, p, buf)
		}
		buf = p
		if o.Kind == OpDepExit {
			exits++
		}
	}
	if last < 0 {
		return nil
	}
	if k.GridSyncParamBarrier() >= 0 {
		return fmt.Errorf("ir: a kernel with dependency counters has a grid barrier too")
	}
	if buf < 0 || exits != 1 || k.Ops[len(k.Ops)-1].Kind != OpDepExit {
		return fmt.Errorf("ir: dependency counters need exactly one dep.exit, as the kernel's last op")
	}
	for i, o := range k.Ops {
		if isDep(o.Kind) || o.Kind == OpParam {
			continue
		}
		for _, a := range o.Args {
			if a != 0 && k.paramOf(a) == buf {
				return fmt.Errorf("ir: op %d (%s) touches the dep buffer %q, which the counters own",
					i, o.Kind, k.Params[buf].Name)
			}
		}
		for _, a := range o.Vargs {
			if k.paramOf(a) == buf {
				return fmt.Errorf("ir: op %d (%s) touches the dep buffer", i, o.Kind)
			}
		}
	}

	type state struct {
		read, wrote [maxParams]bool
	}
	tasks := map[int]*state{}
	covered := map[int]map[int]bool{}
	var order []int
	cur := -1
	seen := map[int]bool{}
	depth := 0
	for i, o := range k.Ops {
		switch o.Kind {
		case OpLoop:
			depth++
		case OpEndLoop:
			depth--
		case OpDepTask:
			if depth != 0 {
				return fmt.Errorf("ir: op %d: dep.task inside a loop", i)
			}
			id := int(o.Imm)
			if id <= cur {
				return fmt.Errorf("ir: op %d: task %d after task %d", i, id, cur)
			}
			cur = id
			tasks[id] = &state{}
			covered[id] = map[int]bool{}
			order = append(order, id)
			seen = map[int]bool{}
		case OpDepWait:
			if cur < 0 {
				return fmt.Errorf("ir: op %d: dep.wait outside a task", i)
			}
			pr := int(o.Imm &^ DepTile)
			if pr >= cur || tasks[pr] == nil {
				return fmt.Errorf("ir: op %d: task %d waits on task %d, not an earlier one", i, cur, pr)
			}
			seen[pr] = true
			if o.Imm&DepTile == 0 {
				covered[cur][pr] = true
				for c := range covered[pr] {
					covered[cur][c] = true
					seen[c] = true
				}
			}
		case OpDepSignal:
			if int(o.Imm) != cur {
				return fmt.Errorf("ir: op %d: task %d signals for task %d", i, cur, o.Imm)
			}
		case OpDepEpoch:
			if depth != 0 {
				return fmt.Errorf("ir: op %d: dep.epoch inside a loop", i)
			}
		case OpDepExit:
			if depth != 0 {
				return fmt.Errorf("ir: op %d: dep.exit inside a loop", i)
			}
		case OpLoad, OpLoadV, OpStore, OpStoreV:
			p := k.paramOf(o.Args[0])
			if p < 0 {
				continue
			}
			if cur < 0 {
				return fmt.Errorf("ir: op %d: a global access before the first dep.task", i)
			}
			st := o.Kind == OpStore || o.Kind == OpStoreV
			for _, t := range order[:len(order)-1] {
				e := tasks[t]
				if (e.wrote[p] || (st && e.read[p])) && !seen[t] {
					return fmt.Errorf("ir: op %d: task %d %s param %d, which task %d %s, without waiting on it",
						i, cur, map[bool]string{true: "writes", false: "reads"}[st], p, t,
						map[bool]string{true: "wrote", false: "read"}[e.wrote[p]])
				}
			}
			if st {
				tasks[cur].wrote[p] = true
			} else {
				tasks[cur].read[p] = true
			}
		}
	}
	return nil
}

func (k *Kernel) GridSyncParamBarrier() int {
	for _, o := range k.Ops {
		if o.Kind == OpGridBarrier {
			return k.paramOf(o.Args[0])
		}
	}
	return -1
}
