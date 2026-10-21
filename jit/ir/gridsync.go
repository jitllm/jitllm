package ir

import "fmt"

const GridSyncWords = 2

func (b *Builder) GridBarrier(sync Value) {
	b.emit(Op{Kind: OpGridBarrier, Args: [3]Value{sync}})
}

func (k *Kernel) GridSync() bool {
	for _, o := range k.Ops {
		if o.Kind == OpGridBarrier || o.Kind == OpDepExit {
			return true
		}
	}
	return false
}

func (k *Kernel) GridSyncParam() int {
	for _, o := range k.Ops {
		if o.Kind == OpGridBarrier || o.Kind == OpDepExit {
			return k.paramOf(o.Args[0])
		}
	}
	return -1
}

func (k *Kernel) validateGrid() error {
	sync := -1
	var loops []bool
	for i, o := range k.Ops {
		switch o.Kind {
		case OpLoop:
			loops = append(loops, o.Imm < 0)
		case OpEndLoop:
			if len(loops) > 0 {
				loops = loops[:len(loops)-1]
			}
		case OpGridBarrier:
			p := k.paramOf(o.Args[0])
			if p < 0 {
				return fmt.Errorf("ir: op %d: a grid barrier's sync buffer must be a parameter", i)
			}
			if k.Params[p].Elem != U32 {
				return fmt.Errorf("ir: op %d: grid barrier sync buffer %q is %s, not u32", i, k.Params[p].Name, k.Params[p].Elem)
			}
			if sync >= 0 && sync != p {
				return fmt.Errorf("ir: op %d: a second grid sync buffer (param %d after %d); one launch has one", i, p, sync)
			}
			sync = p
			for _, rt := range loops {
				if rt {
					return fmt.Errorf("ir: op %d: a grid barrier inside a loop with a runtime count; "+
						"every workgroup must reach it equally often", i)
				}
			}
		}
	}
	if sync < 0 {
		return nil
	}
	for i, o := range k.Ops {
		switch o.Kind {
		case OpGridBarrier, OpParam:
			continue
		}
		for _, a := range o.Args {
			if a != 0 && k.paramOf(a) == sync {
				return fmt.Errorf("ir: op %d (%s) touches the grid sync buffer %q, which the barrier owns",
					i, o.Kind, k.Params[sync].Name)
			}
		}
		for _, a := range o.Vargs {
			if k.paramOf(a) == sync {
				return fmt.Errorf("ir: op %d (%s) touches the grid sync buffer", i, o.Kind)
			}
		}
	}
	return nil
}
