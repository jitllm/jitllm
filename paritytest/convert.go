package paritytest

import (
	oldir "github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/jit/ir"
)

// Old is a kernel of the composer's IR as main's IR, the comparator's: the
// composer's IR is a copy, so every field carries over as it is.
func Old(k *ir.Kernel) *oldir.Kernel {
	o := &oldir.Kernel{Name: k.Name, Group: k.Group, Lanes: k.Lanes, Comment: k.Comment}
	for _, p := range k.Params {
		o.Params = append(o.Params, oldir.Param{Name: p.Name, Elem: oldir.Type(p.Elem), Tile: oldir.TileElem(p.Tile), IsTile: p.IsTile})
	}
	for _, op := range k.Ops {
		n := oldir.Op{Kind: oldir.Kind(op.Kind), Type: oldir.Type(op.Type), Imm: op.Imm,
			Args: [3]oldir.Value{oldir.Value(op.Args[0]), oldir.Value(op.Args[1]), oldir.Value(op.Args[2])}}
		for _, v := range op.Vargs {
			n.Vargs = append(n.Vargs, oldir.Value(v))
		}
		if op.Tile != nil {
			n.Tile = &oldir.TileType{Rows: op.Tile.Rows, Cols: op.Tile.Cols, Elem: oldir.TileElem(op.Tile.Elem), Use: oldir.TileUse(op.Tile.Use)}
		}
		o.Ops = append(o.Ops, n)
	}
	return o
}
