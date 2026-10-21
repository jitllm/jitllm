package jit

import "github.com/jitllm/jitllm/jit/ir"

// A row's history descriptor, the words the runtime writes for each row of
// a step.
const (
	rowTab   = iota // where the row's page table starts
	rowStart        // the first position it attends
	rowEnd          // one past the last
	rowWrite        // where its new key and value go
	rowWords
)

// The step's tables.
var (
	tableAngles = Operand{Kind: OperandTable, Name: "rope.cs"}
	tableQOff   = Operand{Kind: OperandTable, Name: "q.off", Elem: ir.U32}
	tableRows   = Operand{Kind: OperandTable, Name: "rows", Elem: ir.U32}
	tablePages  = Operand{Kind: OperandTable, Name: "pages", Elem: ir.U32}
	tableKeys   = Operand{Kind: OperandTable, Name: "kv.k"}
	tableValues = Operand{Kind: OperandTable, Name: "kv.v"}
)

// page is row r's page and place for its new key and value: its table's
// page holding the write position, and the position within it.
func (x *exprs) page(r int, page int) (pid, off int) {
	U := ir.U32
	at := func(w int) int {
		return x.add(Expr{Op: ExprLoad, In: 1, A: x.op(OpAdd, U, x.op(OpMul, U, r, x.u(rowWords)), x.u(uint32(w)))})
	}
	tab, pos := at(rowTab), at(rowWrite)
	pid = x.add(Expr{Op: ExprLoad, In: 2, A: x.op(OpAdd, U, tab, x.op(OpDiv, U, pos, x.u(uint32(page))))})
	return pid, x.op(OpRem, U, pos, x.u(uint32(page)))
}

// rope is a rotary over w-wide rows of heads of hd, the first nrot of each
// turned in pairs by the step's angles: an element a pair, both turned
// values stored, at the q offset where qoff.
func rope(in, out Operand, rows, w, hd, nrot int, neox, qoff bool) Prim {
	U, F := ir.U32, ir.F32
	pairs, heads := nrot/2, w/hd
	var x exprs
	e := x.add(Expr{Op: ExprElem})
	p := x.op(OpRem, U, e, x.u(uint32(pairs)))
	h := x.op(OpRem, U, x.op(OpDiv, U, e, x.u(uint32(pairs))), x.u(uint32(heads)))
	r := x.op(OpDiv, U, e, x.u(uint32(pairs*heads)))
	first, step := x.op(OpMul, U, p, x.u(2)), 1
	if neox {
		first, step = p, pairs
	}
	at := x.op(OpAdd, U, x.op(OpAdd, U, x.op(OpMul, U, r, x.u(uint32(w))), x.op(OpMul, U, h, x.u(uint32(hd)))), first)
	at2 := x.op(OpAdd, U, at, x.u(uint32(step)))
	x0, x1 := x.add(Expr{Op: ExprLoad, In: 0, A: at}), x.add(Expr{Op: ExprLoad, In: 0, A: at2})
	cs := x.op(OpAdd, U, x.op(OpMul, U, r, x.u(uint32(nrot))), x.op(OpMul, U, p, x.u(2)))
	c, s := x.add(Expr{Op: ExprLoad, In: 1, A: cs}), x.add(Expr{Op: ExprLoad, In: 1, A: x.op(OpAdd, U, cs, x.u(1))})
	y0 := x.op(OpSub, F, x.op(OpMul, F, x0, c), x.op(OpMul, F, x1, s))
	y1 := x.op(OpAdd, F, x.op(OpMul, F, x0, s), x.op(OpMul, F, x1, c))
	ins := []Operand{in, tableAngles}
	o0, o1 := at, at2
	if qoff {
		ins = append(ins, tableQOff)
		base := x.add(Expr{Op: ExprLoad, In: 2, A: x.u(0)})
		o0, o1 = x.op(OpAdd, U, base, at), x.op(OpAdd, U, base, at2)
	}
	return Prim{Kind: PrimMap, N: rows * heads * pairs, In: ins, Expr: x.e,
		Stores: []PrimStore{{Out: out, Index: o0, Value: y0}, {Out: out, Index: o1, Value: y1}}}
}

// appendKey is a row's key written into its page of the keys' pool, a page
// [width][page]: an element a key element.
func appendKey(k Operand, rows, w, page int) Prim {
	U := ir.U32
	var x exprs
	e := x.add(Expr{Op: ExprElem})
	d, r := x.op(OpRem, U, e, x.u(uint32(w))), x.op(OpDiv, U, e, x.u(uint32(w)))
	pid, off := x.page(r, page)
	at := x.op(OpAdd, U, x.op(OpMul, U, pid, x.u(uint32(page*w))), x.op(OpAdd, U, x.op(OpMul, U, d, x.u(uint32(page))), off))
	v := x.add(Expr{Op: ExprLoad, In: 0, A: e})
	return Prim{Kind: PrimMap, N: rows * w, In: []Operand{k, tableRows, tablePages}, Expr: x.e,
		Stores: []PrimStore{{Out: tableKeys, Index: at, Value: v}}}
}

// appendValue is a row's value written into its page of the values' pool, a
// page [page][width].
func appendValue(v Operand, rows, w, page int) Prim {
	U := ir.U32
	var x exprs
	e := x.add(Expr{Op: ExprElem})
	d, r := x.op(OpRem, U, e, x.u(uint32(w))), x.op(OpDiv, U, e, x.u(uint32(w)))
	pid, off := x.page(r, page)
	at := x.op(OpAdd, U, x.op(OpMul, U, x.op(OpAdd, U, x.op(OpMul, U, pid, x.u(uint32(page))), off), x.u(uint32(w))), d)
	val := x.add(Expr{Op: ExprLoad, In: 0, A: e})
	return Prim{Kind: PrimMap, N: rows * w, In: []Operand{v, tableRows, tablePages}, Expr: x.e,
		Stores: []PrimStore{{Out: tableValues, Index: at, Value: val}}}
}
