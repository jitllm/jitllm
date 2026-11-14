package composer

import (
	"fmt"

	"github.com/jitllm/jitllm/composer/ir"
)

// Table is one of the step's tables a primitive reads or writes, what the
// call site binds to it: a step's rows, its pages, a rotary's angles. A
// table of several instances (a model's rotaries, the packed codes of each
// width) tells them apart by Operand.Instance.
type Table uint8

const (
	TableNone Table = iota
	// TableRows is the step's row descriptors, TableWindows their windows,
	// TableCall the rows a call takes, TablePages the history's pages.
	TableRows
	TableWindows
	TableCall
	TablePages
	// TableKeys and TableValues are a block's history pools.
	TableKeys
	TableValues
	// TableQOff is each row's query offset.
	TableQOff
	// TableAngles is a rotary's angles (Instance the model's rotary),
	// TablePlanes its planes, TablePos the rows' positions and TableAxisPos
	// theirs on each axis, TableAxes a pair's axis, TableRopeConst the
	// table's constants.
	TableAngles
	TablePlanes
	TablePos
	TableAxisPos
	TableAxes
	TableRopeConst
	// TableTokens is the step's tokens and TableRequest its request.
	TableTokens
	TableRequest
	// TableExperts is the experts each row routes to.
	TableExperts
	// TableRunIn and TableRunOut are a call's runs of rows.
	TableRunIn
	TableRunOut
	// A recurrent block's state and short convolution's tail, read and
	// written, and its rows' slots.
	TableRecState
	TableRecStateOut
	TableRecConv
	TableRecConvOut
	TableRecRows
	// TableCodes is a packed weight's codes, Instance the word they unpack.
	TableCodes
	// TableRopeRows is the step's rows as the rotary positions read them
	// (each row's cache position and its sequence's marks), TableRopeMarks
	// the sequences' rotary marks (graph.OpRopePos).
	TableRopeRows
	TableRopeMarks
	// TablePicture is the step's picture, its decoded samples as words
	// (graph.OpPicture).
	TablePicture
	// TablePicAcc is a Pillow resize's vertical sums, accumulated over its
	// bands (graph.Attrs.PicPass), and TablePicAccOut the same buffer
	// written.
	TablePicAcc
	TablePicAccOut
	// TablePicHAcc is a Pillow resize's horizontal sums over a band of source
	// rows, accumulated over its bands of source columns (graph.PicCols), and
	// TablePicHAccOut the same buffer written.
	TablePicHAcc
	TablePicHAccOut
	// TableRecKeep and TableRecConvKeep are a recurrent block's state and
	// short convolution's tail after each row of a step that keeps them
	// (graph.Attrs.Keeps), a slot a row's entry of the run order.
	TableRecKeep
	TableRecConvKeep
	// TableSelIn and TableSelOut are a selection's running block scores over
	// a history read in passes: each block's best in the passes before, and
	// with this pass's.
	TableSelIn
	TableSelOut
	numTables
)

func (t Table) String() string {
	switch t {
	case TableNone:
		return ""
	case TableRows:
		return "rows"
	case TableWindows:
		return "rows.win"
	case TableCall:
		return "rows.n"
	case TablePages:
		return "pages"
	case TableKeys:
		return "kv.k"
	case TableValues:
		return "kv.v"
	case TableQOff:
		return "q.off"
	case TableAngles:
		return "rope.cs"
	case TablePlanes:
		return "rope.planes"
	case TablePos:
		return "rope.pos"
	case TableAxisPos:
		return "rope.pos.axes"
	case TableAxes:
		return "rope.axes"
	case TableRopeConst:
		return "rope.const"
	case TableTokens:
		return "tokens"
	case TableRequest:
		return "request"
	case TableExperts:
		return "experts"
	case TableRunIn:
		return "run.in"
	case TableRunOut:
		return "run.out"
	case TableRecState:
		return "rec.state"
	case TableRecStateOut:
		return "rec.state.out"
	case TableRecConv:
		return "rec.conv"
	case TableRecConvOut:
		return "rec.conv.out"
	case TableRecRows:
		return "rec.rows"
	case TableCodes:
		return "codes"
	case TableRopeRows:
		return "rope.rows"
	case TableRopeMarks:
		return "rope.marks"
	case TablePicture:
		return "picture"
	case TablePicAcc:
		return "picture.acc"
	case TablePicAccOut:
		return "picture.acc.out"
	case TablePicHAcc:
		return "picture.hacc"
	case TablePicHAccOut:
		return "picture.hacc.out"
	case TableRecKeep:
		return "rec.state.keep"
	case TableRecConvKeep:
		return "rec.conv.keep"
	case TableSelIn:
		return "sel.in"
	case TableSelOut:
		return "sel.out"
	}
	return fmt.Sprintf("table%d", t)
}

// recurrent says t is a recurrent block's own table.
func (t Table) recurrent() bool {
	return t >= TableRecState && t <= TableRecRows || t == TableRecKeep || t == TableRecConvKeep
}

// storage is the operand naming the buffer o is bound to where a call site
// binds two operands to one: a recurrent block's written state is its read
// state in the shared pool (one half both), so a pass moving a read of one
// past a write of the other moves it past a write of its own buffer.
func (o Operand) storage() Operand {
	if o.Kind == OperandTable && o.Table == TableRecStateOut {
		o.Table, o.Name = TableRecState, tableLabel(TableRecState, o.Instance)
	}
	if o.Kind == OperandTable && o.Table == TablePicAccOut {
		o.Table, o.Name = TablePicAcc, tableLabel(TablePicAcc, o.Instance)
	}
	if o.Kind == OperandTable && o.Table == TablePicHAccOut {
		o.Table, o.Name = TablePicHAcc, tableLabel(TablePicHAcc, o.Instance)
	}
	return o
}

// TableKey is a table as the call site binds it.
type TableKey struct {
	Table    Table
	Instance int
}

// Key is the table o is, as the call site binds it.
func (o Operand) Key() TableKey { return TableKey{o.Table, o.Instance} }

// of is table t's instance inst as an operand; its name is the table's label,
// never what it is matched by.
func (t Table) of(inst int) Operand {
	return Operand{Kind: OperandTable, Table: t, Instance: inst, Name: tableLabel(t, inst), Elem: t.elem(), PerPart: t.perPart()}
}

// op is table t's instance 0.
func (t Table) op() Operand { return t.of(0) }

// elem is the type of table t's elements, Void where it is a part's own
// state or a float table the kernel types itself.
func (t Table) elem() ir.Type {
	switch {
	case t.perPart():
		return ir.Void
	case t == TableAngles, t == TablePlanes, t == TableRunIn, t == TableRunOut, t == TableSelIn, t == TableSelOut:
		return ir.Void
	}
	return ir.U32
}

// perPart says t is a part's own state (Operand.PerPart).
func (t Table) perPart() bool {
	switch t {
	case TableKeys, TableValues, TableRecState, TableRecStateOut, TableRecConv, TableRecConvOut, TableRecKeep, TableRecConvKeep:
		return true
	}
	return false
}

// tableLabel is table t's instance inst as it is named for a person.
func tableLabel(t Table, inst int) string {
	switch {
	case t == TableCodes:
		return fmt.Sprintf("codes.%x", inst)
	case inst > 0:
		return fmt.Sprintf("%s.%d", t, inst)
	}
	return t.String()
}
