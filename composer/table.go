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
	// The entry history's pool, descriptors and pages (graph.StateEnt).
	TableEntKeys
	TableEntRows
	TableEntPages
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
	case TableEntKeys:
		return "ent.k"
	case TableEntRows:
		return "ent.rows"
	case TableEntPages:
		return "ent.pages"
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
	}
	return fmt.Sprintf("table%d", t)
}

// recurrent says t is a recurrent block's own table.
func (t Table) recurrent() bool { return t >= TableRecState && t <= TableRecRows }

// entry says t is the entry history's.
func (t Table) entry() bool { return t >= TableEntKeys && t <= TableEntPages }

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

// elem is the type of table t's elements, Void where it is a pool or a
// float table the kernel types itself.
func (t Table) elem() ir.Type {
	switch t {
	case TableAngles, TablePlanes, TableRunIn, TableRunOut, TableKeys, TableValues, TableEntKeys:
		return ir.Void
	}
	return ir.U32
}

// perPart says t is a part's own state (Operand.PerPart).
func (t Table) perPart() bool {
	switch t {
	case TableKeys, TableValues, TableEntKeys, TableRecState, TableRecStateOut, TableRecConv, TableRecConvOut:
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
