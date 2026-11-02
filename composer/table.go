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

var tableNames = [numTables]string{"", "rows", "rows.win", "rows.n", "pages", "kv.k", "kv.v", "ent.k", "ent.rows", "ent.pages", "q.off",
	"rope.cs", "rope.planes", "rope.pos", "rope.pos.axes", "rope.axes", "rope.const", "tokens", "request", "experts", "run.in", "run.out",
	"rec.state", "rec.state.out", "rec.conv", "rec.conv.out", "rec.rows", "codes", "rope.rows", "rope.marks"}

func (t Table) String() string {
	if t < numTables {
		return tableNames[t]
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

// tableOp is table t's instance inst as an operand of elements elem; its name
// is the table's label, never what it is matched by.
func tableOp(t Table, inst int, elem ir.Type) Operand {
	return Operand{Kind: OperandTable, Table: t, Instance: inst, Name: tableLabel(t, inst), Elem: elem}
}

// partTable is table t, a part's own state (Operand.PerPart).
func partTable(t Table) Operand {
	o := tableOp(t, 0, ir.Void)
	o.PerPart = true
	return o
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
