package gpu

import (
	"fmt"
	"slices"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// rowTable is a row table (jlm.RowTable) a block gathers from, on the
// card, by its role; a block gathers from any number of them. A row table
// is read a row at a time and never resident whole, so the card holds none
// of it: the container's rows are read through a view of the card's frames
// (jlm.Frames.NewRowView), and the rows a call's gathers name are copied
// into slots on the card, each range a gather binds the table from (its
// first row, from) holding its own. A range's slot is an id's place among
// the call's distinct ids of that range, so it stays inside the range
// however wide the call; each id is rewritten to its slot past the range's
// first row before the gathers read it (rowsRun).
type rowTable struct {
	role     jlm.Role
	view     *jlm.RowView
	typ      jlm.Type
	rowBytes int
	// col is each of the gathered row's columns' range, an index into ranges.
	col    []int
	ranges []rowRange
	ids    []uint32
}

// rowTableOf is the table of role r in ts, nil for none.
func rowTableOf(ts []*rowTable, r jlm.Role) *rowTable {
	for _, t := range ts {
		if t.role == r {
			return t
		}
	}
	return nil
}

// rowRange is the rows of the table one gather binds from its first row
// from: the call's distinct ids in it, each its slot, and the slots on the
// card.
type rowRange struct {
	from  int
	slot  map[uint32]uint32
	order []uint32
	rows  resident
	slots int
	stage []byte
}

// tableGather is a row table a block's gathers read (graph.Graph.TableRows):
// its role, the value its gathers read their rows by, and the first row each
// of its columns is gathered from.
type tableGather struct {
	role jlm.Role
	rows graph.ValueID
	cols []int
}

// rowTableGathers is every row table a block's gathers read, an error for a
// column gathered from two first rows or by none.
func rowTableGathers(gr *graph.Graph) ([]tableGather, error) {
	var gs []tableGather
	for _, t := range gr.TableRows() {
		cols, err := tableCols(gr, t)
		if err != nil {
			return nil, err
		}
		gs = append(gs, tableGather{role: t.Role, rows: t.Rows, cols: cols})
	}
	return gs, nil
}

// tableCols is the first row each column of t's gathered row is gathered
// from.
func tableCols(gr *graph.Graph, t graph.TableRow) ([]int, error) {
	rv := t.Rows
	cols := make([]int, gr.Values[rv].Width)
	for i := range cols {
		cols[i] = -1
	}
	cut := map[graph.ValueID][2]int{rv: {0, len(cols)}}
	for _, o := range gr.Ops {
		if o.Kind == graph.OpSlice && len(o.In) == 1 && o.In[0] == rv && len(o.Out) == 1 {
			cut[o.Out[0]] = [2]int{o.Attr.Offset, gr.Values[o.Out[0]].Width}
		}
	}
	for _, o := range gr.Ops {
		if o.Kind != graph.OpEmbed || len(o.Weight) != 1 || o.Weight[0].Role != t.Role {
			continue
		}
		at, ok := cut[o.In[0]]
		if len(o.In) != 1 || !ok || at[0] < 0 || at[0]+at[1] > len(cols) {
			return nil, fmt.Errorf("the row table %v gathered by rows that are not its gathered row's columns", t.Role)
		}
		for c := at[0]; c < at[0]+at[1]; c++ {
			if cols[c] >= 0 && cols[c] != o.Weight[0].From {
				return nil, fmt.Errorf("column %d of the table %v's gathered row gathered from rows %d and %d", c, t.Role, cols[c], o.Weight[0].From)
			}
			cols[c] = o.Weight[0].From
		}
	}
	for c, f := range cols {
		if f < 0 {
			return nil, fmt.Errorf("column %d of the table %v's gathered row is gathered by none", c, t.Role)
		}
	}
	return cols, nil
}

// openRowTable opens block l's row tables on the card where its graph
// gathers them, through fr: what the step's codes read is copied in a call
// at a time (rowsRun).
func (g *devTier) openRowTable(l *layer, p *nn.LayerPlan, fr *jlm.Frames) error {
	if len(l.rowTabs) > 0 || l.graph == nil || p.Spec == nil || len(p.Spec.Tables) == 0 {
		return nil
	}
	gs, err := rowTableGathers(l.graph)
	if err != nil || len(gs) == 0 {
		return err
	}
	if fr == nil {
		return fmt.Errorf("a row table offered with no frames to read its rows through")
	}
	for _, tg := range gs {
		i := slices.IndexFunc(p.Spec.Tables, func(t nn.RowTable) bool { return t.Entry.Role == tg.role })
		if i < 0 {
			continue
		}
		st := p.Spec.Tables[i]
		typ, ok := p.Spec.Formats[tg.role]
		if !ok {
			g.closeRowTable(l)
			return fmt.Errorf("the row table %v stated in no format", tg.role)
		}
		v, err := fr.NewRowView(st.Entry)
		if err != nil {
			g.closeRowTable(l)
			return err
		}
		nt := &rowTable{role: tg.role, view: v, typ: typ, rowBytes: st.RowBytes, col: make([]int, len(tg.cols))}
		for c, f := range tg.cols {
			i := slices.IndexFunc(nt.ranges, func(r rowRange) bool { return r.from == f })
			if i < 0 {
				nt.ranges = append(nt.ranges, rowRange{from: f, slot: map[uint32]uint32{}})
				i = len(nt.ranges) - 1
			}
			nt.col[c] = i
		}
		l.rowTabs = append(l.rowTabs, nt)
	}
	return nil
}

// at is the rows on the card of the range a gather binds from from, nil for
// a first row no gather binds.
func (nt *rowTable) at(from int) *resident {
	for i := range nt.ranges {
		if nt.ranges[i].from == from {
			return &nt.ranges[i].rows
		}
	}
	return nil
}

// rowSlots makes r hold n rows of rb bytes on the card, charged to the
// budget. Callers hold g.mu.
func (g *devTier) rowSlots(r *rowRange, n, rb int) error {
	if n <= r.slots {
		return nil
	}
	b, err := g.dev.Alloc(n * rb)
	if err != nil {
		return fmt.Errorf("%d rows of the row table: %w", n, err)
	}
	g.freeRowSlots(r, rb)
	r.rows = resident{qs: b, nrows: n, ok: true}
	r.slots = n
	g.charge(uint64(n * rb))
	return nil
}

func (g *devTier) freeRowSlots(r *rowRange, rb int) {
	if r.rows.qs != nil {
		r.rows.qs.Free()
		g.refund(uint64(r.slots * rb))
	}
	r.rows, r.slots = resident{}, 0
}

// closeRowTable lets block l's tables go: their rows on the card and their
// views. Callers hold g.mu.
func (g *devTier) closeRowTable(l *layer) {
	for _, nt := range l.rowTabs {
		for i := range nt.ranges {
			g.freeRowSlots(&nt.ranges[i], nt.rowBytes)
		}
		nt.view.Close()
	}
	l.rowTabs = nil
}

// partRowSel is the rows a form gathers of the row table of role: the own
// buffer sel holds them, n a row, ids at the form's rows.
type partRowSel struct {
	role        jlm.Role
	sel, n, ids int
}

// rowHalves finds where a form gathering row tables' rows is cut
// (partForm.rowSplit): at its first code that reads a table, each table's
// rows written by a code before it.
func (f *partForm) rowHalves(gr *graph.Graph, vals map[graph.ValueID]int, rows int) error {
	gs, err := rowTableGathers(gr)
	if err != nil || len(gs) == 0 {
		return err
	}
	f.rowSplit = -1
	for ci, args := range f.args {
		for _, a := range args {
			if a.kind == partWeight && f.rowSplit < 0 && slices.ContainsFunc(gs, func(tg tableGather) bool { return tg.role == a.role }) {
				f.rowSplit = ci
			}
		}
	}
	if f.rowSplit <= 0 {
		return fmt.Errorf("a form gathering row table rows with no code before its gathers (%d)", f.rowSplit)
	}
	for _, tg := range gs {
		i, ok := vals[tg.rows]
		if !ok {
			return fmt.Errorf("the rows the row table %v is gathered by are not one of the form's own values", tg.role)
		}
		written := false
		for _, args := range f.args[:f.rowSplit] {
			for _, a := range args {
				written = written || a.kind == partOwn && a.i == i && a.writes
			}
		}
		if !written {
			return fmt.Errorf("the rows the row table %v is gathered by are written by no code before the gathers", tg.role)
		}
		w := gr.Values[tg.rows].Width
		f.rowSels = append(f.rowSels, partRowSel{role: tg.role, sel: i, n: w, ids: rows * w})
	}
	return nil
}

// rowsRun is runAs for a form gathering row tables' rows
// (partForm.rowSplit): its codes up to the gathers, each table's hashed
// rows read back and read in from the container through the table's view,
// each range's distinct ids copied into its slots on the card, each id
// rewritten to its slot, then the codes that gather.
func (sv *servedPart) rowsRun(g *devTier, lc *launcher, s backend.Session, inst *partInst, kind nn.Part, l *layer, in, out []backend.Buf) error {
	f := sv.f
	copy(sv.in[:sv.n], in)
	copy(sv.out[:sv.m], out)
	m0, err := sv.launchCodes(lc, inst, sv.in[:sv.n], sv.out[:sv.m], 0, f.rowSplit)
	if err != nil {
		return err
	}
	if err := s.Sync(); err != nil {
		return err
	}
	read := 0
	defer dropTableRows(l.rowTabs)
	for _, rs := range f.rowSels {
		nt := rowTableOf(l.rowTabs, rs.role)
		if nt == nil || len(nt.col) != rs.n {
			return fmt.Errorf("a form gathering the row table %v's rows on a block that holds no table of its columns", rs.role)
		}
		n := rs.ids
		if sv.rows > 0 {
			n = min(n, sv.rows*rs.n)
		}
		if err := g.tableRowsIn(s, inst.bufs[rs.sel], nt, rs.n, n); err != nil {
			return err
		}
		read += n
	}
	m1, err := sv.launchCodes(lc, inst, sv.in[:sv.n], sv.out[:sv.m], f.rowSplit, len(f.codes))
	if err != nil {
		return err
	}
	g.Steps.Serve(kind)
	g.TableRowsRead += read
	g.MatrixLaunches[kind] += m0.matrix + m1.matrix
	g.AttendLaunches[kind] += m0.attend + m1.attend
	return nil
}

// dropTableRows lets every table's rows a call read in go.
func dropTableRows(ts []*rowTable) {
	for _, t := range ts {
		t.view.Drop()
	}
}

// tableRowsIn reads back the n ids buf holds, w a row, makes table nt's rows
// they name resident in its view, copies each range's distinct rows into its
// slots on the card and writes each id back as its slot.
func (g *devTier) tableRowsIn(s backend.Session, buf backend.Buf, nt *rowTable, w, n int) error {
	nt.ids = growU32(nt.ids, n)
	ids := nt.ids[:n]
	if err := s.Read(buf, u32b(ids)); err != nil {
		return err
	}
	if err := nt.view.Ensure(ids); err != nil {
		return err
	}
	for i := range nt.ranges {
		r := &nt.ranges[i]
		clear(r.slot)
		r.order = r.order[:0]
	}
	for j, id := range ids {
		if uint64(id) >= nt.view.Rows() {
			return fmt.Errorf("a gathered row named row %d of a table of %d", id, nt.view.Rows())
		}
		r := &nt.ranges[nt.col[j%w]]
		k, ok := r.slot[id]
		if !ok {
			k = uint32(len(r.order))
			r.slot[id] = k
			r.order = append(r.order, id)
		}
		ids[j] = uint32(r.from) + k
	}
	rb, base := nt.rowBytes, nt.view.Base()
	for i := range nt.ranges {
		r := &nt.ranges[i]
		if len(r.order) == 0 {
			continue
		}
		if err := g.rowSlots(r, len(r.order), rb); err != nil {
			return err
		}
		r.stage = slices.Grow(r.stage[:0], len(r.order)*rb)[:len(r.order)*rb]
		for k, id := range r.order {
			copy(r.stage[k*rb:(k+1)*rb], unsafe.Slice((*byte)(unsafe.Add(base, int(id)*rb)), rb))
		}
		if err := s.WriteAt(r.rows.qs, 0, r.stage); err != nil {
			return err
		}
	}
	return s.WriteAt(buf, 0, u32b(ids))
}
