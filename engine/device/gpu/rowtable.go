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
// card. A row table is read a row at a time and never resident whole, so
// the card holds none of it: the container's rows are read through a view
// of the card's frames (jlm.Frames.NewRowView), and the rows a call's
// gathers name are copied into slots on the card, each range a gather binds the table
// from (its first row, from) holding its own. A range's slot is an id's
// place among the call's distinct ids of that range, so it stays inside the
// range however wide the call; each id is rewritten to its slot past the
// range's first row before the gathers read it (rowsRun).
type rowTable struct {
	view     *jlm.RowView
	typ      jlm.Type
	rowBytes int
	// col is each of the gathered row's columns' range, an index into ranges.
	col    []int
	ranges []rowRange
	ids    []uint32
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

// rowTableGather is the value a block's gathers of a row table read their
// rows by (graph.Graph.TableRows), and the first row each of its columns is
// gathered from: -1 for a graph gathering no table's rows, an error for a
// column gathered from two first rows or by none.
func rowTableGather(gr *graph.Graph) (graph.ValueID, []int, error) {
	rv := gr.TableRows()
	if rv < 0 {
		return -1, nil, nil
	}
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
	gathers := false
	for _, o := range gr.Ops {
		if o.Kind != graph.OpEmbed || len(o.Weight) != 1 || !jlm.RowTableRole(o.Weight[0].Role) {
			continue
		}
		gathers = true
		at, ok := cut[o.In[0]]
		if len(o.In) != 1 || !ok || at[0] < 0 || at[0]+at[1] > len(cols) {
			return -1, nil, fmt.Errorf("a row table gathered by rows that are not its gathered row's columns")
		}
		for c := at[0]; c < at[0]+at[1]; c++ {
			if cols[c] >= 0 && cols[c] != o.Weight[0].From {
				return -1, nil, fmt.Errorf("column %d of a gathered row gathered from rows %d and %d", c, cols[c], o.Weight[0].From)
			}
			cols[c] = o.Weight[0].From
		}
	}
	if !gathers {
		return -1, nil, nil
	}
	for c, f := range cols {
		if f < 0 {
			return -1, nil, fmt.Errorf("column %d of a gathered row is gathered by none", c)
		}
	}
	return rv, cols, nil
}

// openRowTable opens block l's row table on the card where its graph
// gathers one, through fr: what the step's codes read is copied in a call
// at a time (rowsRun).
func (g *devTier) openRowTable(l *layer, p *nn.LayerPlan, fr *jlm.Frames) error {
	if l.rowTab != nil || l.graph == nil || p.Spec == nil || p.Spec.Table == nil {
		return nil
	}
	rv, froms, err := rowTableGather(l.graph)
	if err != nil || rv < 0 {
		return err
	}
	if fr == nil {
		return fmt.Errorf("a row table offered with no frames to read its rows through")
	}
	typ, ok := p.Spec.Formats[p.Spec.Table.Role]
	if !ok {
		return fmt.Errorf("a row table stated in no format")
	}
	v, err := fr.NewRowView(p.Spec.Table)
	if err != nil {
		return err
	}
	nt := &rowTable{view: v, typ: typ, rowBytes: p.Spec.TableRow, col: make([]int, len(froms))}
	for c, f := range froms {
		i := slices.IndexFunc(nt.ranges, func(r rowRange) bool { return r.from == f })
		if i < 0 {
			nt.ranges = append(nt.ranges, rowRange{from: f, slot: map[uint32]uint32{}})
			i = len(nt.ranges) - 1
		}
		nt.col[c] = i
	}
	l.rowTab = nt
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

// closeRowTable lets block l's table go: its rows on the card and its view.
// Callers hold g.mu.
func (g *devTier) closeRowTable(l *layer) {
	if nt := l.rowTab; nt != nil {
		for i := range nt.ranges {
			g.freeRowSlots(&nt.ranges[i], nt.rowBytes)
		}
		nt.view.Close()
		l.rowTab = nil
	}
}

// rowHalves finds where a form gathering a row table's rows is cut
// (partForm.rowSplit): at its first code that reads the table, the rows
// written by a code before it.
func (f *partForm) rowHalves(gr *graph.Graph, vals map[graph.ValueID]int, rows int) error {
	rv, _, err := rowTableGather(gr)
	if err != nil || rv < 0 {
		return err
	}
	i, ok := vals[rv]
	if !ok {
		return fmt.Errorf("the rows a row table is gathered by are not one of the form's own values")
	}
	f.rowSel, f.rowN, f.rowIDs = i, gr.Values[rv].Width, rows*gr.Values[rv].Width
	f.rowSplit = -1
	for ci, args := range f.args {
		for _, a := range args {
			if a.kind == partWeight && jlm.RowTableRole(a.role) && f.rowSplit < 0 {
				f.rowSplit = ci
			}
		}
	}
	if f.rowSplit <= 0 {
		return fmt.Errorf("a form gathering row table rows with no code before its gathers (%d)", f.rowSplit)
	}
	written := false
	for _, args := range f.args[:f.rowSplit] {
		for _, a := range args {
			written = written || a.kind == partOwn && a.i == f.rowSel && a.writes
		}
	}
	if !written {
		return fmt.Errorf("the rows a row table is gathered by are written by no code before the gathers")
	}
	return nil
}

// rowsRun is runAs for a form gathering a row table's rows
// (partForm.rowSplit): its codes up to the gathers, the hash's rows read
// back and read in from the container through the table's view, each
// range's distinct ids copied into its slots on the card, each id
// rewritten to its slot, then the codes that gather.
func (sv *servedPart) rowsRun(g *devTier, lc *launcher, s backend.Session, inst *partInst, kind nn.Part, l *layer, in, out []backend.Buf) error {
	f, nt := sv.f, l.rowTab
	if nt == nil || len(nt.col) != f.rowN {
		return fmt.Errorf("a form gathering row table rows on a block that holds no table of its columns")
	}
	n := f.rowIDs
	if sv.rows > 0 {
		n = min(n, sv.rows*f.rowN)
	}
	copy(sv.in[:sv.n], in)
	copy(sv.out[:sv.m], out)
	m0, err := sv.launchCodes(lc, inst, sv.in[:sv.n], sv.out[:sv.m], 0, f.rowSplit)
	if err != nil {
		return err
	}
	if err := s.Sync(); err != nil {
		return err
	}
	nt.ids = growU32(nt.ids, n)
	ids := nt.ids[:n]
	if err := s.Read(inst.bufs[f.rowSel], u32b(ids)); err != nil {
		return err
	}
	if err := nt.view.Ensure(ids); err != nil {
		return err
	}
	defer nt.view.Drop()
	for i := range nt.ranges {
		r := &nt.ranges[i]
		clear(r.slot)
		r.order = r.order[:0]
	}
	for j, id := range ids {
		if uint64(id) >= nt.view.Rows() {
			return fmt.Errorf("a gathered row named row %d of a table of %d", id, nt.view.Rows())
		}
		r := &nt.ranges[nt.col[j%f.rowN]]
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
	if err := s.WriteAt(inst.bufs[f.rowSel], 0, u32b(ids)); err != nil {
		return err
	}
	m1, err := sv.launchCodes(lc, inst, sv.in[:sv.n], sv.out[:sv.m], f.rowSplit, len(f.codes))
	if err != nil {
		return err
	}
	g.Steps.Serve(kind)
	g.TableRowsRead += n
	g.MatrixLaunches[kind] += m0.matrix + m1.matrix
	g.AttendLaunches[kind] += m0.attend + m1.attend
	return nil
}
