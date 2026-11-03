package widgets

import (
	"github.com/jitllm/jitllm/common/session"

	"fmt"

	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"
)

// Placement is re-exported; see [session.Placement].
type Placement = session.Placement

// The placements, re-exported.
const (
	PlaceHost     = session.PlaceHost
	PlaceDevice   = session.PlaceDevice
	PlaceResident = session.PlaceResident
)

// OnDevice is re-exported; see [session.OnDevice].
func OnDevice(k int) Placement { return session.OnDevice(k) }

// SegKind is what a stretch of a memory lane holds.
type SegKind int

const (
	SegDense  SegKind = iota // weights that never page: embedding and head
	SegKV                    // attention history sequences hold
	SegKVFree                // history pool pages nobody holds yet
	SegOther                 // scratch and anything else charged here
	SegFree                  // room left under the cap
)

// MemSeg is one stretch of a lane.
type MemSeg struct {
	Kind  SegKind
	Bytes uint64
}

// MemLane is one tier of memory -- the file, host RAM, a card -- drawn as a
// bar against its own capacity: a cell per block page it holds, then the
// segments.
type MemLane struct {
	Name, Caption string
	// Cells is the block index of each page the lane holds, CellBytes the size
	// of one. Disk lanes draw them hollow: the file is the backing store, not a
	// place blocks run.
	Cells     []int
	CellBytes uint64
	Disk      bool
	Segs      []MemSeg
}

// MemView is one loaded model's memory, as the memory map draws it.
type MemView struct {
	// Blocks is one Placement per block.
	Blocks []byte
	// Devices names device k for the legend.
	Devices []string
	Lanes   []MemLane
	// Pos is how many positions the session holds, of a MaxSeq window.
	Pos, MaxSeq int
}

// MemColors is the memory map's palette. Devices is indexed by ordinal and
// wraps.
type MemColors struct {
	Host, Dense, KV, Other, Track, Line, Seam, Text, Muted, OnCell widget.Color
	Devices                                                        []widget.Color
}

// MemoryMap draws where a model is: its blocks in order coloured by the device
// that runs them, with every CPU/GPU seam marked; one bar per tier of memory
// with a cell per page it holds; and how much of the context window the
// session has filled. It is the website's machine room flattened into rows,
// fed by the engine's own counters rather than a simulation.
type MemoryMap struct {
	widget.WidgetBase

	view   state.ReadonlySignal[MemView]
	colors MemColors
	size   float32
}

// NewMemoryMap builds a map bound to view. size is the caller's type role.
func NewMemoryMap(view state.ReadonlySignal[MemView], colors MemColors, size float32) *MemoryMap {
	var _ widget.Lifecycle = (*MemoryMap)(nil)
	m := &MemoryMap{view: view, colors: colors, size: size}
	m.SetVisible(true)
	m.SetEnabled(true)
	return m
}

const (
	mmLabelW = 120 // the name column
	mmBarH   = 18
	mmGap    = 1
	mmMinW   = 2 // a stretch that holds anything is never drawn narrower
)

func (m *MemoryMap) lineH() float32 { return m.size + 6 }

// rowH is one bar with its caption under it.
func (m *MemoryMap) rowH() float32 { return mmBarH + m.lineH() + 6 }

// Layout fills the width; the height follows the lane count.
func (m *MemoryMap) Layout(_ widget.Context, c geometry.Constraints) geometry.Size {
	v := m.view.Get()
	if len(v.Blocks) == 0 {
		return c.Constrain(geometry.Sz(c.MaxWidth, 0))
	}
	h := m.rowH()*float32(2+len(v.Lanes)) + m.lineH()
	return c.Constrain(geometry.Sz(c.MaxWidth, h))
}

// Draw paints the layers row, the context row, a row per lane and the legend.
func (m *MemoryMap) Draw(_ widget.Context, cv widget.Canvas) {
	v := m.view.Get()
	if len(v.Blocks) == 0 {
		return
	}
	r := m.Bounds()
	x0 := r.Min.X + mmLabelW
	w := r.Width() - mmLabelW
	if w < 40 {
		return
	}
	y := r.Min.Y

	m.label(cv, "layers", r.Min.X, y)
	m.layers(cv, v, x0, y, w)
	m.caption(cv, layersCaption(v), x0, y+mmBarH+3, w)
	y += m.rowH()

	m.label(cv, "context", r.Min.X, y)
	m.context(cv, v, x0, y, w)
	m.caption(cv, contextCaption(v), x0, y+mmBarH+3, w)
	y += m.rowH()

	for _, l := range v.Lanes {
		m.label(cv, l.Name, r.Min.X, y)
		m.lane(cv, v, l, x0, y, w)
		m.caption(cv, l.Caption, x0, y+mmBarH+3, w)
		y += m.rowH()
	}
	m.legend(cv, v, x0, y, w)
}

func (m *MemoryMap) device(k int) widget.Color {
	if k < 0 || len(m.colors.Devices) == 0 {
		return m.colors.Host
	}
	return m.colors.Devices[k%len(m.colors.Devices)]
}

// layers is one cell per block, in order. A host block whose page is out is
// hollow, and a seam -- consecutive blocks on different tiers, where the
// token crosses a bus -- is a bar between them.
func (m *MemoryMap) layers(cv widget.Canvas, v MemView, x, y, w float32) {
	n := len(v.Blocks)
	gap := float32(mmGap)
	cell := (w - gap*float32(n-1)) / float32(n)
	if cell < 2 {
		cell, gap = w/float32(n), 0
	}
	for i, b := range v.Blocks {
		p := Placement(b)
		cx := x + float32(i)*(cell+gap)
		rect := geometry.NewRect(cx, y, cell, mmBarH)
		col := m.device(p.Device())
		if p.Device() < 0 && !p.Resident() {
			cv.StrokeRect(rect, col, 1)
		} else {
			cv.DrawRect(rect, col)
			m.cellText(cv, fmt.Sprint(i), rect)
		}
		if i > 0 && Placement(v.Blocks[i-1]).Device() != p.Device() {
			sx := cx - gap/2
			cv.DrawLine(geometry.Pt(sx, y-3), geometry.Pt(sx, y+mmBarH+3), m.colors.Seam, 2)
		}
	}
}

// context is the window, filled to the session's position.
func (m *MemoryMap) context(cv widget.Canvas, v MemView, x, y, w float32) {
	cv.DrawRect(geometry.NewRect(x, y+4, w, mmBarH-8), m.colors.Track)
	if v.MaxSeq <= 0 || v.Pos <= 0 {
		return
	}
	f := min(float32(v.Pos)/float32(v.MaxSeq), 1)
	cv.DrawRect(geometry.NewRect(x, y+4, max(f*w, mmMinW), mmBarH-8), m.colors.KV)
}

// lane is one tier: a cell per page it holds, then each segment, against the
// lane's own total so a 2 GiB card and a 64 GiB host both read as full bars.
func (m *MemoryMap) lane(cv widget.Canvas, v MemView, l MemLane, x, y, w float32) {
	type unit struct {
		bytes uint64
		cell  int // block index, or -1 for a segment
		kind  SegKind
	}
	var units []unit
	for _, c := range l.Cells {
		units = append(units, unit{l.CellBytes, c, 0})
	}
	for _, s := range l.Segs {
		if s.Bytes > 0 {
			units = append(units, unit{s.Bytes, -1, s.Kind})
		}
	}
	if len(units) == 0 {
		cv.DrawRect(geometry.NewRect(x, y, w, mmBarH), m.colors.Track)
		return
	}
	var total uint64
	for _, u := range units {
		total += max(u.bytes, 1)
	}
	gap := float32(mmGap)
	if float32(len(units))*(mmMinW+gap) > w {
		gap = 0
	}
	// Every unit gets the floor, and what is left is shared by size.
	spare := max(w-float32(len(units))*(mmMinW+gap), 0)
	cx := x
	for _, u := range units {
		uw := mmMinW + spare*float32(max(u.bytes, 1))/float32(total)
		rect := geometry.NewRect(cx, y, uw, mmBarH)
		switch {
		case u.cell >= 0 && l.Disk:
			cv.StrokeRect(rect, m.colors.Line, 1)
		case u.cell >= 0:
			p := PlaceHost
			if u.cell < len(v.Blocks) {
				p = Placement(v.Blocks[u.cell])
			}
			cv.DrawRect(rect, m.device(p.Device()))
			m.cellText(cv, fmt.Sprint(u.cell), rect)
		case u.kind == SegKVFree:
			cv.StrokeRect(rect, m.colors.KV, 1)
		case u.kind == SegFree:
			cv.DrawRect(rect, m.colors.Track)
		default:
			cv.DrawRect(rect, m.segColor(u.kind))
		}
		cx += uw + gap
	}
}

func (m *MemoryMap) segColor(k SegKind) widget.Color {
	switch k {
	case SegDense:
		return m.colors.Dense
	case SegKV, SegKVFree:
		return m.colors.KV
	case SegOther:
		return m.colors.Other
	}
	return m.colors.Track
}

// legend names every colour the map used.
func (m *MemoryMap) legend(cv widget.Canvas, v MemView, x, y, w float32) {
	type key struct {
		c      widget.Color
		label  string
		hollow bool
	}
	keys := []key{{m.colors.Host, "host", false}}
	for k, name := range v.Devices {
		keys = append(keys, key{m.device(k), name, false})
	}
	keys = append(keys,
		key{m.colors.Host, "paged out", true},
		key{m.colors.Dense, "dense", false},
		key{m.colors.KV, "kv", false},
		key{m.colors.KV, "kv pool free", true},
		key{m.colors.Other, "scratch", false},
		key{m.colors.Track, "free", false},
	)
	sw := m.size
	cx := x
	for _, k := range keys {
		tw := cv.MeasureText(k.label, m.size, false)
		if cx+sw+4+tw > x+w {
			return
		}
		rect := geometry.NewRect(cx, y+3, sw, m.size-2)
		if k.hollow {
			cv.StrokeRect(rect, k.c, 1)
		} else {
			cv.DrawRect(rect, k.c)
		}
		cv.DrawText(k.label, geometry.NewRect(cx+sw+4, y, tw, m.lineH()), m.size, m.colors.Muted, false, widget.TextAlignLeft)
		cx += sw + 4 + tw + 14
	}
}

func (m *MemoryMap) label(cv widget.Canvas, s string, x, y float32) {
	m.text(cv, s, geometry.NewRect(x, y, mmLabelW-8, mmBarH), m.colors.Text, true)
}

func (m *MemoryMap) caption(cv widget.Canvas, s string, x, y, w float32) {
	m.text(cv, s, geometry.NewRect(x, y, w, m.lineH()), m.colors.Muted, false)
}

// cellText numbers a cell when the number fits inside it.
func (m *MemoryMap) cellText(cv widget.Canvas, s string, r geometry.Rect) {
	size := m.size - 2
	if cv.MeasureText(s, size, false)+4 > r.Width() {
		return
	}
	cv.DrawText(s, r, size, m.colors.OnCell, false, widget.TextAlignCenter)
}

// text draws s cut to fit: Canvas.DrawText does not clip (UPSTREAM.md #8).
func (m *MemoryMap) text(cv widget.Canvas, s string, r geometry.Rect, c widget.Color, bold bool) {
	if cv.MeasureText(s, m.size, bold) > r.Width() {
		rs := []rune(s)
		for len(rs) > 0 && cv.MeasureText(string(rs)+"…", m.size, bold) > r.Width() {
			rs = rs[:len(rs)-1]
		}
		s = string(rs) + "…"
	}
	cv.DrawText(s, r, m.size, c, bold, widget.TextAlignLeft)
}

func layersCaption(v MemView) string {
	var host, out, seams int
	dev := map[int]int{}
	for i, b := range v.Blocks {
		p := Placement(b)
		if p.Device() < 0 {
			host++
			if !p.Resident() {
				out++
			}
		} else {
			dev[p.Device()]++
		}
		if i > 0 && Placement(v.Blocks[i-1]).Device() != p.Device() {
			seams++
		}
	}
	s := ""
	for k := range len(v.Devices) {
		if dev[k] > 0 {
			s += fmt.Sprintf("%d on %s, ", dev[k], v.Devices[k])
		}
	}
	s += fmt.Sprintf("%d on the host", host)
	if out > 0 {
		s += fmt.Sprintf(" (%d paged out)", out)
	}
	switch seams {
	case 0:
	case 1:
		s += " -- each token crosses 1 seam"
	default:
		s += fmt.Sprintf(" -- each token crosses %d seams", seams)
	}
	return s
}

func contextCaption(v MemView) string {
	if v.MaxSeq <= 0 {
		return "no session"
	}
	return fmt.Sprintf("%d of %d positions held in the kv cache", v.Pos, v.MaxSeq)
}

// Event ignores input; the map is a readout.
func (m *MemoryMap) Event(_ widget.Context, _ event.Event) bool { return false }

// Children returns nil.
func (m *MemoryMap) Children() []widget.Widget { return nil }

// Mount binds the view so a new placement relays out and repaints the map:
// the lane count sets its height.
func (m *MemoryMap) Mount(ctx widget.Context) {
	if sched := ctx.Scheduler(); sched != nil {
		m.AddBinding(state.BindToSchedulerLayout(m.view, m, sched))
	}
}

// Unmount has nothing to release; the binding goes with the WidgetBase. It
// must exist: widget.MountTree calls Mount only on a full Lifecycle, so
// without it the map never binds and keeps the height it had empty.
func (m *MemoryMap) Unmount() {}
