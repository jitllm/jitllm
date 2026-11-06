package cpudev

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/nn"
)

// Tables is a model's rotary tables on the CPU device: for each rotary a
// form of one row and one of a chunk's (a tower's grid table one of a whole
// picture's), composed once at the instance's placement, reading the
// rotary's planes, and a call's rows as the tables read them
// (composer.TableRopeRows): its live count, then a row's cache position and
// where its sequence's marks start.
type Tables struct {
	one, many []Part
	planes    [2][]float32
	widths    [2]int
	w         Weights
	dev       *Device
	rows      []uint32
	// zero is the marks a model of no marks binds, one empty block.
	zero [1]uint32
	// req and pos are a grid table's request and rows (a tower's picture):
	// the grid's width, and the live rows and their positions.
	req, pos []uint32
	call     [4]uint32
	zeros    []uint32
	recRows  [3]uint32
	bind     Call
}

// NewTables is the rotary tables' part on dev: rotary i's forms one[i] and
// many[i], its planes planes[i] and its rows widths[i] wide, their weights
// bound by w.
func NewTables(dev *Device, one, many []Part, planes [2][]float32, widths [2]int, w Weights) *Tables {
	t := &Tables{one: one, many: many, planes: planes, widths: widths, w: w, dev: dev}
	u := 1
	for i := range one {
		for _, p := range []*Part{&t.one[i], &t.many[i]} {
			if p.F != nil {
				u = max(u, p.F.Unit, p.F.Rows)
			}
		}
	}
	t.rows = make([]uint32, 1+2*u)
	t.zeros = make([]uint32, 4)
	return t
}

// SetGrid makes the tables a grid table's over a picture: req its request,
// pos its rows and their positions.
func (t *Tables) SetGrid(req, pos []uint32) { t.req, t.pos = req, pos }

// Tables writes the rotary tables of a call's rows into cs and, where the
// model trains a second rotary, csSWA: row i at the cache position desc[2i],
// its sequence's marks from desc[2i+1] in marks (none where marks is nil) --
// each row's positions on the axes from its marks (graph.OpRopePos), then
// its angles. A grid table reads the picture SetGrid named, desc's rows
// counting them.
func (t *Tables) Tables(desc, marks []uint32, cs, csSWA []float32) error {
	outs := [2][]float32{cs, csSWA}
	rows := len(desc) / 2
	for i := range t.one {
		if i >= len(outs) || len(outs[i]) == 0 || t.widths[i] == 0 {
			continue
		}
		if len(outs[i]) < rows*t.widths[i] {
			return fmt.Errorf("model: the rotary table %d on the CPU: %d values for %d rows of %d", i, len(outs[i]), rows, t.widths[i])
		}
		p := &t.one[i]
		if rows > 1 {
			p = &t.many[i]
		}
		f, why := p.Form()
		var weights []unsafe.Pointer
		if why == nn.DeclineServes {
			weights, why = t.w.Bind(f)
		}
		if why != nn.DeclineServes {
			return fmt.Errorf("model: the rotary table %d on the CPU: %s: %w", i, why, p.Err)
		}
		if t.pos != nil {
			// A grid table reads the picture's rows and request whole.
			if f.Rows < rows {
				return fmt.Errorf("model: the grid table on the CPU turns %d rows, a picture's %d", f.Rows, rows)
			}
			t.run(f, weights, rows, nil, t.zero[:], outs[0], outs[1])
			continue
		}
		for r0 := 0; r0 < rows; {
			k := min(f.Rows, rows-r0)
			t.rows[0] = uint32(k)
			for j := range f.Rows {
				r := r0 + min(j, k-1)
				t.rows[1+2*j], t.rows[2+2*j] = desc[2*r], desc[2*r+1]
			}
			mk := t.zero[:]
			if len(marks) > 0 {
				mk = marks
			}
			cs, swa := outs[0], outs[1]
			if i == 0 {
				cs = outs[0][r0*t.widths[0]:]
			} else {
				swa = outs[1][r0*t.widths[1]:]
			}
			t.run(f, weights, k, t.rows, mk, cs, swa)
			r0 += k
		}
	}
	return nil
}

// run calls form f, its weights at weights, over n rows described by rows
// (nil for a grid's), its marks mk, writing cs and csSWA.
func (t *Tables) run(f *Form, weights []unsafe.Pointer, n int, rows, mk []uint32, cs, csSWA []float32) {
	c := &t.bind
	*c = Call{Own: t.dev.Own(f), N: n, Weights: weights}
	c.Rope = [2]unsafe.Pointer{ptr(cs), ptr(csSWA)}
	c.Planes = [2]unsafe.Pointer{ptr(t.planes[0]), ptr(t.planes[1])}
	t.call[0] = uint32(n)
	t.recRows = [3]uint32{1, 0, 0}
	c.At[SrcCall], c.At[SrcZero] = unsafe.Pointer(&t.call[0]), unsafe.Pointer(&t.zeros[0])
	c.At[SrcRecRows], c.At[SrcRopeMarks] = unsafe.Pointer(&t.recRows[0]), ptr(mk)
	if rows != nil {
		c.At[SrcRopeRows] = ptr(rows)
	} else {
		c.At[SrcRequest], c.At[SrcPos] = ptr(t.req), ptr(t.pos)
	}
	for i := range f.Codes {
		t.dev.Call(c, &f.Codes[i])
	}
}

// ptr is the address of s's first element, nil for an empty s.
func ptr[T any](s []T) unsafe.Pointer {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Pointer(&s[0])
}
