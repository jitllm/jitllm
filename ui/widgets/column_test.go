package widgets

import (
	"testing"

	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/uitest"
	"github.com/gogpu/ui/widget"
)

type clickable struct {
	widget.WidgetBase
	events int
}

func newClickable() *clickable {
	c := &clickable{}
	c.SetVisible(true)
	c.SetEnabled(true)
	return c
}

func (c *clickable) Layout(widget.Context, geometry.Constraints) geometry.Size {
	return geometry.Sz(100, 30)
}
func (c *clickable) Draw(_ widget.Context, cv widget.Canvas) {
	cv.DrawText("row", c.Bounds(), 12, widget.ColorBlack, false, widget.TextAlignLeft)
}
func (c *clickable) Event(widget.Context, event.Event) bool { c.events++; return true }
func (c *clickable) Children() []widget.Widget              { return nil }
func (c *clickable) Mount(widget.Context)                   {}
func (c *clickable) Unmount()                               {}

// An event must reach a child, which core/listview rows cannot do. This is the
// whole reason the column exists.
func TestAnEventReachesAChild(t *testing.T) {
	rows := []*clickable{newClickable(), newClickable()}
	n := state.NewSignal(2)
	col := NewColumn(n.AsReadonly(), 4, func(i int) widget.Widget { return rows[i] })

	ctx := uitest.NewMockContext()
	ctx.SchedulerVal = state.NewScheduler(func([]widget.Widget) {})
	widget.MountTree(col, ctx)
	col.Layout(ctx, geometry.Constraints{MaxWidth: 300, MaxHeight: 600})

	if !col.Event(ctx, &event.MouseEvent{}) {
		t.Fatal("the column consumed nothing: no child saw the event")
	}
	if rows[0].events == 0 {
		t.Error("the first row never received an event")
	}
}

// Children are laid out in order, stacked, with the gap between them.
func TestChildrenStackInOrder(t *testing.T) {
	rows := []*clickable{newClickable(), newClickable(), newClickable()}
	n := state.NewSignal(3)
	col := NewColumn(n.AsReadonly(), 4, func(i int) widget.Widget { return rows[i] })

	ctx := uitest.NewMockContext()
	ctx.SchedulerVal = state.NewScheduler(func([]widget.Widget) {})
	widget.MountTree(col, ctx)
	sz := col.Layout(ctx, geometry.Constraints{MaxWidth: 300, MaxHeight: 6000})

	if want := float32(3*30 + 2*4); sz.Height != want {
		t.Errorf("column is %v high, want %v (three 30px rows and two 4px gaps)", sz.Height, want)
	}
	var last float32 = -1
	for i, r := range rows {
		y := r.Bounds().Min.Y
		if y <= last {
			t.Errorf("row %d is at y=%v, not below the one before at %v", i, y, last)
		}
		last = y
	}
}

// A count change rebuilds, and an unchanged count does not; rebuilding every
// layout would discard each child's state once a frame.
func TestRebuildsOnlyWhenTheCountMoves(t *testing.T) {
	built := 0
	n := state.NewSignal(1)
	col := NewColumn(n.AsReadonly(), 0, func(int) widget.Widget {
		built++
		return newClickable()
	})

	ctx := uitest.NewMockContext()
	ctx.SchedulerVal = state.NewScheduler(func([]widget.Widget) {})
	widget.MountTree(col, ctx)
	col.Layout(ctx, geometry.Constraints{MaxWidth: 300, MaxHeight: 600})
	col.Layout(ctx, geometry.Constraints{MaxWidth: 300, MaxHeight: 600})
	if built != 1 {
		t.Errorf("built %d time(s) for an unchanged count of 1", built)
	}

	n.Set(3)
	col.Layout(ctx, geometry.Constraints{MaxWidth: 300, MaxHeight: 600})
	if built != 4 {
		t.Errorf("built %d time(s) after the count went to 3, want 4 in total", built)
	}
	if got := len(col.Children()); got != 3 {
		t.Errorf("the column has %d children, want 3", got)
	}
}

// A column placed off its parent's corner paints its children there too:
// Layout places children in the column's own frame, so Draw must translate.
func TestColumnDrawsInItsOwnFrame(t *testing.T) {
	n := state.NewSignal(1)
	col := NewColumn(n.AsReadonly(), 0, func(int) widget.Widget { return (&Thumbs{}).New(quadAndDisc(), 16) })
	col.Layout(nil, geometry.Constraints{MaxWidth: 100, MaxHeight: 100})
	col.SetBounds(geometry.FromPointSize(geometry.Pt(12, 10), col.Bounds().Size()))

	c := &uitest.MockCanvas{}
	col.Draw(nil, c)
	if len(c.Transforms) == 0 || c.Transforms[0] != geometry.Pt(12, 10) {
		t.Fatalf("transforms %v: the column's children are painted at its parent's corner", c.Transforms)
	}
	if len(c.Images) != 1 || c.Images[0].At != geometry.Pt(0, 0) {
		t.Fatalf("the child drew %v, want once at its local origin", c.Images)
	}
}
