package app

import (
	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"
)

// pageStack shows the selected page and nothing else, at the whole size it
// is given. Every page stays mounted, so its bindings live while hidden;
// tabPage keeps the hidden ones out of the compositor's traversal.
type pageStack struct {
	widget.WidgetBase
	sel   state.Signal[int]
	pages []widget.Widget
}

func newPageStack(sel state.Signal[int], pages []widget.Widget) *pageStack {
	var _ widget.Lifecycle = (*pageStack)(nil)
	p := &pageStack{sel: sel, pages: pages}
	for _, pg := range pages {
		if s, ok := pg.(interface{ SetParent(widget.Widget) }); ok {
			s.SetParent(p)
		}
	}
	p.SetVisible(true)
	p.SetEnabled(true)
	return p
}

func (p *pageStack) cur() widget.Widget {
	if i := p.sel.Get(); i >= 0 && i < len(p.pages) {
		return p.pages[i]
	}
	return nil
}

func (p *pageStack) Layout(ctx widget.Context, c geometry.Constraints) geometry.Size {
	sz := c.Constrain(geometry.Sz(c.MaxWidth, c.MaxHeight))
	if pg := p.cur(); pg != nil {
		widget.LayoutChild(pg, ctx, geometry.Tight(sz))
	}
	p.SetBounds(geometry.FromPointSize(p.Position(), sz))
	return sz
}

func (p *pageStack) Draw(ctx widget.Context, canvas widget.Canvas) {
	pg := p.cur()
	if pg == nil {
		return
	}
	if s, ok := pg.(interface{ SetBounds(geometry.Rect) }); ok {
		s.SetBounds(p.Bounds())
	}
	widget.StampScreenOrigin(pg, canvas)
	pg.Draw(ctx, canvas)
}

func (p *pageStack) Event(ctx widget.Context, e event.Event) bool {
	if pg := p.cur(); pg != nil {
		return pg.Event(ctx, e)
	}
	return false
}

// Children is the shown page only: the traversals that walk it -- drawing,
// hit-testing, the compositor -- have no business in a hidden one.
func (p *pageStack) Children() []widget.Widget {
	if pg := p.cur(); pg != nil {
		return []widget.Widget{pg}
	}
	return nil
}

// Mount mounts every page, shown or not, and relays out on a switch.
func (p *pageStack) Mount(ctx widget.Context) {
	for _, pg := range p.pages {
		widget.MountTree(pg, ctx)
	}
	if sched := ctx.Scheduler(); sched != nil {
		p.AddBinding(state.BindToSchedulerLayout(p.sel.AsReadonly(), p, sched))
	}
}

func (p *pageStack) Unmount() {
	for _, pg := range p.pages {
		widget.UnmountTree(pg)
	}
}
