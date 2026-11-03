package widgets

import (
	"testing"

	"github.com/gogpu/ui/event"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
)

func mouse(t event.MouseEventType, x, y float32) *event.MouseEvent {
	return &event.MouseEvent{MouseType: t, Button: event.ButtonLeft, Position: geometry.Pt(x, y)}
}

// The toolkit's protocol: Enter and Leave are trusted as sent, and a click is
// a press and a release inside. A widget that re-tested Enter against the
// position stayed lit after the pointer left -- the sidebar's theme row did.
func TestAPressableFollowsTheToolkitsProtocol(t *testing.T) {
	clicks := 0
	n := NewNavItem(IconChat(), "Chat", nil, func() { clicks++ }, NavColors{}, 14)
	n.SetBounds(geometry.NewRect(0, 0, 200, 36))

	n.Event(nil, mouse(event.MouseEnter, 999, 999)) // where the window says, not where it is
	if !n.hover {
		t.Fatal("Enter did not light the row")
	}
	n.Event(nil, mouse(event.MouseLeave, 10, 10))
	if n.hover {
		t.Fatal("Leave did not clear the row: it stays lit after the pointer goes")
	}

	n.Event(nil, mouse(event.MousePress, 10, 10))
	if clicks != 0 {
		t.Fatal("a press alone clicked")
	}
	n.Event(nil, mouse(event.MouseRelease, 10, 10))
	if clicks != 1 {
		t.Fatalf("a press and a release inside clicked %d time(s)", clicks)
	}

	n.Event(nil, mouse(event.MousePress, 10, 10))
	n.Event(nil, mouse(event.MouseRelease, 500, 10))
	if clicks != 1 {
		t.Fatal("releasing outside clicked: a drag off a button must cancel it")
	}

	off := state.NewSignal(true)
	b := NewIconButton(IconSend(), "Send", func() { clicks++ }, IconButtonColors{}).DisabledSignal(off.AsReadonly())
	b.SetBounds(geometry.NewRect(0, 0, 32, 32))
	b.Event(nil, mouse(event.MousePress, 5, 5))
	b.Event(nil, mouse(event.MouseRelease, 5, 5))
	if clicks != 1 {
		t.Fatal("a disabled button clicked")
	}
}
