package app

import (
	"testing"

	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/primitives"
)

func tight(w, h float32) geometry.Constraints {
	return geometry.Constraints{MinWidth: w, MaxWidth: w, MaxHeight: h}
}

// A typography role must carry its line height, not only its size (body is 14
// on a 20px line).
func TestATypographyRoleCarriesItsLineHeight(t *testing.T) {
	p := NewPainters(newTheme(false), nil)

	role := p.Type.BodyMedium
	if role.FontSize == 0 || role.LineHeight == 0 {
		t.Fatalf("the theme's body role is %v: the scale is not wired", role)
	}
	if role.LineHeight == role.FontSize {
		t.Fatalf("body leads at 1.0, so this gate cannot tell a dropped line "+
			"height from a kept one: %v", role)
	}

	w := p.Role(primitives.Text("hello"), role)
	if w == nil {
		t.Fatal("Role returned nothing")
	}
	// The widget's own measurement is the observable: a role with a 20/14
	// ratio must lay out taller than one line of 14.
	sz := w.Layout(nil, tight(400, 400))
	if sz.Height < role.LineHeight-1 {
		t.Errorf("a body-role line measured %v high against a %v line height: "+
			"the ratio was dropped", sz.Height, role.LineHeight)
	}
}

// The scales must actually be populated from the theme.
func TestTheScalesAreWired(t *testing.T) {
	p := NewPainters(newTheme(false), nil)
	if p.Type.TitleLarge.FontSize == 0 || p.Type.LabelSmall.FontSize == 0 {
		t.Error("the type scale is empty")
	}
	if p.Shape.Medium == 0 || p.Shape.ExtraSmall == 0 {
		t.Error("the shape scale is empty")
	}
	if p.Shape.Medium <= p.Shape.ExtraSmall {
		t.Errorf("the shape scale does not increase: extra-small %v, medium %v",
			p.Shape.ExtraSmall, p.Shape.Medium)
	}
}
