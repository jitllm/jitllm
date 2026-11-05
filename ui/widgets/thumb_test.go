package widgets

import (
	"path/filepath"
	"testing"

	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/uitest"
)

// quadAndDisc() is the engine's own VLM test picture: a yellow disc over four
// coloured quadrants, 384x384.
func quadAndDisc() string {
	return filepath.Join("..", "..", "engine", "testdata", "quad-and-disc.png")
}

// A thumbnail must reach the canvas as a real DrawImage, scaled into its box and
// still showing the picture. The pixel is the assertion, not the call: the
// disc is yellow at the centre, so the thumbnail's centre must be too.
func TestThumbDrawsTheScaledPicture(t *testing.T) {
	th := (&Thumbs{}).New(quadAndDisc(), 96)
	if th.Err() != "" {
		t.Fatalf("the test picture did not decode: %s", th.Err())
	}
	th.SetBounds(geometry.FromPointSize(geometry.Pt(10, 20), geometry.Sz(0, 0)))
	sz := th.Layout(nil, geometry.Constraints{MaxWidth: 400, MaxHeight: 400})
	if sz.Width != 96 || sz.Height != 96 {
		t.Fatalf("a 384x384 picture in a 96 box laid out at %v", sz)
	}

	c := &uitest.MockCanvas{}
	th.Draw(nil, c)
	if len(c.Images) != 1 {
		t.Fatalf("%d DrawImage call(s): the picture never reached the canvas", len(c.Images))
	}
	call := c.Images[0]
	if call.At != geometry.Pt(10, 20) {
		t.Errorf("drawn at %v, the widget is at (10,20)", call.At)
	}
	b := call.Image.Bounds()
	if b.Dx() != 96 || b.Dy() != 96 {
		t.Fatalf("the canvas got a %dx%d image: DrawImage does not scale, so an unscaled "+
			"picture covers the window", b.Dx(), b.Dy())
	}
	r, g, bl, _ := call.Image.At(48, 48).RGBA()
	if r < 0xc000 || g < 0xc000 || bl > 0x6000 {
		t.Errorf("the centre of the thumbnail is rgb(%d,%d,%d), not the yellow disc",
			r>>8, g>>8, bl>>8)
	}
}

// A file that does not decode is a labelled placeholder, never an empty space.
func TestThumbNamesAPictureItCannotRead(t *testing.T) {
	th := (&Thumbs{}).New(filepath.Join(t.TempDir(), "missing.png"), 64)
	if th.Err() == "" {
		t.Fatal("a missing file reported no error")
	}
	th.Layout(nil, geometry.Constraints{MaxWidth: 400, MaxHeight: 400})
	c := &uitest.MockCanvas{}
	th.Draw(nil, c)
	if len(c.Images) != 0 {
		t.Error("an unreadable picture drew an image")
	}
	if len(c.Texts) == 0 {
		t.Error("an unreadable picture said nothing about why")
	}
}
