package app

import (
	"strings"
	"testing"

	"github.com/gogpu/ui/core/datatable"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/uitest"
)

// A table cell must be kept to its own column: Canvas.DrawText uses the rect
// for alignment only, so a wide name would overlap the next column.
func TestTableCellIsClippedToItsColumn(t *testing.T) {
	p := NewPainters(newTheme(false), nil)
	canvas := &uitest.MockCanvas{}
	cell := geometry.FromPointSize(geometry.Pt(0, 0), geometry.Sz(180, 28))

	p.DataTable.PaintCell(canvas, datatable.CellPaintState{
		Bounds: cell,
		Value:  "Qwen2-VL-2B-Instruct-Q4_K_M.jlm",
	})

	// Assert the drawn string, not the clip: gg's clip never reaches
	// DrawString, so a clip-only check passed while the bug was live.
	if len(canvas.Texts) != 1 {
		t.Fatalf("expected one DrawText, got %d", len(canvas.Texts))
	}
	drawn := canvas.Texts[0].Text
	if drawn == "Qwen2-VL-2B-Instruct-Q4_K_M.jlm" {
		t.Error("the value was drawn whole: it is wider than its column and " +
			"will be painted over the next one")
	}
	if !strings.HasSuffix(drawn, "\u2026") {
		t.Errorf("drew %q: a truncated value has to say it was truncated", drawn)
	}
	if w := canvas.MeasureText(drawn, 14, false); w > cell.Width()-24 {
		t.Errorf("drew %q, %v wide in a %v column", drawn, w, cell.Width()-24)
	}

	// The clip stays as well, and must not leak into the next cell.
	if canvas.PopClipCount != len(canvas.Clips) {
		t.Errorf("%d clip(s) pushed and %d popped", len(canvas.Clips), canvas.PopClipCount)
	}
}
