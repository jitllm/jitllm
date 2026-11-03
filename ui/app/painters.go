package app

import (
	"github.com/gogpu/ui/core/button"
	"github.com/gogpu/ui/core/checkbox"
	"github.com/gogpu/ui/core/chip"
	"github.com/gogpu/ui/core/collapsible"
	"github.com/gogpu/ui/core/datatable"
	"github.com/gogpu/ui/core/dialog"
	"github.com/gogpu/ui/core/dropdown"
	"github.com/gogpu/ui/core/linechart"
	"github.com/gogpu/ui/core/listview"
	"github.com/gogpu/ui/core/menu"
	"github.com/gogpu/ui/core/progress"
	"github.com/gogpu/ui/core/progressbar"
	"github.com/gogpu/ui/core/radio"
	"github.com/gogpu/ui/core/scrollview"
	"github.com/gogpu/ui/core/slider"
	"github.com/gogpu/ui/core/splitview"
	"github.com/gogpu/ui/core/tabview"
	"github.com/gogpu/ui/core/textfield"
	"github.com/gogpu/ui/core/toolbar"
	"github.com/gogpu/ui/core/treeview"
	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/theme"
	"github.com/gogpu/ui/theme/material3"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/widgets"
)

// Painters is one painter value per widget kind. The toolkit has no
// theme-wide painter default, so build this once and hand it down; a screen
// never constructs a material3.XPainter itself.
type Painters struct {
	Button      button.Painter
	Checkbox    checkbox.Painter
	Chip        chip.Painter
	Collapsible collapsible.Painter
	DataTable   datatable.Painter
	Dialog      dialog.Painter
	Dropdown    dropdown.Painter
	LineChart   linechart.Painter
	ListView    listview.Painter
	Menu        menu.Painter
	Progress    progress.Painter
	Radio       radio.Painter
	Scrollbar   scrollview.Painter
	Slider      slider.Painter
	SplitView   splitview.Painter
	TabView     tabview.Painter
	TextField   textfield.Painter
	Toolbar     toolbar.Painter
	TreeView    treeview.Painter

	// ProgressBar takes a colour scheme rather than a Painter interface.
	ProgressBar progressbar.ProgressBarColorScheme
	// ProgressBarWarn is the same scheme in the error colour, for a gauge that
	// has crossed its budget.
	ProgressBarWarn progressbar.ProgressBarColorScheme

	// Colors are the raw theme tokens, for the places a widget draws itself.
	Colors material3.ColorScheme
	// Warn is the palette's orange, which Material has no role for: the
	// attention history in the memory map.
	Warn widget.Color

	// Type is the theme's type scale and Shape its corner-radius scale. Use
	// the roles (see Role) rather than hardcoded font sizes.
	Type  material3.TypeScale
	Shape material3.ShapeScale
	// Space is the 2/4/8/16/24/32 spacing scale.
	Space theme.SpacingScale
	Dark  bool

	// Fonts is the chain the faces below draw by.
	Fonts *Fonts
}

// Prose is the face for running text; see [Fonts.Prose].
func (p Painters) Prose() widgets.Face { return p.Fonts.Prose() }

// Mono is the face for data; see [Fonts.Mono].
func (p Painters) Mono() widgets.Face { return p.Fonts.Mono() }

// NewPainters builds the set from a Material 3 theme.
func NewPainters(m3 *material3.Theme, fonts *Fonts) Painters {
	return Painters{
		Fonts:       fonts,
		Button:      material3.ButtonPainter{Theme: m3},
		Checkbox:    material3.CheckboxPainter{Theme: m3},
		Chip:        material3.ChipPainter{Theme: m3},
		Collapsible: material3.CollapsiblePainter{Theme: m3},
		DataTable:   clipCells{material3.DataTablePainter{Theme: m3}},
		Dialog:      material3.DialogPainter{Theme: m3},
		Dropdown:    material3.DropdownPainter{Theme: m3},
		LineChart:   material3.LineChartPainter{Theme: m3},
		ListView:    material3.ListViewPainter{Theme: m3},
		Menu:        material3.MenuPainter{Theme: m3},
		Progress:    material3.ProgressPainter{Theme: m3},
		Radio:       material3.RadioPainter{Theme: m3},
		Scrollbar:   material3.ScrollbarPainter{Theme: m3},
		Slider:      material3.SliderPainter{Theme: m3},
		SplitView:   material3.SplitViewPainter{Theme: m3},
		TabView:     material3.TabViewPainter{Theme: m3},
		TextField:   material3.TextFieldPainter{Theme: m3},
		Toolbar:     material3.ToolbarPainter{Theme: m3},
		TreeView:    material3.TreeViewPainter{Theme: m3},

		ProgressBar: progressbar.ProgressBarColorScheme{
			Bar:   m3.Colors.Primary,
			Track: m3.Colors.SurfaceVariant,
			Label: m3.Colors.OnSurface,
		},
		ProgressBarWarn: progressbar.ProgressBarColorScheme{
			Bar:   m3.Colors.Error,
			Track: m3.Colors.SurfaceVariant,
			Label: m3.Colors.OnSurface,
		},

		Colors: m3.Colors,
		Warn:   warn(m3.IsDark()),
		Type:   m3.Typography,
		Shape:  m3.Shape,
		Space:  theme.DefaultSpacing(),
		Dark:   m3.IsDark(),
	}
}

// Text is the default body colour on a surface.
func (p Painters) Text() widget.Color { return p.Colors.OnSurface }

// Muted is the secondary label colour: units, captions, hints.
func (p Painters) Muted() widget.Color { return p.Colors.OnSurfaceVariant }

// Surface is the panel background.
func (p Painters) Surface() widget.Color { return p.Colors.Surface }

// Background is the window background.
func (p Painters) Background() widget.Color { return p.Colors.Background }

// clipCells is the table painter with every cell kept to its own column.
// Canvas.DrawText uses its rect for alignment only, so a value wider than its
// column would be drawn over the next one.
//
// It embeds the datatable.Painter interface rather than the concrete painter,
// so the methods it does not override forward on their own.
type clipCells struct{ datatable.Painter }

// m3 table drawing constants, copied because they are unexported: the
// numbers material3.DataTablePainter draws with (datatable.go:189-195).
const (
	tableCellPadH   float32 = 12
	tableCellSize   float32 = 14
	tableHeaderSize float32 = 13
	tableEllipsis           = "\u2026"
)

// PaintCell truncates a value that does not fit its column before drawing it.
// A clip alone does not work for text (gg's ClipRect never reaches
// DrawString), so the text is cut, measured with the canvas. The clip stays
// for backgrounds and in case gg ever honours it for text.
func (c clipCells) PaintCell(canvas widget.Canvas, st datatable.CellPaintState) {
	canvas.PushClip(st.Bounds)
	defer canvas.PopClip()
	st.Value = fit(canvas, st.Value, st.Bounds.Width()-tableCellPadH*2, tableCellSize)
	c.Painter.PaintCell(canvas, st)
}

// PaintHeaderCell does the same for the header, whose font is a point smaller.
func (c clipCells) PaintHeaderCell(canvas widget.Canvas, b geometry.Rect, st datatable.HeaderCellPaintState) {
	canvas.PushClip(b)
	defer canvas.PopClip()
	st.Title = fit(canvas, st.Title, b.Width()-tableCellPadH*2, tableHeaderSize)
	c.Painter.PaintHeaderCell(canvas, b, st)
}

// fit shortens s until it measures within w, ending in an ellipsis. It cuts by
// rune, never leaving half a multi-byte character.
func fit(canvas widget.Canvas, s string, w, size float32) string {
	if s == "" || w <= 0 || canvas.MeasureText(s, size, false) <= w {
		return s
	}
	r := []rune(s)
	// Binary search the longest prefix that fits with the ellipsis.
	lo, hi := 0, len(r)
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if canvas.MeasureText(string(r[:mid])+tableEllipsis, size, false) <= w {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	if lo == 0 {
		return tableEllipsis
	}
	return string(r[:lo]) + tableEllipsis
}

// Role applies a whole typography role -- size, line height and weight -- to a
// text widget. It takes the role rather than a size so the line height that
// the scale pairs with each size is never dropped.
func (p *Painters) Role(t *primitives.TextWidget, role material3.TextStyle) *primitives.TextWidget {
	t = t.FontSize(role.FontSize)
	if role.LineHeight > 0 && role.FontSize > 0 {
		t = t.LineHeight(role.LineHeight / role.FontSize)
	}
	if role.Bold {
		t = t.Bold()
	}
	return t
}

// Card is the surface a grouped block sits on: a container tone, the shape
// scale's medium radius and one step of elevation.
func (p *Painters) Card(w widget.Widget) *primitives.BoxWidget {
	return primitives.Box(w).
		Background(p.Colors.SurfaceContainerLow).
		Rounded(p.Shape.Medium).
		ShadowLevel(1)
}

// Field is the outline an input-bearing box wears.
func (p *Painters) Field(w widget.Widget) *primitives.BoxWidget {
	return primitives.Box(w).
		Rounded(p.Shape.ExtraSmall).
		BorderStyle(1, p.Colors.OutlineVariant)
}

// Table is the table painter with an empty state of its own: what to do next,
// where the toolkit would say "No data".
func (p Painters) Table(empty string) datatable.Painter {
	return emptyTable{Painter: p.DataTable, msg: empty, color: p.Muted()}
}

type emptyTable struct {
	datatable.Painter
	msg   string
	color widget.Color
}

// PaintEmptyState centres the message, cut to the table's width: the canvas
// does not clip text.
func (e emptyTable) PaintEmptyState(canvas widget.Canvas, b geometry.Rect) {
	if b.IsEmpty() {
		return
	}
	canvas.DrawText(fit(canvas, e.msg, b.Width()-tableCellPadH*2, tableCellSize), b, tableCellSize, e.color, false, widget.TextAlignCenter)
}

// BareTextField is the text field painter without its own box: for a field
// that sits inside a card which already draws one.
func (p Painters) BareTextField() textfield.Painter { return bareField{p.TextField} }

type bareField struct{ textfield.Painter }

func (b bareField) PaintTextField(cv widget.Canvas, st *textfield.PaintState) {
	b.Painter.PaintTextField(noBox{Canvas: cv, box: st.Bounds}, st)
}

// noBox drops the fill and the outline drawn at the field's own bounds and
// passes everything else -- text, selection, cursor -- through.
type noBox struct {
	widget.Canvas
	box geometry.Rect
}

func (n noBox) DrawRoundRect(r geometry.Rect, c widget.Color, radius float32) {
	if r != n.box {
		n.Canvas.DrawRoundRect(r, c, radius)
	}
}

func (n noBox) StrokeRoundRect(r geometry.Rect, c widget.Color, radius, w float32) {
	if r != n.box {
		n.Canvas.StrokeRoundRect(r, c, radius, w)
	}
}
