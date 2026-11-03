package screen

import (
	"strings"

	"github.com/gogpu/ui/core/scrollview"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/widgets"
)

// The pieces Discover's layout is made of, for the pages that list files:
// a heading with its actions, rows with a letter tile, and a panel beside them.

// pageHead is a page's title, one sentence under it, and its actions on the
// right.
func pageHead(sh *app.Shell, title, sub string, actions ...widget.Widget) widget.Widget {
	p := sh.P
	text := primitives.VBox(
		p.Role(primitives.Text(title).Color(p.Text()), p.Type.HeadlineSmall).Bold(),
		widgets.NewParagraph(sub).FontSize(p.Type.BodyMedium.FontSize).Color(p.Muted()).Font(sh.P.Prose()),
	).Gap(p.Space.XS).CrossAlign(primitives.CrossAxisStretch)
	row := append([]widget.Widget{primitives.Expanded(text)}, actions...)
	return primitives.HBox(row...).Gap(p.Space.S)
}

// tileLook is a file's letter tile: the first letter of its family, in a
// colour that family keeps wherever it appears.
func tileLook(p app.Painters, name string) widgets.TileLook {
	// A vision tower is named for its model: mmproj-SmolVLM-... is SmolVLM's.
	name = strings.TrimPrefix(strings.TrimPrefix(name, "mmproj-"), "mmproj_")
	fam := strings.FieldsFunc(name, func(r rune) bool { return r == '-' || r == '_' || r == ' ' || r == '.' })
	if len(fam) == 0 {
		return widgets.TileLook{}
	}
	h := 0
	for _, r := range strings.ToLower(fam[0]) {
		h += int(r)
	}
	palette := []widget.Color{p.Colors.Primary, p.Colors.Secondary, p.Colors.Tertiary, p.Warn}
	return widgets.TileLook{Letter: strings.ToUpper(firstRune(fam[0])), Bg: palette[h%len(palette)]}
}

// listRow is one row of a list page: the tile, a title over a line of facts,
// and two short lines on the right -- what state it is in and what it means
// here. A click selects it.
func listRow(sh *app.Shell, look state.ReadonlySignal[widgets.TileLook], title, meta, right, rightNote state.ReadonlySignal[string],
	selected state.ReadonlySignal[bool], onClick func()) widget.Widget {
	p := sh.P
	small := p.Type.BodySmall.FontSize
	row := primitives.Box(primitives.HBox(
		widgets.NewLetterTile("", 36, p.Colors.Primary, p.Colors.OnPrimary).Bind(look),
		primitives.Expanded(primitives.VBox(
			widgets.NewParagraph("").ContentSignal(title).FontSize(p.Type.BodyLarge.FontSize).
				Color(p.Text()).MaxLines(1).Font(sh.P.Prose()).Bold(),
			widgets.NewParagraph("").ContentSignal(meta).FontSize(small).
				Color(p.Muted()).MaxLines(1).Font(sh.P.Prose()),
		).Gap(2)),
		primitives.Box(primitives.VBox(
			widgets.NewParagraph("").ContentSignal(right).FontSize(small).Color(p.Text()).
				MaxLines(1).Font(sh.P.Prose()).Align(widget.TextAlignRight),
			widgets.NewParagraph("").ContentSignal(rightNote).FontSize(small).Color(p.Muted()).
				MaxLines(1).Font(sh.P.Prose()).Align(widget.TextAlignRight),
		).Gap(2)).Width(rowSideWidth),
	).Gap(p.Space.S+p.Space.XS)).PaddingXY(p.Space.S+p.Space.XS, p.Space.S+p.Space.XS)
	return widgets.NewClickable(row, selected, onClick,
		p.Colors.SurfaceContainer, p.Colors.SurfaceContainerHigh, p.Colors.OutlineVariant)
}

// rowSideWidth is the right-hand column of a list row: "Pages from disk" and
// "Needs reconverting" at the body-small size.
const rowSideWidth = 140

// panel is the card beside a list, in Discover's style.
func panel(sh *app.Shell, content ...widget.Widget) widget.Widget {
	p := sh.P
	return primitives.Box(primitives.VBox(content...).Gap(p.Space.S).CrossAlign(primitives.CrossAxisStretch)).
		Padding(p.Space.M+p.Space.XS).
		Rounded(p.Shape.Large).
		BorderStyle(1, p.Colors.OutlineVariant).
		Background(p.Colors.SurfaceContainer)
}

// metric is one labelled figure in a panel.
func metric(sh *app.Shell, label string, value state.ReadonlySignal[string]) widget.Widget {
	p := sh.P
	return primitives.VBox(
		p.Role(primitives.Text(label).Color(p.Muted()), p.Type.LabelMedium),
		widgets.NewParagraph("").ContentSignal(value).FontSize(p.Type.BodyMedium.FontSize).
			Color(p.Text()).Font(sh.P.Prose()).Bold().MaxLines(2),
	).Gap(2)
}

// pair is two metrics side by side.
func pair(sh *app.Shell, a, b widget.Widget) widget.Widget {
	return primitives.HBox(primitives.Expanded(a), primitives.Expanded(b)).Gap(sh.P.Space.M)
}

// listPage is a page of rows beside a panel, under a heading, the whole of it
// scrolling.
func listPage(sh *app.Shell, head, list, side widget.Widget, sideWidth float32) widget.Widget {
	p := sh.P
	body := primitives.VBox(
		head,
		primitives.HBox(
			primitives.Expanded(list),
			primitives.Box(side).Width(sideWidth),
		).Gap(p.Space.M),
	).Gap(p.Space.L).CrossAlign(primitives.CrossAxisStretch).Padding(p.Space.L)
	return primitives.Box(primitives.VBox(
		primitives.Expanded(scrollview.New(body, scrollview.PainterOpt(p.Scrollbar))),
	)).Background(p.Background())
}
