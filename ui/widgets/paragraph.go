package widgets

import (
	"strings"
	"unicode"

	"github.com/gogpu/ui/geometry"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"
)

// charWidthRatio is the toolkit's own estimate of glyph width against font
// size (primitives.estimatedCharWidth), and a monospace face's exact advance.
// It is what a Paragraph wraps by when its Face does not say.
const charWidthRatio float32 = 0.6

// Face is what a Paragraph is set in: the family that draws each rune, and
// the mean advance per rune, in ems, that it wraps by. Layout has no canvas to
// measure with, so the advance is how a line is broken; it must not be under
// the face's real one, because the canvas does not clip text.
type Face struct {
	For     func(r rune, prefer string) string
	Advance float32
	// BoldAdvance is Advance for bold text, which is wider; 0 takes Advance.
	BoldAdvance float32
}

// Paragraph is multi-line text that actually wraps.
//
// It exists because primitives.Text measures a multi-line height but draws the
// whole string as one run, so a long reply runs off the edge. It reads its
// signal in Layout and Draw, as widgets.BlockStrip does; Shell.Post pairs every
// state write with a RequestRedraw, so a streamed reply appears token by token.
type Paragraph struct {
	widget.WidgetBase

	sig    state.ReadonlySignal[string]
	static string

	// colorSig overrides color while it is set, for text whose colour is
	// state (a transcript name that turns the model's colour once known).
	colorSig state.ReadonlySignal[widget.Color]

	fontSize float32
	lineH    float32
	color    widget.Color
	bold     bool
	align    widget.TextAlign
	family   string
	fontFor  func(r rune, prefer string) string
	adv      float32 // ems per rune, see Face
	advBold  float32 // ems per rune when bold, or 0 for adv
	// maxLines caps the lines drawn; 0 is unlimited.
	maxLines int
	// intrinsic measures to the TEXT rather than to the constraint.
	intrinsic bool

	// selectable turns dragging and copying on; sel is the current selection.
	selectable bool
	sel        selection
	selColor   widget.Color
	// drawn is the lines exactly as painted -- wrapped, capped and reordered
	// for display -- and logical the same lines before reordering. The
	// selection indexes drawn, because that is what was clicked; Selected maps
	// back through logical.
	drawn   []string
	logical []string

	// width is the constraint Layout last resolved, so Draw can re-wrap to the
	// same column budget.
	width float32
}

// NewParagraph builds a paragraph over static text.
func NewParagraph(s string) *Paragraph {
	p := &Paragraph{static: s, fontSize: 14, lineH: 1.35, color: widget.ColorBlack, adv: charWidthRatio}
	p.SetVisible(true)
	p.SetEnabled(true)
	return p
}

// Mount binds the content signal so a change invalidates the layout and not
// only the paint. A RequestRedraw alone repaints without re-measuring, so a
// growing reply would keep the height it was first measured at.
func (p *Paragraph) Mount(ctx widget.Context) {
	if p.sig == nil || ctx == nil {
		return
	}
	if sched := ctx.Scheduler(); sched != nil {
		p.AddBinding(state.BindToSchedulerLayout(p.sig, p, sched))
	}
}

// Unmount is the twin widget.Lifecycle requires; the bindings themselves are
// cleaned up by WidgetBase.
func (p *Paragraph) Unmount() {}

// ContentSignal binds the paragraph to a signal, for text that changes while
// it is on screen.
func (p *Paragraph) ContentSignal(sig state.ReadonlySignal[string]) *Paragraph {
	p.sig = sig
	return p
}

// FontSize, LineHeight, Color and Bold mirror primitives.Text's setters so a
// call site reads the same either way.
func (p *Paragraph) FontSize(v float32) *Paragraph       { p.fontSize = v; return p }
func (p *Paragraph) Align(a widget.TextAlign) *Paragraph { p.align = a; return p }
func (p *Paragraph) LineHeight(v float32) *Paragraph     { p.lineH = v; return p }
func (p *Paragraph) Color(c widget.Color) *Paragraph     { p.color = c; return p }

// Intrinsic makes the paragraph as wide as its text instead of as wide as it
// is allowed to be. Layout is greedy by default, and two greedy children in an
// HBox land on top of each other.
//
// It is opt-in rather than implied by MaxLines(1): a right-aligned, Expanded
// single line needs the full width to align within.
func (p *Paragraph) Intrinsic() *Paragraph { p.intrinsic = true; return p }

// ColorSignal binds the colour, for text whose colour changes on screen. It
// needs no scheduler binding: a colour change is a repaint, never a relayout.
func (p *Paragraph) ColorSignal(sig state.ReadonlySignal[widget.Color]) *Paragraph {
	p.colorSig = sig
	return p
}

// colour is the colour to paint with: the signal while one is bound.
func (p *Paragraph) colour() widget.Color {
	if p.colorSig != nil {
		return p.colorSig.Get()
	}
	return p.color
}
func (p *Paragraph) Bold() *Paragraph { p.bold = true; return p }

// FontFamily selects a registered font, which is what makes a script the
// embedded default does not cover render at all. The name must have been
// registered; see app.Fonts.Load.
func (p *Paragraph) FontFamily(name string) *Paragraph { p.family = name; return p }

// MaxLines caps the paragraph at n lines and ends the last one with an
// ellipsis when there is more. primitives.Text.Ellipsis() does not truncate
// the drawn string and the canvas does not clip, so this does it in Draw,
// against the bounds the parent actually gave.
func (p *Paragraph) MaxLines(n int) *Paragraph { p.maxLines = n; return p }

// FontFor supplies a font chain instead of a single family: given a rune and
// the family the current run is already using, it returns the family that
// should draw that rune. A line is then split into runs of consecutive runes
// the same family covers, and each run is drawn with it. Asking "can this font
// draw it" rather than "what script is this" needs no per-script table.
//
// It does not shape: cursive joining needs a GSUB/GPOS shaper, which the text
// stack lacks, so that is not fixed up per script here.
func (p *Paragraph) Font(f Face) *Paragraph {
	p.fontFor = f.For
	if f.Advance > 0 {
		p.adv = f.Advance
	}
	p.advBold = f.BoldAdvance
	return p
}

// run is a stretch of a line drawn with one family.
type run struct {
	text   string
	family string
}

// splitRuns breaks s wherever the chain changes family. With no chain it is one
// run, which is the single-family path unchanged.
func (p *Paragraph) splitRuns(s string) []run {
	if p.fontFor == nil || s == "" {
		return []run{{text: s, family: p.family}}
	}
	var runs []run
	cur := run{family: p.fontFor([]rune(s)[0], p.family)}
	for _, r := range s {
		f := p.fontFor(r, cur.family)
		if f != cur.family {
			runs = append(runs, cur)
			cur = run{family: f}
		}
		cur.text += string(r)
	}
	return append(runs, cur)
}

// style is the paragraph as the styled-text path wants it.
func (p *Paragraph) style() widget.TextStyle { return p.styleFor(p.family) }

func (p *Paragraph) styleFor(family string) widget.TextStyle {
	return widget.TextStyle{
		FontFamily: family,
		FontSize:   p.fontSize,
		Bold:       p.bold,
		Color:      p.colour(),
		Align:      p.align,
	}
}

// cols is how many runes fit in w, measured against the canvas when it can
// measure and estimated when it cannot. The estimate is Latin-shaped and
// underestimates full-width CJK by about half.
func (p *Paragraph) colsFor(w float32, sd widget.StyledTextDrawer) int {
	if w <= 0 || w >= float32(1<<29) {
		return 1 << 30
	}
	if sd != nil {
		// Measure a representative run rather than per-line: one call gives
		// the average advance in this font at this size.
		const probe = "the quick brown fox jumps over the lazy dog"
		if m := sd.MeasureStyledText(probe, p.style()); m > 0 {
			per := m / float32(len([]rune(probe)))
			if per > 0 {
				return int(w / per)
			}
		}
	}
	return int(w / (p.fontSize * p.advance()))
}

// advance is the ems per rune this paragraph wraps by, in its weight.
func (p *Paragraph) advance() float32 {
	if p.bold && p.advBold > 0 {
		return p.advBold
	}
	return p.adv
}

// Content is the text this paragraph is currently showing.
func (p *Paragraph) Content() string {
	if p.sig != nil {
		return p.sig.Get()
	}
	return p.static
}

// Layout wraps at the available width and takes exactly the height the wrapped
// lines need.
func (p *Paragraph) Layout(_ widget.Context, c geometry.Constraints) geometry.Size {
	w := c.MaxWidth
	p.width = w
	lines := p.wrapTo(w, nil)
	if p.maxLines > 0 && len(lines) > p.maxLines {
		lines = lines[:p.maxLines]
	}

	h := float32(len(lines)) * p.fontSize * p.lineH
	if len(lines) == 0 {
		h = 0
	}
	if p.intrinsic || w <= 0 || w >= float32(1<<29) {
		// One character of slack: an estimate a pixel short would wrap the
		// last word in Draw, which under MaxLines(1) becomes an ellipsis.
		w = widest(lines, p.fontSize*p.advance()) + p.fontSize*p.advance()
	}
	sz := c.Constrain(geometry.Sz(w, h))
	p.SetBounds(geometry.FromPointSize(p.Position(), sz))
	return sz
}

// wrapTo breaks the CURRENT content to a width. sd is the canvas when it can
// measure, nil otherwise.
func (p *Paragraph) wrapTo(w float32, sd widget.StyledTextDrawer) []string {
	return WrapText(p.Content(), p.colsFor(w, sd))
}

// widest is the longest line's width at per pixels a rune.
func widest(lines []string, per float32) float32 {
	var n int
	for _, l := range lines {
		if r := len([]rune(l)); r > n {
			n = r
		}
	}
	return float32(n) * per
}

// Draw paints one line per row, each in its own rect so the canvas's vertical
// centring lands on that line rather than on the middle of the whole block.
func (p *Paragraph) Draw(_ widget.Context, canvas widget.Canvas) {
	if !p.IsVisible() {
		return
	}
	r := p.Bounds()
	// Wrapped here from the current content, not reused from Layout:
	// core/listview caches a row's widget, so only Draw runs when a streamed
	// reply grows.
	w := r.Width()
	if w <= 0 {
		w = p.width
	}
	sd, _ := canvas.(widget.StyledTextDrawer)
	// The same estimate Layout used (nil measurer), so the painted line count
	// matches the measured height; Layout has no canvas to measure with.
	lines := p.wrapTo(w, nil)
	if len(lines) == 0 {
		return
	}
	lh := p.fontSize * p.lineH

	// Record the lines as painted, in both orders, for the selection.
	p.drawn = p.drawn[:0]
	p.logical = p.logical[:0]
	for i, line := range lines {
		if p.maxLines > 0 && i >= p.maxLines {
			break
		}
		if p.maxLines > 0 && i == p.maxLines-1 && len(lines) > p.maxLines {
			line = ellipsise(line)
		}
		// A one-line paragraph is a label, so it can afford a real measurement:
		// the estimate is a mean, and a name of capitals and digits is wider
		// than the mean. Only where the line is cut changes, never the line
		// count, so the height Layout gave still holds.
		if p.maxLines == 1 && sd != nil {
			line = fitLine(line, w, func(s string) float32 { return sd.MeasureStyledText(s, p.style()) })
		}
		p.logical = append(p.logical, line)
		p.drawn = append(p.drawn, visualOrder(line))
	}
	p.resolvePending(canvas, r, lh)

	for i, line := range lines {
		if line == "" {
			continue
		}
		if p.maxLines > 0 && i >= len(p.drawn) {
			break
		}
		line = p.drawn[i]
		top := r.Min.Y + float32(i)*lh
		// A row whose height was measured before the reply grew cannot show
		// every line; stop at the bound rather than paint over the turn below.
		if top >= r.Max.Y {
			break
		}
		if p.maxLines > 0 && i >= p.maxLines {
			break
		}
		box := geometry.FromPointSize(geometry.Pt(r.Min.X, top), geometry.Sz(w, lh))
		p.paintSelection(canvas, i, line, box)
		// The styled path reaches registered fonts; plain DrawText only has
		// the embedded default.
		if sd != nil {
			runs := p.splitRuns(line)
			// One run is every line that is not mixed, which is nearly all of
			// them: draw it exactly as a single family would, in one call.
			if len(runs) == 1 && runs[0].family != "" {
				sd.DrawStyledText(line, box, p.styleFor(runs[0].family))
				continue
			}
			if len(runs) > 1 {
				x := r.Min.X
				for _, rn := range runs {
					st := p.styleFor(rn.family)
					// Per-run boxes must not each re-centre their own text.
					st.Align = widget.TextAlignLeft
					rb := geometry.FromPointSize(geometry.Pt(x, top), geometry.Sz(r.Max.X-x, lh))
					sd.DrawStyledText(rn.text, rb, st)
					x += sd.MeasureStyledText(rn.text, st)
				}
				continue
			}
		}
		canvas.DrawText(line, box, p.fontSize, p.colour(), p.bold, p.align)
	}
}

func wrapOne(s string, cols int) []string {
	var out []string
	var line []rune
	flush := func() {
		if len(line) > 0 {
			out = append(out, string(line))
			line = line[:0]
		}
	}
	for _, word := range splitWords(s) {
		w := []rune(word)
		// A word that cannot fit on a line of its own is split hard, which is
		// what stops a 200-character token or a URL from running off.
		for len(w) > cols {
			flush()
			out = append(out, string(w[:cols]))
			w = w[cols:]
		}
		switch {
		case len(line) == 0:
			line = append(line, w...)
		case len(line)+len(w) <= cols:
			line = append(line, w...)
		default:
			flush()
			// A wrapped line never begins with the space that broke it.
			for len(w) > 0 && unicode.IsSpace(w[0]) {
				w = w[1:]
			}
			line = append(line, w...)
		}
	}
	flush()
	if len(out) == 0 {
		return []string{""}
	}
	return out
}

// splitWords keeps each word together with the spaces that follow it, so a
// break lands between words and the spacing inside a line is preserved.
func splitWords(s string) []string {
	var out []string
	r := []rune(s)
	for i := 0; i < len(r); {
		j := i
		for j < len(r) && !unicode.IsSpace(r[j]) {
			j++
		}
		for j < len(r) && unicode.IsSpace(r[j]) {
			j++
		}
		out = append(out, string(r[i:j]))
		i = j
	}
	return out
}

// fitLine cuts line, with an ellipsis, until it measures w or less.
func fitLine(line string, w float32, measure func(string) float32) string {
	if measure(line) <= w {
		return line
	}
	r := []rune(strings.TrimSuffix(line, "\u2026"))
	for len(r) > 1 {
		r = r[:len(r)-1]
		if t := strings.TrimRight(string(r), " ") + "\u2026"; measure(t) <= w {
			return t
		}
	}
	return "\u2026"
}

// ellipsise replaces the tail of a line with a marker, cutting by RUNE so a
// multi-byte character is never left halved.
func ellipsise(s string) string {
	r := []rune(s)
	if len(r) == 0 {
		return "\u2026"
	}
	// One rune of room for the marker, and never an empty result.
	n := len(r) - 1
	if n < 1 {
		n = 1
	}
	return strings.TrimRight(string(r[:n]), " ") + "\u2026"
}

// Children returns nil: a paragraph is a leaf.
func (p *Paragraph) Children() []widget.Widget { return nil }

// WrapText breaks s into lines of at most cols runes, on word boundaries where
// it can and inside a word where it must. It is pure so tests can pin it.
func WrapText(s string, cols int) []string {
	if s == "" {
		return nil
	}
	if cols < 1 {
		cols = 1
	}
	var out []string
	for _, para := range strings.Split(s, "\n") {
		if para == "" {
			out = append(out, "")
			continue
		}
		out = append(out, wrapOne(para, cols)...)
	}
	return out
}
