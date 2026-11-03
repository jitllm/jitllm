package main

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// The palette is Tokyo Night, the website's and the window's; the brand is
// its green.
const (
	cBg      = lipgloss.Color("#1a1b26")
	cPanel   = lipgloss.Color("#1f2335")
	cRaised  = lipgloss.Color("#292e42")
	cLine    = lipgloss.Color("#3b4261")
	cDim     = lipgloss.Color("#565f89")
	cMuted   = lipgloss.Color("#a9b1d6")
	cFg      = lipgloss.Color("#c0caf5")
	cBrand   = lipgloss.Color("#9ece6a")
	cBlue    = lipgloss.Color("#7aa2f7")
	cCyan    = lipgloss.Color("#7dcfff")
	cMagenta = lipgloss.Color("#bb9af7")
	cOrange  = lipgloss.Color("#ff9e64")
	cYellow  = lipgloss.Color("#e0af68")
	cRed     = lipgloss.Color("#f7768e")
)

// modelColours() is the transcript's palette, indexed by LoadedModel.Colour so
// each model keeps its colour for the session.
func modelColours() []lipgloss.Color {
	return []lipgloss.Color{cBrand, cBlue, cMagenta, cCyan, cOrange, cYellow}
}

// deviceColours() paints a block by the device that runs it.
func deviceColours() []lipgloss.Color {
	return []lipgloss.Color{cCyan, cMagenta, cYellow, cBlue, cOrange}
}

func colourOf(i int, from []lipgloss.Color) lipgloss.Color {
	return from[((i%len(from))+len(from))%len(from)]
}

func sDim() lipgloss.Style { return lipgloss.NewStyle().Foreground(cDim) }

func sMuted() lipgloss.Style { return lipgloss.NewStyle().Foreground(cMuted) }

func sFg() lipgloss.Style { return lipgloss.NewStyle().Foreground(cFg) }

func sBold() lipgloss.Style { return lipgloss.NewStyle().Foreground(cFg).Bold(true) }

func sBrand() lipgloss.Style { return lipgloss.NewStyle().Foreground(cBrand).Bold(true) }

func sErr() lipgloss.Style { return lipgloss.NewStyle().Foreground(cRed) }

func sThink() lipgloss.Style { return lipgloss.NewStyle().Foreground(cDim).Italic(true) }

func sPanel() lipgloss.Style {
	return lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(cLine).Padding(0, 1)
}

func sBubble() lipgloss.Style {
	return lipgloss.NewStyle().Background(cRaised).Foreground(cFg).Padding(0, 1)
}

func sCard() lipgloss.Style {
	return lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(cLine).Padding(0, 1)
}

// inner is the content size of a panel w x h cells outside.
func inner(w, h int) (int, int) {
	return max(1, w-sPanel().GetHorizontalFrameSize()), max(1, h-sPanel().GetVerticalFrameSize())
}

// panel frames body in exactly w x h cells (h 0 fits the body), with title
// set into the top border in accent. The body is cut to fit, never wrapped
// past it.
func panel(title, body string, w, h int, accent lipgloss.Color) string {
	iw, ih := inner(w, h)
	body = fitWidth(body, iw)
	st := sPanel().Width(w - sPanel().GetHorizontalBorderSize())
	if h > 0 {
		body = fitHeight(body, ih)
		st = st.Height(h - sPanel().GetVerticalBorderSize())
	}
	out := st.Render(body)
	if title == "" {
		return out
	}
	top, rest, _ := strings.Cut(out, "\n")
	label := " " + lipgloss.NewStyle().Foreground(accent).Bold(true).Render(ansi.Truncate(title, w-6, "…")) + " "
	border := lipgloss.NewStyle().Foreground(cLine)
	fill := max(0, lipgloss.Width(top)-3-lipgloss.Width(label))
	return border.Render("╭─") + label + border.Render(strings.Repeat("─", fill)+"╮") + "\n" + rest
}

// fitWidth cuts every line of s to w cells.
func fitWidth(s string, w int) string {
	lines := strings.Split(s, "\n")
	for i, l := range lines {
		if lipgloss.Width(l) > w {
			lines[i] = ansi.Truncate(l, w, "…")
		}
	}
	return strings.Join(lines, "\n")
}

// fitHeight keeps the first h lines of s.
func fitHeight(s string, h int) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) > h {
		lines = lines[:h]
	}
	return strings.Join(lines, "\n")
}

// pill is a bold label on a coloured ground.
func pill(text string, fg, bg lipgloss.Color) string {
	return lipgloss.NewStyle().Foreground(fg).Background(bg).Bold(true).Padding(0, 1).Render(text)
}

// chip is a quiet tag: coloured text on the raised ground.
func chip(text string, fg lipgloss.Color) string {
	return lipgloss.NewStyle().Foreground(fg).Background(cRaised).Padding(0, 1).Render(text)
}

func truncate(s string, w int) string {
	if w <= 1 || lipgloss.Width(s) <= w {
		return s
	}
	return ansi.Truncate(s, w, "…")
}

// kvLine is a label at the left and its value at the right of w cells.
func kvLine(k, v string, w int) string {
	v = truncate(v, max(4, w-lipgloss.Width(k)-1))
	return sDim().Render(k) + strings.Repeat(" ", max(1, w-lipgloss.Width(k)-lipgloss.Width(v))) + sFg().Render(v)
}

// meter is a static bar of w cells at frac, the progress bubble's look
// without its animation.
func meter(frac float64, w int, from, to lipgloss.Color) string {
	p := progress.New(progress.WithGradient(string(from), string(to)), progress.WithoutPercentage(),
		progress.WithWidth(w), progress.WithFillCharacters('━', '━'))
	p.EmptyColor = string(cLine)
	return p.ViewAs(min(max(frac, 0), 1))
}

func lerp(a, b lipgloss.Color, t float64) lipgloss.Color {
	var ar, ag, ab, br, bg, bb int
	fmt.Sscanf(string(a), "#%02x%02x%02x", &ar, &ag, &ab)
	fmt.Sscanf(string(b), "#%02x%02x%02x", &br, &bg, &bb)
	mix := func(x, y int) int { return x + int(float64(y-x)*t+0.5) }
	return lipgloss.Color(fmt.Sprintf("#%02x%02x%02x", mix(ar, br), mix(ag, bg), mix(ab, bb)))
}

// overlay draws fg over bg with its top-left corner at x, y. Bubble Tea
// renders one string, so a modal is a splice of the two.
func overlay(bg, fg string, x, y int) string {
	bl := strings.Split(bg, "\n")
	for i, l := range strings.Split(fg, "\n") {
		row := y + i
		if row < 0 || row >= len(bl) {
			continue
		}
		base := bl[row]
		left := ansi.Truncate(base, x, "")
		if pad := x - lipgloss.Width(left); pad > 0 {
			left += strings.Repeat(" ", pad)
		}
		bl[row] = left + l + ansi.TruncateLeft(base, x+lipgloss.Width(l), "")
	}
	return strings.Join(bl, "\n")
}

// dim fades s to the border colour, the ground a modal sits on.
func dim(s string) string {
	lines := strings.Split(ansi.Strip(s), "\n")
	st := lipgloss.NewStyle().Foreground(cLine)
	for i, l := range lines {
		lines[i] = st.Render(l)
	}
	return strings.Join(lines, "\n")
}
