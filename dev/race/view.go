package main

import (
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

	"github.com/jitllm/jitllm/dev/httpbench/hb"
)

// The palette is Tokyo Night, the website's and tui/'s; the brand is its green.
const (
	cLine   = lipgloss.Color("#3b4261")
	cDim    = lipgloss.Color("#565f89")
	cMuted  = lipgloss.Color("#a9b1d6")
	cFg     = lipgloss.Color("#c0caf5")
	cBrand  = lipgloss.Color("#9ece6a")
	cBlue   = lipgloss.Color("#7aa2f7")
	cYellow = lipgloss.Color("#e0af68")
	cRed    = lipgloss.Color("#f7768e")
)

func sDim() lipgloss.Style   { return lipgloss.NewStyle().Foreground(cDim) }
func sMuted() lipgloss.Style { return lipgloss.NewStyle().Foreground(cMuted) }
func sFg() lipgloss.Style    { return lipgloss.NewStyle().Foreground(cFg) }
func sBold() lipgloss.Style  { return lipgloss.NewStyle().Foreground(cFg).Bold(true) }
func sErr() lipgloss.Style   { return lipgloss.NewStyle().Foreground(cRed) }

// sideColour(): the left side is the brand's, the right the blue.
func sideColour() [2]lipgloss.Color { return [2]lipgloss.Color{cBrand, cBlue} }

// pane is one side's live state.
type pane struct {
	text   strings.Builder
	chunks int
	first  time.Duration // the first text chunk, since the start; 0 before it
	last   time.Duration
	done   *hb.Sample
	doneAt time.Duration
	rank   int // 1 for the first to finish
}

type (
	eventMsg Event
	tickMsg  time.Time
	startMsg struct{}
	quitMsg  struct{}
)

type model struct {
	sides   [2]Side
	header  []string // lines under the title: hardware, model, the setup
	events  chan Event
	start   func() // starts the race; called once, on startMsg
	t0      time.Time
	now     time.Duration
	panes   [2]pane
	ended   bool
	hold    time.Duration
	w, h    int
	results [2]hb.Sample
}

func (m *model) Init() tea.Cmd {
	return tea.Batch(tea.Tick(time.Second, func(time.Time) tea.Msg { return startMsg{} }))
}

func (m *model) next() tea.Cmd {
	return func() tea.Msg {
		e, ok := <-m.events
		if !ok {
			return nil
		}
		return eventMsg(e)
	}
}

func tick() tea.Cmd {
	return tea.Tick(100*time.Millisecond, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func (m *model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.w, m.h = msg.Width, msg.Height
	case tea.KeyMsg:
		if s := msg.String(); s == "q" || s == "ctrl+c" || s == "esc" {
			return m, tea.Quit
		}
	case startMsg:
		m.t0 = time.Now()
		m.start()
		return m, tea.Batch(m.next(), tick())
	case tickMsg:
		if m.ended {
			return m, nil
		}
		m.now = time.Since(m.t0)
		return m, tick()
	case quitMsg:
		return m, tea.Quit
	case eventMsg:
		p := &m.panes[msg.Side]
		if msg.Done != nil {
			p.done, p.doneAt = msg.Done, msg.At
			m.results[msg.Side] = *msg.Done
			other := m.panes[1-msg.Side]
			p.rank = 1
			if other.done != nil {
				p.rank = 2
			}
			if m.panes[0].done != nil && m.panes[1].done != nil {
				m.ended = true
				m.now = time.Since(m.t0)
				return m, tea.Tick(m.hold, func(time.Time) tea.Msg { return quitMsg{} })
			}
		} else {
			if p.chunks == 0 {
				p.first = msg.At
			}
			p.chunks++
			p.last = msg.At
			p.text.WriteString(msg.Text)
		}
		return m, m.next()
	}
	return m, nil
}

func (m *model) View() string {
	w, h := max(m.w, 40), max(m.h, 12)
	title := lipgloss.NewStyle().Foreground(cBrand).Bold(true).Render("jitllm race") +
		sDim().Render("  · same prompt, same model, both started at the same instant")
	lines := []string{title}
	for _, l := range m.header {
		lines = append(lines, sMuted().Render(ansi.Truncate(l, w, "…")))
	}
	head := strings.Join(lines, "\n")

	foot := sDim().Render("waiting for both engines…")
	if m.ended {
		foot = lipgloss.NewStyle().Foreground(cYellow).Bold(true).Render(ansi.Wrap(Summary(m.sides, m.results), w, ""))
	} else if m.t0.IsZero() {
		foot = sDim().Render("ready…")
	}
	ph := h - lipgloss.Height(head) - lipgloss.Height(foot)
	pw := w / 2
	left := m.paneView(0, pw, ph)
	right := m.paneView(1, w-pw, ph)
	return lipgloss.JoinVertical(lipgloss.Left, head, lipgloss.JoinHorizontal(lipgloss.Top, left, right), foot)
}

// paneView draws one side, outer size w x h: name and label, the stats, then
// as much of the streamed text's tail as fits.
func (m *model) paneView(i, w, h int) string {
	p := &m.panes[i]
	c := sideColour()[i]
	box := lipgloss.NewStyle().Border(lipgloss.ThickBorder()).BorderForeground(c).Padding(0, 1)
	if p.done != nil && p.rank == 1 {
		box = box.BorderStyle(lipgloss.DoubleBorder())
	}
	iw := max(10, w-box.GetHorizontalFrameSize())
	ih := max(4, h-box.GetVerticalFrameSize())

	name := lipgloss.NewStyle().Foreground(c).Bold(true).Render(m.sides[i].Name)
	status := sDim().Render("streaming")
	switch {
	case p.done != nil && p.done.Err != "":
		status = sErr().Render("error")
	case p.done != nil:
		status = lipgloss.NewStyle().Foreground(c).Bold(true).Render(fmt.Sprintf("✔ finished (%s)", ordinal(p.rank)))
	case p.chunks == 0 && !m.t0.IsZero():
		status = sDim().Render("prefilling…")
	}
	gap := max(1, iw-lipgloss.Width(name)-lipgloss.Width(status))
	top := name + strings.Repeat(" ", gap) + status
	label := sMuted().Render(ansi.Truncate(m.sides[i].Label, iw, "…"))

	ttft, rate, total, toks, prompt := "—", "—", "—", fmt.Sprint(p.chunks), "—"
	if p.chunks > 0 {
		ttft = msOf(p.first)
	}
	if p.done != nil && p.done.Err == "" {
		ttft = msOf(p.done.TTFT)
		rate = fmt.Sprintf("%.1f tok/s", p.done.DecodeRate())
		total = fmt.Sprintf("%.2f s", p.done.Total.Seconds())
		toks = fmt.Sprint(p.done.CompletionTokens)
		prompt = fmt.Sprint(p.done.PromptTokens)
	} else if !m.t0.IsZero() {
		if r := liveRate(p.chunks, p.first, m.now); r > 0 {
			rate = fmt.Sprintf("%.1f tok/s", r)
		}
		total = fmt.Sprintf("%.1f s", m.now.Seconds())
	}
	stat := func(k, v string) string { return sDim().Render(k+" ") + sBold().Render(v) }
	stats := []string{
		stat("TTFT", ttft) + "   " + stat("decode", rate),
		stat("total", total) + "   " + stat("tokens", toks) + "   " + stat("prompt", prompt),
	}
	if p.done != nil && p.done.Err != "" {
		stats = append(stats, sErr().Render(ansi.Wrap(p.done.Err, iw, "")))
	}
	rule := lipgloss.NewStyle().Foreground(cLine).Render(strings.Repeat("─", iw))

	headLines := append([]string{top, label}, stats...)
	headLines = append(headLines, rule)
	room := max(1, ih-len(headLines))
	body := strings.Split(ansi.Wrap(p.text.String(), iw, ""), "\n")
	if len(body) > room {
		body = body[len(body)-room:]
	}
	content := strings.Join(headLines, "\n") + "\n" + sFg().Render(strings.Join(body, "\n"))
	return box.Width(iw + box.GetHorizontalPadding()).Height(ih).MaxHeight(h).Render(content)
}

func ordinal(n int) string {
	if n == 1 {
		return "1st"
	}
	return "2nd"
}
