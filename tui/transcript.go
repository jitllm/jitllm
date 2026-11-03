package main

import (
	"fmt"
	"path/filepath"
	"strings"

	"charm.land/glamour/v2"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
	"github.com/jitllm/jitllm/common/session"
)

// transcript is the conversation in a viewport. Each finished turn is
// rendered once and kept; only the reply still streaming is drawn again, and
// the viewport's content is replaced only when something in it changed.
type transcript struct {
	env *env
	vp  viewport.Model
	md  markdown

	// done holds each finished turn's rendering, keyed by what it was
	// rendered from, so a turn that has not changed is never rendered twice.
	done map[int]rendered
	// liveText is the streaming reply's Markdown as last rendered.
	liveText, liveOut string
	// empty is true while the transcript shows the logo instead of turns.
	empty bool
}

type rendered struct {
	key, out string
}

func newTranscript(env *env) *transcript {
	vp := viewport.New(80, 20)
	vp.KeyMap = viewport.KeyMap{
		PageUp:       bind("pgup", "scroll up", "pgup"),
		PageDown:     bind("pgdn", "scroll down", "pgdown"),
		HalfPageUp:   key.NewBinding(key.WithKeys("ctrl+u")),
		HalfPageDown: key.NewBinding(key.WithKeys("ctrl+d")),
	}
	return &transcript{env: env, vp: vp, done: map[int]rendered{}}
}

func (t *transcript) SetSize(w, h int) {
	if w != t.vp.Width {
		t.done = map[int]rendered{}
		t.liveText = ""
	}
	t.vp.Width, t.vp.Height = w, h
	t.refresh()
}

// reset drops every rendering, for a chat that is not the one shown.
func (t *transcript) reset() {
	t.done = map[int]rendered{}
	t.liveText = ""
	t.refresh()
	t.vp.GotoBottom()
}

func (t *transcript) Update(msg tea.Msg) tea.Cmd {
	switch msg := msg.(type) {
	case engineMsg:
		if msg.turns || msg.changed.has(fStream, fStreaming, fProblem, fLoads, fLoaded) {
			t.refresh()
		}
	case animMsg:
		if t.empty || t.env.f.st.Streaming.Get() {
			t.refresh()
		}
	case tea.KeyMsg, tea.MouseMsg:
		var cmd tea.Cmd
		t.vp, cmd = t.vp.Update(msg)
		return cmd
	}
	return nil
}

func (t *transcript) Animating() bool { return t.empty || t.env.f.st.Streaming.Get() }

// refresh rebuilds the content from the cached turns and the live reply,
// keeping the view at the bottom when it was there.
func (t *transcript) refresh() {
	follow := t.vp.AtBottom() || t.vp.TotalLineCount() <= t.vp.Height
	t.vp.SetContent(t.content())
	if follow {
		t.vp.GotoBottom()
	}
}

func (t *transcript) content() string {
	st := t.env.f.st
	turns, open, alert := t.env.f.snapshot()
	w := max(20, t.vp.Width)
	t.empty = len(turns) == 0
	if t.empty {
		return t.emptyState(w)
	}
	var b strings.Builder
	showThink := st.ShowThinking.Get()
	streaming := st.Streaming.Get()
	for i, turn := range turns {
		// A reply stopped before its first token is nothing to read; the
		// template drops it too (session.ChatMessages).
		if turn.Role == session.RoleAssistant && turn.Text == "" && turn.Think == "" &&
			!(i == len(turns)-1 && streaming) {
			continue
		}
		live := i == len(turns)-1 && streaming && turn.Role == session.RoleAssistant && turn.Text == "" && turn.Think == ""
		if live {
			b.WriteString(t.live(turn, w))
		} else {
			k := fmt.Sprintf("%d|%s|%s|%d|%v|%v|%d", turn.Role, turn.Text, turn.Think, turn.Tokens, open[i], showThink, w)
			r, ok := t.done[i]
			if !ok || r.key != k {
				r = rendered{k, t.turn(turn, open[i] || showThink, w)}
				t.done[i] = r
			}
			b.WriteString(r.out)
		}
		b.WriteString("\n\n")
	}
	for i := range t.done {
		if i >= len(turns) {
			delete(t.done, i)
		}
	}
	if p := st.Problem.Get(); p.Seq != 0 {
		b.WriteString(sCard().BorderForeground(cRed).Width(w-2).Render(sErr().Bold(true).Render(p.Title)+"\n"+sFg().Render(p.Detail)) + "\n")
	}
	if alert != "" {
		b.WriteString(sErr().Render(alert) + "\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

func (t *transcript) turn(turn session.Turn, thinkOpen bool, w int) string {
	switch turn.Role {
	case session.RoleUser:
		text := lipgloss.NewStyle().Width(min(w*3/4, lipgloss.Width(turn.Text)+sBubble().GetHorizontalFrameSize())).
			Inherit(sBubble()).Render(turn.Text)
		return lipgloss.PlaceHorizontal(w, lipgloss.Right, sDim().Render("you")) + "\n" +
			lipgloss.PlaceHorizontal(w, lipgloss.Right, text)
	case session.RoleError:
		return sCard().BorderForeground(cRed).Foreground(cRed).Width(w - 2).Render(turn.Text)
	}
	var body strings.Builder
	if turn.Think != "" {
		if thinkOpen {
			body.WriteString(sThink().Render("▾ thinking") + "\n")
			body.WriteString(sThink().Width(w-4).Render(turn.Think) + "\n\n")
		} else {
			body.WriteString(sThink().Render(fmt.Sprintf("▸ thought for %d words · ctrl+t", len(strings.Fields(turn.Think)))) + "\n")
		}
	}
	if turn.Text != "" {
		body.WriteString(t.md.render(turn.Text, w-3))
	} else {
		body.WriteString(sDim().Render("(no reply)"))
	}
	if turn.Tokens > 0 {
		body.WriteString("\n" + sDim().Render(fmt.Sprintf("%d tokens · %.1f tok/s", turn.Tokens, turn.TokPerSec)))
	}
	return t.head(turn, false) + "\n" + t.rule(turn).Render(strings.TrimRight(body.String(), "\n"))
}

// live is the reply still streaming: its Markdown rendered again only when
// the text grew, and a cursor that blinks on the frame clock.
func (t *transcript) live(turn session.Turn, w int) string {
	st := t.env.f.st
	think, text := st.StreamThink.Get(), st.Stream.Get()
	var body strings.Builder
	if think != "" && (text == "" || st.ShowThinking.Get()) {
		body.WriteString(sThink().Render("▾ thinking") + "\n")
		body.WriteString(sThink().Width(w-4).Render(think) + "\n\n")
	} else if think != "" {
		body.WriteString(sThink().Render(fmt.Sprintf("▸ thought for %d words · ctrl+t", len(strings.Fields(think)))) + "\n")
	}
	if text != "" {
		if text != t.liveText {
			t.liveText, t.liveOut = text, t.md.render(text, w-3)
		}
		body.WriteString(t.liveOut)
		if t.env.frame%8 < 4 {
			body.WriteString(lipgloss.NewStyle().Foreground(colourOf(turn.Colour, modelColours())).Render("▍"))
		}
	} else if think == "" {
		body.WriteString(sDim().Render(strings.Repeat("·", 1+t.env.frame/3%3)))
	}
	return t.head(turn, true) + "\n" + t.rule(turn).Render(body.String())
}

func (t *transcript) head(turn session.Turn, live bool) string {
	name := strings.TrimSuffix(turn.Model, ".jlm")
	if name == "" {
		name = "assistant"
	}
	h := lipgloss.NewStyle().Foreground(colourOf(turn.Colour, modelColours())).Bold(true).Render("● " + name)
	if live {
		h += " " + t.env.spin.View()
	}
	return h
}

func (t *transcript) rule(turn session.Turn) lipgloss.Style {
	return lipgloss.NewStyle().Border(lipgloss.ThickBorder(), false, false, false, true).
		BorderForeground(colourOf(turn.Colour, modelColours())).PaddingLeft(1)
}

func (t *transcript) emptyState(w int) string {
	st := t.env.f.st
	var hint string
	switch {
	case len(st.Loads.Get()) > 0:
		l := st.Loads.Get()[0]
		hint = t.env.spin.View() + " " + sMuted().Render("loading "+filepath.Base(l.Path)+" · "+l.Stage)
	case st.Loaded.Get():
		hint = sMuted().Render("ready. ask anything, or type / for commands.")
	default:
		hint = sMuted().Render("f2 to pick a model · ctrl+k for every command")
	}
	tag := sDim().Render("every kernel generated at run time for the machine in front of it")
	block := lipgloss.JoinVertical(lipgloss.Center, wordmark(1, t.env.frame*2), "", tag, "", hint)
	return lipgloss.Place(w, max(t.vp.Height, lipgloss.Height(block)), lipgloss.Center, lipgloss.Center, block)
}

// markdown renders a reply through glamour, in the window's palette.
// glamour v1 overflowed a line by one cell per hyphen in it (reflow's word
// wrap writes a breakpoint without counting it), and its second wrap then
// stranded a word on a line of its own; v2 wraps with lipgloss.
type markdown struct {
	width int
	r     *glamour.TermRenderer
}

func (md *markdown) render(text string, w int) string {
	if md.r == nil || md.width != w {
		r, err := glamour.NewTermRenderer(glamour.WithStandardStyle("tokyo-night"), glamour.WithWordWrap(w))
		if err != nil {
			return text
		}
		md.r, md.width = r, w
	}
	out, err := md.r.Render(text)
	if err != nil {
		return text
	}
	// glamour indents every line by its document margin; the rule beside the
	// reply is the margin here. The margin sits behind the line's colour
	// codes, so it is cut by cells, not by bytes.
	lines := strings.Split(strings.Trim(out, "\n"), "\n")
	for i, l := range lines {
		if strings.HasPrefix(ansi.Strip(l), "  ") {
			lines[i] = ansi.TruncateLeft(l, 2, "")
		}
	}
	return strings.Join(lines, "\n")
}
