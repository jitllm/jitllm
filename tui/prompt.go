package main

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// submitMsg is the prompt box's text, sent with enter.
type submitMsg struct{ text string }

// history is the prompts sent this session, walked with up and down the way
// a shell's is. pos == len(items) is the draft being typed.
type history struct {
	items []string
	pos   int
	draft string
}

func (h *history) add(s string) {
	if n := len(h.items); n == 0 || h.items[n-1] != s {
		h.items = append(h.items, s)
	}
	h.pos, h.draft = len(h.items), ""
}

// step moves by d and returns the text to show, or false at either end.
func (h *history) step(d int, current string) (string, bool) {
	next := h.pos + d
	if next < 0 || next > len(h.items) {
		return "", false
	}
	if h.pos == len(h.items) {
		h.draft = current
	}
	h.pos = next
	if next == len(h.items) {
		return h.draft, true
	}
	return h.items[next], true
}

type promptKeys struct {
	Send, Newline, Complete, Older, Newer, Clear key.Binding
}

// prompt is the box the user types into, with its history and the menu of
// slash commands that opens as one is typed.
type prompt struct {
	ta      textarea.Model
	hist    history
	keys    promptKeys
	focused bool
}

func newPrompt() *prompt {
	k := promptKeys{
		Send:     bind("enter", "send"),
		Newline:  bind("ctrl+j", "newline", "ctrl+j", "alt+enter"),
		Complete: bind("tab", "complete", "tab"),
		Older:    bind("↑", "history", "up"),
		Newer:    bind("↓", "history", "down"),
		Clear:    bind("esc", "clear", "esc"),
	}
	ta := textarea.New()
	ta.Placeholder = "Ask anything…  (/ for commands)"
	ta.ShowLineNumbers = false
	ta.Prompt = ""
	ta.CharLimit = 0
	ta.SetHeight(3)
	ta.KeyMap.InsertNewline = k.Newline
	focused := ta.FocusedStyle
	focused.CursorLine = lipgloss.NewStyle()
	focused.Placeholder = sDim()
	focused.Text = sFg()
	ta.FocusedStyle = focused
	blurred := focused
	blurred.Text = sMuted()
	ta.BlurredStyle = blurred
	return &prompt{ta: ta, keys: k}
}

func (p *prompt) Focus() tea.Cmd { p.focused = true; return p.ta.Focus() }
func (p *prompt) Blur()          { p.focused = false; p.ta.Blur() }
func (p *prompt) SetWidth(w int) { p.ta.SetWidth(w) }
func (p *prompt) Height() int    { return p.ta.Height() }
func (p *prompt) Value() string  { return p.ta.Value() }
func (p *prompt) SetValue(s string) {
	p.ta.SetValue(s)
	p.ta.CursorEnd()
}

// commanding is true while a slash command is being typed.
func (p *prompt) commanding() bool { return strings.HasPrefix(p.ta.Value(), "/") }

func (p *prompt) Update(msg tea.Msg) tea.Cmd {
	k, ok := msg.(tea.KeyMsg)
	if !ok {
		var cmd tea.Cmd
		p.ta, cmd = p.ta.Update(msg)
		return cmd
	}
	switch {
	case key.Matches(k, p.keys.Send):
		text := strings.TrimSpace(p.ta.Value())
		if text == "" {
			return nil
		}
		p.hist.add(text)
		p.ta.Reset()
		return func() tea.Msg { return submitMsg{text} }
	case key.Matches(k, p.keys.Complete) && p.commanding():
		if ms := slashMatches(p.ta.Value()); len(ms) > 0 {
			p.SetValue("/" + ms[0].name + " ")
		}
		return nil
	case key.Matches(k, p.keys.Older) && p.ta.Line() == 0,
		key.Matches(k, p.keys.Newer) && p.ta.Line() == p.ta.LineCount()-1:
		d := -1
		if key.Matches(k, p.keys.Newer) {
			d = 1
		}
		if s, ok := p.hist.step(d, p.ta.Value()); ok {
			p.SetValue(s)
		}
		return nil
	}
	var cmd tea.Cmd
	p.ta, cmd = p.ta.Update(msg)
	return cmd
}

func (p *prompt) View() string { return p.ta.View() }

// menu is the slash commands matching what is typed, or "" when none is
// being typed.
func (p *prompt) menu() string {
	if !p.commanding() {
		return ""
	}
	ms := slashMatches(p.ta.Value())
	if len(ms) == 0 {
		return ""
	}
	var b strings.Builder
	for i, c := range ms {
		name := "/" + c.name
		if c.args != "" {
			name += " " + c.args
		}
		row := fmt.Sprintf(" %-20s %s ", name, c.help)
		if i == 0 {
			b.WriteString(lipgloss.NewStyle().Foreground(cBg).Background(cBrand).Bold(true).Render(row) + sDim().Render(" tab") + "\n")
		} else {
			b.WriteString(sFg().Render(row) + "\n")
		}
	}
	return sCard().BorderForeground(cBrand).Background(cPanel).Render(strings.TrimRight(b.String(), "\n"))
}
