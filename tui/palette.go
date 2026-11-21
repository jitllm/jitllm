package main

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jitllm/jitllm/common/engine"
	"github.com/sahilm/fuzzy"
)

// action is one entry of the command palette.
type action struct {
	group, title, keys string
	run                func(r *root) tea.Cmd
}

// closePaletteMsg closes the palette, after its action has run.
type closePaletteMsg struct{}

type paletteKeys struct{ Up, Down, Run, Close key.Binding }

// palette is the ctrl+k overlay: every action, fuzzy-filtered as you type.
type palette struct {
	in    textinput.Model
	keys  paletteKeys
	all   []action
	shown []int
	sel   int
}

const paletteRows = 12

func newPalette(r *root) *palette {
	in := textinput.New()
	in.Placeholder = "Type a command…"
	in.Prompt = sBrand().Render("❯ ")
	in.PlaceholderStyle = sDim()
	in.TextStyle = sFg()
	in.Cursor.Style = sBrand()
	p := &palette{in: in, all: actions(r), keys: paletteKeys{
		Up:    bind("↑", "up", "up", "ctrl+p"),
		Down:  bind("↓", "down", "down", "ctrl+n"),
		Run:   bind("enter", "run"),
		Close: bind("esc", "close", "esc", "ctrl+k"),
	}}
	p.filter()
	return p
}

func (p *palette) Focus() tea.Cmd { return p.in.Focus() }

func actions(r *root) []action {
	st := r.env.f.st
	go2 := func(s screen) func(*root) tea.Cmd { return func(*root) tea.Cmd { return show(s) } }
	as := []action{
		{"go", "Chat", "f1", go2(scrChat)},
		{"go", "Models", "f2", go2(scrModels)},
		{"go", "Discover models to download", "f3", go2(scrDiscover)},
		{"go", "Engine cockpit", "f4", go2(scrEngine)},
		{"go", "Settings", "f5", go2(scrSettings)},
		{"engine", r.env.badge(), "", go2(scrSettings)},
		{"chat", "New chat", "ctrl+n", func(r *root) tea.Cmd { return tea.Batch(r.chat.newChat(), show(scrChat)) }},
		{"chat", "Previous chat", "ctrl/⌥+↑", func(r *root) tea.Cmd { return tea.Batch(r.chat.step(-1), show(scrChat)) }},
		{"chat", "Next chat", "ctrl/⌥+↓", func(r *root) tea.Cmd { return tea.Batch(r.chat.step(1), show(scrChat)) }},
		{"chat", "Delete this chat", "/delete", func(r *root) tea.Cmd { return tea.Batch(r.chat.deleteChat(), show(scrChat)) }},
		{"chat", "Copy last reply", "ctrl+y", func(r *root) tea.Cmd { return r.chat.copyReply() }},
		{"chat", "Show or fold reasoning", "ctrl+t", func(r *root) tea.Cmd {
			st.ShowThinking.Set(!st.ShowThinking.Get())
			return nil
		}},
	}
	if r.env.cfg.Chat {
		as = append(as, action{"chat", "Switch to raw completion", "/completion", func(r *root) tea.Cmd { return setChat(r.env, false) }})
	} else {
		as = append(as, action{"chat", "Switch to the chat template", "/chat", func(r *root) tea.Cmd { return setChat(r.env, true) }})
	}
	if st.Busy.Get() {
		as = append(as, action{"chat", "Stop the reply", "esc", func(r *root) tea.Cmd { r.env.e.Stop(); return nil }})
	}
	for _, lm := range st.Models.Get() {
		name := strings.TrimSuffix(lm.Name, ".jlm")
		path := lm.Path
		if !lm.Active {
			as = append(as, action{"model", "Chat with " + name, "", func(*root) tea.Cmd { return load(path) }})
		}
		as = append(as, action{"model", "Unload " + name, "", func(r *root) tea.Cmd { r.env.e.Unload(path); return nil }})
	}
	if n := len(st.BlockMap.Get()); n > 0 {
		as = append(as,
			action{"engine", "Every block on the device", "end", func(*root) tea.Cmd { return seamTo(n) }},
			action{"engine", "Every block on the host", "home", func(*root) tea.Cmd { return seamTo(0) }},
			action{"engine", fmt.Sprintf("Half on the device (%d)", n/2), "", func(*root) tea.Cmd { return seamTo(n / 2) }},
		)
	}
	if rem, ok := r.env.e.(*engine.Remote); ok && r.env.where.Lost {
		as = append(as, action{"engine", "Reconnect to jitllmd at " + r.env.where.Addr, "", func(*root) tea.Cmd { rem.Reconnect(); return nil }})
	}
	return append(as, action{"app", "Quit", "ctrl+c", func(r *root) tea.Cmd { return r.quit() }})
}

func (p *palette) filter() {
	q := strings.TrimSpace(p.in.Value())
	p.shown = p.shown[:0]
	if q == "" {
		for i := range p.all {
			p.shown = append(p.shown, i)
		}
	} else {
		titles := make([]string, len(p.all))
		for i, a := range p.all {
			titles[i] = a.group + " " + a.title
		}
		for _, m := range fuzzy.Find(q, titles) {
			p.shown = append(p.shown, m.Index)
		}
	}
	p.sel = min(p.sel, max(0, len(p.shown)-1))
}

func (p *palette) Update(r *root, msg tea.Msg) tea.Cmd {
	k, ok := msg.(tea.KeyMsg)
	if !ok {
		var cmd tea.Cmd
		p.in, cmd = p.in.Update(msg)
		return cmd
	}
	switch {
	case key.Matches(k, p.keys.Close):
		return func() tea.Msg { return closePaletteMsg{} }
	case key.Matches(k, p.keys.Up):
		p.sel = max(0, p.sel-1)
		return nil
	case key.Matches(k, p.keys.Down):
		p.sel = min(len(p.shown)-1, p.sel+1)
		return nil
	case key.Matches(k, p.keys.Run):
		closing := func() tea.Msg { return closePaletteMsg{} }
		if len(p.shown) == 0 {
			return closing
		}
		return tea.Sequence(closing, p.all[p.shown[p.sel]].run(r))
	}
	var cmd tea.Cmd
	p.in, cmd = p.in.Update(msg)
	p.filter()
	return cmd
}

func (p *palette) View(w int) string {
	w = min(72, w-8)
	frame := sCard().GetHorizontalFrameSize()
	iw := w - frame
	var b strings.Builder
	b.WriteString(p.in.View() + "\n")
	b.WriteString(lipgloss.NewStyle().Foreground(cLine).Render(strings.Repeat("─", iw)) + "\n")
	from := max(0, p.sel-paletteRows+1)
	for i := from; i < min(len(p.shown), from+paletteRows); i++ {
		a := p.all[p.shown[i]]
		keys := a.keys
		title := truncate(a.title, iw-8-lipgloss.Width(keys)-2)
		pad := strings.Repeat(" ", max(1, iw-8-lipgloss.Width(title)-lipgloss.Width(keys)))
		if i == p.sel {
			b.WriteString(lipgloss.NewStyle().Foreground(cBg).Background(cBrand).Bold(true).Width(iw).
				Render(fmt.Sprintf("%-8s%s%s%s", a.group, title, pad, keys)) + "\n")
		} else {
			b.WriteString(lipgloss.NewStyle().Width(8).Foreground(cDim).Render(a.group) + sFg().Render(title) + pad + sDim().Render(keys) + "\n")
		}
	}
	if len(p.shown) == 0 {
		b.WriteString(sDim().Render("nothing matches") + "\n")
	}
	return sCard().BorderForeground(cBrand).Background(cPanel).Width(w - sCard().GetHorizontalBorderSize()).
		Render(strings.TrimRight(b.String(), "\n"))
}
