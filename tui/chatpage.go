package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/atotto/clipboard"
	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/jitllm/jitllm/common/session"
	"github.com/muesli/termenv"
)

const (
	chatsW = 28 // the chat list's outer width, when the terminal has room
	sideW  = 36 // the engine sidebar's
)

type chatKeys struct {
	Stop, Think, Copy, New, Prev, Next, Scroll key.Binding
}

// chatPage is the conversation: the saved chats at the left (the window's
// chats.json, so a chat started in one front end is in the other's list),
// the transcript and the prompt in the middle, the engine at the right.
type chatPage struct {
	env    *env
	keys   chatKeys
	tr     *transcript
	prompt *prompt
	chats  session.ChatList

	w, h             int
	listW, sideW     int
	wasBusy          bool
	promptW, promptH int
}

func newChatPage(env *env) *chatPage {
	c := &chatPage{
		env: env,
		keys: chatKeys{
			Stop:   bind("esc", "stop · clear"),
			Think:  bind("ctrl+t", "thinking"),
			Copy:   bind("ctrl+y", "copy reply"),
			New:    bind("ctrl+n", "new chat"),
			Prev:   bind("ctrl/⌥+↑", "prev chat", "ctrl+up", "alt+up"),
			Next:   bind("ctrl/⌥+↓", "next chat", "ctrl+down", "alt+down"),
			Scroll: bind("pgup/pgdn", "scroll", "pgup", "pgdown", "ctrl+u", "ctrl+d"),
		},
		tr:     newTranscript(env),
		prompt: newPrompt(),
	}
	c.chats = session.LoadChats(env.cfg.ChatsPath())
	c.show(0)
	return c
}

func (c *chatPage) Help() []key.Binding {
	return []key.Binding{c.prompt.keys.Send, c.keys.Stop, c.prompt.keys.Older, c.keys.Copy, c.keys.New, c.keys.Prev, c.keys.Next, c.keys.Think, c.keys.Scroll}
}

func (c *chatPage) Focus() tea.Cmd  { return c.prompt.Focus() }
func (c *chatPage) Blur()           { c.prompt.Blur() }
func (c *chatPage) Capturing() bool { return c.prompt.commanding() }
func (c *chatPage) Animating() bool { return c.tr.Animating() }

func (c *chatPage) SetSize(w, h int) {
	c.w, c.h = w, h
	c.listW, c.sideW = 0, 0
	if w >= 150 {
		c.listW = chatsW
	}
	if w >= 110 {
		c.sideW = sideW
	}
	mid := w - c.listW - c.sideW
	c.promptW = mid - sCard().GetHorizontalFrameSize()
	c.prompt.SetWidth(c.promptW)
	c.promptH = c.prompt.Height() + sCard().GetVerticalFrameSize()
	iw, ih := inner(mid, h-c.promptH)
	c.tr.SetSize(iw, ih)
}

func (c *chatPage) Update(msg tea.Msg) tea.Cmd {
	st := c.env.f.st
	switch msg := msg.(type) {
	case submitMsg:
		return c.submit(msg.text)
	case engineMsg:
		cmd := c.tr.Update(msg)
		if msg.turns {
			c.park()
		}
		// A reply that just finished is saved with its chat.
		if busy := st.Busy.Get(); msg.changed.has(fBusy) {
			if c.wasBusy && !busy {
				c.save()
			}
			c.wasBusy = busy
		}
		return cmd
	case animMsg:
		return c.tr.Update(msg)
	case tea.MouseMsg:
		return c.tr.Update(msg)
	case tea.KeyMsg:
		switch {
		case key.Matches(msg, c.keys.Stop):
			if st.Busy.Get() {
				c.env.e.Stop()
				return notify("stopped")
			}
			c.prompt.SetValue("")
			return nil
		case key.Matches(msg, c.keys.Think):
			st.ShowThinking.Set(!st.ShowThinking.Get())
			return nil
		case key.Matches(msg, c.keys.Copy):
			return c.copyReply()
		case key.Matches(msg, c.keys.New):
			return c.newChat()
		case key.Matches(msg, c.keys.Prev):
			return c.step(-1)
		case key.Matches(msg, c.keys.Next):
			return c.step(1)
		case key.Matches(msg, c.keys.Scroll):
			return c.tr.Update(msg)
		}
		return c.prompt.Update(msg)
	}
	return c.prompt.Update(msg)
}

// submit sends the prompt, or runs it as a command when it starts with /.
func (c *chatPage) submit(text string) tea.Cmd {
	if strings.HasPrefix(text, "/") {
		return c.slash(text)
	}
	st := c.env.f.st
	switch {
	case len(st.Loads.Get()) > 0:
		c.prompt.SetValue(text)
		return notify("still loading, one moment")
	case !st.Loaded.Get():
		c.prompt.SetValue(text)
		return notify("no model yet: f2 to pick one")
	case st.Busy.Get():
		c.prompt.SetValue(text)
		return notify("busy: esc stops the reply")
	}
	c.send(text)
	return nil
}

// send starts a reply to text, the way the window's session screen builds
// its request.
func (c *chatPage) send(text string) {
	env := c.env
	hist, _, _ := env.f.snapshot()
	colour, name := activeModel(env)
	env.f.appendTurn(session.Turn{Role: session.RoleUser, Text: text})
	req := session.ChatRequest{
		Chat:      env.cfg.Chat,
		Prompt:    text,
		Reply:     env.f.appendTurn(session.Turn{Role: session.RoleAssistant, Model: name, Colour: colour}),
		MaxTokens: session.DefaultMaxTokens,
		Sampling:  env.cfg.Sampling,
	}
	if env.cfg.Chat {
		req.Messages = session.ChatMessages(env.cfg.System, hist, text)
	}
	if p := env.f.st.Active.Get(); p != "" {
		c.chats.SetModel(p)
	}
	c.tr.vp.GotoBottom()
	env.e.Send(req)
}

func activeModel(env *env) (int, string) {
	for _, lm := range env.f.st.Models.Get() {
		if lm.Active {
			return lm.Colour, lm.Name
		}
	}
	return 0, ""
}

// park copies the transcript and draft into the current chat.
func (c *chatPage) park() {
	turns, _, _ := c.env.f.snapshot()
	c.chats.Park(turns, c.prompt.Value())
}

// show makes chat i the transcript, and brings back the model it was
// talking to when that model is open.
func (c *chatPage) show(i int) {
	ch := c.chats.Show(i)
	c.env.f.setTurns(ch.Turns)
	c.prompt.SetValue(ch.Draft)
	c.tr.reset()
	if ch.Model != "" && ch.Model != c.env.f.st.Active.Get() && isOpen(c.env, ch.Model) {
		c.env.e.Use(ch.Model)
	}
}

func isOpen(env *env, path string) bool {
	for _, lm := range env.f.st.Models.Get() {
		if lm.Path == path {
			return true
		}
	}
	return false
}

func (c *chatPage) save() tea.Cmd {
	c.park()
	if err := c.chats.Save(c.env.cfg.ChatsPath()); err != nil {
		return notify("chats not saved: " + err.Error())
	}
	return nil
}

func (c *chatPage) newChat() tea.Cmd {
	if c.env.f.st.Busy.Get() {
		c.env.e.Stop()
	}
	c.park()
	c.show(c.chats.New(c.env.f.st.Active.Get()))
	return tea.Batch(c.save(), notify("new chat"))
}

// step moves d chats down the list.
func (c *chatPage) step(d int) tea.Cmd {
	if c.env.f.st.Busy.Get() {
		return notify("busy: esc stops the reply first")
	}
	c.park()
	if i, ok := c.chats.Select(c.chats.Cur + d); ok {
		c.show(i)
	}
	return nil
}

func (c *chatPage) deleteChat() tea.Cmd {
	if c.env.f.st.Busy.Get() {
		c.env.e.Stop()
	}
	c.park()
	if i, ok := c.chats.Delete(c.chats.Cur); ok {
		c.show(i)
		return tea.Batch(c.save(), notify("chat deleted"))
	}
	return nil
}

func (c *chatPage) copyReply() tea.Cmd {
	turns, _, _ := c.env.f.snapshot()
	for i := len(turns) - 1; i >= 0; i-- {
		if t := turns[i]; t.Role == session.RoleAssistant && t.Text != "" {
			// The system clipboard when there is one, and the terminal's
			// (OSC 52) as well, which reaches the desktop over ssh.
			clipboard.WriteAll(t.Text)
			termenv.NewOutput(os.Stdout).Copy(t.Text)
			return notify(fmt.Sprintf("copied %d characters", len(t.Text)))
		}
	}
	return notify("nothing to copy yet")
}

// slashCmd is one command the prompt box takes.
type slashCmd struct {
	name, args, help string
	run              func(c *chatPage, arg string) tea.Cmd
}

func slashCmds() []slashCmd {
	return []slashCmd{
		{"new", "", "start a new chat", func(c *chatPage, _ string) tea.Cmd { return c.newChat() }},
		{"delete", "", "delete this chat", func(c *chatPage, _ string) tea.Cmd { return c.deleteChat() }},
		{"models", "", "the models screen", func(*chatPage, string) tea.Cmd { return show(scrModels) }},
		{"engine", "", "the engine cockpit", func(*chatPage, string) tea.Cmd { return show(scrEngine) }},
		{"chat", "", "use the chat template", func(c *chatPage, _ string) tea.Cmd { return setChat(c.env, true) }},
		{"completion", "", "raw completion, no template", func(c *chatPage, _ string) tea.Cmd { return setChat(c.env, false) }},
		{"system", "<text>", "set the system prompt", func(c *chatPage, a string) tea.Cmd {
			c.env.cfg.System = a
			return tea.Batch(saveConfig(c.env), notify("system prompt set"))
		}},
		{"temp", "<value>", "sampling temperature", func(c *chatPage, a string) tea.Cmd {
			v, err := strconv.ParseFloat(a, 32)
			if err != nil || v < 0 {
				return notify("usage: /temp 0.7")
			}
			c.env.cfg.Sampling.Temp = float32(v)
			return tea.Batch(saveConfig(c.env), notify(fmt.Sprintf("temperature %.2f", v)))
		}},
		{"seam", "<blocks>", "blocks on the device", func(c *chatPage, a string) tea.Cmd {
			n, err := strconv.Atoi(a)
			if err != nil || n < 0 {
				return notify("usage: /seam 12")
			}
			return seamTo(n)
		}},
		{"copy", "", "copy the last reply", func(c *chatPage, _ string) tea.Cmd { return c.copyReply() }},
		{"think", "", "show or fold reasoning", func(c *chatPage, _ string) tea.Cmd {
			c.env.f.st.ShowThinking.Set(!c.env.f.st.ShowThinking.Get())
			return nil
		}},
		{"quit", "", "leave", func(*chatPage, string) tea.Cmd { return tea.Quit }},
	}
}

// slashMatches is every command whose name starts with what is typed.
func slashMatches(text string) []slashCmd {
	word, _, _ := strings.Cut(strings.TrimPrefix(text, "/"), " ")
	var out []slashCmd
	for _, c := range slashCmds() {
		if strings.HasPrefix(c.name, word) {
			out = append(out, c)
		}
	}
	return out
}

func (c *chatPage) slash(text string) tea.Cmd {
	word, arg, _ := strings.Cut(strings.TrimPrefix(text, "/"), " ")
	arg = strings.TrimSpace(arg)
	for _, s := range slashCmds() {
		if s.name == word {
			return s.run(c, arg)
		}
	}
	if ms := slashMatches(text); len(ms) == 1 {
		return ms[0].run(c, arg)
	}
	return notify("no command /" + word)
}

// seamMsg asks the engine page to move the seam to n device blocks, so its
// "moving" indicator knows a move is on its way.
type seamMsg struct{ n int }

func seamTo(n int) tea.Cmd { return func() tea.Msg { return seamMsg{n} } }

func setChat(env *env, on bool) tea.Cmd {
	env.cfg.Chat = on
	env.f.st.Chat.Set(on)
	if on {
		return tea.Batch(saveConfig(env), notify("chat template on"))
	}
	return tea.Batch(saveConfig(env), notify("raw completion"))
}

// saveConfig writes the shared settings, so the window starts the same way.
func saveConfig(env *env) tea.Cmd {
	if err := env.cfg.Save(); err != nil {
		return notify("settings not saved: " + err.Error())
	}
	return nil
}

func (c *chatPage) View() string {
	mid := c.w - c.listW - c.sideW
	focus := cBrand
	if c.env.f.st.Busy.Get() {
		focus = cLine
	}
	box := sCard().BorderForeground(focus).Width(mid - sCard().GetHorizontalBorderSize()).Render(c.prompt.View())
	title := "conversation"
	if n := c.env.f.turnCount(); n > 0 {
		title += fmt.Sprintf(" · %d", n)
	}
	tr := panel(title, c.tr.vp.View(), mid, c.h-c.promptH, cBlue)
	if menu := c.prompt.menu(); menu != "" {
		tr = overlay(tr, menu, 2, lipgloss.Height(tr)-lipgloss.Height(menu)-1)
	}
	cols := []string{lipgloss.JoinVertical(lipgloss.Left, tr, box)}
	if c.listW > 0 {
		cols = append([]string{c.chatList()}, cols...)
	}
	if c.sideW > 0 {
		cols = append(cols, sidebar(c.env, c.sideW, c.h))
	}
	return lipgloss.JoinHorizontal(lipgloss.Top, cols...)
}

// chatList is the column of conversations.
func (c *chatPage) chatList() string {
	iw, ih := inner(c.listW, c.h)
	var b strings.Builder
	for i, info := range c.chats.Infos() {
		title := info.Title
		if title == "" {
			title = "New chat"
		}
		if i == c.chats.Cur {
			b.WriteString(lipgloss.NewStyle().Foreground(cBg).Background(cBrand).Bold(true).Width(iw).
				Render(truncate(" "+title, iw)) + "\n")
		} else {
			b.WriteString(sMuted().Render(truncate(" "+title, iw)) + "\n")
		}
		if info.Model != "" {
			b.WriteString(sDim().Render(truncate("  "+strings.TrimSuffix(filepath.Base(info.Model), ".jlm"), iw)) + "\n")
		}
	}
	hint := sDim().Render(truncate("ctrl+↑↓ · ctrl+n new", iw))
	list := lipgloss.NewStyle().Height(ih - 1).Render(fitHeight(b.String(), ih-1))
	return panel(fmt.Sprintf("chats · %d", len(c.chats.Chats)), list+"\n"+hint, c.listW, c.h, cMagenta)
}

// sidebar is the engine at a glance beside the chat: the models open, the
// rates with their recent history, where every block runs, and memory.
func sidebar(env *env, w, h int) string {
	st := env.f.st
	iw, _ := inner(w, h)
	var b strings.Builder
	section := func(title string) { b.WriteString(sBrand().Render(title) + "\n") }
	kv := func(k, v string) { b.WriteString(kvLine(k, v, iw) + "\n") }

	section("MODELS")
	models := st.Models.Get()
	if len(models) == 0 {
		b.WriteString(sDim().Render("none loaded · f2") + "\n")
	}
	for _, lm := range models {
		name := truncate(strings.TrimSuffix(lm.Name, ".jlm"), iw-2)
		// Every loaded model is loaded and usable; the chat's own is bold.
		ns := sMuted()
		if lm.Active {
			ns = sBold()
		}
		b.WriteString(lipgloss.NewStyle().Foreground(colourOf(lm.Colour, modelColours())).Render("●") + " " + ns.Render(name) + "\n")
		b.WriteString("  " + sDim().Render(fmt.Sprintf("%d blocks", lm.Blocks)) + "\n")
	}

	b.WriteString("\n")
	section("SPEED")
	if r := st.DecodeTokS.Get(); r > 0 {
		b.WriteString(sBrand().Render(fmt.Sprintf("%.1f", r)) + sDim().Render(" tok/s decode") + "\n")
	} else {
		b.WriteString(sDim().Render("—") + "\n")
	}
	if ch := env.tele.decode.chart(iw, 2, cBrand); ch != "" {
		b.WriteString(ch + "\n")
	}
	if r := st.PromptTokS.Get(); r > 0 {
		kv("prompt", fmt.Sprintf("%.0f tok/s", r))
	}
	if g := st.GBs.Get(); g >= 0.05 {
		kv("bandwidth", fmt.Sprintf("%.1f GB/s", g))
	}
	if bt := st.BytesPerTok.Get(); bt > 0 {
		kv("per token", session.Bytes(uint64(bt)))
	}

	if bm := st.BlockMap.Get(); len(bm) > 0 {
		b.WriteString("\n")
		section("BLOCKS")
		b.WriteString(env.tele.grid(bm, iw, false, env.frame) + "\n")
		b.WriteString(blockLegend(bm) + "\n")
	}

	a := st.Alloc.Get()
	if a.NBlocks > 0 || a.MaxSeq > 0 {
		b.WriteString("\n")
		section("MEMORY")
		if a.HostBudget > 0 {
			kv("host", session.Bytes(a.HostUsed)+" / "+session.Bytes(a.HostBudget))
			b.WriteString(meter(float64(a.HostUsed)/float64(a.HostBudget), iw, cBrand, cBlue) + "\n")
		}
		for i, d := range a.Devices {
			v := session.Bytes(d.Used)
			if d.Limit > 0 {
				v += " / " + session.Bytes(d.Limit)
			}
			kv(deviceName(d.Name, iw/2), v)
			if d.Limit > 0 {
				c := colourOf(i, deviceColours())
				b.WriteString(meter(float64(d.Used)/float64(d.Limit), iw, c, lerp(c, "#ffffff", 0.3)) + "\n")
			}
		}
		if a.MaxSeq > 0 {
			kv("context", fmt.Sprintf("%d / %d", a.Pos, a.MaxSeq))
			b.WriteString(meter(float64(a.Pos)/float64(a.MaxSeq), iw, cMagenta, cBlue) + "\n")
		}
	}
	return panel("engine", b.String(), w, h, cBrand)
}
