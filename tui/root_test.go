package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/charmbracelet/bubbles/list"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/jitllm/jitllm/common/config"
	"github.com/jitllm/jitllm/common/engine"
	"github.com/jitllm/jitllm/common/session"
)

// newTestRoot is the app over a real engine with no model, a models folder
// holding one file, and settings that are never written anywhere.
func newTestRoot(t *testing.T) *root {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "x.gguf"), []byte("not a model"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := config.DefaultConfig()
	cfg.ModelDirs = []string{dir}
	f := newFront()
	f.st = newState(f, "cpu", "", 1024, false, false)
	e := engine.New(f, f.st)
	t.Cleanup(e.Close)
	// Nothing drains the queue in a test but the test itself.
	go func() {
		for range f.queue {
		}
	}()
	r := newRoot(f, e, cfg, cfg.ModelDirs)
	r.Update(tea.WindowSizeMsg{Width: 160, Height: 44})
	r.splashing = false
	return r
}

// Every screen renders exactly the terminal's size, at a wide and a narrow
// width, with and without the palette over it.
func TestEveryScreenFillsTheTerminalExactly(t *testing.T) {
	r := newTestRoot(t)
	for _, size := range [][2]int{{160, 44}, {100, 30}, {80, 24}} {
		r.Update(tea.WindowSizeMsg{Width: size[0], Height: size[1]})
		for s := range screenNames {
			r.setScreen(screen(s))
			for _, palette := range []bool{false, true} {
				r.palette = nil
				if palette {
					r.palette = newPalette(r)
				}
				v := r.View()
				lines := strings.Split(v, "\n")
				if len(lines) != size[1] {
					t.Errorf("%s %dx%d palette=%v: %d lines", screenNames[s], size[0], size[1], palette, len(lines))
				}
				for i, l := range lines {
					if w := len([]rune(stripANSI(l))); w > size[0] {
						t.Errorf("%s %dx%d palette=%v: line %d is %d wide", screenNames[s], size[0], size[1], palette, i, w)
						break
					}
				}
			}
		}
	}
	r.palette = nil
}

// View changes nothing: two renders in a row are the same string.
func TestViewIsPure(t *testing.T) {
	r := newTestRoot(t)
	for s := range screenNames {
		r.setScreen(screen(s))
		if a, b := r.View(), r.View(); a != b {
			t.Errorf("%s: two Views differ", screenNames[s])
		}
	}
}

// The frame clock stops when nothing on screen moves.
func TestTheClockStopsWhenNothingMoves(t *testing.T) {
	r := newTestRoot(t)
	r.discover.probed = true
	r.setScreen(scrModels)
	if r.animWanted() {
		t.Error("the models screen with no job wants frames")
	}
	r.setScreen(scrChat)
	r.env.f.appendTurn(session.Turn{Role: session.RoleUser, Text: "hi"})
	r.Update(wakeMsg{})
	if r.animWanted() {
		t.Error("a chat with a finished transcript wants frames")
	}
}

// Tab completes a command in the prompt and never changes the screen; esc
// goes back to the chat from anywhere, after first clearing a filter.
func TestNavigationIsEscBackToTheChat(t *testing.T) {
	r := newTestRoot(t)
	r.setScreen(scrChat)
	r.chat.prompt.SetValue("/te")
	r.Update(tea.KeyMsg{Type: tea.KeyTab})
	if got := r.chat.prompt.Value(); got != "/temp " {
		t.Fatalf("tab completed to %q", got)
	}
	r.chat.prompt.SetValue("")
	r.Update(tea.KeyMsg{Type: tea.KeyTab})
	if r.cur != scrChat {
		t.Fatalf("tab left the chat for %s", screenNames[r.cur])
	}
	r.Update(tea.KeyMsg{Type: tea.KeyF2})
	r.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("x")})
	if r.models.list.FilterState() == list.Unfiltered {
		t.Fatal("typing on Models did not filter")
	}
	r.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if r.cur != scrModels {
		t.Fatal("esc with a filter up left Models instead of clearing it")
	}
	for i := 0; i < 2 && r.cur == scrModels; i++ {
		r.Update(tea.KeyMsg{Type: tea.KeyEsc})
	}
	if r.cur != scrChat {
		t.Fatalf("esc on Models went to %s, not the chat", screenNames[r.cur])
	}
}

// Only the focused page's text box takes typing.
func TestOnlyTheFocusedPageTakesKeys(t *testing.T) {
	r := newTestRoot(t)
	r.setScreen(scrEngine)
	r.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("x")})
	if v := r.chat.prompt.Value(); v != "" {
		t.Fatalf("typing on Models reached the chat prompt: %q", v)
	}
}

// A slash command runs instead of being sent to the model.
func TestASlashCommandIsNotAPrompt(t *testing.T) {
	r := newTestRoot(t)
	r.setScreen(scrChat)
	r.chat.Update(submitMsg{"/temp 0.5"})
	if r.env.cfg.Sampling.Temp != 0.5 {
		t.Errorf("temperature %v after /temp 0.5", r.env.cfg.Sampling.Temp)
	}
	if n := r.env.f.turnCount(); n != 0 {
		t.Errorf("a command left %d turns in the transcript", n)
	}
}

// History walks back through what was sent and forward to the draft.
func TestHistoryReturnsToTheDraft(t *testing.T) {
	var h history
	h.add("one")
	h.add("two")
	if s, _ := h.step(-1, "draft"); s != "two" {
		t.Fatalf("up gave %q", s)
	}
	if s, _ := h.step(-1, ""); s != "one" {
		t.Fatalf("up again gave %q", s)
	}
	if _, ok := h.step(-1, ""); ok {
		t.Fatal("up past the first prompt moved")
	}
	h.step(1, "")
	if s, _ := h.step(1, ""); s != "draft" {
		t.Fatalf("down to the end gave %q, want the draft", s)
	}
}

// A new chat parks the old one, and stepping back shows it again.
func TestChatsKeepTheirTurns(t *testing.T) {
	r := newTestRoot(t)
	r.env.f.appendTurn(session.Turn{Role: session.RoleUser, Text: "first chat"})
	r.chat.newChat()
	if n := r.env.f.turnCount(); n != 0 {
		t.Fatalf("new chat shows %d turns", n)
	}
	r.chat.step(1)
	turns, _, _ := r.env.f.snapshot()
	if len(turns) != 1 || turns[0].Text != "first chat" {
		t.Fatalf("back to the first chat shows %v", turns)
	}
}

// macOS takes ctrl+arrows for Mission Control, so Option (alt) arrows switch
// chats too, and the help says so.
func TestOptionArrowsSwitchChatsAndAreInTheHelp(t *testing.T) {
	r := newTestRoot(t)
	r.env.f.appendTurn(session.Turn{Role: session.RoleUser, Text: "first chat"})
	r.chat.newChat()
	r.setScreen(scrChat)
	r.Update(tea.KeyMsg{Type: tea.KeyDown, Alt: true})
	if turns, _, _ := r.env.f.snapshot(); len(turns) != 1 || turns[0].Text != "first chat" {
		t.Fatalf("alt+down shows %v", turns)
	}
	r.Update(tea.KeyMsg{Type: tea.KeyUp, Alt: true})
	if n := r.env.f.turnCount(); n != 0 {
		t.Fatalf("alt+up back to the new chat shows %d turns", n)
	}
	var prev, next bool
	for _, b := range r.chat.Help() {
		prev = prev || b.Help().Key == "ctrl/⌥+↑"
		next = next || b.Help().Key == "ctrl/⌥+↓"
	}
	if !prev || !next {
		t.Fatal("the help does not show the Option bindings")
	}
}

// A finished turn is rendered once: a second refresh reuses it.
func TestAFinishedTurnIsRenderedOnce(t *testing.T) {
	r := newTestRoot(t)
	r.env.f.appendTurn(session.Turn{Role: session.RoleAssistant, Text: "**bold** reply"})
	tr := r.chat.tr
	tr.refresh()
	first := tr.done[0]
	if first.out == "" {
		t.Fatal("the turn was not rendered")
	}
	tr.done[0] = rendered{first.key, "SENTINEL"}
	tr.refresh()
	if !strings.Contains(tr.vp.View(), "SENTINEL") {
		t.Fatal("an unchanged turn was rendered again")
	}
}

func stripANSI(s string) string {
	var b strings.Builder
	esc := false
	for _, r := range s {
		switch {
		case r == '\x1b':
			esc = true
		case esc && (r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z'):
			esc = false
		case !esc:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// Two loaded models both read "loaded" on the models page and in the chat's
// sidebar, and the palette offers each an explicit unload.
func TestTwoLoadedModelsAreBothLoaded(t *testing.T) {
	r := newTestRoot(t)
	dir := r.env.dirs[0]
	a, b := filepath.Join(dir, "a.jlm"), filepath.Join(dir, "b.jlm")
	for _, p := range []string{a, b} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	r.env.f.st.Models.Set([]session.LoadedModel{
		{Path: a, Name: "a.jlm", Arch: "llama", Blocks: 2},
		{Path: b, Name: "b.jlm", Arch: "llama", Blocks: 2, Colour: 1, Active: true},
	})
	r.env.f.st.Active.Set(b)
	r.pages[scrModels].(*modelsPage).scan()
	r.setScreen(scrModels)
	v := stripANSI(r.View())
	if n := strings.Count(v, "● loaded"); n != 2 {
		t.Errorf("%d rows read \"● loaded\", want 2:\n%s", n, v)
	}
	for _, w := range []string{"● active", "○ open", "2 open"} {
		if strings.Contains(v, w) {
			t.Errorf("the models page still says %q", w)
		}
	}
	side := stripANSI(sidebar(r.env, 40, 30))
	if n := strings.Count(side, "●"); n != 2 {
		t.Errorf("the sidebar marks %d models loaded, want 2:\n%s", n, side)
	}
	var titles []string
	for _, a := range actions(r) {
		titles = append(titles, a.title)
	}
	all := strings.Join(titles, "|")
	for _, w := range []string{"Unload a", "Unload b", "Chat with a"} {
		if !strings.Contains(all, w) {
			t.Errorf("the palette has no %q: %s", w, all)
		}
	}
}
