package main

import (
	"github.com/jitllm/jitllm/common/api"
	"time"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/jitllm/jitllm/common/config"
	"github.com/jitllm/jitllm/common/engine"
	"github.com/jitllm/jitllm/internal/localserver"
)

// screen is which page fills the body.
type screen int

const (
	scrChat screen = iota
	scrModels
	scrDiscover
	scrEngine
	scrSettings
)

func screenNames() []string {
	return []string{"Chat", "Models", "Discover", "Engine", "Settings"}
}

// page is one screen. The root sizes it, focuses it, routes it every message
// and draws its View; a page asks the root for anything beyond itself with a
// message (showMsg, notifyMsg, rescanMsg).
type page interface {
	Update(msg tea.Msg) tea.Cmd
	View() string
	SetSize(w, h int)
	Focus() tea.Cmd
	Blur()
	// Help is the page's keys, for the footer.
	Help() []key.Binding
	// Capturing is true while the page takes text or has something esc
	// undoes (a filter), so esc goes to the page instead of back to the chat.
	Capturing() bool
	// Animating is true while the page draws something that moves, which is
	// when the root runs its frame clock.
	Animating() bool
}

// env is what every page shares: the engine and its state, the settings,
// the frame clock and the spinner.
type env struct {
	f    *front
	e    engine.Driver
	cfg  *config.Config
	dirs []string
	keys *globalKeys
	jobs *jobs
	tele *telemetry
	// frame counts animation ticks; spin is the one spinner every page shows.
	frame int
	spin  spinner.Model
	// api is the API this app serves, nil when it is off.
	api *api.Server
	// where is which engine this app drives, for the badge.
	where engine.Where
	// hub is the Hub downloads read from; "" is huggingface.co.
	hub string
}

// syncAPI starts, restarts or stops the API to match the settings, and
// returns its status line. The API serves the terminal's own engine, and its
// model folder is the first of the terminal's.
func (env *env) syncAPI() string {
	emb, ok := env.e.(*engine.Engine)
	if !ok {
		// Connected: the service serves the API; this app runs no engine to
		// serve one from.
		if env.cfg.API {
			return "the API is the service's, at " + env.where.Addr
		}
		return ""
	}
	srv := emb.Server()
	if len(env.dirs) > 0 {
		srv.SetModelDir(env.dirs[0])
	}
	var msg string
	var held localserver.Info
	env.api, msg, held = api.Toggle(env.api, env.cfg.API, env.cfg.APIAddr, srv, api.WithKey(env.cfg.ServingKey()))
	if held.Addr != "" {
		// Refused: a jitllmd holds the address. The setting goes back off
		// rather than claiming an API this app does not serve.
		env.cfg.API = false
		msg += " (tui --connect " + held.Addr + ")"
	}
	return msg
}

// badge is the engine line the status bar always shows (engine.Badge).
func (env *env) badge() string {
	a := ""
	if env.api != nil {
		a = env.api.Addr()
	}
	return engine.Badge(env.where, a)
}

// Messages a page sends the root.
type (
	showMsg   struct{ s screen }
	notifyMsg struct{ text string }
	// rescanMsg asks every page that lists the model folders to list them
	// again, after a job wrote into one or a folder was added.
	rescanMsg struct{}
	// loadMsg asks the root to open path (or switch to it) and show the chat.
	loadMsg struct{ path string }
)

func show(s screen) tea.Cmd      { return func() tea.Msg { return showMsg{s} } }
func notify(text string) tea.Cmd { return func() tea.Msg { return notifyMsg{text} } }
func rescan() tea.Cmd            { return func() tea.Msg { return rescanMsg{} } }
func load(path string) tea.Cmd   { return func() tea.Msg { return loadMsg{path} } }

// fps is the frame clock's rate while something animates.
const fps = 12

type animMsg struct{}

func frameTick() tea.Cmd {
	return tea.Tick(time.Second/fps, func(time.Time) tea.Msg { return animMsg{} })
}

// globalKeys are the bindings the root takes before any page.
type globalKeys struct {
	Quit, Palette, Back key.Binding
	Screens             []key.Binding
}

func bind(help, desc string, ks ...string) key.Binding {
	if len(ks) == 0 {
		ks = []string{help}
	}
	return key.NewBinding(key.WithKeys(ks...), key.WithHelp(help, desc))
}

func newGlobalKeys() *globalKeys {
	return &globalKeys{
		Quit:    bind("ctrl+c", "quit", "ctrl+c"),
		Palette: bind("ctrl+k", "commands", "ctrl+k", "ctrl+p"),
		Back:    bind("esc", "back to chat", "esc"),
		Screens: []key.Binding{
			bind("f1", "chat", "f1", "alt+1"),
			bind("f2", "models", "f2", "alt+2", "ctrl+o"),
			bind("f3", "discover", "f3", "alt+3"),
			bind("f4", "engine", "f4", "alt+4", "ctrl+e"),
			bind("f5", "settings", "f5", "alt+5"),
		},
	}
}
