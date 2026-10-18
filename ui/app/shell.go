// Package app is the desktop shell: the window, the navigation, the shared
// state and the async-to-UI bridge every screen is written against.
//
// The dependency direction is fixed and there is no cycle in it:
//
//	widgets  -> gogpu/ui only          (a leaf)
//	catalog  -> jitllm/jlm only        (a leaf)
//	app      -> widgets, catalog
//	engine   -> app, jitllm            (writes signals, never widgets)
//	screen   -> app, widgets, engine
//	main     -> all of the above
//
// engine depends on app rather than the other way round because the Store is
// the contract between them, and a Store that could not name an engine type
// would push those types somewhere worse.
package app

import (
	"strings"
	"sync"
	"time"

	"github.com/gogpu/gogpu"
	uiapp "github.com/gogpu/ui/app"
	"github.com/gogpu/ui/core/scrollview"
	"github.com/gogpu/ui/dnd"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/theme/material3"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/widgets"
)

// ScreenFunc builds a screen's root widget. It may be called more than once
// (a theme swap rebuilds the whole tree), so mutable state belongs in the
// Store, never in a closure over the first build or a package variable.
type ScreenFunc func(*Shell) widget.Widget

type screenReg struct {
	label string
	build ScreenFunc
}

// Shell owns the window, the theme, the shared state and the task queue.
type Shell struct {
	// GPU is the platform window. It is also the only route to the native file
	// dialogs and the clipboard, and both must be called from the UI
	// goroutine.
	GPU *gogpu.App
	// UI is the widget application.
	UI *uiapp.App
	// M3 is the live theme; P is the painter set built from it.
	M3 *material3.Theme
	P  Painters
	// Store is every piece of state the screens share.
	Store *Store
	// Cfg is the persisted settings.
	Cfg *Config

	q       chan func()
	screens []screenReg

	mu        sync.Mutex
	saveTimer *time.Timer
	shutdown  []func()
	onDrop    func(paths []string)
	// drop is the registered dnd target, kept so its bounds can follow the
	// window instead of registering a second one on every resize.
	drop dnd.DropTarget

	// ConnectTo switches the window to the jitllmd at addr; nil where it
	// cannot (the mock), and OfferService then only says what is running.
	ConnectTo func(addr string)
	// ApplyEngine switches to what the engine setting (Store.EngineMode,
	// ServiceAddr) now chooses; nil where there is one engine only.
	ApplyEngine func()
}

// NewShell builds the shell around an already-created gogpu app.
//
// It creates the ui app too, because a theme swap has to reach both.
func NewShell(gpuApp *gogpu.App, cfg *Config) *Shell {
	if cfg == nil {
		cfg = DefaultConfig()
	}
	m3 := newTheme(!cfg.Light)

	s := &Shell{
		GPU:   gpuApp,
		M3:    m3,
		P:     NewPainters(m3),
		Store: NewStore(),
		Cfg:   cfg,
		q:     make(chan func(), uiQueueDepth),
	}

	// Seed the store from the persisted settings so a screen reads one source.
	s.Store.Chat.Set(cfg.Chat)
	s.Store.System.Set(cfg.System)
	s.Store.DeviceSpec.Set(cfg.DeviceSpec)
	s.Store.Placement.Set(cfg.Placement)
	s.Store.Sampling.Set(cfg.Sampling)
	s.Store.MaxSeq.Set(cfg.MaxSeq)
	s.Store.ShowTuning.Set(!cfg.HideTuning)
	s.Store.KVCache.Set(!cfg.NoKVCache)
	s.Store.API.Set(cfg.API)
	s.Store.APIAddr.Set(cfg.APIAddr)
	s.Store.EngineMode.Set(cfg.Engine)
	s.Store.ServiceAddr.Set(cfg.ServiceAddr)
	s.Store.ServiceKey.Set(cfg.ServiceKey)
	s.Store.APIKey.Set(cfg.APIKey)
	s.Store.ModelDirs.Set(append([]string(nil), cfg.ModelDirs...))
	if p := cfg.ChatsPath(); p != "" {
		s.Store.LoadChats(p)
	}
	// Only a config read from disk is saved; a test's is never written.
	if cfg.Path() != "" {
		s.autosave()
	}

	if gpuApp != nil {
		s.UI = uiapp.New(
			uiapp.WithWindowProvider(gpuApp),
			uiapp.WithPlatformProvider(gpuApp),
			uiapp.WithEventSource(gpuApp.EventSource()),
			uiapp.WithTheme(m3.AsTheme()),
		)
	}
	return s
}

// Register adds a screen. Order is tab order. Call before [Shell.Build].
func (s *Shell) Register(label string, build ScreenFunc) {
	s.screens = append(s.screens, screenReg{label: label, build: build})
}

// Build constructs the whole widget tree and returns the root: a sidebar of
// pages, the selected page, and the status line.
//
// Every page stays mounted, so switching does not tear down the transcript or
// the loaded model. Each is wrapped in a tabPage so the compositor skips the
// ones not shown; see tabpage.go.
func (s *Shell) Build() widget.Widget {
	// The drop target is registered here, where the window exists: desktop.Run
	// hit-tests the window's dnd.Manager for OS drops.
	s.acceptDrops(float32(s.Cfg.WindowW), float32(s.Cfg.WindowH))

	pages := make([]widget.Widget, len(s.screens))
	for i, sc := range s.screens {
		pages[i] = newTabPage(i, s.Store.Tab, sc.build(s))
	}

	root := primitives.VBox(
		s.crashPanel(),
		primitives.Expanded(primitives.HBox(
			s.sidebar(),
			primitives.Box().Width(1).Background(s.P.Colors.OutlineVariant),
			primitives.Expanded(newPageStack(s.Store.Tab, pages)),
		)),
		s.statusBar(),
	).Background(s.P.Background())

	return widgets.NewDialogHost(root, s.Store.Dialog, s.P.Dialog)
}

// navOrder is the sidebar's order, which is the order of the work -- talk to a
// model, find one, manage them, convert, look at the machine -- not the
// registration order the tab constants fix.
var navOrder = []struct {
	tab  int
	icon widgets.Icon
}{
	{TabSession, widgets.IconChat},
	{TabDownload, widgets.IconDiscover},
	{TabModels, widgets.IconModels},
	{TabConvert, widgets.IconConvert},
	{TabMachine, widgets.IconMachine},
}

func (s *Shell) sidebar() widget.Widget {
	c := s.P.Colors
	colors := widgets.NavColors{
		Text: c.OnSurface, Muted: c.OnSurfaceVariant,
		Active: c.Primary, ActiveBg: c.SurfaceContainerHigh, HoverBg: c.SurfaceContainer,
	}
	size := s.P.Type.BodyMedium.FontSize
	items := []widget.Widget{}
	for _, n := range navOrder {
		if n.tab >= len(s.screens) {
			continue
		}
		tab := n.tab
		on := state.NewComputed(func() bool { return s.Store.Tab.Get() == tab }, s.Store.Tab.AsReadonly())
		items = append(items, widgets.NewNavItem(n.icon, s.screens[tab].label, on, func() { s.SelectTab(tab) }, colors, size))
	}

	// SetDark rebuilds the window, so the row is decided here, per build.
	themeIcon, themeLabel, toDark := widgets.IconSun, "Light theme", false
	if !s.M3.IsDark() {
		themeIcon, themeLabel, toDark = widgets.IconMoon, "Dark theme", true
	}
	theme := widgets.NewNavItem(themeIcon, themeLabel, nil, func() { s.SetDark(toDark) }, colors, size)
	// Settings sits with the theme at the foot: it is visited, not worked in.
	var settings widget.Widget = primitives.Box()
	if TabSettings < len(s.screens) {
		on := state.NewComputed(func() bool { return s.Store.Tab.Get() == TabSettings }, s.Store.Tab.AsReadonly())
		settings = widgets.NewNavItem(widgets.IconSettings, s.screens[TabSettings].label, on, s.GoSettings, colors, size)
	}

	mark := widgets.NewLogo(markBands(s.M3 == nil || s.M3.IsDark()), 2)

	return newRail(primitives.Box(primitives.VBox(
		mark,
		primitives.Box().Height(6),
		// Which engine the window drives, always in view; it opens the setting.
		widgets.NewChatRow(s.Badge(), nil, s.GoSettings, colors, s.P.Type.LabelSmall.FontSize),
		primitives.Box().Height(6),
		primitives.VBox(items...).Gap(2),
		primitives.Box().Height(16),
		widgets.NewNavItem(widgets.IconPlus, "New chat", nil, func() { s.chatAction(s.Store.NewChat) }, colors, size),
		widgets.NewNavHeading("Recent", s.P.Type.LabelSmall.FontSize, c.OnSurfaceVariant),
		primitives.Expanded(scrollview.New(s.chatList(colors, size), scrollview.PainterOpt(s.P.Scrollbar))),
		settings,
		theme,
	).CrossAlign(primitives.CrossAxisStretch)).
		PaddingXY(10, 14).
		Background(c.SurfaceContainerLowest))
}

// chatList is every conversation, newest first.
func (s *Shell) chatList(colors widgets.NavColors, size float32) widget.Widget {
	st := s.Store
	n := state.NewComputed(func() int { return len(st.Chats.Get()) }, st.Chats.AsReadonly())
	return widgets.NewColumn(n, 2, func(i int) widget.Widget {
		title := state.NewComputed(func() string {
			if cs := st.Chats.Get(); i < len(cs) {
				return cs[i].Title
			}
			return ""
		}, st.Chats.AsReadonly())
		on := state.NewComputed(func() bool { return st.ChatSel.Get() == i }, st.ChatSel.AsReadonly())
		return widgets.NewChatRow(title, on, func() { s.chatAction(func() { st.SelectChat(i) }) }, colors, size)
	})
}

// chatAction switches conversations and shows the chat. A reply being
// written goes on into the chat that asked for it (Store.SetTurn).
func (s *Shell) chatAction(do func()) {
	do()
	s.GoSession()
}

func (s *Shell) statusBar() widget.Widget {
	return primitives.HBox(
		primitives.Expanded(
			primitives.Text("").
				ContentSignal(s.Store.Status).
				FontSize(s.P.Type.BodySmall.FontSize).
				Color(s.P.Muted()).
				MaxLines(1).
				Ellipsis(),
		),
		primitives.Text("").
			ContentSignal(s.Store.ProgressLabel).
			FontSize(s.P.Type.BodySmall.FontSize).
			Color(s.P.Muted()).
			MaxLines(1).
			Ellipsis(),
	).
		Gap(12).
		PaddingXY(12, 6).
		Background(s.P.Colors.SurfaceContainerLowest)
}

// SetStatus writes the status line. Safe from any goroutine.
func (s *Shell) SetStatus(msg string) { s.Store.Status.Set(msg) }

// Alert raises a modal alert. Safe from any goroutine.
func (s *Shell) Alert(title, message string) {
	s.raise(DialogReq{Kind: DialogAlert, Title: title, Message: message})
}

// Confirm raises a modal confirmation. onOK and onCancel run on the UI
// goroutine. Safe from any goroutine.
func (s *Shell) Confirm(title, message, okLabel string, onOK func()) {
	s.raise(DialogReq{Kind: DialogConfirm, Title: title, Message: message, OK: okLabel, OnOK: onOK})
}

// raise publishes a dialog request from the UI goroutine, through Post like
// every other action (the README's concurrency rule 4). Post's RequestRedraw
// runs a frame, and widgets.DialogHost raises it from that frame's tick.
func (s *Shell) raise(r DialogReq) {
	r.Seq = widgets.NextDialogSeq()
	s.Post(func() { s.Store.Dialog.Set(r) })
}

// Report shows a problem on the session screen and puts its title in the
// status line. Safe from any goroutine; published on the UI goroutine like a
// dialog.
func (s *Shell) Report(p Problem) {
	p.Seq = widgets.NextDialogSeq()
	s.Post(func() {
		s.Store.Problem.Set(p)
		s.Store.Status.Set(p.Title)
	})
}

// SelectTab switches to a tab by index. Safe from any goroutine.
func (s *Shell) SelectTab(i int) { s.Store.Tab.Set(i) }

// OnFilesDropped registers the handler for files dropped on the window. The
// handler runs on the UI goroutine.
func (s *Shell) OnFilesDropped(fn func(paths []string)) {
	s.mu.Lock()
	s.onDrop = fn
	s.mu.Unlock()
}

// DropFiles is the gogpu drag-and-drop callback. Install it with
// gogpuApp.OnDragDrop(shell.DropFiles).
func (s *Shell) DropFiles(paths []string, _, _ float64) {
	s.mu.Lock()
	fn := s.onDrop
	s.mu.Unlock()
	if fn != nil {
		fn(paths)
	}
}

// OnShutdown registers a function to run when the window closes, in
// registration order, so main need not know the engine.
func (s *Shell) OnShutdown(fn func()) {
	s.mu.Lock()
	s.shutdown = append(s.shutdown, fn)
	s.mu.Unlock()
}

// Close runs the shutdown hooks and saves the settings. Install it with
// gogpuApp.OnClose(shell.Close). It runs on the render thread.
func (s *Shell) Close() {
	s.mu.Lock()
	hooks := s.shutdown
	s.shutdown = nil
	s.mu.Unlock()

	for _, fn := range hooks {
		fn()
	}

	if s.GPU != nil && s.Cfg != nil {
		if w, h := s.GPU.Size(); w > 0 && h > 0 {
			s.Cfg.WindowW, s.Cfg.WindowH = w, h
		}
	}
	// Best effort on the way out: a settings file we cannot write is not a
	// reason to fail a shutdown, and there is no one left to tell.
	s.SaveSettings()
}

// SaveSettings writes the settings and the chats now. UI goroutine only: it
// reads the Store into Cfg.
func (s *Shell) SaveSettings() {
	if s.Cfg == nil {
		return
	}
	st := s.Store
	s.Cfg.Chat = st.Chat.Get()
	s.Cfg.System = st.System.Get()
	s.Cfg.DeviceSpec = st.DeviceSpec.Get()
	s.Cfg.Placement = st.Placement.Get()
	s.Cfg.Sampling = st.Sampling.Get()
	s.Cfg.MaxSeq = st.MaxSeq.Get()
	s.Cfg.HideTuning = !st.ShowTuning.Get()
	s.Cfg.NoKVCache = !st.KVCache.Get()
	s.Cfg.API = st.API.Get()
	s.Cfg.APIAddr = st.APIAddr.Get()
	s.Cfg.Engine = st.EngineMode.Get()
	s.Cfg.ServiceAddr = st.ServiceAddr.Get()
	s.Cfg.ServiceKey = strings.TrimSpace(st.ServiceKey.Get())
	s.Cfg.APIKey = strings.TrimSpace(st.APIKey.Get())
	s.Cfg.ModelDirs = append([]string(nil), st.ModelDirs.Get()...)
	s.Cfg.LastModel = st.ModelPath.Get()
	if err := s.Cfg.Save(); err != nil {
		s.SetStatus("settings not saved: " + err.Error())
	}
	if err := st.SaveChats(s.Cfg.ChatsPath()); err != nil {
		s.SetStatus("chats not saved: " + err.Error())
	}
}

// SaveSoon saves the settings once they have stopped changing for a moment,
// so a slider dragged or a prompt typed is one write, not one per step. Safe
// from any goroutine.
func (s *Shell) SaveSoon() {
	if s.Cfg == nil || s.Cfg.Path() == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.saveTimer != nil {
		s.saveTimer.Stop()
	}
	s.saveTimer = time.AfterFunc(saveDelay, func() { s.Post(s.SaveSettings) })
}

// saveDelay is how long the settings must be still before they are written.
const saveDelay = 500 * time.Millisecond

// autosave saves whenever anything persisted changes, so a crash or a killed
// process keeps what was set rather than what was set at the last clean exit.
func (s *Shell) autosave() {
	st := s.Store
	save := s.SaveSoon
	on(st.Chat, save)
	on(st.System, save)
	on(st.DeviceSpec, save)
	on(st.Placement, save)
	on(st.Sampling, save)
	on(st.MaxSeq, save)
	on(st.ShowTuning, save)
	on(st.KVCache, save)
	on(st.API, save)
	on(st.APIAddr, save)
	on(st.EngineMode, save)
	on(st.ServiceAddr, save)
	// A key is used the moment the API or the engine is next started, which
	// can be before the delayed save: Cfg follows the field at once.
	on(st.ServiceKey, func() { s.Cfg.ServiceKey = strings.TrimSpace(st.ServiceKey.Get()); save() })
	on(st.APIKey, func() { s.Cfg.APIKey = strings.TrimSpace(st.APIKey.Get()); save() })
	on(st.ModelDirs, save)
	on(st.ModelPath, save)
	on(st.Chats, save)
	on(st.Turns, save)
	on(st.Revision, save)
}

// on calls f on every change of sig, for the life of the shell.
func on[T any](sig state.Signal[T], f func()) { sig.SubscribeForever(func(T) { f() }) }

// AddModelDir adds a folder models are found in, and reports whether it was
// new. UI goroutine only.
func (s *Shell) AddModelDir(dir string) bool {
	for _, d := range s.Store.ModelDirs.Get() {
		if d == dir {
			return false
		}
	}
	s.SetModelDirs(append(append([]string(nil), s.Store.ModelDirs.Get()...), dir))
	return true
}

// SetModelDirs replaces the folders models are found in; the first is where
// downloads go. UI goroutine only.
func (s *Shell) SetModelDirs(dirs []string) {
	s.Cfg.ModelDirs = append([]string(nil), dirs...)
	s.Store.ModelDirs.Set(append([]string(nil), dirs...))
}

// PickFile opens the native file picker. Call it from the UI goroutine only
// (a click handler, or inside [Shell.Post]): it drives OS window APIs.
func (s *Shell) PickFile(title string, filters []gogpu.FileTypeFilter, initialDir string) []string {
	if s.GPU == nil {
		return nil
	}
	paths, err := s.GPU.ShowOpenFileDialog(gogpu.FileDialogOptions{
		Title:            title,
		Filters:          filters,
		InitialDirectory: initialDir,
	})
	if err != nil {
		s.SetStatus("file dialog: " + err.Error())
		return nil
	}
	return paths
}

// PickDir opens the native directory picker. UI goroutine only; see
// [Shell.PickFile].
func (s *Shell) PickDir(title, initialDir string) string {
	if s.GPU == nil {
		return ""
	}
	paths, err := s.GPU.ShowOpenFileDialog(gogpu.FileDialogOptions{
		Title:            title,
		Directory:        true,
		InitialDirectory: initialDir,
	})
	if err != nil || len(paths) == 0 {
		return ""
	}
	return paths[0]
}

// Copy puts text on the clipboard. UI goroutine only.
func (s *Shell) Copy(text string) {
	if s.GPU == nil {
		return
	}
	if err := s.GPU.ClipboardWrite(text); err != nil {
		s.SetStatus("clipboard: " + err.Error())
	}
}
