package main

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
	"github.com/jitllm/jitllm/common/config"
	"github.com/jitllm/jitllm/common/engine"
)

// root is the program's model: it owns the pages, routes every message,
// keeps focus, and runs the frame clock and the spinner only while
// something on screen needs them.
type root struct {
	env   *env
	pages []page
	cur   screen

	chat     *chatPage
	models   *modelsPage
	discover *discoverPage
	engine   *enginePage
	settings *settingsPage

	palette *palette
	help    help.Model

	// splash is the opening, built at the first size the terminal reports;
	// splashing is true until it ends or a key skips it.
	splash    *compile
	splashing bool

	toast    string
	toastSeq int

	w, h, bodyH int

	ticking, spinning bool
}

// toastDoneMsg clears notice seq, if it is still the one showing.
type toastDoneMsg struct{ seq int }

// helpKeys is a list of bindings as the help bubble reads one.
type helpKeys []key.Binding

func (k helpKeys) ShortHelp() []key.Binding  { return k }
func (k helpKeys) FullHelp() [][]key.Binding { return [][]key.Binding{k} }

func newRoot(f *front, e engine.Driver, cfg *config.Config, dirs []string) *root {
	env := &env{
		f: f, e: e, cfg: cfg, dirs: dirs, keys: newGlobalKeys(), tele: &telemetry{},
		spin: spinner.New(spinner.WithSpinner(spinner.MiniDot), spinner.WithStyle(sBrand())),
	}
	env.jobs = newJobs(env)
	r := &root{env: env, cur: scrModels, splashing: true}
	r.chat = newChatPage(env)
	r.models = newModelsPage(env)
	r.discover = newDiscoverPage(env)
	r.engine = newEnginePage(env)
	r.settings = newSettingsPage(env)
	r.pages = []page{r.chat, r.models, r.discover, r.engine, r.settings}
	r.help = help.New()
	r.help.Styles.ShortKey = sMuted().Bold(true)
	r.help.Styles.ShortDesc = sDim()
	r.help.Styles.ShortSeparator = sDim()
	r.help.Styles.Ellipsis = sDim()
	return r
}

func (r *root) page() page { return r.pages[r.cur] }

func (r *root) Init() tea.Cmd {
	if r.env.cfg.API {
		return tea.Batch(r.page().Focus(), notify(r.env.syncAPI()))
	}
	return r.page().Focus()
}

// setScreen moves focus to s.
func (r *root) setScreen(s screen) tea.Cmd {
	if s == r.cur {
		return nil
	}
	r.page().Blur()
	r.cur = s
	return r.page().Focus()
}

func (r *root) quit() tea.Cmd {
	return tea.Sequence(r.chat.save(), tea.Quit)
}

func (r *root) Update(msg tea.Msg) tea.Cmd {
	var cmds []tea.Cmd
	switch msg := msg.(type) {
	case wakeMsg:
		em := r.env.f.collect()
		r.env.tele.observe(r.env, em)
		for _, p := range r.pages {
			cmds = append(cmds, p.Update(em))
		}
	case postMsg:
		msg.fn()
	case jobMsg, progress.FrameMsg:
		cmds = append(cmds, r.env.jobs.Update(msg))
	case machineMsg:
		cmds = append(cmds, r.discover.Update(msg))
	case rescanMsg:
		cmds = append(cmds, r.models.Update(msg), r.discover.Update(msg))
	case probedMsg:
		cmds = append(cmds, r.models.Update(msg))
	case settingsDoneMsg, settingsCancelMsg:
		cmds = append(cmds, r.settings.Update(msg))
	case seamMsg:
		cmds = append(cmds, r.engine.Update(msg), r.setScreen(scrEngine))
	case showMsg:
		cmds = append(cmds, r.setScreen(msg.s))
	case loadMsg:
		cmds = append(cmds, r.load(msg.path))
	case notifyMsg:
		r.toastSeq++
		r.toast = msg.text
		seq := r.toastSeq
		cmds = append(cmds, tea.Tick(3*time.Second, func(time.Time) tea.Msg { return toastDoneMsg{seq} }))
	case toastDoneMsg:
		if msg.seq == r.toastSeq {
			r.toast = ""
		}
	case closePaletteMsg:
		r.palette = nil
		cmds = append(cmds, r.page().Focus())
	case tea.WindowSizeMsg:
		r.w, r.h = msg.Width, msg.Height
		r.layout()
		if r.splashing && r.splash == nil {
			r.splash = newCompile(r.w, r.h-2)
		}
	case animMsg:
		r.ticking = false
		r.env.frame++
		if r.splash != nil && r.splash.done() {
			r.splashing = false
		}
		cmds = append(cmds, r.page().Update(msg))
	case spinner.TickMsg:
		if r.spinWanted() {
			var c tea.Cmd
			r.env.spin, c = r.env.spin.Update(msg)
			cmds = append(cmds, c)
		} else {
			r.spinning = false
		}
	case tea.KeyMsg:
		cmds = append(cmds, r.key(msg))
	default:
		if r.palette != nil {
			cmds = append(cmds, r.palette.Update(r, msg))
		} else {
			cmds = append(cmds, r.page().Update(msg))
		}
	}
	if r.animWanted() && !r.ticking {
		r.ticking = true
		if r.splashing {
			// The opening flies pixels; it gets a frame rate to match.
			cmds = append(cmds, tea.Tick(time.Second/30, func(time.Time) tea.Msg { return animMsg{} }))
		} else {
			cmds = append(cmds, frameTick())
		}
	}
	if r.spinWanted() && !r.spinning {
		r.spinning = true
		cmds = append(cmds, r.env.spin.Tick)
	}
	return tea.Batch(cmds...)
}

// key routes a key: quit always, then the palette when it is open, then
// the root's own bindings unless the page is taking text, then the page.
func (r *root) key(k tea.KeyMsg) tea.Cmd {
	g := r.env.keys
	if r.splashing {
		r.splashing = false
		return nil
	}
	if key.Matches(k, g.Quit) {
		return r.quit()
	}
	if r.palette != nil {
		return r.palette.Update(r, k)
	}
	if key.Matches(k, g.Palette) {
		r.page().Blur()
		r.palette = newPalette(r)
		return r.palette.Focus()
	}
	for i, b := range g.Screens {
		if key.Matches(k, b) {
			return r.setScreen(screen(i))
		}
	}
	if r.cur != scrChat && !r.page().Capturing() && key.Matches(k, g.Back) {
		return r.setScreen(scrChat)
	}
	return r.page().Update(k)
}

// load opens path, or switches to it when it is open, and shows the chat.
func (r *root) load(path string) tea.Cmd {
	switch {
	case !isOpen(r.env, path):
		r.env.e.Load(path)
	case path != r.env.f.st.Active.Get():
		r.env.e.Use(path)
	}
	return r.setScreen(scrChat)
}

func (r *root) animWanted() bool {
	return r.splashing || r.page().Animating() || r.env.jobs.busy() || r.env.tele.flashing(r.env.frame)
}

func (r *root) spinWanted() bool {
	st := r.env.f.st
	return st.Busy.Get() || len(st.Loads.Get()) > 0 || r.env.jobs.busy() || !r.discover.probed || r.engine.asked
}

// layout gives every page the body: the terminal less the header and the
// footer, as they measure.
func (r *root) layout() {
	r.help.Width = r.w - 2
	r.bodyH = max(4, r.h-lipgloss.Height(r.header())-lipgloss.Height(r.footer()))
	for _, p := range r.pages {
		p.SetSize(r.w, r.bodyH)
	}
}

func (r *root) View() string {
	if r.w == 0 {
		return ""
	}
	if r.splashing && r.splash != nil {
		return r.splashView()
	}
	body := r.page().View()
	if r.palette != nil {
		pv := r.palette.View(r.w)
		body = overlay(dim(body), pv, max(0, (r.w-lipgloss.Width(pv))/2), max(0, (r.bodyH-lipgloss.Height(pv))/3))
	}
	return lipgloss.JoinVertical(lipgloss.Left, r.header(), body, r.footer())
}

// header is the wordmark, the screens as tabs, and what the engine is doing.
func (r *root) header() string {
	st := r.env.f.st
	left := pill("◆ jitllm", cBg, cBrand) + " "
	for i, n := range screenNames() {
		fkey := fmt.Sprintf("F%d", i+1)
		if screen(i) == r.cur {
			left += " " + sBrand().Render(n)
		} else if r.w >= 110 {
			left += " " + sDim().Render(fkey+" ") + sMuted().Render(n)
		}
	}
	var right string
	switch {
	case r.env.jobs.busy():
		right = r.env.jobs.header()
	case len(st.Loads.Get()) > 0:
		l := st.Loads.Get()[0]
		right = r.env.spin.View() + " " + sMuted().Render("loading "+filepath.Base(l.Path)+" · "+l.Stage)
	case st.Streaming.Get():
		right = r.env.spin.View() + " " + sBrand().Render("generating")
		if v := st.DecodeTokS.Get(); v > 0 {
			right += sMuted().Render(fmt.Sprintf(" · %.1f tok/s", v))
		}
	case st.Active.Get() != "":
		colour, _ := activeModel(r.env)
		mode := "completion"
		if r.env.cfg.Chat {
			mode = "chat"
		}
		right = pill(strings.TrimSuffix(filepath.Base(st.Active.Get()), ".jlm"), cBg, colourOf(colour, modelColours())) + " " + sDim().Render(mode)
	}
	right = ansi.Truncate(right, max(0, r.w-lipgloss.Width(left)-3), "…")
	gap := max(1, r.w-lipgloss.Width(left)-lipgloss.Width(right)-2)
	return " " + left + strings.Repeat(" ", gap) + right + " "
}

// footer is the keys of the screen in front, or a notice when one is up,
// with the engine's status line at the right.
func (r *root) footer() string {
	g := r.env.keys
	keys := helpKeys(r.page().Help())
	if r.cur != scrChat {
		keys = append(keys, g.Back)
	}
	keys = append(keys, g.Palette, g.Quit)
	left := r.help.View(keys)
	if r.palette != nil {
		left = r.help.View(helpKeys{r.palette.keys.Up, r.palette.keys.Run, r.palette.keys.Close})
	}
	if r.toast != "" {
		left = pill("✓", cBg, cBrand) + " " + sFg().Render(r.toast)
	}
	// The engine badge is always shown, whatever else gives way: which engine
	// this is cannot be a thing that scrolls off.
	badge := sMuted().Render(r.env.badge())
	if r.env.where.Lost {
		badge = pill(r.env.badge(), cBg, cRed)
	}
	if w := r.w - 4 - lipgloss.Width(badge); lipgloss.Width(left) > w {
		left = ansi.Truncate(left, max(0, w), "…")
	}
	room := r.w - 7 - lipgloss.Width(left) - lipgloss.Width(badge)
	right := badge
	if room > 16 {
		right = ansi.Truncate(sDim().Render(r.env.f.st.Status.Get()), room, "…") + sDim().Render(" · ") + badge
	}
	line := " " + left + strings.Repeat(" ", max(1, r.w-2-lipgloss.Width(left)-lipgloss.Width(right))) + right + " "
	return ansi.Truncate(line, r.w, "")
}

// splashView is the website's opening: the wordmark compiling out of the
// noise, and the line under it once it has landed.
func (r *root) splashView() string {
	tag := ""
	if r.splash.built() {
		tag = sDim().Render("an inference OS: every kernel generated for the machine in front of it")
	}
	return r.splash.String() + "\n\n" + lipgloss.PlaceHorizontal(r.w, lipgloss.Center, tag)
}

// program is the root as Bubble Tea's Model: Update returns the root itself.
type program struct{ *root }

func (p program) Update(msg tea.Msg) (tea.Model, tea.Cmd) { return p, p.root.Update(msg) }
