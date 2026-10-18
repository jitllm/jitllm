package main

import (
	"errors"
	"fmt"
	"net"
	"strconv"
	"strings"

	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/huh"
	"github.com/jitllm/jitllm/common/api"
	"github.com/jitllm/jitllm/common/hardware"
	"github.com/jitllm/jitllm/common/session"
)

// settingsPage is the window's settings as a huh form over the same file:
// a change here is a change there. The form edits a draft; submitting it
// (enter on the last field) applies and saves the draft, esc throws it away.
type settingsPage struct {
	env  *env
	form *huh.Form
	d    *draft
	w, h int
}

// draft is the settings as the form edits them: text for every number, so a
// half-typed value is the form's to validate rather than lost.
type draft struct {
	chat, kvCache bool
	api           bool
	apiAddr       string
	// serviceKey is the key to ask a jitllmd with, apiKey the one this app's
	// own API requires; showKeys unmasks both. Blank is no key.
	serviceKey, apiKey string
	showKeys           bool
	system, spec, ctx  string
	placement          string
	knobs              []string
	seed, folders      string
}

func newSettingsPage(env *env) *settingsPage {
	p := &settingsPage{env: env}
	p.rebuild()
	return p
}

func theme() *huh.Theme {
	t := huh.ThemeCharm()
	t.Focused.Base = t.Focused.Base.BorderForeground(cBrand)
	t.Focused.Title = t.Focused.Title.Foreground(cBrand).Bold(true)
	t.Focused.Description = t.Focused.Description.Foreground(cDim)
	t.Focused.TextInput.Cursor = t.Focused.TextInput.Cursor.Foreground(cBrand)
	t.Focused.TextInput.Prompt = t.Focused.TextInput.Prompt.Foreground(cBrand)
	t.Focused.TextInput.Text = t.Focused.TextInput.Text.Foreground(cFg)
	t.Focused.FocusedButton = t.Focused.FocusedButton.Background(cBrand).Foreground(cBg)
	t.Focused.BlurredButton = t.Focused.BlurredButton.Background(cRaised).Foreground(cMuted)
	t.Focused.ErrorIndicator = t.Focused.ErrorIndicator.Foreground(cRed)
	t.Focused.ErrorMessage = t.Focused.ErrorMessage.Foreground(cRed)
	t.Blurred = t.Focused
	t.Blurred.Base = t.Blurred.Base.BorderForeground(cLine)
	t.Blurred.Title = t.Blurred.Title.Foreground(cMuted).Bold(false)
	t.Blurred.TextInput.Text = t.Blurred.TextInput.Text.Foreground(cMuted)
	t.Blurred.FocusedButton = t.Blurred.FocusedButton.Background(cRaised).Foreground(cMuted)
	t.Group.Title = t.Group.Title.Foreground(cBrand).Bold(true)
	return t
}

// rebuild starts a fresh form over the settings as they are now.
func (p *settingsPage) rebuild() {
	cfg := p.env.cfg
	d := &draft{
		chat: cfg.Chat, kvCache: !cfg.NoKVCache, api: cfg.API, apiAddr: cfg.APIAddr,
		serviceKey: cfg.ServiceKey, apiKey: cfg.APIKey,
		system: cfg.System, spec: cfg.DeviceSpec, placement: cfg.Placement, ctx: strconv.Itoa(cfg.MaxSeq),
		seed: strconv.FormatInt(cfg.Sampling.Seed, 10), folders: strings.Join(cfg.ModelDirs, "\n"),
	}
	knobs := make([]huh.Field, len(session.SamplingKnobs))
	d.knobs = make([]string, len(session.SamplingKnobs))
	for i, k := range session.SamplingKnobs {
		k := k
		d.knobs[i] = strconv.FormatFloat(float64(k.Get(cfg.Sampling)), 'f', -1, 32)
		knobs[i] = huh.NewInput().Title(k.Name).Value(&d.knobs[i]).
			Description(fmt.Sprintf("%g to %g", k.Min, k.Max)).
			Validate(func(s string) error {
				v, err := strconv.ParseFloat(strings.TrimSpace(s), 32)
				if err != nil || float32(v) < k.Min || float32(v) > k.Max {
					return fmt.Errorf("a number from %g to %g", k.Min, k.Max)
				}
				return nil
			})
	}
	knobs = append(knobs, huh.NewInput().Title("Seed").Description("0 is random").Value(&d.seed).
		Validate(func(s string) error {
			if _, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64); err != nil {
				return errors.New("a whole number")
			}
			return nil
		}))

	p.d = d
	p.form = huh.NewForm(
		huh.NewGroup(
			huh.NewConfirm().Title("Chat template").Description("off is a raw completion of the prompt, as the benchmarks run").
				Affirmative("on").Negative("off").Value(&d.chat),
			huh.NewText().Title("System prompt").Lines(3).Value(&d.system),
			huh.NewConfirm().Title("Prompt cache on disk").Description("a prompt seen before resumes from its cached keys and values").
				Affirmative("on").Negative("off").Value(&d.kvCache),
		).Title("Chat"),
		huh.NewGroup(
			huh.NewInput().Title("Run models on").Description("auto, all, cpu, gpu, gpu:0, cuda:0, vulkan:1, metal · the next load").
				Value(&d.spec).Validate(func(s string) error {
				if strings.TrimSpace(s) == "" {
					return nil
				}
				_, err := hardware.ValidateSpec(strings.TrimSpace(s))
				return err
			}),
			huh.NewInput().Title("Block placement").
				Description("blocks to devices, e.g. 0-15=cuda:0,16-31=vulkan:0 · names must be in Run models on · empty lets jitllm place · the next load").
				Value(&d.placement).Validate(func(s string) error {
				if strings.TrimSpace(s) == "" {
					return nil
				}
				return hardware.ValidatePlacement(s, d.spec)
			}),
			huh.NewInput().Title("Context").Description("tokens · the next load").Value(&d.ctx).
				Validate(func(s string) error {
					if n, err := strconv.Atoi(strings.TrimSpace(s)); err != nil || n < 256 {
						return errors.New("a whole number of tokens, 256 or more")
					}
					return nil
				}),
			huh.NewText().Title("Model folders").Description("one per line; the first is where conversions go").
				Lines(4).Value(&d.folders).Validate(func(s string) error {
				if strings.TrimSpace(s) == "" {
					return errors.New("at least one folder")
				}
				return nil
			}),
			huh.NewConfirm().Title("API").Description("serve the OpenAI, Anthropic and Connect API while this app runs").
				Affirmative("on").Negative("off").Value(&d.api),
			huh.NewInput().Title("API address").Description("host:port · empty is the default, "+api.DefaultAddr+", this machine only").
				Value(&d.apiAddr).Validate(func(s string) error {
				if strings.TrimSpace(s) == "" {
					return nil
				}
				if _, _, err := net.SplitHostPort(strings.TrimSpace(s)); err != nil {
					return errors.New("host:port, e.g. localhost:4242")
				}
				return nil
			}),
			huh.NewConfirm().Title("Show API keys").Description("unmask the two key fields that follow").
				Affirmative("show").Negative("hide").Value(&d.showKeys),
		).Title("Engine"),
		// The same two values twice, masked and plain; the toggle above picks
		// which group is on screen.
		keyGroup(d, true),
		keyGroup(d, false),
		huh.NewGroup(knobs...).Title("Sampling"),
	).WithTheme(theme()).WithShowHelp(false).WithLayout(huh.LayoutColumns(3))
	p.form.SubmitCmd = func() tea.Msg { return settingsDoneMsg{} }
	p.form.CancelCmd = func() tea.Msg { return settingsCancelMsg{} }
	p.layout()
}

type (
	settingsDoneMsg   struct{}
	settingsCancelMsg struct{}
)

// apply writes the draft into the settings and the engine's state, and saves.
func (p *settingsPage) apply() tea.Cmd {
	cfg, st, d := p.env.cfg, p.env.f.st, p.d
	cfg.Chat = d.chat
	st.Chat.Set(d.chat)
	cfg.NoKVCache = !d.kvCache
	st.KVCache.Set(d.kvCache)
	cfg.System = strings.TrimSpace(d.system)
	cfg.DeviceSpec = strings.TrimSpace(d.spec)
	if cfg.DeviceSpec == "" {
		cfg.DeviceSpec = "auto"
	}
	st.DeviceSpec.Set(cfg.DeviceSpec)
	cfg.Placement = strings.TrimSpace(d.placement)
	st.Placement.Set(cfg.Placement)
	if n, err := strconv.Atoi(strings.TrimSpace(d.ctx)); err == nil {
		cfg.MaxSeq = n
		st.MaxSeq.Set(n)
	}
	for i, k := range session.SamplingKnobs {
		if v, err := strconv.ParseFloat(strings.TrimSpace(d.knobs[i]), 32); err == nil {
			k.Set(&cfg.Sampling, float32(v))
		}
	}
	if v, err := strconv.ParseInt(strings.TrimSpace(d.seed), 10, 64); err == nil {
		cfg.Sampling.Seed = v
	}
	var dirs []string
	for _, l := range strings.Split(d.folders, "\n") {
		if l = strings.TrimSpace(l); l != "" {
			dirs = append(dirs, l)
		}
	}
	folders := strings.Join(dirs, "\n") != strings.Join(cfg.ModelDirs, "\n")
	cfg.ModelDirs = dirs
	p.env.dirs = dirs
	cfg.API, cfg.APIAddr = d.api, strings.TrimSpace(d.apiAddr)
	cfg.ServiceKey, cfg.APIKey = strings.TrimSpace(d.serviceKey), strings.TrimSpace(d.apiKey)
	msg := "settings saved"
	if s := p.env.syncAPI(); s != "" {
		msg += " · " + s
	}
	cmds := []tea.Cmd{saveConfig(p.env), notify(msg)}
	if folders {
		cmds = append(cmds, rescan())
	}
	return tea.Batch(cmds...)
}

// keyGroup is the two key fields, masked or plain. Exactly one of the pair is
// shown, by the draft's showKeys.
func keyGroup(d *draft, masked bool) *huh.Group {
	mode := huh.EchoModeNormal
	if masked {
		mode = huh.EchoModePassword
	}
	return huh.NewGroup(
		huh.NewInput().Title("Key for the service").EchoMode(mode).Value(&d.serviceKey).
			Description("sent to a jitllmd that was started with -api-key · blank is none · next start"),
		huh.NewInput().Title("Key for this app's API").EchoMode(mode).Value(&d.apiKey).
			Description("required of clients of the API above · blank is none"),
	).Title("API keys").WithHideFunc(func() bool { return d.showKeys == masked })
}

func (p *settingsPage) Help() []key.Binding {
	return []key.Binding{bind("tab", "next field"), bind("shift+tab", "previous"), bind("enter", "next · save on the last"), bind("esc", "discard")}
}

func (p *settingsPage) Focus() tea.Cmd { return p.form.Init() }
func (p *settingsPage) Blur()          {}

// Capturing: the form takes tab and every printable key.
func (p *settingsPage) Capturing() bool { return true }
func (p *settingsPage) Animating() bool { return false }

func (p *settingsPage) SetSize(w, h int) {
	p.w, p.h = w, h
	p.layout()
}

func (p *settingsPage) layout() {
	iw, ih := inner(p.w, p.h)
	// The columns' borders and gaps are the form's own; it is given a little
	// less than the panel so they land inside it.
	p.form.WithWidth(iw - 4).WithHeight(ih - 1)
	if iw < 120 {
		p.form.WithLayout(huh.LayoutDefault)
	} else {
		p.form.WithLayout(huh.LayoutColumns(3))
	}
}

func (p *settingsPage) Update(msg tea.Msg) tea.Cmd {
	switch msg.(type) {
	case settingsDoneMsg:
		cmd := p.apply()
		p.rebuild()
		return tea.Batch(cmd, p.form.Init())
	case settingsCancelMsg:
		p.rebuild()
		return tea.Batch(notify("changes discarded"), p.form.Init())
	}
	m, cmd := p.form.Update(msg)
	if f, ok := m.(*huh.Form); ok {
		p.form = f
	}
	return cmd
}

func (p *settingsPage) View() string {
	foot := sDim.Render("saved to " + p.env.cfg.Path() + " · shared with the window")
	iw, _ := inner(p.w, p.h)
	return panel("settings", p.form.View()+"\n"+truncate(foot, iw), p.w, p.h, cBrand)
}
