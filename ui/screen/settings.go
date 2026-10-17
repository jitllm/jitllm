package screen

import (
	"github.com/jitllm/jitllm/common/api"
	"os"

	"github.com/gogpu/ui/core/button"
	"github.com/gogpu/ui/core/checkbox"
	"github.com/gogpu/ui/core/chip"
	"github.com/gogpu/ui/core/scrollview"
	"github.com/gogpu/ui/core/textfield"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/common/config"
	core "github.com/jitllm/jitllm/common/engine"
	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/widgets"
)

// Settings is every choice the app remembers, in one place, and where it
// remembers them. Each change is saved as it is made (Shell.SaveSoon), to the
// settings file in this system's config folder.
func Settings(sh *app.Shell) widget.Widget {
	p := sh.P
	title := p.Role(primitives.Text("Settings").Color(p.Text()), p.Type.HeadlineSmall).Bold()
	sub := widgets.NewParagraph("Saved as you change them, in this computer's settings folder.").
		FontSize(p.Type.BodyMedium.FontSize).Color(p.Muted()).Font(app.Prose)

	cards := []widget.Widget{
		primitives.VBox(title, sub).Gap(p.Space.XS).CrossAlign(primitives.CrossAxisStretch),
		// Which engine runs comes first: every other setting is that engine's.
		card(sh, "Engine", engineSettings(sh)),
		card(sh, "Appearance", appearance(sh)),
		card(sh, "Model folders", modelFolders(sh)),
		card(sh, "Loading models", nextLoad(sh)),
		card(sh, "Chat", chatSettings(sh)),
		card(sh, "API", apiSettings(sh)),
	}
	if cl := commandLine(sh); cl != nil {
		cards = append(cards, card(sh, "Command line", cl))
	}
	cards = append(cards, card(sh, "Files", files(sh)))
	body := primitives.VBox(cards...).Gap(p.Space.M).CrossAlign(primitives.CrossAxisStretch).Padding(p.Space.L)

	return primitives.Box(primitives.VBox(
		primitives.Expanded(scrollview.New(body, scrollview.PainterOpt(p.Scrollbar))),
	)).Background(p.Background())
}

func appearance(sh *app.Shell) widget.Widget {
	p := sh.P
	dark := sh.M3 == nil || sh.M3.IsDark()
	theme := func(label string, d bool) widget.Widget {
		return chip.New(
			chip.Label(label),
			chip.Selectable(true),
			// SetDark rebuilds the window, so the selection is decided per build.
			chip.SelectedReadonlySignal(state.NewSignal(dark == d).AsReadonly()),
			chip.OnClick(func() {
				sh.SetDark(d)
				sh.SaveSoon()
			}),
			chip.PainterOpt(p.Chip),
		)
	}
	return primitives.VBox(
		label(sh, "Theme"),
		primitives.HBox(theme("Dark", true), theme("Light", false)).Gap(p.Space.S),
		checkbox.New(
			checkbox.LabelOpt("Show sampling settings beside the chat"),
			checkbox.CheckedSignal(sh.Store.ShowTuning),
			checkbox.PainterOpt(p.Checkbox),
		),
	).Gap(p.Space.S)
}

// modelFolders is where the Models tab looks for models; the first is also
// where Discover saves what it downloads.
func modelFolders(sh *app.Shell) widget.Widget {
	p := sh.P
	st := sh.Store
	n := state.NewComputed(func() int { return len(st.ModelDirs.Get()) }, st.ModelDirs.AsReadonly())
	list := widgets.NewColumn(n, p.Space.XS, func(i int) widget.Widget {
		dir := state.NewComputed(func() string {
			if ds := st.ModelDirs.Get(); i < len(ds) {
				return ds[i]
			}
			return ""
		}, st.ModelDirs.AsReadonly())
		role := "Models are found here"
		if i == 0 {
			role = "Downloads are saved here"
		}
		remove := widgets.NewIconButton(widgets.IconTrash, "Stop looking in this folder", func() {
			ds := append([]string(nil), st.ModelDirs.Get()...)
			if i < len(ds) {
				sh.SetModelDirs(append(ds[:i], ds[i+1:]...))
			}
		}, iconColors(sh))
		open := widgets.NewIconButton(widgets.IconFolder, "Open this folder", func() { OpenFolder(sh, dir.Get()) }, iconColors(sh))
		return primitives.HBox(
			primitives.Expanded(primitives.VBox(
				widgets.NewParagraph("").ContentSignal(dir).FontSize(p.Type.BodyMedium.FontSize).
					Color(p.Text()).MaxLines(1).Font(app.Mono),
				line(sh, state.NewSignal(role).AsReadonly()),
			).Gap(2)),
			open, remove,
		).Gap(p.Space.XS)
	})
	empty := wrapped(sh, state.NewComputed(func() string {
		if len(st.ModelDirs.Get()) == 0 {
			return "No folders. Add one, or download a model from Discover."
		}
		return ""
	}, st.ModelDirs.AsReadonly()))
	add := button.New(
		button.TextOpt("Add folder"),
		button.SizeOpt(button.Small),
		button.VariantOpt(button.Outlined),
		button.OnClick(func() {
			start := ""
			if ds := st.ModelDirs.Get(); len(ds) > 0 {
				start = ds[0]
			}
			if dir := sh.PickDir("Add a model folder", start); dir != "" && !sh.AddModelDir(dir) {
				sh.SetStatus(dir + " is already a model folder")
			}
		}),
		button.PainterOpt(p.Button),
	)
	return primitives.VBox(list, empty, primitives.HBox(add)).Gap(p.Space.S).CrossAlign(primitives.CrossAxisStretch)
}

func chatSettings(sh *app.Shell) widget.Widget {
	p := sh.P
	st := sh.Store
	mode := func(l string, chat bool) widget.Widget {
		return chip.New(
			chip.Label(l),
			chip.Selectable(true),
			chip.SelectedReadonlySignal(state.NewComputed(func() bool { return st.Chat.Get() == chat }, st.Chat.AsReadonly())),
			chip.OnClick(func() { st.Chat.Set(chat) }),
			chip.PainterOpt(p.Chip),
		)
	}
	system := textfield.New(
		textfield.ValueSignal(st.System),
		textfield.Placeholder("None"),
		textfield.PainterOpt(p.TextField),
	)
	resetSampling := button.New(
		button.TextOpt("Reset sampling"),
		button.SizeOpt(button.Small),
		button.VariantOpt(button.Outlined),
		button.OnClick(func() {
			st.Sampling.Set(app.DefaultConfig().Sampling)
			sh.SetStatus("sampling reset to its defaults")
		}),
		button.PainterOpt(p.Button),
	)
	return primitives.VBox(
		label(sh, "New messages are sent as"),
		primitives.HBox(mode("Chat", true), mode("Completion", false)).Gap(p.Space.S),
		note(sh, "Chat wraps each message in the model's own chat format. Completion continues the text exactly as typed."),
		label(sh, "System prompt"),
		p.Field(system),
		note(sh, "Sent before every chat with a model that has a chat format."),
		checkbox.New(
			checkbox.LabelOpt("Remember prompts between runs"),
			checkbox.CheckedSignal(st.KVCache),
			checkbox.PainterOpt(p.Checkbox),
		),
		note(sh, "Keeps the work done on a prompt on disk, so the same start of a chat is not read again. Applies to the next model loaded."),
		primitives.HBox(resetSampling),
	).Gap(p.Space.S).CrossAlign(primitives.CrossAxisStretch)
}

// engineModes are the engine setting's choices, in the order shown.
var engineModes = []struct{ mode, label string }{
	{core.ModeAuto, "Auto"},
	{core.ModeEmbedded, "Embedded"},
	{core.ModeService, "Service"},
}

// engineSettings is which engine the window runs: a jitllmd it connects to,
// or its own. A choice applies with Apply, since switching closes the models
// the engine in use holds.
func engineSettings(sh *app.Shell) widget.Widget {
	p := sh.P
	st := sh.Store
	mode := func() string {
		if m := st.EngineMode.Get(); m != "" {
			return m
		}
		return core.ModeAuto
	}
	chips := make([]widget.Widget, 0, len(engineModes))
	for _, m := range engineModes {
		m := m
		chips = append(chips, chip.New(
			chip.Label(m.label),
			chip.Selectable(true),
			chip.SelectedReadonlySignal(state.NewComputed(func() bool { return mode() == m.mode }, st.EngineMode.AsReadonly())),
			chip.OnClick(func() { st.EngineMode.Set(m.mode) }),
			chip.PainterOpt(p.Chip),
		))
	}
	addr := textfield.New(
		textfield.ValueSignal(st.ServiceAddr),
		textfield.Placeholder(localserver.DefaultAddr),
		textfield.PainterOpt(p.TextField),
	)
	apply := button.New(
		button.TextOpt("Apply"),
		button.SizeOpt(button.Small),
		button.VariantOpt(button.Outlined),
		button.OnClick(func() { applyEngine(sh) }),
		button.PainterOpt(p.Button),
	)
	return primitives.VBox(
		label(sh, "Now"),
		wrapped(sh, sh.Badge()),
		primitives.HBox(append(chips, primitives.Expanded(primitives.Box()), apply)...).Gap(p.Space.S),
		note(sh, "Auto connects to a jitllmd that answers at the address and runs this app's own engine only when none does. Service connects and nothing else; Embedded always runs its own."),
		label(sh, "Service address"),
		p.Field(addr),
		note(sh, "Empty is the login service's address. Apply closes the models the engine in use holds."),
		label(sh, "Service API key"),
		keyField(sh, st.ServiceKey),
		showKeys(sh),
		note(sh, "Sent to a jitllmd that was started with an API key. Blank is none."),
	).Gap(p.Space.S).CrossAlign(primitives.CrossAxisStretch)
}

// applyEngine switches to what the setting chooses.
func applyEngine(sh *app.Shell) {
	if sh.ApplyEngine == nil {
		sh.SetStatus("this window has one engine only")
		return
	}
	sh.ApplyEngine()
}

// apiSettings turns on the API the app serves (Shell.ServeAPI).
func apiSettings(sh *app.Shell) widget.Widget {
	p := sh.P
	st := sh.Store
	addr := textfield.New(
		textfield.ValueSignal(st.APIAddr),
		textfield.Placeholder(api.DefaultAddr),
		textfield.PainterOpt(p.TextField),
	)
	return primitives.VBox(
		checkbox.New(
			checkbox.LabelOpt("Serve the API while the app runs"),
			checkbox.CheckedSignal(st.API),
			checkbox.PainterOpt(p.Checkbox),
		),
		note(sh, "OpenAI- and Anthropic-compatible endpoints and jitllm's own Connect API. A model open here is a model of the API, and its requests run beside the chat."),
		label(sh, "Address"),
		p.Field(addr),
		note(sh, "Empty is "+api.DefaultAddr+", this computer only. A new address applies the next time the API is turned on."),
		label(sh, "API key"),
		keyField(sh, st.APIKey),
		showKeys(sh),
		note(sh, "Required of clients of the API when set; blank is none. Applies the next time the API is turned on."),
	).Gap(p.Space.S).CrossAlign(primitives.CrossAxisStretch)
}

// files is where everything the app keeps lives, each a click from the
// system's file browser.
func files(sh *app.Shell) widget.Widget {
	p := sh.P
	st := sh.Store
	where := func(name, path, what string) widget.Widget {
		if path == "" {
			path = "not saved: this window was opened without a settings file"
		}
		open := widgets.NewIconButton(widgets.IconFolder, "Open the folder", func() { OpenFolder(sh, config.FolderOf(path)) }, iconColors(sh))
		copyPath := widgets.NewIconButton(widgets.IconCopy, "Copy the path", func() {
			sh.Copy(path)
			sh.SetStatus("copied " + path)
		}, iconColors(sh))
		return primitives.HBox(
			primitives.Expanded(primitives.VBox(
				p.Role(primitives.Text(name).Color(p.Text()), p.Type.BodyMedium),
				widgets.NewParagraph(path).FontSize(p.Type.BodySmall.FontSize).Color(p.Muted()).MaxLines(1).Font(app.Mono),
				line(sh, state.NewSignal(what).AsReadonly()),
			).Gap(2)),
			open, copyPath,
		).Gap(p.Space.XS)
	}

	deleteChats := button.New(
		button.TextOpt("Delete all chats"),
		button.SizeOpt(button.Small),
		button.VariantOpt(button.Outlined),
		button.DisabledReadonlySignal(st.Streaming.AsReadonly()),
		button.OnClick(func() {
			sh.Confirm("Delete all chats?", "Every conversation is removed. This cannot be undone.", "Delete", func() {
				if st.Streaming.Get() {
					sh.SetStatus("a reply is being written: stop it first")
					return
				}
				for len(st.Chats.Get()) > 1 {
					st.DeleteChat(0)
				}
				st.DeleteChat(0)
				sh.SetStatus("all chats deleted")
			})
		}),
		button.PainterOpt(p.Button),
	)
	clearCache := button.New(
		button.TextOpt("Clear prompt cache"),
		button.SizeOpt(button.Small),
		button.VariantOpt(button.Outlined),
		button.DisabledReadonlySignal(st.Loaded.AsReadonly()),
		button.OnClick(func() {
			sh.Confirm("Clear the prompt cache?", "Prompts seen before will be read again the next time.", "Clear", func() {
				if err := os.RemoveAll(app.KVCacheDir()); err != nil {
					sh.SetStatus("prompt cache: " + err.Error())
					return
				}
				sh.SetStatus("prompt cache cleared")
			})
		}),
		button.PainterOpt(p.Button),
	)
	reset := button.New(
		button.TextOpt("Reset all settings"),
		button.SizeOpt(button.Small),
		button.VariantOpt(button.Outlined),
		button.OnClick(func() {
			sh.Confirm("Reset all settings?", "Every setting goes back to its default. Chats and models are kept.", "Reset", func() {
				ResetSettings(sh)
				sh.SetStatus("settings reset")
			})
		}),
		button.PainterOpt(p.Button),
	)

	return primitives.VBox(
		where("Settings", sh.Cfg.Path(), "Everything on this page."),
		where("Chats", sh.Cfg.ChatsPath(), "Every conversation in the sidebar."),
		where("Prompt cache", app.KVCacheDir(), "Can be cleared at any time; it is only a speed-up."),
		primitives.HBox(deleteChats, clearCache, reset).Gap(p.Space.S),
		wrapped(sh, state.NewComputed(func() string {
			if st.Loaded.Get() {
				return "Unload the models to clear the prompt cache: they hold it."
			}
			return ""
		}, st.Loaded.AsReadonly())),
	).Gap(p.Space.M).CrossAlign(primitives.CrossAxisStretch)
}

// ResetSettings puts every setting back to its default. Model folders, the
// chats and the window size are the person's data, not settings, and stay.
func ResetSettings(sh *app.Shell) {
	d := app.DefaultConfig()
	st := sh.Store
	st.Chat.Set(d.Chat)
	st.System.Set(d.System)
	st.DeviceSpec.Set(d.DeviceSpec)
	st.Placement.Set(d.Placement)
	st.Sampling.Set(d.Sampling)
	st.MaxSeq.Set(d.MaxSeq)
	st.ShowTuning.Set(!d.HideTuning)
	st.KVCache.Set(!d.NoKVCache)
	st.API.Set(d.API)
	st.APIAddr.Set(d.APIAddr)
	sh.Cfg.MaxMem = d.MaxMem
	sh.SaveSoon()
	sh.SetDark(!d.Light)
}

// keyField is a masked text field over sig, swapped for a plain one while
// ShowKeys is on. The two share the signal, so the value is the same text.
func keyField(sh *app.Shell, sig state.Signal[string]) widget.Widget {
	p := sh.P
	masked := textfield.New(
		textfield.ValueSignal(sig),
		textfield.InputTypeOpt(textfield.TypePassword),
		textfield.PainterOpt(p.TextField),
	)
	plain := textfield.New(
		textfield.ValueSignal(sig),
		textfield.PainterOpt(p.TextField),
	)
	show := sh.Store.ShowKeys
	hidden := state.NewComputed(func() bool { return !show.Get() }, show.AsReadonly())
	return primitives.VBox(
		widgets.Hide(hidden, p.Field(masked)),
		widgets.Hide(show.AsReadonly(), p.Field(plain)),
	).CrossAlign(primitives.CrossAxisStretch)
}

// showKeys is the checkbox that unmasks the key fields.
func showKeys(sh *app.Shell) widget.Widget {
	return checkbox.New(
		checkbox.LabelOpt("Show keys"),
		checkbox.CheckedSignal(sh.Store.ShowKeys),
		checkbox.PainterOpt(sh.P.Checkbox),
	)
}

// label is a small heading over one control.
func label(sh *app.Shell, s string) widget.Widget {
	return sh.P.Role(primitives.Text(s).Color(sh.P.Muted()), sh.P.Type.LabelMedium)
}

// OpenFolder shows a folder in the system's file browser.
func OpenFolder(sh *app.Shell, dir string) {
	if err := config.OpenFolder(dir); err != nil {
		sh.SetStatus("could not open " + dir + ": " + err.Error())
	}
}
