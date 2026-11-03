package screen

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/gogpu/gogpu"

	"github.com/gogpu/ui/core/button"
	"github.com/gogpu/ui/core/chip"
	"github.com/gogpu/ui/core/textfield"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/widgets"
)

// iconColors is the palette every icon button on the session screen uses.
func iconColors(sh *app.Shell) widgets.IconButtonColors {
	c := sh.P.Colors
	return widgets.IconButtonColors{
		Icon: c.OnSurfaceVariant, Hover: c.SurfaceContainerHigh, Disabled: c.Outline,
		Fill: c.Primary, On: c.OnPrimary,
	}
}

// sessionStatus is what the engine is doing, as the header's pill says it.
// Prefill produces no token for however long the prompt takes, so it must not
// read as idle.
func sessionStatus(sh *app.Shell) state.ReadonlySignal[widgets.Status] {
	st, c := sh.Store, sh.P.Colors
	return state.NewComputed(func() widgets.Status {
		switch {
		case st.Loading():
			return widgets.Status{Label: st.Loads.Get()[0].Stage, Dot: sh.P.Warn}
		case !st.Loaded.Get():
			return widgets.Status{Label: "No model", Dot: c.Outline}
		case st.Streaming.Get() && st.DecodeTokS.Get() > 0 && st.Stream.Get()+st.StreamThink.Get() != "":
			return widgets.Status{Label: "Generating · " + app.Rate(st.DecodeTokS.Get()), Dot: sh.P.Warn}
		case st.Busy.Get():
			return widgets.Status{Label: "Reading the prompt", Dot: sh.P.Warn}
		default:
			return widgets.Status{Label: "Ready", Dot: c.Primary}
		}
	}, st.Loaded.AsReadonly(), st.Streaming.AsReadonly(), st.Busy.AsReadonly(),
		st.DecodeTokS.AsReadonly(), st.Stream.AsReadonly(), st.StreamThink.AsReadonly(), st.Loads.AsReadonly())
}

// sessionHeader is the open models, what the engine is doing, and the
// conversation's three actions: copy it, clear it, and open the sampling panel.
func sessionHeader(sh *app.Shell) widget.Widget {
	st := sh.Store
	ic := iconColors(sh)

	noTurns := state.NewComputed(func() bool { return st.Turns.Get() == 0 }, st.Turns.AsReadonly())
	copyAll := widgets.NewIconButton(widgets.IconCopy(), "Copy the conversation", func() {
		if t := transcriptText(st); t != "" {
			sh.Copy(t)
			sh.SetStatus("conversation copied")
		}
	}, ic).DisabledSignal(noTurns)
	clear := widgets.NewIconButton(widgets.IconTrash(), "Delete this chat",
		func() { sessionClear(sh) }, ic).DisabledSignal(sessionClearDisabled(st))
	tune := widgets.NewIconButton(widgets.IconSliders(), "Sampling settings",
		func() { st.ShowTuning.Set(!st.ShowTuning.Get()) }, ic)

	// A Paragraph capped at one line: Text.Ellipsis() only affects measuring
	// (UPSTREAM.md #10). It has its own row; beside the controls the HBox drew
	// them over it.
	summary := widgets.NewParagraph("").
		ContentSignal(st.ModelSummary.AsReadonly()).
		FontSize(sh.P.Type.BodySmall.FontSize).
		LineHeight(sh.P.Type.BodySmall.LineHeight / sh.P.Type.BodySmall.FontSize).
		Color(sh.P.Muted()).
		MaxLines(1).
		Font(sh.P.Prose())

	return primitives.VBox(
		primitives.Box(primitives.VBox(
			primitives.HBox(
				// The bar hides itself with no model open and then measures zero,
				// so it sits in a row that always takes the space.
				primitives.Expanded(modelBar(sh)),
				primitives.Box(widgets.NewStatusPill(sessionStatus(sh), sh.P.Muted(), sh.P.Type.BodySmall.FontSize)).PaddingTop(2),
				copyAll, clear, tune,
			).Gap(sh.P.Space.XS),
			// With no model the transcript's empty state says so; a second
			// "no model loaded" here would only repeat it.
			widgets.Hide(st.Loaded.AsReadonly(), summary),
		).Gap(sh.P.Space.XS).CrossAlign(primitives.CrossAxisStretch)).
			PaddingXY(sh.P.Space.M, sh.P.Space.S),
		divider(sh),
	).CrossAlign(primitives.CrossAxisStretch)
}

// divider is a hairline between two regions that share a background.
func divider(sh *app.Shell) widget.Widget {
	return primitives.Box().Height(1).Background(sh.P.Colors.OutlineVariant)
}

// sessionClear asks, then deletes the chat on screen; the newest other chat
// takes its place, or an empty one. The model needs no reset: the engine
// resets before every turn and a chat turn is rendered from the transcript
// alone. It refuses while a reply runs, because a confirm raised earlier can be
// answered mid-decode and the finished text would land in a turn index that no
// longer exists.
func sessionClear(sh *app.Shell) {
	st := sh.Store
	sh.Confirm("Delete this chat?",
		"Removes the conversation, the draft and any attached pictures. The model stays loaded.",
		"Delete", func() {
			if st.Busy.Get() {
				sh.SetStatus("a reply is being generated: stop it before deleting the chat")
				return
			}
			st.DeleteChat(st.ChatSel.Get())
			st.PromptTokS.Set(0)
			st.DecodeTokS.Set(0)
			st.BytesPerTok.Set(0)
			st.GBs.Set(0)
			sh.SetStatus("chat deleted")
		})
}

// sessionClearDisabled is true while a reply runs, or while there is nothing
// to delete: the only chat, and empty.
func sessionClearDisabled(st *app.Store) state.ReadonlySignal[bool] {
	return state.NewComputed(func() bool {
		return st.Busy.Get() || (len(st.Chats.Get()) <= 1 &&
			st.Turns.Get() == 0 && strings.TrimSpace(st.Draft.Get()) == "" && len(st.Attach.Get()) == 0)
	}, st.Busy.AsReadonly(), st.Turns.AsReadonly(), st.Draft.AsReadonly(), st.Attach.AsReadonly(), st.Chats.AsReadonly())
}

// sessionComposer is one card: the pictures waiting to go, the prompt, and a
// row of the things that shape the next message -- attach, reasoning, chat or
// completion -- ending in Send, which is Stop while a reply runs. Under it, one
// line of what the last reply cost. The field is one line because gogpu/ui has
// no multiline text field; a long prompt comes in through the file button.
func sessionComposer(sh *app.Shell, o SessionOptions) widget.Widget {
	st := sh.Store
	ic := iconColors(sh)
	busy := st.Busy.AsReadonly()

	field := textfield.New(
		textfield.Placeholder("Message the model\u2026"),
		textfield.ValueSignal(st.Draft),
		textfield.OnSubmit(func(string) { sessionSend(sh, o) }),
		textfield.DisabledFn(func() bool { return st.Busy.Get() }),
		textfield.A11yLabel("prompt"),
		textfield.PainterOpt(sh.P.BareTextField()),
	)

	image := widgets.NewIconButton(widgets.IconImage(), "Attach a picture for a vision model",
		func() { sessionPickImage(sh) }, ic).
		DisabledSignal(state.NewComputed(func() bool {
			return st.Busy.Get() || !st.Vision.Get()
		}, st.Busy.AsReadonly(), st.Vision.AsReadonly()))
	fromFile := widgets.NewIconButton(widgets.IconFile(), "Send a prompt from a file",
		func() { sessionPickPrompt(sh) }, ic).DisabledSignal(busy)

	// Opens or closes every turn's reasoning at once.
	reasoning := chip.New(
		chip.LabelFn(func() string { return "Reasoning" }),
		chip.SelectedReadonlySignal(st.ShowThinking.AsReadonly()),
		chip.OnClick(func() { st.SetAllThinkOpen(!st.ShowThinking.Get()) }),
		chip.PainterOpt(sh.P.Chip),
	)
	// Chat wraps the prompt in the model's own template; Completion continues
	// the text as given. Two chips, because both should be visible at once.
	modeChip := func(label string, chat bool) widget.Widget {
		return chip.New(
			chip.Label(label),
			chip.Selectable(true),
			chip.SelectedReadonlySignal(state.NewComputed(func() bool {
				return st.Chat.Get() == chat
			}, st.Chat.AsReadonly())),
			chip.OnClick(func() { st.Chat.Set(chat) }),
			chip.DisabledReadonlySignal(busy),
			chip.PainterOpt(sh.P.Chip),
		)
	}

	send := widgets.NewIconButton(widgets.IconSend(), "Send", func() { sessionSend(sh, o) }, ic).
		Filled().DisabledSignal(sessionSendDisabled(st))
	// Stop cancels the job and leaves the model open, so the next prompt
	// starts from a warm engine.
	stop := widgets.NewIconButton(widgets.IconStop(), "Stop the reply", func() {
		if eng := depsOf(sh).Engine; eng != nil {
			eng.Stop()
		}
	}, ic).Filled()

	card := primitives.Box(primitives.VBox(
		sessionAttachments(sh),
		field,
		primitives.HBox(
			image, fromFile, reasoning, modeChip("Chat", true), modeChip("Completion", false),
			primitives.Expanded(primitives.Box()),
			widgets.Hide(sessionNot(busy), send),
			widgets.Hide(busy, stop),
		).Gap(sh.P.Space.XS),
	).Gap(sh.P.Space.XS).CrossAlign(primitives.CrossAxisStretch)).
		PaddingXY(sh.P.Space.S+sh.P.Space.XS, sh.P.Space.S).
		Rounded(sh.P.Shape.ExtraLarge).
		BorderStyle(1, sh.P.Colors.OutlineVariant).
		Background(sh.P.Colors.SurfaceContainer)

	// Figures, so mono, and muted: the conversation is the thing on the page.
	footer := widgets.NewParagraph("").
		ContentSignal(sessionFooter(sh)).
		FontSize(sh.P.Type.LabelSmall.FontSize).
		Color(sh.P.Muted()).
		MaxLines(1).
		Font(sh.P.Mono())

	return primitives.Box(reading(primitives.VBox(card, primitives.Box(footer).PaddingXY(sh.P.Space.S, 0)).
		Gap(sh.P.Space.XS).CrossAlign(primitives.CrossAxisStretch))).
		PaddingXY(sh.P.Space.M, sh.P.Space.S)
}

// sessionAttachments is the strip of pictures waiting to go with the next
// message: a thumbnail, the file's name and a way to take it back.
func sessionAttachments(sh *app.Shell) widget.Widget {
	st := sh.Store
	n := state.NewComputed(func() int { return len(st.Attach.Get()) }, st.Attach.AsReadonly())
	col := widgets.NewColumn(n, 6, func(i int) widget.Widget {
		paths := st.Attach.Get()
		if i >= len(paths) {
			return primitives.Box()
		}
		path := paths[i]
		remove := button.New(
			button.TextOpt("Remove"),
			button.SizeOpt(button.Small),
			button.VariantOpt(button.TextOnly),
			button.DisabledSignal(st.Busy),
			button.OnClick(func() { st.Attach.Set(sessionWithout(st.Attach.Get(), path)) }),
			button.PainterOpt(sh.P.Button),
		)
		return primitives.HBox(
			sh.Thumbs.New(path, sessionAttachThumb),
			primitives.Expanded(widgets.NewParagraph(filepath.Base(path)).
				FontSize(sh.P.Type.BodySmall.FontSize).Color(sh.P.Muted()).MaxLines(1).
				Font(sh.P.Prose())),
			remove,
		).Gap(8).CrossAlign(primitives.CrossAxisCenter)
	})
	// Says why Send is off when the model has no vision tower. The pictures
	// are kept, so switching back to a vision model sends them.
	blind := widgets.Hide(
		state.NewComputed(func() bool { return n.Get() > 0 && !st.Vision.Get() }, n, st.Vision.AsReadonly()),
		widgets.NewParagraph("This model can't see pictures. Remove them to send, or switch back to a vision model.").
			FontSize(sh.P.Type.BodySmall.FontSize).Color(sh.P.Colors.Error).Font(sh.P.Prose()),
	)
	return widgets.Hide(state.NewComputed(func() bool { return n.Get() > 0 }, n),
		primitives.VBox(col, blind).Gap(4).CrossAlign(primitives.CrossAxisStretch))
}

// sessionAttachThumb is the composer's thumbnail edge, and sessionTurnThumb
// the transcript's: the draft only has to say which picture.
const (
	sessionAttachThumb = 48
	sessionTurnThumb   = 192
)

// sessionWithout is paths without one of them.
func sessionWithout(paths []string, drop string) []string {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		if p != drop {
			out = append(out, p)
		}
	}
	return out
}

// sessionImageExts() are the formats model.Preprocess decodes. The picker's
// filter and the drop classifier both read this, so what can be chosen and
// what can be dropped cannot disagree.
func sessionImageExts() []string {
	return []string{"png", "jpg", "jpeg"}
}

// IsImage reports whether a path names a picture the engine can read.
func IsImage(path string) bool {
	ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(path), "."))
	return slices.Contains(sessionImageExts(), ext)
}

// sessionPickImage attaches a picture through the native dialog.
func sessionPickImage(sh *app.Shell) {
	paths := sh.PickFile("Attach an image",
		[]gogpu.FileTypeFilter{{Name: "Images", Extensions: sessionImageExts()}}, "")
	AttachImages(sh, paths)
}

// AttachImages adds pictures to the next message, and is the one route in:
// the Image button and a drop on the window both come through here.
func AttachImages(sh *app.Shell, paths []string) {
	st := sh.Store
	if !st.Vision.Get() {
		sh.SetStatus("the loaded model has no vision tower, so it cannot take a picture")
		return
	}
	cur := st.Attach.Get()
	added := 0
	for _, p := range paths {
		if IsImage(p) && !slices.Contains(cur, p) {
			cur = append(slices.Clip(cur), p)
			added++
		}
	}
	if added == 0 {
		return
	}
	st.Attach.Set(cur)
	sh.SetStatus(fmt.Sprintf("%d picture(s) attached to the next message", len(cur)))
}

// sessionPickPrompt reads a prompt off disk into the draft.
//
// It runs on the UI goroutine, because the native file dialog drives OS
// window APIs.
func sessionPickPrompt(sh *app.Shell) {
	paths := sh.PickFile("Prompt file", []gogpu.FileTypeFilter{
		{Name: "Text", Extensions: []string{"txt", "md", "prompt"}},
		{Name: "Any file", Extensions: []string{"*"}},
	}, "")
	if len(paths) == 0 {
		return
	}
	text, err := sessionReadPrompt(paths[0])
	if err != nil {
		sh.Alert("Could not read the prompt", err.Error())
		return
	}
	sh.Store.Draft.Set(text)
	sh.SetStatus("prompt loaded from " + paths[0])
}

// sessionSendDisabled is true while there is nothing to send, or nothing to
// send it to. It is a computed signal so the button greys itself out as the
// draft empties.
func sessionSendDisabled(st *app.Store) state.ReadonlySignal[bool] {
	return state.NewComputed(func() bool {
		return st.Busy.Get() || !st.Loaded.Get() ||
			(strings.TrimSpace(st.Draft.Get()) == "" && len(st.Attach.Get()) == 0) ||
			// Pictures attached for a model with no vision tower.
			(len(st.Attach.Get()) > 0 && !st.Vision.Get())
	}, st.Busy.AsReadonly(), st.Loaded.AsReadonly(), st.Draft.AsReadonly(), st.Attach.AsReadonly(), st.Vision.AsReadonly())
}

// sessionNot inverts a boolean signal, for the widgets whose only option is
// Disabled and whose natural condition is Enabled.
func sessionNot(sig state.ReadonlySignal[bool]) state.ReadonlySignal[bool] {
	return state.NewComputed(func() bool { return !sig.Get() }, sig)
}

// sessionReadPrompt reads a prompt file. It caps the read so a long file
// cannot become a megabyte-long single-line text field, and refuses a file
// that is not text (a NUL or invalid UTF-8).
func sessionReadPrompt(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if len(b) > sessionMaxPromptBytes {
		b = b[:sessionMaxPromptBytes]
		// A cap can split a rune; drop the partial tail rather than refuse.
		for len(b) > 0 && !utf8.Valid(b) {
			b = b[:len(b)-1]
		}
	}
	if IsImage(path) {
		return "", fmt.Errorf("%s is a picture; attach it with Image\u2026 instead", filepath.Base(path))
	}
	if bytes.IndexByte(b, 0) >= 0 || !utf8.Valid(b) {
		return "", fmt.Errorf("%s is not a text file", filepath.Base(path))
	}
	return strings.TrimRight(string(b), "\r\n"), nil
}

// sessionMaxPromptBytes bounds a prompt read from disk.
const sessionMaxPromptBytes = 1 << 20
