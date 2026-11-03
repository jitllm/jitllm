package screen

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gogpu/ui/core/button"
	"github.com/gogpu/ui/core/collapsible"
	"github.com/gogpu/ui/core/progress"
	"github.com/gogpu/ui/core/scrollview"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/widgets"
)

// sessionTranscript builds the committed conversation, the watcher that keeps
// it scrolled to the newest turn, and the scroll offset. The watcher draws
// nothing, so it is returned for the caller to mount anywhere rather than
// parented beside the list. The offset is returned so the follow rule can be
// tested.
func sessionTranscript(sh *app.Shell) (widget.Widget, widget.Widget, state.Signal[float32]) {
	st := sh.Store

	// A plain column in a scrollview, not core/listview: a listview row's
	// children receive no events (itemDecorator.Event returns false, Draw never
	// stamps the child's origin, and the list's ClickRecognizer wins the arena),
	// its height cache is not invalidated on resize, and its rows are
	// RepaintBoundaries that bleed across tabs. The price is virtualisation:
	// every turn is built, which is fine at tens of turns.
	col := widgets.NewColumn(st.Turns.AsReadonly(), sh.P.Space.S, func(i int) widget.Widget {
		return sessionBubble(sh, i, st.Turn(i), isLiveTurn(st, i))
	}).KeyedBy(st.Shown.AsReadonly())

	// The offset is a signal so it can be driven; scrollview has no ScrollTo.
	scrollY := state.NewSignal(float32(0))
	sv := scrollview.New(col,
		scrollview.ScrollYSignal(scrollY),
		scrollview.PainterOpt(sh.P.Scrollbar),
	)

	// Following is posted to the UI goroutine: turns are appended from the
	// engine's goroutine and the widget has no lock.
	toBottom := func() {
		sh.Post(func() {
			// Content minus viewport, read after layout, and zero when it all fits:
			// scrollview draws the offset unclamped, so a stale offset after a Clear
			// would cut off the next message.
			scrollY.Set(max(0, sv.ContentSize().Height-sv.ViewportSize().Height))
		})
	}

	// The view follows the reply as it streams, but only while it is already at
	// the bottom: a reader who scrolled up is left where they are, and scrolling
	// back down re-arms following.
	atBottom := func() bool {
		gap := sv.ContentSize().Height - sv.ViewportSize().Height
		if gap <= 0 {
			return true
		}
		return gap-scrollY.Get() <= followSlack
	}
	followIfAnchored := func() {
		if atBottom() {
			toBottom()
		}
	}

	// A new turn always scrolls: sending a message is itself the intent to
	// see the answer, whatever was being read before. So does a Clear, to 0.
	follow := primitives.VBox(
		newSignalWatcher(st.Turns.AsReadonly(), func(int) { toBottom() }),
		newSignalWatcher(st.Stream.AsReadonly(), func(string) { followIfAnchored() }),
		newSignalWatcher(st.StreamThink.AsReadonly(), func(string) { followIfAnchored() }),
	).Gap(0)

	// An empty transcript is the first screen a user meets, so it says what to
	// do next instead of showing a blank region.
	empty := widgets.Hide(
		state.NewComputed(func() bool { return st.Turns.Get() == 0 }, st.Turns.AsReadonly()),
		sessionEmpty(sh),
	)

	return primitives.Box(
		primitives.VBox(empty, primitives.Expanded(sv)).
			CrossAlign(primitives.CrossAxisStretch),
	).Background(sh.P.Surface()), follow, scrollY
}

// sessionEmpty is what stands where the conversation will be. With no model
// it points at the Models tab; with one it names the model and says how to
// start.
func sessionEmpty(sh *app.Shell) widget.Widget {
	st := sh.Store

	loading := state.NewComputed(st.Loading, st.Loads.AsReadonly())
	head := state.NewComputed(func() string {
		if ls := st.Loads.Get(); len(ls) > 0 {
			return "Loading " + modelLabel(ls[0].Path)
		}
		if !st.Loaded.Get() {
			return "No model loaded"
		}
		if n := modelLabel(st.ModelPath.Get()); n != "" {
			return n + " is ready"
		}
		return "Ready"
	}, st.Loaded.AsReadonly(), st.ModelPath.AsReadonly(), st.Loads.AsReadonly())

	body := state.NewComputed(func() string {
		if ls := st.Loads.Get(); len(ls) > 0 {
			s := ls[0].Stage + "... A large model takes a few seconds; it stays open once it is."
			if len(ls) > 1 {
				s += fmt.Sprintf(" %d more waiting.", len(ls)-1)
			}
			return s
		}
		if !st.Loaded.Get() {
			return "Pick a model on the Models tab and load it. A GGUF file is " +
				"converted there first."
		}
		return "Type a message below and press Enter; Escape stops a reply. " +
			"Sampling settings on the right apply to the next reply."
	}, st.Loaded.AsReadonly(), st.Loads.AsReadonly())

	open := widgets.Hide(
		state.NewComputed(func() bool { return !st.Loaded.Get() && !loading.Get() },
			st.Loaded.AsReadonly(), loading),
		button.New(
			button.TextOpt("Open Models"),
			button.SizeOpt(button.Medium),
			button.VariantOpt(button.Filled),
			button.OnClick(sh.GoModels),
			button.PainterOpt(sh.P.Button),
		),
	)

	return primitives.Box(
		sh.P.Card(
			primitives.VBox(
				// The gap goes with the spinner: a hidden child keeps the row's
				// gap, and the title would sit indented from the text below it.
				primitives.HBox(
					widgets.Hide(loading, primitives.HBox(
						progress.New(
							progress.Indeterminate(true),
							progress.Size(20),
							progress.StrokeWidth(2.5),
							progress.PainterOpt(sh.P.Progress),
						),
						primitives.Box().Width(sh.P.Space.S),
					)),
					sh.P.Role(primitives.Text("").ContentSignal(head).Color(sh.P.Text()), sh.P.Type.TitleMedium),
				),
				widgets.NewParagraph("").
					ContentSignal(body).
					FontSize(sh.P.Type.BodyMedium.FontSize).
					LineHeight(sh.P.Type.BodyMedium.LineHeight/sh.P.Type.BodyMedium.FontSize).
					Color(sh.P.Muted()).
					Font(sh.P.Prose()),
				open,
			).Gap(sh.P.Space.S).CrossAlign(primitives.CrossAxisStart).
				Padding(sh.P.Space.M),
		),
	).PaddingXY(sh.P.Space.M, sh.P.Space.M)
}

// modelLabel is a container path as a person would name it.
func modelLabel(path string) string {
	if path == "" {
		return ""
	}
	return strings.TrimSuffix(filepath.Base(path), ".jlm")
}

// followSlack is how far from the bottom still counts as "following". It is
// one line of body text rather than zero because the content grows between
// the token that extends it and the frame that scrolls there.
const followSlack float32 = 24

// readingWidth is the widest the conversation sets a line: past ~90
// characters a reader loses the start of the next line. A wider window gets
// margins, not longer lines.
const readingWidth float32 = 760

// userBubbleWidth is the widest a message of yours grows before it wraps.
const userBubbleWidth float32 = 560

// reading centres a turn in a column readingWidth wide.
func reading(w widget.Widget) widget.Widget {
	return primitives.HBox(
		primitives.Expanded(primitives.Box()),
		primitives.Box(w).MaxWidthValue(readingWidth),
		primitives.Expanded(primitives.Box()),
	)
}

// sessionBubble renders one turn. Yours is a tinted bubble on the right; a
// reply is plain text on the page under the model's name, because it is the
// thing being read; an engine error is a card. A live bubble takes its body
// from Store.Stream so tokens appear as they arrive; every other bubble is
// static.
func sessionBubble(sh *app.Shell, idx int, t app.Turn, live bool) widget.Widget {
	if t.Role == app.RoleUser {
		return reading(userBubble(sh, t))
	}
	var bg widget.Color
	fg, who := sh.P.Text(), "assistant"
	switch t.Role {
	case app.RoleSystem:
		fg, who = sh.P.Muted(), "system"
	case app.RoleError:
		bg, fg, who = sh.P.Colors.ErrorContainer, sh.P.Colors.OnErrorContainer, "engine"
	}

	// An assistant turn is named by the model that produced it, since the model
	// can change mid-conversation; "assistant" is the fallback.
	//
	// The head re-reads the turn on every Store.Revision rather than capturing
	// it: a bubble is built when its empty placeholder is appended and Column
	// rebuilds only when the count moves, so fields SetTurn fills in later would
	// never reach the screen. Each field is a computed rather than a row rebuild,
	// which would destroy an in-progress selection.
	rev := sh.Store.Revision.AsReadonly()
	// A reply still being written names the model writing it: the turn's own
	// name arrives with its last token.
	name := state.NewComputed(func() string {
		t := sh.Store.Turn(idx)
		switch {
		case t.Role != app.RoleAssistant:
			return who
		case t.Model != "":
			return t.Model
		case live && sh.Store.ModelPath.Get() != "":
			return filepath.Base(sh.Store.ModelPath.Get())
		}
		return who
	}, rev, sh.Store.ModelPath.AsReadonly())
	nameFg := state.NewComputed(func() widget.Color {
		return turnColour(sh, sh.Store.Turn(idx), fg)
	}, rev)
	footer := state.NewComputed(func() string {
		return sessionTurnFooter(sh.Store.Turn(idx))
	}, rev)

	head := primitives.HBox(
		primitives.Expanded(headText(sh, name).ColorSignal(nameFg)),
		// The name is unbounded so it is Expanded; the footer is short and must
		// not be cut, so it is intrinsic. It is figures, so it is set in mono.
		headText(sh, footer).Intrinsic().Color(sh.P.Muted()).Font(sh.P.Mono()),
	).Gap(8)

	// A Paragraph, not primitives.Text: Text measures as wrapped but paints one
	// line. The reply is selectable, which works only because the transcript is
	// a column rather than a listview.
	var body widget.Widget = widgets.NewParagraph(t.Text).
		FontSize(sh.P.Type.BodyMedium.FontSize).Color(fg).
		LineHeight(bodyLineHeight).
		Font(sh.P.Prose()).
		SelectionColor(sh.P.Colors.PrimaryContainer).
		Selectable()
	var think widget.Widget = thinkingSection(sh, idx, t.Think, nil)
	if live {
		// The live bubble reads Store.Stream so tokens appear as they arrive, and
		// falls back to the turn's own text when the stream is empty: the last
		// assistant turn stays live for the rest of the session, and anything that
		// clears the stream would otherwise blank the reply.
		body = widgets.NewParagraph("").
			ContentSignal(state.NewComputed(func() string {
				if live := sh.Store.Stream.Get(); live != "" {
					return live
				}
				return t.Text
			}, sh.Store.Stream.AsReadonly())).
			FontSize(sh.P.Type.BodyMedium.FontSize).Color(fg).
			LineHeight(bodyLineHeight).
			Font(sh.P.Prose()).
			SelectionColor(sh.P.Colors.PrimaryContainer).
			Selectable()
		// The reasoning falls back to the turn's own copy for the same reason.
		think = thinkingSection(sh, idx, t.Think, state.NewComputed(func() string {
			if live := sh.Store.StreamThink.Get(); live != "" {
				return live
			}
			return t.Think
		}, sh.Store.StreamThink.AsReadonly()))
	}

	parts := []widget.Widget{head, think}
	if len(t.Images) > 0 {
		// The pictures come before the text because that is the order the
		// model received them in -- ChatMessage puts Images before Content.
		row := make([]widget.Widget, len(t.Images))
		for i, p := range t.Images {
			row[i] = sh.Thumbs.New(p, sessionTurnThumb)
		}
		parts = append(parts, primitives.HBox(row...).Gap(8))
	}
	parts = append(parts, body)
	if t.Role == app.RoleAssistant {
		parts = append(parts, sessionTurnActions(sh, idx, live))
	}
	turn := primitives.Box(
		primitives.VBox(parts...).Gap(6).CrossAlign(primitives.CrossAxisStretch),
	).PaddingXY(sh.P.Space.M, sh.P.Space.S+sh.P.Space.XS)
	if bg != (widget.Color{}) {
		turn = turn.Rounded(sh.P.Shape.Medium).Background(bg)
	}
	return reading(turn)
}

// userBubble is a message of yours: as wide as its text up to
// userBubbleWidth, against the right edge of the reading column. The pictures
// come before the text, the order the model receives them in.
func userBubble(sh *app.Shell, t app.Turn) widget.Widget {
	fg := sh.P.Colors.OnSecondaryContainer
	body := widgets.NewParagraph(t.Text).
		FontSize(sh.P.Type.BodyMedium.FontSize).Color(fg).
		LineHeight(bodyLineHeight).
		Font(sh.P.Prose()).
		SelectionColor(sh.P.Colors.PrimaryContainer).
		Selectable().
		Intrinsic()
	parts := []widget.Widget{}
	if len(t.Images) > 0 {
		row := make([]widget.Widget, len(t.Images))
		for i, p := range t.Images {
			row[i] = sh.Thumbs.New(p, sessionTurnThumb)
		}
		parts = append(parts, primitives.HBox(row...).Gap(8))
	}
	parts = append(parts, body)
	bubble := primitives.Box(primitives.VBox(parts...).Gap(8).CrossAlign(primitives.CrossAxisEnd)).
		PaddingXY(sh.P.Space.M, sh.P.Space.S+sh.P.Space.XS).
		Rounded(sh.P.Shape.Large).
		Background(sh.P.Colors.SecondaryContainer).
		MaxWidthValue(userBubbleWidth)
	return primitives.Box(primitives.HBox(primitives.Expanded(primitives.Box()), bubble)).
		PaddingXY(sh.P.Space.M, sh.P.Space.S)
}

// sessionTurnActions is the row under a reply: copy it, and -- on the last
// one -- write it again. A reply still being written has no actions yet.
func sessionTurnActions(sh *app.Shell, idx int, live bool) widget.Widget {
	st := sh.Store
	ic := iconColors(sh)
	copyBtn := widgets.NewIconButton(widgets.IconCopy(), "Copy the reply", func() {
		if t := st.Turn(idx).Text; t != "" {
			sh.Copy(t)
			sh.SetStatus("reply copied")
		}
	}, ic).Size(28)
	retry := widgets.NewIconButton(widgets.IconRetry(), "Write the reply again",
		func() { Regenerate(sh) }, ic).Size(28)
	last := state.NewComputed(func() bool { return idx == st.Turns.Get()-1 }, st.Turns.AsReadonly())
	row := primitives.HBox(copyBtn, widgets.Hide(last, retry)).Gap(2)
	if !live {
		return row
	}
	return widgets.Hide(sessionNot(st.Busy.AsReadonly()), row)
}

// Regenerate writes the last reply again: it drops the reply and the message
// that asked for it, and sends that message once more, pictures included, the
// way the Send button would. The engine renders a turn from the transcript
// alone, so nothing else needs undoing.
func Regenerate(sh *app.Shell) {
	st := sh.Store
	if st.Busy.Get() {
		return
	}
	u := st.Turns.Get() - 1
	for u >= 0 && st.Turn(u).Role != app.RoleUser {
		u--
	}
	if u < 0 {
		return
	}
	t := st.Turn(u)
	st.TruncateTurns(u)
	st.Draft.Set(t.Text)
	st.Attach.Set(t.Images)
	Submit(sh)
}

// bodyLineHeight is the conversation's leading: looser than the UI's, as
// running text wants.
const bodyLineHeight float32 = 1.6

// headText is one line of a bubble's head, bound to a signal. MaxLines(1)
// keeps a long model name from wrapping and pushing the reply down; see
// [line].
func headText(sh *app.Shell, sig state.ReadonlySignal[string]) *widgets.Paragraph {
	p := widgets.NewParagraph("").
		ContentSignal(sig).
		FontSize(sh.P.Type.LabelSmall.FontSize).
		Font(sh.P.Prose()).
		MaxLines(1)
	if sh.P.Type.LabelSmall.LineHeight > 0 && sh.P.Type.LabelSmall.FontSize > 0 {
		p = p.LineHeight(sh.P.Type.LabelSmall.LineHeight / sh.P.Type.LabelSmall.FontSize)
	}
	if sh.P.Type.LabelSmall.Bold {
		p = p.Bold()
	}
	return p
}

// sessionTurnFooter is the per-turn cost (token count and rate), or empty
// when the turn did not cost any. It carries both numbers or neither.
func sessionTurnFooter(t app.Turn) string {
	if t.Tokens <= 0 && t.TokPerSec <= 0 {
		return ""
	}
	if t.TokPerSec <= 0 {
		return strconv.Itoa(t.Tokens) + " tok"
	}
	return strconv.Itoa(t.Tokens) + " tok  " + app.Rate(t.TokPerSec)
}

// isLiveTurn says whether row i is the bubble a generation is filling: the
// last turn, an assistant one, in the chat the reply is written into -- any
// other chat's last reply is its own text, not the stream. It is a function so a test can gate it;
// getting it wrong means the answer silently never appears.
func isLiveTurn(st *app.Store, i int) bool {
	return st.Turn(i).Role == app.RoleAssistant && i == st.Turns.Get()-1 && st.ReplyShown()
}

// thinkingSection is the model's reasoning, collapsed because it is usually
// longer than the answer and sits above it. It uses core/collapsible, which
// draws hidden content inside an empty clip so stale textures are culled.
// A turn with no reasoning gets nothing, live or not; a live turn's section
// appears only once reasoning arrives.
func thinkingSection(sh *app.Shell, idx int, static string, sig state.ReadonlySignal[string]) widget.Widget {
	if sig == nil {
		if static == "" {
			return primitives.Box().Height(0)
		}
	}

	body := widgets.NewParagraph(static).
		FontSize(sh.P.Type.BodySmall.FontSize).Color(sh.P.Muted()).
		LineHeight(sh.P.Type.BodySmall.LineHeight / sh.P.Type.BodySmall.FontSize).
		Font(sh.P.Prose())
	if sig != nil {
		body = body.ContentSignal(sig)
	}

	// The title counts the reasoning's words, which also shows progress.
	title := state.NewComputed(func() string {
		s := static
		if sig != nil {
			s = sig.Get()
		}
		if s == "" {
			return "thinking…"
		}
		return fmt.Sprintf("thinking — %d words", len(strings.Fields(s)))
	}, thinkDep(sig))

	// Expanded + OnToggle + a watcher calling SetExpanded, not ExpandedSignal:
	// the widget's open state is an animation progress reconciled only at
	// Mount, in its own click, and in SetExpanded, and SetExpanded
	// short-circuits when current equals wanted. Bound to the signal, an
	// external write (the Reasoning chip, the engine) would never animate
	// open. With Expanded(bool) the widget keeps its own field, so such writes
	// differ from it; a click round-trips through OnToggle and the watcher's
	// call is then a no-op.
	open := sh.Store.ThinkOpen(idx)
	c := collapsible.New(
		collapsible.TitleReadonlySignal(title),
		collapsible.Content(primitives.Box(body).PaddingXY(8, 6)),
		collapsible.Expanded(open.Get()),
		collapsible.OnToggle(open.Set),
		collapsible.PainterOpt(sh.P.Collapsible),
	)
	// The watcher takes no space; it owns the subscription's lifetime, since
	// widgets.Column discards every row each time a turn is appended.
	sec := primitives.VBox(c, newSignalWatcher(open.AsReadonly(), c.SetExpanded)).Gap(0)
	if sig == nil {
		return sec
	}
	// A live turn's disclosure appears when its reasoning does.
	return widgets.Hide(state.NewComputed(func() bool {
		return sig.Get() != ""
	}, sig), sec)
}

// thinkDep is the signal the title recomputes from, or a constant one for a
// finished turn whose reasoning can no longer change.
func thinkDep(sig state.ReadonlySignal[string]) state.ReadonlySignal[string] {
	if sig != nil {
		return sig
	}
	return state.NewSignal("").AsReadonly()
}
