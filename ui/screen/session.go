package screen

import (
	"github.com/gogpu/ui/event"
	"strings"

	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/widgets"
)

// The chat screen. It holds nothing itself: one *model.Model and one
// *model.Instance stay open in the engine across many turns, and this screen is
// the prompt box, the streaming view and the stats strip around them.

// ChatRequest and ChatMessage are re-exported; see [session.ChatRequest].
type (
	ChatRequest = session.ChatRequest
	ChatMessage = session.ChatMessage
)

// SessionOptions is what an integrator hands the screen.
type SessionOptions struct {
	// Side is the tuning panel ([Tuning]), shown beside the conversation.
	// Nil leaves the conversation the whole tab.
	Side app.ScreenFunc
	// MaxTokens is the decode cap. Zero takes DefaultMaxTokens.
	MaxTokens int
}

// tuningWidth is the sampling panel's width beside the chat.
const tuningWidth = 340

// DefaultMaxTokens is re-exported; see [session.DefaultMaxTokens].
const DefaultMaxTokens = session.DefaultMaxTokens

// Session is the [app.ScreenFunc] main registers. It builds the screen with no
// engine attached, which is what makes the shell runnable before the engine
// exists; use [SessionWith] once there is one.
func Session(sh *app.Shell) widget.Widget { return SessionWith(SessionOptions{})(sh) }

// SessionWith returns the screen with, optionally, the runtime panel beside
// it. The engine is the shell's ([Attach]):
//
//	sh.Register(app.TabNames()[app.TabSession], screen.SessionWith(screen.SessionOptions{
//	    Side: screen.Runtime,
//	}))
//
// It is a constructor because an app.ScreenFunc may be called again (a theme
// swap rebuilds the tree).
func SessionWith(o SessionOptions) app.ScreenFunc {
	if o.MaxTokens <= 0 {
		o.MaxTokens = DefaultMaxTokens
	}
	return func(sh *app.Shell) widget.Widget {
		conv := sessionColumn(sh, o)
		if o.Side == nil {
			// CrossAlign stretch, or the column is laid out at its content's width and
			// the stats strip's background stops half way. Expanded would distribute
			// the main axis, which here is height.
			return primitives.Box(conv).
				CrossAlign(primitives.CrossAxisStretch).
				Background(sh.P.Background())
		}
		// The panel is a fixed width the header's sliders button opens and
		// closes: the conversation is what the page is for, so it takes the rest.
		panel := primitives.HBox(
			primitives.Box().Width(1).Background(sh.P.Colors.OutlineVariant),
			primitives.Box(o.Side(sh)).Width(tuningWidth),
		)
		return primitives.Box(primitives.HBox(
			primitives.Expanded(conv),
			widgets.Hide(sh.Store.ShowTuning.AsReadonly(), panel),
		)).Background(sh.P.Background())
	}
}

// sessionColumn is the chat column: header, transcript, the in-flight reply,
// the stats strip and the composer.
func sessionColumn(sh *app.Shell, o SessionOptions) widget.Widget {
	tr, follow, _ := sessionTranscript(sh)

	// The key watcher is last because primitives.Box offers non-mouse events to
	// children in reverse order, so it sees Escape before the composer. It claims
	// the key only while a reply is in flight.
	stop := newKeyWatcher(event.KeyEscape, func() bool {
		eng := depsOf(sh).Engine
		if !sh.Store.Busy.Get() || eng == nil {
			return false
		}
		eng.Stop()
		sh.SetStatus("stopped")
		return true
	})

	return primitives.VBox(
		sessionHeader(sh),
		sessionProblem(sh),
		primitives.Expanded(tr),
		sessionComposer(sh, o),
		follow,
		stop,
		newSignalWatcher(sh.Store.ChatSel.AsReadonly(), func(int) { chatModel(sh) }),
	).CrossAlign(primitives.CrossAxisStretch)
}

// chatModel brings back the model the chat on screen was talking to, when it
// is open and another one is running. One that was closed is not reopened:
// that is a load, and a click on a chat should not start one.
func chatModel(sh *app.Shell) {
	st := sh.Store
	c, ok := st.ChatAt(st.ChatSel.Get())
	eng := depsOf(sh).Engine
	if !ok || c.Model == "" || c.Model == st.ModelPath.Get() || eng == nil {
		return
	}
	for _, m := range st.Models.Get() {
		if m.Path == c.Model {
			eng.Use(c.Model)
			return
		}
	}
}

// sessionSend builds a request out of the draft and hands it to the engine.
// It runs on the UI goroutine and does only Store writes and one non-blocking
// engine call.
func sessionSend(sh *app.Shell, o SessionOptions) {
	st := sh.Store

	text := strings.TrimSpace(st.Draft.Get())
	imgs := st.Attach.Get()
	if text == "" && len(imgs) == 0 {
		return
	}
	eng := depsOf(sh).Engine
	if eng == nil {
		sh.SetStatus("no engine is attached: see screen.Attach")
		return
	}
	if !st.Loaded.Get() {
		sh.SetStatus("load a model first")
		sh.GoModels()
		return
	}
	if st.Busy.Get() {
		return
	}
	if len(imgs) > 0 && !st.Vision.Get() {
		sh.Alert("This model cannot see pictures",
			"The loaded model has no vision tower. Load a vision model (one converted "+
				"with its mmproj), or remove the attached picture.")
		return
	}

	// The history is read before the new turns are appended, so the empty
	// assistant placeholder never reaches the model.
	hist := st.AllTurns()
	chat := st.Chat.Get()

	st.SetChatModel(st.ModelPath.Get())
	st.AppendTurn(app.Turn{Role: app.RoleUser, Text: text, Images: imgs})
	reply := st.AppendTurn(app.Turn{Role: app.RoleAssistant})
	st.MarkReply()
	st.Draft.Set("")
	st.Attach.Set(nil)
	// Clear both halves of the live bubble, or the new turn shows the previous
	// reply's reasoning until its own arrives.
	st.Stream.Set("")
	st.StreamThink.Set("")

	req := ChatRequest{
		Chat:      chat,
		Prompt:    text,
		Reply:     reply,
		MaxTokens: o.MaxTokens,
		Sampling:  st.Sampling.Get(),
	}
	switch {
	case chat:
		req.Messages = ChatMessages(st.System.Get(), hist, text, imgs...)
	case len(imgs) > 0:
		// A picture goes through the model's template even in completion mode, as
		// the CLI's -image does. One turn, no history, no system prompt.
		req.Messages = ChatMessages("", nil, text, imgs...)
	}
	eng.Send(req)
}

// ChatMessages is re-exported; see [session.ChatMessages].
func ChatMessages(system string, turns []app.Turn, prompt string, images ...string) []ChatMessage {
	return session.ChatMessages(system, turns, prompt, images...)
}

// Submit sends the composer's draft as the Send button does, with the default
// decode cap. It is how a scripted run (package mock's scenarios) presses Send.
func Submit(sh *app.Shell) { sessionSend(sh, SessionOptions{MaxTokens: DefaultMaxTokens}) }
