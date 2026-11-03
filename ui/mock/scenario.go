package mock

import (
	"fmt"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/common/crash"
	core "github.com/jitllm/jitllm/common/engine"
	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/screen"
)

// Step is one thing a person or the engine does, and the frame after it is
// what the step is judged by.
type Step struct {
	Name string
	Do   func(sh *app.Shell, e *Engine)
	// Until, when set, is waited for -- frame after frame -- before the step's
	// frame is taken: work the app finishes on its own goroutine, like a scan.
	Until func(sh *app.Shell) bool
	// Want says what the step must have done, or "" if it did: a step that
	// silently does nothing renders a clean frame of the wrong thing.
	Want func(sh *app.Shell) string
	// Shows is text the step's frame must paint: the store can be right while
	// the screen still shows what was there before.
	Shows []string
}

func loaded(name string) func(*app.Shell) string {
	return func(sh *app.Shell) string {
		if !sh.Store.Loaded.Get() || !strings.HasSuffix(sh.Store.ModelPath.Get(), name) {
			return "not loaded: " + sh.Store.ModelPath.Get()
		}
		return ""
	}
}

func streaming(sh *app.Shell) string {
	if !sh.Store.Streaming.Get() || sh.Store.Stream.Get()+sh.Store.StreamThink.Get() == "" {
		return "no reply is streaming"
	}
	return ""
}

func replied(sh *app.Shell) string {
	n := sh.Store.Turns.Get()
	if sh.Store.Busy.Get() || n == 0 || sh.Store.Turn(n-1).Text == "" {
		return "the reply did not land in its turn"
	}
	return ""
}

// Scenario is a run of steps on one tab, starting from a window with nothing
// loaded: the order the app meets its data in.
type Scenario struct {
	Name  string
	Tab   int
	Steps []Step
}

func load(m Model) func(*app.Shell, *Engine) {
	return func(_ *app.Shell, e *Engine) { e.Load(m.Entry.Path) }
}

func ask(text string) func(*app.Shell, *Engine) {
	return func(sh *app.Shell, _ *Engine) {
		sh.Store.Draft.Set(text)
		screen.Submit(sh)
	}
}

func steps(n int) func(*app.Shell, *Engine) {
	return func(_ *app.Shell, e *Engine) {
		for i := 0; i < n && e.Step(); i++ {
		}
	}
}

func finish(_ *app.Shell, e *Engine) {
	for e.Step() {
	}
}

// Scenarios is every scripted run the stage tests check and the shots render.
func Scenarios() []Scenario {
	return []Scenario{
		{Name: "session", Tab: app.TabSession, Steps: []Step{
			{Name: "loaded", Do: load(Llama()), Want: loaded("Llama-3.2-1B-Instruct-Q4_K_M.jlm")},
			{Name: "streaming", Do: func(sh *app.Shell, e *Engine) {
				ask("What is the capital of France?")(sh, e)
				steps(8)(sh, e)
			}, Want: streaming},
			{Name: "replied", Do: finish, Want: replied},
			{Name: "second-reply", Do: func(sh *app.Shell, e *Engine) {
				ask("And how many people live there?")(sh, e)
				finish(sh, e)
			}, Want: func(sh *app.Shell) string {
				if n := sh.Store.Turns.Get(); n != 4 {
					return fmt.Sprintf("%d turn(s), want 4", n)
				}
				return replied(sh)
			}},
			{Name: "regenerated", Do: func(sh *app.Shell, e *Engine) {
				screen.Regenerate(sh)
				finish(sh, e)
			}, Want: func(sh *app.Shell) string {
				if n := sh.Store.Turns.Get(); n != 4 {
					return fmt.Sprintf("%d turn(s) after regenerating, want 4: the old reply was kept", n)
				}
				if got := sh.Store.Turn(2).Text; got != "And how many people live there?" {
					return "the message asked again is " + got
				}
				return replied(sh)
			}},
		}},
		{Name: "settings", Tab: app.TabSettings, Steps: []Step{
			{Name: "page", Shows: []string{"Engine", "Auto", "Service address", "Embedded · API off"}},
			// The badge and the Engine card for a window connected to a jitllmd.
			{Name: "connected", Do: func(sh *app.Shell, _ *Engine) {
				sh.Store.EngineMode.Set(core.ModeService)
				sh.Store.Engine.Set(core.Where{Service: true, Addr: "localhost:4242", Version: "0.9.0"})
			}, Shows: []string{"Service · localhost:4242 (jitllmd 0.9.0)"}},
			{Name: "lost", Do: func(sh *app.Shell, _ *Engine) {
				sh.Store.Engine.Set(core.Where{Service: true, Addr: "localhost:4242", Version: "0.9.0", Lost: true})
			}, Shows: []string{"Service · not answering · localhost:4242 (jitllmd 0.9.0)"}},
		}},
		{Name: "chats", Tab: app.TabSession, Steps: []Step{
			{Name: "loaded", Do: load(Llama()), Want: loaded("Llama-3.2-1B-Instruct-Q4_K_M.jlm")},
			{Name: "first", Do: func(sh *app.Shell, e *Engine) {
				ask("What is the capital of France?")(sh, e)
				finish(sh, e)
			}, Want: wantChats("What is the capital of France?")},
			{Name: "new", Do: func(sh *app.Shell, _ *Engine) { sh.Store.NewChat() },
				Want: func(sh *app.Shell) string {
					if n := sh.Store.Turns.Get(); n != 0 || sh.Store.ChatSel.Get() != 0 {
						return fmt.Sprintf("a new chat shows %d turn(s) at %d", n, sh.Store.ChatSel.Get())
					}
					return wantChats("", "What is the capital of France?")(sh)
				}},
			{Name: "other-model", Do: load(Qwen()), Want: loaded("Qwen3-30B-A3B-Q4_K_M.jlm")},
			{Name: "second-model", Do: func(sh *app.Shell, e *Engine) {
				ask("Write a haiku about paging")(sh, e)
				finish(sh, e)
			}, Want: func(sh *app.Shell) string {
				if w := wantChats("Write a haiku about paging", "What is the capital of France?")(sh); w != "" {
					return w
				}
				return loaded("Qwen3-30B-A3B-Q4_K_M.jlm")(sh)
			}},
			{Name: "back", Do: func(sh *app.Shell, _ *Engine) { sh.Store.SelectChat(1) },
				Shows: []string{"What is the capital of France?"},
				Want: func(sh *app.Shell) string {
					if got := sh.Store.Turn(0).Text; got != "What is the capital of France?" || sh.Store.Turns.Get() != 2 {
						return fmt.Sprintf("the first chat came back as %d turn(s) starting %q", sh.Store.Turns.Get(), got)
					}
					// Its model comes back with it.
					return loaded("Llama-3.2-1B-Instruct-Q4_K_M.jlm")(sh)
				}},
		}},
		{Name: "switch-streaming", Tab: app.TabSession, Steps: []Step{
			{Name: "loaded", Do: load(Llama()), Want: loaded("Llama-3.2-1B-Instruct-Q4_K_M.jlm")},
			{Name: "asked", Do: func(sh *app.Shell, e *Engine) {
				ask("What is the capital of France?")(sh, e)
				steps(4)(sh, e)
			}, Want: streaming},
			// Another chat while the reply is being written: it stays empty, and
			// the reply goes on into the chat that asked.
			{Name: "other-chat", Do: func(sh *app.Shell, e *Engine) {
				sh.Store.NewChat()
				steps(8)(sh, e)
			}, Want: func(sh *app.Shell) string {
				if n := sh.Store.Turns.Get(); n != 0 || sh.Store.ChatSel.Get() != 0 {
					return fmt.Sprintf("the new chat shows %d turn(s) at %d", n, sh.Store.ChatSel.Get())
				}
				return streaming(sh)
			}},
			{Name: "back-streaming", Do: func(sh *app.Shell, e *Engine) {
				sh.Store.SelectChat(1)
				steps(4)(sh, e)
			}, Shows: []string{"What is the capital of France?"},
				Want: func(sh *app.Shell) string {
					if n := sh.Store.Turns.Get(); n != 2 {
						return fmt.Sprintf("the asking chat came back with %d turn(s), want 2", n)
					}
					return streaming(sh)
				}},
			{Name: "away-finished", Do: func(sh *app.Shell, e *Engine) {
				sh.Store.SelectChat(0)
				finish(sh, e)
			}, Want: func(sh *app.Shell) string {
				if n := sh.Store.Turns.Get(); n != 0 {
					return fmt.Sprintf("the other chat holds %d turn(s) after the reply finished, want 0", n)
				}
				return ""
			}},
			{Name: "back-replied", Do: func(sh *app.Shell, _ *Engine) { sh.Store.SelectChat(1) },
				Want: func(sh *app.Shell) string {
					if got := sh.Store.Turn(1).Text; got != strings.Join(strings.Fields(Llama().Reply), " ") {
						return fmt.Sprintf("the asking chat's reply is %q, want the whole reply", got)
					}
					return replied(sh)
				}},
		}},
		{Name: "loading", Tab: app.TabSession, Steps: []Step{
			{Name: "reading", Do: func(sh *app.Shell, e *Engine) {
				e.Hold = true
				load(Llama())(sh, e)
				load(Qwen())(sh, e)
			}, Want: wantLoads("Reading the model", "Waiting")},
			{Name: "placing", Do: func(_ *app.Shell, e *Engine) { e.Stage("Placing layers on the GPU") },
				Want: wantLoads("Placing layers on the GPU", "Waiting")},
			{Name: "second", Do: func(_ *app.Shell, e *Engine) { e.Release() },
				Want: func(sh *app.Shell) string {
					if w := wantLoads("Reading the model")(sh); w != "" {
						return w
					}
					return loaded("Llama-3.2-1B-Instruct-Q4_K_M.jlm")(sh)
				}},
			{Name: "loaded", Do: func(_ *app.Shell, e *Engine) { e.Release() },
				Want: func(sh *app.Shell) string {
					if w := wantLoads()(sh); w != "" {
						return w
					}
					if n := len(sh.Store.Models.Get()); n != 2 {
						return fmt.Sprintf("%d model(s) open, want 2", n)
					}
					return loaded("Qwen3-30B-A3B-Q4_K_M.jlm")(sh)
				}},
		}},
		{Name: "thinking", Tab: app.TabSession, Steps: []Step{
			{Name: "loaded", Do: load(Qwen()), Want: loaded("Qwen3-30B-A3B-Q4_K_M.jlm")},
			{Name: "reasoning", Do: func(sh *app.Shell, e *Engine) {
				ask("How do I write an HTTP server in Rust?")(sh, e)
				steps(12)(sh, e)
			}, Want: func(sh *app.Shell) string {
				if sh.Store.StreamThink.Get() == "" || sh.Store.Stream.Get() != "" {
					return "not in the reasoning: think " + sh.Store.StreamThink.Get()
				}
				return ""
			}},
			{Name: "answering", Do: steps(30), Want: func(sh *app.Shell) string {
				if sh.Store.Stream.Get() == "" {
					return "the answer has not started"
				}
				return streaming(sh)
			}},
			{Name: "replied", Do: finish, Want: replied},
		}},
		{Name: "problem", Tab: app.TabSession, Steps: []Step{
			{Name: "banner", Do: func(sh *app.Shell, e *Engine) {
				load(Llama())(sh, e)
				sh.Post(func() {
					sh.Store.Problem.Set(app.Problem{Seq: 1,
						Title:  "This model was converted by an older jitllm",
						Detail: "gemma-2b-v13.jlm is container format 13 and this build reads a newer one. Reconvert it from the file it came from.",
						Action: "Reconvert", Do: func() {}})
				})
			}},
		}},
		{Name: "machine", Tab: app.TabMachine, Steps: []Step{
			{Name: "llama", Do: load(Llama()), Want: loaded("Llama-3.2-1B-Instruct-Q4_K_M.jlm")},
			{Name: "relocated", Do: func(_ *app.Shell, e *Engine) { e.Relocate(10) }, Want: func(sh *app.Shell) string {
				if a := sh.Store.Alloc.Get(); a.DeviceBlocks != 10 {
					return fmt.Sprintf("%d block(s) on the device, want 10", a.DeviceBlocks)
				}
				return ""
			}},
			{Name: "qwen", Do: load(Qwen()), Want: func(sh *app.Shell) string {
				if len(sh.Store.Models.Get()) != 2 {
					return "the first model was not kept open"
				}
				return loaded("Qwen3-30B-A3B-Q4_K_M.jlm")(sh)
			}},
			{Name: "back-to-llama", Do: func(_ *app.Shell, e *Engine) { e.Use(Llama().Entry.Path) },
				Want: loaded("Llama-3.2-1B-Instruct-Q4_K_M.jlm")},
		}},
		{Name: "models", Tab: app.TabModels, Steps: []Step{
			{Name: "scanned", Until: func(sh *app.Shell) bool { return sh.Store.CatalogRows.Get() >= len(Models()) }},
			{Name: "selected", Do: func(sh *app.Shell, _ *Engine) { sh.Store.CatalogSel.Set(1) }},
			{Name: "loaded", Do: load(Qwen()), Want: loaded("Qwen3-30B-A3B-Q4_K_M.jlm")},
		}},
		{Name: "discover", Tab: app.TabDownload, Steps: []Step{
			{Name: "probed", Until: func(sh *app.Shell) bool { return sh.Store.Machine.Get().Probed },
				Want: func(sh *app.Shell) string {
					if sh.Store.Machine.Get().MemWall <= 0 {
						return "the machine card has no measured bandwidth"
					}
					return ""
				}},
			{Name: "smartest", Do: func(sh *app.Shell, _ *Engine) { screen.SetBalance(sh, 4) }},
			{Name: "streaming", Do: func(sh *app.Shell, _ *Engine) {
				screen.ShowDownload(sh, 0.42, "Downloading and converting as it arrives...")
			}, Shows: []string{"Stop"}},
		}},
		{Name: "convert", Tab: app.TabConvert, Steps: []Step{
			{Name: "listed", Until: func(sh *app.Shell) bool { return sh.Store.SourcesRows.Get() == len(Sources()) },
				Shows: []string{"Not converted", "Converted", "Goes with its model"}},
			{Name: "picked", Do: func(sh *app.Shell, _ *Engine) { screen.OfferConvert(sh, Sources()[0].Path) },
				Want: func(sh *app.Shell) string {
					if e, ok := sh.Store.SelectedSource(); !ok || e.Path != Sources()[0].Path {
						return "the picked file is not selected in the list"
					}
					return ""
				}},
		}},
		{Name: "crash", Tab: app.TabSession, Steps: []Step{
			{Name: "reported", Do: func(sh *app.Shell, _ *Engine) {
				// A real recovered panic on a goroutine, as a worker's would be.
				sh.Crashes().OnReport(sh.ShowCrash)
				go func() {
					defer sh.Crashes().Recover("a mock worker")
					var rows []int
					fmt.Println(rows[3])
				}()
			}, Until: func(sh *app.Shell) bool { return sh.Store.Crash.Get().Text != "" },
				Shows: []string{"jitllm crashed", "Copy report", "Open an issue", "Dismiss", "index out of range"}},
			{Name: "dismissed", Do: func(sh *app.Shell, _ *Engine) {
				sh.Crashes().OnReport(nil)
				sh.Store.Crash.Set(crash.Report{})
			}, Want: func(sh *app.Shell) string {
				if sh.Store.Crash.Get().Text != "" {
					return "the report is still up"
				}
				return ""
			}},
		}},
	}
}

// wantLoads checks the loads shown, in order, by stage.
func wantLoads(stages ...string) func(*app.Shell) string {
	return func(sh *app.Shell) string {
		ls := sh.Store.Loads.Get()
		got := make([]string, len(ls))
		for i, l := range ls {
			got[i] = l.Stage
		}
		if !slices.Equal(got, stages) {
			return fmt.Sprintf("loads shown %q, want %q", got, stages)
		}
		return ""
	}
}

// wantChats checks the chat list's titles, newest first.
func wantChats(titles ...string) func(*app.Shell) string {
	return func(sh *app.Shell) string {
		cs := sh.Store.Chats.Get()
		got := make([]string, len(cs))
		for i, c := range cs {
			got[i] = c.Title
		}
		if !slices.Equal(got, titles) {
			return fmt.Sprintf("chats %q, want %q", got, titles)
		}
		return ""
	}
}
