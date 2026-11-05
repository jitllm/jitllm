# jitllm desktop

A desktop front end for jitllm: an alternative to the CLI, not a wrapper around
it. The engine is driven **in process** — shelling out to the `jitllm` binary
cannot hold a model open across turns, which is the whole point of the session
screen, and it turns structured counters back into text to scrape.

## Why this is its own module

`github.com/jitllm/jitllm` stays lean: its `go.mod` has one dependency and
this GUI must not be the reason that changes. The UI lives in
`github.com/jitllm/jitllm/ui`, which requires the engine through a
`replace` directive pointing at `../`.

## Build and run

Every build and test goes through the repo's cgroup cap. A bare `go build` is
refused by a hook.

```
cd ui && ../scripts/cap 8G -- go build ./...
cd ui && ../scripts/cap 8G -- go test ./... -count=1
cd ui && ../scripts/cap 8G -- go run .
```

`8G`, not more. A full build of this module is roughly a million lines of Go
and an ~18 MB binary.

**Never build this module while a board row is being taken.** A compile is
invisible to every measurement gate in this repo and it lands mid-round; the
project's own evidence records a subagent compiling Go in the same tree as the
cause of a nine-point phantom regression.

## Packaging: a desktop application, not a command-line program

`cmd/pack` (pure Go, runs on any host) makes the binary an application on the
two systems that tell one from a terminal program by its packaging:

```
cd ui && ../scripts/cap 8G -- go run ./cmd/pack syso -arch amd64 -version 1.2.3
cd ui && GOOS=windows GOARCH=amd64 ../scripts/cap 8G -- go build -ldflags -H=windowsgui -o jitllm-ui.exe .
cd ui && ../scripts/cap 8G -- go run ./cmd/pack check jitllm-ui.exe
cd ui && ../scripts/cap 8G -- go run ./cmd/pack app -bin jitllm-ui -version 1.2.3 -o dist-app/jitllm.app
```

- **Windows**: linked with `-H=windowsgui`, so no console window opens beside
  it, and with `rsrc_windows_<arch>.syso` beside `main.go` the linker embeds
  the icon and the version information. `check` refuses an .exe whose
  subsystem is not GUI or whose resources are missing.
- **macOS**: `jitllm.app` with Info.plist, the `.icns` and the binary under
  `Contents/MacOS`, and with `-cli` the CLI and the server under
  `Contents/Resources/bin`, which the app offers to link into PATH
  (`install/`, `screen/cli.go`; `packaging/README.md` has the dmg and the
  cask); on a Mac it is signed ad-hoc under the hardened runtime
  with `.github/release/entitlements.plist`. gogpu sets the regular activation
  policy and brings the window to the front itself.
- Every icon is drawn from `app.IconAt`, the window icon, so there is no image
  file to keep in step.
- With no console to read it (a GUI .exe from Explorer, an app from Finder),
  the log goes to `jitllm-ui.log` beside the settings file. A GUI .exe run
  from cmd or PowerShell writes to that terminal.
- A crash is a report (package `common/crash`): the runtime writes a fatal
  error's traceback, every goroutine's, to `crash.log` beside the settings
  (`%AppData%\jitllm` on Windows, `~/Library/Application Support/jitllm` on
  macOS, `~/.config/jitllm` on Linux), and the next launch shows it over every
  screen with Copy report, Open an issue and Dismiss, saved as
  `crash-report.txt`. A panic on one of the app's own goroutines (the engine
  worker, the model scan, the hardware probe, a download) is reported the same
  way while the window stays up. Home paths are shortened to `~`; no prompt or
  chat text is in a report.

`.goreleaser.yaml` runs the same steps per target; CI's `ui` job runs them on
its Windows and macOS hosts.

## Do not float the toolkit

The versions in `go.mod` are pinned and `go get -u` breaks the build. `gg` at
**v0.52.5** pulls `gputypes v0.8.0`, whose struct-argument `Draw` does not
compile against `gg`. Pinning `gg` to **v0.52.3** — what `ui v0.1.54` itself
requires — resolves `gputypes v0.5.2` and builds.

`CGO_ENABLED=0` is the default and must stay that way: enabling cgo breaks
`goffi`.

## Environment

The toolkit reads a few debug switches of its own:

| variable | effect |
|---|---|
| `GOGPU_DEBUG_LAYOUT=1` | re-runs each cached layout and asserts the result matches |
| `GOGPU_DEBUG_STAMP=1` | logs every screen-origin stamp |
| `GOGPU_DEBUG_DIRTY=1` | overlays the dirty-widget regions |

jitllm's own `JITLLM_*` names are read by `cmd/jitllm`, not by the libraries.
This app configures the engine through the `With...()` options instead, which
is the caller that option API was written for.

## Layout

```
main.go            window, screen registration, drag-and-drop, desktop.Run
app/               the shell: window, navigation, shared state, the async->UI bridge
widgets/           composite widgets gogpu/ui does not ship (no app imports)
screen/            the pages -- Chat, Discover, Models, Convert, Machine -- and the interfaces they reach the world through (deps.go)
engine/            attaches common/engine to the window: a Front over the shell, a State over the Store's signals
mock/              stand-ins for the engine, the hardware and the files; the scenarios
stage/             headless frames of the mounted window, and what is wrong with them
cmd/shots/         every scenario's frames as PNGs
cmd/pack/          the Windows resource and GUI check, the macOS bundle, the installer icon
install/           the macOS app linking its CLIs into PATH and its launchd agent
```

The dependency direction is fixed and has no cycle:

```
widgets   -> gogpu/ui, common/session (the Placement type)
app       -> widgets, common/catalog, common/session
engine    -> app, screen, common/engine   (hands the worker the Store's signals)
screen    -> app, widgets, common/hardware, common/catalog
mock      -> app, screen, widgets, common/hardware, common/catalog
stage     -> app, screen, mock
main      -> all of the above
```

The GUI-free half lives in the `common/` module, shared with the terminal UI
(`tui/`): `common/catalog` (model-file enumeration and the cheap container
probe), `common/hardware` (the machine probe), `common/session` (the
transcript, request and report types the screens read) and `common/engine`
(the single worker goroutine that owns every jitllm object, reporting through
its `Front` interface and `State` values). None of it imports gogpu.

## What works today

Every tab is live: `main.go` installs the screens with `screen.Install` and
attaches the engine, and closes it on shutdown. Loading, chatting (text and
pictures), converting, switching between open models and moving the seam all
run through `common/engine`.

## Everything outside the widgets is an interface

The screens reach the world through three interfaces in `screen/deps.go` --
`Engine`, `Machine` (the hardware probe) and `Files` (the model catalog) --
bound per shell with `screen.Attach`. `engine.Engine`, `hardware.Host` and
`catalog.Disk` are the real ones; package `mock` implements all three with
fixtures, publishing through `Shell.Post` exactly as the engine does. The
converter and the downloader are still called directly.

```
cd ui && ../scripts/cap 8G -- go run . -mock       # the app, no model, no GPU
cd ui && ../scripts/cap 8G -- go run ./cmd/shots -out /tmp/shots
```

## Checking what it looks like

`mock.Scenarios()` are scripted runs -- load, send, stream, reply, relocate,
switch models -- each starting from an empty window. `stage` plays them: it
mounts the real window once, drains the Post queue and lays out before every
frame as `OnUpdate` does, and keeps the tree between frames. That order matters:
a fixture set before the first layout hides every widget that fails to
re-measure when its data arrives, which is how the memory map shipped drawing
over the rows below it.

Each frame is checked for what reads as broken: text painted over text, text
outside its clip (the toolkit does not clip text, UPSTREAM.md #8), text past the
window -- counting only what is still visible after later opaque fills. Each
step also states what it must have done (`Step.Want`), because a step that did
nothing renders a clean frame. `stage.TestEveryScenarioRendersClean` runs every
scenario at 1440x900 and 1000x700 in both themes; `cmd/shots` writes the same
frames and prints the same problems.

## The concurrency rules

Only four things in either library are safe off the UI goroutine, and each was
verified in source: `state.Signal.Set/Get/Update` (RWMutex),
`gogpuApp.RequestRedraw()`, `linechart.Widget.PushValue`, and
`state.Scheduler.MarkDirty/Flush/Batch`. Across all 27 `core/` packages only
two files contain a mutex at all.

`examples/taskmanager/main.go:289` in gogpu/ui calls `progressbar.SetValue` —
which has no lock — from a ticker goroutine. **That shipped example is racy.
Do not copy it.**

1. One engine goroutine owns every jitllm object. `engine.State` has no lock.
2. The UI goroutine never calls jitllm. Click handlers send commands.
3. Values flow out as signals. Never call a widget mutator from a worker.
4. Actions flow out through `Shell.Post`, drained in `OnUpdate`.
5. Dialogs are raised with `Shell.Alert` / `Shell.Confirm`, from any goroutine:
   they publish `Store.Dialog` through `Shell.Post`, and `widgets.DialogHost`
   shows them from its per-frame `TickAnimation`, where a `widget.Context`
   exists.
6. Token streaming is coalesced in the worker to 30 Hz.
7. Native file dialogs and the clipboard are UI-goroutine only.
