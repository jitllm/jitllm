// Command tui is jitllm's terminal interface, driven through the same engine
// and the same settings and chats as the window (common/...).
//
//	tui [-chat=true|false] [-device auto] [-ctx 8192] [-models dir] [-mouse] [--connect addr | --embedded] [model.jlm]
//
// A flag overrides its setting for this run. The engine is a jitllmd that
// answers at the configured address (the login service's by default) and the
// app's own only when none does; --connect and --embedded force one. With no model it opens on the
// models screen.
//
//	tui demo relocate [-connect addr] [-model name] [-devices ids] [-auto]
//
// is the relocation demo (demo.go): a model's blocks and their history moving
// between the host and the devices of a running jitllmd while a reply streams.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/jitllm/jitllm/common/config"
	"github.com/jitllm/jitllm/common/engine"
	"github.com/jitllm/jitllm/internal/localserver"
	"github.com/muesli/termenv"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "demo" {
		os.Exit(runDemo(os.Args[2:]))
	}
	config.UseEnv(config.Env{Models: os.Getenv("JITLLM_MODELS"), DataHome: os.Getenv("XDG_DATA_HOME"),
		APIKey: os.Getenv("JITLLM_API_KEY")})
	cfg := config.LoadConfig()
	chat := flag.Bool("chat", cfg.Chat, "apply the model's chat template")
	device := flag.String("device", cfg.DeviceSpec, "device spec, as the window's setting")
	ctx := flag.Int("ctx", cfg.MaxSeq, "context length")
	dir := flag.String("models", "", "list this folder instead of the configured ones")
	mouse := flag.Bool("mouse", false, "scroll with the mouse wheel (the terminal's own text selection then needs shift)")
	connectTo := flag.String("connect", "", "connect to the jitllmd at this address and nothing else")
	embedded := flag.Bool("embedded", false, "run this app's own engine, even where a jitllmd answers")
	flag.Parse()
	if flag.NArg() > 1 || (*connectTo != "" && *embedded) {
		fmt.Fprintln(os.Stderr, "usage: tui [-chat=true|false] [-device auto] [-ctx 8192] [-models dir] [-mouse] [--connect addr | --embedded] [model.jlm]")
		os.Exit(2)
	}
	mode, addr := cfg.Engine, cfg.ServiceAddr
	switch {
	case *connectTo != "":
		mode, addr = engine.ModeService, *connectTo
	case *embedded:
		mode = engine.ModeEmbedded
	}
	dirs := cfg.ModelDirs
	if *dir != "" {
		dirs = []string{*dir}
	}
	cfg.Chat, cfg.DeviceSpec, cfg.MaxSeq = *chat, *device, *ctx

	f := newFront()
	f.st = newState(f, cfg.DeviceSpec, cfg.Placement, cfg.MaxSeq, cfg.Chat, !cfg.NoKVCache)
	// Service first: under Auto a jitllmd that answers is used, and this app
	// runs its own engine only when none does. Asked before the API starts
	// (root.Init), or Auto would find this app.
	ch := engine.Choose(context.Background(), mode, addr, localserver.WithAPIKey(cfg.ServiceAPIKey()))
	var r *root
	var e engine.Driver
	if ch.Conn != nil {
		e = engine.NewRemote(f, f.st, ch.Conn, ch.Addr, func(w engine.Where) {
			f.Post(func() {
				if r != nil {
					r.env.where = w
				}
			})
		})
	} else {
		e = engine.New(f, f.st)
	}
	defer e.Close()
	if ch.Warning != "" {
		f.st.Status.Set(ch.Warning)
	}
	r = newRoot(f, e, cfg, dirs)
	if rem, ok := e.(*engine.Remote); ok {
		r.env.where = rem.Where()
	}
	defer func() {
		if r.env.api != nil {
			r.env.api.Close()
		}
	}()
	probeMachine(r.env)
	if flag.NArg() == 1 {
		r.cur = scrChat
		e.Load(flag.Arg(0))
	}

	opts := []tea.ProgramOption{tea.WithAltScreen()}
	if *mouse {
		opts = append(opts, tea.WithMouseCellMotion())
	}
	// The app paints on its own ground, whatever the terminal's theme: the
	// terminal's background is set for the run and reset to its default on
	// exit (OSC 111).
	termenv.NewOutput(os.Stdout).SetBackgroundColor(termenv.RGBColor(string(cBg)))
	defer fmt.Print("\x1b]111\x07")

	p := tea.NewProgram(program{r}, opts...)
	go func() {
		for msg := range f.queue {
			p.Send(msg)
		}
	}()
	if _, err := p.Run(); err != nil {
		fmt.Print("\x1b]111\x07")
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
