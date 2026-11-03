package screen

import (
	"errors"
	"os"
	"path/filepath"

	"github.com/gogpu/ui/core/button"
	"github.com/gogpu/ui/core/checkbox"
	"github.com/gogpu/ui/primitives"
	"github.com/gogpu/ui/state"
	"github.com/gogpu/ui/widget"

	"github.com/jitllm/jitllm/ui/app"
	"github.com/jitllm/jitllm/ui/install"
)

// commandLine is the settings card of the macOS app, which carries jitllm and
// jitllmd inside its bundle: put them on PATH, and start jitllmd at login. It
// is nil where there is no bundle to link from (Windows' installer and Linux's
// install script do this themselves).
func commandLine(sh *app.Shell) widget.Widget {
	bin, ok := install.Bundled()
	if !ok {
		return nil
	}
	p := sh.P
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}
	where := state.NewSignal(linkedLine(bin, home))
	agent := install.Agent{Home: home}

	add := button.New(
		button.TextOpt("Add to PATH"),
		button.SizeOpt(button.Small),
		button.VariantOpt(button.Outlined),
		button.OnClick(func() { go linkCLIs(sh, bin, home, where) }),
		button.PainterOpt(p.Button),
	)
	remove := button.New(
		button.TextOpt("Remove from PATH"),
		button.SizeOpt(button.Small),
		button.VariantOpt(button.Outlined),
		button.OnClick(func() {
			for _, d := range install.LinkDirs(home) {
				if err := install.Unlink(bin, d); err != nil {
					// /usr/local/bin is root's: the links there are removed
					// as they were made, by hand or with the admin prompt.
					sh.SetStatus("could not remove the link in " + d + ": " + err.Error())
				}
			}
			where.Set(linkedLine(bin, home))
		}),
		button.PainterOpt(p.Button),
	)
	atLogin := checkbox.New(
		checkbox.LabelOpt("Start the server at login"),
		checkbox.CheckedFn(agent.Installed),
		checkbox.OnToggle(func(on bool) {
			go func() {
				var err error
				if on {
					models := sh.Cfg.DefaultModelDir()
					if dirs := sh.Store.ModelDirs.Get(); len(dirs) > 0 {
						models = dirs[0]
					}
					err = agent.Install(filepath.Join(bin, "jitllmd"), models)
				} else {
					err = agent.Remove()
				}
				switch {
				case err != nil:
					sh.SetStatus("server at login: " + err.Error())
				case on:
					sh.SetStatus("jitllmd starts at login and is running on " + install.AgentAddr)
				default:
					sh.SetStatus("jitllmd no longer starts at login")
				}
			}()
		}),
		checkbox.PainterOpt(p.Checkbox),
	)
	return primitives.VBox(
		line(sh, where.AsReadonly()),
		primitives.HBox(add, remove).Gap(p.Space.S),
		note(sh, "Links jitllm and jitllmd from this app into /usr/local/bin, so a terminal finds them. Your password is asked for if that folder needs it; without it they go to ~/.local/bin."),
		atLogin,
		note(sh, "Runs jitllmd in the background on "+install.AgentAddr+", over your first model folder, from login until you turn this off. It takes the same address as the API above: use one or the other."),
	).Gap(p.Space.S).CrossAlign(primitives.CrossAxisStretch)
}

func linkedLine(bin, home string) string {
	if d := install.Linked(bin, install.LinkDirs(home)); d != "" {
		return "jitllm and jitllmd are on PATH, linked in " + d + "."
	}
	return "jitllm and jitllmd are inside this app and not on PATH yet."
}

// linkCLIs links the bundle's programs into /usr/local/bin, through the
// administrator prompt when the folder is not writable, and into
// ~/.local/bin when the prompt is declined. Off the UI goroutine: the prompt
// blocks until it is answered.
func linkCLIs(sh *app.Shell, bin, home string, where state.Signal[string]) {
	defer func() { where.Set(linkedLine(bin, home)) }()
	sys, local := install.LinkDirs(home)[0], install.LinkDirs(home)[2]
	err := install.Link(bin, sys)
	if err != nil && !errors.Is(err, install.ErrNotALink) {
		err = install.LinkAsAdmin(bin, sys)
	}
	if err == nil {
		sh.SetStatus("linked jitllm and jitllmd in " + sys)
		return
	}
	if lerr := install.Link(bin, local); lerr != nil {
		sh.SetStatus("could not link into " + sys + " (" + err.Error() + ") or " + local + " (" + lerr.Error() + ")")
		return
	}
	sh.SetStatus("linked in " + local + "; add it to your shell's PATH if it is not there")
}

// OfferCLI asks once, on the macOS app's first launch, whether to put jitllm
// and jitllmd on PATH. It does nothing outside a bundle, after the first
// launch, or when they are already linked (a Homebrew cask links them).
func OfferCLI(sh *app.Shell) {
	bin, ok := install.Bundled()
	if !ok || sh.Cfg == nil || sh.Cfg.CLIOffered {
		return
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return
	}
	sh.Cfg.CLIOffered = true
	sh.SaveSoon()
	if install.Linked(bin, install.LinkDirs(home)) != "" {
		return
	}
	where := state.NewSignal("")
	sh.Confirm("Use jitllm from the terminal?",
		"This links jitllm and jitllmd into /usr/local/bin. Your password may be asked for. Settings > Command line can undo it, and can start the server at login.",
		"Add to PATH", func() { go linkCLIs(sh, bin, home, where) })
}
