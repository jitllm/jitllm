// Package service installs jitllmd as the person's own background service,
// the way each system starts a per-user program at login: a systemd user
// unit on Linux, a launchd user agent on macOS, and on Windows the login
// script and Run key the installer (packaging/windows) writes. `jitllmd
// service` and the macOS app's "start the server at login" both go through
// it, so there is one definition of what the service runs.
//
// It reads no environment and owns no process: every path is derived from
// the directories in Service, and every system command goes through Run, so
// a test installs into a temporary home and records the commands.
package service

import (
	"errors"
	"fmt"
	"html"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/internal/localserver"
)

// Label names the launchd agent: the macOS app's bundle identifier with the
// program's name. It is also the Windows Run value's name's stem.
const Label = "org.jitllm.service"

// Unit is the systemd user unit's default name, without .service.
const Unit = "jitllm"

// Service is the service for one person, on one system.
type Service struct {
	// GOOS is the system the files are for ("linux", "darwin", "windows").
	GOOS string
	// Home is the person's home directory.
	Home string
	// ConfigHome is $XDG_CONFIG_HOME, where systemd looks for user units
	// (default Home/.config).
	ConfigHome string
	// LocalAppData is %LOCALAPPDATA% on Windows (default Home\AppData\Local).
	LocalAppData string
	// Name replaces the unit's name (Unit) and the agent's label (Label); a
	// test uses a name of its own so it never touches the person's service.
	Name string
	// UID is the person's user id, for launchd's gui/<uid> domain.
	UID int
	// Run runs a system command and returns its combined output. Nil runs
	// it for real.
	Run func(name string, args ...string) ([]byte, error)
}

// ForThisUser is the service for the person running this process.
// configHome is $XDG_CONFIG_HOME and localAppData %LOCALAPPDATA%, which the
// command reads and hands over.
func ForThisUser(goos, configHome, localAppData string) (Service, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return Service{}, err
	}
	return Service{GOOS: goos, Home: home, ConfigHome: configHome, LocalAppData: localAppData, UID: os.Getuid()}, nil
}

func (s Service) name() string {
	if s.Name != "" {
		return s.Name
	}
	if s.GOOS == "darwin" {
		return Label
	}
	return Unit
}

func (s Service) run(name string, args ...string) ([]byte, error) {
	if s.Run != nil {
		return s.Run(name, args...)
	}
	out, err := exec.Command(name, args...).CombinedOutput()
	if err != nil {
		err = fmt.Errorf("%s %s: %v: %s", name, strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return out, err
}

// File is the file that defines the service: the unit, the property list,
// or the Windows login script.
func (s Service) File() string {
	switch s.GOOS {
	case "darwin":
		// A launchd path, so slash-separated wherever this is built.
		return path.Join(s.Home, "Library", "LaunchAgents", s.name()+".plist")
	case "windows":
		return winJoin(s.windowsDir(), "jitllmd-login.vbs")
	}
	cfg := s.ConfigHome
	if cfg == "" || !path.IsAbs(cfg) {
		cfg = path.Join(s.Home, ".config")
	}
	return path.Join(cfg, "systemd", "user", s.name()+".service")
}

// Log is where the service's output goes. On Linux it is the journal, and
// this is the command that reads it.
func (s Service) Log() string {
	switch s.GOOS {
	case "darwin":
		return path.Join(s.Home, "Library", "Logs", "jitllm", "jitllmd.log")
	case "windows":
		return winJoin(s.windowsDir(), "jitllmd.log")
	}
	return "journalctl --user -u " + s.name()
}

// winJoin joins Windows path elements with backslashes, on any host: the
// service's paths are in its target system's spelling, not this host's.
func winJoin(elem ...string) string {
	out := strings.TrimRight(elem[0], `\/`)
	for _, e := range elem[1:] {
		out += `\` + strings.Trim(e, `\/`)
	}
	return out
}

// windowsDir is %LOCALAPPDATA%\jitllm, beside the models folder.
func (s Service) windowsDir() string {
	l := s.LocalAppData
	if l == "" {
		l = winJoin(s.Home, "AppData", "Local")
	}
	return winJoin(l, "jitllm")
}

// Installed reports whether the service's file is in place.
func (s Service) Installed() bool {
	_, err := os.Stat(s.File())
	return err == nil
}

// Args is what the service runs: jitllmd serve on addr (DefaultAddr when
// empty) over the models folder, then any further serve flags (such as
// "-api-key", KEY).
func Args(jitllmd, models, addr string, serveFlags ...string) []string {
	if addr == "" {
		addr = localserver.DefaultAddr
	}
	return append([]string{jitllmd, "serve", "-addr", addr, "-models", models}, serveFlags...)
}

// Render is the service's file for these arguments.
func (s Service) Render(args []string) string {
	switch s.GOOS {
	case "darwin":
		return s.plist(args)
	case "windows":
		return s.vbs(args)
	}
	return s.unit(args)
}

// unit is a systemd user unit: restarted after a crash, not after a clean
// stop, started with the person's default target (at login, or at boot with
// lingering on). Its output goes to the journal.
func (s Service) unit(args []string) string {
	q := make([]string, len(args))
	for i, a := range args {
		q[i] = systemdQuote(a)
	}
	return `[Unit]
Description=jitllm inference server (jitllmd)
After=network.target

[Service]
ExecStart=` + strings.Join(q, " ") + `
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
`
}

// systemdQuote quotes one ExecStart word: double quotes, with backslash and
// quote escaped, and % doubled since systemd expands specifiers in it.
func systemdQuote(a string) string {
	a = strings.NewReplacer(`\`, `\\`, `"`, `\"`, `%`, `%%`).Replace(a)
	return `"` + a + `"`
}

// plist is the launchd agent: at login and after a crash (KeepAlive on an
// unsuccessful exit only, so jitllmd exiting cleanly is not restarted).
func (s Service) plist(args []string) string {
	e := html.EscapeString
	var argv strings.Builder
	for _, a := range args {
		argv.WriteString("\t\t<string>" + e(a) + "</string>\n")
	}
	return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>` + e(s.name()) + `</string>
	<key>ProgramArguments</key>
	<array>
` + argv.String() + `	</array>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<dict>
		<key>SuccessfulExit</key>
		<false/>
	</dict>
	<key>ProcessType</key>
	<string>Background</string>
	<key>StandardOutPath</key>
	<string>` + e(s.Log()) + `</string>
	<key>StandardErrorPath</key>
	<string>` + e(s.Log()) + `</string>
</dict>
</plist>
`
}

// vbs is the installer's login script: jitllmd started hidden (window style
// 0, so a console program opens no window) without waiting, its output
// appended to the log through cmd.
func (s Service) vbs(args []string) string {
	q := make([]string, len(args))
	for i, a := range args {
		q[i] = `"` + a + `"`
	}
	// cmd /c strips one outer pair of quotes, so the whole line is wrapped
	// in one more; a VBScript string literal then doubles every quote.
	line := `cmd /c "` + strings.Join(q, " ") + ` >> "` + s.Log() + `" 2>&1"`
	line = strings.ReplaceAll(line, `"`, `""`)
	return `CreateObject("WScript.Shell").Run "` + line + `", 0, False` + "\r\n"
}

// runKey is the Windows Run key the installer writes the login entry under.
const runKey = `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`

func (s Service) runValue() string {
	if s.Name != "" {
		return s.Name
	}
	return "jitllmd"
}

func (s Service) domain() string { return "gui/" + strconv.Itoa(s.UID) }

// Install writes the service and starts it now and at every login. One
// already there is replaced, so a moved binary or a new models folder takes
// effect.
func (s Service) Install(args []string) error {
	if err := os.MkdirAll(filepath.Dir(s.File()), 0o755); err != nil {
		return err
	}
	if s.GOOS != "linux" && s.GOOS != "" {
		if err := os.MkdirAll(filepath.Dir(s.Log()), 0o755); err != nil {
			return err
		}
	}
	if s.GOOS == "darwin" && s.Installed() {
		// Not loaded is not a failure: the file can outlive a session.
		s.run("launchctl", "bootout", s.domain()+"/"+s.name())
	}
	// 0600: the arguments may carry the service's API key.
	if err := os.WriteFile(s.File(), []byte(s.Render(args)), 0o600); err != nil {
		return err
	}
	switch s.GOOS {
	case "darwin":
		_, err := s.run("launchctl", "bootstrap", s.domain(), s.File())
		return err
	case "windows":
		if _, err := s.run("reg", "add", runKey, "/v", s.runValue(), "/t", "REG_SZ", "/d",
			`wscript.exe "`+s.File()+`"`, "/f"); err != nil {
			return err
		}
		return s.Start()
	}
	if _, err := s.run("systemctl", "--user", "daemon-reload"); err != nil {
		return err
	}
	// restart, not start: a reinstall must pick up the new file.
	if _, err := s.run("systemctl", "--user", "enable", s.name()+".service"); err != nil {
		return err
	}
	_, err := s.run("systemctl", "--user", "restart", s.name()+".service")
	return err
}

// Uninstall stops the service and removes it. One that is not installed is
// not an error.
func (s Service) Uninstall() error {
	if !s.Installed() {
		return nil
	}
	switch s.GOOS {
	case "darwin":
		// As in Install: an agent that is not loaded has nothing to stop.
		s.run("launchctl", "bootout", s.domain()+"/"+s.name())
	case "windows":
		// A value already gone is the state wanted.
		s.run("reg", "delete", runKey, "/v", s.runValue(), "/f")
		s.Stop()
	default:
		// A unit already stopped or disabled is the state wanted.
		s.run("systemctl", "--user", "disable", "--now", s.name()+".service")
	}
	if err := os.Remove(s.File()); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if s.GOOS == "linux" || s.GOOS == "" {
		s.run("systemctl", "--user", "daemon-reload")
	}
	return nil
}

// Start starts an installed service now.
func (s Service) Start() error {
	if !s.Installed() {
		return ErrNotInstalled
	}
	var err error
	switch s.GOOS {
	case "darwin":
		_, err = s.run("launchctl", "kickstart", s.domain()+"/"+s.name())
	case "windows":
		_, err = s.run("wscript.exe", s.File())
	default:
		_, err = s.run("systemctl", "--user", "start", s.name()+".service")
	}
	return err
}

// Stop stops a running service until the next login or Start. A clean stop
// is not restarted on any system.
func (s Service) Stop() error {
	if !s.Installed() {
		return ErrNotInstalled
	}
	var err error
	switch s.GOOS {
	case "darwin":
		_, err = s.run("launchctl", "kill", "SIGTERM", s.domain()+"/"+s.name())
	case "windows":
		// /F: a console program started hidden has no window to ask. The
		// filter spares this process, which may itself be a jitllmd (`jitllmd
		// service stop`).
		_, err = s.run("taskkill", "/F", "/IM", "jitllmd.exe", "/FI", "PID ne "+strconv.Itoa(os.Getpid()))
	default:
		_, err = s.run("systemctl", "--user", "stop", s.name()+".service")
	}
	return err
}

// Manager names what runs the service on this system, for status.
func (s Service) Manager() string {
	switch s.GOOS {
	case "darwin":
		return "launchd agent " + s.name()
	case "windows":
		return `login entry ` + runKey + `\` + s.runValue()
	}
	return "systemd user unit " + s.name() + ".service"
}

// ErrNotInstalled is Start or Stop on a service with no file.
const ErrNotInstalled sentinel = "the jitllm service is not installed: run `jitllmd service install`"

type sentinel string

func (e sentinel) Error() string { return string(e) }
