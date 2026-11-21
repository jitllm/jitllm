// Package crash turns a crash of a front end into a report a person can read
// and file: the panic, the whole stack, the version, the system and the GPU,
// with the home directory shortened to ~ and nothing a prompt or a chat said.
//
// Two kinds of crash reach it. A panic on a goroutine the app owns is
// recovered ([Recover]) and reported while the window is still up. A fatal
// error recover cannot catch -- an unrecovered panic, a concurrent map write,
// running out of memory, an access violation in a driver -- is written by the
// runtime to the file [Arm] names (debug.SetCrashOutput), and [Arm] turns it
// into a report on the next launch. A Windows program linked as a GUI program
// has no standard error to print a traceback to, so that file is the only
// trace there is.
package crash

import (
	"bufio"
	"bytes"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"time"
)

// Report is one crash, ready to show.
type Report struct {
	// Title is one line: what crashed and the panic's first line.
	Title string
	// Text is the whole report: the system, the panic, every stack the runtime
	// wrote and the log lines before it, home paths shortened.
	Text string
	// Path is the file the report was saved to, "" when it could not be.
	Path string
}

// Info is what a report says about the program and the machine. Set its
// fields before anything can crash; GPU may be filled in later, once the
// window has an adapter.
type Info struct {
	App     string
	Version string
	GPU     string
}

// Guard is one program's crash reporting: what its reports say, where they
// are saved, and who is shown them. The zero Guard reports into nothing.
type Guard struct {
	mu       sync.Mutex
	info     Info
	dir      string
	handler  func(Report)
	fatalOut *os.File
}

// SetInfo records what a report says about the program.
func (g *Guard) SetInfo(i Info) {
	g.mu.Lock()
	g.info = i
	g.mu.Unlock()
}

// SetGPU records the GPU a report names.
// The GPU is also written beside crash.log, so the report of a fatal error,
// built by the next launch before it has a window, names the GPU the crashed
// run had.
func (g *Guard) SetGPU(gpu string) {
	g.mu.Lock()
	g.info.GPU = gpu
	d := g.dir
	g.mu.Unlock()
	if d != "" {
		// A report without the GPU is still a report; nothing to do on failure.
		os.WriteFile(filepath.Join(d, gpuFile), []byte(gpu), 0o644)
	}
}

// GPU is the GPU a report of this run names.
func (g *Guard) GPU() string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.info.GPU
}

// OnReport sets the function a recovered panic's report is handed to, on the
// goroutine that panicked. A front end shows it.
func (g *Guard) OnReport(fn func(Report)) {
	g.mu.Lock()
	g.handler = fn
	g.mu.Unlock()
}

// The files in the crash directory.
const (
	// fatalFile is where the runtime writes a fatal error's traceback.
	fatalFile = "crash.log"
	// reportFile is the last report, as it was shown.
	reportFile = "crash-report.txt"
	// gpuFile is the GPU the last run had (SetGPU).
	gpuFile = "gpu.txt"
)

// Arm makes fatal errors reach d: every goroutine's stack is printed
// (GOTRACEBACK=all), and the runtime writes it to d/crash.log. A traceback
// left there by the previous run is returned as a report, saved to
// d/crash-report.txt, and the file emptied so the same crash is shown once.
// logTail is the previous run's last log lines, put in the report.
//
// noConsole says nothing reads the process's standard error (a GUI program
// started from Explorer or Finder). The runtime prints a fatal error's first
// lines ("fatal error: stack overflow", a Windows exception code) only there,
// not to the crash output, so then standard error itself is pointed at
// d/crash.log. With a console it stays where it is, and the file gets the
// traceback alone.
func (g *Guard) Arm(d string, logTail []string, noConsole bool) (*Report, error) {
	debug.SetTraceback("all")
	if err := os.MkdirAll(d, 0o755); err != nil {
		return nil, err
	}
	g.mu.Lock()
	g.dir = d
	g.mu.Unlock()

	fatal := filepath.Join(d, fatalFile)
	var rep *Report
	if b, err := os.ReadFile(fatal); err == nil && crashed(b) {
		prevGPU, err := os.ReadFile(filepath.Join(d, gpuFile))
		if err != nil {
			prevGPU = nil
		}
		r := g.build("the previous run", string(b), logTail, string(prevGPU))
		rep = &r
		rep.Path = g.save(r.Text)
	}
	// Truncated: what is there was just reported, and the next fatal error
	// must be the only thing in it.
	f, err := os.OpenFile(fatal, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return rep, err
	}
	if noConsole {
		if err := redirectStderr(f); err == nil {
			// Kept open for the life of the process: on Windows the standard
			// handle is f's own, not a duplicate.
			g.mu.Lock()
			g.fatalOut = f
			g.mu.Unlock()
			return rep, nil
		}
	}
	// The runtime keeps its own duplicate of the descriptor, so f is closed
	// here without losing the crash output.
	defer f.Close()
	return rep, debug.SetCrashOutput(f, debug.CrashOptions{})
}

// crashed reports whether b, what a run left in crash.log, holds a Go
// traceback. With standard error pointed at the file it also collects what
// drivers print there ("libEGL warning: ..."), which is not a crash.
func crashed(b []byte) bool {
	return bytes.Contains(b, []byte("\ngoroutine ")) || bytes.HasPrefix(b, []byte("goroutine ")) ||
		bytes.Contains(b, []byte("fatal error:")) || bytes.Contains(b, []byte("panic:"))
}

// Recover reports a panic on the calling goroutine and lets the goroutine end
// rather than the process. Defer it first thing in every goroutine the app
// starts: `defer crash.Recover("model scan")`.
func (g *Guard) Recover(where string) {
	r := recover()
	if r == nil {
		return
	}
	g.Handle(where, r, debug.Stack())
}

// Handle reports a recovered panic: it builds the report, saves it and hands
// it to the [OnReport] handler.
func (g *Guard) Handle(where string, r any, stack []byte) Report {
	body := fmt.Sprintf("panic: %v [recovered in %s]\n\n%s", r, where, stack)
	rep := g.build(where, body, nil, g.GPU())
	rep.Path = g.save(rep.Text)
	g.mu.Lock()
	h := g.handler
	g.mu.Unlock()
	if h != nil {
		h(rep)
	}
	return rep
}

// build is the report's text: a header, the traceback, the log tail.
func (g *Guard) build(where, trace string, logTail []string, gpu string) Report {
	g.mu.Lock()
	in := g.info
	g.mu.Unlock()
	if gpu == "" {
		gpu = "unknown"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s %s crashed (%s)\n", in.App, in.Version, where)
	fmt.Fprintf(&b, "os: %s/%s, %s, %d CPUs\n", runtime.GOOS, runtime.GOARCH, runtime.Version(), runtime.NumCPU())
	fmt.Fprintf(&b, "gpu: %s\n", gpu)
	fmt.Fprintf(&b, "reported: %s\n\n", time.Now().UTC().Format(time.RFC3339))
	b.WriteString(strings.TrimRight(trace, "\n"))
	b.WriteString("\n")
	if len(logTail) > 0 {
		b.WriteString("\nlast log lines:\n")
		for _, l := range logTail {
			b.WriteString(l)
			b.WriteString("\n")
		}
	}
	text := Scrub(b.String())
	return Report{Title: Scrub(title(in.App, trace)), Text: text}
}

// title is the issue's title: the app and the first line that says what
// went wrong.
func title(app, trace string) string {
	sc := bufio.NewScanner(strings.NewReader(trace))
	for sc.Scan() {
		l := strings.TrimSpace(sc.Text())
		if strings.HasPrefix(l, "panic:") || strings.HasPrefix(l, "fatal error:") ||
			strings.HasPrefix(l, "Exception ") || strings.HasPrefix(l, "unexpected fault") ||
			strings.HasPrefix(l, "runtime:") || strings.HasPrefix(l, "SIG") {
			if len(l) > 120 {
				l = l[:120]
			}
			return app + " crash: " + l
		}
	}
	return app + " crash"
}

// save writes the report beside the fatal file and returns where.
func (g *Guard) save(text string) string {
	g.mu.Lock()
	d := g.dir
	g.mu.Unlock()
	if d == "" {
		return ""
	}
	p := filepath.Join(d, reportFile)
	if err := os.WriteFile(p, []byte(text), 0o644); err != nil {
		return ""
	}
	return Scrub(p)
}

// Scrub shortens the home directory to ~ and replaces the user's name
// wherever else it appears, so a report can be pasted into a public issue.
// Both slash directions are handled: a Windows path may be written either way.
func Scrub(s string) string {
	home, err := os.UserHomeDir()
	if err != nil || len(home) < 2 {
		return s
	}
	forms := pathForms(home)
	for _, f := range forms {
		for _, h := range []string{f, filepath.ToSlash(f), strings.ReplaceAll(f, "/", `\`)} {
			s = replaceFold(s, h, "~")
		}
	}
	for _, f := range forms {
		if name := filepath.Base(f); len(name) > 2 {
			s = replaceFold(s, name, "<user>")
		}
	}
	return s
}

// replaceFold replaces old in s ignoring case, since Windows paths are
// case-insensitive and a stack may spell the profile differently.
func replaceFold(s, old, repl string) string {
	if old == "" {
		return s
	}
	lo, lold := strings.ToLower(s), strings.ToLower(old)
	if len(lo) != len(s) {
		// A case mapping that changes byte lengths would misalign the indices;
		// fall back to an exact replacement.
		return strings.ReplaceAll(s, old, repl)
	}
	var b strings.Builder
	for {
		i := strings.Index(lo, lold)
		if i < 0 {
			b.WriteString(s)
			return b.String()
		}
		b.WriteString(s[:i])
		b.WriteString(repl)
		s, lo = s[i+len(old):], lo[i+len(old):]
	}
}

// IssueBase is where an issue is opened.
const IssueBase = "https://github.com/jitllm/jitllm/issues/new"

// MaxIssueURL bounds the prefilled issue link. Browsers and GitHub accept
// links of a few thousand bytes; past that the whole link is refused, so a
// report that does not fit is summarised and the person pastes the copy.
const MaxIssueURL = 6000

// IssueURL is a link opening an issue with the report's title and as much of
// its text as fits. When the text does not fit, the body is the report's
// head, its first stack frames, and a request to paste the copied report.
func IssueURL(r Report) string {
	link := func(body string) string {
		return IssueBase + "?" + url.Values{"title": {r.Title}, "body": {body}}.Encode()
	}
	full := "```\n" + r.Text + "```\n"
	if u := link(full); len(u) <= MaxIssueURL {
		return u
	}
	const note = "\nThe report was too long for a link. It has been copied to the clipboard; please paste it here in full.\n"
	lines := strings.Split(r.Text, "\n")
	for n := len(lines); n > 0; n /= 2 {
		u := link("```\n" + strings.Join(lines[:n], "\n") + "\n...\n```\n" + note)
		if len(u) <= MaxIssueURL {
			return u
		}
	}
	return link(note)
}

// LogTail is the last n lines of the file at p that come before any
// traceback in it, or nil when it cannot be read.
func LogTail(p string, n int) []string {
	b, err := os.ReadFile(p)
	if err != nil {
		return nil
	}
	var lines []string
	sc := bufio.NewScanner(bytes.NewReader(b))
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		l := sc.Text()
		if strings.HasPrefix(l, "panic:") || strings.HasPrefix(l, "fatal error:") || strings.HasPrefix(l, "goroutine ") {
			break
		}
		lines = append(lines, l)
	}
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return lines
}

// OpenURL opens u in the default browser.
func OpenURL(u string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", u)
	case "windows":
		// Not `cmd /c start`: cmd reads the & between query fields as a
		// command separator.
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", u)
	default:
		cmd = exec.Command("xdg-open", u)
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	// Reaped in the background: the browser outlives this call.
	go cmd.Wait()
	return nil
}
