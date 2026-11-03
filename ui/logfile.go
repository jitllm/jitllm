package main

import (
	"log"
	"os"
	"path/filepath"

	"github.com/jitllm/jitllm/common/config"
)

// logToFileWithoutAConsole sends the app's log output to jitllm-ui.log beside
// the settings file when nothing would read standard error: a Windows GUI
// program started from Explorer has no console (its handle does not stat), and
// a macOS app opened from Finder or `open` has its standard error on
// /dev/null. Started from a terminal, the output stays there. noConsole is
// !stderrIsRead(), asked once: on Windows asking attaches the console.
func logToFileWithoutAConsole(held *stdio, noConsole bool) {
	if !noConsole {
		return
	}
	p := config.ConfigPath()
	if p == "" {
		return
	}
	dir := filepath.Dir(p)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	// One run's log, truncated at start, so it never grows without bound.
	f, err := os.OpenFile(filepath.Join(dir, logName), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return
	}
	held.replace(f)
	log.SetOutput(f)
}

// logName is the log file beside the settings. A crash's traceback is not in
// it: the runtime writes that to crash.log (package crash), which is shown on
// the next launch, since this file is truncated by the launch that follows.
const logName = "jitllm-ui.log"

// stderrIsRead reports whether standard error goes anywhere a person sees.
func stderrIsRead(held *stdio) bool {
	fi, err := os.Stderr.Stat()
	if err != nil {
		// A GUI program run from a terminal has no console of its own; it
		// writes to the terminal's when it can attach to it.
		return attachParentConsole(held)
	}
	null, err := os.Stat(os.DevNull)
	if err != nil {
		return true
	}
	return !os.SameFile(fi, null)
}

// stdio holds the standard files os.Stdout and os.Stderr held before they were
// replaced. An *os.File closes its descriptor when collected, and descriptor
// 2 is where the runtime writes a fatal error: dropped, the collector closes
// it, the next file opened takes its number, and a crash's traceback is
// written into that file -- or nowhere. The caller keeps it alive for the
// life of the process.
type stdio struct{ files []*os.File }

// replace points os.Stdout and os.Stderr at f, keeping the old ones.
func (s *stdio) replace(f *os.File) {
	s.files = append(s.files, os.Stdout, os.Stderr)
	os.Stdout, os.Stderr = f, f
}
