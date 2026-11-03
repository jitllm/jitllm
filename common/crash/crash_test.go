package crash

import (
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/debug"
	"strings"
	"testing"
	"time"
)

// The child half of the fatal tests: it arms the directory and dies the way
// the env var says, in a way recover cannot catch.
func TestMain(m *testing.M) {
	if d := os.Getenv("CRASH_CHILD_DIR"); d != "" {
		g := &Guard{}
		if _, err := g.Arm(d, nil, os.Getenv("CRASH_CHILD_KIND") == "overflow"); err != nil {
			panic(err)
		}
		home, err := os.UserHomeDir()
		if err != nil {
			panic(err)
		}
		g.SetGPU("Test Adapter (Vulkan, DiscreteGPU)")
		switch os.Getenv("CRASH_CHILD_KIND") {
		case "panic":
			go func() {
				panic("boom in a worker reading " + filepath.Join(home, "models"))
			}()
			// Waits for the panic to end the process; nothing else does.
			select {}
		case "overflow":
			debug.SetMaxStack(1 << 20)
			var f func(int) int
			f = func(n int) int { return f(n+1) + 1 }
			f(0)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// crashDir is a directory for Arm. The cleanup lets go of the crash output
// before the directory is removed: Windows will not delete an open file.
func crashDir(t *testing.T) string {
	d := t.TempDir()
	t.Cleanup(func() {
		if err := debug.SetCrashOutput(nil, debug.CrashOptions{}); err != nil {
			t.Error(err)
		}
	})
	return d
}

func runChild(t *testing.T, dir, kind string) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^$")
	cmd.Env = append(os.Environ(), "CRASH_CHILD_DIR="+dir, "CRASH_CHILD_KIND="+kind)
	if err := cmd.Run(); err == nil {
		t.Fatalf("the %s child exited cleanly; it was to crash", kind)
	}
}

// A fatal error recover cannot catch reaches the next launch as a report with
// the whole traceback, home shortened, shown once. The panic child has a
// console (the crash output alone); the overflow child has none, so its
// standard error is the file and the "fatal error" line, which the runtime
// prints only there, is in the report.
func TestFatalCrashIsReportedOnTheNextLaunch(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil || len(home) < 2 {
		t.Fatal("no home directory to scrub")
	}
	for _, c := range []struct{ kind, want string }{
		{"panic", "panic: boom in a worker reading ~"},
		{"overflow", "fatal error: stack overflow"},
	} {
		t.Run(c.kind, func(t *testing.T) {
			dir := crashDir(t)
			runChild(t, dir, c.kind)
			g := &Guard{}
			g.SetInfo(Info{App: "jitllm-desktop", Version: "test"})
			rep, err := g.Arm(dir, []string{"loaded " + filepath.Join(home, "m.jlm")}, false)
			if err != nil {
				t.Fatal(err)
			}
			if rep == nil {
				t.Fatal("no report from a run that crashed")
			}
			for _, w := range []string{c.want, "goroutine ", "crash_test.go", "os: ", "last log lines:", "loaded ~", "gpu: Test Adapter (Vulkan, DiscreteGPU)"} {
				if !strings.Contains(rep.Text, w) {
					t.Errorf("report lacks %q:\n%s", w, rep.Text)
				}
			}
			if strings.Contains(strings.ToLower(rep.Text), strings.ToLower(home)) {
				t.Errorf("report carries the home directory %q", home)
			}
			if !strings.HasPrefix(rep.Title, "jitllm-desktop crash: ") {
				t.Errorf("title %q", rep.Title)
			}
			saved, err := os.ReadFile(filepath.Join(dir, reportFile))
			if err != nil || string(saved) != rep.Text {
				t.Errorf("saved report differs from the shown one (%v)", err)
			}
			again, err := g.Arm(dir, nil, false)
			if err != nil || again != nil {
				t.Errorf("the same crash reported twice (%v)", err)
			}
		})
	}
}

// What a driver prints to standard error lands in crash.log too, and is not a
// crash: a launch after it shows nothing. Against the violation (any bytes
// counting as a crash) a report comes back.
func TestDriverNoiseIsNotACrash(t *testing.T) {
	dir := crashDir(t)
	noise := "libEGL warning: DRI3 error: Could not get DRI3 device\nvulkan: No DRI3 support detected\n"
	if err := os.WriteFile(filepath.Join(dir, fatalFile), []byte(noise), 0o644); err != nil {
		t.Fatal(err)
	}
	rep, err := (&Guard{}).Arm(dir, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if rep != nil {
		t.Fatalf("driver noise reported as a crash:\n%s", rep.Text)
	}
}

// A recovered panic is handed to the handler with the panicking goroutine's
// stack, and the goroutine ends rather than the process.
func TestRecoverReportsAndKeepsTheProcess(t *testing.T) {
	dir := crashDir(t)
	g := &Guard{}
	if _, err := g.Arm(dir, nil, false); err != nil {
		t.Fatal(err)
	}
	got := make(chan Report, 1)
	g.OnReport(func(r Report) { got <- r })
	defer g.OnReport(nil)
	go func() {
		defer g.Recover("test worker")
		var m map[string]int
		m["x"] = 1
	}()
	var r Report
	select {
	case r = <-got:
	case <-time.After(10 * time.Second):
		t.Fatal("the recovered panic was never reported")
	}
	for _, w := range []string{"assignment to entry in nil map", "recovered in test worker", "crash_test.go", "goroutine "} {
		if !strings.Contains(r.Text, w) {
			t.Errorf("report lacks %q:\n%s", w, r.Text)
		}
	}
	if r.Path == "" {
		t.Error("report not saved")
	}
}

// The issue link opens the right page with a title, and stays under the bound
// however long the report.
func TestIssueURLIsWellFormedAndBounded(t *testing.T) {
	for _, n := range []int{10, 100000} {
		r := Report{Title: "jitllm-desktop crash: panic: x & y", Text: strings.Repeat("goroutine 1 [running]:\n", n/20+1)}
		u := IssueURL(r)
		if len(u) > MaxIssueURL {
			t.Fatalf("link of %d bytes, bound %d", len(u), MaxIssueURL)
		}
		p, err := url.Parse(u)
		if err != nil {
			t.Fatal(err)
		}
		if p.Scheme+"://"+p.Host+p.Path != IssueBase {
			t.Errorf("link goes to %s", u)
		}
		q := p.Query()
		if q.Get("title") != r.Title || !strings.Contains(q.Get("body"), "goroutine 1") {
			t.Errorf("title %q body %q", q.Get("title"), q.Get("body"))
		}
		long := len(r.Text) > MaxIssueURL
		if long != strings.Contains(q.Get("body"), "paste it here") {
			t.Errorf("a %d-byte report: summary note present = %v", len(r.Text), !long)
		}
	}
}

func TestScrubShortensHomeInBothSlashes(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil || len(home) < 2 {
		t.Fatal("no home directory")
	}
	in := home + "/a " + strings.ReplaceAll(home, "/", `\`) + `\b ` + strings.ToUpper(home)
	out := Scrub(in)
	if strings.Contains(strings.ToLower(out), strings.ToLower(filepath.Base(home))) {
		t.Errorf("Scrub(%q) = %q", in, out)
	}
}
