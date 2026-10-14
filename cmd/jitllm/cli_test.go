package main

import (
	"bytes"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
)

// The command line: which subcommand runs, what a bad flag does, and the
// lines other tools read off a run's output. The functions are called
// directly where they return; only what EXITS -- usage() and a flag set built
// with ExitOnError -- needs a process of its own.

// cliArgsEnv carries a child's argv, separated by \x1f.
const cliArgsEnv = "JITLLM_CLI_ARGS"

// TestMain runs the command's own main() when this binary is re-executed by
// cli(). A child never reaches m.Run.
func TestMain(m *testing.M) {
	if args, ok := os.LookupEnv(cliArgsEnv); ok {
		os.Args = []string{"jitllm"}
		if args != "" {
			os.Args = append(os.Args, strings.Split(args, "\x1f")...)
		}
		main()
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// cli runs `jitllm args...` as a child and returns its exit status and output.
func cli(t *testing.T, args ...string) (code int, stdout, stderr string) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^$")
	cmd.Env = append(os.Environ(), cliArgsEnv+"="+strings.Join(args, "\x1f"))
	var o, e bytes.Buffer
	cmd.Stdout, cmd.Stderr = &o, &e
	err := cmd.Run()
	var ee *exec.ExitError
	switch {
	case errors.As(err, &ee):
		code = ee.ExitCode()
	case err != nil:
		t.Fatalf("running the child: %v", err)
	}
	return code, o.String(), e.String()
}

// TestDispatchAndFlagErrorsExitWithTheirStatus: a usage mistake is status 2
// with the usage text, a run that fails is status 1 with its reason, and a
// flag value that does not parse names the flag rather than running with the
// default.
func TestDispatchAndFlagErrorsExitWithTheirStatus(t *testing.T) {
	for _, tc := range []struct {
		args []string
		code int
		says string
	}{
		{nil, 2, "usage:"},
		{[]string{"frobnicate", "x"}, 2, "usage:"},
		{[]string{"run"}, 2, "usage:"},
		{[]string{"info"}, 2, "usage:"},
		{[]string{"run", "m.jlm"}, 2, "usage:"},
		{[]string{"tokenize", "m.jlm"}, 2, "usage:"},
		// -maxmem takes a byte count with a unit (3G is accepted) and refuses
		// what is not one by name rather than reading it as zero, which would
		// mean "work it out".
		{[]string{"run", "-maxmem", "lots", "m.jlm", "hi"}, 2, `invalid value "lots" for flag -maxmem`},
		{[]string{"run", "-n", "many", "m.jlm", "hi"}, 2, `invalid value "many" for flag -n`},
		{[]string{"run", "-temp", "hot", "m.jlm", "hi"}, 2, `invalid value "hot" for flag -temp`},
		{[]string{"run", "-bogus", "m.jlm", "hi"}, 2, "flag provided but not defined: -bogus"},
		{[]string{"run", "m.jlm", "-chat", "hi"}, 1, "-chat came after the file"},
		{[]string{"run", filepath.Join(t.TempDir(), "absent.jlm"), "hi"}, 1, "absent.jlm"},
		{[]string{"run", "model.gguf", "hi"}, 1, "jitllm convert model.gguf"},
		{[]string{"bench", "a", "b"}, 1, "usage: jitllm bench"},
		{[]string{"embed", "m.jlm"}, 1, "usage: jitllm embed"},
		{[]string{"library", "-bogus"}, 1, "flag provided but not defined"},
	} {
		code, _, stderr := cli(t, tc.args...)
		if code != tc.code || !strings.Contains(stderr, tc.says) {
			t.Errorf("jitllm %v: exit %d, want %d saying %q; stderr:\n%s", tc.args, code, tc.code, tc.says,
				firstLines(stderr, 4))
		}
	}
	code, stdout, _ := cli(t, "version")
	if code != 0 || stdout != "dev\n" {
		t.Errorf("jitllm version: exit %d, stdout %q", code, stdout)
	}
}

func firstLines(s string, n int) string {
	ls := strings.SplitN(s, "\n", n+1)
	if len(ls) > n {
		ls = ls[:n]
	}
	return strings.Join(ls, "\n")
}

// TestRunRefusesContradictoryFlagsBeforeLoading: each of these is caught by
// runCmd before a model is opened, so the file need not exist -- and the
// error must not be "file not found", which would send the user the wrong
// way.
func TestRunRefusesContradictoryFlagsBeforeLoading(t *testing.T) {
	absent := filepath.Join(t.TempDir(), "absent.jlm")
	for _, tc := range []struct {
		args []string
		says string
	}{
		{[]string{absent, "-chat", "hi"}, "came after the file"},
		{[]string{"-system", "be terse", absent, "hi"}, "-system needs -chat"},
		{[]string{"-placement-strict", absent, "hi"}, "-placement-strict needs -placement"},
		{[]string{"-placement", "0:host", absent, "hi"}, "is not SEL=WHERE"},
		{[]string{"-relocate", "-devices", "cuda:0", absent, "hi"}, "-relocate moves blocks to the host"},
		{[]string{"-tokenizer", filepath.Join(t.TempDir(), "no-tokenizer.json"), absent, "hi"}, "no-tokenizer.json"},
	} {
		err := runCmd(tc.args)
		if err == nil || !strings.Contains(err.Error(), tc.says) || strings.Contains(err.Error(), "absent.jlm") {
			t.Errorf("run %v: %v, want a refusal saying %q", tc.args, err, tc.says)
		}
	}
	if err := tokenizeCmd([]string{absent, "-tokenizer", "x"}); err == nil || !strings.Contains(err.Error(), "came after") {
		t.Errorf("tokenize with a flag after the file: %v", err)
	}
	if err := benchCmd([]string{absent, "-tokenizer", "x"}); err == nil || !strings.Contains(err.Error(), "came after") {
		t.Errorf("bench with a flag after the file: %v", err)
	}
	if err := embedCmd([]string{absent, "-ids", "x"}); err == nil || !strings.Contains(err.Error(), "came after") {
		t.Errorf("embed with a flag after the file: %v", err)
	}
}

// ---------------------------------------------------------------- output lines

// modelPath is name in the model directory, or a loud skip.
func modelPath(t *testing.T, name string) string {
	t.Helper()
	p := testmodels.Path(name)
	fresh.Must(t, p)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	return p
}

// capture runs f with os.Stdout and os.Stderr redirected, as the command
// writes to both directly.
func capture(t *testing.T, f func() error) (stdout, stderr string, err error) {
	t.Helper()
	ro, wo, perr := os.Pipe()
	if perr != nil {
		t.Fatal(perr)
	}
	re, we, perr := os.Pipe()
	if perr != nil {
		t.Fatal(perr)
	}
	outc, errc := make(chan string, 1), make(chan string, 1)
	go func() { b, _ := io.ReadAll(ro); outc <- string(b) }()
	go func() { b, _ := io.ReadAll(re); errc <- string(b) }()
	oldOut, oldErr := os.Stdout, os.Stderr
	os.Stdout, os.Stderr = wo, we
	defer func() {
		os.Stdout, os.Stderr = oldOut, oldErr
	}()
	err = f()
	wo.Close()
	we.Close()
	return <-outc, <-errc, err
}

// The patterns the scripts in scripts/ apply to a run's stderr, transcribed
// from their sed expressions. A change to a line that breaks one of these
// breaks a measurement harness silently: its rate comes back empty.
var (
	// vs-llamacpp.sh, paging.sh: the decode rate.
	reDecodeRate = regexp.MustCompile(`decode [0-9]* tok in [^(]*\(([0-9.]*) tok/s\)`)
	// vs-llamacpp.sh: the prompt rate.
	rePromptRate = regexp.MustCompile(`prompt [0-9]* tok in [^(]*\(([0-9.]*) tok/s\)`)
	// paging-rate.sh: the decode count and its duration.
	reDecodeDur = regexp.MustCompile(`decode ([0-9]*) tok in ([^ ]*) `)
	// paging.sh: the ids, prompt and completion together.
	reIDs = regexp.MustCompile(`(?m)^ids (\[.*\])$`)

	reMemory = regexp.MustCompile(`(?m)^memory   budget ([0-9.]+) GiB   weights ([0-9.]+) GiB   kv ([0-9.]+) GiB reserved ` +
		`\(-n ([0-9]+), commits as the context grows\)   touches ([0-9.]+) GiB of weights a token(   ★ OVER BUDGET by ([0-9.]+) GiB)?$`)
	rePages = regexp.MustCompile(`(?m)^pages    ([0-9]+) frame\(s\), ([0-9]+) page-in\(s\), ([0-9]+) eviction\(s\), ` +
		`([0-9]+) fresh frame\(s\), ([0-9.]+) GiB read in ([0-9]+) request\(s\) at ([0-9]+) KiB chunks$`)
	rePagesShort = regexp.MustCompile(`(?m)^pages    ([0-9]+) of ([0-9]+) host block\(s\) resident at once`)
	rePromptLine = regexp.MustCompile(`(?m)^prompt ([0-9]+) tok in `)
)

func atoi(t *testing.T, s string) int {
	t.Helper()
	n, err := strconv.Atoi(s)
	if err != nil {
		t.Fatalf("%q is not a count: %v", s, err)
	}
	return n
}

// runSmall runs stories260K on the host, greedy, and returns what it printed.
func runSmall(t *testing.T, n int, maxmem uint64) (stdout, stderr string) {
	t.Helper()
	path := modelPath(t, "stories260K.jlm")
	sm := &model.Sampler{RepeatPen: 1, RepeatLastN: 64}
	stdout, stderr, err := capture(t, func() error {
		return run(path, "Once upon a time", n, 0, "cpu", -1, 0, maxmem, false, false, false, false,
			sm, "", nil, nil, "", 0, 0, nil, nil)
	})
	if err != nil {
		t.Fatalf("run: %v\n%s", err, stderr)
	}
	return stdout, stderr
}

// TestRunPrintsTheLinesTheScriptsParse: every line a harness reads is present,
// matches the pattern that harness applies, and its numbers agree with each
// other -- the ids line holds exactly the prompt and the decoded tokens.
func TestRunPrintsTheLinesTheScriptsParse(t *testing.T) {
	stdout, stderr := runSmall(t, 6, 1<<30)

	if !strings.HasPrefix(stdout, "Once upon a time") || len(stdout) <= len("Once upon a time\n") {
		t.Fatalf("stdout is not the prompt then its completion: %q", stdout)
	}
	for name, re := range map[string]*regexp.Regexp{"decode rate": reDecodeRate, "prompt rate": rePromptRate} {
		m := re.FindStringSubmatch(stderr)
		if m == nil && strings.Contains(stderr, "0s (rate under the clock's resolution)") {
			// A run inside one tick of a coarse clock (Windows) has no rate to
			// print; the line says so rather than +Inf.
			continue
		}
		if m == nil {
			t.Fatalf("no %s line a harness can read:\n%s", name, stderr)
		}
		if _, err := strconv.ParseFloat(m[1], 64); err != nil {
			t.Fatalf("the %s %q is not a number", name, m[1])
		}
	}
	d := reDecodeDur.FindStringSubmatch(stderr)
	if d == nil {
		t.Fatalf("no decode count and duration:\n%s", stderr)
	}
	if _, err := time.ParseDuration(d[2]); err != nil {
		t.Fatalf("the decode duration %q does not parse: %v", d[2], err)
	}
	decoded := atoi(t, d[1])
	if decoded < 1 || decoded > 6 {
		t.Fatalf("decoded %d tokens with -n 6", decoded)
	}
	prompt := atoi(t, rePromptLine.FindStringSubmatch(stderr)[1])

	idm := reIDs.FindStringSubmatch(stderr)
	if idm == nil {
		t.Fatalf("no ids line:\n%s", stderr)
	}
	if got := len(strings.Fields(strings.Trim(idm[1], "[]"))); got != prompt+decoded {
		t.Fatalf("the ids line holds %d ids; the prompt was %d and the decode %d", got, prompt, decoded)
	}

	mem := reMemory.FindStringSubmatch(stderr)
	if mem == nil {
		t.Fatalf("no memory line in the shape it is read in:\n%s", stderr)
	}
	if mem[1] != "1.00" || mem[4] != "6" || mem[6] != "" {
		t.Fatalf("memory line says budget %s GiB, -n %s, over-budget %q; asked 1 GiB, -n 6, and it fits", mem[1], mem[4], mem[6])
	}
	if !strings.Contains(stderr, "\ndevice cpu\n") {
		t.Fatalf("a host run does not say so:\n%s", stderr)
	}
	pg := rePages.FindStringSubmatch(stderr)
	if pg == nil {
		t.Fatalf("no pages line:\n%s", stderr)
	}
	if atoi(t, pg[1]) != 5 || atoi(t, pg[3]) != 0 {
		t.Fatalf("a model that fits reads %s frame(s) and %s eviction(s), want 5 and 0", pg[1], pg[3])
	}
	if rePagesShort.MatchString(stderr) {
		t.Fatalf("a model that fits printed the short-budget line:\n%s", stderr)
	}
}

// TestATinyBudgetPagesAndSaysSo: a -maxmem below what the dense weights and
// the KV already take leaves the pager nothing -- and the pager reads a zero
// budget as UNLIMITED. Before setPageBudget clamped it, a run over budget
// printed the warning and then held every block anyway, evicting nothing.
func TestATinyBudgetPagesAndSaysSo(t *testing.T) {
	_, stderr := runSmall(t, 4, 4096)
	mem := reMemory.FindStringSubmatch(stderr)
	if mem == nil || mem[6] == "" {
		t.Fatalf("a 4 KiB budget against a 1.2 MB model is not marked over budget:\n%s", stderr)
	}
	short := rePagesShort.FindStringSubmatch(stderr)
	if short == nil {
		t.Fatalf("no line saying how few blocks the budget holds:\n%s", stderr)
	}
	if got, host := atoi(t, short[1]), atoi(t, short[2]); got >= host || host != 5 {
		t.Fatalf("the short-budget line says %d of %d resident", got, host)
	}
	pg := rePages.FindStringSubmatch(stderr)
	if pg == nil || atoi(t, pg[3]) == 0 {
		t.Fatalf("a run that cannot hold its blocks evicted nothing:\n%s", stderr)
	}
}

// TestTokenizePrintsEveryPiece: the count, the ids, and one piece per id.
func TestTokenizePrintsEveryPiece(t *testing.T) {
	stdout, _, err := capture(t, func() error {
		return tokenizeCmd([]string{modelPath(t, "stories260K.jlm"), "Once", "upon", "a", "time"})
	})
	if err != nil {
		t.Fatal(err)
	}
	m := regexp.MustCompile(`^([0-9]+) tokens: \[([0-9 ]+)\]\n`).FindStringSubmatch(stdout)
	if m == nil {
		t.Fatalf("tokenize printed %q", stdout)
	}
	n := atoi(t, m[1])
	if len(strings.Fields(m[2])) != n || strings.Count(stdout, " -> ") != n {
		t.Fatalf("tokenize says %d tokens and lists %q with %d pieces", n, m[2], strings.Count(stdout, " -> "))
	}
	if strings.Contains(stdout, "round trip differs") {
		t.Fatalf("a plain English prompt does not round trip:\n%s", stdout)
	}
}

// TestInfoDescribesTheCommittedGGUF needs no model directory: the file is in
// the repository.
func TestInfoDescribesTheCommittedGGUF(t *testing.T) {
	gguf := filepath.Join("..", "..", "testdata", "models", "stories260K.gguf")
	stdout, _, err := capture(t, func() error { return info(gguf, true) })
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"file      " + gguf + "\n",
		"arch      llama ",
		"L=5  d=64  ffn=172  heads=8/4  head_dim=8",
		"\nkv\n", // -kv prints the header
		"\nblock 0   ",
		"blk.0  ",
		"... 2 more",
		"   1,280 B/position kv\n", // 2 x 5 blocks x 4 kv heads x 8 x 4 bytes
		"roofline  ",
	} {
		if !strings.Contains(stdout, want) {
			t.Errorf("info does not print %q:\n%s", want, stdout)
		}
	}
	if err := info(filepath.Join(t.TempDir(), "absent.gguf"), false); err == nil {
		t.Error("info on a missing file succeeded")
	}
}

// TestCommasGroupsThousands: info's byte counts are read by eye.
func TestCommasGroupsThousands(t *testing.T) {
	for in, want := range map[uint64]string{0: "0", 999: "999", 1000: "1,000", 1280: "1,280", 816010912: "816,010,912"} {
		if got := commas(in); got != want {
			t.Errorf("commas(%d) = %q, want %q", in, got, want)
		}
	}
}

// TestLibraryRefsAreOneLinePerModelForScripts: `library -refs` is the form a
// script reads, so every line is NAME BYTES hf://... [hf://...].
func TestLibraryRefsAreOneLinePerModelForScripts(t *testing.T) {
	stdout, _, err := capture(t, func() error { return libraryCmd([]string{"-refs"}) })
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSuffix(stdout, "\n"), "\n")
	line := regexp.MustCompile(`^\S+ [0-9]+ hf://\S+( hf://\S+)?$`)
	for _, l := range lines {
		if !line.MatchString(l) {
			t.Fatalf("library -refs line %q is not NAME BYTES REF [TOWER]", l)
		}
	}
	human, _, err := capture(t, func() error { return libraryCmd(nil) })
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(human, "\n"); got != len(lines) || !strings.Contains(human, " GB") {
		t.Fatalf("library lists %d models and -refs %d", got, len(lines))
	}
}

// TestSubcommandsRefuseAModelTheyCannotUse: each says why, rather than
// running and printing nothing useful.
func TestSubcommandsRefuseAModelTheyCannotUse(t *testing.T) {
	path := modelPath(t, "stories260K.jlm")
	if _, _, err := capture(t, func() error { return embedCmd([]string{path, "hello"}) }); err == nil ||
		!strings.Contains(err.Error(), "not an embedding model") {
		t.Errorf("embed on a generative model: %v", err)
	}
	if _, _, err := capture(t, func() error { return batchCmd([]string{"-devices", "cpu", path, "hello"}) }); err == nil ||
		!strings.Contains(err.Error(), "batch needs a device") {
		t.Errorf("batch with no device: %v", err)
	}
	_, stderr, err := capture(t, func() error {
		return run(path, "hi", 2, 0, "cpu", -1, 0, 1<<30, false, false, false, false,
			&model.Sampler{}, "", nil, &chatSpec{}, "", 0, 0, nil, nil)
	})
	if err == nil || !strings.Contains(err.Error(), "carries no chat template") {
		t.Errorf("-chat on a base model: %v\n%s", err, stderr)
	}
	_, _, err = capture(t, func() error {
		return run(path, "hi", 2, 0, "cpu", -1, 0, 1<<30, false, false, false, false,
			&model.Sampler{}, "x.png", nil, nil, "", 0, 0, nil, nil)
	})
	if err == nil || !strings.Contains(err.Error(), "vision tower") {
		t.Errorf("-image on a model with no tower: %v", err)
	}
	_, _, err = capture(t, func() error {
		return run(path, "", 2, 8, "cpu", -1, 0, 1<<30, false, false, false, false,
			&model.Sampler{}, "", nil, nil, "", 0, 0, nil, nil)
	})
	// An empty prompt still carries the BOS, so -depth has something to
	// repeat; it must run rather than refuse.
	if err != nil {
		t.Errorf("-depth over an empty prompt: %v", err)
	}
	_, _, err = capture(t, func() error {
		return run(path, "hi", 2, 0, "cpu", -1, 0, 1<<30, false, false, false, false,
			&model.Sampler{}, "", nil, nil, "", 0, 1<<20, nil, nil)
	})
	if err == nil || !strings.Contains(err.Error(), "-kv-budget needs -kv-cache") {
		t.Errorf("-kv-budget with no -kv-cache: %v", err)
	}
}

// TestSpeedPrintsOneRatePerRepetition: speed's line is what a board reads, so
// every mode prints its line with one rate per timed repetition. No rate is
// checked against anything: this is the shape, not a measurement.
func TestSpeedPrintsOneRatePerRepetition(t *testing.T) {
	path := modelPath(t, "stories260K.jlm")
	if err := speedCmd([]string{"a.jlm", "b.jlm"}); err == nil || !strings.Contains(err.Error(), "usage: jitllm speed") {
		t.Fatalf("speed with two models: %v", err)
	}
	rate := `[0-9]+\.[0-9]{2}`
	// Each timed run is tens of milliseconds even on a fast host: Go's clock on
	// Windows ticks every 0.5 to 15.6 ms, and a run under one tick measures
	// zero, which speed refuses rather than print as a rate.
	for _, tc := range []struct {
		args []string
		line *regexp.Regexp
	}{
		{[]string{"-devices", "cpu", "-p", "256", "-n", "200", "-r", "2", "-gcstats", path},
			regexp.MustCompile(`(?m)^pp256 ` + rate + ` / ` + rate + ` tok/s   tg200 ` + rate + ` / ` + rate + ` tok/s$`)},
		{[]string{"-devices", "cpu", "-p", "8", "-r", "3", "-ttft", path},
			regexp.MustCompile(`(?m)^ttft [0-9]+ tok   cold [0-9.]+ ms \(open to the first prompt's token\)   warm ` +
				rate + ` / ` + rate + ` / ` + rate + ` ms \(median [0-9.]+, a fresh session\)$`)},
		{[]string{"-devices", "cpu", "-p", "4", "-n", "100", "-r", "1", "-sessions", "2", path},
			regexp.MustCompile(`(?m)^sessions 2 x tg100: one after another ` + rate + ` tok/s, together ` + rate +
				` tok/s \([0-9.]+x\)$`)},
	} {
		stdout, stderr, err := capture(t, func() error { return speedCmd(tc.args) })
		if err != nil {
			t.Fatalf("speed %v: %v\n%s", tc.args, err, stderr)
		}
		if !tc.line.MatchString(stdout) {
			t.Errorf("speed %v printed no line matching %s:\n%s", tc.args, tc.line, stdout)
		}
	}
}

// TestARunTheClockDidNotSeeIsNotARate: a zero duration, which a coarse clock
// gives a short run, is refused rather than divided into +Inf tok/s.
func TestARunTheClockDidNotSeeIsNotARate(t *testing.T) {
	if r, err := perSecond("tg4", 4, 0); err == nil {
		t.Fatalf("a zero-length run gave the rate %v", r)
	}
	if r, err := perSecond("tg4", 4, 2*time.Millisecond); err != nil || r != 2000 {
		t.Fatalf("4 tokens in 2 ms gave %v, %v; want 2000", r, err)
	}
}

// TestVerifyAgreesWithTheHostOnARealDevice runs the device gate the command
// ships, end to end, on a model every block of which a card can take.
func TestVerifyAgreesWithTheHostOnARealDevice(t *testing.T) {
	path := modelPath(t, "stories15M-q8_0.jlm")
	if err := verifyCmd([]string{"a", "b"}); err == nil || !strings.Contains(err.Error(), "usage: jitllm verify") {
		t.Fatalf("verify with two models: %v", err)
	}
	stdout, stderr, err := capture(t, func() error {
		return verifyCmd([]string{"-devices", "gpu:0", "-n", "8", "-dlogit", "0.94", path})
	})
	if err != nil && !strings.Contains(stdout, "device: ") {
		t.Skipf("NO DEVICE: verify could not open -devices gpu:0 (%v) -- this gate proved nothing", err)
	}
	if err != nil {
		t.Fatalf("verify: %v\n%s\n%s", err, stdout, stderr)
	}
	for _, want := range []string{"device: ", "model:  " + path + "  L=6 d=288", "\ncpu  ", "\ngpu  ",
		"6 blocks on device", "per token"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("verify printed no %q:\n%s", want, stdout)
		}
	}
}
