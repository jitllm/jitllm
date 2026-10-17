// deploysearch sweeps a bounded set of jitllmd serving settings on a named
// request mix and reports the Pareto set against the unchanged adaptive
// default, measured in the same pass.
//
// The settings are jitllmd serve flags (-max-batch, -prompt-chunk, -step-cost,
// -fairness, -session-pool, -mem-cache-max, ...), each of which maps to a
// server.Config field. A setting is a restart: every sample starts a fresh
// server, warms it, runs the mix through dev/httpbench's harness (package hb)
// at one concurrency, records host and device memory, and stops it. Rounds
// are interleaved over the configs (A B C, then C B A), and the first -bin's
// "default" (no extra flags) is the control every ratio is taken against.
// An identical second copy of the control ("aa") is run as a candidate, so a
// ratio that should be 1.0 shows what this box can resolve (RULE 2).
//
// More than one -bin compares builds, for example main against an integration
// branch: each build's default joins the sweep, and -grid-all sweeps the knobs
// on every build rather than the first alone.
//
// It changes no default. It prints "candidate" only for a setting with a
// quotable win on goodput or p99 TTFT and no quotable regression; that is a
// reason to repeat the experiment, not to change a default.
//
// A timing row on a shared GPU box runs under that card's lock, both arms on
// the same card in one pass:
//
//	flock ~/gpu3.lock env CUDA_VISIBLE_DEVICES=3 deploysearch -bin main=./jitllmd \
//	    -models $JITLLM_MODELS -model m.jlm -mix mixed -base-arg -devices -base-arg cuda
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/jitllm/jitllm/dev/httpbench/hb"
)

type list []string

func (l *list) String() string     { return strings.Join(*l, ",") }
func (l *list) Set(v string) error { *l = append(*l, v); return nil }

// mixes are the named request mixes. Prompt lengths are words, about a token
// each; the SLOs define goodput. A server's -max-seq must cover the longest
// prompt plus max_tokens (about 1.4 tokens a word).
var mixes = map[string]hb.Config{
	"mixed": {
		Levels: []int{4}, Requests: 16, PromptLens: []int{32, 128, 512}, MaxTokens: 64,
		SLOTTFT: 3 * time.Second, SLOTPOT: 100 * time.Millisecond,
	},
	"long": {
		Levels: []int{4}, Requests: 16, PromptLens: []int{1024, 2048}, MaxTokens: 64,
		SLOTTFT: 12 * time.Second, SLOTPOT: 150 * time.Millisecond,
	},
	"decode": {
		Levels: []int{32}, Requests: 64, PromptLens: []int{64}, MaxTokens: 256,
		SLOTTFT: time.Second, SLOTPOT: 80 * time.Millisecond,
	},
}

// defaultKnobs is the bounded sweep, one factor at a time around the default.
var defaultOrder = []string{"max-batch", "prompt-chunk", "step-cost", "fairness", "sessions", "mem-cache-max"}
var defaultKnobs = map[string][]string{
	"max-batch":     {"8"},
	"prompt-chunk":  {"128", "512"},
	"step-cost":     {"2", "4"},
	"fairness":      {"0", "50"},
	"sessions":      {"4"},
	"mem-cache-max": {"256M"},
}

func main() {
	var bins, knobs, combos, baseArgs list
	flag.Var(&bins, "bin", "NAME=PATH of a jitllmd build (repeat; the first build's default is the control)")
	flag.Var(&knobs, "knob", "FLAG=V1,V2 jitllmd serve flag to sweep one factor at a time (repeat; replaces the built-in set)")
	flag.Var(&combos, "combo", "FLAG=V,FLAG=V a bundle of flags run as one candidate (repeat)")
	flag.Var(&baseArgs, "base-arg", "an argument given to every serve (repeat: -base-arg -devices -base-arg cuda)")
	models := flag.String("models", os.Getenv("JITLLM_MODELS"), "model directory")
	model := flag.String("model", "", "the .jlm to load")
	id := flag.String("id", "model", "model id")
	mix := flag.String("mix", "mixed", "named request mix: mixed, long or decode")
	mixFile := flag.String("mix-file", "", "a JSON hb.Config (one arm is filled in) replacing the named mix")
	rounds := flag.Int("rounds", 6, "interleaved rounds; a quotable ratio needs at least 6")
	warmup := flag.Int("warmup", 2, "requests before each timed run, not measured")
	gridAll := flag.Bool("grid-all", false, "sweep the knobs on every -bin, not the first alone")
	aa := flag.Bool("aa", true, "run an identical copy of the control as a candidate (A/A self-control)")
	port := flag.Int("port", 4290, "first TCP port; each sample uses the next")
	start := flag.Duration("start-timeout", 20*time.Minute, "how long a server may take to load and listen")
	outDir := flag.String("o", "deploysearch-out", "directory for the server logs and results.json")
	flag.Parse()

	if len(bins) == 0 || *model == "" {
		fmt.Fprintln(os.Stderr, "usage: deploysearch -bin main=./jitllmd -models DIR -model m.jlm [-mix mixed] [flags]")
		os.Exit(2)
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fail(err)
	}

	var base hb.Config
	if *mixFile != "" {
		b, err := os.ReadFile(*mixFile)
		if err != nil {
			fail(err)
		}
		if err := json.Unmarshal(b, &base); err != nil {
			fail(fmt.Errorf("%s: %v", *mixFile, err))
		}
	} else {
		m, ok := mixes[*mix]
		if !ok {
			fail(fmt.Errorf("-mix %q: want mixed, long or decode", *mix))
		}
		base = m
	}
	base.Models = []hb.Model{{Name: *id, Weight: 1}}
	base.Rounds, base.Warmup, base.Seed = 1, *warmup, 1

	order, kmap := defaultOrder, defaultKnobs
	if len(knobs) > 0 {
		order, kmap = nil, map[string][]string{}
		for _, k := range knobs {
			f, v, ok := strings.Cut(k, "=")
			if !ok {
				fail(fmt.Errorf("-knob %q: want FLAG=V1,V2", k))
			}
			order = append(order, f)
			kmap[f] = strings.Split(v, ",")
		}
	}

	var cfgs []Config
	for i, b := range bins {
		name, path, ok := strings.Cut(b, "=")
		if !ok {
			fail(fmt.Errorf("-bin %q: want NAME=PATH", b))
		}
		var g []Config
		if i == 0 || *gridAll {
			var err error
			if g, err = Grid(path, order, kmap, combos); err != nil {
				fail(err)
			}
		} else {
			g = []Config{{Name: "default", Bin: path}}
		}
		for _, c := range g {
			if i > 0 {
				c.Name = name + "/" + c.Name
			}
			cfgs = append(cfgs, c)
		}
	}
	if *aa {
		cfgs = append(cfgs, Config{Name: "aa", Bin: cfgs[0].Bin})
	}
	control := cfgs[0].Name

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	var samples []Sample
	port0 := *port
	for r := 0; r < *rounds; r++ {
		for _, i := range Order(len(cfgs), r) {
			c := cfgs[i]
			fmt.Fprintf(os.Stderr, "round %d/%d  %s\n", r+1, *rounds, c.Name)
			m, err := runOne(ctx, c, base, runSpec{
				models: *models, model: *model, id: *id, base: baseArgs, port: port0, start: *start,
				log: filepath.Join(*outDir, fmt.Sprintf("serve-%s-r%d.log", sanitize(c.Name), r+1)),
			})
			port0++
			if err != nil {
				fmt.Fprintf(os.Stderr, "  %s round %d: %v\n", c.Name, r+1, err)
				if ctx.Err() != nil {
					goto done
				}
				continue
			}
			fmt.Fprintf(os.Stderr, "  TTFT p50 %.0f p99 %.0f ms  ITL p99 %.1f ms  goodput %.2f/s  %.0f tok/s  host %.0f MiB  dev %.0f MiB  errors %d/%d\n",
				m.TTFTp50, m.TTFTp99, m.ITLp99, m.Goodput, m.AggRate, m.HostMiB, m.VRAMMiB, m.Errors, m.Requests)
			samples = append(samples, Sample{Config: c.Name, Round: r + 1, M: m})
		}
	}
done:
	rs := Compare(samples, control)
	report(os.Stdout, rs, control, *rounds)
	b, _ := json.MarshalIndent(struct {
		Mix     string
		Control string
		Configs []Config
		Samples []Sample
		Ratios  []Ratios
	}{*mix, control, cfgs, samples, rs}, "", " ")
	if err := os.WriteFile(filepath.Join(*outDir, "results.json"), b, 0o644); err != nil {
		fail(err)
	}
}

func sanitize(s string) string {
	return strings.Map(func(r rune) rune {
		if r == '/' || r == '=' || r == '+' || r == ',' {
			return '_'
		}
		return r
	}, s)
}

func report(w io.Writer, rs []Ratios, control string, rounds int) {
	front := Pareto(rs)
	fmt.Fprintf(w, "\ncontrol: %s (unchanged defaults), %d rounds; ratios are config/control per round, median [IQR/median]\n", control, rounds)
	fmt.Fprintf(w, "latency and memory below 1 is better, goodput above 1 is better; a quotable ratio needs 6 rounds and IQR/median <= 0.10\n\n")
	fmt.Fprintf(w, "%-34s %5s %15s %15s %15s %15s %15s  %s\n", "config", "pareto", "TTFT p99", "ITL p99", "goodput", "memory", "load time", "verdict")
	cell := func(g hb.RatioGate) string {
		s := fmt.Sprintf("%.3f [%.3f]", float64(g.Median), float64(g.IQR)/float64(g.Median))
		if g.Refused != "" {
			s += "*"
		}
		return s
	}
	for _, r := range rs {
		p := ""
		if front[r.Config] {
			p = "yes"
		}
		v := Verdict(r)
		if r.Config == control {
			v = "control"
		}
		fmt.Fprintf(w, "%-34s %5s %15s %15s %15s %15s %15s  %s\n", r.Config, p, cell(r.TTFTp99), cell(r.ITLp99), cell(r.Goodput), cell(r.Mem), cell(r.LoadSec), v)
	}
	fmt.Fprintf(w, "\n* refused by the gate (too few rounds, a failed round or IQR/median above 0.10)\n\n")
	fmt.Fprintf(w, "%-34s %12s %12s %14s %12s\n", "median absolute", "TTFT p99 ms", "ITL p99 ms", "goodput req/s", "memory MiB")
	for _, r := range rs {
		fmt.Fprintf(w, "%-34s %12.0f %12.1f %14.2f %12.0f\n", r.Config, r.MedianTTFT99, r.MedianITL99, r.MedianGoodput, r.MedianMemMiB)
	}
}

type runSpec struct {
	models, model, id string
	base              []string
	port              int
	start             time.Duration
	log               string
}

// runOne starts one server with c's flags, runs the mix against it, and
// stops it.
func runOne(ctx context.Context, c Config, mix hb.Config, sp runSpec) (Metrics, error) {
	var m Metrics
	addr := "127.0.0.1:" + strconv.Itoa(sp.port)
	args := []string{"serve", "-addr", addr, "-models", sp.models, "-load", sp.model, "-id", sp.id}
	args = append(args, sp.base...)
	args = append(args, c.Args()...)
	lf, err := os.Create(sp.log)
	if err != nil {
		return m, err
	}
	defer lf.Close()
	cmd := exec.Command(c.Bin, args...)
	cmd.Stdout, cmd.Stderr = lf, lf
	t0 := time.Now()
	if err := cmd.Start(); err != nil {
		return m, err
	}
	exited := make(chan struct{})
	go func() {
		cmd.Wait()
		close(exited)
	}()
	defer func() {
		cmd.Process.Signal(os.Interrupt)
		select {
		case <-exited:
		case <-time.After(30 * time.Second):
			cmd.Process.Kill()
			<-exited
		}
	}()

	url := "http://" + addr
	if err := waitServing(ctx, url, sp.id, sp.start, exited); err != nil {
		return m, err
	}
	m.LoadSec = time.Since(t0).Seconds()

	var (
		mu   sync.Mutex
		vram float64
	)
	pollCtx, cancel := context.WithCancel(ctx)
	polled := make(chan struct{})
	go func() {
		defer close(polled)
		for {
			if v := deviceMiB(cmd.Process.Pid); v > 0 {
				mu.Lock()
				vram = max(vram, v)
				mu.Unlock()
			}
			select {
			case <-pollCtx.Done():
				return
			case <-time.After(time.Second):
			}
		}
	}()

	cfg := mix
	cfg.Arms = []hb.Arm{{Name: c.Name, URL: url}}
	res, err := hb.Run(ctx, cfg, nil)
	cancel()
	<-polled
	if err != nil {
		return m, err
	}
	m.HostMiB = hostPeakMiB(cmd.Process.Pid)
	mu.Lock()
	m.VRAMMiB = vram
	mu.Unlock()

	var all []hb.Sample
	var wall time.Duration
	for _, lv := range res.Levels {
		all = append(all, lv.Samples...)
		wall += lv.Wall
	}
	st := hb.Summarise(all, wall, res.Config)
	m.TTFTp50, m.TTFTp99, m.ITLp99 = float64(st.TTFTp50), float64(st.TTFTp99), float64(st.ITLp99)
	m.Goodput, m.AggRate = float64(st.Goodput), float64(st.AggRate)
	m.Errors, m.Requests = st.Errors, st.Requests
	if st.Errors > 0 {
		return m, fmt.Errorf("%d of %d requests failed", st.Errors, st.Requests)
	}
	return m, nil
}

// waitServing polls /v1/models until the model id is listed.
func waitServing(ctx context.Context, url, id string, limit time.Duration, exited <-chan struct{}) error {
	deadline := time.Now().Add(limit)
	for time.Now().Before(deadline) {
		select {
		case <-exited:
			return fmt.Errorf("server exited before serving (see its log)")
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
		resp, err := http.Get(url + "/v1/models")
		if err != nil {
			continue
		}
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode == 200 && strings.Contains(string(b), `"`+id+`"`) {
			return nil
		}
	}
	return fmt.Errorf("not serving %q after %v", id, limit)
}

// hostPeakMiB is the process's peak resident set (VmHWM) in MiB, 0 when
// /proc is absent.
func hostPeakMiB(pid int) float64 {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/status")
	if err != nil {
		return 0
	}
	for _, ln := range strings.Split(string(b), "\n") {
		if rest, ok := strings.CutPrefix(ln, "VmHWM:"); ok {
			f := strings.Fields(rest)
			if len(f) > 0 {
				kb, _ := strconv.ParseFloat(f[0], 64)
				return kb / 1024
			}
		}
	}
	return 0
}

// deviceMiB is the device memory nvidia-smi attributes to pid, 0 when there is
// none to report.
func deviceMiB(pid int) float64 {
	out, err := exec.Command("nvidia-smi", "--query-compute-apps=pid,used_memory",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0
	}
	for _, ln := range strings.Split(string(out), "\n") {
		p, u, ok := strings.Cut(ln, ",")
		if !ok {
			continue
		}
		if n, _ := strconv.Atoi(strings.TrimSpace(p)); n == pid {
			v, _ := strconv.ParseFloat(strings.TrimSpace(u), 64)
			return v
		}
	}
	return 0
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "deploysearch:", err)
	os.Exit(1)
}
