// httpbench is the one HTTP load harness every engine is measured with: a
// closed-loop concurrency sweep of streamed OpenAI-compatible requests, two or
// more endpoints run as interleaved arms. .notes/perf/http-bench.md is how to
// set each engine up for a fair arm.
//
//	go run ./dev/httpbench -arm jitllm=http://127.0.0.1:4242 -model stories15M
//	go run ./dev/httpbench -arm jitllm=http://a:4242 -arm llama=http://b:8081 -rounds 8
//	go run ./dev/httpbench -aa -arm jitllm=http://a:4242 -rounds 8   # A/A self-control
//	go run ./dev/httpbench -config run.json                          # hb.Config as JSON
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"time"

	"github.com/jitllm/jitllm/dev/httpbench/hb"
)

type list []string

func (l *list) String() string     { return strings.Join(*l, ",") }
func (l *list) Set(v string) error { *l = append(*l, v); return nil }

func ints(s string) ([]int, error) {
	var out []int
	for _, f := range strings.Split(s, ",") {
		if f = strings.TrimSpace(f); f == "" {
			continue
		}
		n, err := strconv.Atoi(f)
		if err != nil {
			return nil, err
		}
		out = append(out, n)
	}
	return out, nil
}

func main() {
	var arms, models, routes list
	flag.Var(&arms, "arm", "NAME=URL of an endpoint (repeat for arms; the first is the base of every ratio)")
	flag.Var(&models, "model", "NAME[:WEIGHT] of a workload model (repeat for a multi-model mix)")
	flag.Var(&routes, "route", "ARM/MODEL=URL[#SERVED_NAME]: where one arm serves one model (one server per model)")
	cfgPath := flag.String("config", "", "a JSON hb.Config; replaces every workload flag")
	apiKey := flag.String("api-key", os.Getenv("OPENAI_API_KEY"), "bearer token sent to every arm")
	http2 := flag.Bool("http2", false, "speak HTTP/2 (h2c on http://, ALPN on https://)")
	api := flag.String("api", "completions", "completions or chat")
	levels := flag.String("c", "1,2,4,8,16,32,64", "concurrency levels")
	requests := flag.Int("requests", 0, "requests per level per round per arm (0: 4 per slot, at least 8)")
	rounds := flag.Int("rounds", 1, "interleaved rounds per level; a ratio needs at least 6")
	warmup := flag.Int("warmup", 2, "requests per arm before the sweep, not measured")
	warmLevel := flag.Int("warmup-c", 1, "warm-up requests in flight at once (warm at the widest -c to pass first-use compilation of its batch widths)")
	lens := flag.String("prompt-lens", "128,512", "generated prompt lengths in words, one drawn per request")
	promptFile := flag.String("prompts", "", "prompt file: one per line, or {\"prompt\":...} per line in a .jsonl")
	seed := flag.Uint64("seed", 1, "workload seed")
	maxTok := flag.Int("max-tokens", 128, "max_tokens of every request")
	temp := flag.Float64("temperature", 0, "temperature of every request (0: greedy on every engine)")
	noIgnoreEOS := flag.Bool("no-ignore-eos", false, "do not send ignore_eos (the at-max column shows what each engine did)")
	sloTTFT := flag.Duration("slo-ttft", 0, "goodput TTFT bound (0: none)")
	sloTPOT := flag.Duration("slo-tpot", 0, "goodput bound on a request's mean inter-chunk gap (0: none)")
	timeout := flag.Duration("timeout", 10*time.Minute, "per-request timeout")
	aa := flag.Bool("aa", false, "A/A self-control: run the one -arm as two arms")
	out := flag.String("o", "httpbench.json", "JSON file of the config, every sample and the report")
	flag.Parse()

	var c hb.Config
	if *cfgPath != "" {
		b, err := os.ReadFile(*cfgPath)
		if err != nil {
			fail(err)
		}
		if err := json.Unmarshal(b, &c); err != nil {
			fail(fmt.Errorf("%s: %v", *cfgPath, err))
		}
	} else {
		for _, a := range arms {
			name, url, ok := strings.Cut(a, "=")
			if !ok {
				fail(fmt.Errorf("-arm %q: want NAME=URL", a))
			}
			c.Arms = append(c.Arms, hb.Arm{Name: name, URL: url, APIKey: *apiKey, HTTP2: *http2})
		}
		if *aa {
			if len(c.Arms) != 1 {
				fail(fmt.Errorf("-aa takes exactly one -arm"))
			}
			b := c.Arms[0]
			c.Arms[0].Name, b.Name = b.Name+"-A", b.Name+"-B"
			c.Arms = append(c.Arms, b)
		}
		for _, m := range models {
			name, w, ok := strings.Cut(m, ":")
			wt := 1.0
			if ok {
				f, err := strconv.ParseFloat(w, 64)
				if err != nil {
					fail(fmt.Errorf("-model %q: %v", m, err))
				}
				wt = f
			}
			c.Models = append(c.Models, hb.Model{Name: name, Weight: wt})
		}
		for _, r := range routes {
			lhs, rhs, ok1 := strings.Cut(r, "=")
			arm, mdl, ok2 := strings.Cut(lhs, "/")
			if !ok1 || !ok2 {
				fail(fmt.Errorf("-route %q: want ARM/MODEL=URL[#SERVED_NAME]", r))
			}
			url, served, _ := strings.Cut(rhs, "#")
			found := false
			for i := range c.Arms {
				if c.Arms[i].Name == arm || (*aa && strings.TrimSuffix(strings.TrimSuffix(c.Arms[i].Name, "-A"), "-B") == arm) {
					if c.Arms[i].Models == nil {
						c.Arms[i].Models = map[string]hb.Route{}
					}
					c.Arms[i].Models[mdl] = hb.Route{URL: url, Name: served}
					found = true
				}
			}
			if !found {
				fail(fmt.Errorf("-route %q: no arm %q", r, arm))
			}
		}
		var err error
		if c.Levels, err = ints(*levels); err != nil {
			fail(fmt.Errorf("-c: %v", err))
		}
		if c.PromptLens, err = ints(*lens); err != nil {
			fail(fmt.Errorf("-prompt-lens: %v", err))
		}
		c.API, c.Requests, c.Rounds, c.Warmup = *api, *requests, *rounds, *warmup
		c.WarmupLevel = *warmLevel
		c.PromptFile, c.Seed, c.MaxTokens, c.Temperature = *promptFile, *seed, *maxTok, *temp
		c.Extra = map[string]any{"ignore_eos": !*noIgnoreEOS}
		if *noIgnoreEOS {
			c.Extra = map[string]any{}
		}
		c.SLOTTFT, c.SLOTPOT, c.Timeout = *sloTTFT, *sloTPOT, *timeout
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	res, err := hb.Run(ctx, c, func(s string) { fmt.Fprintln(os.Stderr, s) })
	if res == nil {
		fail(err)
	}
	rep := hb.Reduce(res)
	rep.Print(os.Stdout, res.Config)
	b, jerr := json.MarshalIndent(struct {
		*hb.Result
		Report hb.Report `json:"report"`
	}{res, rep}, "", " ")
	if jerr != nil {
		fail(jerr)
	}
	if werr := os.WriteFile(*out, b, 0o644); werr != nil {
		fail(werr)
	}
	fmt.Fprintf(os.Stderr, "samples: %s\n", *out)
	if err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "httpbench:", err)
	os.Exit(1)
}
