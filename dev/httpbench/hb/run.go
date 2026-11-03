// Package hb is the HTTP load harness behind dev/httpbench: one closed-loop
// concurrency sweep, measured the same way against every OpenAI-compatible
// engine, with two or more engines run as interleaved arms.
package hb

import (
	"bufio"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

func nan() float64 {
	return math.NaN()
}

// Arm is one endpoint under test.
type Arm struct {
	Name   string `json:"name"`
	URL    string `json:"url"`
	APIKey string `json:"api_key,omitempty"`
	// HTTP2 speaks HTTP/2: cleartext (h2c, prior knowledge) on an http://
	// URL, ALPN on https://. Off, every request is HTTP/1.1.
	HTTP2 bool `json:"http2"`
	// Models maps a workload model name onto where this arm serves it. An
	// engine that serves one model per process has a URL per model; one that
	// names a model differently has its own name for it. A model not listed
	// is served at URL under the workload's name.
	Models map[string]Route `json:"models,omitempty"`
	// Extra is merged into every request body, replacing the workload's
	// default extras when a key collides; a null value removes the default.
	Extra map[string]any `json:"extra,omitempty"`
}

// Route is where an arm serves one model.
type Route struct {
	URL  string `json:"url,omitempty"`
	Name string `json:"name,omitempty"`
}

// Model is one model of the workload and its share of the requests.
type Model struct {
	Name   string  `json:"name"`
	Weight float64 `json:"weight"`
}

// Config is a whole run. dev/httpbench fills it from flags or reads it from a
// JSON file of this shape.
type Config struct {
	Arms   []Arm   `json:"arms"`
	Models []Model `json:"models"`
	API    string  `json:"api"` // "completions" (default) or "chat"

	Levels []int `json:"levels"`
	// Requests is how many requests one level sends per round per arm; 0 is
	// four per concurrent slot, at least 8.
	Requests int `json:"requests"`
	Rounds   int `json:"rounds"`
	Warmup   int `json:"warmup"`
	// WarmupLevel is how many warm-up requests run at once: zero is one at a
	// time. An engine that compiles or tunes a kernel per batch width on first
	// use is only warm for the widths its warm-up reached, so a measurement at
	// concurrency c warms at c.
	WarmupLevel int `json:"warmup_level,omitempty"`

	// PromptLens is the mix of generated prompt lengths, in words() (about a
	// token each on a llama-family tokenizer; the measured count is in each
	// sample's prompt_tokens). PromptFile replaces generation.
	PromptLens []int  `json:"prompt_lens"`
	PromptFile string `json:"prompt_file,omitempty"`
	Seed       uint64 `json:"seed"`

	MaxTokens   int            `json:"max_tokens"`
	Temperature float64        `json:"temperature"`
	Extra       map[string]any `json:"extra"`

	// SLOTTFT and SLOTPOT define goodput: a request counts when its TTFT and
	// its mean inter-chunk gap are both within them. Zero is no bound.
	SLOTTFT time.Duration `json:"slo_ttft_ns"`
	SLOTPOT time.Duration `json:"slo_tpot_ns"`

	Timeout time.Duration `json:"timeout_ns"`
}

// Defaults fills what a Config left zero.
func (c *Config) Defaults() {
	if c.API == "" {
		c.API = "completions"
	}
	if len(c.Levels) == 0 {
		c.Levels = []int{1, 2, 4, 8, 16, 32, 64}
	}
	if c.Rounds == 0 {
		c.Rounds = 1
	}
	if len(c.PromptLens) == 0 {
		c.PromptLens = []int{128, 512}
	}
	if c.MaxTokens == 0 {
		c.MaxTokens = 128
	}
	if c.Extra == nil {
		c.Extra = map[string]any{"ignore_eos": true}
	}
	if c.Timeout == 0 {
		c.Timeout = 10 * time.Minute
	}
	for i := range c.Models {
		if c.Models[i].Weight == 0 {
			c.Models[i].Weight = 1
		}
	}
}

// Validate refuses a Config that cannot produce a row.
func (c *Config) Validate() error {
	if len(c.Arms) == 0 {
		return fmt.Errorf("no arm: name at least one endpoint")
	}
	if len(c.Models) == 0 {
		return fmt.Errorf("no model: name at least one")
	}
	if c.API != "completions" && c.API != "chat" {
		return fmt.Errorf("api %q: completions or chat", c.API)
	}
	seen := map[string]bool{}
	for _, a := range c.Arms {
		if a.Name == "" || a.URL == "" {
			return fmt.Errorf("an arm needs a name and a url: %+v", a)
		}
		if seen[a.Name] {
			return fmt.Errorf("arm %q named twice", a.Name)
		}
		seen[a.Name] = true
	}
	for _, m := range c.Models {
		if m.Weight < 0 {
			return fmt.Errorf("model %q: negative weight", m.Name)
		}
	}
	for _, l := range c.Levels {
		if l < 1 {
			return fmt.Errorf("concurrency %d", l)
		}
	}
	return nil
}

// job is one request of a level, before it is bound to an arm: the same jobs
// go to every arm of a round, so the arms see the same prompts and the same
// model mix.
type job struct {
	model  string
	prompt string
}

// words() is the generated prompts' vocabulary: common English words(), so a
// llama-family tokenizer spends about a token on each.
func words() []string {
	return strings.Fields(`the of and to in is was that for it with as his on be at by had
are but from or have an they which one you were all we her she there would their
will when who him been has more if no out so said what up its about than into them
can only other time new some could these two may first then do any like my now over
such our man me even most made after also did many before must through back years
where much your way well down should because each just those people how too little
state good very make world still own see men work long get here between both life
being under never day same another know while last might us great old year off come
since against go came right used take three`)
}

func generate(r *rand.Rand, n int) string {
	var b strings.Builder
	vocab := words()
	for i := range n {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(vocab[r.IntN(len(vocab))])
	}
	return b.String()
}

// LoadPrompts reads a prompt file: JSON lines of {"prompt": "..."} when the
// name ends .jsonl, otherwise one prompt per line.
func LoadPrompts(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	jsonl := strings.HasSuffix(path, ".jsonl")
	for sc.Scan() {
		line := sc.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		if jsonl {
			var p struct {
				Prompt string `json:"prompt"`
			}
			if err := json.Unmarshal([]byte(line), &p); err != nil {
				return nil, fmt.Errorf("%s: %v", path, err)
			}
			line = p.Prompt
		}
		out = append(out, line)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s: no prompts", path)
	}
	return out, nil
}

// jobs is a level's requests for one round, from a seed fixed by the level and
// the round so every arm of the round gets the same list and a rerun gets the
// same run.
func (c *Config) jobs(prompts []string, level, round, n int) []job {
	r := rand.New(rand.NewPCG(c.Seed, uint64(level)<<32|uint64(round)))
	var total float64
	for _, m := range c.Models {
		total += m.Weight
	}
	out := make([]job, n)
	for i := range out {
		x := r.Float64() * total
		out[i].model = c.Models[len(c.Models)-1].Name
		for _, m := range c.Models {
			if x < m.Weight {
				out[i].model = m.Name
				break
			}
			x -= m.Weight
		}
		if len(prompts) > 0 {
			out[i].prompt = prompts[r.IntN(len(prompts))]
		} else {
			out[i].prompt = generate(r, c.PromptLens[r.IntN(len(c.PromptLens))])
		}
	}
	return out
}

// Client builds an arm's HTTP client: no connection cap, idle connections
// kept for the widest level so a level does not pay connection setup, and
// HTTP/2 only when the arm asks for it.
func Client(a Arm, maxLevel int) *http.Client {
	tr := &http.Transport{
		MaxIdleConns:        0,
		MaxIdleConnsPerHost: maxLevel + 1,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  true,
		TLSClientConfig:     &tls.Config{},
	}
	p := new(http.Protocols)
	if a.HTTP2 {
		if strings.HasPrefix(a.URL, "https://") {
			p.SetHTTP2(true)
		} else {
			p.SetUnencryptedHTTP2(true)
		}
	} else {
		p.SetHTTP1(true)
	}
	tr.Protocols = p
	return &http.Client{Transport: tr}
}

// Level is one arm at one concurrency for one round.
type Level struct {
	Arm     string        `json:"arm"`
	Level   int           `json:"concurrency"`
	Round   int           `json:"round"`
	Wall    time.Duration `json:"wall_ns"`
	Samples []Sample      `json:"samples"`
}

// Result is a whole run: every level of every arm, in the order they ran.
type Result struct {
	Config Config  `json:"config"`
	Levels []Level `json:"levels"`
}

// Run runs the sweep. For each concurrency, round by round, every arm runs the
// round's requests in turn (A B A B ...), so a drift in the box over the run
// lands on both arms rather than on whichever ran second.
func Run(ctx context.Context, c Config, log func(string)) (*Result, error) {
	c.Defaults()
	if err := c.Validate(); err != nil {
		return nil, err
	}
	var prompts []string
	if c.PromptFile != "" {
		p, err := LoadPrompts(c.PromptFile)
		if err != nil {
			return nil, err
		}
		prompts = p
	}
	maxLevel := 0
	for _, l := range c.Levels {
		maxLevel = max(maxLevel, l)
	}
	clients := make([]*http.Client, len(c.Arms))
	for i, a := range c.Arms {
		clients[i] = Client(a, maxLevel)
		defer clients[i].CloseIdleConnections()
	}
	if c.Warmup > 0 {
		for i, a := range c.Arms {
			for _, s := range runJobs(ctx, c, a, clients[i], c.jobs(prompts, 0, 0, c.Warmup), max(c.WarmupLevel, 1)) {
				if s.Err != "" {
					return nil, fmt.Errorf("arm %s: warm-up request failed: %s", a.Name, s.Err)
				}
			}
		}
	}
	res := &Result{Config: c}
	for _, level := range c.Levels {
		n := c.Requests
		if n == 0 {
			n = max(4*level, 8)
		}
		for round := range c.Rounds {
			js := c.jobs(prompts, level, round+1, n)
			for i, a := range c.Arms {
				if err := ctx.Err(); err != nil {
					return res, err
				}
				t0 := time.Now()
				ss := runJobs(ctx, c, a, clients[i], js, level)
				lv := Level{Arm: a.Name, Level: level, Round: round + 1, Wall: time.Since(t0), Samples: ss}
				res.Levels = append(res.Levels, lv)
				if log != nil {
					log(fmt.Sprintf("%-12s c=%-3d round %d/%d  %s", a.Name, level, round+1, c.Rounds, summaryLine(lv)))
				}
			}
		}
	}
	return res, nil
}

// runJobs runs js against arm a with `level` requests in flight, each worker
// taking the next job when its last one ends (a closed loop).
func runJobs(ctx context.Context, c Config, a Arm, hc *http.Client, js []job, level int) []Sample {
	out := make([]Sample, len(js))
	next := make(chan int)
	var wg sync.WaitGroup
	for range min(level, len(js)) {
		wg.Go(func() {
			for i := range next {
				out[i] = do(ctx, c, a, hc, js[i])
				out[i].Arm, out[i].Model, out[i].Level = a.Name, js[i].model, level
			}
		})
	}
	for i := range js {
		next <- i
	}
	close(next)
	wg.Wait()
	return out
}

func do(ctx context.Context, c Config, a Arm, hc *http.Client, j job) Sample {
	r := Request{URL: a.URL, APIKey: a.APIKey, API: c.API, Model: j.model, Prompt: j.prompt,
		Max: c.MaxTokens, Temp: c.Temperature, Extra: map[string]any{}}
	if rt, ok := a.Models[j.model]; ok {
		if rt.URL != "" {
			r.URL = rt.URL
		}
		if rt.Name != "" {
			r.Model = rt.Name
		}
	}
	for k, v := range c.Extra {
		r.Extra[k] = v
	}
	for k, v := range a.Extra {
		// null removes a default extra the arm's engine refuses.
		if v == nil {
			delete(r.Extra, k)
			continue
		}
		r.Extra[k] = v
	}
	rctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()
	return Do(rctx, hc, r)
}
