// race streams one prompt through two OpenAI-compatible engines at the same
// instant and shows them side by side: the text as it arrives, time to first
// token, the live and final decode rate, the total time, the tokens
// generated, and which finished first. It is a demonstration, not a
// measurement: one request per side is one sample, and a board row comes from
// dev/httpbench or scripts/vs-llamacpp.sh (AGENTS.md RULE 2).
//
//	go run . -left jitllm=http://127.0.0.1:4242 -right other=http://127.0.0.1:8080 \
//	  -model model -prompt "Explain how a CPU cache works." -max-tokens 256 -warm 2
//
// Each side is NAME=URL, the URL the endpoint's base without /v1. The header
// lines (-hw, -setup) and each side's label (-left-label, -right-label) are
// where every difference between the two arms is stated: the format, the
// version, the device each one runs on.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"net/http"
	"os"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/jitllm/jitllm/dev/httpbench/hb"
)

func main() {
	left := flag.String("left", "", "NAME=URL of the left engine")
	right := flag.String("right", "", "NAME=URL of the right engine")
	modelName := flag.String("model", "model", "the model name sent to both sides")
	leftModel := flag.String("left-model", "", "the left side's model name (default -model)")
	rightModel := flag.String("right-model", "", "the right side's model name (default -model)")
	prompt := flag.String("prompt", "Explain, step by step, how a CPU cache works.", "the prompt sent to both sides")
	promptFile := flag.String("prompt-file", "", "read the prompt from this file instead")
	leftPrompt := flag.String("left-prompt", "", "the left side's prompt (default -prompt)")
	rightPrompt := flag.String("right-prompt", "", "the right side's prompt (default -prompt)")
	leftLabel := flag.String("left-label", "", "a line under the left engine's name: version, backend, format, device")
	rightLabel := flag.String("right-label", "", "a line under the right engine's name")
	hw := flag.String("hw", "", "header line: the hardware")
	setup := flag.String("setup", "", "header line: the model, quantization and how the arms differ")
	api := flag.String("api", "chat", "chat or completions")
	maxTok := flag.Int("max-tokens", 256, "max_tokens of both requests")
	ignoreEOS := flag.Bool("ignore-eos", true, "send ignore_eos so both sides generate exactly -max-tokens")
	warm := flag.Int("warm", 1, "throwaway races before the shown one")
	hold := flag.Duration("hold", 4*time.Second, "how long the finished screen stays before exiting")
	out := flag.String("o", "", "write both samples and the summary as JSON here")
	timeout := flag.Duration("timeout", 10*time.Minute, "the whole race's timeout")
	leftExtra := flag.String("left-extra", "", "a JSON object merged into the left side's request body")
	rightExtra := flag.String("right-extra", "", "a JSON object merged into the right side's request body, e.g. {\"cache_prompt\":false}")
	noTUI := flag.Bool("no-tui", false, "race without the screen and print the summary only")
	flag.Parse()

	if *promptFile != "" {
		b, err := os.ReadFile(*promptFile)
		if err != nil {
			fail(err)
		}
		*prompt = strings.TrimSpace(string(b))
	}
	var sides [2]Side
	for i, s := range []struct{ arm, model, prompt, label, extra string }{
		{*left, *leftModel, *leftPrompt, *leftLabel, *leftExtra},
		{*right, *rightModel, *rightPrompt, *rightLabel, *rightExtra},
	} {
		name, url, ok := strings.Cut(s.arm, "=")
		if !ok || name == "" || url == "" {
			fail(fmt.Errorf("-left and -right are NAME=URL, got %q", s.arm))
		}
		r := hb.Request{URL: url, API: *api, Model: or(s.model, *modelName), Prompt: or(s.prompt, *prompt), Max: *maxTok}
		r.Extra = map[string]any{}
		if s.extra != "" {
			if err := json.Unmarshal([]byte(s.extra), &r.Extra); err != nil {
				fail(fmt.Errorf("%s: -extra: %v", name, err))
			}
		}
		if *ignoreEOS {
			r.Extra["ignore_eos"] = true
		}
		sides[i] = Side{Name: name, Label: s.label, Req: r}
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	clients := [2]*http.Client{hb.Client(hb.Arm{URL: sides[0].Req.URL}, 1), hb.Client(hb.Arm{URL: sides[1].Req.URL}, 1)}
	if *warm > 0 {
		fmt.Fprintf(os.Stderr, "warming both engines (%d run(s))…\n", *warm)
		if err := Warm(ctx, sides, clients, *warm); err != nil {
			fail(err)
		}
	}

	m := &model{sides: sides, hold: *hold, events: make(chan Event, 1024)}
	m.header = []string{*hw, *setup}
	if *noTUI {
		m.results = Race(ctx, sides, clients, nil)
	} else {
		m.start = func() {
			go Race(ctx, sides, clients, m.events)
		}
		if _, err := tea.NewProgram(m, tea.WithAltScreen()).Run(); err != nil {
			fail(err)
		}
		if m.panes[0].done == nil || m.panes[1].done == nil {
			os.Exit(1)
		}
	}
	sum := Summary(sides, m.results)
	fmt.Println(sum)
	if *out != "" {
		b, err := json.MarshalIndent(map[string]any{"sides": []any{
			map[string]any{"name": sides[0].Name, "label": sides[0].Label, "sample": m.results[0], "decode_tok_s": finite(m.results[0].DecodeRate())},
			map[string]any{"name": sides[1].Name, "label": sides[1].Label, "sample": m.results[1], "decode_tok_s": finite(m.results[1].DecodeRate())},
		}, "summary": sum, "header": m.header}, "", "  ")
		if err != nil {
			fail(err)
		}
		if err := os.WriteFile(*out, b, 0o644); err != nil {
			fail(err)
		}
	}
}

func or(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// finite is x, or 0 where JSON has no number for it.
func finite(x float64) float64 {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return 0
	}
	return x
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "race:", err)
	os.Exit(1)
}
