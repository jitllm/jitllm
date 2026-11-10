package server

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// The end-to-end gate: a real container, a real prefill, real tokens.
//
// It is opt-in because it occupies the decode cores and reads the whole
// model, which would disturb any measurement beside it. It runs only when
// JITLLM_SERVER_MODEL names a container (a path, or a bare name in
// JITLLM_MODELS):
//
//	JITLLM_SERVER_MODEL=<something>.jlm \
//	  ./scripts/cap 24G -- taskset -c 0,2,4,6,8,10 \
//	  go test ./server/ -run TestEndToEndAgainstARealContainer -count=1 -v
//
// Unset, it skips with that command in the message. Set, it never skips: a
// missing file, a GGUF or an unopenable container is a failure.
//
// It covers what the fake-backed gates cannot: model.Open's refusal on a real
// file, prefill and Forward, DecodeChat, the device offer, the residency
// counters, and two real sessions taking turns.
func TestEndToEndAgainstARealContainer(t *testing.T) {
	path := testmodels.Resolve(os.Getenv("JITLLM_SERVER_MODEL"))
	if path == "" {
		t.Skip("JITLLM_SERVER_MODEL is unset, so the end-to-end gate did not run. " +
			"Set it to a .jlm container (a bare name is looked up in JITLLM_MODELS) and re-run:\n" +
			"  JITLLM_SERVER_MODEL=<model>.jlm \\\n" +
			"    ./scripts/cap 24G -- taskset -c 0,2,4,6,8,10 go test ./server/ -run TestEndToEnd -count=1 -v")
	}
	// Past this point nothing skips.
	if fi, err := os.Stat(path); err != nil {
		t.Fatalf("JITLLM_SERVER_MODEL=%q: %v", path, err)
	} else if fi.IsDir() {
		t.Fatalf("JITLLM_SERVER_MODEL=%q is a directory", path)
	}
	if !strings.EqualFold(filepath.Ext(path), model.Ext) {
		t.Fatalf("JITLLM_SERVER_MODEL=%q is not a %s container; the engine reads one weight "+
			"format and a GGUF must be converted first", path, model.Ext)
	}

	e := New(admitConfig(Config{ModelDir: filepath.Dir(path), Probe: noProbe}))
	defer e.Close()

	lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "e2e"})
	if err != nil {
		t.Fatalf("LoadModel(%q): %v", path, err)
	}
	t.Logf("loaded %s: blocks=%d n_embd=%d n_ctx=%d vocab=%d chat=%v",
		lm.Name(), lm.m.Cfg.NLayer, lm.m.Cfg.NEmbd, lm.m.Cfg.NCtx,
		lm.m.Cfg.NVocab, lm.m.ChatCapable())

	// ---- 1. the ConnectRPC path, streaming.
	t.Run("generate streams real tokens", func(t *testing.T) {
		var events []string
		var text strings.Builder
		var started *Started
		var fin *Finished
		err := e.Generate(context.Background(), GenerateOptions{
			ModelID:   "e2e",
			Prompt:    Prompt{Kind: PromptText, Text: "The capital of France is"},
			MaxTokens: 8,
		}, func(ev Event) error {
			switch ev.Kind {
			case EventStarted:
				events = append(events, "started")
				started = ev.Started
			case EventToken:
				events = append(events, "token")
				text.WriteString(ev.Token.Text)
			case EventFinished:
				events = append(events, "finished")
				fin = ev.Finished
			}
			return nil
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if len(events) < 3 || events[0] != "started" || events[len(events)-1] != "finished" {
			t.Fatalf("event sequence %v does not start with started and end with finished", events)
		}
		if strings.TrimSpace(text.String()) == "" {
			t.Fatalf("the model produced no text over %d events", len(events))
		}
		if started.PromptTokens == 0 {
			t.Fatal("the prompt tokenized to nothing")
		}
		if fin.CompletionTokens == 0 {
			t.Fatal("no completion tokens were produced")
		}
		// A rate is quoted with the bytes behind it.
		t.Logf("%q  prompt=%d completion=%d prefill=%s decode=%s %.2f tok/s, %d B/token",
			text.String(), fin.PromptTokens, fin.CompletionTokens,
			fin.Prefill.Round(time.Millisecond), fin.Decode.Round(time.Millisecond),
			fin.TokensPerSecond, fin.BytesPerToken)
		if fin.BytesPerToken == 0 {
			t.Fatal("bytes_per_token is 0 on a real model: the rate is then unfalsifiable")
		}
	})

	// ---- 2. a stop string, end to end.
	t.Run("a stop string truncates and is never echoed", func(t *testing.T) {
		var text strings.Builder
		var fin *Finished
		err := e.Generate(context.Background(), GenerateOptions{
			ModelID:   "e2e",
			Prompt:    Prompt{Kind: PromptText, Text: "One two three four five six seven"},
			MaxTokens: 24,
			Stop:      []string{" "},
		}, func(ev Event) error {
			if ev.Kind == EventToken {
				text.WriteString(ev.Token.Text)
			}
			if ev.Kind == EventFinished {
				fin = ev.Finished
			}
			return nil
		})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if fin.Reason != FinishStop {
			t.Logf("no stop match (reason %v) -- the model did not emit the stop string in 24 tokens", fin.Reason)
		} else if strings.Contains(text.String(), fin.StopMatched) {
			t.Fatalf("the completion %q contains the stop string %q, which is never part of it",
				text.String(), fin.StopMatched)
		}
	})

	// ---- 3. the two compat shims over the same engine.
	t.Run("the compat shims stream the same engine", func(t *testing.T) {
		if !lm.m.ChatCapable() {
			t.Logf("%s carries no chat template, so the chat shims are exercised through "+
				"/v1/completions only", lm.Name())
		}
		srv := httptest.NewServer(CompatHandler(e))
		defer srv.Close()

		resp := post(t, srv, "/v1/completions",
			`{"model":"e2e","prompt":"The capital of France is","max_tokens":8}`)
		defer resp.Body.Close()
		var out struct {
			Choices []struct {
				Text string `json:"text"`
			} `json:"choices"`
			Jitllm *oaJitllmExtra `json:"jitllm"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			t.Fatalf("decoding /v1/completions: %v", err)
		}
		if len(out.Choices) != 1 || strings.TrimSpace(out.Choices[0].Text) == "" {
			t.Fatalf("/v1/completions produced no text: %+v", out.Choices)
		}
		if out.Jitllm == nil || out.Jitllm.SessionID == "" {
			t.Fatal("the response carries no jitllm extension, so a caller cannot see where it ran")
		}
		t.Logf("/v1/completions -> %q (session %s, %d device blocks)",
			out.Choices[0].Text, out.Jitllm.SessionID, out.Jitllm.DeviceBlocks)

		if lm.m.ChatCapable() {
			r2 := post(t, srv, "/v1/messages",
				`{"model":"e2e","max_tokens":16,"system":"Be terse.","messages":[{"role":"user","content":"What is the capital of France?"}]}`)
			defer r2.Body.Close()
			var an anResponse
			if err := json.NewDecoder(r2.Body).Decode(&an); err != nil {
				t.Fatalf("decoding /v1/messages: %v", err)
			}
			if len(an.Content) != 1 || an.Content[0].Text == nil || strings.TrimSpace(*an.Content[0].Text) == "" {
				t.Fatalf("/v1/messages produced no text: %+v", an)
			}
			t.Logf("/v1/messages -> %q", *an.Content[0].Text)
		}
	})

	// ---- 4. the serialisation, with real work in it.
	//
	// Two real sessions on the host must take turns, and the second must be
	// told it waited.
	t.Run("two sessions take turns and the second is told", func(t *testing.T) {
		a, err := e.CreateSession(SessionOptions{ModelID: "e2e"})
		if err != nil {
			t.Fatal(err)
		}
		defer e.CloseSession(a.ID())
		b, err := e.CreateSession(SessionOptions{ModelID: "e2e"})
		if err != nil {
			t.Fatal(err)
		}
		defer e.CloseSession(b.ID())

		var wg sync.WaitGroup
		waits := make([]time.Duration, 2)
		for i, s := range []*Session{a, b} {
			wg.Add(1)
			go func(i int, s *Session) {
				defer wg.Done()
				e.Generate(context.Background(), GenerateOptions{
					SessionID: s.ID(),
					Prompt:    Prompt{Kind: PromptText, Text: "Count: one two three"},
					MaxTokens: 12,
				}, func(ev Event) error {
					if ev.Kind == EventStarted {
						waits[i] = ev.Started.QueuedFor
					}
					return nil
				})
			}(i, s)
		}
		wg.Wait()
		t.Logf("queued for: %v and %v", waits[0], waits[1])
		if waits[0] == 0 && waits[1] == 0 {
			t.Fatal("neither session reported a queue wait, yet the host gate is width 1: " +
				"either they did not overlap, or queued_millis is not being measured")
		}
	})

	// ---- 5. residency, which needs a real container.
	t.Run("residency reports whole blocks", func(t *testing.T) {
		r := pbResidency(lm)
		t.Logf("residency: %d/%d blocks resident, frames=%d, budget=%s, fits=%v, "+
			"page-ins=%d evictions=%d bytes read=%s",
			r.GetResidentBlocks(), r.GetBlocks(), r.GetFrames(),
			r.GetBudget().GetHuman(), r.GetFits(), r.GetPageIns(), r.GetEvictions(),
			r.GetBytesRead().GetHuman())
		if r.GetBlocks() == 0 {
			t.Fatal("residency reports 0 blocks for a loaded container")
		}
		if r.GetFits() && r.GetEvictions() > 0 {
			t.Fatalf("residency says the budget fits and yet %d page(s) were evicted; "+
				"a budget that holds every block never evicts", r.GetEvictions())
		}
	})
}
