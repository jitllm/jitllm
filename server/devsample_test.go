package server

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/engine/model"
)

// TestTheRowIsReadBackWhenTheReplyReadsIt: with the head on the device, a reply
// that asks for logprobs or runs under a grammar reads every row back and
// samples on the host -- those read the raw logits, which a device-selected
// token never brings home -- whatever the device sampler's mode. Asserted on
// the session's own count of device-selected tokens, under two modes:
//
//   - forced on (LoadOptions.DeviceSample) and auto: a plain sampled generate
//     draws every token on the device, so the arm the logprobs and grammar
//     arms must stay off is shown to be taken.
func TestTheRowIsReadBackWhenTheReplyReadsIt(t *testing.T) {
	for _, mode := range []model.DeviceSampleMode{model.DeviceSampleOn, model.DeviceSampleAuto} {
		t.Run(fmt.Sprintf("mode=%d", mode), func(t *testing.T) { rowComesHome(t, mode) })
	}
}

func rowComesHome(t *testing.T, mode model.DeviceSampleMode) {
	path := modelPath(t, instructModel)
	e := New(admitConfig(Config{ModelDir: filepath.Dir(path), Version: "test", MaxBatchRows: 1}))
	defer e.Close()
	if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "inst", DeviceIDs: []string{"gpu:0"},
		Sessions: 1, DeviceSample: mode}); err != nil {
		t.Skipf("NO DEVICE: %v -- this gate proved nothing", err)
	}
	sm := model.Sampler{Temp: 0.9, TopK: 40, TopP: 0.9, Seed: 3}
	s, err := e.CreateSession(SessionOptions{ModelID: "inst", SessionID: "a", MaxSeq: 256, Sampling: sm})
	if err != nil {
		t.Fatal(err)
	}
	type count struct {
		dev  int64
		head bool
	}
	read := func() (c count) {
		s.Inspect(func(st *model.Instance) {
			c = count{st.SampledOnDevice, st.HeadOnDevice()}
		})
		return c
	}
	if !read().head {
		t.Skip("CARD TOO SMALL: the head is not on the device -- this gate proved nothing")
	}
	max := 12
	gen := func(o GenerateOptions) (tokens, logprobs int) {
		t.Helper()
		if o.MaxTokens == 0 {
			o.MaxTokens = 12
		}
		o.SessionID, o.Prompt, o.IgnoreEOS = "a", Prompt{Kind: PromptText, Text: "The capital of France is"}, o.Grammar == ""
		err := e.Generate(context.Background(), o, func(ev Event) error {
			if ev.Kind == EventToken && ev.Token.ID >= 0 {
				tokens++
				if ev.Token.Logprob != nil {
					logprobs++
				}
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return tokens, logprobs
	}

	c0 := read()
	tok, _ := gen(GenerateOptions{MaxTokens: max})
	c1 := read()
	if tok != max {
		t.Fatalf("a plain generate made %d tokens, want %d", tok, max)
	}
	// Every token is followed by one ForwardSample, the last included.
	steps, dev := int64(tok), c1.dev-c0.dev
	if dev != steps {
		t.Fatalf("mode %d: %d of %d tokens drawn on the device", mode, dev, steps)
	}

	tok, lps := gen(GenerateOptions{Logprobs: true})
	c2 := read()
	if c2.dev != c1.dev || lps != tok {
		t.Fatalf("logprobs: %d tokens drawn on the device, %d of %d tokens carry a logprob", c2.dev-c1.dev, lps, tok)
	}
	tok, _ = gen(GenerateOptions{Grammar: `root ::= "Paris" | "Lyon"` + "\n"})
	c3 := read()
	if c3.dev != c2.dev || tok == 0 {
		t.Fatalf("grammar: %d of %d tokens drawn on the device", c3.dev-c2.dev, tok)
	}
	t.Logf("plain: %d on the device; logprobs and grammar: none", dev)
}
