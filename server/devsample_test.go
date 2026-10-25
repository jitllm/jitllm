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
//   - forced on (LoadOptions.DeviceSample): a plain sampled generate selects
//     every token on the device, so the arm the logprobs and grammar arms
//     must stay off is shown to be taken;
//   - auto: a plain generate follows the measured choice -- inside the probe
//     its device turns (the first and last of each ABBA quartet), after it
//     every token where the verdict is the device and none where it is the
//     host (model.State.DeviceSampleChoice).
func TestTheRowIsReadBackWhenTheReplyReadsIt(t *testing.T) {
	for _, mode := range []model.DeviceSampleMode{model.DeviceSampleOn, model.DeviceSampleAuto} {
		t.Run(fmt.Sprintf("mode=%d", mode), func(t *testing.T) { rowComesHome(t, mode) })
	}
}

func rowComesHome(t *testing.T, mode model.DeviceSampleMode) {
	path := modelPath(t, instructModel)
	e := New(Config{ModelDir: filepath.Dir(path), Version: "test", MaxBatchRows: 1})
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
		dev, probes int64
		head        bool
	}
	read := func() (c count) {
		s.Inspect(func(st *model.State) {
			c = count{st.SampledOnDevice, st.SampleProbes, st.HeadOnDevice()}
		})
		return c
	}
	if !read().head {
		t.Skip("CARD TOO SMALL: the head is not on the device -- this gate proved nothing")
	}
	// Auto's plain arm runs past one whole probe (32 tokens), so the verdict
	// is exercised as well as the probe's turns.
	max := 12
	if mode == model.DeviceSampleAuto {
		max = 44
	}
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
	switch mode {
	case model.DeviceSampleOn:
		if dev != steps || c1.probes != c0.probes {
			t.Fatalf("forced on: %d of %d tokens selected on the device, %d probe tokens",
				dev, steps, c1.probes-c0.probes)
		}
	case model.DeviceSampleAuto:
		probes := c1.probes - c0.probes
		var want int64
		for i := range probes {
			if q := (c0.probes + i) % 4; q == 0 || q == 3 {
				want++
			}
		}
		var device bool
		s.Inspect(func(st *model.State) { _, device, _, _, _ = st.DeviceSampleChoice(&sm) })
		if device {
			want += steps - probes
		}
		if probes == 0 || dev != want {
			t.Fatalf("auto: %d probe tokens, %d of %d selected on the device; the probe's turns "+
				"and its verdict (device %v) say %d", probes, dev, steps, device, want)
		}
		t.Logf("auto: %d probe tokens, verdict device=%v, %d on the device", probes, device, dev)
	}

	tok, lps := gen(GenerateOptions{Logprobs: true})
	c2 := read()
	if c2.dev != c1.dev || c2.probes != c1.probes || lps != tok {
		t.Fatalf("logprobs: %d tokens selected on the device, %d probed, %d of %d tokens carry a logprob",
			c2.dev-c1.dev, c2.probes-c1.probes, lps, tok)
	}
	tok, _ = gen(GenerateOptions{Grammar: `root ::= "Paris" | "Lyon"` + "\n"})
	c3 := read()
	if c3.dev != c2.dev || c3.probes != c2.probes || tok == 0 {
		t.Fatalf("grammar: %d of %d tokens selected on the device, %d probed", c3.dev-c2.dev, tok, c3.probes-c2.probes)
	}
	t.Logf("plain: %d on the device; logprobs and grammar: none", dev)
}
