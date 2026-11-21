package server

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"
)

// TestPostSamplingProbsAreTheDraws: /completion with post_sampling_probs
// reports at each sampled token the distribution its draw drew from, on
// stories15M, through the decode loop's rows and through the one-at-a-time
// generate. With top_k 3 and n_probs 5 every survivor is listed, so the
// shares sum to 1, the drawn token is among them at its own share, and they
// come the likeliest first; a greedy draw lists its token alone at 1. The
// raw distribution, asked without it, lists five under their log names --
// what reporting it in place of the draw's would read as.
func TestPostSamplingProbsAreTheDraws(t *testing.T) {
	sm := structuredModels()[0]
	path := modelPath(t, sm.file)
	type alt struct {
		ID      int32    `json:"id"`
		Prob    *float64 `json:"prob"`
		Logprob *float64 `json:"logprob"`
	}
	type entry struct {
		alt
		Top    []alt `json:"top_probs"`
		TopLog []alt `json:"top_logprobs"`
	}
	for _, rows := range []int{0, 1} {
		t.Run(fmt.Sprintf("max rows %d", rows), func(t *testing.T) {
			e := New(admitConfig(Config{ModelDir: filepath.Dir(path), Probe: oneCardProbe, Version: "test", NoOnDemand: true, MaxBatchRows: rows}))
			t.Cleanup(e.Close)
			if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: sm.id}); err != nil {
				t.Fatal(err)
			}
			c := serveEngine(t, e)
			ask := func(extra string) []entry {
				var out struct {
					Probs []entry `json:"completion_probabilities"`
				}
				body := fmt.Sprintf(`{"model":%q,"prompt":%q,"n_predict":6,"seed":11,"n_probs":5%s}`, sm.id, sm.prompt, extra)
				if code := postJSON(t, c.url, "/completion", body, &out); code != 200 || len(out.Probs) == 0 {
					t.Fatalf("%s: %d, %d entries", body, code, len(out.Probs))
				}
				return out.Probs
			}
			for i, p := range ask(`,"temperature":0.9,"top_k":3,"post_sampling_probs":true`) {
				if p.Prob == nil || p.Logprob != nil || p.TopLog != nil || len(p.Top) == 0 || len(p.Top) > 3 {
					t.Fatalf("token %d: prob %v, logprob %v, %d top_probs, %d top_logprobs", i, p.Prob, p.Logprob, len(p.Top), len(p.TopLog))
				}
				sum, own := 0.0, -1.0
				for j, a := range p.Top {
					if a.Prob == nil || *a.Prob <= 0 || j > 0 && *a.Prob > *p.Top[j-1].Prob {
						t.Fatalf("token %d: alternative %d is %+v after %+v", i, j, a, p.Top[max(j-1, 0)])
					}
					sum += *a.Prob
					if a.ID == p.ID {
						own = *a.Prob
					}
				}
				if math.Abs(sum-1) > 1e-4 || math.Abs(own-*p.Prob) > 1e-6 {
					t.Fatalf("token %d (%d at %.6f): its survivors sum to %.6f, it is listed at %.6f", i, p.ID, *p.Prob, sum, own)
				}
			}
			for i, p := range ask(`,"temperature":0,"post_sampling_probs":true`) {
				if p.Prob == nil || *p.Prob != 1 || len(p.Top) != 1 || p.Top[0].ID != p.ID || p.Top[0].Prob == nil || *p.Top[0].Prob != 1 {
					t.Fatalf("greedy token %d: %+v, top %+v", i, p.alt, p.Top)
				}
			}
			for i, p := range ask(`,"temperature":0.9,"top_k":3`) {
				if p.Logprob == nil || p.Prob != nil || len(p.TopLog) != 5 || p.Top != nil {
					t.Fatalf("raw token %d: logprob %v, prob %v, %d top_logprobs, %d top_probs", i, p.Logprob, p.Prob, len(p.TopLog), len(p.Top))
				}
			}
		})
	}
}
