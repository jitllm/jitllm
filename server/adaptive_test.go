package server

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"
)

// TestAdaptivePReachesTheDraw: /completion with adaptive_p in samplers draws
// through adaptive-p (engine.Sampler.Adaptive) on stories15M, through the
// decode loop's rows and the one-at-a-time generate: its first token's
// distribution after every sampler (post_sampling_probs) is a distribution
// over the chain's survivors, summing to 1, and not the plain chain's at the
// same prompt and seed -- a front end that dropped the field would report
// the same one.
func TestAdaptivePReachesTheDraw(t *testing.T) {
	sm := structuredModels()[0]
	path := modelPath(t, sm.file)
	type alt struct {
		ID   int32    `json:"id"`
		Prob *float64 `json:"prob"`
	}
	for _, rows := range []int{0, 1} {
		t.Run(fmt.Sprintf("max rows %d", rows), func(t *testing.T) {
			e := New(admitConfig(Config{ModelDir: filepath.Dir(path), Probe: oneCardProbe, Version: "test", NoOnDemand: true, MaxBatchRows: rows}))
			t.Cleanup(e.Close)
			if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: sm.id}); err != nil {
				t.Fatal(err)
			}
			c := serveEngine(t, e)
			first := func(samplers string) map[int32]float64 {
				var out struct {
					Probs []struct {
						Top []alt `json:"top_probs"`
					} `json:"completion_probabilities"`
				}
				body := fmt.Sprintf(`{"model":%q,"prompt":%q,"n_predict":1,"seed":3,"n_probs":5,"top_k":4,"temperature":1,"post_sampling_probs":true,"adaptive_target":0.1,"adaptive_decay":0.9,"samplers":%s}`,
					sm.id, sm.prompt, samplers)
				if code := postJSON(t, c.url, "/completion", body, &out); code != 200 || len(out.Probs) != 1 {
					t.Fatalf("%s: %d, %d entries", samplers, code, len(out.Probs))
				}
				d, sum := map[int32]float64{}, 0.0
				for _, a := range out.Probs[0].Top {
					d[a.ID] = *a.Prob
					sum += *a.Prob
				}
				if len(d) == 0 || len(d) > 4 || math.Abs(sum-1) > 1e-4 {
					t.Fatalf("%s: %d survivors summing to %.6f", samplers, len(d), sum)
				}
				return d
			}
			plain, adapt := first(`["top_k","temperature"]`), first(`["top_k","temperature","adaptive_p"]`)
			moved := 0.0
			for id, p := range plain {
				moved = max(moved, math.Abs(adapt[id]-p))
			}
			if moved < 1e-3 {
				t.Fatalf("adaptive-p left the distribution as the plain chain's: %v against %v", adapt, plain)
			}
		})
	}
}
