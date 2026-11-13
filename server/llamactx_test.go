package server

import (
	"fmt"
	"path/filepath"
	"testing"
)

// TestLlamaContextEndIsALimit: /completion asked for more than its context
// holds stops at the context's end as llama-server does without a context
// shift: stop_type limit and truncated, the prompt and the reply filling the
// session; n_keep, n_discard and n_cache_reuse beside it change nothing, and
// a reply that stops short of the end is not truncated. On stories15M,
// through the decode loop's rows and the one-at-a-time generate.
func TestLlamaContextEndIsALimit(t *testing.T) {
	sm := structuredModels()[0]
	path := modelPath(t, sm.file)
	for _, rows := range []int{0, 1} {
		t.Run(fmt.Sprintf("max rows %d", rows), func(t *testing.T) {
			e := New(admitConfig(Config{ModelDir: filepath.Dir(path), Probe: oneCardProbe, Version: "test", NoOnDemand: true, MaxBatchRows: rows}))
			t.Cleanup(e.Close)
			if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: sm.id}); err != nil {
				t.Fatal(err)
			}
			c := serveEngine(t, e)
			type reply struct {
				StopType  string `json:"stop_type"`
				Truncated bool   `json:"truncated"`
				Predicted int    `json:"tokens_predicted"`
				Evaluated int    `json:"tokens_evaluated"`
				Settings  struct {
					NCtx int `json:"n_ctx"`
				} `json:"generation_settings"`
			}
			ask := func(extra string) reply {
				var out reply
				body := fmt.Sprintf(`{"model":%q,"prompt":%q,"seed":5,"ignore_eos":true%s}`, sm.id, sm.prompt, extra)
				if code := postJSON(t, c.url, "/completion", body, &out); code != 200 {
					t.Fatalf("%s: %d", body, code)
				}
				return out
			}
			full := ask(`,"n_predict":-1,"n_keep":4,"n_discard":8,"n_cache_reuse":16`)
			if full.StopType != "limit" || !full.Truncated || full.Predicted == 0 || full.Evaluated+full.Predicted != full.Settings.NCtx {
				t.Fatalf("a reply to the context's end: stop_type %q, truncated %v, %d evaluated and %d predicted of %d", full.StopType,
					full.Truncated, full.Evaluated, full.Predicted, full.Settings.NCtx)
			}
			if short := ask(`,"n_predict":4`); short.StopType != "limit" || short.Truncated || short.Predicted != 4 {
				t.Fatalf("a reply of 4: stop_type %q, truncated %v, %d predicted", short.StopType, short.Truncated, short.Predicted)
			}
		})
	}
}
