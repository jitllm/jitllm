package server

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// TestPredictTimeEndsAtTheNextNewline: a reply past its prediction time
// (llama-server's t_max_predict_ms) ends at its first token whose text holds a
// newline, as FinishMaxTokens, alone and as a row of the step loop; the
// tokens up to it are the reply's without the limit, and a limit not reached
// changes nothing.
func TestPredictTimeEndsAtTheNextNewline(t *testing.T) {
	path := modelPath(t, smallModel)
	for _, rows := range []int{1, 0} {
		e := New(admitConfig(Config{ModelDir: filepath.Dir(path), Probe: oneCardProbe, Version: "test",
			NoOnDemand: true, MaxBatchRows: rows}))
		t.Cleanup(e.Close)
		if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "small"}); err != nil {
			t.Fatalf("LoadModel(%s): %v", path, err)
		}
		reply := func(limit time.Duration) ([]int32, []string, *Started, *Finished) {
			var ids []int32
			var texts []string
			var st *Started
			var fin *Finished
			err := e.Generate(context.Background(), GenerateOptions{ModelID: "small",
				Prompt: Prompt{Kind: PromptText, Text: story}, MaxTokens: 200, PredictTime: limit}, func(ev Event) error {
				switch ev.Kind {
				case EventStarted:
					st = ev.Started
				case EventToken:
					if ev.Token.ID >= 0 {
						ids = append(ids, ev.Token.ID)
						texts = append(texts, ev.Token.Text)
					}
				case EventFinished:
					fin = ev.Finished
				}
				return nil
			})
			if err != nil || st == nil || fin == nil {
				t.Fatalf("rows %d, limit %v: %v (started %v, finished %v)", rows, limit, err, st != nil, fin != nil)
			}
			return ids, texts, st, fin
		}
		ref, texts, st, _ := reply(0)
		if st.Batched != (rows != 1) {
			t.Fatalf("rows %d: the generate ran batched=%v", rows, st.Batched)
		}
		nl := slices.IndexFunc(texts, func(s string) bool { return strings.Contains(s, "\n") })
		if nl < 0 || nl == len(ref)-1 {
			t.Fatalf("rows %d: the reference reply has no newline before its last token to end at: %q", rows, strings.Join(texts, ""))
		}
		cut, _, _, fin := reply(time.Nanosecond)
		if !slices.Equal(cut, ref[:nl+1]) || fin.Reason != FinishMaxTokens || fin.CompletionTokens != nl+1 {
			t.Fatalf("rows %d: past its time the reply gave %d tokens (%v, %d counted), want the %d up to its first newline",
				rows, len(cut), fin.Reason, fin.CompletionTokens, nl+1)
		}
		long, _, _, _ := reply(time.Hour)
		if !slices.Equal(long, ref) {
			t.Fatalf("rows %d: a limit not reached changed the reply (%d tokens, want %d)", rows, len(long), len(ref))
		}
		t.Logf("rows %d: %d tokens, the first newline at %d", rows, len(ref), nl)
	}
}
