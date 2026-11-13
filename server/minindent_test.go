package server

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestMinIndentEndsAtAShortLine: a reply with a MinIndent (llama-server's
// n_indent) ends at the token where a line past its first newline starts with
// fewer spaces or tabs than that before its first other character, as
// FinishMaxTokens, alone and as a row of the step loop; the tokens up to it
// are the reply's without the rule.
func TestMinIndentEndsAtAShortLine(t *testing.T) {
	path := modelPath(t, smallModel)
	for _, rows := range []int{1, 0} {
		e := New(admitConfig(Config{ModelDir: filepath.Dir(path), Probe: oneCardProbe, Version: "test",
			NoOnDemand: true, MaxBatchRows: rows}))
		t.Cleanup(e.Close)
		if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "small"}); err != nil {
			t.Fatalf("LoadModel(%s): %v", path, err)
		}
		reply := func(indent int) ([]int32, []string, *Started, *Finished) {
			var ids []int32
			var texts []string
			var st *Started
			var fin *Finished
			err := e.Generate(context.Background(), GenerateOptions{ModelID: "small",
				Prompt: Prompt{Kind: PromptText, Text: story}, MaxTokens: 200, MinIndent: indent}, func(ev Event) error {
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
				t.Fatalf("rows %d, n_indent %d: %v (started %v, finished %v)", rows, indent, err, st != nil, fin != nil)
			}
			return ids, texts, st, fin
		}
		ref, texts, st, _ := reply(0)
		if st.Batched != (rows != 1) {
			t.Fatalf("rows %d: the generate ran batched=%v", rows, st.Batched)
		}
		// The first character past the first newline that is not a space, a
		// tab or another newline: the first a line past it starts with.
		all := strings.Join(texts, "")
		nl := strings.IndexByte(all, '\n')
		at := -1
		if nl >= 0 {
			if j := strings.IndexFunc(all[nl+1:], func(r rune) bool { return r != ' ' && r != '\t' && r != '\n' }); j >= 0 {
				at = nl + 1 + j
			}
		}
		if at < 0 {
			t.Fatalf("rows %d: the reference reply has no line past its first newline that starts with text: %q", rows, all)
		}
		lineAt := strings.LastIndexByte(all[:at], '\n')
		end, off := -1, 0
		for i, s := range texts {
			if off += len(s); off > at {
				end = i
				break
			}
		}
		if end < 0 || end == len(ref)-1 {
			t.Fatalf("rows %d: that line's first character is in no token before the last", rows)
		}
		// Its indentation is under the rule's by one.
		indent := at - lineAt - 1 + 1
		cut, _, _, fin := reply(indent)
		if !slices.Equal(cut, ref[:end+1]) || fin.Reason != FinishMaxTokens || fin.CompletionTokens != end+1 {
			t.Fatalf("rows %d: n_indent %d gave %d tokens (%v, %d counted), want the %d up to that line's first character",
				rows, indent, len(cut), fin.Reason, fin.CompletionTokens, end+1)
		}
		// One short of it is met by every line up to there.
		if ok, _, _, _ := reply(indent - 1); len(ok) <= end+1 {
			t.Fatalf("rows %d: n_indent %d, which that line meets, ended the reply at %d tokens", rows, indent-1, len(ok))
		}
		t.Logf("rows %d: %d tokens, the first line past a newline starts in token %d at indentation %d", rows, len(ref), end, indent-1)
	}
}
