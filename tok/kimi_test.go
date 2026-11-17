package tok

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/jitllm/jitllm/tok/pretok"
)

// TestKimiK2SplitsAsMoonshotDoes runs the kimi-k2 pre-tokenizer against a
// golden generated from Moonshot's own pattern.
//
// The authority is Moonshot's `pat_str`. A split that walks one
// `is_letter && !is_han` run is lossy in a way this golden pins:
//
//	input         Moonshot's own      one letter run
//	getUserName   [get User Name]     [getUserName]
//	iPhone        [i Phone]           [iPhone]
//	helloWorld    [hello World]       [helloWorld]
//
// where the pattern specifies o200k's two case-split alternatives (jitllm's
// CaseRuns).
//
// \p{Han} is the pinned 15.1.0 set on both sides (the golden's generator reads
// tok/unicodedata.go). The corpus includes U+2E80, U+3005, U+F900 and U+2EBF0,
// which separate the pinned set from newer Unicode and from a CJK-blocks-only
// Han.
func TestKimiK2SplitsAsMoonshotDoes(t *testing.T) {
	ops, ok := pretok.PreOps("kimi-k2")
	if !ok {
		t.Fatal("pretok has no kimi-k2 entry -- Kimi-K2 and Moonlight cannot load")
	}
	pipe, err := buildPipeline(ops)
	if err != nil {
		t.Fatalf("kimi-k2 does not compose: %v", err)
	}

	b, err := os.ReadFile(filepath.Join("testdata", "kimi_k2_split.json"))
	if err != nil {
		t.Fatalf("golden: %v (RULE 11: regenerate it, do not skip)", err)
	}
	var golden []struct {
		Text   string   `json:"text"`
		Pieces []string `json:"pieces"`
	}
	if err := json.Unmarshal(b, &golden); err != nil {
		t.Fatal(err)
	}
	if len(golden) < 10 {
		t.Fatalf("golden has %d case(s) -- too few to prove anything", len(golden))
	}

	han, camel := 0, 0
	for _, c := range golden {
		got := pipe.apply(c.Text)
		if !reflect.DeepEqual(got, c.Pieces) {
			t.Errorf("%q\n  got  %q\n  want %q", c.Text, got, c.Pieces)
		}
		for _, r := range c.Text {
			if ucIsHan(r) {
				han++
				break
			}
		}
		if len(c.Pieces) > 1 && c.Text == "getUserName" {
			camel++
		}
	}
	// The corpus must contain both things this entry is about, or ordinary
	// English would pass with HanRuns off and CaseRuns wrong.
	if han < 4 {
		t.Errorf("only %d golden case(s) contain Han -- the flag under test is barely exercised", han)
	}
	if camel == 0 {
		t.Error("no camelCase case in the golden -- the case-split alternatives are untested")
	}
	t.Logf("%d case(s), %d with Han", len(golden), han)
}

// TestKimiK2NeedsHanRuns is the violation: the same pipeline with HanRuns off
// is gpt-4o's, and it must not reproduce the golden; otherwise a splitter that
// ignored the flag would pass.
func TestKimiK2NeedsHanRuns(t *testing.T) {
	ops, ok := pretok.PreOps("kimi-k2")
	if !ok {
		t.Fatal("no kimi-k2 entry")
	}
	off := make([]pretok.Op, len(ops))
	copy(off, ops)
	found := false
	for i := range off {
		if off[i].Kind == pretok.OpSplit && off[i].Split.HanRuns {
			off[i].Split.HanRuns = false
			found = true
		}
	}
	if !found {
		t.Fatal("kimi-k2's entry has no HanRuns stage -- it is gpt-4o's pipeline")
	}
	on, err := buildPipeline(ops)
	if err != nil {
		t.Fatal(err)
	}
	no, err := buildPipeline(off)
	if err != nil {
		t.Fatal(err)
	}
	// Each of these separates the two, and each is a different reason:
	// a Han/Latin boundary, the radical block a CJK-blocks-only Han omits, and
	// the iteration mark that is Lm rather than Lo.
	for _, text := range []string{"中文abc", "⺀一x", "々y", "日本語テスト"} {
		a, b := on.apply(text), no.apply(text)
		if reflect.DeepEqual(a, b) {
			t.Errorf("%q splits the same with HanRuns on and off (%q) -- the flag is not wired", text, a)
		} else {
			t.Logf("%-16q  HanRuns %q   off %q", text, a, b)
		}
	}
}
