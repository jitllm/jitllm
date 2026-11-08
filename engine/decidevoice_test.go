package engine

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestD1OmniDecidesOverItsVoice asks d1-omni the voice request's questions
// about its clip on the CPU device: each question's prompt after the clip's
// rows is the reference's token for token (the options as the audio
// questions were trained, jlm.PromptClips), and the answers are the
// reference's (scripts/d1omnigold.py, its own code) within 0.03, the int8
// blocks' band, and llama.cpp's within the same. The same questions with no
// clip are the violation: their answers must part from the clip's.
func TestD1OmniDecidesOverItsVoice(t *testing.T) {
	m := d1OmniOpen(t)
	defer m.Close()
	in, err := NewInstance(m, 4096, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	d, err := NewDecider(in)
	if err != nil {
		t.Fatal(err)
	}
	clip, _ := voiceClip(t)
	sp, err := m.ClipSpan(clip)
	if err != nil {
		t.Fatal(err)
	}
	state, qs := decisionRequest(t, "voice")
	b, err := os.ReadFile(filepath.Join("testdata", "decision", "d1-omni.voice.reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var gold struct {
		Questions []struct {
			ID      string  `json:"id"`
			IDs     []int32 `json:"ids"`
			Markers []int   `json:"markers"`
			Prefix  int     `json:"prefix"`
		} `json:"questions"`
	}
	if err := json.Unmarshal(b, &gold); err != nil {
		t.Fatal(err)
	}
	if len(gold.Questions) != len(qs) {
		t.Fatalf("%d questions, the reference's %d", len(qs), len(gold.Questions))
	}
	for i, g := range gold.Questions {
		if sp.ClipRows != g.Prefix {
			t.Fatalf("%s: the clip takes %d rows, the reference's %d", g.ID, sp.ClipRows, g.Prefix)
		}
		ids, marks, err := markedPrompt(d.mp, d.encodePlain, state, &qs[i], in.MaxSeq()-sp.ClipRows, 0, 1)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(ids, g.IDs) || !slices.Equal(marks, g.Markers) {
			t.Fatalf("%s: ids %v markers %v, the reference's %v %v", g.ID, ids, marks, g.IDs, g.Markers)
		}
	}
	ans, err := d.Decide(state, qs, sp)
	if err != nil {
		t.Fatal(err)
	}
	worst := checkDecisions(t, qs, ans, readDecisionGold(t, "d1-omni.voice.reference.json"), 0.03)
	// llama.cpp writes TypeSafe's confidence where d1-omni's code writes the
	// largest probability (docs/divergences.md): its probabilities, nouls
	// and scores are held, its confidences replaced by the reference's.
	ref, lc := readDecisionGold(t, "d1-omni.voice.reference.json"), readDecisionGold(t, "d1-omni-600M-F16.voice.llamacpp.json")
	for id, a := range lc.Answers {
		a.Confidence = ref.Answers[id].Confidence
		lc.Answers[id] = a
	}
	worstL := checkDecisions(t, qs, ans, lc, 0.03)
	t.Logf("a clip of %d rows: worst %.5f from the reference, %.5f from llama.cpp, %d prompt rows", sp.ClipRows, worst, worstL, d.InputTokens())
	deaf, err := d.Decide(state, qs, []model.Span{}...)
	if err != nil {
		t.Fatal(err)
	}
	if !decisionsDiffer(qs, ans, deaf) {
		t.Fatal("VIOLATION arm: the answers without the clip are the answers with it")
	}
}
