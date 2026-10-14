package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// d1OmniVoiceGold is the voice request's golden (scripts/d1omnigold.py): the
// questions' as every golden's, the clip's first four log-mel frames and the
// prefix rows' first 16 values.
type d1OmniVoiceGold struct {
	d1OmniGold
	MelFrames int         `json:"mel_frames"`
	Mel       [][]float64 `json:"mel"`
	Prefix    [][]float64 `json:"prefix_rows"`
}

func d1OmniVoice(t *testing.T) (*DecisionBody, []float32, d1OmniVoiceGold) {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "decision", "voice.json"))
	if err != nil {
		t.Fatal(err)
	}
	req, err := ParseDecisionRequest(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.Audio) != 1 || len(req.Images) != 0 {
		t.Fatalf("the voice request parsed to %d clips and %d pictures", len(req.Audio), len(req.Images))
	}
	clip, err := DecodeAudioDataURL(req.Audio[0])
	if err != nil {
		t.Fatal(err)
	}
	var g d1OmniVoiceGold
	b, err := os.ReadFile(filepath.Join("testdata", "decision", "d1-omni.voice.reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	return &req, clip, g
}

// TestD1OmniHearsAsItsReference holds d1-omni's audio path to its own code
// (audio.py, the F16-rounded weights): the log-mel features, the prefix rows
// out of the conformer and the adapter, the prompt written as the audio
// questions were trained, the head's logits and the uncalibrated answers.
func TestD1OmniHearsAsItsReference(t *testing.T) {
	m := openD1Omni(t)
	defer m.Close()
	if m.audio == nil {
		t.Fatal("the container carries no audio tower, so this gate proves nothing")
	}
	d, err := m.NewDecider(0)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	req, clip, g := d1OmniVoice(t)
	var melW float64
	d.emb.audioTap = func(stage string, rows []float32) {
		if stage != "mel" {
			return
		}
		nm := int(m.audio.cfg.MelBins)
		for f, frame := range g.Mel {
			for i, want := range frame {
				melW = max(melW, math.Abs(float64(rows[f*nm+i])-want))
			}
		}
	}
	rows, err := d.emb.encodeAudio(d.d1OmniSamples(clip))
	if err != nil {
		t.Fatal(err)
	}
	d.emb.audioTap = nil
	pre := append([]float32(nil), rows...)
	nd := m.Cfg.NEmbd
	if len(pre)/nd != len(g.Prefix) {
		t.Fatalf("the clip encodes to %d prefix rows, the reference's %d", len(pre)/nd, len(g.Prefix))
	}
	var preN, preD float64
	for r, row := range g.Prefix {
		for i, want := range row {
			got := float64(pre[r*nd+i])
			preD += (got - want) * (got - want)
			preN += want * want
		}
	}
	t.Logf("log-mel worst %.3g; prefix NMSE %.3e over %d rows", melW, preD/preN, len(g.Prefix))
	if melW > 2e-3 || preD/preN > 1e-2 {
		t.Errorf("past the bound (log-mel 2e-3, prefix NMSE 1e-2)")
	}
	d.omniPre, d.omniSpoken = pre, true
	defer func() { d.omniPre, d.omniSpoken = d.omniPre[:0], false }()
	worstL, worstP := 0.0, 0.0
	for _, gq := range g.Questions {
		var q *DecisionQuestion
		for i := range req.Questions {
			if req.Questions[i].ID == gq.ID {
				q = &req.Questions[i]
			}
		}
		if q == nil {
			t.Fatalf("the request has no question %q", gq.ID)
		}
		ids, markers, err := d.d1OmniSequence(req.State, q, d.d1OmniMaxLen())
		if err != nil {
			t.Fatal(err)
		}
		if !equalIDs(ids, gq.IDs) || !equalInts(markers, gq.Markers) {
			t.Fatalf("%s: the sequence differs from the reference's\n got %v %v\nwant %v %v", q.ID, ids, markers, gq.IDs, gq.Markers)
		}
		s, err := d.d1OmniForward(ids, q.Type, markers, 0)
		if err != nil {
			t.Fatal(err)
		}
		for j := range s {
			worstL = max(worstL, math.Abs(float64(s[j])-gq.Logits[j]))
		}
		a, err := d.answer(q, [][]float32{s})
		if err != nil {
			t.Fatal(err)
		}
		probs := a.Probs
		if q.Type == jlm.QuestionNoul {
			probs = []float64{a.Noul, 1 - a.Noul}
		}
		for j := range probs {
			worstP = max(worstP, math.Abs(probs[j]-gq.Probs[j]))
		}
	}
	t.Logf("worst difference from the reference: logit %.3g, probability %.3g", worstL, worstP)
	if worstL > 0.4 || worstP > 0.02 {
		t.Errorf("past the bound (logit 0.4, probability 0.02)")
	}
	// The whole path, as a request runs it.
	ans, err := d.DecideAudio(req.State, req.Questions, clip)
	if err != nil {
		t.Fatal(err)
	}
	if d.InputTokens() != g.Usage.InputTokens {
		t.Errorf("read %d positions, the reference %d", d.InputTokens(), g.Usage.InputTokens)
	}
	if len(ans) != len(req.Questions) {
		t.Fatalf("%d answers to %d questions", len(ans), len(req.Questions))
	}
}

// TestD1OmniAudioFeaturesAreLoadBearing removes what is particular to the
// audio path and requires the answers to move past five times their clean
// distance from the reference: the attention's relative-position term, the
// convolution module, and the questions written as the audio questions were
// trained. A warm clip, encoded and answered again, makes no allocation.
func TestD1OmniAudioFeaturesAreLoadBearing(t *testing.T) {
	m := openD1Omni(t)
	defer m.Close()
	d, err := m.NewDecider(0)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	req, clip, g := d1OmniVoice(t)
	worst := func() float64 {
		ans, err := d.DecideAudio(req.State, req.Questions, clip)
		if err != nil {
			t.Fatal(err)
		}
		w := 0.0
		for i, gq := range g.Questions {
			a := ans[i]
			probs := a.Probs
			if req.Questions[i].Type == jlm.QuestionNoul {
				probs = []float64{a.Noul, 1 - a.Noul}
			}
			if req.Questions[i].ID != gq.ID {
				t.Fatalf("question %d is %q, the reference's %q", i, req.Questions[i].ID, gq.ID)
			}
			for j := range probs {
				w = max(w, math.Abs(probs[j]-gq.Probs[j]))
			}
		}
		return w
	}
	clean := worst()
	t.Logf("clean: worst probability %.4f off the reference", clean)
	for _, v := range []string{"no-relpos", "no-conv-module"} {
		d.emb.violation = v
		w := worst()
		d.emb.violation = ""
		t.Logf("%s: %.4f", v, w)
		if !(w > 5*clean && w > 0.05) {
			t.Errorf("%s moved the answers %.4f, under five times the clean %.4f or 0.05", v, w, clean)
		}
	}
	d.violation = "unspoken"
	w := worst()
	d.violation = ""
	t.Logf("unspoken: %.4f", w)
	if !(w > 5*clean && w > 0.05) {
		t.Errorf("writing the questions as text moved the answers %.4f, under five times the clean %.4f or 0.05", w, clean)
	}
	worst()
	if n := testing.AllocsPerRun(2, func() {
		if _, err := d.emb.encodeAudio(d.d1OmniSamples(clip)); err != nil {
			t.Fatal(err)
		}
	}); n != 0 {
		t.Errorf("a warm clip's encoding made %v allocations", n)
	}
}

// TestD1OmniVoiceMatchesLlamaCpp answers the voice request as llama.cpp's
// d1-omni server does (scripts/decisiongold.sh): the same positions read and
// the same answers within the tower's Q8_0 and int8 band (llama.cpp reads the
// mmproj's F16 and F32).
func TestD1OmniVoiceMatchesLlamaCpp(t *testing.T) {
	m := openD1Omni(t)
	defer m.Close()
	d, err := m.NewDecider(0)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	req, clip, _ := d1OmniVoice(t)
	gold := readDecisionGold(t, "d1-omni-600M-F16.voice.llamacpp.json")
	ans, err := d.DecideAudio(req.State, req.Questions, clip)
	if err != nil {
		t.Fatal(err)
	}
	if d.InputTokens() != gold.Usage.InputTokens {
		t.Errorf("read %d positions, llama.cpp %d: the prefix or the prompts differ", d.InputTokens(), gold.Usage.InputTokens)
	}
	checkDecisions(t, req.Questions, ans, gold, 0.03, false)
}
