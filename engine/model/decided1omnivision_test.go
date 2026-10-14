package model

import (
	"image"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

const d1OmniProj = "d1-omni/mmproj-d1-omni-600M-F16.gguf"

// openD1Omni is d1-omni converted with its mmproj, so it carries the tower.
func openD1Omni(t *testing.T) *Model {
	t.Helper()
	text, proj := testmodels.Path(d1OmniModel), testmodels.Path(d1OmniProj)
	if !vlmAvailable(text, proj) {
		testmodels.Missing(t, "MODEL MISSING: %s and its mmproj (RULE 11)", d1OmniModel)
	}
	m, err := Open(jlmOfPair(t, text, proj), noTune)
	if err != nil {
		t.Fatal(err)
	}
	if m.Tower() == nil || m.Decision() != jlm.DecisionD1Omni {
		m.Close()
		t.Fatalf("the container carries tower %v and readout %v", m.Tower() != nil, m.Decision())
	}
	return m
}

// d1OmniPictures is testdata/decision/pictures.json: its state, questions
// and decoded pictures.
func d1OmniPictures(t *testing.T) (*DecisionBody, []image.Image) {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "decision", "pictures.json"))
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseDecisionRequest(body)
	if err != nil {
		t.Fatal(err)
	}
	imgs := make([]image.Image, len(parsed.Images))
	for i, u := range parsed.Images {
		if imgs[i], err = DecodeImageDataURL(u); err != nil {
			t.Fatal(err)
		}
	}
	if len(imgs) < 2 {
		t.Fatal("the request carries fewer than two pictures, so the swap proves nothing")
	}
	return &parsed, imgs
}

// d1OmniPicturesWorst runs the pictures request's questions over imgs and
// returns the worst logit and probability difference from the golden.
func d1OmniPicturesWorst(t *testing.T, d *Decider, req *DecisionBody, imgs []image.Image) (worstL, worstP float64) {
	t.Helper()
	g := readD1OmniGold(t, "pictures")
	if err := d.d1OmniPrefix(imgs); err != nil {
		t.Fatal(err)
	}
	defer func() { d.omniPre = d.omniPre[:0] }()
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
		if p := len(d.omniPre) / d.m.Cfg.NEmbd; p != gq.Prefix {
			t.Fatalf("a prefix of %d rows, the reference's %d", p, gq.Prefix)
		}
		ids, markers, err := d.d1OmniSequence(req.State, q, d.d1OmniMaxLen())
		if err != nil {
			t.Fatal(err)
		}
		if !equalIDs(ids, gq.IDs) || !equalInts(markers, gq.Markers) {
			t.Fatalf("%s: the sequence differs from the reference's\n got %v %v\nwant %v %v",
				q.ID, ids, markers, gq.IDs, gq.Markers)
		}
		s, err := d.d1OmniForward(ids, q.Type, markers, 0)
		if err != nil {
			t.Fatal(err)
		}
		for j := range s {
			if math.IsNaN(float64(s[j])) || math.IsInf(float64(s[j]), 0) {
				t.Fatalf("%s: score %d is %v", q.ID, j, s[j])
			}
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
	return worstL, worstP
}

// TestD1OmniDecidesOverPicturesAsItsReference holds d1-omni's picture path
// to its own code: the prefix's length (every tile and thumbnail of both
// pictures through the tower and projector), the prompt after it, the head's
// logits over the text rows and the uncalibrated probabilities. Two
// violations show the gate sees what is particular to the prefix: the
// prefix's rows reading the text, and the pictures swapped.
func TestD1OmniDecidesOverPicturesAsItsReference(t *testing.T) {
	m := openD1Omni(t)
	defer m.Close()
	d, err := m.NewDecider(0)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	req, imgs := d1OmniPictures(t)
	l, p := d1OmniPicturesWorst(t, d, req, imgs)
	t.Logf("worst difference from the reference: logit %.2e, probability %.2e", l, p)
	// The converter stores the tower Q8_0 and it runs on int8 activations
	// (convert/vision.go, the one lossy step), against the reference's F16
	// weights: the prefix rows sit at NMSE 1.6e-3 (an outlier channel near
	// -110 is 26 off), the logits 0.21 and the probabilities 5.2e-3 away on
	// amd64, 0.32 and 8.1e-3 on the M4.
	const boundL, boundP = 0.4, 0.02
	if l > boundL || p > boundP {
		t.Errorf("past the bound (logit %g, probability %g)", boundL, boundP)
	}
	d.emb.violation = "prefix-reads-text"
	vl, _ := d1OmniPicturesWorst(t, d, req, imgs)
	d.emb.violation = ""
	t.Logf("violation, the prefix reads the text: worst logit %.3g", vl)
	if !(vl > 2*l && vl > boundL) {
		t.Errorf("the prefix reading the text moved no logit past twice the clean %.3g and the bound", l)
	}
	sl, _ := d1OmniPicturesWorst(t, d, req, []image.Image{imgs[1], imgs[0]})
	t.Logf("violation, the pictures swapped: worst logit %.3g", sl)
	if !(sl > 2*l && sl > boundL) {
		t.Errorf("swapping the pictures moved no logit past twice the clean %.3g and the bound", l)
	}
}

// TestD1OmniPicturesMatchLlamaCpp answers the pictures request as llama.cpp's
// d1-omni server does (scripts/decisiongold.sh): the same prefix rows, the
// same prompt and the same uncalibrated answers, within the towers' Q8_0
// and int8 band (llama.cpp reads the mmproj's F16).
func TestD1OmniPicturesMatchLlamaCpp(t *testing.T) {
	m := openD1Omni(t)
	defer m.Close()
	d, err := m.NewDecider(0)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	req, imgs := d1OmniPictures(t)
	gold := readDecisionGold(t, "d1-omni-600M-F16.pictures.llamacpp.json")
	ans, err := d.Decide(req.State, req.Questions, imgs)
	if err != nil {
		t.Fatal(err)
	}
	if d.InputTokens() != gold.Usage.InputTokens {
		t.Errorf("ran %d input tokens, llama.cpp %d: the prefix or the prompts differ", d.InputTokens(), gold.Usage.InputTokens)
	}
	checkDecisions(t, req.Questions, ans, gold, 0.02, false)
}
