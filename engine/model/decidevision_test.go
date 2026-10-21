package model

import (
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
	"github.com/jitllm/jitllm/tok/jinja"
)

// d1-3B's decisions over pictures, against the checkpoint's own SystemOne
// (runner.py: cap_pixels, the processor, the hybrid stack) in float32 on the
// pictures scripts/lfm2visiongold.py ran: with a text state and with none,
// a choice, a noul and a score per request.

type d1VisionCase struct {
	Image     string
	State     *string
	Questions json.RawMessage
	Response  struct {
		Answers map[string]struct {
			Type          string
			Noul          float64
			Probabilities map[string]float64
		}
	}
}

func readD1VisionCases(t *testing.T) []d1VisionCase {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(d1Gold, "decisions.json"))
	if err != nil {
		testmodels.Missing(t, "GOLDEN MISSING: %v -- run scripts/lfm2visiongold.py (RULE 11)", err)
	}
	var cs []d1VisionCase
	if err := json.Unmarshal(b, &cs); err != nil {
		t.Fatal(err)
	}
	if len(cs) == 0 {
		t.Fatal("the golden holds no case")
	}
	return cs
}

// d1VisionWorst runs every case on d and returns the worst probability
// difference against the reference.
func d1VisionWorst(t *testing.T, d *Decider, cs []d1VisionCase) float64 {
	t.Helper()
	worst := 0.0
	for _, c := range cs {
		qv, err := jinja.FromJSON(c.Questions)
		if err != nil {
			t.Fatal(err)
		}
		qs, err := ParseDecisionQuestions(qv)
		if err != nil {
			t.Fatal(err)
		}
		state := jinja.None()
		if c.State != nil {
			state = jinja.NewString(*c.State)
		}
		img := decodeImage(t, filepath.Join(d1Gold, c.Image))
		ans, err := d.Decide(state, qs, []image.Image{img})
		if err != nil {
			t.Fatal(err)
		}
		for i, q := range qs {
			ref, ok := c.Response.Answers[q.ID]
			if !ok {
				t.Fatalf("the golden has no answer to %s", q.ID)
			}
			if ref.Type == "noul" {
				worst = math.Max(worst, math.Abs(ans[i].Noul-ref.Noul))
				continue
			}
			if len(ans[i].Probs) != len(q.Options) {
				t.Fatalf("%s: %d probabilities for %d options", q.ID, len(ans[i].Probs), len(q.Options))
			}
			for j, o := range q.Options {
				worst = math.Max(worst, math.Abs(ans[i].Probs[j]-ref.Probabilities[o.Key]))
			}
		}
		t.Logf("%s, state %v: %d questions, %d positions", c.Image, c.State != nil, len(qs), d.InputTokens())
	}
	return worst
}

// TestD1DecidesOverPicturesAsItsReference holds d1-3B's answers over each
// picture to its own code's, on the host and with every block on each
// device; a picture left out of the prompt is the violation.
func TestD1DecidesOverPicturesAsItsReference(t *testing.T) {
	m := openD1VL(t)
	defer m.Close()
	cs := readD1VisionCases(t)
	d, err := m.NewDecider(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	// Q8_0 weights and int8 activations against the reference's float32,
	// through 27 tower blocks and 30 text blocks; the text-only gate reads
	// 0.021 against llama.cpp.
	const bound = 0.05
	worst := d1VisionWorst(t, d, cs)
	t.Logf("host: worst probability difference %.4f against d1's own code", worst)
	if math.IsNaN(worst) || worst > bound {
		t.Fatalf("host: worst probability difference %.4f (bound %g)", worst, bound)
	}

	// The violation: the same requests with the picture's rows replaced by
	// another picture's.
	other := make([]d1VisionCase, len(cs))
	for i, c := range cs {
		other[i] = c
		other[i].Image = cs[(i+2)%len(cs)].Image
	}
	if other[0].Image != cs[0].Image {
		bad := d1VisionWorst(t, d, other)
		t.Logf("violation, each request answered over another picture: worst %.4f", bad)
		if !(bad > 2*worst && bad > bound) {
			t.Errorf("another picture reads %.4f against a clean %.4f: the gate cannot see the picture", bad, worst)
		}
	}

	for _, spec := range stepDevices() {
		gpu, err := openTier(t, tier.WithDevices(spec))
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			dd, err := m.NewDecider(4096)
			if err != nil {
				t.Fatal(err)
			}
			defer dd.Close()
			if err := dd.SetDevice(gpu, -1); err != nil {
				t.Fatal(err)
			}
			// Every block where the card holds the model (the V100s); on a
			// smaller card the partial seam the bar asks for as well, the
			// count asserted and printed.
			n := dd.DeviceBlocks()
			t.Logf("%s holds %d of %d text blocks (%s)", spec, n, m.Cfg.NLayer, gpu.Err())
			if n == 0 {
				t.Fatalf("%s took no text block, so this arm ran on the host", spec)
			}
			w := d1VisionWorst(t, dd, cs)
			t.Logf("%s, every text block: worst %.4f against d1's own code", spec, w)
			if math.IsNaN(w) || w > bound {
				t.Fatal(fmt.Sprintf("%s: worst %.4f (bound %g)", spec, w, bound))
			}
		})
	}
}

// TestD1PicturesMatchLlamaCpp holds the two picture requests -- two pictures
// over a null state, and a chat-message state carrying its picture as an
// image_url part -- to llama-server's answers on the same GGUF and mmproj
// (scripts/decisiongold.sh). llama.cpp runs the projector's GELU in its tanh
// form and jitllm in the reference's erf form (RULE 7m); both pictures are
// under d1's pixel bound, so the runner's resize is in neither.
func TestD1PicturesMatchLlamaCpp(t *testing.T) {
	m := openD1VL(t)
	defer m.Close()
	for _, req := range []string{"pictures", "chatpicture"} {
		t.Run(req, func(t *testing.T) {
			body, err := os.ReadFile(filepath.Join("testdata", "decision", req+".json"))
			if err != nil {
				t.Fatal(err)
			}
			parsed, err := ParseDecisionRequest(body)
			state, qs, urls := parsed.State, parsed.Questions, parsed.Images
			if err != nil {
				t.Fatal(err)
			}
			if len(urls) == 0 {
				t.Fatal("the request carries no picture, so this gate proves nothing")
			}
			imgs := make([]image.Image, len(urls))
			for i, u := range urls {
				if imgs[i], err = DecodeImageDataURL(u); err != nil {
					t.Fatal(err)
				}
			}
			gold := readDecisionGold(t, "d1-3B-Q8_0."+req+".llamacpp.json")
			d, err := m.NewDecider(4096)
			if err != nil {
				t.Fatal(err)
			}
			defer d.Close()
			ans, err := d.Decide(state, qs, imgs)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%d picture(s), %d positions; llama.cpp read %d", len(imgs), d.InputTokens(), gold.Usage.InputTokens)
			if d.InputTokens() != gold.Usage.InputTokens {
				t.Errorf("ran %d positions, llama.cpp %d: the prompts differ", d.InputTokens(), gold.Usage.InputTokens)
			}
			checkDecisions(t, qs, ans, gold, 0.05, true)

			// The server's path: the picture request in a batch beside a text
			// request runs alone and answers as Decide did; the text request
			// answers from the batch.
			text, tqs := decisionRequest(t, filepath.Join("testdata", "decision", "support.json"))
			bo, err := d.DecideBatch([]DecisionRequest{{State: state, Questions: qs, Images: imgs},
				{State: text, Questions: tqs}})
			if err != nil {
				t.Fatal(err)
			}
			if bo[0].Err != nil || bo[1].Err != nil {
				t.Fatalf("the batch: %v, %v", bo[0].Err, bo[1].Err)
			}
			for i := range qs {
				if probDiffBatch(bo[0].Answers[i], ans[i]) != 0 {
					t.Fatalf("%s: the batch's answer to the picture request is not Decide's", qs[i].ID)
				}
			}
			if len(bo[1].Answers) != len(tqs) {
				t.Fatalf("the text request has %d answers for %d questions", len(bo[1].Answers), len(tqs))
			}

			// Paging: the same request under a budget of four text pages and
			// two tower pages evicts and re-reads, and answers the same to
			// the bit.
			budget := uint64(4*m.container.PageBytes(0) + 2*m.container.PageBytes(int(m.container.H.NBlocks)))
			mp := openD1VL(t, WithPageBudget(budget))
			defer mp.Close()
			dp, err := mp.NewDecider(4096)
			if err != nil {
				t.Fatal(err)
			}
			defer dp.Close()
			pans, err := dp.Decide(state, qs, imgs)
			if err != nil {
				t.Fatal(err)
			}
			if _, ev := mp.container.Faults(); ev == 0 {
				t.Fatal("the budget evicted nothing: paging was not exercised")
			}
			for i := range ans {
				if ans[i].Noul != pans[i].Noul || fmt.Sprint(ans[i].Probs) != fmt.Sprint(pans[i].Probs) {
					t.Fatalf("%s: paged %v/%v, resident %v/%v", qs[i].ID, pans[i].Noul, pans[i].Probs, ans[i].Noul, ans[i].Probs)
				}
			}
			t.Logf("under a %d-byte budget: every answer bit-identical", budget)
		})
	}
}

// probDiffBatch is the largest difference between two answers' probabilities,
// P(true) for a noul.
func probDiffBatch(a, b DecisionAnswer) float64 {
	w := math.Abs(a.Noul - b.Noul)
	for i := range a.Probs {
		if i < len(b.Probs) {
			w = math.Max(w, math.Abs(a.Probs[i]-b.Probs[i]))
		}
	}
	return w
}
