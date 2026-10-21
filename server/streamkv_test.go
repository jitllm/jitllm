package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestSixtyFourRequestsWhoseHistoryStreams is the load that killed jitllmd on
// a V100 (httpbench at 64 concurrent requests, Llama 3.1 8B) and then failed
// 172 of 256 requests there: 64 completions at once over HTTP, twice -- the
// second wave on the pooled States the first left -- with the card's budget
// held below what their histories take. Admission (stepLoop.kvRoom) keeps the
// admitted rows' histories on the card and holds the rest back until rows
// retire, so the rows that run step together: every request completes, the
// budget bound (admissions were put off), and joint steps ran in both waves.
// Before admission every row was admitted, the oldest pages of most sessions
// went home, and a step with two sequences' pages at home is refused as one
// step, so each session ran alone and re-streamed its evicted prefix every
// token, a Sync a pass. Its violation is
// TestSixtyFourRequestsGateDiscriminates.
func TestSixtyFourRequestsWhoseHistoryStreams(t *testing.T) {
	for wave, w := range sixtyFour(t, 2, false) {
		t.Logf("wave %d: %d of 64 requests failed; %d joint steps, %d admissions put off for room, "+
			"%d refusals, %d evictions, %d streamed passes", wave, w.failed, w.joint, w.waits, w.refusals,
			w.evictions, w.passes)
		if w.failed > 0 {
			t.Fatalf("wave %d: %d of 64 requests failed", wave, w.failed)
		}
		if w.joint == 0 {
			t.Fatalf("wave %d: no joint step ran: the rows stepped one session at a time", wave)
		}
		if w.waits == 0 {
			t.Fatalf("wave %d: no admission was put off: the budget never bound and the gate proved nothing", wave)
		}
	}
}

// TestSixtyFourRequestsGateDiscriminates runs one wave of the gate with every
// waiting request admitted whatever the card has room for, and demands the
// gate's checks fail and the card thrash as it did before admission: pages
// sent home and joint steps refused.
func TestSixtyFourRequestsGateDiscriminates(t *testing.T) {
	w := sixtyFour(t, 1, true)[0]
	t.Logf("violation: %d of 64 requests failed; %d joint steps, %d admissions put off, %d refusals, "+
		"%d evictions, %d streamed passes", w.failed, w.joint, w.waits, w.refusals, w.evictions, w.passes)
	if w.waits != 0 {
		t.Fatalf("%d admissions were put off with admission off", w.waits)
	}
	if w.evictions == 0 || w.refusals == 0 {
		t.Fatalf("with every request admitted the card neither evicted (%d) nor refused a joint step (%d): "+
			"the load does not bind and the gate cannot tell admission from none", w.evictions, w.refusals)
	}
}

// TestSixtyFourRequestsTimeSliced is the same load at fairness level 100:
// rows are parked for the requests waiting on the card's room, and every
// request still completes, in both waves, with joint steps.
func TestSixtyFourRequestsTimeSliced(t *testing.T) {
	for wave, w := range sixtyFour(t, 2, false, 100) {
		t.Logf("wave %d: %d of 64 requests failed; %d joint steps, %d parks, %d resumes parked again, "+
			"%d admissions put off, %d refusals, %d evictions", wave, w.failed, w.joint, w.parks, w.short,
			w.waits, w.refusals, w.evictions)
		if w.failed > 0 {
			t.Fatalf("wave %d: %d of 64 requests failed", wave, w.failed)
		}
		if w.joint == 0 || w.parks == 0 {
			t.Fatalf("wave %d: %d joint steps, %d parks: the gate did not time-slice joint rows", wave, w.joint, w.parks)
		}
	}
}

// waveCounts is what one wave of sixtyFour did.
type waveCounts struct {
	failed                                    int
	joint, waits, refusals, evictions, passes int64
	parks, short                              int64
}

// sixtyFour loads Llama 3.2 1B wholly onto gpu:0 with the budget 1 GiB past
// what placement spent, and sends waves of 64 concurrent greedy completions,
// half of 128 words and half of 512; admitAll is the violation.
func sixtyFour(t *testing.T, waves int, admitAll bool, level ...int) []waveCounts {
	const name = "Llama-3.2-1B-Instruct-Q4_K_M.jlm"
	requireDevice(t, "gpu:0")
	path := modelPath(t, name)
	cfg := Config{Probe: oneCardProbe, Version: "test", DefaultMaxSeq: 1024, MaxBatchRows: 64}
	if len(level) > 0 {
		cfg.Fairness = &level[0]
	}
	e := New(cfg)
	t.Cleanup(e.Close)
	lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "dev", DeviceIDs: []string{"gpu:0"},
		tierConfig: func(c *tier.Config) { c.KVPage = 64 }})
	if err != nil {
		t.Fatalf("loading %s onto -devices gpu:0: %v", name, err)
	}
	if lm.loop == nil || lm.gpu == nil {
		t.Fatal("a model loaded onto a device has no step loop")
	}
	lm.loop.admitAll = admitAll
	requireWholeOnDevice(t, e, lm)
	c := serveEngine(t, e)
	if _, err := lm.gpu.SetBudget(lm.gpu.Stats().BudgetUsed + 1<<30); err != nil {
		t.Fatal(err)
	}
	words := strings.Fields(strings.Repeat("the clerk counted barrels of salt on the upper floor while rain fell ", 50))
	const n = 64
	st := &lm.loop.stats
	var out []waveCounts
	for wave := range waves {
		j0, w0, r0 := st.jointSteps.Load(), st.kvWaits.Load(), st.refusals.Load()
		p0, q0 := st.parks.Load(), st.shortResumes.Load()
		s0 := lm.gpu.Stats()
		errs := make([]error, n)
		var wg sync.WaitGroup
		for i := range n {
			wg.Add(1)
			go func() {
				defer wg.Done()
				w := 128
				if i%2 == 1 {
					w = 512
				}
				errs[i] = complete64(c.url, strings.Join(words[:w], " "))
			}()
		}
		wg.Wait()
		var wc waveCounts
		for i, err := range errs {
			if err != nil {
				wc.failed++
				if wc.failed <= 3 && !admitAll {
					t.Errorf("wave %d, request %d: %v", wave, i, err)
				}
			}
		}
		s1 := lm.gpu.Stats()
		wc.joint, wc.waits, wc.refusals = st.jointSteps.Load()-j0, st.kvWaits.Load()-w0, st.refusals.Load()-r0
		wc.evictions, wc.passes = int64(s1.KVEvictions-s0.KVEvictions), int64(s1.KVStreamPasses-s0.KVStreamPasses)
		wc.parks, wc.short = st.parks.Load()-p0, st.shortResumes.Load()-q0
		out = append(out, wc)
	}
	return out
}

// complete64 posts one greedy 128-token completion of prompt and reports
// anything but a 200 carrying a choice and its tokens.
func complete64(url, prompt string) error {
	// The prompt is plain lowercase words: %q is its JSON string.
	body := fmt.Sprintf(`{"model":"dev","prompt":%q,"max_tokens":128,"temperature":0,"ignore_eos":true}`, prompt)
	resp, err := http.Post(url+"/v1/completions", "application/json", strings.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	var out struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if resp.StatusCode != 200 || json.Unmarshal(b, &out) != nil || len(out.Choices) != 1 ||
		out.Usage.CompletionTokens == 0 {
		return fmt.Errorf("status %d: %s", resp.StatusCode, b)
	}
	return nil
}
