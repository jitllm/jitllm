package model

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestManySessionsStreamTheirHistory is a server's step loop in miniature,
// the shape that crashed jitllmd at 64 concurrent requests on a V100: many
// sessions stepping together through StepRuns on one device whose KV budget
// their histories outgrow, requests finishing and their pooled States reset
// and taking a new prompt while the others decode, relocation on. Pages go
// home and stream back under every mix of joint steps, one-State calls,
// blocks moving and sequences restarting lower than their evicted history.
//
// Every request is then replayed alone on the same tier with the budget
// lifted, teacher-forced through the ids it chose, and its logits held to
// that run within the eviction gate's band: a page at home written into, or
// streamed from a stale copy, is tenths. KVEvictions and KVStreamPasses are
// the selection check, and Reset must have met evicted history (rewinds).
func TestManySessionsStreamTheirHistory(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")), noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, spec := range testdevices.Specs("cuda:0", "vulkan:0") {
		t.Run(spec, func(t *testing.T) {
			manySessionsStream(t, m, spec)
		})
	}
}

// streamReq is one request of TestManySessionsStreamTheirHistory: its prompt,
// the ids fed after it, and a sample of the logits after the prompt and
// after each id.
type streamReq struct {
	prompt, ids []int32
	logits      [][]float32
}

// streamSample is every streamStride-th logit: enough of the row to see a
// lost page, small enough to keep every step of every request.
func streamSample(lg []float32) []float32 {
	out := make([]float32, 0, len(lg)/streamStride+1)
	for i := 0; i < len(lg); i += streamStride {
		out = append(out, lg[i])
	}
	return out
}

const streamStride = 97

func manySessionsStream(t *testing.T, m *Model, spec string) {
	g := atOnceTier(t, spec, func(c *tier.Config) {
		c.KVPage = 64
		c.StagedDecode = true
	})
	const n, seq, gen, steps = 12, 1024, 128, 600
	words := strings.Repeat("The clerk counted barrels of salt on the upper floor while rain fell. ", 60)
	all := m.Vocab.Encode(words, true)
	type live struct {
		st   *State
		next int32
		left int
		req  *streamReq
	}
	var done []*streamReq
	start := func(l *live, k int) {
		if l.req != nil {
			done = append(done, l.req)
		}
		if l.st == nil {
			l.st = m.NewState(seq)
			if err := l.st.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			l.st.SetRelocate(true)
		} else {
			l.st.Reset()
		}
		// Each prompt starts at its own word, so a request reusing a State
		// writes rows that differ from the history it left: a stale page reads
		// as another text, not the same one.
		off := 1 + (k*7)%13
		p := append([]int32{all[0]}, all[off:off+63+(k*97)%400]...)
		lg, err := l.st.Prefill(p)
		if err != nil {
			t.Fatalf("prompt of %d: %v (%s)", len(p), err, g.Err())
		}
		l.req = &streamReq{prompt: p, logits: [][]float32{streamSample(lg)}}
		l.next, l.left = Greedy(lg), gen/2+(k*31)%gen
	}
	ss := make([]live, n)
	for k := range ss {
		start(&ss[k], k)
	}
	defer func() {
		for _, s := range ss {
			s.st.Close()
		}
	}()
	before := g.Stats()
	if _, err := g.SetBudget(before.BudgetUsed + 32<<20); err != nil {
		t.Fatal(err)
	}
	refused := 0
	for step := 0; step < steps; step++ {
		runs := make([]Run, n)
		for k, s := range ss {
			runs[k] = Run{State: s.st, Tokens: []int32{s.next}, Logits: true}
		}
		// A step the device refuses runs its rows one State at a time, as
		// the server's step loop does.
		out, err := StepRuns(runs)
		if err != nil {
			refused++
			out = make([][]float32, n)
			for k, r := range runs {
				o, err := StepRuns([]Run{r})
				if err != nil {
					t.Fatalf("step %d, session %d alone: %v (%s)", step, k, err, g.Err())
				}
				out[k] = o[0]
			}
		}
		for k := range ss {
			s := &ss[k]
			s.req.ids = append(s.req.ids, s.next)
			s.req.logits = append(s.req.logits, streamSample(out[k]))
			s.next = Greedy(out[k])
			if s.left--; s.left <= 0 {
				start(s, k+step)
			}
		}
	}
	st := g.Stats()
	for k := range ss {
		ss[k].st.Close()
	}
	ss = nil
	t.Logf("%s: %d evictions, %d streamed passes, %d joint steps refused, %d requests", spec,
		st.KVEvictions, st.KVStreamPasses, refused, len(done))
	if st.KVEvictions == 0 || st.KVStreamPasses == 0 {
		t.Fatalf("the budget never bound: %d evictions, %d passes", st.KVEvictions, st.KVStreamPasses)
	}
	// Each finished request alone, with room for its whole history.
	if _, err := g.SetBudget(before.BudgetUsed + 1<<30); err != nil {
		t.Fatal(err)
	}
	alone := m.NewState(seq)
	defer alone.Close()
	if err := alone.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	ev := g.Stats().KVEvictions
	worst := 0.0
	for i, r := range done {
		alone.Reset()
		lg, err := alone.Prefill(r.prompt)
		if err != nil {
			t.Fatal(err)
		}
		var num, den float64
		add := func(got, want []float32) {
			for j := range want {
				d := float64(got[j] - want[j])
				num += d * d
				den += float64(want[j]) * float64(want[j])
			}
		}
		add(r.logits[0], streamSample(lg))
		for j, id := range r.ids {
			if lg, err = alone.Forward(id); err != nil {
				t.Fatal(err)
			}
			add(r.logits[j+1], streamSample(lg))
		}
		e := num / den
		if math.IsNaN(e) || math.IsInf(e, 0) {
			t.Fatalf("request %d: logits NMSE %v", i, e)
		}
		worst = max(worst, e)
		if !(e < evictNMSE) {
			t.Errorf("request %d (prompt %d, %d steps): logits NMSE %.3e against the request alone, bound "+
				"%.0e: its history was lost or read stale", i, len(r.prompt), len(r.ids), e, evictNMSE)
		}
	}
	if d := g.Stats().KVEvictions - ev; d != 0 {
		t.Fatalf("the requests alone sent %d page(s) home: the reference evicted too", d)
	}
	t.Logf("%s: %d requests held to themselves alone, worst NMSE %.3e", spec, len(done), worst)
	if slices.ContainsFunc(done, func(r *streamReq) bool { return len(r.ids) == 0 }) {
		t.Fatal("a request ran no step")
	}
}
