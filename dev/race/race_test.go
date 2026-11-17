package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jitllm/jitllm/dev/httpbench/hb"
)

// fake is an OpenAI-compatible chat endpoint that streams n tokens, the first
// after ttft and each next after gap, and records when each request arrived.
type fake struct {
	ttft, gap time.Duration
	n         int
	mu        sync.Mutex
	arrived   []time.Time
}

func (f *fake) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.arrived = append(f.arrived, time.Now())
	f.mu.Unlock()
	if r.URL.Path != "/v1/chat/completions" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	fl := w.(http.Flusher)
	fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n")
	fl.Flush()
	time.Sleep(f.ttft)
	for i := range f.n {
		if i > 0 {
			time.Sleep(f.gap)
		}
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"t%d \"}}]}\n\n", i)
		fl.Flush()
	}
	fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":%d}}\n\ndata: [DONE]\n\n", f.n)
	fl.Flush()
}

func sides(a, b *httptest.Server) [2]Side {
	req := func(u string) hb.Request {
		return hb.Request{URL: u, API: "chat", Model: "m", Prompt: "p", Max: 8}
	}
	return [2]Side{{Name: "fast", Req: req(a.URL)}, {Name: "slow", Req: req(b.URL)}}
}

// TestRaceStreamsBothAndTimesEach: both sides' text arrives as events, each
// side's final sample has its own TTFT and decode rate, the faster side
// finishes first, and the event channel closes after both.
func TestRaceStreamsBothAndTimesEach(t *testing.T) {
	fa := &fake{ttft: 20 * time.Millisecond, gap: 5 * time.Millisecond, n: 8}
	fb := &fake{ttft: 120 * time.Millisecond, gap: 20 * time.Millisecond, n: 8}
	a, b := httptest.NewServer(fa), httptest.NewServer(fb)
	defer a.Close()
	defer b.Close()
	ss := sides(a, b)
	cl := [2]*http.Client{a.Client(), b.Client()}

	events := make(chan Event, 64)
	var got [2]strings.Builder
	var doneOrder []int
	finished := make(chan struct{})
	go func() {
		for e := range events {
			if e.Done != nil {
				doneOrder = append(doneOrder, e.Side)
				continue
			}
			got[e.Side].WriteString(e.Text)
		}
		close(finished)
	}()
	res := Race(context.Background(), ss, cl, events)
	<-finished

	want := "t0 t1 t2 t3 t4 t5 t6 t7 "
	if got[0].String() != want || got[1].String() != want {
		t.Fatalf("streamed text %q / %q", got[0].String(), got[1].String())
	}
	if len(doneOrder) != 2 || doneOrder[0] != 0 {
		t.Fatalf("finish order %v, want the fast side first", doneOrder)
	}
	for i, s := range res {
		if s.Err != "" || s.CompletionTokens != 8 || s.PromptTokens != 7 || s.Arm != ss[i].Name {
			t.Fatalf("side %d: %+v", i, s)
		}
	}
	if res[0].TTFT < 20*time.Millisecond || res[1].TTFT < 120*time.Millisecond || res[0].TTFT >= res[1].TTFT {
		t.Fatalf("TTFT %v / %v", res[0].TTFT, res[1].TTFT)
	}
	// 7 tokens after the first over 7 gaps: 200 tok/s and 50 tok/s at most,
	// less by whatever the scheduler adds to each gap.
	if r := res[0].DecodeRate(); r > 200 || r < 50 {
		t.Fatalf("fast decode %.1f tok/s", r)
	}
	if r := res[1].DecodeRate(); r > 50 || r < 15 {
		t.Fatalf("slow decode %.1f tok/s", r)
	}
	if res[0].DecodeRate() <= res[1].DecodeRate() {
		t.Fatal("the fast side decoded slower")
	}
}

// TestRaceStartsBothAtOnce: the two requests reach their servers within a
// few milliseconds of each other, whichever side is slower.
func TestRaceStartsBothAtOnce(t *testing.T) {
	fa := &fake{n: 2}
	fb := &fake{n: 2}
	a, b := httptest.NewServer(fa), httptest.NewServer(fb)
	defer a.Close()
	defer b.Close()
	cl := [2]*http.Client{a.Client(), b.Client()}
	ss := sides(a, b)
	if err := Warm(context.Background(), ss, cl, 1); err != nil {
		t.Fatal(err)
	}
	for range 5 {
		Race(context.Background(), ss, cl, nil)
	}
	worst := time.Duration(0)
	for i := range fa.arrived {
		d := fa.arrived[i].Sub(fb.arrived[i]).Abs()
		worst = max(worst, d)
	}
	if len(fa.arrived) != 6 || len(fb.arrived) != 6 {
		t.Fatalf("arrivals %d / %d, want 6 each", len(fa.arrived), len(fb.arrived))
	}
	if worst > 20*time.Millisecond {
		t.Fatalf("the two sides started %v apart", worst)
	}
}

// TestWarmReportsAFailingSide: a side that answers with an error stops the
// warm-up with that side's name, before anything is shown.
func TestWarmReportsAFailingSide(t *testing.T) {
	ok := httptest.NewServer(&fake{n: 2})
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "no such model", http.StatusNotFound)
	}))
	defer ok.Close()
	defer bad.Close()
	err := Warm(context.Background(), sides(ok, bad), [2]*http.Client{ok.Client(), bad.Client()}, 1)
	if err == nil || !strings.Contains(err.Error(), "slow") || !strings.Contains(err.Error(), "404") {
		t.Fatalf("err = %v", err)
	}
}

func TestSummaryNamesBothAndTheRatios(t *testing.T) {
	ss := [2]Side{{Name: "jitllm"}, {Name: "other"}}
	mk := func(ttft time.Duration, toks int, gap time.Duration) hb.Sample {
		s := hb.Sample{TTFT: ttft, CompletionTokens: toks}
		for range toks - 1 {
			s.ITL = append(s.ITL, gap)
		}
		return s
	}
	got := Summary(ss, [2]hb.Sample{mk(50*time.Millisecond, 101, 10*time.Millisecond), mk(100*time.Millisecond, 101, 20*time.Millisecond)})
	want := "jitllm: TTFT 50 ms, 100.0 tok/s · other: TTFT 100 ms, 50.0 tok/s · jitllm/other: decode 2.00x, TTFT 0.50x"
	if got != want {
		t.Fatalf("got  %q\nwant %q", got, want)
	}
	if got := Summary(ss, [2]hb.Sample{{Err: "x"}, mk(time.Millisecond, 2, time.Millisecond)}); !strings.HasPrefix(got, "jitllm: error · ") || strings.Contains(got, "decode") {
		t.Fatalf("an errored side: %q", got)
	}
}

// TestViewFitsTheTerminal: the screen is exactly the terminal's height, so
// the summary row is never scrolled off the bottom of a recording.
func TestViewFitsTheTerminal(t *testing.T) {
	for _, h := range []int{24, 36} {
		m := &model{sides: [2]Side{{Name: "a", Label: "x"}, {Name: "b", Label: "y"}}, header: []string{"hw", "setup"}, w: 120, h: h}
		for i := range m.panes {
			m.panes[i].text.WriteString(strings.Repeat("**Heading**\n\nsome words in a paragraph, ", 300))
			m.panes[i].chunks = 2000
		}
		s := hb.Sample{TTFT: time.Millisecond, CompletionTokens: 3, ITL: []time.Duration{time.Millisecond, time.Millisecond}}
		m.panes[0].done, m.panes[0].rank = &s, 1
		m.panes[1].done, m.panes[1].rank = &s, 2
		m.results = [2]hb.Sample{s, s}
		m.ended = true
		if got := strings.Count(m.View(), "\n") + 1; got != h {
			t.Fatalf("a %d-row terminal got a %d-row screen", h, got)
		}
	}
}

func TestLiveRate(t *testing.T) {
	if r := liveRate(11, time.Second, 2*time.Second); r != 10 {
		t.Fatalf("%v", r)
	}
	if liveRate(1, time.Second, 2*time.Second) != 0 || liveRate(5, time.Second, time.Second) != 0 {
		t.Fatal("a rate from one chunk or no time")
	}
}
