package engine

import (
	"runtime"
	"slices"
	"testing"
	"time"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
)

// preloadIDs opens path with opts, decodes n greedy tokens from prompt and
// returns them with the model, still open, for the caller's counters.
func preloadIDs(t *testing.T, path string, n int, opts ...model.Option) ([]int32, *model.Model) {
	t.Helper()
	m, err := model.Open(path, opts...)
	if err != nil {
		t.Fatal(err)
	}
	in := m.Vocab.Encode("Once upon a time", true)
	st := testInstance(t, m, len(in)+n+1)
	defer st.Close()
	logits, err := st.Prefill(in)
	if err != nil {
		t.Fatal(err)
	}
	var ids []int32
	for range n {
		best := int32(0)
		for j, v := range logits {
			if v > logits[best] {
				best = int32(j)
			}
		}
		ids = append(ids, best)
		if logits, err = st.Forward(best); err != nil {
			t.Fatal(err)
		}
	}
	return ids, m
}

// TestPreloadReadsEveryPageAndKeepsTheAnswer: WithPreload reads every page
// with more than one request outstanding, the answer is the one a model that
// faults its pages in gives, and a budget below the model preloads nothing.
func TestPreloadReadsEveryPageAndKeepsTheAnswer(t *testing.T) {
	for _, name := range []string{"stories15M", "qwen3moe"} {
		t.Run(name, func(t *testing.T) {
			path := jlmOf(t, models()[name])
			want, m0 := preloadIDs(t, path, 8)
			coldReads, coldPeak := m0.PagerReads(), m0.ReadConcurrency()
			m0.Close()

			m, err := model.Open(path, model.WithPreload(4))
			if err != nil {
				t.Fatal(err)
			}
			cf := testFrames(t, m)
			if err := cf.WaitPreload(); err != nil {
				t.Fatal(err)
			}
			c := m.Container
			if got, all := cf.ResidentPages(), c.NPages(); got != all {
				t.Fatalf("preload left %d of %d pages resident", got, all)
			}
			var page uint64
			for i := range c.NPages() {
				page += c.PageBytes(i)
			}
			t.Logf("%s: %d pages, %d bytes read in %d requests, peak %d outstanding "+
				"(faulted in by the tokens: %d requests, peak %d)", name, c.NPages(),
				m.ReadBytes(), m.PagerReads(), m.ReadConcurrency(), coldReads, coldPeak)
			t.Logf("%s: preload peak pages in flight %d", name, m.PreloadDepth(cf))
			if c.NPages() >= 16 && m.PreloadDepth(cf) < 2 {
				t.Fatalf("preload had at most %d page in flight: the pages were read one at a time",
					m.PreloadDepth(cf))
			}
			if uint64(m.ReadBytes()) < page {
				t.Fatalf("preload read %d bytes of %d bytes of pages", m.ReadBytes(), page)
			}
			before := m.PagerReads()
			m.Close()

			got, m2 := preloadIDs(t, path, 8, model.WithPreload(4))
			if after := m2.PagerReads(); after < before {
				t.Fatalf("preload then decode issued %d requests, fewer than the preload alone (%d)",
					after, before)
			}
			m2.Close()
			if !slices.Equal(got, want) {
				t.Fatalf("preloaded %v, faulted in %v", got, want)
			}

			// A budget below the model: the forward pass decides residency.
			m3, err := model.Open(path, model.WithPreload(4))
			if err != nil {
				t.Fatal(err)
			}
			if err := testFrames(t, m3, jlm.Budget(page/2)).WaitPreload(); err != nil {
				t.Fatal(err)
			}
			m4, err := model.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			// Frames over it too, which read the dense region: what differs is
			// the preload alone.
			testFrames(t, m4, jlm.Budget(page/2))
			if r, open := m3.PagerReads(), m4.PagerReads(); r != open {
				t.Fatalf("preload under a budget below the model issued %d reads, frames alone %d", r, open)
			}
			m3.Close()
			m4.Close()
		})
	}
}

// TestCloseStopsAPreload: closing the frames in the middle of a preload waits
// for its reads and leaves no goroutine behind.
func TestCloseStopsAPreload(t *testing.T) {
	path := jlmOf(t, models()["qwen3moe"])
	base := runtime.NumGoroutine()
	for range 4 {
		m, err := model.Open(path, model.WithPreload(4))
		if err != nil {
			t.Fatal(err)
		}
		fr, err := m.NewFrames()
		if err != nil {
			t.Fatal(err)
		}
		fr.Close()
		if err := m.Close(); err != nil {
			t.Fatal(err)
		}
	}
	deadline := time.Now().Add(5 * time.Second)
	for runtime.NumGoroutine() > base && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if n := runtime.NumGoroutine(); n > base {
		t.Fatalf("%d goroutines after four frames closed under a preload, %d before", n, base)
	}
}
