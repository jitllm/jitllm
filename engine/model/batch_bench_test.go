//go:build jitllmbench && (amd64 || arm64)

// A measurement, not a gate: opt-in by build tag, because a measurement on a
// busy box is worse than none and an env-var skip puts a green line in every
// default run. Like jitllmtest (nn/disable.go) and jitllmfault (tier/fault.go):
//
//	go test -tags jitllmbench -run <Name> ./<pkg>
//
// Any JITLLM_* variable the test reads still selects its parameters.

package model

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/dev/bench"
)

// TestABBatchThroughput prices batched decode: how many tokens per second the
// engine delivers in aggregate at each batch width, against the single-sequence
// decode path a user gets.
//
// bench.AB's ratio is per step and a step is N tokens, so the throughput
// speedup is Median*N, and Median is how much slower one batched step is than
// one single token. At the memory-bound limit Median is 1.0 and the speedup is
// N; where the kernel becomes the constraint the curve flattens.
//
//	JITLLM_BENCH_MODEL=<path-to-q-quantized-model> \
//	  ./scripts/cap 8G -- taskset -c 0,2,4,6,8,10 go test ./engine/model/ \
//	  -run TestABBatchThroughput -count=1 -v
//
// JITLLM_BATCH_N sets the widths, JITLLM_DEPTH the context every arm is seeded to.
func TestABBatchThroughput(t *testing.T) {
	path := benchModel(t)
	if err := bench.Guard(false); err != nil {
		t.Skip(err)
	}
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	widths := []int{2, 4, 8, 16}
	if v := os.Getenv("JITLLM_BATCH_N"); v != "" {
		widths = widths[:0]
		for _, f := range strings.Split(v, ",") {
			if n, err := strconv.Atoi(strings.TrimSpace(f)); err == nil && n > 0 {
				widths = append(widths, n)
			}
		}
	}
	depth := envInt("JITLLM_DEPTH", 0)
	iters := envInt("JITLLM_ITERS", 24)
	rounds := envInt("JITLLM_ROUNDS", 8)
	bpt := m.BytesPerToken()

	// A fixed in-range token, repeated: this measures throughput, and a constant
	// input keeps every round doing identical work.
	tok := int32(min(1000, m.Cfg.NVocab-1))

	single := m.NewInstance(depth + iters + 1)
	defer single.Close()
	seed(t, single, tok, depth)
	aFn := func(n int) {
		single.pos = depth
		for i := 0; i < n; i++ {
			if _, err := single.Forward(tok); err != nil {
				t.Error(err)
				return
			}
		}
	}

	for _, nseq := range widths {
		b := newSessBatch(m, nseq, depth+iters+1)
		row := make([]int32, nseq)
		for i := range row {
			row[i] = tok
		}
		for i := 0; i < depth; i++ {
			if _, err := b.ForwardBatch(row); err != nil {
				b.Close()
				t.Fatal(err)
			}
		}
		bFn := func(n int) {
			for _, se := range b.sess {
				se.pos, se.bpos[0] = depth, depth
			}
			for i := 0; i < n; i++ {
				if _, err := b.ForwardBatch(row); err != nil {
					t.Error(err)
					return
				}
			}
		}
		// Both sides converge before either is timed: the GEMM tile is tuned per
		// shape off real calls, so a cold arm measures the search.
		aFn(iters)
		bFn(iters)

		// One weight pass per iteration either way (that is the claim), so the
		// physics guard sees the same denominator on both arms.
		r := bench.AB(
			bench.Case{Name: "single", BytesPerIter: bpt, Threads: single.Workers(), Fn: aFn},
			bench.Case{Name: "batch" + itoa(nseq), BytesPerIter: bpt, Threads: b.Workers(), Fn: bFn},
			rounds, iters)
		one := float64(iters) / r.AMedian.Seconds()
		agg := float64(iters*nseq) / r.BMedian.Seconds()
		t.Logf("N=%-3d depth=%-5d single %6.2f tok/s   batch %7.2f tok/s aggregate (%.2fx)   per-step %.4fx  IQR/median %.1f%%%s",
			nseq, depth, one, agg, agg/one, r.Median, 100*r.IQR/r.Median, unstable(r))
		b.Close()
	}
}

func unstable(r bench.Result) string {
	if r.Stable() {
		return ""
	}
	return "   UNSTABLE -- do not quote"
}

func seed(t *testing.T, s *Instance, tok int32, depth int) {
	t.Helper()
	for i := 0; i < depth; i++ {
		if _, err := s.Forward(tok); err != nil {
			t.Fatal(err)
		}
	}
}

// TestBatchProfile prints the per-op split of a batched step beside a single
// token's, which is how to see where a batch of N stops being N times cheaper.
// Set JITLLM_BENCH_MODEL; JITLLM_BATCH_N picks the width.
func TestBatchProfile(t *testing.T) {
	path := benchModel(t)
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()
	nseq := envInt("JITLLM_BATCH_N", 8)
	depth, iters := envInt("JITLLM_DEPTH", 0), envInt("JITLLM_ITERS", 24)
	tok := int32(min(1000, m.Cfg.NVocab-1))

	show := func(label string, ntok int, run func()) {
		run() // warm: the tuners converge before the counters are read
		m.ResetProfile()
		start := time.Now()
		run()
		took := time.Since(start)
		names, ns := m.OpProfile()
		var tot int64
		for _, v := range ns {
			tot += v
		}
		line := ""
		for i, nm := range names {
			if ns[i] > 0 {
				line += fmt.Sprintf("  %s %.1f%%", nm, 100*float64(ns[i])/float64(took.Nanoseconds()))
			}
		}
		t.Logf("%-10s %6.2f ms/step  %6.2f ms/token  accounted %.0f%%%s",
			label, float64(took.Milliseconds())/float64(iters),
			float64(took.Milliseconds())/float64(iters*ntok),
			100*float64(tot)/float64(took.Nanoseconds()), line)
	}

	s := m.NewInstance(depth + iters + 1)
	defer s.Close()
	seed(t, s, tok, depth)
	show("single", 1, func() {
		s.pos = depth
		for i := 0; i < iters; i++ {
			if _, err := s.Forward(tok); err != nil {
				t.Fatal(err)
			}
		}
	})

	b := newSessBatch(m, nseq, depth+iters+1)
	defer b.Close()
	row := make([]int32, nseq)
	for i := range row {
		row[i] = tok
	}
	for i := 0; i < depth; i++ {
		if _, err := b.ForwardBatch(row); err != nil {
			t.Fatal(err)
		}
	}
	show("batch"+itoa(nseq), nseq, func() {
		for _, se := range b.sess {
			se.pos, se.bpos[0] = depth, depth
		}
		for i := 0; i < iters; i++ {
			if _, err := b.ForwardBatch(row); err != nil {
				t.Fatal(err)
			}
		}
	})
}
