package server

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestStepRowsAreTheirOwn: on a model split over two devices whose steps are
// one kernel for every row count (gpu.GPU.RowsComposed), a session's logits
// are bit for bit the same whatever other sessions share its steps -- its
// prompt alone or beside another's, its decode alone, beside a shorter
// history or beside a longer one, its prompt in chunks of any size -- which is
// what lets the decode groups split rows without moving a token.
//
// The one-at-a-time path (model.StepRuns of one instance: Prefill's tier
// kernels, Forward's one-row form) is other arithmetic, and the server's
// loop moves a row onto it whenever the row steps alone, so a row's tokens
// still follow the schedule: those arms are measured and logged as the gap
// that keeps the load gate's one-group arm from repeating itself, and not
// held equal (Qwen3-0.6B on the laptop: up to 0.30 of a logit, the joint
// arm at one row 59.7 ms a step where Forward is 18.7).
func TestStepRowsAreTheirOwn(t *testing.T) {
	devs := testdevices.Take(t, 2, "cuda:0,vulkan:1")
	eachBatchModel(t, func(t *testing.T, name string) {
		gl := loadUnderLoad(t, name, devs, 4, 1024)
		if !gl.lm.gpuTier().RowsComposed() {
			t.Fatal("the steps are not one kernel for every row count")
		}
		const dec = 16
		c := gl.lm.m.Cfg
		vocab := uint32(min(c.NVocab-1, 20000))
		mk := func(seed uint32, n int) []int32 {
			p := make([]int32, n)
			x := seed*2654435761 + 1
			for j := range p {
				x ^= x << 13
				x ^= x >> 17
				x ^= x << 5
				p[j] = 1 + int32(x%vocab)
			}
			return p
		}
		long := min(900, int(c.NCtx)-dec-2)
		a, b, l := mk(1, 61), mk(2, 29), mk(3, long)
		state := func() *engine.Session {
			s, err := gl.e.CreateSession(SessionOptions{ModelID: gl.lm.id})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { gl.e.CloseSession(s.id) })
			return s.ms
		}
		// lone runs a step of one instance the server's way (model.StepRuns),
		// its prompt through Prefill and a token through Forward.
		lone := false
		step := func(runs ...engine.Run) [][]float32 {
			stepf := engine.StepJoint
			if lone && len(runs) == 1 {
				stepf = engine.StepRuns
			}
			out, err := stepf(runs)
			if err != nil {
				t.Fatal(err)
			}
			cp := make([][]float32, len(out))
			for i, o := range out {
				cp[i] = append([]float32(nil), o...)
			}
			return cp
		}
		// run is a step's run of st's, its session leased for the phase the
		// server's loop takes for it.
		run := func(st *engine.Session, toks []int32) engine.Run {
			p := engine.PhaseGenerate
			if len(toks) > 1 {
				p = engine.PhasePrefill
			}
			if err := st.Acquire(context.Background(), p); err != nil {
				t.Fatal(err)
			}
			return engine.Run{Session: st, Tokens: toks, Logits: true}
		}
		argmax := func(x []float32) int32 {
			best := 0
			for i, v := range x {
				if v > x[best] {
					best = i
				}
			}
			return int32(best)
		}
		// trace is a's prompt logits and its next dec decode steps' logits
		// with company: the other states' prompts beside a's, and each
		// decode step beside theirs.
		// With pre, each other state's prompt runs alone first, and a's
		// prompt steps beside their first decode.
		trace := func(others [][]int32, pre bool, chunk int) ([][]float32, time.Duration) {
			sa := state()
			pa := a
			for chunk > 0 && len(pa) > chunk {
				step(run(sa, pa[:chunk]))
				pa = pa[chunk:]
			}
			if chunk < 0 {
				// The prompt but its last token through Prefill, its last token
				// joint: Prefill's blocks without its head.
				if _, err := engine.StepRuns([]engine.Run{run(sa, pa[:len(pa)-1])}); err != nil {
					t.Fatal(err)
				}
				pa = pa[len(pa)-1:]
			}
			sts := make([]*engine.Session, len(others))
			runs := []engine.Run{}
			next := make([]int32, len(others)+1)
			for i, o := range others {
				sts[i] = state()
				if pre {
					next[i] = argmax(step(run(sts[i], o))[0])
					o = next[i : i+1]
				}
				runs = append(runs, run(sts[i], o))
			}
			out := step(append(runs, run(sa, pa))...)
			got := [][]float32{out[len(out)-1]}
			for i := range next {
				next[i] = argmax(out[i])
			}
			t0 := time.Now()
			for range dec {
				runs = runs[:0]
				for i := range others {
					runs = append(runs, run(sts[i], next[i:i+1]))
				}
				out = step(append(runs, run(sa, next[len(others):]))...)
				got = append(got, out[len(out)-1])
				for i := range next {
					next[i] = argmax(out[i])
				}
			}
			return got, time.Since(t0) / dec
		}
		alone, dt := trace(nil, false, 0)
		t.Logf("alone: %v a decode step", dt)
		for _, arm := range []struct {
			name   string
			others [][]int32
			pre    bool
			chunk  int
			lone   bool
		}{
			{"beside a shorter prompt", [][]int32{b}, false, 0, false},
			{"beside a longer history", [][]int32{l}, true, 0, false},
			{"beside a longer and a shorter history", [][]int32{l, b}, true, 0, false},
			{"beside two prompts", [][]int32{b, mk(4, 17)}, false, 0, false},
			{"its prompt in chunks of 20", nil, false, 20, false},
			{"its prompt in chunks of 7, beside a longer history", [][]int32{l}, true, 7, false},
			{"its prompt a token at a time", nil, false, 1, false},
			{"its prompt through Prefill but its last token", nil, false, -1, false},
			{"alone, the one-at-a-time path", nil, false, 0, true},
			{"in chunks of 20, the one-at-a-time path", nil, false, 20, true},
		} {
			lone = arm.lone
			got, dt := trace(arm.others, arm.pre, arm.chunk)
			t.Logf("%s: %v a decode step", arm.name, dt)
			lone = false
			for k := range alone {
				diff, n := 0.0, 0
				for i := range alone[k] {
					if math.Float32bits(alone[k][i]) != math.Float32bits(got[k][i]) {
						n++
						diff = max(diff, math.Abs(float64(alone[k][i]-got[k][i])))
					}
				}
				switch {
				case n > 0 && arm.chunk >= 0 && !arm.lone:
					t.Errorf("%s, step %d: %d of %d logits differ from alone, by up to %.3g", arm.name, k, n, len(alone[k]), diff)
				case n > 0 && k < 2:
					t.Logf("%s, step %d: %d of %d logits differ from the joint arm, by up to %.3g", arm.name, k, n, len(alone[k]), diff)
				}
			}
		}
	})
}
