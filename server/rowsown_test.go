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
// composed for every row count (gpu.GPU.RowsComposed), a session's rows are
// its own. Beside other sessions -- a shorter prompt, a longer history, two
// prompts -- its logits are bit for bit those of the same steps beside other
// sessions of the same lengths: the same forms run, and only the other rows
// differ. Against its steps alone, or its prompt in chunks, other forms run
// (a form per row count, its own arithmetic: RULE 8), so they are held to a
// band: three times how far the session's two paths alone stand apart, the
// joint step and the one-at-a-time path. Every arm's tokens are the arm alone's, teacher-forced, so one
// argmax flipped inside the band does not part every step after it.
//
// The one-at-a-time path (engine.StepRuns of one instance: Prefill, then
// Forward's one-row form) is the other path the server's loop runs a row
// alone on; how far it stands from the joint step alone sets the band.
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
		// lone runs a step of one instance the server's way (engine.StepRuns),
		// its prompt through Prefill and a token through Forward.
		lone := false
		step := func(runs ...engine.Run) [][]float32 {
			stepf := func(runs []engine.Run) ([][]float32, error) { return engine.StepJoint(runs, nil) }
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
		var forced []int32
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
			if forced != nil {
				next[len(others)] = forced[0]
			}
			t0 := time.Now()
			for k := range dec {
				runs = runs[:0]
				for i := range others {
					runs = append(runs, run(sts[i], next[i:i+1]))
				}
				out = step(append(runs, run(sa, next[len(others):]))...)
				got = append(got, out[len(out)-1])
				for i := range next {
					next[i] = argmax(out[i])
				}
				if forced != nil {
					next[len(others)] = forced[k+1]
				}
			}
			return got, time.Since(t0) / dec
		}
		alone, dt := trace(nil, false, 0)
		t.Logf("alone: %v a decode step", dt)
		for _, row := range alone {
			forced = append(forced, argmax(row))
		}
		// apart is how far got's steps are from want's: the elements that
		// differ, the largest difference and the worst step's NMSE.
		apart := func(want, got [][]float32) (n int, diff, worst float64) {
			for k := range want {
				var num, den float64
				for i := range want[k] {
					d := float64(want[k][i] - got[k][i])
					if d != 0 || math.Float32bits(want[k][i]) != math.Float32bits(got[k][i]) {
						n++
					}
					diff = max(diff, math.Abs(d))
					num += d * d
					den += float64(want[k][i]) * float64(want[k][i])
				}
				worst = max(worst, num/max(den, 1e-30))
			}
			return n, diff, worst
		}
		// other is a prompt of each of others' lengths, of other tokens.
		lone = true
		loneAlone, _ := trace(nil, false, 0)
		lone = false
		_, _, paths := apart(alone, loneAlone)
		band := 3 * max(paths, 1e-6)
		t.Logf("the two paths alone stand %.3e apart: the band is %.3e", paths, band)
		other := func(others [][]int32) [][]int32 {
			out := make([][]int32, len(others))
			for i, o := range others {
				out[i] = mk(uint32(100+i), len(o))
			}
			return out
		}
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
			{"in chunks of 20, the one-at-a-time path", nil, false, 20, true},
		} {
			lone = arm.lone
			got, dt := trace(arm.others, arm.pre, arm.chunk)
			t.Logf("%s: %v a decode step", arm.name, dt)
			if len(arm.others) > 0 {
				twin, _ := trace(other(arm.others), arm.pre, arm.chunk)
				if n, diff, _ := apart(got, twin); n > 0 {
					t.Errorf("%s: %d logits differ beside other sessions of the same lengths, by up to %.3g: "+
						"a row read another's", arm.name, n, diff)
				}
			}
			lone = false
			n, diff, worst := apart(alone, got)
			switch {
			case worst > band:
				t.Errorf("%s: NMSE %.3e from alone (%d logits apart, by up to %.3g), past the band %.3e",
					arm.name, worst, n, diff, band)
			case n > 0:
				t.Logf("%s: NMSE %.3e from alone, by up to %.3g", arm.name, worst, diff)
			}
		}
	})
}
