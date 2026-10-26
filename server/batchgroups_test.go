package server

import (
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// TestBatchDecodeGroupsKeepTheTokens: a model whose blocks are split over two
// devices decodes concurrent greedy requests in decode groups
// (batchgroups.go), each group stepping on its own, and every row's tokens are
// bit for bit the ones it decodes in one group (DecodeGroups 1, every decoding
// row in each step) -- with a group per card, and with a group per row. The
// tier's StageOverlaps says a stage began on one card while another ran
// another group's: the groups filled the pipeline rather than taking it in
// turn. Every arm runs on one load, so the device's tuners chose once for all
// of them.
//
// The loop groups by itself (DecodeGroups 0) only where the tier's steps are
// one kernel for every row count (tier.GPU.RowsComposed); elsewhere the arm
// must not group at all, and the forced arms, whose tokens then move with
// their step's rows, log where they part instead of holding them equal.
//
// Against groups forced to step one after another (stepLoop.serialGroups), the
// tokens are the same and StageOverlaps stays at zero, which the grouped arms'
// assertion refuses.
//
// JITLLM_TEST_DEVICES names the two devices (default the laptop's card under
// CUDA and Vulkan); JITLLM_SERVER_BATCH_MODELS adds models, as the batch gates
// read it.
func TestBatchDecodeGroupsKeepTheTokens(t *testing.T) {
	devs := testdevices.Take(t, 2, "cuda:0,vulkan:1")
	eachBatchModel(t, func(t *testing.T, name string) {
		const n, gen = 8, 20
		gl := loadGrouped(t, name, devs, n)
		one := gl.run(t, 1, false)
		if one.groupSteps != 0 {
			t.Fatalf("one group ran %d grouped step(s)", one.groupSteps)
		}
		invariant := gl.lm.gpuTier().RowsComposed()
		t.Logf("the steps are one kernel for every row count: %v", invariant)
		for _, arm := range []struct {
			what   string
			groups int
			serial bool
		}{
			{"a group per card", 0, false},
			{"two groups forced", 2, false},
			{"a group per row", n, false},
			{"groups forced serial", 2, true},
		} {
			t.Run(arm.what, func(t *testing.T) {
				got := gl.run(t, arm.groups, arm.serial)
				if arm.groups == 0 && !invariant {
					if got.groupSteps != 0 {
						t.Fatalf("%d grouped step(s) on steps whose kernels follow the row count", got.groupSteps)
					}
					return
				}
				for i := range n {
					if slices.Equal(got.ids[i], one.ids[i]) {
						continue
					}
					if invariant {
						t.Fatalf("row %d: %v in groups, %v in one group", i, got.ids[i], one.ids[i])
					}
					j := 0
					for j < len(got.ids[i]) && got.ids[i][j] == one.ids[i][j] {
						j++
					}
					t.Logf("row %d parts at token %d in forced groups on kernels chosen per row count", i, j)
				}
				if got.groupSteps == 0 {
					t.Fatalf("no grouped step ran: %+v", got)
				}
				t.Logf("%d grouped steps in %d round(s), %d stage overlap(s) (one group: %d)",
					got.groupSteps, got.groupRounds, got.overlaps, one.overlaps)
				if arm.serial {
					if got.overlaps != 0 {
						t.Fatalf("groups stepped one after another, yet %d stage(s) overlapped", got.overlaps)
					}
					return
				}
				// A stage overlaps another card's at nearly every grouped step
				// once the groups fill the pipeline (39 of 40 on the laptop's
				// stories15M); a quarter leaves room for a host-bound tiny model.
				if got.overlaps*4 < int(got.groupSteps) {
					t.Fatalf("%d stage(s) of %d grouped steps overlapped another card's: the groups took the cards in turn",
						got.overlaps, got.groupSteps)
				}
			})
		}
	})
}

// groupedLoad is a model loaded split over two devices, its loop, and how
// many requests each arm sends.
type groupedLoad struct {
	e  *Engine
	lm *LoadedModel
	c  clients
	n  int
}

type groupedResult struct {
	ids                     [][]int32
	groupSteps, groupRounds int64
	overlaps                int
}

// loadGrouped loads name with its first half of blocks on devs[0] and the
// rest on devs[1], room for n sessions, every decode step joint.
func loadGrouped(t *testing.T, name string, devs []string, n int) *groupedLoad {
	t.Helper()
	path := modelPath(t, name)
	probe := New(Config{Version: "test", NoMemCache: true})
	lm, err := probe.LoadModel(LoadOptions{Path: path, ModelID: "dev", DeviceIDs: devs})
	if err != nil {
		probe.Close()
		t.Fatalf("loading %s onto -devices %s: %v", name, strings.Join(devs, ","), err)
	}
	nb := lm.m.Cfg.NLayer
	probe.Close()
	place := fmt.Sprintf("0-%d=%s,%d-%d=%s", nb/2-1, devs[0], nb/2, nb-1, devs[1])
	e := New(Config{Version: "test", NoMemCache: true, DefaultMaxSeq: 256, JointSteps: JointAlways})
	t.Cleanup(e.Close)
	lm, err = e.LoadModel(LoadOptions{Path: path, ModelID: "dev", DeviceIDs: devs, Sessions: n + 1,
		Placement: place, PlacementStrict: true})
	if err != nil {
		t.Fatalf("loading %s with %q: %v", name, place, err)
	}
	if lm.loop == nil {
		t.Fatal("a model loaded onto devices has no step loop")
	}
	s, err := e.CreateSession(SessionOptions{ModelID: lm.id})
	if err != nil {
		t.Fatal(err)
	}
	why, d := s.st.StepRefusal(), lm.gpuTier().PipelineDepth()
	if err := e.CloseSession(s.id); err != nil {
		t.Fatal(err)
	}
	if why != nil || d != 2 {
		t.Fatalf("a session's blocks are on %d device(s), want 2 (%q), and its step refusal is %v", d, place, why)
	}
	return &groupedLoad{e: e, lm: lm, c: serveEngine(t, e), n: n}
}

// run sends the load's n greedy requests together (20 tokens each, EOS
// ignored so every row runs to the end) with the loop's decode groups set to
// groups, and returns their tokens and the loop's and the tier's counts.
func (gl *groupedLoad) run(t *testing.T, groups int, serial bool) groupedResult {
	t.Helper()
	const gen = 20
	lp := gl.lm.loop
	// The loop is idle between arms: its next serve reads these after the
	// requests below wake it.
	lp.groupsCfg, lp.serialGroups = groups, serial
	steps0, rounds0 := lp.stats.groupSteps.Load(), lp.stats.groupRounds.Load()
	overlaps0 := gl.lm.gpuTier().Stats().StageOverlaps
	release := holdGates(t, gl.e, gl.lm)
	out := make([]completion, gl.n)
	var wg sync.WaitGroup
	for i := range gl.n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out[i] = complete(gl.c, &v1.GenerateRequest{ModelId: "dev",
				Prompt: text(fmt.Sprintf("%s %d", prompts[i%len(prompts)], i)), MaxTokens: gen, IgnoreEos: true})
		}()
	}
	waitFor(t, "every request to wait for a row", func() bool { return waitingFor(lp) == gl.n })
	release()
	wg.Wait()
	r := groupedResult{ids: make([][]int32, gl.n)}
	for i, o := range out {
		if o.err != nil {
			t.Fatalf("request %d: %v", i, o.err)
		}
		if !o.started.GetBatched() {
			t.Fatalf("request %d did not run as a row", i)
		}
		if len(o.ids) != gen {
			t.Fatalf("request %d: %d tokens, want %d", i, len(o.ids), gen)
		}
		r.ids[i] = o.ids
	}
	r.groupSteps = lp.stats.groupSteps.Load() - steps0
	r.groupRounds = lp.stats.groupRounds.Load() - rounds0
	r.overlaps = gl.lm.gpuTier().Stats().StageOverlaps - overlaps0
	return r
}
