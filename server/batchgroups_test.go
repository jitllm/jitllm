package server

import (
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

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
// The loop groups by itself (DecodeGroups -1) only where the tier's steps are
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
		// The default groups nothing until the load gate passes
		// (TestDecodeGroupsHoldUnderLoad).
		if d := gl.run(t, 0, false); d.groupSteps != 0 {
			t.Fatalf("the default ran %d grouped step(s)", d.groupSteps)
		}
		invariant := gl.lm.gpuTier().RowsComposed()
		t.Logf("the steps are one kernel for every row count: %v", invariant)
		for _, arm := range []struct {
			what   string
			groups int
			serial bool
		}{
			{"a group per card", -1, false},
			{"two groups forced", 2, false},
			{"a group per row", n, false},
			{"groups forced serial", 2, true},
		} {
			t.Run(arm.what, func(t *testing.T) {
				got := gl.run(t, arm.groups, arm.serial)
				if arm.groups < 0 && !invariant {
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

// TestDecodeGroupsHoldUnderLoad: a model split over two devices serves conc
// requests at once, prompts long enough to page history alternating with
// short ones, once with one decode group and once with the groups on (the
// loop's own count where the steps are one kernel for every row count, two
// forced elsewhere). Each arm runs the requests two ways: in waves of conc
// admitted together (holdGates), whose schedule is the same in both arms, so
// every request's tokens must be the ones it decodes with one group wherever
// the steps are one kernel for every row count; and as a closed loop of conc
// clients, each sending its next request as its last ends, as a server under
// load sees them, where when a request arrives moves which rows share its
// prompt's steps, so its tokens may part at a near tie and only its answer is
// required. Every request answers in full, the groups ran grouped steps, and
// the devices serve a request after each arm.
//
// JITLLM_SERVER_GROUP_LOAD is conc,reqs,long,short,gen (default
// 16,2,3072,384,64, the prompts cut to the model's context).
func TestDecodeGroupsHoldUnderLoad(t *testing.T) {
	devs := testdevices.Take(t, 2, "cuda:0,vulkan:1")
	conc, reqs, long, short, gen := 16, 2, 3072, 384, 64
	if v := os.Getenv("JITLLM_SERVER_GROUP_LOAD"); v != "" {
		if _, err := fmt.Sscanf(v, "%d,%d,%d,%d,%d", &conc, &reqs, &long, &short, &gen); err != nil {
			t.Fatalf("JITLLM_SERVER_GROUP_LOAD=%q: want conc,reqs,long,short,gen: %v", v, err)
		}
	}
	eachBatchModel(t, func(t *testing.T, name string) {
		gl := loadUnderLoad(t, name, devs, conc, long+gen+1)
		c := gl.lm.m.Cfg
		long := min(long, int(c.NCtx)-gen-1)
		short := min(short, long/2)
		vocab := uint32(min(c.NVocab-1, 20000))
		prompt := func(i int) []int32 {
			n := short
			if i%2 == 0 {
				n = long
			}
			p := make([]int32, n)
			x := uint32(i)*2654435761 + 1
			for j := range p {
				x ^= x << 13
				x ^= x >> 17
				x ^= x << 5
				p[j] = 1 + int32(x%vocab)
			}
			return p
		}
		lp := gl.lm.loop
		invariant := gl.lm.gpuTier().RowsComposed()
		t.Logf("%d requests at once, %d each way, prompts of %d and %d tokens, %d out; steps one kernel for every row count: %v",
			conc, reqs, long, short, gen, invariant)
		one := func(i int) completion {
			return complete(gl.c, &v1.GenerateRequest{ModelId: "dev", Prompt: ids(prompt(i)...),
				MaxTokens: int32(gen), IgnoreEos: true})
		}
		answers := func(what string, out []completion) [][]int32 {
			got := make([][]int32, len(out))
			failed := 0
			for i, o := range out {
				switch {
				case o.err != nil:
					t.Errorf("%s, request %d: %v", what, i, o.err)
					failed++
				case len(o.ids) != gen:
					t.Errorf("%s, request %d: %d tokens, want %d", what, i, len(o.ids), gen)
					failed++
				}
				got[i] = o.ids
			}
			if failed > 0 {
				t.Fatalf("%s: %d of %d requests failed", what, failed, len(out))
			}
			// The devices still serve: a fault a step met leaves them unable to.
			if o := complete(gl.c, &v1.GenerateRequest{ModelId: "dev", Prompt: ids(prompt(1)...), MaxTokens: 4}); o.err != nil {
				t.Fatalf("%s: a request after it: %v", what, o.err)
			}
			return got
		}
		arm := func(groups int) (waves, loop [][]int32, steps int64) {
			lp.groupsCfg = groups
			steps0 := lp.stats.groupSteps.Load()
			s0, t0 := gl.lm.gpuTier().Stats(), time.Now()
			defer func() {
				s1 := gl.lm.gpuTier().Stats()
				t.Logf("groups %d: %s; captures %d, formed at a step %d, lanes %d (waited %d), KV grows %d, evictions %d, page-ins %d, stage overlaps %d",
					groups, time.Since(t0).Round(time.Millisecond), s1.Captures-s0.Captures, s1.FormedAtStep-s0.FormedAtStep,
					s1.Lanes, s1.LanesWaited-s0.LanesWaited, s1.KVGrows-s0.KVGrows, s1.KVEvictions-s0.KVEvictions,
					s1.PageIns-s0.PageIns, s1.StageOverlaps-s0.StageOverlaps)
			}()
			out := make([]completion, conc*reqs)
			for w := range reqs {
				release := holdGates(t, gl.e, gl.lm)
				var wg sync.WaitGroup
				for k := range conc {
					wg.Add(1)
					go func() {
						defer wg.Done()
						out[w*conc+k] = one(w*conc + k)
					}()
				}
				waitFor(t, "every request to wait for a row", func() bool { return waitingFor(lp) == conc })
				release()
				wg.Wait()
			}
			waves = answers(fmt.Sprintf("groups %d in waves", groups), out)
			out = make([]completion, conc*reqs)
			var wg sync.WaitGroup
			for k := range conc {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for w := range reqs {
						out[k*reqs+w] = one(k*reqs + w)
					}
				}()
			}
			wg.Wait()
			loop = answers(fmt.Sprintf("groups %d in a closed loop", groups), out)
			return waves, loop, lp.stats.groupSteps.Load() - steps0
		}
		oneW, oneL, oneSteps := arm(1)
		if oneSteps != 0 {
			t.Fatalf("one group ran %d grouped step(s)", oneSteps)
		}
		groups := -1
		if !invariant {
			groups = 2
		}
		if v := os.Getenv("JITLLM_SERVER_GROUP_COUNT"); v != "" {
			if _, err := fmt.Sscanf(v, "%d", &groups); err != nil {
				t.Fatalf("JITLLM_SERVER_GROUP_COUNT=%q: %v", v, err)
			}
		}
		gotW, gotL, steps := arm(groups)
		if steps == 0 {
			t.Fatal("the groups arm ran no grouped step")
		}
		parted := func(got, want [][]int32, hold bool) int {
			n := 0
			for i := range got {
				if slices.Equal(got[i], want[i]) {
					continue
				}
				n++
				j := 0
				for j < gen && got[i][j] == want[i][j] {
					j++
				}
				if !hold {
					continue
				}
				// A step's attention plan follows its rows' longest history
				// (pagedKeys), so a group's reduction can differ from one
				// group's at a near tie: the host, in neither arm, says
				// whether the two tokens were one.
				if gap := hostGap(t, gl.lm.m, prompt(i), got[i][:j], got[i][j], want[i][j]); gap > tieMargin {
					t.Errorf("request %d parts at token %d, %.3f apart on the host: %v in groups, %v in one group",
						i, j, gap, got[i][j:], want[i][j:])
				} else {
					t.Logf("request %d parts at token %d, a tie on the host (%.3f)", i, j, gap)
				}
			}
			return n
		}
		t.Logf("%d grouped steps; requests parting from one group's tokens: %d of %d in waves, %d of %d in a closed loop",
			steps, parted(gotW, oneW, invariant), len(gotW), parted(gotL, oneL, false), len(gotL))
	})
}

// loadUnderLoad loads name over devs as jitllmd does, its first half of
// blocks on devs[0] and the rest on devs[1] (JITLLM_SERVER_GROUP_PLACEMENT
// names another placement, "auto" the devices' own), every step that can be
// joint joint, room for conc sessions of maxSeq positions, and requires a
// session's blocks on both devices.
func loadUnderLoad(t *testing.T, name string, devs []string, conc, maxSeq int) *groupedLoad {
	t.Helper()
	path := modelPath(t, name)
	o := LoadOptions{Path: path, ModelID: "dev", DeviceIDs: devs, Sessions: conc + 1}
	switch p := os.Getenv("JITLLM_SERVER_GROUP_PLACEMENT"); p {
	case "auto":
	case "":
		probe := New(Config{Version: "test", NoMemCache: true})
		lm, err := probe.LoadModel(LoadOptions{Path: path, ModelID: "dev", DeviceIDs: devs})
		if err != nil {
			probe.Close()
			t.Fatalf("loading %s onto %s: %v", name, strings.Join(devs, ","), err)
		}
		nb := lm.m.Cfg.NLayer
		probe.Close()
		o.Placement = fmt.Sprintf("0-%d=%s,%d-%d=%s", nb/2-1, devs[0], nb/2, nb-1, devs[1])
		o.PlacementStrict = true
	default:
		o.Placement, o.PlacementStrict = p, true
	}
	e := New(Config{Version: "test", NoMemCache: true, DefaultMaxSeq: maxSeq, JointSteps: JointAlways,
		MaxBatchRows: conc, NoOrchestrate: true})
	t.Cleanup(e.Close)
	lm, err := e.LoadModel(o)
	if err != nil {
		t.Fatalf("loading %s onto %s (%q): %v", name, strings.Join(devs, ","), o.Placement, err)
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
		t.Fatalf("a session's blocks are on %d device(s), want 2 (%q), and its step refusal is %v", d, o.Placement, why)
	}
	return &groupedLoad{e: e, lm: lm, c: serveEngine(t, e), n: conc}
}
