package engine

import (
	"fmt"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// groupedStepAllocs is the server's decode groups (server/batchgroups.go) as
// TestDecodeDoesNotAllocate's arm: the model split over two devices, four
// sessions in two groups, each group stepped by StepJoint on a long-lived
// goroutine of its own, the groups not waiting for each other. A warm step
// allocates nothing on either goroutine, and the tier's StageOverlaps says
// the groups ran on the two cards at once. Two physical cards where
// JITLLM_TEST_DEVICES names them.
func groupedStepAllocs(t *testing.T, path string, mopts ...model.Option) {
	d := testdevices.Take(t, 2, "cuda:0,vulkan:1")
	g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(d[0] + "," + d[1])}, testTierOpts(t)...)...)
	if err != nil || g == nil {
		noDevice(t, d[0]+","+d[1], err)
	}
	defer g.Close()
	probe, err := model.Open(jlmOf(t, path), mopts...)
	if err != nil {
		t.Fatal(err)
	}
	nb := probe.Cfg.NLayer
	probe.Close()
	first := map[int]Place{}
	for li := range nb / 2 {
		first[li] = Place{On: d[0]}
	}
	instOpts2 := append([]Option(nil), WithStreamTrial(false),
		WithPlacement(Placement{Blocks: first, Rest: &Place{On: d[1]}, Strict: true}))
	m, err := model.Open(jlmOf(t, path), mopts...)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompts := []string{"The capital of France is", "Water boils at a temperature of",
		"The clerk counted barrels of salt on the upper floor, and", "Once upon a time"}
	const warm, n = 128, 64
	groups := [2][]Run{}
	toks := make([][1]int32, len(prompts))
	for i, pr := range prompts {
		ids := m.Vocab.Encode(pr, true)
		st := testInstance(t, m, len(ids)+warm+n+8, instOpts2...)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if st.GPULayers() != nb || !st.HeadOnDevice() {
			t.Fatalf("%d of %d blocks on the devices, head %v: a grouped step needs them all",
				st.GPULayers(), nb, st.HeadOnDevice())
		}
		if _, err := st.Prefill(ids); err != nil {
			t.Fatal(err)
		}
		groups[i%2] = append(groups[i%2], Run{Instance: st, Tokens: toks[i][:], Logits: true})
	}
	if depth := g.PipelineDepth(); depth != 2 {
		t.Fatalf("the blocks are on %d device(s), want 2", depth)
	}
	// Each group's worker runs steps of its own, the count its start says,
	// and reports on done: long-lived, as the server's are.
	var start [2]chan int
	var done sync.WaitGroup
	var fail [2]error
	for k := range groups {
		start[k] = make(chan int, 1)
		go func() {
			for steps := range start[k] {
				for s := range steps {
					for _, r := range groups[k] {
						r.Tokens[0] = int32(3 + (s+k)%64)
					}
					if _, err := StepJoint(groups[k]); err != nil && fail[k] == nil {
						fail[k] = err
					}
				}
				done.Done()
			}
		}()
	}
	defer func() {
		for k := range start {
			close(start[k])
		}
	}()
	cycle := func(steps int) {
		done.Add(len(start))
		for k := range start {
			start[k] <- steps
		}
		done.Wait()
	}
	cycle(warm)
	for k, err := range fail {
		if err != nil {
			t.Fatalf("group %d: %v", k, err)
		}
	}
	if err := prefaultExperts(m); err != nil {
		t.Fatal(err)
	}
	c0 := g.Stats()
	r0 := m.Container.Reads()
	w := countAllocs(func() { cycle(n) })
	c1 := g.Stats()
	// Filling the pipeline, nearly every stage begins beside the other
	// group's (127 of 128 on the laptop); half a group's steps is the floor.
	if over := c1.StageOverlaps - c0.StageOverlaps; over < n/2 {
		t.Fatalf("%d stage(s) of one group began while the other ran on the other card, over %d steps each", over, n)
	} else {
		t.Logf("%d stage overlap(s) over %d steps a group", over, n)
	}
	if rows := c1.SessionRows - c0.SessionRows; rows == 0 {
		t.Fatalf("no row stepped across sessions: %s", declinesOf(c1))
	}
	allocVerdict(t, fmt.Sprintf("2 groups of 2 runs on %s and %s", d[0], d[1]), w, 2*n,
		m.Container.Reads()-r0, c1.Captures-c0.Captures)
}

// groupedCensus steps four sessions on g as the server's decode groups do --
// two groups of two, each group's steps on a goroutine of its own, the groups
// not waiting for each other -- for TestPlacementComposesOnce's census: rows
// are a run-time argument of every composed form, so groups of any width
// compose nothing.
func groupedCensus(t *testing.T, m *model.Model, g *gpu.GPU, ids []int32, io ...Option) {
	t.Helper()
	const steps = 16
	var groups [2][]Run
	toks := make([][1]int32, 4)
	for i := range toks {
		st := testInstance(t, m, 256, io...)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		l, err := st.Prefill(ids[i*20 : i*20+9+i])
		if err != nil {
			t.Fatal(err)
		}
		toks[i][0] = argmaxID(l)
		groups[i%2] = append(groups[i%2], Run{Instance: st, Tokens: toks[i][:], Logits: true})
	}
	o0 := g.Stats().StageOverlaps
	var wg sync.WaitGroup
	var fail [2]error
	for k := range groups {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range steps {
				out, err := StepJoint(groups[k])
				if err != nil {
					fail[k] = err
					return
				}
				for j, r := range groups[k] {
					r.Tokens[0] = argmaxID(out[j])
				}
			}
		}()
	}
	wg.Wait()
	for k, err := range fail {
		if err != nil {
			t.Fatalf("group %d: %v", k, err)
		}
	}
	t.Logf("two groups of two, %d steps each: %d stage overlap(s)", steps, g.Stats().StageOverlaps-o0)
}
