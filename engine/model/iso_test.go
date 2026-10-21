//go:build linux

package model

import (
	"sync"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// isoSetup opens the model and every device on the host.
func isoSetup(t *testing.T) (*Model, *tier.GPU) {
	t.Helper()
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := Open(path)
	if err != nil {
		t.Skip(err)
	}
	g, err := openTier(t, tier.WithDevices("all"), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		m.Close()
		noDevice(t, "devices", err)
	}
	return m, g
}

// TestTwoStatesOnOneTierShareAKVCache: two States attached to the same tier.GPU
// must not corrupt each other. Layers takes a position but no state identity,
// so each session needs its own device KV cache (tier.GPU.Attach); interleaved
// States must match the same sequences run alone.
func TestTwoStatesOnOneTierShareAKVCache(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := Open(path)
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	seqA := []int32{101, 202, 303, 404, 505, 606}
	seqB := []int32{909, 808, 707, 606, 505, 404}

	g, err := openTier(t, tier.WithDevices("gpu:0"), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()

	// The reference is the device running one sequence, not the host: host and
	// device differ by f32 reduction order, which would fold noise into the
	// thing under test.
	ref := func(seq []int32) []float32 {
		s := m.NewState(32)
		defer s.Close()
		s.SetDevice(g)
		var lg []float32
		for _, id := range seq {
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return append([]float32(nil), lg...)
	}
	wantA, wantB := ref(seqA), ref(seqB)

	a := m.NewState(32)
	defer a.Close()
	b := m.NewState(32)
	defer b.Close()
	a.SetDevice(g)
	b.SetDevice(g)
	t.Logf("A placed %d blocks, B placed %d blocks", a.GPULayers(), b.GPULayers())
	if a.GPULayers() == 0 {
		t.Skip("no blocks placed; nothing to share")
	}

	// Interleaved, A, B, A, B, as two sessions on one engine run.
	var gotA, gotB []float32
	for i := range seqA {
		if gotA, err = a.Forward(seqA[i]); err != nil {
			t.Fatal(err)
		}
		if gotB, err = b.Forward(seqB[i]); err != nil {
			t.Fatal(err)
		}
	}

	nmse := func(got, want []float32) float64 {
		var num, den float64
		for i := range want {
			d := float64(got[i] - want[i])
			num += d * d
			den += float64(want[i]) * float64(want[i])
		}
		if den == 0 {
			return 0
		}
		return num / den
	}
	na, nb := nmse(gotA, wantA), nmse(gotB, wantB)
	t.Logf("interleaved on ONE tier: NMSE against the solo run  A=%.3e  B=%.3e", na, nb)
	t.Logf("greedy: A got %d want %d   B got %d want %d",
		Greedy(gotA), Greedy(wantA), Greedy(gotB), Greedy(wantB))

	// Bit equality is unavailable (the split is measured per process), but 1e-6
	// is far below what a shared cache produces (~2e-1).
	const bound = 1e-6
	if na > bound || nb > bound {
		t.Errorf("TWO STATES ON ONE tier.GPU CORRUPT EACH OTHER: NMSE A=%.3e B=%.3e "+
			"against a %.0e bound. The device KV cache is per-BLOCK (l.kc/l.vc) and "+
			"nn.LayerDevice.Layers carries no state identity, so both sequences write "+
			"the same cache at the same positions.", na, nb, bound)
	}
}

// TestConcurrentPlacementPlacesAsManyBlocks: attaching from several goroutines
// must place what attaching serially places.
//
// The assertion is the count: States that lose the race run on the host with
// every answer right.
// BeginPlacement/EndPlacement bracket the whole offer sequence so each State
// sees a budget nobody else is spending.
func TestConcurrentPlacementPlacesAsManyBlocks(t *testing.T) {
	m, g := isoSetup(t)
	defer m.Close()
	defer g.Close()

	solo := m.NewState(32)
	solo.SetDevice(g)
	want := solo.GPULayers()
	solo.Close()
	if want == 0 {
		t.Skip("no blocks placed even alone")
	}

	const n = 4
	ss := make([]*State, n)
	for i := range ss {
		ss[i] = m.NewState(32)
		defer ss[i].Close()
	}
	var wg sync.WaitGroup
	for i := range ss {
		wg.Add(1)
		go func(i int) { defer wg.Done(); ss[i].SetDevice(g) }(i)
	}
	wg.Wait()

	for i, s := range ss {
		got := s.GPULayers()
		t.Logf("state %d placed %d blocks (solo places %d)", i, got, want)
		// A later session may get fewer blocks (its KV cache must fit
		// beside the others'), but not one where the budget held many.
		if got == 0 || (want > 4 && got < 2) {
			t.Errorf("state %d placed %d of a possible %d: the offer sequences "+
				"interleaved and each saw a budget the others were spending", i, got, want)
		}
	}
}
