package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// TestSplitPiecesReadTheirSessionsRows: synth-gemma4-e -- per-layer inputs
// and blocks at two geometries -- with every block on two cards, the lower
// blocks on one and the upper ones and the head on the other, so a prompt
// chunk wider than a submission runs in pieces, each through every run
// before the next (GPU.pieces). Two instances share the cards, and the
// second's 1300-token prompt is held to the CPU's logits: each piece reads
// its own rows' per-layer inputs, the calling session's and not the
// device's own. The band is the same prompt's on the first card alone,
// where a chunk is one submission wide and nothing goes in pieces;
// Stats.SplitPieces counts the pieces, or the gate saw none.
func TestSplitPiecesReadTheirSessionsRows(t *testing.T) {
	m, _, io := openGemma4(t, "synth-gemma4-e")
	defer m.Close()
	if m.Cfg.SideWidth() == 0 || !m.Cfg.MixedGeom() {
		t.Fatalf("synth-gemma4-e states side width %d and mixed geometry %v", m.Cfg.SideWidth(), m.Cfg.MixedGeom())
	}
	const n = 1300
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32((i*7919 + 13) % 50000)
	}
	want, err := testInstance(t, m, n+8, io...).Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)
	nmse := func(got []float32) float64 {
		var num, den float64
		for i := range want {
			d := float64(got[i] - want[i])
			num, den = num+d*d, den+float64(want[i])*float64(want[i])
		}
		if den == 0 || math.IsNaN(num) {
			return math.Inf(1)
		}
		return num / den
	}
	a, b := moveDevices(t)
	nl := m.Cfg.NLayer

	g1, err := openGPU(t, gpu.WithDeviceTune(gpu.TuneOff), gpu.WithDevices(a))
	if err != nil || g1 == nil {
		t.Fatalf("no GPU device: %v", err)
	}
	defer g1.Close()
	one := testInstance(t, m, n+8, io...)
	if err := one.SetDevice(g1); err != nil {
		t.Fatal(err)
	}
	if one.GPULayers() != nl {
		t.Fatalf("one card took %d of %d blocks: %s", one.GPULayers(), nl, g1.Err())
	}
	lg, err := one.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	band := nmse(lg)
	if splitPieces(g1) != 0 {
		t.Fatalf("one card ran %d pieces: the band is not a chunk one submission wide", splitPieces(g1))
	}

	g, err := openGPU(t, gpu.WithDeviceTune(gpu.TuneOff), gpu.WithDevices(a+","+b))
	if err != nil || g == nil {
		t.Fatalf("no GPU device: %v", err)
	}
	defer g.Close()
	distinctCards(t, g)
	// The second card from the first block whose history a block of the
	// upper half reads: a block sharing another's history runs beside it.
	lo := nl / 2
	for li := nl / 2; li < nl; li++ {
		lo = min(lo, kvSourceOf(m, li))
	}
	up := map[int]Place{}
	for li := lo; li < nl; li++ {
		up[li] = Place{On: b}
	}
	opts := append(append([]Option(nil), io...),
		WithPlacement(Placement{Blocks: up, Rest: &Place{On: a}, Head: &Place{On: b}, Strict: true}))
	first, second := testInstance(t, m, n+8, opts...), testInstance(t, m, n+8, opts...)
	for _, s := range []*Instance{first, second} {
		if err := s.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if s.GPULayers() != nl {
			t.Fatalf("the cards took %d of %d blocks: %s", s.GPULayers(), nl, g.Err())
		}
	}
	if _, err := first.Prefill(ids[:40]); err != nil {
		t.Fatal(err)
	}
	before := splitPieces(g)
	lg, err = second.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	ran := splitPieces(g) - before
	if ran == 0 {
		piped := 0
		for _, st := range g.DevStats() {
			piped += st.PipelinePieces
		}
		t.Fatalf("no piece ran (%d pipelined pieces, placed %v): the prompt never went in pieces over the two geometries", piped, g.Placed())
	}
	got := nmse(lg)
	if bound := max(3*band, 1e-9); got > bound || Greedy(lg) != Greedy(want) {
		t.Fatalf("the second instance's prompt in pieces stands at NMSE %.3e from the CPU's (greedy %d, the CPU's %d), "+
			"above %.3e, three times one card's %.3e", got, Greedy(lg), Greedy(want), bound, band)
	}
	t.Logf("%d pieces: NMSE %.3e from the CPU's, one card's %.3e", ran, got, band)
}

// splitPieces is the pieces over two geometries g's devices ran.
func splitPieces(g *gpu.GPU) int {
	n := 0
	for _, s := range g.DevStats() {
		n += s.SplitPieces
	}
	return n
}
