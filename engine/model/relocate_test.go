package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// relocationModel opens the model the relocation gates run, and measures what a
// tier holds with every block placed and one page of history -- the budget both
// gates start from.
func relocationModel(t *testing.T) (*Model, []int32, uint64) {
	t.Helper()
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	// No KV prewarm: these gates squeeze the budget to what attaching
	// spends and need the history's growth to outrun it, which pages grown
	// ahead into the attach's room would cover.
	m, err := Open(p, noTune, WithKVPrewarm(false))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	ids := m.Vocab.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 200), true)[:relocN]
	probe, err := tier.OpenWith(tier.WithDeviceTune(tier.TuneOff), onePage)
	if err != nil || probe == nil {
		noDevice(t, "device", err)
	}
	defer probe.Close()
	st := m.NewState(swaMaxSeq)
	defer st.Close()
	st.SetDevice(probe)
	if st.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("placed %d of %d blocks unconstrained: %q", st.GPULayers(), m.Cfg.NLayer, probe.Err())
	}
	if _, err := st.Forward(ids[0]); err != nil {
		t.Fatal(err)
	}
	return m, ids, probe.Bytes()
}

// onePage makes a device reserve one page of history (tier.Config.KVReserve)
// rather than the context it is asked for, so a context past it has to GROW --
// which is what these gates exist to exercise.
var onePage = tier.WithConfig(func(c *tier.Config) { c.KVReserve = relocPage })

// relocN positions run past relocPage, the tier's first capacity (onePage),
// and relocTail of them run after the second gate hands the device back.
const relocN, relocPage, relocTail = 336, 256, 16

// relocTier is a device with room for every block and one page of history: the
// doubling (~1 MiB a block on this model) does not fit.
func relocTier(t *testing.T, used uint64) *tier.GPU {
	t.Helper()
	g, err := tier.OpenWith(tier.WithDeviceTune(tier.TuneOff), tier.WithBudget(used+2<<20), onePage)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	return g
}

// TestRelocateKeepsTheDeviceWhenTheContextOutgrowsIt gives a device room for
// every block and one page of history, runs past that page, and holds the run
// to the host. The control is the same budget without relocation, which must
// demote (so the budget bites). A relocated block whose history did not follow
// it attends over an empty host cache, far outside the device's band.
func TestRelocateKeepsTheDeviceWhenTheContextOutgrowsIt(t *testing.T) {
	m, ids, used := relocationModel(t)
	const page = relocPage
	host := swaRun(t, m, nil, ids, false)

	// arm is what one run leaves behind: its tier and State are closed before
	// the next opens, because a 4 GiB card holds one copy of this model.
	type arm struct {
		out                      [][]float32
		relocated, demoted, kept int
	}
	run := func(relocate, batched bool) arm {
		g := relocTier(t, used)
		defer g.Close()
		st := m.NewState(swaMaxSeq)
		defer st.Close()
		st.SetDevice(g)
		if st.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("placed %d of %d blocks at the probed budget: %q", st.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		st.SetRelocate(relocate)
		keep := func(lg []float32) []float32 { return append([]float32(nil), lg[:min(len(lg), swaSlice)]...) }
		var out [][]float32
		if batched {
			// 128-row chunks: the third asks for the second page with 256
			// positions already written, so a relocated block has history.
			lg, err := st.Prefill(ids)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, keep(lg))
		} else {
			for _, id := range ids {
				lg, err := st.Forward(id)
				if err != nil {
					t.Fatal(err)
				}
				out = append(out, keep(lg))
			}
		}
		return arm{out, st.Relocations(), st.DeviceDemotions(), st.GPULayers()}
	}
	for _, batched := range []bool{false, true} {
		href := host
		if batched {
			href = host[len(host)-1:]
		}
		// The device's own band: every block placed, nothing constrained.
		gb := swaTier(t)
		band := swaMedian(swaRun(t, m, gb, ids, batched), href, page)
		gb.Close()
		bound := max(1e-3, 10*band)
		ctl := run(false, batched)
		if ctl.demoted == 0 {
			t.Fatalf("batched=%v: without relocation the budget demoted nothing (%d of %d blocks "+
				"kept): it does not bite and this gate would prove nothing", batched, ctl.kept, m.Cfg.NLayer)
		}
		// The demotion itself must be right: a growth refused halfway must not
		// leave blocks read home at the wrong stride.
		if c := swaMedian(ctl.out, href, page); c > bound {
			t.Errorf("batched=%v: after the demotion the host run reads %.2e against the host "+
				"(bound %.2e): the history it brought home was read wrong", batched, c, bound)
		}
		a := run(true, batched)
		d := swaMedian(a.out, href, page)
		t.Logf("batched=%v: %d block(s) relocated, %d of %d kept; past the page device-vs-host %.2e (band %.2e)",
			batched, a.relocated, a.kept, m.Cfg.NLayer, d, band)
		switch {
		case a.demoted != 0:
			t.Fatalf("batched=%v: the device was demoted %d time(s) with relocation on", batched, a.demoted)
		case a.relocated == 0 || a.kept != m.Cfg.NLayer-a.relocated:
			t.Fatalf("batched=%v: %d relocation(s) and %d of %d blocks on the device",
				batched, a.relocated, a.kept, m.Cfg.NLayer)
		case a.kept == 0:
			t.Fatalf("batched=%v: relocation emptied the device: that is a demotion by another name", batched)
		}
		if d > bound {
			t.Errorf("batched=%v: device-vs-host %.2e past the page against a bound of %.2e: the "+
				"relocated block's history did not come home, or the device's did not survive the "+
				"growth", batched, d, bound)
		}
	}
}

// refusing is a tier made to answer the way the FIRST of two cards does: its
// refusals name blocks [0, len(firstCard)) whatever the card actually holds.
// One card is enough to exercise what two would, and it runs on every host.
type refusing struct {
	*tier.GPU
	firstCard []int
}

func (r *refusing) Attach() nn.LayerDevice { return &refusingView{r.GPU.Attach(), r} }

type refusingView struct {
	nn.LayerDevice
	r *refusing
}

func (v *refusingView) Refused() []int {
	if len(v.r.GPU.Refused()) == 0 {
		return nil
	}
	return v.r.firstCard
}

// TestRelocationTakesFromTheDeviceThatRefused runs the relocation gate's budget
// on a tier whose refusals come from the first half of the model, then hands
// the device back and finishes on the host. The block that moves must be the
// refusing card's highest (the model's top block frees nothing on that card).
// The placement then has a hole, so shrinking must bring home the history of
// blocks past it too, not only the prefix up to gpuLayers.
func TestRelocationTakesFromTheDeviceThatRefused(t *testing.T) {
	m, ids, used := relocationModel(t)
	host := swaRun(t, m, nil, ids, false)
	half := m.Cfg.NLayer / 2
	first := make([]int, half)
	for i := range first {
		first[i] = i
	}
	g := relocTier(t, used)
	defer g.Close()
	st := m.NewState(swaMaxSeq)
	defer st.Close()
	st.SetDevice(&refusing{GPU: g, firstCard: first})
	if st.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("placed %d of %d blocks: %q", st.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	st.SetRelocate(true)
	var out [][]float32
	for i, id := range ids {
		if i == relocN-relocTail {
			if st.Relocations() != 1 || st.devAt(half-1) || !st.devAt(m.Cfg.NLayer-1) || st.devCount() != m.Cfg.NLayer-1 {
				t.Fatalf("%d relocation(s); block %d placed %v, block %d placed %v, %d of %d on the "+
					"device -- want the refusing card's highest block, and only it, on the host",
					st.Relocations(), half-1, st.devAt(half-1), m.Cfg.NLayer-1,
					st.devAt(m.Cfg.NLayer-1), st.devCount(), m.Cfg.NLayer)
			}
			if st.SetGPULayers(0); st.devCount() != 0 {
				t.Fatalf("SetGPULayers(0) left %d blocks on the device", st.devCount())
			}
		}
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg[:min(len(lg), swaSlice)]...))
	}
	gb := swaTier(t)
	band := swaMedian(swaRun(t, m, gb, ids, false), host, relocPage)
	gb.Close()
	bound := max(1e-3, 10*band)
	hole := swaMedian(out[:relocN-relocTail], host[:relocN-relocTail], relocPage)
	tail := swaMedian(out[relocN-relocTail:], host[relocN-relocTail:], 0)
	t.Logf("block %d relocated around a hole: %.2e past the page; after handing the device back %.2e (band %.2e)",
		half-1, hole, tail, band)
	if hole > bound {
		t.Errorf("with a host block between two device runs: %.2e against a bound of %.2e", hole, bound)
	}
	if tail > bound {
		t.Errorf("after SetGPULayers(0): %.2e against a bound of %.2e -- the blocks past the hole "+
			"did not bring their history home", tail, bound)
	}
}

// TestRelocatedBlocksComeBack relocates a block past the first page, then gives
// the card room two ways and holds each run to the host: a shorter sequence
// (Reset and a 64-token prompt; the block returns before the first chunk), and
// a budget raised mid-sequence (the block returns at the next token and must
// bring the history it built on the host).
func TestRelocatedBlocksComeBack(t *testing.T) {
	defer fusedRunsCheck(t, true)()
	m, ids, used := relocationModel(t)
	host := swaRun(t, m, nil, ids, false)
	gb := swaTier(t)
	band := swaMedian(swaRun(t, m, gb, ids, false), host, relocPage)
	gb.Close()
	bound := max(1e-3, 10*band)
	keep := func(lg []float32) []float32 { return append([]float32(nil), lg[:min(len(lg), swaSlice)]...) }

	t.Run("shorter-prompt", func(t *testing.T) {
		g := relocTier(t, used)
		defer g.Close()
		st := m.NewState(swaMaxSeq)
		defer st.Close()
		st.SetDevice(g)
		st.SetRelocate(true)
		for _, id := range ids[:relocPage+16] {
			if _, err := st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		if st.Relocations() != 1 || st.GPULayers() != m.Cfg.NLayer-1 {
			t.Fatalf("%d relocation(s), %d of %d on the device before the shorter prompt",
				st.Relocations(), st.GPULayers(), m.Cfg.NLayer)
		}
		st.Reset()
		const short, more = 64, 32
		lg, err := st.Prefill(ids[:short])
		if err != nil {
			t.Fatal(err)
		}
		if st.Reclaims() != 1 || st.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("after a %d-token prompt: %d reclaim(s), %d of %d on the device (%q)",
				short, st.Reclaims(), st.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		out := [][]float32{keep(lg)}
		for _, id := range ids[short : short+more] {
			lg, err := st.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, keep(lg))
		}
		d := swaMedian(out, host[short-1:short+more], 0)
		t.Logf("every block back for a %d-token prompt; device-vs-host %.2e (band %.2e)", short, d, band)
		if d > bound {
			t.Errorf("device-vs-host %.2e against a bound of %.2e", d, bound)
		}
	})

	t.Run("budget-raised", func(t *testing.T) {
		g := relocTier(t, used)
		defer g.Close()
		st := m.NewState(swaMaxSeq)
		defer st.Close()
		st.SetDevice(g)
		st.SetRelocate(true)
		const raise = relocN - relocTail
		var out [][]float32
		for i, id := range ids {
			if i == raise {
				if st.Relocations() != 1 {
					t.Fatalf("%d relocation(s) before the budget was raised", st.Relocations())
				}
				g.SetBudget(used + 1<<30)
			}
			lg, err := st.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, keep(lg))
		}
		if st.Reclaims() != 1 || st.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("after the budget was raised: %d reclaim(s), %d of %d on the device (%q)",
				st.Reclaims(), st.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		d := swaMedian(out[raise:], host[raise:], 0)
		t.Logf("block back mid-sequence at position %d; device-vs-host after it %.2e (band %.2e)", raise, d, band)
		if d > bound {
			t.Errorf("device-vs-host %.2e after the reclaim, against a bound of %.2e: the block came "+
				"back without its history", d, bound)
		}
	})
}

// TestFusedRunsFollowARelocation: a block relocated off the card mid-sequence
// and reclaimed later changes which blocks a fused run (fuse.go) bridges, and
// the fused decode keeps the per-op decode's answer through both moves, to the
// bit. Each arm runs the same schedule on its own tier.
func TestFusedRunsFollowARelocation(t *testing.T) {
	m, ids, used := relocationModel(t)
	const raise = relocN - relocTail
	run := func(fused bool) ([][]float32, int64) {
		g := relocTier(t, used)
		defer g.Close()
		st := m.NewState(swaMaxSeq)
		defer st.Close()
		st.SetBlockFusion(fused)
		st.SetDevice(g)
		st.SetRelocate(true)
		var out [][]float32
		for i, id := range ids {
			if i == raise {
				g.SetBudget(used + 1<<30)
			}
			lg, err := st.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		if st.Relocations() != 1 || st.Reclaims() != 1 {
			t.Fatalf("fused=%v: %d relocation(s), %d reclaim(s): the schedule did not move a block both ways",
				fused, st.Relocations(), st.Reclaims())
		}
		return out, st.FusedRuns
	}
	per, n0 := run(false)
	fu, n1 := run(true)
	if n0 != 0 || n1 == 0 {
		t.Fatalf("fused runs: per-op arm %d, fused arm %d", n0, n1)
	}
	for i := range per {
		if Greedy(per[i]) != Greedy(fu[i]) {
			t.Fatalf("position %d: fused token %d, per op %d", i, Greedy(fu[i]), Greedy(per[i]))
		}
		for j := range per[i] {
			if math.Float32bits(per[i][j]) != math.Float32bits(fu[i][j]) {
				t.Fatalf("position %d logit %d: fused %v, per op %v", i, j, fu[i][j], per[i][j])
			}
		}
	}
	t.Logf("%d positions through a relocation and a reclaim, same bits; %d fused runs", len(ids), n1)
}
