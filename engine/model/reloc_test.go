package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestSeamMovesWhileServing moves blocks between host and device mid-sequence
// and requires the same tokens as a static placement. It guards the KV cache
// following the block, a captured graph outliving its placement, and kernels
// assuming a fixed output buffer.
//
// It compares migrated against static on one tier, not device against host
// (they legitimately part on knife-edge argmaxes). Both arms share one
// tier.GPU so the split tuner's reduction order is the same.
func TestSeamMovesWhileServing(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))

	g, err := tier.Open(0)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()

	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const ntok = 10
	in := m.Vocab.Encode("The capital of France is", true)

	// run decodes ntok tokens greedily. move, when set, is called before each
	// step and returns the seam it wants; the block count actually reached is
	// recorded so the test can prove the seam MOVED rather than assuming it.
	// margin[i] is the static arm's gap between its top two logits at token
	// i: where it is a tie, host and device may pick either.
	var margin []float32
	run := func(move func(i, full int) int) ([]int32, []int) {
		st := m.NewState(len(in) + ntok + 1)
		defer st.Close()
		st.SetDevice(g)
		full := st.GPULayers()
		if full == 0 {
			return nil, nil
		}
		logits, err := st.Prefill(in)
		if err != nil {
			t.Fatal(err)
		}
		seen := []int{full}
		out := make([]int32, 0, ntok)
		for i := 0; i < ntok; i++ {
			best, bv, second := int32(0), float32(-1e30), float32(-1e30)
			for j, v := range logits {
				if v > bv {
					best, bv, second = int32(j), v, bv
				} else if v > second {
					second = v
				}
			}
			out = append(out, best)
			if move == nil {
				margin = append(margin, bv-second)
			}
			if move != nil {
				// Moved mid-sequence, so there is KV history to travel.
				got := st.SetGPULayers(move(i, full))
				seen = append(seen, got)
			}
			if logits, err = st.Forward(best); err != nil {
				t.Fatal(err)
			}
		}
		return out, seen
	}

	static, _ := run(nil)
	if static == nil {
		t.Skip("the device took no blocks: there is no seam to move")
	}

	// Down to zero and back up, so both directions run and the final placement
	// is the one it started from.
	walk := func(i, full int) int {
		return []int{full / 2, 0, full / 4, full, full / 2, full}[i%6]
	}
	moved, seen := run(walk)

	// Assert the seam actually moved: SetGPULayers returns what it reached, and
	// a tier that ignored every request would match trivially.
	distinct := map[int]bool{}
	for _, n := range seen {
		distinct[n] = true
	}
	if len(distinct) < 3 {
		t.Fatalf("the seam did not move: block counts %v over the walk, want at "+
			"least three distinct placements", seen)
	}
	if !distinct[0] {
		t.Errorf("the seam never reached ZERO device blocks, so the full "+
			"host round trip was never exercised: %v", seen)
	}
	t.Logf("%s: seam walked %v of %d blocks", g.Name(), seen, seen[0])

	if len(moved) != len(static) {
		t.Fatalf("migrated run produced %d tokens, static %d", len(moved), len(static))
	}
	// Up to the first difference: the seam walks to zero, so the moved arm runs
	// some steps on the host, and host and device legitimately part on a tie.
	// A difference anywhere else is the migration.
	for i := range static {
		if moved[i] == static[i] {
			continue
		}
		// 0.2, the bottom of the f32 reduction band, not tieMargin's 0.5:
		// this prompt has margins of 0.3-0.5 at four tokens, and a migration
		// that broke there must not read as a tie.
		if margin[i] < 0.2 {
			t.Logf("the arms part at token %d, a tie (margin %.3f): migrated %v, static %v", i, margin[i], moved, static)
			return
		}
		t.Fatalf("relocation CHANGED THE ANSWER at token %d (margin %.3f): migrated %v, static %v",
			i, margin[i], moved, static)
	}
}

// TestPagesMoveWhileADeviceHoldsBlocks is the mixed case host-only
// TestPagingDoesNotChangeTheAnswer cannot reach: the device holds half the
// blocks while the host pages the other half through two frames.
func TestPagesMoveWhileADeviceHoldsBlocks(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))

	g, err := tier.Open(0)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()

	const ntok = 8
	ids := func(frames, devLayers int) ([]int32, int) {
		m, err := Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		if frames > 0 {
			m.SetPageBudget(uint64(frames) * m.PageSize())
		}
		in := m.Vocab.Encode("The capital of France is", true)
		st := m.NewState(len(in) + ntok + 1)
		defer st.Close()
		st.SetDeviceLayers(g, devLayers)
		took := st.GPULayers()
		logits, err := st.Prefill(in)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]int32, 0, ntok)
		for i := 0; i < ntok; i++ {
			best, bv := int32(0), float32(-1e30)
			for j, v := range logits {
				if v > bv {
					best, bv = int32(j), v
				}
			}
			out = append(out, best)
			if logits, err = st.Forward(best); err != nil {
				t.Fatal(err)
			}
		}
		return out, took
	}

	// Half the blocks on the device so the other half must page on the host.
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	half := len(m.layers) / 2
	m.Close()

	resident, took := ids(0, half)
	if took == 0 {
		t.Skip("the device took no blocks")
	}
	// Two frames forces the host half to evict and re-read every token.
	paged, took2 := ids(2, half)
	if took2 != took {
		t.Fatalf("the device took %d blocks resident and %d paged; the arms are "+
			"not comparable", took, took2)
	}
	t.Logf("%s: %d device blocks, host half fully resident vs two frames", g.Name(), took)
	for i := range resident {
		if paged[i] != resident[i] {
			t.Fatalf("paging under a device CHANGED THE ANSWER at token %d: "+
				"paged %v, resident %v", i, paged, resident)
		}
	}
}

// TestPageBudgetResizesWhileServing resizes the page budget mid-sequence.
// SetPageBudget reseats every frame, so the weights move under an in-flight
// sequence while the KV does not; the tokens must match a static run.
func TestPageBudgetResizesWhileServing(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	const ntok = 10

	run := func(resize func(m *Model, i int) bool) ([]int32, int) {
		m, err := Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		in := m.Vocab.Encode("The capital of France is", true)
		st := m.NewState(len(in) + ntok + 1)
		defer st.Close()
		logits, err := st.Prefill(in)
		if err != nil {
			t.Fatal(err)
		}
		resizes := 0
		out := make([]int32, 0, ntok)
		for i := 0; i < ntok; i++ {
			best, bv := int32(0), float32(-1e30)
			for j, v := range logits {
				if v > bv {
					best, bv = int32(j), v
				}
			}
			out = append(out, best)
			if resize != nil && resize(m, i) {
				resizes++
			}
			if logits, err = st.Forward(best); err != nil {
				t.Fatal(err)
			}
		}
		return out, resizes
	}

	static, _ := run(nil)

	// Squeeze to two frames, open back up, squeeze again -- so both directions
	// run and a frame that was reused for another block has to be re-read.
	walk := func(m *Model, i int) bool {
		n := []int{2, 22, 3, 22, 2}[i%5]
		m.SetPageBudget(uint64(n) * m.PageSize())
		return true
	}
	resized, n := run(walk)
	if n == 0 {
		t.Fatal("the budget was never resized: this gate proved nothing")
	}
	t.Logf("%d live budget changes across %d tokens", n, ntok)
	for i := range static {
		if resized[i] != static[i] {
			t.Fatalf("resizing the page pool CHANGED THE ANSWER at token %d: "+
				"resized %v, static %v", i, resized, static)
		}
	}
}

// TestLayerRelocatesToAnotherDevice moves a sequence to a second tier after the
// first token and requires the same answer. It skips with the reason when the
// second tier takes no blocks.
func TestLayerRelocatesToAnotherDevice(t *testing.T) {
	// Every block is on a card here, so no host run exists to fuse; the
	// fused relocation is TestFusedRunsFollowARelocation.
	defer fusedRunsCheck(t, false)()
	for _, name := range relocModels() {
		t.Run(name, func(t *testing.T) { layerRelocatesToAnotherDevice(t, jlmOf(t, testmodels.Path(name))) })
	}
}

// layerRelocatesToAnotherDevice is TestLayerRelocatesToAnotherDevice on one model.
func layerRelocatesToAnotherDevice(t *testing.T, path string) {

	// Both tiers' split tuner off: each tier tunes its own splits, so two
	// tuned tiers can sum a matvec in different orders, and the two arms then
	// part at a knife-edge argmax whatever the relocation did (it read token
	// 7 of tinyllama either way on a V100, on main and on a branch alike).
	// An equality gate pins every knob its equality depends on (RULE 11c).
	a, err := openTier(t, tier.WithDeviceTune(tier.TuneOff))
	if err != nil || a == nil {
		noDevice(t, "device", err)
	}
	defer a.Close()
	b, err := openTier(t, tier.WithDeviceTune(tier.TuneOff))
	if err != nil || b == nil {
		noDevice(t, "second tier", err)
	}
	defer b.Close()

	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const ntok = 8
	in := m.Vocab.Encode("The capital of France is", true)

	// hop moves the sequence from tier a to tier b after the first token, so
	// the history has to follow.
	run := func(hop bool) ([]int32, []int) {
		st := m.NewState(len(in) + ntok + 1)
		defer st.Close()
		st.SetDevice(a)
		placed := []int{st.GPULayers()}
		logits, err := st.Prefill(in)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]int32, 0, ntok)
		for i := 0; i < ntok; i++ {
			best, bv := int32(0), float32(-1e30)
			for j, v := range logits {
				if v > bv {
					best, bv = int32(j), v
				}
			}
			out = append(out, best)
			if hop && i == 0 {
				st.SetDevice(b)
				placed = append(placed, st.GPULayers())
			}
			if logits, err = st.Forward(best); err != nil {
				t.Fatal(err)
			}
		}
		return out, placed
	}

	static, _ := run(false)
	hopped, placed := run(true)
	t.Logf("%s -> %s: blocks %v", a.Name(), b.Name(), placed)
	if len(placed) < 2 || placed[1] == 0 {
		t.Skipf("the second tier took no blocks (%v): cross-device relocation is "+
			"the missing PLURAL in Scope, not a regression here", placed)
	}
	for i := range static {
		if hopped[i] != static[i] {
			t.Fatalf("moving the sequence to a second tier CHANGED THE ANSWER at "+
				"token %d: hopped %v, static %v", i, hopped, static)
		}
	}
}
