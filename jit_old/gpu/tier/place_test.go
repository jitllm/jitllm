package tier

import (
	"math"
	"testing"
)

// measuredCover is the coverage curve measured on Qwen3-30B-A3B: the share of
// all expert selections served by the r hottest experts of a layer. It is the
// measurement rather than a synthetic power law, which could give an expert
// more than one use per token.
var measuredCover = []struct {
	r    int
	frac float64
}{{0, 0}, {1, .082}, {2, .147}, {4, .247}, {8, .393}, {16, .590},
	{24, .728}, {32, .822}, {48, .930}, {64, .978}, {96, 1}, {128, 1}}

// expertUses turns that cumulative curve into a per-expert probability, which
// is what Uses means: how often each byte of this unit is read per token.
// Every value is in [0,1] and they sum to NExpertUsed.
func expertUses(nExpert, nUsed int) []float64 {
	u := make([]float64, nExpert)
	for i := 1; i < len(measuredCover); i++ {
		lo, hi := measuredCover[i-1], measuredCover[i]
		if lo.r >= nExpert {
			break
		}
		each := (hi.frac - lo.frac) / float64(hi.r-lo.r) * float64(nUsed)
		for e := lo.r; e < hi.r && e < nExpert; e++ {
			u[e] = each
		}
	}
	return u
}

func qwen30B(nLayer, nExpert, nUsed int) []Unit {
	const (
		attnB   = 11_100_000 // q/k/v/o at Q4_K, packed
		routerB = 1_048_576  // F32
		expertB = 2_650_000  // gate+up+down for one expert, packed
	)
	eu := expertUses(nExpert, nUsed)
	var us []Unit
	for l := 0; l < nLayer; l++ {
		us = append(us,
			Unit{Layer: l, Kind: KindAttn, Bytes: attnB, Uses: 1},
			Unit{Layer: l, Kind: KindRouter, Bytes: routerB, Uses: 1})
		for e := 0; e < nExpert; e++ {
			us = append(us, Unit{Layer: l, Kind: KindExpert, Expert: e,
				Bytes: expertB, Uses: eu[e]})
		}
	}
	return us
}

// TestExpertUsesIsAProbability guards the fixture itself: a profile whose
// entries exceed 1 is not a routing distribution, and it silently inverts the
// ranking the policy under test is supposed to produce.
func TestExpertUsesIsAProbability(t *testing.T) {
	u := expertUses(128, 8)
	var sum float64
	for e, x := range u {
		if x < 0 || x > 1 {
			t.Fatalf("expert %d has Uses %.3f, outside [0,1]", e, x)
		}
		sum += x
	}
	if math.Abs(sum-8) > 1e-9 {
		t.Fatalf("uses sum to %.4f, want NExpertUsed = 8", sum)
	}
}

// TestPlaceBeatsWholeBlocks is the claim the whole design rests on: for the
// SAME device budget, ranking units by hotness per byte covers far more of a
// token than taking whole blocks until the budget runs out.
//
// The control is the policy it replaces, at the same budget, not an absolute
// number.
func TestPlaceBeatsWholeBlocks(t *testing.T) {
	units := qwen30B(48, 128, 8)
	const budget = 2_500_000_000

	got := Place(units, budget)

	// The incumbent: whole blocks, in order, until the budget is gone.
	var blockBytes uint64
	var perBlockUses float64
	for _, u := range units {
		if u.Layer != 0 {
			continue
		}
		blockBytes += u.Bytes
		perBlockUses += u.Uses * float64(u.Bytes)
	}
	nBlocks := int(budget / blockBytes)
	whole := float64(nBlocks) * perBlockUses / got.Offered

	t.Logf("budget %.2f GiB", float64(budget)/(1<<30))
	t.Logf("  whole blocks: %d of 48, covers %.1f%% of per-token bytes",
		nBlocks, 100*whole)
	t.Logf("  by hotness:   %d units, %.2f GiB, covers %.1f%%",
		len(got.Resident), float64(got.Bytes)/(1<<30), 100*got.Covered)

	if got.Covered <= whole {
		t.Fatalf("hotness placement covers %.1f%%, whole blocks %.1f%% -- no better",
			100*got.Covered, 100*whole)
	}
	if got.Bytes > budget {
		t.Fatalf("placement is %d bytes over budget", got.Bytes-budget)
	}
}

// TestPlaceTakesTheDensestFirst pins the ordering: attention is read by every
// token and is the densest unit in the model, so it must be resident in every
// layer before any expert is.
func TestPlaceTakesTheDensestFirst(t *testing.T) {
	units := qwen30B(48, 128, 8)
	// Enough for all 48 attention groups and routers, and little else.
	p := Place(units, 48*(11_100_000+1_048_576))
	attn, router, expert := 0, 0, 0
	for _, u := range p.Resident {
		switch u.Kind {
		case KindAttn:
			attn++
		case KindRouter:
			router++
		case KindExpert:
			expert++
		}
	}
	if attn != 48 || router != 48 {
		t.Errorf("got %d attention and %d router units, want 48 and 48", attn, router)
	}
	if expert != 0 {
		t.Errorf("placed %d experts before the attention groups were all resident", expert)
	}
}

// TestPlaceFillsTheTail is the reason Place continues past a unit that does not
// fit instead of stopping. The list is mixed-size, so the first miss is
// routinely followed by many units that do fit.
func TestPlaceFillsTheTail(t *testing.T) {
	units := []Unit{
		{Layer: 0, Kind: KindAttn, Bytes: 100, Uses: 10},   // density 0.1
		{Layer: 1, Kind: KindFFN, Bytes: 1000, Uses: 90},   // density 0.09, does not fit
		{Layer: 2, Kind: KindExpert, Bytes: 10, Uses: 0.5}, // density 0.05, fits
	}
	p := Place(units, 150)
	if len(p.Resident) != 2 {
		t.Fatalf("got %d units, want 2 (the big one skipped, the small one taken)", len(p.Resident))
	}
	if p.Resident[1].Kind != KindExpert {
		t.Errorf("second unit is %v, want the expert that fit after the miss", p.Resident[1].Kind)
	}
}

// TestPlaceIsReproducible: two runs of one profile must place identically, or
// units relocate across the bus for no gain.
func TestPlaceIsReproducible(t *testing.T) {
	units := qwen30B(8, 16, 2)
	// Equal density on purpose: without a total order these tie and may swap.
	for i := range units {
		units[i].Uses = 1
		units[i].Bytes = 1000
	}
	a, b := Place(units, 5000), Place(units, 5000)
	if len(a.Resident) != len(b.Resident) {
		t.Fatal("different lengths")
	}
	for i := range a.Resident {
		if a.Resident[i] != b.Resident[i] {
			t.Fatalf("unit %d differs: %+v vs %+v", i, a.Resident[i], b.Resident[i])
		}
	}
}
