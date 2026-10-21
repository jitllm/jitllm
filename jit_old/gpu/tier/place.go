package tier

import "sort"

// Placement decides which units of a model live on the device.
//
// The unit is not the block: in a mixture most of a block is expert banks a
// token reads a small fraction of, while attention is read in full by every
// token, so ranking by hotness covers far more of a token's bytes than taking
// whole blocks for the same budget.
//
// Hotness is measured at runtime because it is a property of the workload
// (different prompts pick different hot experts), so this takes counts rather
// than a table baked at build time.
//
// A unit that does not fit is computed on the host concurrently, so the bus is
// never on the critical path; placing by hotness pays each transfer once
// rather than paging on a miss through a link slower than host DRAM.
type Placement struct {
	// Resident is the units chosen, in the order they should be uploaded --
	// hottest per byte first, so a budget that runs out mid-list still holds
	// the most valuable prefix.
	Resident []Unit
	// Bytes is what Resident costs on the device.
	Bytes uint64
	// Covered is the share of per-token weight bytes that Resident serves,
	// between 0 and 1: the number to report and to compare policies on.
	Covered float64
	// Offered is the total per-token weight bytes across every unit, so
	// Covered's denominator is visible rather than implied.
	Offered float64
}

// Unit is one placeable piece of a model.
type Unit struct {
	Layer int
	Kind  UnitKind
	// Expert is meaningful only for KindExpert.
	Expert int
	// Bytes is what this unit occupies on the device, which can exceed its
	// file size after packing; pricing in file bytes would overcommit the card.
	//
	// For KindAttn it must include the KV cache (PrepLayer charges
	// MaxSeq*NKVHead*HeadDim*4 for K and again for V), which at a real context
	// is larger than the attention weights themselves.
	Bytes uint64
	// Uses is how many times this unit was read, in whatever window the caller
	// counted over. Only the ratio between units matters.
	Uses float64
}

type UnitKind uint8

const (
	// KindAttn is q/k/v/o and the norms: read in full by every token, so it is
	// the densest unit in any model and is placed first whenever it fits.
	KindAttn UnitKind = iota
	// KindRouter is ffn_gate_inp. Tiny, and read every token.
	KindRouter
	// KindFFN is a dense block's gate/up/down.
	KindFFN
	// KindExpert is one expert's gate/up/down out of a bank.
	KindExpert
	// KindHead is the output norm and the vocabulary projection: read in full
	// every token, so joint-densest with attention, and often the largest
	// single unit worth placing. Its Layer is meaningless; there is one per
	// model.
	KindHead
)

func (k UnitKind) String() string {
	return [...]string{"attn", "router", "ffn", "expert", "head"}[k]
}

// Place ranks units by reads per resident byte and takes them while the budget
// lasts. A greedy knapsack is enough: experts in a layer are all one size, the
// other sizes (attention, router) are the densest and always taken, and the
// items are megabytes against a budget of gigabytes.
func Place(units []Unit, budget uint64) Placement {
	p := Placement{}
	for _, u := range units {
		p.Offered += u.Uses * float64(u.Bytes)
	}
	ranked := append([]Unit(nil), units...)
	sort.SliceStable(ranked, func(i, j int) bool {
		a, b := ranked[i], ranked[j]
		// Rank by Uses, not Uses/Bytes: placing a unit removes Uses*Bytes of
		// per-token host traffic at a cost of Bytes, so value per byte is Uses
		// alone. Dividing again by Bytes over-values small units
		// (TestPlaceTakesTheDensestFirst).
		if a.Uses != b.Uses {
			return a.Uses > b.Uses
		}
		// Equal density: prefer the smaller unit, so more of them fit.
		if a.Bytes != b.Bytes {
			return a.Bytes < b.Bytes
		}
		// A total order, so a placement is reproducible run to run.
		if a.Layer != b.Layer {
			return a.Layer < b.Layer
		}
		if a.Kind != b.Kind {
			return a.Kind < b.Kind
		}
		return a.Expert < b.Expert
	})
	var covered float64
	for _, u := range ranked {
		if p.Bytes+u.Bytes > budget {
			// Keep going: the list is mixed-size, so smaller units later on
			// may still fit.
			continue
		}
		p.Bytes += u.Bytes
		covered += u.Uses * float64(u.Bytes)
		p.Resident = append(p.Resident, u)
	}
	if p.Offered > 0 {
		p.Covered = covered / p.Offered
	}
	return p
}
