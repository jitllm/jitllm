package cpudev

import (
	"unsafe"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/nn"
)

// The placement's vocabulary is nn's; these name it where cpudev did.
type (
	Tensors     = nn.Tensors
	Source      = nn.Source
	ExpertPager = nn.ExpertPager
	Lease       = nn.Lease
	PartPlan    = nn.PartPlan
	PartsPlan   = nn.PartsPlan
	TablePlan   = nn.TablePlan
	KVConvPlan  = nn.KVConvPlan
	TowerPlans  = nn.TowerPlans
)

// boundWeights is a part's weights as the execution's bindings read them.
type boundWeights struct {
	b *Bindings
	w nn.PartWeights
}

func (w boundWeights) Bind(f *Form) ([]unsafe.Pointer, nn.Decline) {
	return w.b.Bind(f, w.w.Block, w.w.Of)
}

// PlaceLanes is Place (nn.LanePlacer).
func (e *Exec) PlaceLanes(n int, l nn.Lease, planes [2][]float32, nExpert int, prof bool) {
	e.Place(n, l, planes, nExpert, prof)
}

// PlaceParts composes every part of p (nn.PartsPlacer): a part the plan
// states none of is nil.
func (e *Exec) PlaceParts(p *nn.PartsPlan) nn.PlacedParts {
	defer e.comp.Place()()
	var out nn.PlacedParts
	pp := e.placeParts(p)
	if pp.Head != nil {
		out.Head = pp.Head
	}
	if pp.Streams != nil {
		out.Streams = pp.Streams
	}
	if pp.Pool != nil {
		out.Pool = pp.Pool
	}
	if pp.Entry != nil {
		out.Entry = pp.Entry
	}
	if pp.Inputs != nil {
		out.Inputs = pp.Inputs
	}
	if pp.Tables != nil {
		out.Tables = pp.Tables
	}
	for f, c := range pp.KVConv {
		if c != nil {
			out.KVConv[f] = c
		}
	}
	return out
}

// PlaceTower composes a tower's parts on the execution (nn.TowerPlacer).
func (e *Exec) PlaceTower(p nn.TowerPlans, dense nn.Tensors, taps []nn.Tensors) (nn.TowerParts, error) {
	defer e.comp.Place()()
	t, err := NewTower(e, p, dense, taps)
	if err != nil {
		return nil, err
	}
	return t, nil
}

// RowsServed says whether block li's rows run as one call, and why not
// (nn.BlockRows).
func (e *Exec) RowsServed(li int) nn.Decline {
	cb, why := e.Block(li)
	switch {
	case cb == nil:
		return why
	case cb.Rows == nil:
		return max(cb.Why, cb.RowsWhy)
	}
	return nn.DeclineServes
}

// BankViews is the bindings' (nn.BankViewer).
func (e *Exec) BankViews(li, n int, ent [3]*jlm.Entry) bool { return e.b.BankViews(li, n, ent) }
