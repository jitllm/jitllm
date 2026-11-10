package engine

import (
	"fmt"
	"sync"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// blockLease is what a block's call on the CPU device pages in during it,
// through the instance's pager (cpudev.Resources): the routed experts a
// selection picks, held in hold under mu and counted in the residency and,
// for a step, the hotness; and the n-gram table's rows, read into the
// weights' row view.
type blockLease struct {
	m      *model.Model
	fr     *jlm.Frames
	mu     *sync.Mutex
	hold   *model.ExpertHold
	res    *model.Residency
	hot    *Hotness
	layers []model.Layer
	lo     int
	// trace is the instance's tracer (WithTrace), read at each call: a
	// call's selection is traced as moe_topk, its experts in the order its
	// rows pick them, each once (cpudev.Runner.experts); picks is its
	// values.
	trace *func(layer int, name string, v []float32)
	picks []float32
}

// Held is nn.Source's: block li's page is in.
func (l *blockLease) Held(li int) bool { return l.m.BlockHeld(l.fr, li) }

// Ensure is nn.Source's: block li's own weights read into its page and
// this instance's copy of the block bound at them (model.Model.PageIn).
func (l *blockLease) Ensure(li int) (nn.Tensors, error) {
	if li < l.lo || li >= l.lo+len(l.layers) {
		return nil, fmt.Errorf("engine: block %d is not one of this instance's [%d, %d)", li, l.lo, l.lo+len(l.layers))
	}
	t := &l.layers[li-l.lo]
	l.mu.Lock()
	defer l.mu.Unlock()
	return t, l.m.PageIn(l.fr, li, t)
}

func (l *blockLease) Experts(li int, sel []int32, count bool) error {
	l.mu.Lock()
	err := l.m.EnsureExperts(l.fr, li, sel, l.hold, &l.layers[li-l.lo])
	if err == nil && l.trace != nil && *l.trace != nil {
		l.picks = l.picks[:0]
		for _, e := range sel {
			l.picks = append(l.picks, float32(e))
		}
		(*l.trace)(li, "moe_topk", l.picks)
	}
	l.mu.Unlock()
	if err != nil {
		return err
	}
	l.res.Touch(li, l.m.Cfg.NExpert, sel)
	if count {
		l.hot.add(li, sel)
	}
	return nil
}

func (l *blockLease) ReleaseExperts() { l.hold.Release() }
