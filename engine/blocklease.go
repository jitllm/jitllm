package engine

import (
	"fmt"
	"sync"

	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
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
}

// Held is cpudev.Source's: block li's page is in.
func (l *blockLease) Held(li int) bool { return l.m.BlockHeld(l.fr, li) }

// Ensure is cpudev.Source's: block li's own weights read into its page and
// this instance's copy of the block bound at them (model.Model.PageIn).
func (l *blockLease) Ensure(li int) (cpudev.Tensors, error) {
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
