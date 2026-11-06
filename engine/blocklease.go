package engine

import (
	"sync"

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
	ws     *weightSource
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

func (l *blockLease) Rows(own []uint32) error { return l.ws.b.Rows(own) }

func (l *blockLease) DropRows() { l.ws.b.DropRows() }
