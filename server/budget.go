package server

import (
	"fmt"
)

// backgroundShare is the fraction of the host budget that all backgrounded
// models divide between them when the active model is given priority.
//
// A backgrounded model is not closed, so it still needs bytes: it keeps its
// pager and buys residency back by paging when next used. An eighth keeps the
// dense region and a few block pages.
const backgroundShare = 8

// shares divides the host weight budget between the open models.
//
// Something here has to divide it because nothing in the engine does:
// sched.MemBudget() is stateless and model.Open, gpu.OpenWith and NewInstance
// each read it independently, so two models would each believe they hold the
// machine. The cgroup then throttles into reclaim, and neither pager can see
// the other model. (The vision tower avoided this by sharing one container
// and one budget; two models are two files and cannot.)
// pins are per-model caps the user set. They are honoured exactly and come off
// the top; the rest divide what is left.
//
// total is the budget, read once by the caller: sched.MemBudget follows the
// machine's free memory, so two reads of it are two different numbers, and a
// division checked against another moment's budget can exceed it.
func shares(total uint64, paths []string, active string, priority bool, pins map[string]uint64) map[string]uint64 {
	out := make(map[string]uint64, len(paths))
	if len(paths) == 0 {
		return out
	}

	// Pins first, so an unpinned model can never be handed bytes a pin has
	// already claimed.
	var free uint64 = total
	var unpinned []string
	for _, p := range paths {
		if n := pins[p]; n > 0 {
			if n > free {
				n = free
			}
			out[p] = n
			free -= n
			continue
		}
		unpinned = append(unpinned, p)
	}
	if len(unpinned) == 0 {
		return out
	}

	// Without priority the unpinned models split what is left evenly.
	if !priority {
		each := free / uint64(len(unpinned))
		for _, p := range unpinned {
			out[p] = each
		}
		return out
	}

	// Priority gives the active model the rest. Switching re-runs this, so the
	// bytes follow the model in use.
	bg := free / backgroundShare
	if len(unpinned) == 1 {
		out[unpinned[0]] = free
		return out
	}
	perBg := bg / uint64(len(unpinned)-1)
	var activeUnpinned bool
	for _, p := range unpinned {
		if p == active {
			activeUnpinned = true
			continue
		}
		out[p] = perBg
	}
	if activeUnpinned {
		out[active] = free - perBg*uint64(len(unpinned)-1)
		return out
	}
	// The active model is pinned, so the remainder goes back to the others.
	each := free / uint64(len(unpinned))
	for _, p := range unpinned {
		out[p] = each
	}
	return out
}

// hostTotal is the host budget every loaded model's share is divided from:
// Config.HostBudget when the caller named one, else what the host offered at
// the last load (rereadHost). Guarded by e.mu.
func (e *Engine) hostTotal() uint64 {
	if e.cfg.HostBudget > 0 {
		return e.cfg.HostBudget
	}
	if e.total == 0 {
		e.rereadHostLocked()
	}
	return e.total
}

// rereadHostLocked asks the host again, at a load: sched.MemBudget is the
// smallest of the cgroup, the bound NUMA nodes and what is available now, so
// memory other processes took since the last load is no longer counted as
// ours. What this engine's models hold is added back -- MemBudget reads it as
// taken, and it is the engine's to divide. e.mu held.
func (e *Engine) rereadHostLocked() {
	e.total = e.availNow() + e.heldLocked()
}

// heldLocked is the host memory this engine's models hold: resident weights,
// the memory caches, every session's and pooled session's history, and each
// model's host step scratch. e.mu
// held.
func (e *Engine) heldLocked() uint64 {
	var n uint64
	for _, lm := range e.models {
		n += lm.m.HostBytes() + lm.stepScratchBytes()
		if st := lm.ttft.store; st != nil {
			n += st.Bytes()
		}
		for _, s := range lm.sessions {
			n += s.snapKV.Load()
		}
	}
	return n
}

// availNow is the host budget as the host states it now. A test replaces it
// (Engine.hostAvail) to hold the host's answer still.
func (e *Engine) availNow() uint64 {
	if e.hostAvail != nil {
		return e.hostAvail()
	}
	return e.cfg.GC.Budget()
}

// sharesLocked divides the host budget between the loaded models, plus extra
// when a load is about to add one. e.mu held.
func (e *Engine) sharesLocked(extra string) map[string]uint64 {
	ids := append([]string(nil), e.order...)
	if extra != "" {
		ids = append(ids, extra)
	}
	pins := map[string]uint64{}
	for id, lm := range e.models {
		if lm.pin > 0 {
			pins[id] = lm.pin
		}
	}
	return shares(e.hostTotal(), ids, e.favored, e.priority, pins)
}

// rebudget re-divides the host budget and applies each model's share. A share
// is a function of the whole set, so it runs on every load, unload, pin and
// change of priority.
func (e *Engine) rebudget() {
	e.mu.Lock()
	sh := e.sharesLocked("")
	models := make([]*LoadedModel, 0, len(e.models))
	var sum, mapped uint64
	for id, lm := range e.models {
		lm.mu.Lock()
		lm.budget = sh[id]
		lm.mu.Unlock()
		sum += sh[id]
		if lm.m != nil {
			mapped += lm.m.MappedBytes()
		}
		models = append(models, lm)
	}
	// Under e.mu, so two rebudgets cannot hand the limit their sums out of
	// order.
	if e.cfg.OffHeap != nil {
		e.cfg.OffHeap(sum, mapped)
	}
	e.mu.Unlock()
	for _, lm := range models {
		e.applyPageBudget(lm, nil)
	}
}

// Pin caps one model's host budget at bytes, taken off the top before the
// others divide the rest; 0 releases it back to the division.
func (e *Engine) Pin(id string, bytes uint64) error {
	lm, err := e.Model(id)
	if err != nil {
		return err
	}
	lm.mu.Lock()
	lm.pin = bytes
	lm.mu.Unlock()
	e.rebudget()
	return nil
}

// SetPriority gives the favoured model (Favor) everything but an eighth of the
// budget, the rest dividing that eighth; off, the models divide it evenly.
func (e *Engine) SetPriority(on bool) {
	e.mu.Lock()
	e.priority = on
	e.mu.Unlock()
	e.rebudget()
}

// Favor names the model in use, the one priority gives the budget to.
func (e *Engine) Favor(id string) {
	e.mu.Lock()
	e.favored = id
	e.mu.Unlock()
	e.rebudget()
}

// Budget is the total this model's pager is divided from now.
func (lm *LoadedModel) Budget() uint64 {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	return lm.budget
}

// Pinned reports whether the model's budget is a pin rather than a share.
func (lm *LoadedModel) Pinned() bool {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	return lm.pin > 0
}

// The priorities a request may name (GenerateOptions.Priority).
const (
	PriorityHigh   = "high"
	PriorityNormal = "normal"
)

// prioritise applies a request's priority. "high" makes its model the
// favoured one with priority on, so the host budget follows the model in use:
// it gets all but an eighth, the others divide the eighth and page. "normal"
// changes nothing, so a request that does not care cannot take the budget
// from one that asked. The re-division runs only when the answer changes.
func (e *Engine) prioritise(lm *LoadedModel, p string) error {
	switch p {
	case PriorityNormal:
		return nil
	case PriorityHigh:
	default:
		return fmt.Errorf("%w: priority %q: want %q or %q", ErrInvalid, p, PriorityHigh, PriorityNormal)
	}
	e.mu.Lock()
	same := e.priority && e.favored == lm.id
	e.priority, e.favored = true, lm.id
	e.mu.Unlock()
	if !same {
		e.rebudget()
	}
	return nil
}
