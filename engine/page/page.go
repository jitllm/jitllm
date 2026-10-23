package page

import "github.com/jitllm/jitllm/engine/sched"

// Rec is one block's page record: what its pageable weights cost on a
// device while resident, and whether they are. Every block of every model
// has one, whether or not the model fits: the budget decides only whether
// eviction ever fires.
type Rec struct {
	// Bytes is what the block's weights cost on the device while resident;
	// Round the driver's page rounding of those buffers beyond Bytes, which
	// a page-in also has to find room for.
	Bytes, Round uint64
	// In is whether the weights are resident right now.
	In bool
	// Uses is how many submissions -- running, or paged in and about to run
	// -- read the block: it is no victim while any does.
	Uses int
}

// Pick chooses the block a page-in evicts: of the candidates offered, the
// resident one whose next use is furthest away. On the cyclic layer scan
// that is Belady's optimum and it is computable (sched.NextUse): LRU is
// pessimal there, while this pins a prefix of slots-1 blocks and thrashes
// only the last slot. A block outside the scan is never used again in it,
// so it goes first; ties break on the lowest index, so the choice does not
// depend on the order the candidates come in. A block being loaded (cur),
// one a submission reads, and one whose page-out frees nothing are never
// chosen.
type Pick struct {
	cur, lo, hi int
	best, bestD int
}

// NewPick is a choice for loading block cur while scan [lo, hi) runs.
func NewPick(cur, lo, hi int) Pick { return Pick{cur: cur, lo: lo, hi: hi, best: -1, bestD: -1} }

// Offer puts block li, record r, among the candidates.
func (p *Pick) Offer(li int, r *Rec) {
	if r == nil || !r.In || li == p.cur || r.Uses > 0 || r.Bytes == 0 {
		return
	}
	d := sched.NextUse(li, p.cur, p.lo, p.hi)
	if d > p.bestD || (d == p.bestD && li < p.best) {
		p.best, p.bestD = li, d
	}
}

// Victim is the block chosen, -1 for none.
func (p *Pick) Victim() int { return p.best }
