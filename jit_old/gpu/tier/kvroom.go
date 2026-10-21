package tier

// KVRoom is how many more positions of attention history the devices can take
// without sending a page home: in every layer a device's pool holds, the pages
// it has free and the pages the budget lets it grow by, priced at one page of
// every layer at once, and the fewest of any device, since a sequence grows in
// every layer together. page is the pool's positions a page (0 with no pool);
// a sequence takes whole pages, so a caller adding up sequences rounds each up
// to it. A device with no attention layer does not bound it.
//
// It is what a scheduler admits by: rows whose histories fit it all stay on
// the card and step together, where rows past it send their oldest pages home
// and stream them back every token (kvevict.go) -- correct, and slower by the
// whole history crossing the bus once a step.
func (g *GPU) KVRoom() (positions, page int) {
	positions = -1
	for _, d := range g.devs {
		d.mu.Lock()
		n, p := d.kvRoom()
		d.mu.Unlock()
		if p == 0 {
			continue
		}
		page = p
		if positions < 0 || n < positions {
			positions = n
		}
	}
	return max(positions, 0), page
}

// kvRoom is KVRoom for one device. Callers hold g.mu.
func (g *devTier) kvRoom() (positions, page int) {
	kp := g.kvp
	if kp == nil || len(kp.layers) == 0 {
		return 0, 0
	}
	var per uint64
	free := -1
	for _, l := range kp.layers {
		per += kp.pageBytes(l)
		if n := len(l.free) + len(l.fenced); free < 0 || n < free {
			free = n
		}
	}
	g.settleRounding()
	var grow uint64
	if lim, used := g.limit, g.used+g.reserved(); used < lim {
		grow = lim - used
	}
	if g.kvTight {
		// A recorded ceiling goes stale as the card's other tenants and this
		// device's own scratch come and go, and a stale low one admitted one
		// row at a time for a whole wave; the driver's figure does not.
		free, _, err := g.dev.Mem()
		switch {
		case err != nil || free <= kvTightMargin:
			grow = 0
		default:
			grow = min(grow, free-kvTightMargin)
		}
	}
	if g.pool != nil && g.pool.limit > 0 {
		g.pool.mu.Lock()
		if g.pool.used < g.pool.limit {
			grow = min(grow, g.pool.limit-g.pool.used)
		} else {
			grow = 0
		}
		g.pool.mu.Unlock()
	}
	// The one page eviction keeps free to stream through is not room.
	pages := max(free-1, 0) + int(grow/max(per, 1))
	return pages * kp.p, kp.p
}

// kvTightMargin is what KVRoom leaves of the driver's free figure on a tight
// card: the driver's own allocations and a growth's rounding.
const kvTightMargin = 64 << 20

// PromiseKV tells the devices that a scheduler has admitted rows whose
// histories will grow by positions more, page-rounded per row, and that the
// room for them is spoken for: a batched scratch of a width nobody reserved
// is not built into it (prepBatch), so a scratch built after admission does
// not take the pages the admitted rows were promised and send their history
// home. 0 withdraws the promise.
func (g *GPU) PromiseKV(positions int) {
	for _, d := range g.devs {
		d.mu.Lock()
		d.kvPromise = max(positions, 0)
		d.mu.Unlock()
	}
}

// PrewarmKV is nn.KVPrewarmer: every device's pool grows so each of its
// layers has positions positions' worth of free pages, where the device's
// budget has the room as it stands -- nothing is paged out for it, and the
// pages go to no sequence, so another session can take them as it would
// take a page any growth made. A layer it cannot grow is left as it is; the
// first prompt grows it as it always did.
func (g *GPU) PrewarmKV(positions int) {
	for _, d := range g.devs {
		d.prewarmKV(positions)
	}
}

// prewarmKV is PrewarmKV on one device.
func (g *devTier) prewarmKV(positions int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	kp := g.kvp
	if !g.paged || kp == nil || positions <= 0 || kp.p <= 0 {
		return
	}
	pages := (positions + kp.p - 1) / kp.p
	for _, l := range kp.layers {
		// DeepSeek V4's entries grow at their own rate with their block.
		if l.rate > 0 || len(l.free) >= pages {
			continue
		}
		// At least a doubling, so growKVLayer grows to exactly this and asks
		// for no more room than room() was asked for.
		want := max(l.n+pages-len(l.free), 2*l.n)
		// The room left must also hold another scratch as large as every
		// one built so far: a step's batched scratch is built when the rows
		// arrive, and pages grown ahead and then filled by histories would
		// refuse it every step (an olmoe step on a 4 GB card did). Where that
		// is not left, the pools stay at their first frame.
		if want > kp.maxPages(l) || !g.room(kp.pageBytes(l)*uint64(want)+g.scratch) ||
			!g.keepsSlot(kp.pageBytes(l)*uint64(want-l.n)) {
			continue
		}
		before := l.n
		if err := g.growKVLayer(kp, l, want); err != nil {
			continue
		}
		g.KVPrewarmPages += l.n - before
	}
}

// promisedBytes is what the promise needs beyond the pool's free pages, at
// one page of every layer a page. Callers hold g.mu.
func (g *devTier) promisedBytes() uint64 {
	kp := g.kvp
	if g.kvPromise <= 0 || kp == nil || len(kp.layers) == 0 || kp.p <= 0 {
		return 0
	}
	var per uint64
	free := -1
	for _, l := range kp.layers {
		per += kp.pageBytes(l)
		if n := len(l.free) + len(l.fenced); free < 0 || n < free {
			free = n
		}
	}
	pages := (g.kvPromise + kp.p - 1) / kp.p
	return uint64(max(pages-max(free-1, 0), 0)) * per
}
