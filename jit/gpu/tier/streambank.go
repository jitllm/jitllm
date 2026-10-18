package tier

import (
	"fmt"
	"sync"
	"time"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// streamBank is the host side of a block whose routed experts are assembled per
// token, and the suspension that makes that possible.
//
// A mixture routing few of many experts can make a resident bank tens of times
// the bytes a token reads, so on a small card the choice is streaming or the
// host. The price is a host round trip mid-block: the selection must come home
// before anyone knows which sheets to send, so the submission suspends after
// ExpertRank and resumes after the upload, and such a block cannot be captured
// into a graph (a recording has nowhere to put a readback).
type streamBank struct {
	// sh is the per-expert sheet size of each plane, in bytes, for gate, up and
	// down in that order: [matrix][qs, d, sc].
	sh [3][3]int
	// sel is the selection, read home each token. It is len NExpertUsed and is
	// reused rather than allocated per token.
	sel []uint32
	// w is the host side of this block. ensure re-points its spans at wherever
	// the host pager holds the block now, a different frame every token on a
	// model that pages (see nn.LayerWeights.Ensure).
	w      nn.LayerWeights
	ensure func(*nn.LayerWeights) error
	// sel2 is ensure narrowed to the routed sheets. When it is non-nil the
	// whole-bank read never happens; see nn.LayerWeights.EnsureExperts.
	sel2 func(*nn.LayerWeights, []uint32) error
	// pre is sel2's read half, safe to call from another goroutine. When it is
	// non-nil fill pipelines: group g uploads while group g+1 reads. See
	// nn.LayerWeights.PrefetchExperts.
	pre func([]uint32) error
	// rd and grp are fill's reads in flight and its groups, kept so a token
	// allocates neither; items and want are its work list and the experts
	// it reads.
	rd    []streamRead
	grp   [][2]int
	items []streamItem
	want  []uint32
	// The expert cache (initCache), nil without one: slotOf maps an expert
	// to the slot holding it or -1, owner a slot to its expert or -1, used a
	// slot to the tick it was last selected; slotSel is this token's
	// selection in slots.
	slotOf  []int32
	owner   []int32
	used    []uint64
	tick    uint64
	slotSel []uint32
	bank    *resident
	// plain is the bank a cache sized after placement replaced (cacheBlock),
	// kept so dropCaches can give the cache back.
	plain [3]*resident
	// nExpert is the bank's expert count, so a cache can be set up after
	// placement (GPU.sizeAutoCaches).
	nExpert int
	// The cross-layer prefetch (prefetch): the experts it reads, whether a
	// read is in flight, its join and its error.
	pfWant []uint32
	// spec hands the host's read service a guess and nil abandons the
	// unstarted ones (nn.LayerWeights.SpeculateExperts).
	spec func(sel []uint32)
	// host is the block's routed experts on the host (hybrid execution), and
	// hin, hw and hout its input, weights and output, reused every token.
	host          func(sel []uint32, w, in, out []float32) error
	hybrid        bool // the experts run on the host (runHost), not sent
	li            int  // the block, for the session's host side (hostFor)
	hin, hw, hout []float32
	hsel          []uint32
	// rowSel and rowW are a split step's misses for the host (splitRows).
	rowSel []uint32
	rowW   []float32
	// rowOrig is the step's selections as the router chose them (splitRows
	// rewrites the rows to slots), rowMiss the experts learnRows offers.
	rowOrig, rowMiss []uint32
	rowHit           []bool
	// The split (hybrid with a cache): heat counts each expert's selections,
	// halved every heatHalf tokens; hit marks which of this token's
	// selection the cache holds; dw is the card's weights (a miss's zeroed),
	// and msel and mw the misses the host runs.
	heat []uint32
	own  []uint32
	// owned counts the tokens st.own has counted, for its halving.
	owned uint64
	hit   []bool
	dw    []float32
	msel  []uint32
	mw    []float32
	// pred is this block's selection as the previous block's probe predicted
	// it (Config.StreamProbe), valid while havePred; score compares it with
	// the real one and clears it.
	pred     []uint32
	havePred bool
	// prefillWaves' scratch: wslot is an expert's slot while its wave runs
	// (-1 unseen), waveHit and waveMiss the chunk's experts the cache held
	// and did not.
	wslot             []int32
	waveHit, waveMiss []uint32
	// peer is this block's banks on the expert card, nil without one
	// (peerbank.go).
	peer *peerBank
}

// score charges this token's selection to the probe's counters and hands it
// to Config.StreamSelLog. Both are measurement: a run without either pays a
// branch.
func (st *streamBank) score(g *devTier, li int) {
	if g.StreamSelLog != nil {
		g.StreamSelLog(li, st.sel)
	}
	if !st.havePred {
		return
	}
	st.havePred = false
	for _, e := range st.sel {
		for _, q := range st.pred {
			if q == e {
				g.ProbeHits++
				break
			}
		}
	}
	g.ProbeExperts += len(st.sel)
}

// growF32 returns s resized to n, reusing its array when it is big enough.
func growF32(s []float32, n int) []float32 {
	if cap(s) < n {
		return make([]float32, n)
	}
	return s[:n]
}

// growU32 returns s resized to n, reusing its array when it is big enough.
func growU32(s []uint32, n int) []uint32 {
	if cap(s) < n {
		return make([]uint32, n)
	}
	return s[:n]
}

// fill uploads the k sheets this token routed to, in selection order, so that
// compact slot i is selected expert i and bs.rw[i] is its weight. The order is
// the contract: ExpertRank writes rsel and rtop in descending-logit order and
// the combine pairs slot i with rw[i], so sorting by expert id would weight
// every expert with another's gate.
func (st *streamBank) fill(g *devTier, s backend.Session, l *layer, nExpert int) error {
	// ensure and the uploads below are one window over a reused host frame, so an
	// eviction in between would send another block's weights to the card.
	// model.enterPager holds the pager for the whole Forward, which closes this
	// window and the other three like it; jlm.Lease is the finer primitive.
	//
	// Every group's read is issued up front and group g uploads as soon as its
	// own read has landed, so the pager has the whole selection in flight and
	// the transfers overlap the reads still running. Only the first group goes
	// through sel2, which re-points st.w and binds; the reads go through pre,
	// which touches nothing shared -- two sel2 calls in flight would race on
	// the model's layer. A bank in expert pages gets nothing from sel2 (its
	// sheets are reached through Sheet), so the first group is read through pre
	// too: without it every page was faulted in by Sheet one at a time, inside
	// the upload loop (placement.md 16c).
	// The work is a list of (expert, slot) items: every selected expert into
	// slot i without a cache, and only the misses, into the slots they evict,
	// with one.
	st.items = st.items[:0]
	if st.cached() {
		// A bank paged out and back in is a new buffer with nothing in it, so
		// the map is only good for the buffer it was built over.
		if l.down != st.bank {
			st.forget()
			st.bank = l.down
		}
		st.lookup(g)
	} else {
		for i, e := range st.sel {
			st.items = append(st.items, streamItem{e: e, slot: i})
		}
	}
	if len(st.items) == 0 {
		return nil
	}
	st.want = st.want[:0]
	for _, it := range st.items {
		st.want = append(st.want, it.e)
	}
	grp := st.groups(g, len(st.items))
	t0 := time.Now()
	// The previous block's guesses that no reader has started are dropped:
	// whatever this fill needs, it reads itself below, and a guess already
	// being read is joined by the read of the same page rather than waited
	// for (jlm.File.Speculate), so a wrong guess delays nothing.
	if st.spec != nil {
		st.spec(nil)
	}
	if st.pre != nil {
		if cap(st.rd) < len(grp) {
			st.rd = make([]streamRead, len(grp))
		}
		st.rd = st.rd[:len(grp)]
		for gi := 1; gi < len(grp); gi++ {
			st.rd[gi].wg.Add(1)
			g.StreamOverlaps++
			go st.read(gi, grp[gi])
		}
	}
	var err error
	if st.pre != nil {
		err = st.pre(st.want[grp[0][0]:grp[0][1]])
	}
	switch {
	case err != nil:
	case st.sel2 != nil:
		// The narrow read is preferred: the whole bank is correct but tens of
		// times the bytes the token consumes.
		err = st.sel2(&st.w, st.want[grp[0][0]:grp[0][1]])
	case st.ensure != nil:
		err = st.ensure(&st.w)
	}
	g.TStreamRead += time.Since(t0)
	for gi := range grp {
		// Every read is joined, even after a failure: a goroutine still reading
		// into a frame this block is about to release is the eviction hazard
		// above.
		if gi > 0 && st.pre != nil {
			t2 := time.Now()
			st.rd[gi].wg.Wait()
			g.TStreamRead += time.Since(t2)
			if err == nil {
				err = st.rd[gi].err
			}
		}
		if err == nil {
			t1 := time.Now()
			err = st.put(g, s, l, nExpert, grp[gi])
			g.TStreamPut += time.Since(t1)
		}
	}
	return err
}

// prefetch starts reading this block's predicted experts into host frames
// while the block before it finishes (Config.StreamPrefetch). It reads and
// never uploads: a wrong guess costs disk bytes, not link bytes or cache
// slots, and the fill that follows still reads, maps and sends exactly the
// experts the router chose. An expert the cache already holds is not read.
func (st *streamBank) prefetch(g *devTier) {
	if st.pre == nil || st.spec == nil {
		return
	}
	st.pfWant = st.pfWant[:0]
	for _, e := range st.pred {
		if st.cached() && int(e) < len(st.slotOf) && st.slotOf[e] >= 0 {
			continue
		}
		st.pfWant = append(st.pfWant, e)
	}
	if len(st.pfWant) == 0 {
		return
	}
	g.StreamPrefetched += len(st.pfWant)
	st.spec(st.pfWant)
}

// runHostRows is runHost for a chunk of rows: the selections (row stride
// k+1, ExpertRank's layout), the weights (stride k) and the rows of in come
// home, the host runs each row's experts, and the rows of the sum go back.
func (st *streamBank) runHostRows(sid uint64, g *devTier, s backend.Session, sel, w, in, out backend.Buf, n, k, rows int) error {
	if cap(st.hin) < rows*n {
		st.hin = make([]float32, rows*n)
		st.hout = make([]float32, rows*n)
	}
	st.hin, st.hout = st.hin[:rows*n], st.hout[:rows*n]
	if cap(st.hw) < rows*k {
		st.hw = make([]float32, rows*k)
	}
	st.hw = st.hw[:rows*k]
	if cap(st.hsel) < rows*(k+1) {
		st.hsel = make([]uint32, rows*(k+1))
	}
	st.hsel = st.hsel[:rows*(k+1)]
	t0 := time.Now()
	if err := s.Sync(); err != nil {
		return err
	}
	for _, rd := range [3]struct {
		b backend.Buf
		p []byte
	}{{in, f32b(st.hin)}, {w, f32b(st.hw)}, {sel, u32b(st.hsel)}} {
		if err := s.Read(rd.b, rd.p); err != nil {
			return err
		}
	}
	if hr := g.hostFns[sid][st.li].rows; hr != nil {
		// Expert-major: each chosen expert read once for the chunk.
		if err := hr(st.hsel, st.hw, st.hin, st.hout, rows, k); err != nil {
			return err
		}
		err := s.WriteAt(out, 0, f32b(st.hout))
		g.THybrid += time.Since(t0)
		g.HybridRows += rows
		return err
	}
	host := g.hostFor(sid, st)
	if host == nil {
		return fmt.Errorf("tier: this session offered no host side for a hybrid block")
	}
	for r := 0; r < rows; r++ {
		if err := host(st.hsel[r*(k+1):r*(k+1)+k], st.hw[r*k:(r+1)*k],
			st.hin[r*n:(r+1)*n], st.hout[r*n:(r+1)*n]); err != nil {
			return err
		}
	}
	err := s.WriteAt(out, 0, f32b(st.hout))
	g.THybrid += time.Since(t0)
	g.HybridRows += rows
	return err
}

// splitRows is split for a step of several rows (a speculation step's
// verification, a few-row chunk): every row's selection goes through split
// and splitWeights as one token's does -- its heat counted, its hits mapped
// to their cache slots, nothing evicted or admitted. The rows' input, weights
// and selections come home; sel and w are written back as the card's share
// (split's slots, splitWeights' weights with the misses zeroed) for the
// grouped expert kernels over the cache bank, each cached expert read once
// for every row that chose it, and the misses are kept (st.rowSel, a hit's
// place holding noExpert) for hostRowsLate, which runs them while the card
// runs the hits. rows is the step's padded width, valid its real rows: a
// padding row is all zero weights at slot 0, since its selection names
// experts the cache bank does not hold. It returns the hits.
func (st *streamBank) splitRows(g *devTier, s backend.Session, sel, w, in backend.Buf, n, k, rows, valid int) (int, error) {
	if cap(st.hin) < valid*n {
		st.hin = make([]float32, valid*n)
		st.hout = make([]float32, valid*n)
	}
	st.hin, st.hout = st.hin[:valid*n], st.hout[:valid*n]
	st.hw = growF32(st.hw, rows*k)
	st.hsel = growU32(st.hsel, rows*(k+1))
	st.rowSel = growU32(st.rowSel, valid*(k+1))
	st.rowW = growF32(st.rowW, valid*k)
	st.rowOrig = growU32(st.rowOrig, valid*k)
	if err := s.Sync(); err != nil {
		return 0, err
	}
	if err := s.Read(in, f32b(st.hin)); err != nil {
		return 0, err
	}
	if err := s.Read(w, f32b(st.hw)); err != nil {
		return 0, err
	}
	if err := s.Read(sel, u32b(st.hsel)); err != nil {
		return 0, err
	}
	// split reads st.sel, the decode step's selection buffer: it is pointed
	// at each row in turn and put back.
	tok := st.sel
	defer func() { st.sel = tok }()
	hits := 0
	for r := 0; r < rows; r++ {
		if r >= valid {
			for j := 0; j < k; j++ {
				st.hsel[r*(k+1)+j], st.hw[r*k+j] = 0, 0
			}
			continue
		}
		row := st.hsel[r*(k+1) : r*(k+1)+k]
		st.sel = row
		hits += st.split(g)
		st.splitWeights(st.hw[r*k : (r+1)*k])
		st.rowSel[r*(k+1)+k] = st.hsel[r*(k+1)+k]
		for j := 0; j < k; j++ {
			st.rowW[r*k+j] = st.hw[r*k+j]
			st.rowOrig[r*k+j] = row[j]
			st.rowSel[r*(k+1)+j] = row[j]
			if st.hit[j] {
				st.rowSel[r*(k+1)+j] = noExpert
			}
		}
		copy(st.hw[r*k:(r+1)*k], st.dw)
		copy(row, st.slotSel[:k])
	}
	if err := s.WriteAt(sel, 0, u32b(st.hsel)); err != nil {
		return 0, err
	}
	return hits, s.WriteAt(w, 0, f32b(st.hw))
}

// noExpert holds a hit's place in the host's share of a split step: an id no
// expert has, which the host's expert-major pass matches to nothing.
const noExpert = ^uint32(0)

// learnRows is what decode's count and admit do for a token, for the valid
// rows of a split step: every row's selection counts into the session's
// expert counts, the slots the step's hits read are held, and the misses are
// queued for the cache (admit's rules: an empty slot takes any, an occupied
// one only an expert selected clearly more often, within the token's swap
// budget). The sheets go out beside the card's launches and are in use from
// the next step (finishAdmits). Without it a speculating run, whose every
// step has several rows, never fills the cache it split against.
func (st *streamBank) learnRows(g *devTier, s backend.Session, l *layer, uses []uint32, nExpert, k, valid int) error {
	tok, hit := st.sel, st.hit
	defer func() { st.sel, st.hit = tok, hit }()
	st.rowMiss = st.rowMiss[:0]
	for r := 0; r < valid; r++ {
		st.sel = st.rowOrig[r*k : (r+1)*k]
		st.count(uses)
	}
	st.tick++
	for _, e := range st.rowOrig[:valid*k] {
		if sl := st.slotOf[e]; sl >= 0 {
			st.used[sl] = st.tick
		} else {
			st.rowMiss = append(st.rowMiss, e)
		}
	}
	if len(st.rowMiss) == 0 {
		return nil
	}
	if cap(st.rowHit) < len(st.rowMiss) {
		st.rowHit = make([]bool, len(st.rowMiss))
	}
	st.rowHit = st.rowHit[:len(st.rowMiss)]
	clear(st.rowHit)
	st.sel, st.hit = st.rowMiss, st.rowHit
	return st.admit(g, s, l, nExpert, g.splitAdmit()*valid)
}

// hostRowsLate runs splitRows' misses on the host, row by row as decode's
// hostLate runs one token's (a few rows share few of their experts, so the
// host's bytes are the same either way, and the session's host side picks
// its own order for a row), while the card runs the hits queued before it,
// and writes the rows' sum to dst. A row with no miss sums to zero.
func (st *streamBank) hostRowsLate(sid uint64, g *devTier, s backend.Session, dst backend.Buf, k, valid int) error {
	t0 := time.Now()
	host := g.hostFor(sid, st)
	if host == nil {
		return fmt.Errorf("tier: this session offered no host side for a hybrid block")
	}
	n := len(st.hin) / valid
	for r := 0; r < valid; r++ {
		st.msel, st.mw = st.msel[:0], st.mw[:0]
		for j := 0; j < k; j++ {
			if e := st.rowSel[r*(k+1)+j]; e != noExpert {
				st.msel = append(st.msel, e)
				st.mw = append(st.mw, st.rowW[r*k+j])
			}
		}
		out := st.hout[r*n : (r+1)*n]
		if len(st.msel) == 0 {
			clear(out)
			continue
		}
		if err := host(st.msel, st.mw, st.hin[r*n:(r+1)*n], out); err != nil {
			return err
		}
	}
	g.TSplitHost += time.Since(t0)
	g.SplitRowSteps++
	g.HybridRows += valid
	tw := time.Now()
	err := s.WriteAt(dst, 0, f32b(st.hout))
	g.TLateWrite += time.Since(tw)
	return err
}

// hostFor is session sid's host side for st's block, nil when this
// session never offered one. Never the placing session's: that State may be
// closed, and its generated code with it. Callers hold g.mu.
func (g *devTier) hostFor(sid uint64, st *streamBank) func(sel []uint32, w, in, out []float32) error {
	return g.hostFns[sid][st.li].one
}

// hostSide is a session's host side for one hybrid block: a row at a time,
// and a chunk's rows expert-major.
type hostSide struct {
	one  func(sel []uint32, w, in, out []float32) error
	rows func(sel []uint32, w, in, out []float32, rows, k int) error
	// uses is the session's counts of this block's experts
	// (nn.LayerWeights.ExpertUses).
	uses []uint32
}

// splitIn brings a split step's input and routing weights home and divides
// the weights between the card and the host. The Sync before the selection
// was read covers in.
func (st *streamBank) splitIn(s backend.Session, in, w backend.Buf, n, k int) error {
	if err := st.home(s, in, w, n, k); err != nil {
		return err
	}
	st.splitWeights(st.hw[:k])
	return nil
}

// home reads a hybrid step's expert input and routing weights back.
func (st *streamBank) home(s backend.Session, in, w backend.Buf, n, k int) error {
	if cap(st.hin) < n {
		st.hin = make([]float32, n)
		st.hout = make([]float32, n)
	}
	st.hin, st.hout = st.hin[:n], st.hout[:n]
	if cap(st.hw) < k+1 {
		st.hw = make([]float32, k+1)
	}
	st.hw = st.hw[:k+1]
	if err := s.Read(in, f32b(st.hin)); err != nil {
		return err
	}
	return s.Read(w, f32b(st.hw))
}

// hybridIn is splitIn for a step with no hit: every selected expert is the
// host's.
func (st *streamBank) hybridIn(s backend.Session, in, w backend.Buf, n, k int) error {
	if err := st.home(s, in, w, n, k); err != nil {
		return err
	}
	st.msel = append(st.msel[:0], st.sel...)
	st.mw = append(st.mw[:0], st.hw[:k]...)
	return nil
}

// hostLate runs the host's share of a step -- a split's misses, or a hybrid
// step's whole selection -- while the card runs what was queued before it,
// and queues the sum into dst.
func (st *streamBank) hostLate(sid uint64, g *devTier, s backend.Session, dst backend.Buf, split bool) error {
	t0 := time.Now()
	host := g.hostFor(sid, st)
	if host == nil {
		return fmt.Errorf("tier: this session offered no host side for a hybrid block")
	}
	if len(st.msel) == 0 {
		clear(st.hout)
	} else if err := host(st.msel, st.mw, st.hin, st.hout); err != nil {
		return err
	}
	// The host's arithmetic alone: the write after it can wait for the
	// card to finish what was queued before it.
	if split {
		g.TSplitHost += time.Since(t0)
		g.SplitRuns++
		g.SplitHits += len(st.sel) - len(st.msel)
	} else {
		g.THybrid += time.Since(t0)
	}
	tw := time.Now()
	err := s.WriteAt(dst, 0, f32b(st.hout))
	g.TLateWrite += time.Since(tw)
	return err
}

// splitSwaps is Config.SplitSwaps with its default.
func (g *devTier) splitSwaps() int {
	if g.SplitSwaps != 0 {
		return max(g.SplitSwaps, 0)
	}
	return 8
}

// splitAdmit is Config.SplitAdmit with its default.
func (g *devTier) splitAdmit() int {
	if g.SplitAdmit > 0 {
		return g.SplitAdmit
	}
	return 2
}

// runHost is the hybrid suspension: in (the experts' input, n wide -- the
// expert width, which is the latent on Kimi-K3) and the k routing weights
// come home, the host's experts sum into hout, and the sum is
// written to out, where the combine kernel would have put it. The Sync makes
// the launches that write in land first.
func (st *streamBank) runHost(sid uint64, g *devTier, s backend.Session, in, w, out backend.Buf, n, k int) error {
	if cap(st.hin) < n {
		st.hin = make([]float32, n)
		st.hout = make([]float32, n)
	}
	st.hin, st.hout = st.hin[:n], st.hout[:n]
	if cap(st.hw) < k+1 {
		st.hw = make([]float32, k+1)
	}
	st.hw = st.hw[:k+1]
	t0 := time.Now()
	if err := s.Sync(); err != nil {
		return err
	}
	if err := s.Read(in, f32b(st.hin)); err != nil {
		return err
	}
	if err := s.Read(w, f32b(st.hw)); err != nil {
		return err
	}
	host := g.hostFor(sid, st)
	if host == nil {
		return fmt.Errorf("tier: this session offered no host side for a hybrid block")
	}
	if err := host(st.sel, st.hw[:k], st.hin, st.hout); err != nil {
		return err
	}
	err := s.WriteAt(out, 0, f32b(st.hout))
	g.THybrid += time.Since(t0)
	return err
}

// streamItem is one sheet set a fill sends: expert e into compact slot slot.
type streamItem struct {
	e    uint32
	slot int
}

// initCache sets a block's bank up as an expert cache of n slots when n is
// more than the selection; at n == k there is no cache and fill sends every
// selected expert into slot i.
func (st *streamBank) initCache(n, nExpert int) {
	if n <= len(st.sel) {
		return
	}
	st.slotOf = make([]int32, nExpert)
	for i := range st.slotOf {
		st.slotOf[i] = -1
	}
	st.owner = make([]int32, n)
	for i := range st.owner {
		st.owner[i] = -1
	}
	st.used = make([]uint64, n)
	st.slotSel = make([]uint32, len(st.sel))
}

// uncache returns a block to its plain bank: no cache, nothing held.
func (st *streamBank) uncache() {
	st.slotOf, st.owner, st.used, st.slotSel = nil, nil, nil, nil
	st.heat, st.own, st.hit, st.bank = nil, nil, nil, nil
}

// cached reports whether this block's bank is an expert cache.
func (st *streamBank) cached() bool { return st.owner != nil }

// lookup maps this token's selection onto cache slots: a hit keeps its slot,
// a miss takes the least recently used slot this token does not need, and
// only the misses become items. slotSel is the selection in slots, which the
// kernels index by.
func (st *streamBank) lookup(g *devTier) {
	st.tick++
	for _, e := range st.sel {
		if s := st.slotOf[e]; s >= 0 {
			st.used[s] = st.tick
		}
	}
	for i, e := range st.sel {
		if s := st.slotOf[e]; s >= 0 {
			st.slotSel[i] = uint32(s)
			g.StreamCacheHits++
			continue
		}
		// The victim: the oldest slot not touched this token. There always is
		// one, since the cache holds more slots than a token selects.
		v := -1
		for s := range st.owner {
			if st.used[s] != st.tick && (v < 0 || st.used[s] < st.used[v]) {
				v = s
			}
		}
		if o := st.owner[v]; o >= 0 {
			st.slotOf[o] = -1
		}
		st.owner[v], st.slotOf[e], st.used[v] = int32(e), int32(v), st.tick
		st.slotSel[i] = uint32(v)
		st.items = append(st.items, streamItem{e: e, slot: v})
		g.StreamCacheMisses++
	}
}

// heatHalf is how many tokens a block's own selection counts last at full
// weight, for a session that hands the tier none (nn.LayerWeights.ExpertUses
// is the model's, which halves its own).
const heatHalf = 256

// heatMargin is the floor of the hysteresis a miss clears to take an occupied
// slot: twice the slot expert's selections and this many more.
const heatMargin = 2

// count adds this token's selection, which the card routed, to uses -- the
// session's own counts of the block's experts (nn.LayerWeights.ExpertUses,
// model.Hotness), one source for the model's measure of its experts and for
// the expert cache -- or to the block's own counts when the session handed
// none, halved every heatHalf tokens. The cache keeps by st.heat.
func (st *streamBank) count(uses []uint32) {
	if len(uses) == st.nExpert && st.nExpert > 0 {
		st.heat = uses
	} else {
		if len(st.own) != st.nExpert {
			st.own = make([]uint32, st.nExpert)
		}
		st.heat = st.own
		st.owned++
		if st.owned%heatHalf == 0 {
			for i := range st.heat {
				st.heat[i] >>= 1
			}
		}
	}
	for _, e := range st.sel {
		if int(e) < len(st.heat) && st.heat[e] != ^uint32(0) {
			st.heat[e]++
		}
	}
}

// split maps this token's selection onto a hybrid block's cache without
// evicting anything: a hit runs on the card from its slot, a miss on the
// host. It counts the selection into uses -- the session's own expert counts
// (nn.LayerWeights.ExpertUses, model.Hotness), one source for the model's
// placement and this cache -- or the block's own when the session gave none,
// and returns the hits. A miss's slot
// is the first hit's, so the card's kernels read a sheet that holds an
// expert (its weight is zero in dw); with no hit nothing runs on the card.
func (st *streamBank) split(g *devTier) int {
	if st.hit == nil {
		st.hit = make([]bool, len(st.sel))
	}
	st.tick++
	hits, first := 0, uint32(0)
	for i, e := range st.sel {
		sl := st.slotOf[e]
		st.hit[i] = sl >= 0
		if sl < 0 {
			continue
		}
		st.used[sl] = st.tick
		st.slotSel[i] = uint32(sl)
		if hits == 0 {
			first = uint32(sl)
		}
		hits++
	}
	for i := range st.sel {
		if !st.hit[i] {
			st.slotSel[i] = first
		}
	}
	g.StreamCacheHits += hits
	g.StreamCacheMisses += len(st.sel) - hits
	return hits
}

// splitWeights divides the routing weights w (the selection's order) into
// the card's, a miss's zeroed, and the host's misses with theirs.
func (st *streamBank) splitWeights(w []float32) {
	st.dw = append(st.dw[:0], w...)
	st.msel, st.mw = st.msel[:0], st.mw[:0]
	for i, e := range st.sel {
		if !st.hit[i] {
			st.dw[i] = 0
			st.msel = append(st.msel, e)
			st.mw = append(st.mw, w[i])
		}
	}
}

// admit takes this token's misses into a hybrid block's cache, each into the
// coldest slot this token does not read, when the miss has been selected more
// often than that slot's expert (an empty slot takes any). At most max of
// them change a slot's expert; filling an empty slot spends neither max nor
// the swap budget, since the only cost is the sheet, once, and a cache left
// empty is all misses. The misses are the token's own, so a fill is bounded by
// its selection. The uploads are queued behind the launches that read this token's
// slots, so a slot changes only after its last reader.
func (st *streamBank) admit(g *devTier, s backend.Session, l *layer, nExpert, max int) error {
	st.items = st.items[:0]
	// The swaps -- a slot changing hands -- share a budget per token across
	// the blocks (Config.SplitSwaps); filling an empty slot does not spend it.
	// A block at or before the last one that admitted starts a new token.
	if st.li <= g.swapLi {
		g.swapsLeft = g.splitSwaps()
	}
	g.swapLi = st.li
	nswap := 0
	for i, e := range st.sel {
		if st.hit[i] || st.slotOf[e] != -1 {
			continue
		}
		v := -1
		for sl, o := range st.owner {
			if st.used[sl] == st.tick {
				continue
			}
			if o < 0 {
				v = sl
				break
			}
			if v < 0 || st.heat[o] < st.heat[st.owner[v]] {
				v = sl
			}
		}
		if v < 0 {
			break
		}
		if o := st.owner[v]; o >= 0 {
			// A slot changes hands only to an expert selected clearly more
			// often: a sheet costs a transfer, and two experts trading one
			// slot back and forth pay it every token.
			if st.heat[e] < 2*st.heat[o]+heatMargin || g.swapsLeft <= 0 || nswap >= max {
				continue
			}
			nswap++
			g.swapsLeft--
			st.slotOf[o] = -1
		}
		st.owner[v], st.slotOf[e], st.used[v] = int32(e), slotInFlight, st.tick
		st.items = append(st.items, streamItem{e: e, slot: v})
	}
	if len(st.items) == 0 {
		return nil
	}
	// The sheets are sent later, where the host would wait anyway: at the
	// next block's wait for its router, or at the call's end (flushAdmits).
	lo := len(g.admItems)
	g.admItems = append(g.admItems, st.items...)
	g.adm = append(g.adm, admJob{st: st, l: l, nExpert: nExpert, lo: lo, hi: len(g.admItems)})
	return nil
}

// slotInFlight marks an expert whose sheet is on its way to a slot: a miss
// until the sheet has landed (finishAdmits), and not admitted again.
const slotInFlight = -2

// admJob is one block's admissions waiting to be sent: g.admItems[lo:hi].
type admJob struct {
	st      *streamBank
	l       *layer
	nExpert int
	lo, hi  int
}

// flushAdmits sends the admissions admit queued. It runs where the host waits
// on the card anyway -- before a block's router readback, and at the call's
// end -- so a sheet's page read and host copy are off the token's path. On a
// session with a copy queue (backend.Uploader) the sheets are copied into a
// page-locked ring and sent beside the launches, and the slots are taken
// into use at the call's end (finishAdmits), once the copies have landed;
// otherwise they are sent in order with the launches (put) and in use at once.
func (g *devTier) flushAdmits(s backend.Session) error {
	if len(g.adm) == 0 {
		return nil
	}
	defer func(t0 time.Time) { g.TAdmit += time.Since(t0) }(time.Now())
	up, _ := s.(backend.Uploader)
	if up != nil && !g.upRingReady() {
		up = nil
	}
	var err error
	for _, j := range g.adm {
		st := j.st
		items := g.admItems[j.lo:j.hi]
		st.want = st.want[:0]
		for _, it := range items {
			st.want = append(st.want, it.e)
		}
		// Sheets the expert card serves are not read on the host at all.
		if !st.peerAll(g) {
			switch {
			case st.pre != nil:
				err = st.pre(st.want)
			case st.ensure != nil:
				err = st.ensure(&st.w)
			}
			if err == nil && st.sel2 != nil {
				err = st.sel2(&st.w, st.want)
			}
		}
		if err == nil && up != nil {
			err = st.upload(g, up, j.l, j.nExpert, items)
			if err == nil {
				g.admPub = append(g.admPub, j)
				g.SplitQueued += len(items)
			}
		} else if err == nil {
			st.items = append(st.items[:0], items...)
			if err = st.put(g, s, j.l, j.nExpert, [2]int{0, len(items)}); err == nil {
				st.publish(items)
			}
		}
		if err != nil {
			// A sheet half sent is not the expert the map names.
			st.forget()
			break
		}
		g.SplitAdmits += len(items)
	}
	g.adm = g.adm[:0]
	return err
}

// finishAdmits sends what is queued, waits for the copy queue and takes the
// landed slots into use. It runs at the end of every call that admitted,
// so no sheet is in flight between calls (dropCaches frees banks there).
func (g *devTier) finishAdmits(s backend.Session) error {
	err := g.flushAdmits(s)
	if len(g.admPub) > 0 {
		t0 := time.Now()
		if up, ok := s.(backend.Uploader); ok {
			if werr := up.UploadWait(); err == nil {
				err = werr
			}
		}
		for _, j := range g.admPub {
			if err != nil {
				j.st.forget()
				continue
			}
			j.st.publish(g.admItems[j.lo:j.hi])
		}
		g.admPub = g.admPub[:0]
		g.TAdmit += time.Since(t0)
	}
	if len(g.adm) == 0 {
		g.admItems = g.admItems[:0]
	}
	g.ringUsed = 0
	return err
}

// publishAdmits waits for the copy queue and takes every sheet sent so far
// into use, before a block's split reads the cache map.
func (g *devTier) publishAdmits(s backend.Session) error {
	t0 := time.Now()
	var err error
	if up, ok := s.(backend.Uploader); ok {
		err = up.UploadWait()
	}
	for _, j := range g.admPub {
		if err != nil {
			j.st.forget()
			continue
		}
		j.st.publish(g.admItems[j.lo:j.hi])
	}
	g.admPub = g.admPub[:0]
	g.TAdmit += time.Since(t0)
	return err
}

// predictAdmit queues the predicted experts of this block (st.pred, from the
// block before's row through this block's router) that the cache does not
// hold, at most max, each into the coldest slot whose expert is selected no
// more often than the prediction -- a prediction only moves sheets, and the
// real router still decides. They are sent where the host waits before this
// block's router (flushAdmits) and taken into use after it (publishAdmits).
func (st *streamBank) predictAdmit(g *devTier, l *layer, nExpert, max int) {
	if st.heat == nil || st.owner == nil {
		return
	}
	st.items = st.items[:0]
	for _, e := range st.pred {
		if len(st.items) >= max {
			break
		}
		if int(e) >= len(st.slotOf) || st.slotOf[e] != -1 {
			continue
		}
		v := -1
		for sl, o := range st.owner {
			if o >= 0 && int(o) < len(st.slotOf) && st.slotOf[o] == slotInFlight {
				continue
			}
			if o < 0 {
				v = sl
				break
			}
			if v < 0 || st.heat[o] < st.heat[st.owner[v]] {
				v = sl
			}
		}
		if v < 0 {
			break
		}
		if o := st.owner[v]; o >= 0 {
			if st.heat[o] > st.heat[e] {
				continue
			}
			st.slotOf[o] = -1
		}
		st.owner[v], st.slotOf[e] = int32(e), slotInFlight
		st.items = append(st.items, streamItem{e: e, slot: v})
	}
	if len(st.items) == 0 {
		return
	}
	g.SplitPredicted += len(st.items)
	lo := len(g.admItems)
	g.admItems = append(g.admItems, st.items...)
	g.adm = append(g.adm, admJob{st: st, l: l, nExpert: nExpert, lo: lo, hi: len(g.admItems)})
}

// splitPredict is Config.SplitPredict.
func (g *devTier) splitPredict() int { return max(g.SplitPredict, 0) }

// publish takes landed sheets into use: an expert whose slot is still its own
// (no forget or later eviction moved it) is a hit from the next token.
func (st *streamBank) publish(items []streamItem) {
	for _, it := range items {
		if st.owner != nil && it.slot < len(st.owner) && st.owner[it.slot] == int32(it.e) &&
			int(it.e) < len(st.slotOf) && st.slotOf[it.e] == slotInFlight {
			st.slotOf[it.e] = int32(it.slot)
		}
	}
}

// upRingBytes is the page-locked ring flushAdmits copies sheets through.
const upRingBytes = 64 << 20

// upRingReady allocates the ring on first use, and reports false where the
// device has no page-locked memory.
func (g *devTier) upRingReady() bool {
	if g.upRing != nil {
		return true
	}
	hp, ok := g.dev.(backend.HostPinner)
	if !ok || g.noPin || g.StreamNoPin {
		return false
	}
	b, err := hp.PinHost(upRingBytes)
	if err != nil {
		return false
	}
	g.upRing = b
	return true
}

// upload copies items' sheets through the ring onto the copy queue, at their
// slots. A sheet plane larger than the ring goes the ordered way.
func (st *streamBank) upload(g *devTier, up backend.Uploader, l *layer, nExpert int, items []streamItem) error {
	for m, pair := range [3]struct {
		r *resident
		w nn.Weight
	}{{l.gate, st.w.Gate}, {l.up, st.w.Up}, {l.down, st.w.Down}} {
		if m == 0 && pair.r == nil && pair.w.Packed == nil && pair.w.Data == nil {
			continue
		}
		if pair.r == nil || !pair.r.ok || pair.w.Packed == nil {
			return fmt.Errorf("tier: streamed bank %d has no host side", m)
		}
		pk := pair.w.Packed
		for pl := 0; pl < 3; pl++ {
			n := st.sh[m][pl]
			if n == 0 {
				continue
			}
			if n > len(g.upRing) {
				return fmt.Errorf("tier: a %d-byte sheet plane is wider than the upload ring", n)
			}
			dst := [3]backend.Buf{pair.r.qs, pair.r.d, pair.r.sc}[pl]
			for _, it := range items {
				if int(it.e) >= nExpert {
					return fmt.Errorf("tier: the router selected expert %d of %d", it.e, nExpert)
				}
				if ok, err := st.peerSheet(g, m, pl, int(it.e), it.slot, n, dst); err != nil {
					return err
				} else if ok {
					continue
				}
				if g.ringUsed+n > len(g.upRing) {
					// Every copy from the ring has landed before it is reused.
					if err := up.UploadWait(); err != nil {
						return err
					}
					g.ringUsed = 0
				}
				buf := g.upRing[g.ringUsed : g.ringUsed+n : g.ringUsed+n]
				if pk.Sheet != nil {
					sp, rel, err := pk.Sheet(int(it.e))
					if err != nil {
						rel()
						return err
					}
					src := [3][]byte{sp.QS, sp.D, sp.SC}[pl]
					if len(src) != n {
						rel()
						return fmt.Errorf("tier: expert %d plane %d is %d bytes, want %d", it.e, pl, len(src), n)
					}
					copy(buf, src)
					rel()
				} else {
					whole := [3][]byte{pk.QS, pk.D, pk.SC}[pl]
					off := int(it.e) * n
					if off+n > len(whole) {
						return fmt.Errorf("tier: expert %d plane %d wants bytes %d..%d of %d",
							it.e, pl, off, off+n, len(whole))
					}
					copy(buf, whole[off:off+n])
				}
				g.ringUsed += n
				if err := up.Upload(dst, it.slot*n, buf); err != nil {
					return err
				}
				g.StreamBytes += int64(n)
			}
		}
	}
	return nil
}

// forget empties the cache, for when the bank's bytes can no longer be
// trusted to be the experts the map says (a fill that failed midway).
func (st *streamBank) forget() {
	for i := range st.slotOf {
		st.slotOf[i] = -1
	}
	for i := range st.owner {
		st.owner[i] = -1
	}
}

// cacheSlotsFor is a streamed block's bank size in sheets: the selection, or
// want (Config.StreamCacheSlots) when that is bigger and the device has room for
// the block's own bank of that many beside what it already holds. A model
// whose expert kernels read the selection itself (DenseMoE's per-expert
// scale) keeps the plain bank: the cache rewrites the selection as slots.
func (g *devTier) cacheSlotsFor(p *nn.LayerPlan, ws []nn.Weight, want, slots, bank int, biased bool) int {
	// Nor one whose experts carry biases (gpt-oss): the bias kernels index
	// them by the selection, which the cache rewrites as slots. A second
	// buffer of true ids would lift it.
	if p.DenseMoE || biased {
		return slots
	}
	var sheet uint64
	for i := 4; i <= 6; i++ {
		x := ws[i]
		if len(x.Data) == 0 && x.Packed == nil {
			continue
		}
		q, ok := quantOf(x.T)
		if !ok {
			return slots
		}
		pq, pd, psc, err := kernels.PackedWords(q, x.Rows/slots, x.K)
		if err != nil {
			return slots
		}
		sheet += uint64(pq+pd+psc) * 4
	}
	n := min(want, bank)
	if n <= slots {
		return slots
	}
	need := sheet * uint64(n)
	if !g.room(need) {
		g.StreamCacheShort++
		return slots
	}
	g.StreamCacheSize = max(g.StreamCacheSize, n)
	g.StreamCacheSlots += n
	g.StreamCacheBlocks++
	return n
}

// streamRead is one group's read in flight.
type streamRead struct {
	wg  sync.WaitGroup
	err error
}

// read is group gi's read, run on its own goroutine. A method rather than a
// closure, so issuing it allocates nothing on the heap.
func (st *streamBank) read(gi int, gr [2]int) {
	defer st.rd[gi].wg.Done()
	st.rd[gi].err = st.pre(st.want[gr[0]:gr[1]])
}

// groups splits the selection into contiguous runs so a read can overlap an
// upload. One group is the unpipelined path and is what a host with no
// PrefetchExperts gets.
func (st *streamBank) groups(g *devTier, k int) [][2]int {
	n := g.streamGroups(g.StreamBlocks)
	if st.pre == nil || n < 2 || k < 2 {
		st.grp = append(st.grp[:0], [2]int{0, k})
		return st.grp
	}
	if n > k {
		n = k
	}
	st.grp = st.grp[:0]
	per := (k + n - 1) / n
	for lo := 0; lo < k; lo += per {
		hi := min(lo+per, k)
		st.grp = append(st.grp, [2]int{lo, hi})
	}
	return st.grp
}

// directSheet is the plane size from which a sheet is sent straight out of
// its host frame instead of being gathered first. A gather turns many small
// transfers into one, which pays below a megabyte; above it the per-call cost
// is noise and the gather is a second pass over the bytes -- on Kimi-K3 the
// largest term of the fill, 96 ms of 182 (placement.md 16c).
const directSheet = 1 << 20

// put uploads one group of sheets into the compact bank. The compact bank is k
// sheets back to back, so slots [lo, hi) are the byte range [lo*n, hi*n) of
// every plane. A plane of directSheet or more is sent sheet by sheet from the
// host frame; a smaller one is gathered into the device's staging buffer and
// sent once.
func (st *streamBank) put(g *devTier, s backend.Session, l *layer, nExpert int, gr [2]int) (err error) {
	items := st.items[gr[0]:gr[1]]
	// Without a cache the items are slots lo..hi-1 in order, so a small plane
	// can be gathered into one transfer; a cache's misses land in scattered
	// slots and every plane is sent sheet by sheet.
	lo, hi := items[0].slot, items[0].slot+len(items)
	contiguous := !st.cached()
	g.pieces = g.pieces[:0]
	// Every held sheet is released however put leaves, and the direct ones
	// are sent first.
	defer func() {
		if err == nil {
			err = g.sendPieces(s, g.pieces)
		}
		for i := range g.pieces {
			g.pieces[i].release()
			g.pieces[i] = sheetPiece{}
		}
		g.pieces = g.pieces[:0]
	}()
	for m, pair := range [3]struct {
		r *resident
		w nn.Weight
	}{{l.gate, st.w.Gate}, {l.up, st.w.Up}, {l.down, st.w.Down}} {
		// Ungated experts (Nemotron 3) have no gate bank to send.
		if m == 0 && pair.r == nil && pair.w.Packed == nil && pair.w.Data == nil {
			continue
		}
		if pair.r == nil || !pair.r.ok || pair.w.Packed == nil {
			return fmt.Errorf("tier: streamed bank %d has no host side", m)
		}
		pk := pair.w.Packed
		for pl := 0; pl < 3; pl++ {
			n := st.sh[m][pl]
			if n == 0 {
				continue // this format has no such plane
			}
			dst := [3]backend.Buf{pair.r.qs, pair.r.d, pair.r.sc}[pl]
			direct := n >= g.directBytes() || !contiguous
			var buf []byte
			if !direct {
				buf = g.streamStage(m, pl, n*(hi-lo))
			}
			for _, it := range items {
				e, i := it.e, it.slot
				// The bound is checked because the index was read back from a device;
				// a garbage word would otherwise be a panic mid-token.
				if int(e) >= nExpert {
					return fmt.Errorf("tier: the router selected expert %d of %d", e, nExpert)
				}
				// A cache's miss the expert card holds is copied card to card.
				if !contiguous {
					if ok, err := st.peerSheet(g, m, pl, int(e), i, n, dst); err != nil {
						return err
					} else if ok {
						continue
					}
				}
				// A bank in expert pages hands over one sheet at a time; a
				// contiguous one is sliced at e*n.
				var src []byte
				release := func() {}
				ts := time.Now()
				if pk.Sheet != nil {
					sp, rel, err := pk.Sheet(int(e))
					release = rel
					if err != nil {
						release()
						return err
					}
					src = [3][]byte{sp.QS, sp.D, sp.SC}[pl]
				} else {
					whole := [3][]byte{pk.QS, pk.D, pk.SC}[pl]
					off := int(e) * n
					if off+n > len(whole) {
						return fmt.Errorf("tier: expert %d plane %d wants bytes %d..%d of %d",
							e, pl, off, off+n, len(whole))
					}
					src = whole[off : off+n]
				}
				if len(src) != n {
					release()
					return fmt.Errorf("tier: expert %d plane %d is %d bytes, want %d", e, pl, len(src), n)
				}
				tc := time.Now()
				g.TStreamSheet += tc.Sub(ts)
				if direct {
					// The frame stays held until sendPieces has copied it out.
					g.pieces = append(g.pieces, sheetPiece{dst: dst, off: i * n, src: src, release: release})
					continue
				}
				copy(buf[(i-lo)*n:(i-lo+1)*n], src)
				release()
				g.TStreamCopy += time.Since(tc)
			}
			if direct {
				continue
			}
			th := time.Now()
			if err := s.WriteAt(dst, lo*n, buf); err != nil {
				return err
			}
			g.TStreamH2D += time.Since(th)
			g.StreamBytes += int64(len(buf))
			g.StreamGathered++
		}
	}
	return nil
}

// sheetPiece is one direct sheet plane: where it goes on the card, its bytes
// in a held host frame, and the release of that hold.
type sheetPiece struct {
	dst     backend.Buf
	off     int
	src     []byte
	release func()
}

// pinHalf is the size of each page-locked half: two of Kimi-K3's 5.6 MiB
// planes. The first half of every group is copied before any transfer can
// start, so a smaller half is less of the group spent waiting; 32 MiB cost
// 29 s of copy wait over a 32-token run, and a sheet a group 62 s
// (placement.md 16c).
const pinHalf = 12 << 20

// copyChunk is the most one copy goroutine moves: a 5.6 MiB plane copied by
// one core ran under the link, so a half is copied by several.
const copyChunk = 1 << 20

// sendPieces puts direct sheets on the card. With page-locked memory it is a
// two-stage pipeline: the pieces are copied into one half (concurrently, one
// goroutine a piece, since a single memcpy from a remote node runs well under
// the link) while the other half is transferred, and a transfer out of
// page-locked memory runs at the link's rate where a pageable one went
// through the driver's bounce buffer at 5.76 GiB/s (placement.md 16c).
// Without it each piece is a pageable transfer from its frame.
func (g *devTier) sendPieces(s backend.Session, ps []sheetPiece) error {
	if len(ps) == 0 {
		return nil
	}
	pin := g.pinHalves(ps)
	if pin[0] == nil {
		for i := range ps {
			th := time.Now()
			if err := s.WriteAt(ps[i].dst, ps[i].off, ps[i].src); err != nil {
				return err
			}
			g.TStreamH2D += time.Since(th)
			g.StreamBytes += int64(len(ps[i].src))
			g.StreamDirect++
		}
		return nil
	}
	tc := time.Now()
	i, j := 0, g.packHalf(pin[0], ps, 0)
	g.TStreamCopy += time.Since(tc)
	cur := 0
	for i < len(ps) {
		next := j
		if j < len(ps) {
			g.packWG.Add(1)
			go g.packNext(pin[1-cur], ps, j)
		}
		off := 0
		var err error
		th := time.Now()
		for k := i; k < j && err == nil; k++ {
			n := len(ps[k].src)
			err = s.WriteAt(ps[k].dst, ps[k].off, pin[cur][off:off+n])
			off += n
			g.StreamBytes += int64(n)
			g.StreamDirect++
			g.StreamPinned++
		}
		g.TStreamH2D += time.Since(th)
		if j < len(ps) {
			// Joined even on failure: the copy reads frames put releases.
			tw := time.Now()
			g.packWG.Wait()
			g.TStreamCopy += time.Since(tw)
			next = g.packEnd
		}
		if err != nil {
			return err
		}
		i, j, cur = j, next, 1-cur
	}
	return nil
}

// pinHalves returns the two page-locked halves, each big enough for the
// largest piece, allocating them on first use; nils when the device has no
// page-locked memory or the caller turned it off.
func (g *devTier) pinHalves(ps []sheetPiece) [2][]byte {
	hp, ok := g.dev.(backend.HostPinner)
	if !ok || g.noPin || g.StreamNoPin {
		return [2][]byte{}
	}
	need := g.pinHalfBytes()
	for i := range ps {
		need = max(need, len(ps[i].src))
	}
	for h := range g.pin {
		if len(g.pin[h]) >= need {
			continue
		}
		if g.pin[h] != nil {
			hp.UnpinHost(g.pin[h])
			g.pin[h] = nil
		}
		b, err := hp.PinHost(need)
		if err != nil {
			// Page-locked memory is a speed, not a requirement: the pageable
			// path is correct, so a refusal turns this off for the device.
			g.noPin = true
			for k, b := range g.pin {
				if b != nil {
					hp.UnpinHost(b)
					g.pin[k] = nil
				}
			}
			return [2][]byte{}
		}
		g.pin[h] = b
	}
	return g.pin
}

// packNext is packHalf on its own goroutine, its end left in g.packEnd.
func (g *devTier) packNext(buf []byte, ps []sheetPiece, i int) {
	defer g.packWG.Done()
	g.packEnd = g.packHalf(buf, ps, i)
}

// packHalf copies pieces from i into buf, back to back, as many as fit, one
// goroutine a copyChunk, and returns the index after the last one copied. At
// least one always fits: pinHalves sized buf for the largest.
func (g *devTier) packHalf(buf []byte, ps []sheetPiece, i int) int {
	// The half in use is the tuned size, never more than was locked: a half
	// tuned down keeps its bigger buffer and uses a prefix of it.
	if h := g.pinHalfBytes(); h < len(buf) {
		big := 0
		for k := i; k < len(ps); k++ {
			big = max(big, len(ps[k].src))
		}
		buf = buf[:max(h, big)]
	}
	j, n := i, 0
	for j < len(ps) && n+len(ps[j].src) <= len(buf) {
		n += len(ps[j].src)
		j++
	}
	off := 0
	for k := i; k < j; k++ {
		src := ps[k].src
		for c := 0; c < len(src); c += copyChunk {
			e := min(c+copyChunk, len(src))
			g.copyWG.Add(1)
			go g.copyPiece(buf[off+c:off+e], src[c:e])
		}
		off += len(src)
	}
	g.copyWG.Wait()
	return j
}

// copyPiece is one of packHalf's copies. Only one packHalf runs at a time on
// a device, so the WaitGroup is the device's and nothing is allocated.
func (g *devTier) copyPiece(dst, src []byte) {
	defer g.copyWG.Done()
	copy(dst, src)
}

// streamStage is the reused host buffer one plane's gathered sheets go into.
// One set per device, not per block: a device fills its streamed blocks one
// after another, and a set per block was 24 GiB of heap on Kimi-K3 that the
// runtime gave back and faulted in again every token (placement.md 16c).
func (g *devTier) streamStage(m, pl, n int) []byte {
	i := m*3 + pl
	if len(g.sheetStage[i]) < n {
		g.sheetStage[i] = make([]byte, n)
	}
	return g.sheetStage[i][:n]
}

// prefillWaveRows is the fewest valid rows of a chunk whose hybrid block runs
// its experts on the card in waves (streamBank.prefillWaves); fewer go through
// splitRows, whose host share is a few experts' reads. A first cut, to be set
// from a paired measurement.
const prefillWaveRows = 64

// prefillWaves runs a prompt chunk's routed experts of a hybrid block on the
// card, through the block's expert cache, instead of reading every chosen
// expert on the host. The chunk's selection comes home; the experts the cache
// holds run first (wave 0), then the rest are admitted into the least
// recently used slots, as many as the cache has slots at a time, and each
// wave runs the grouped kernels over what the card holds then. A pair whose
// expert is not in the wave names none (nExpert, which the grouping routes to
// its spare column) with a zero weight, so run's weighted sum is that wave's
// share and the caller accumulates the waves. run(first) executes one wave:
// its first call writes the sum, the others add to it.
//
// The sheets travel the admission path (queued on admItems, sent through the
// upload ring and the copy queue, taken into use after UploadWait), so the
// pager, the ring and the cache map are the decode path's. A wave starts after
// a Sync, which is also what makes its slots free: the previous wave's
// kernels have read them.
func (st *streamBank) prefillWaves(g *devTier, s backend.Session, l *layer, nExpert int, sel, w backend.Buf, k, rows, valid int, run func(first bool) error) error {
	if len(st.owner) > nExpert {
		return fmt.Errorf("tier: a cache of %d slots over a bank of %d experts", len(st.owner), nExpert)
	}
	if st.bank != l.down {
		st.forget()
		st.bank = l.down
	}
	st.hsel = growU32(st.hsel, rows*(k+1))
	st.hw = growF32(st.hw, rows*k)
	st.rowSel = growU32(st.rowSel, rows*(k+1))
	st.rowW = growF32(st.rowW, rows*k)
	if err := s.Sync(); err != nil {
		return err
	}
	if err := s.Read(w, f32b(st.hw)); err != nil {
		return err
	}
	if err := s.Read(sel, u32b(st.hsel)); err != nil {
		return err
	}
	if len(st.wslot) != nExpert {
		st.wslot = make([]int32, nExpert)
		for i := range st.wslot {
			st.wslot[i] = -1
		}
		// The wave lists hold at most every expert once: sized here, they
		// never grow in a step.
		st.waveHit = make([]uint32, 0, nExpert)
		st.waveMiss = make([]uint32, 0, nExpert)
	}
	// The chunk's experts, each once: those the cache holds, then the rest.
	// wslot marks an expert seen (waveSeen) and holds a running wave's
	// expert's slot.
	st.waveHit, st.waveMiss = st.waveHit[:0], st.waveMiss[:0]
	defer func() {
		for _, e := range st.waveHit {
			st.wslot[e] = -1
		}
		for _, e := range st.waveMiss {
			st.wslot[e] = -1
		}
		if len(g.adm) == 0 && len(g.admPub) == 0 {
			g.admItems = g.admItems[:0]
		}
	}()
	for r := 0; r < valid; r++ {
		for j := 0; j < k; j++ {
			e := st.hsel[r*(k+1)+j]
			if int(e) >= nExpert {
				return fmt.Errorf("tier: the router selected expert %d of %d", e, nExpert)
			}
			if st.wslot[e] != -1 {
				continue
			}
			st.wslot[e] = waveSeen
			if st.slotOf[e] >= 0 {
				st.waveHit = append(st.waveHit, e)
			} else {
				st.waveMiss = append(st.waveMiss, e)
			}
		}
	}
	if moeFaulted("prefillalloc") {
		// The violation: the lists are rebuilt per chunk, a heap allocation a
		// warm step must not make.
		st.waveHit = append([]uint32(nil), st.waveHit...)
	}
	g.PrefillBlocks++
	g.PrefillHits += len(st.waveHit)
	first := true
	// wave runs the grouped kernels over the experts whose wslot is a slot.
	wave := func() error {
		if !first {
			// The previous wave's kernels have read their slots and hsum.
			if err := s.Sync(); err != nil {
				return err
			}
		}
		for r := 0; r < rows; r++ {
			for j := 0; j < k; j++ {
				id, wt := uint32(nExpert), float32(0)
				if r < valid {
					if sl := st.wslot[st.hsel[r*(k+1)+j]]; sl >= 0 {
						id, wt = uint32(sl), st.hw[r*k+j]
					} else if moeFaulted("prefillunnamed") {
						// The violation: a pair whose expert is not in the wave
						// reads some slot's sheets with its own weight.
						id, wt = st.hsel[r*(k+1)+j]%uint32(len(st.owner)), st.hw[r*k+j]
					}
				}
				st.rowSel[r*(k+1)+j], st.rowW[r*k+j] = id, wt
			}
			st.rowSel[r*(k+1)+k] = st.hsel[r*(k+1)+k]
		}
		if err := s.WriteAt(sel, 0, u32b(st.rowSel)); err != nil {
			return err
		}
		if err := s.WriteAt(w, 0, f32b(st.rowW)); err != nil {
			return err
		}
		g.PrefillWaves++
		err := run(first)
		first = false
		return err
	}
	st.tick++
	if len(st.waveHit) > 0 {
		for _, e := range st.waveHit {
			sl := st.slotOf[e]
			st.used[sl] = st.tick
			st.wslot[e] = sl
		}
		if err := wave(); err != nil {
			return err
		}
		for _, e := range st.waveHit {
			st.wslot[e] = waveSeen
		}
	}
	for lo := 0; lo < len(st.waveMiss); lo += len(st.owner) {
		exps := st.waveMiss[lo:min(lo+len(st.owner), len(st.waveMiss))]
		if !first {
			if err := s.Sync(); err != nil {
				return err
			}
		}
		st.tick++
		if cap(st.items) < len(st.owner) {
			st.items = make([]streamItem, 0, len(st.owner))
		}
		st.items = st.items[:0]
		for _, e := range exps {
			v := -1
			for sl := range st.owner {
				if st.used[sl] != st.tick && (v < 0 || st.used[sl] < st.used[v]) {
					v = sl
				}
			}
			if o := st.owner[v]; o >= 0 {
				st.slotOf[o] = -1
			}
			st.owner[v], st.slotOf[e], st.used[v] = int32(e), slotInFlight, st.tick
			st.items = append(st.items, streamItem{e: e, slot: v})
		}
		jlo := len(g.admItems)
		g.admItems = append(g.admItems, st.items...)
		g.adm = append(g.adm, admJob{st: st, l: l, nExpert: nExpert, lo: jlo, hi: len(g.admItems)})
		if err := g.flushAdmits(s); err != nil {
			return err
		}
		if err := g.publishAdmits(s); err != nil {
			return err
		}
		g.PrefillUploaded += len(exps)
		for _, it := range st.items {
			st.wslot[it.e] = int32(it.slot)
		}
		if err := wave(); err != nil {
			return err
		}
		for _, e := range exps {
			st.wslot[e] = waveSeen
		}
	}
	return nil
}

// waveSeen marks an expert of the chunk that is not in the running wave.
const waveSeen = -3
