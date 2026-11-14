package gpu

import (
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"time"
	"unsafe"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// A whole transformer block on the device.
//
// The point is one Session per block with exactly one Read: serving matvecs one
// at a time costs a blocking round trip each, which alone was a quarter of a
// gemma token. That is also why every op between the matvecs is a kernel -- a
// single host-side RMSNorm would drag the residual back across the bus.

// layer is one block's device-resident state.
type layer struct {
	// cut, where set, is the width of the block's crossing row: the layer
	// is the first half of a block cut in two (nn.LayerPlan.Cut), writing
	// its rows to the scratch's cut, read back to the session's (SetCut).
	cut int
	// ngram is an n-gram block's table, read in a call at a time.
	ngram *ngramRows
	// graph is the block as ops from the model's front end
	// (nn.LayerPlan.Graph): what serving it whole forms (served.go), and
	// win and chunked the window its attention states.
	graph   *graph.Graph
	win     int
	chunked bool
	// vecs is every vector the file stores for the block that no field below
	// holds (nn.LayerWeights.Vectors), uploaded as the file stores it.
	vecs           []roleBuf
	wq, wk, wv, wo *resident
	gate, up, down *resident
	// qwen3next's shared expert: an always-on feed-forward added to the routed
	// sum, scaled by a sigmoid of one logit (its gate is in vecs).
	shGate, shUp, shDown *resident
	// extra is every other matrix the block stores and its graph binds, by
	// role (nn.LayerWeights.Extra), uploaded as the file stores it.
	extra []roleRes
	// pleOff is the block's li*PLEDim, the offset of its slice in a row's
	// per-layer inputs.
	pleOff backend.Buf
	sparse bool
	// kvSrc is the block whose history this one attends to: its own index, or
	// for a KV-sharing block (nn.LayerPlan.KVShared) the source's, whose
	// cache this block reads and never writes.
	kvSrc int
	// runs is the sessions that placed the block here. A block holding no
	// history of its own -- a KV-sharing one, a tower's -- is still theirs.
	runs map[uint64]struct{}
	// A linear block's three matrices (its vectors are in vecs). Wq above
	// carries the fused q|k|v for such a block (the layer kind decides the
	// consumer, as in nn.SSMWeights), so these are the z projection, the
	// beta/alpha head and the output projection. linear says this block runs
	// the gated delta rule instead of attention. It is set from the plan, not
	// inferred from a nil weight.
	linear                 bool
	ssmGate, ssmBA, ssmOut *resident
	// KDA's two low-rank gates, nil on qwen3next.
	ssmFA, ssmFB, ssmGA, ssmGB *resident
	// ssmTaps is the convolution's taps a channel's contiguous, which the
	// composer's convolutions read; the file stores them tap-major.
	ssmTaps resident
	// withAttn is a block that runs attention beside its mixer
	// (nn.RecurrentPlan.WithAttn): it keeps a KV history and a recurrent
	// state both, and its mixer's projection is one of extra, wq being the
	// attention's query.
	withAttn bool
	// noFFN is a block that is its mixer alone (nn.LayerPlan.NoFFN): the
	// emit ends it at the first residual.
	noFFN bool
	// Multi-head latent attention's five matrices and two norms. wkb and wvb are
	// banks, one sheet per head, so their matvecs carry slots like a mixture's and
	// the "expert" indexed is the head. wq carries the one-step query where the
	// architecture has one and is empty where wqa/wqb do; wk and wv are always
	// empty. mla is set from the plan, like linear.
	mla                      bool
	wqa, wqb, wkva, wkb, wvb *resident
	// wide marks a block of several residual streams or of experts at a
	// width of their own.
	wide bool

	nAttn, nFFN backend.Buf // norm weights, f32
	// clampB is a vision block's clipped-linear bounds (nn.LayerWeights.Clamp,
	// Gemma 4), 28 floats the clamp kernels read; nil elsewhere.
	clampB backend.Buf
	// LayerNorm biases, nil for every block that RMSNorms.
	nAttnB, nFFNB backend.Buf
	nQ, nK        backend.Buf // per-head q/k RMSNorm weights (qwen3), nil otherwise
	// The KV cache is per sequence, not per block: nn.LayerDevice.Layers carries a
	// position and no state identity, so two model.instances sharing one pair would
	// silently corrupt each other. See docs/design/device-sessions.md.
	//
	// kv is keyed by session id; the weights, norms and biases above are the
	// model's and stay shared. [MaxSeq*KVDim] f32 per entry.
	kv map[uint64]*kvPair
	// rec is the recurrent state of a linear block, per session: which half
	// of pool the session reads. It is a separate map from kv because the two
	// never coexist: a block keeps an attention history or a running summary,
	// never both.
	rec map[uint64]*recPair
	// pool holds every session's recurrent state for a linear block, each at
	// its seat's slots (recpool.go); nil until a session needs it.
	pool *recPool
	// nonCausal says the block's rows attend over their call's key runs and
	// keep no history (nn.LayerPlan.NonCausal): a call over it is one whole
	// run, submitted in its geometry's scratch set (geoNonCausal).
	nonCausal bool
	// windowed says the block attends inside the windowed runs its call was
	// handed (nn.LayerPlan.Windowed, SetKeyRuns). Per block: the scratch's
	// plan is its first block's, and Qwen2.5-VL's blocks alternate.
	windowed bool
	// picPool is the device's transient pair as the pool a composed
	// non-causal block reads it through (towerStep).
	picPool kvLayerPool
	// A non-causal block's own MLP and rotary, which may differ from the
	// first block's the set was built from: an encoder's decision head
	// (Laya's) is an ungated ReLU MLP wider than the encoder's gated GELU,
	// and rotates nothing where the encoder rotates every head. nffn is its
	// width, ungated whether it has no gate, act its activation, noRope
	// that it rotates nothing. Read on the non-causal path only.
	nffn    int
	ungated bool
	noRope  bool
	// swa says the block attends at the sliding layers' geometry, so it runs
	// in that geometry's scratch set (geomsets.go). outScale is its output
	// scalar, one float, nil where it has none.
	geo      geoKey
	vFromK   bool
	outScale backend.Buf
	// router is a mixture block's router, in its device layout and the type
	// the file stores it in; nil on a block that does not route. gate/up/down
	// above hold the banks in that case, and their mv carries slots > 0.
	router   *resident
	kvBytes  uint64 // this block's KV cache, so ReleaseLayers can refund it
	recBytes uint64 // and its recurrent state, refunded on the same walks
	recHist  uint64 // one session's recurrent pair; see devTier.reserved
	// auxBytes is everything else charged to the budget for this block (the
	// norms), so ReleaseLayers refunds it too.
	auxBytes uint64
	// pg is this block's pageable weights -- the seven matrices and nothing else.
	// It is built for every block, on every model, because the budget rather than
	// the fit is what decides whether anything ever moves; see page.go.
	pg *page
	// stream is non-nil when this block's routed bank is not resident: gate, up
	// and down hold NExpertUsed sheets each, refilled every token out of the
	// router's own selection. nil is the ordinary block, where the whole bank
	// sits on the card and nothing crosses the bus inside a token.
	stream *streamBank
	ok     bool
}

// streamBank is the host side of a block whose routed experts are assembled per
// token, and the suspension that makes that possible.
//
// A mixture routing few of many experts can make a resident bank tens of times
// the bytes a token reads, so on a small card the choice is streaming or the
// host. The price is a host round trip mid-block: the selection must be read back
// before anyone knows which sheets to send, so the submission suspends after
// ExpertRank and resumes after the upload, and such a block cannot be captured
// into a graph (a recording has nowhere to put a readback).
type streamBank struct {
	// sh is the per-expert sheet size of each plane, in bytes, for gate, up and
	// down in that order: [matrix][qs, d, sc].
	sh [3][3]int
	// sel is the selection, read back each token. It is len NExpertUsed and is
	// reused rather than allocated per token.
	sel []uint32
	// w is the host side of this block. ensure re-points its spans at wherever
	// the card's frames (fr) hold the block now, a different frame every
	// token on a model that pages (see nn.LayerWeights.Ensure).
	w      nn.LayerWeights
	fr     *jlm.Frames
	ensure func(*nn.LayerWeights, *jlm.Frames) error
	// sel2 is ensure narrowed to the routed sheets. When it is non-nil the
	// whole-bank read never happens; see nn.LayerWeights.EnsureExperts.
	sel2 func(*nn.LayerWeights, []uint32, *jlm.Frames) error
	// pre is sel2's read half, safe to call from another goroutine. When it is
	// non-nil fill pipelines: group g uploads while group g+1 reads. See
	// nn.LayerWeights.PrefetchExperts.
	pre func([]uint32, *jlm.Frames) error
	// rd and grp are fill's reads in flight and its groups, kept so a token
	// allocates neither; items and want are its work list and the experts
	// it reads.
	rd    []streamRead
	grp   [][2]int
	items []streamItem
	want  []uint32
	// The expert cache (initCache), nil without one: slotOf maps an expert
	// to the slot holding it or -1, owner a slot to its expert or -1, used a
	// slot to the tick it was last selected.
	slotOf []int32
	owner  []int32
	used   []uint64
	tick   uint64
	bank   *resident
	// plain is the bank a cache sized after placement replaced (cacheBlock),
	// kept so dropCaches can give the cache back.
	plain [3]*resident
	// nExpert is the bank's expert count, so a cache can be set up after
	// placement (GPU.sizeAutoCaches).
	nExpert int
	// spec hands the host's read service a guess and nil abandons the
	// unstarted ones (nn.LayerWeights.SpeculateExperts).
	spec func(sel []uint32, fr *jlm.Frames)
	li   int // the block, for the session's host side (hostFor)
	// heat counts each expert's selections, halved every heatHalf tokens;
	// hit marks which of this token's selection the cache holds.
	heat []uint32
	own  []uint32
	// owned counts the tokens st.own has counted, for its halving.
	owned uint64
	hit   []bool
	// tab is a composed step's page table (servedPart.pagedRun): each
	// expert's sheet of the bank, reused every token; rsel the selections
	// of a call of several rows.
	tab, rsel []uint32
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
	// The instance holds the frames for the whole Forward (jlm.Frames.Enter),
	// which closes this window and the other three like it; jlm.Lease is the
	// finer primitive.
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
		st.spec(nil, st.fr)
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
		err = st.pre(st.want[grp[0][0]:grp[0][1]], st.fr)
	}
	switch {
	case err != nil:
	case st.sel2 != nil:
		// The narrow read is preferred: the whole bank is correct but tens of
		// times the bytes the token consumes.
		err = st.sel2(&st.w, st.want[grp[0][0]:grp[0][1]], st.fr)
	case st.ensure != nil:
		err = st.ensure(&st.w, st.fr)
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

// hostSide is a session's own side of a streamed block: its counts of the
// block's experts (nn.LayerWeights.ExpertUses).
type hostSide struct {
	uses []uint32
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
}

// uncache returns a block to its plain bank: no cache, nothing held.
func (st *streamBank) uncache() {
	st.slotOf, st.owner, st.used = nil, nil, nil
	st.heat, st.own, st.hit, st.bank = nil, nil, nil, nil
}

// cached reports whether this block's bank is an expert cache.
func (st *streamBank) cached() bool { return st.owner != nil }

// lookup maps this token's selection onto cache slots: a hit keeps its slot,
// a miss takes the least recently used slot this token does not need, and
// only the misses become items.
func (st *streamBank) lookup(g *devTier) {
	st.tick++
	for _, e := range st.sel {
		if s := st.slotOf[e]; s >= 0 {
			st.used[s] = st.tick
		}
	}
	for _, e := range st.sel {
		if st.slotOf[e] >= 0 {
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
		st.items = append(st.items, streamItem{e: e, slot: v})
		g.StreamCacheMisses++
	}
}

// heatHalf is how many tokens a block's own selection counts last at full
// weight, for a session that hands the tier none (nn.LayerWeights.ExpertUses
// is the model's, which halves its own).
const heatHalf = 256

// count adds this token's selection, which the card routed, to uses -- the
// session's own counts of the block's experts (nn.LayerWeights.ExpertUses,
// engine.Hotness), one source for the model's measure of its experts and for
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

// rawStride is an expert's bytes in raw bank q of rows k elements wide, its
// sheet n bytes: a float bank's sheet is its rows, a verbatim bank's rounds
// its blocks up to a word.
func rawStride(q layout.Quant, n, k int) int {
	row := layout.RawRowBytes(q, k)
	return n / row * row
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

// compactNeed is what a streamed block's banks spend on the card when it is
// placed: its own bank of cslots sheets (an expert cache), or the device's
// shared bank of slots sheets where none of that shape is held yet.
func (g *devTier) compactNeed(ws []nn.Weight, slots, cslots int) uint64 {
	var n uint64
	for i := 4; i <= 6; i++ {
		x := ws[i]
		q, ok := quantOf(x.T)
		if !ok || x.Rows == 0 {
			continue
		}
		rows, sheets := x.Rows/slots, cslots
		if cslots <= slots {
			sheets = slots
			if r, held := g.res[sharedKey(i, q, rows, x.K, slots)]; held && r.ok {
				continue
			}
		}
		pq, pd, psc, err := layout.PackedWords(q, rows, x.K)
		if err != nil {
			continue
		}
		n += g.planeFootprint(pq*sheets) + g.planeFootprint(pd*sheets) + g.planeFootprint(psc*sheets)
	}
	return n
}

// cacheSlotsFor is a streamed block's bank size in sheets: the selection, or
// want (Config.StreamCacheSlots) when that is bigger and the device has room for
// the block's own bank of that many beside what it already holds. The
// composed form reads the banks in pages through the page table and every
// per-expert vector (the experts' biases, a scale of each expert's output) by
// the selection the card routed, so a cache serves a biased bank as any other.
func (g *devTier) cacheSlotsFor(p *nn.LayerPlan, ws []nn.Weight, want, slots, bank int) int {
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
		pq, pd, psc, err := layout.PackedWords(q, x.Rows/slots, x.K)
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
	st.rd[gi].err = st.pre(st.want[gr[0]:gr[1]], st.fr)
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
		pk := pair.w.Packed
		raw := pk == nil && layout.IsRaw(pair.r.t) && len(pair.w.Data) > 0
		if pair.r == nil || !pair.r.ok || pk == nil && !raw {
			return fmt.Errorf("tier: streamed bank %d has no host side", m)
		}
		for pl := 0; pl < 3; pl++ {
			n := st.sh[m][pl]
			if n == 0 {
				continue // this format has no such plane
			}
			// A verbatim bank's expert is its blocks, which the sheet rounds
			// up to a word: such an expert is copied, never sent from its
			// frame, whose n bytes would run into the next expert's.
			stride := n
			if raw {
				stride = rawStride(pair.r.t, n, pair.r.k)
			}
			dst := [3]backend.Buf{pair.r.qs, pair.r.d, pair.r.sc}[pl]
			direct := (n >= g.directBytes() || !contiguous) && stride == n
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
				// A bank in expert pages hands over one sheet at a time; a
				// contiguous one is sliced at e*n.
				var src []byte
				release := func() {}
				ts := time.Now()
				if !raw && pk.Sheet != nil {
					sp, rel, err := pk.Sheet(int(e))
					release = rel
					if err != nil {
						release()
						return err
					}
					src = [3][]byte{sp.QS, sp.D, sp.SC}[pl]
				} else {
					// A raw bank's rows are its one plane, an expert's
					// stride bytes of it.
					whole := pair.w.Data
					if !raw {
						whole = [3][]byte{pk.QS, pk.D, pk.SC}[pl]
					}
					off := int(e) * stride
					if off+stride > len(whole) {
						return fmt.Errorf("tier: expert %d plane %d wants bytes %d..%d of %d",
							e, pl, off, off+stride, len(whole))
					}
					src = whole[off : off+stride]
				}
				if len(src) != stride {
					release()
					return fmt.Errorf("tier: expert %d plane %d is %d bytes, want %d", e, pl, len(src), stride)
				}
				tc := time.Now()
				g.TStreamSheet += tc.Sub(ts)
				if direct {
					// The frame stays held until sendPieces has copied it out.
					g.pieces = append(g.pieces, sheetPiece{dst: dst, off: i * n, src: src, release: release,
						locked: g.pinnedFr.holds(src)})
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
	// locked is that src lies in a page-locked frame (pinnedSet).
	locked bool
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

// sendPieces puts direct sheets on the card. A sheet in a page-locked frame
// -- a card's frames are, where its backend pins host memory -- is sent from
// it. The rest, in pageable memory, go through a two-stage pipeline where
// the device has page-locked memory: the pieces are copied into one half
// (concurrently, one goroutine a piece, since a single memcpy from a remote
// node runs well under the link) while the other half is transferred, and a
// transfer out of page-locked memory runs at the link's rate where a
// pageable one went through the driver's bounce buffer at 5.76 GiB/s
// (placement.md 16c). Without it each is a pageable transfer from its frame.
func (g *devTier) sendPieces(s backend.Session, ps []sheetPiece) error {
	g.staged = g.staged[:0]
	for i := range ps {
		if !ps[i].locked {
			g.staged = append(g.staged, ps[i])
			continue
		}
		th := time.Now()
		if err := s.WriteAt(ps[i].dst, ps[i].off, ps[i].src); err != nil {
			return err
		}
		g.TStreamH2D += time.Since(th)
		g.StreamBytes += int64(len(ps[i].src))
		g.StreamDirect++
		g.StreamFromFrame++
	}
	ps = g.staged
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

type mv struct {
	// pad is set on a matvec over a ragged tower's padded width
	// (nn.LayerPlan.KPad): its activation is the padded int8 rows, so a
	// batched twin that reads the float rows instead (an f16 conversion
	// at k) would read them at the wrong stride, and is not taken.
	pad  bool
	rows int
	// slots is 0 for an ordinary matvec and NExpertUsed for an indexed one, in
	// which case the launch is rows*slots*split threads and the kernel takes a
	// pSel parameter after pOut.
	slots int
	alt   *mv
	// q/k are the shape this was built from, so the batched twin can be
	// generated later without re-reading the weight.
	q layout.Quant
	k int
	// experts and slotAct are an indexed matvec's bank size and whether each
	// slot reads its own activation -- what retuneIndexed rebuilds it from.
	experts int
	slotAct bool
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

// kvPair is one sequence's attention history for one block.
type kvPair struct {
	kc, vc backend.Buf
	// bytes is what this pair charged, so growing it adjusts the ledger by the
	// difference rather than guessing.
	bytes uint64
	// paged marks history kept in the device's pool (pagedattn.go): kc and vc
	// are nil, the pages are the sequence's, and the pool carries the charge.
	paged bool
	// shared marks the one k/v pair every vision block of the device uses
	// (transientkv.go): it is released by reference count, never by one block.
	shared bool
}

// recPair is one session's recurrent state on one linear block: the gated
// delta rule's per-head matrices and the causal convolution's window, held at
// the session's seat in the block's pool (recpool.go), each in two halves.
// ir.Validate refuses a param that is both loaded and stored (clamped surplus
// threads would apply the update twice), so the state arrives through one
// half, leaves through the other, and the halves swap. Both are constant in
// the context length, so doubling them is cheap. In the shared form (one
// state per slot and the device's recNext, taken when two halves do not fit)
// only the window has halves.
type recPair struct {
	// cur is the half a submission reads; it writes the other and flips.
	cur int
	// steps counts the flips, so a rewind can tell one step from several
	// (RecRewind); mark is the half and count RecMark saw.
	steps     uint64
	mark      int
	markSteps uint64
	marked    bool
}

// recOf returns this session's recurrent state for the block, or nil.
func (l *layer) recOf(sid uint64) *recPair {
	if l == nil || l.rec == nil {
		return nil
	}
	return l.rec[sid]
}

// permBytes is what admitting a fresh block charges beyond its weights and
// never gives back while it is resident: this session's history (a KV cache,
// or a linear block's recurrent pair) and the norms. prepLayer prices it into
// the room check, so a block is declined -- and the next device asked -- rather
// than admitted into room its own cache does not have. Callers hold g.mu, after
// initKVCap.
func (g *devTier) permBytes(p *nn.LayerPlan, w *nn.LayerWeights) uint64 {
	var n uint64
	for _, v := range w.Vectors {
		n += uint64(4 * len(v.V))
	}
	// The router stays while the block is placed, paged or not.
	if r := w.Router; len(r.Data) != 0 {
		if q, ok := quantOf(r.T); ok {
			if f, err := g.planesFootprint(q, r.Rows, r.K); err == nil {
				n += f
			}
		}
	}
	if r := p.Recurrent; r.Conv > 0 {
		return n + g.sessions()*recHist(p)
	}
	if g.pagedPlan(p) {
		return n + g.pagedLayerBytes(p)
	}
	cp := g.capped(p)
	cache, vcache := g.kvCacheBytes(p, cp.MaxSeq)
	if p.NonCausal {
		// One pair for every non-causal block of a set, charged once
		// (transientkv.go).
		if v := g.visOf(geoOf(p)); v.tkv != nil && v.tkv.bytes >= uint64(cache+vcache) {
			return n
		}
		return n + uint64(cache+vcache)
	}
	return n + g.sessions()*uint64(cache+vcache)
}

// recHist is one session's recurrent pair for a linear block.
func recHist(p *nn.LayerPlan) uint64 {
	r := p.Recurrent
	return 2 * uint64(r.StateLen*4+r.ConvState*4)
}

func (g *devTier) sessions() uint64 { return uint64(max(1, g.Sessions)) }

// reserved is what the device holds back for sessions that have not attached
// yet: for every placed linear block, one recurrent pair per seat of
// Config.Sessions that no live session occupies. room() counts it, so neither
// a block nor a page can spend it; a session taking a seat (addSessionRec)
// spends its own share. An attention history holds no seat: every causal
// block's history is paged (initKVCap sets g.paged with the first text plan),
// and a page is taken as a sequence writes its rows, not when it attaches.
// Callers hold g.mu.
func (g *devTier) reserved() uint64 {
	if g.Sessions <= 1 {
		return 0
	}
	var n uint64
	for _, l := range g.layers {
		if l == nil || !l.ok || l.nonCausal || !l.linear {
			continue
		}
		if free := g.Sessions - len(l.rec); free > 0 {
			n += uint64(free) * l.recHist
		}
	}
	return n
}

// addSessionRec gives session sid the two-half recurrent state for a
// linear block: a seat in the device's pools (one slot, unless its batch has
// grown) and a pool for the block if it has none. Callers hold g.mu. The state
// is zeroed because a summary of garbage is a history the model never saw,
// and every token after it is fluent and wrong (RULE 13).
func (g *devTier) addSessionRec(sid uint64, l *layer, p *nn.LayerPlan) bool {
	r := p.Recurrent
	if r.Conv == 0 {
		return true // not a linear block; nothing to carry
	}
	g.recS, g.recC = r.StateLen, r.ConvState
	need := 2 * uint64(r.StateLen*4+r.ConvState*4)
	l.recHist = need
	if g.recSeats == nil {
		g.recSeats = map[uint64]recSeat{}
	}
	st, seated := g.recSeats[sid]
	if !seated {
		st = recSeat{base: g.seatPlace(sid, 1), rows: 1}
	}
	slots := max(g.recSlots, st.base+st.rows)
	// The device's first pool takes the form Config.SharedRec asks for; a
	// later one takes the pools' form.
	shared := g.recShared
	if g.recSlots == 0 {
		shared = g.SharedRec
	}
	slot := g.recSlotBytes(shared)
	cost := uint64(slots-g.recSlots) * slot * uint64(len(g.recPools()))
	if l.pool == nil {
		cost += uint64(slots) * slot
	}
	// A card its pages fill makes room for the state as for a history
	// growing (stateFits): a page goes home before a session is refused.
	if cost > 0 && !g.roomSeat(cost, len(l.rec), need) && !g.stateFits(cost, nil) {
		g.LastErr = "a session's recurrent state does not fit in this device's budget"
		g.SessionDeclines++
		return false
	}
	fail := func(err error) bool { g.LastErr = err.Error(); return false }
	if slots != g.recSlots || shared != g.recShared {
		if err := g.recResize(slots, shared); err != nil {
			return fail(err)
		}
	}
	if !seated {
		g.recSeats[sid] = st
		if err := g.recZero(nil, st.base, st.rows); err != nil {
			return fail(err)
		}
	}
	if l.pool == nil {
		np, err := g.newRecPool(g.recSlots, g.recShared, nil)
		if err != nil {
			return fail(err)
		}
		l.pool = np
		n := uint64(g.recSlots) * slot
		g.charge(n)
		l.recBytes += n
		g.RecBytes += n
		if g.recShared {
			g.RecShared++
		}
	} else if err := g.recZero(l, st.base, st.rows); err != nil {
		// The seat's slots in a block this session left and came back to
		// hold its old summary.
		return fail(err)
	}
	if g.recShared && !g.ensureRecNext(st.rows) {
		return false
	}
	if l.rec == nil {
		l.rec = map[uint64]*recPair{}
	}
	l.rec[sid] = &recPair{}
	return true
}

// ensureRecNext makes the shared form's next-state scratch hold rows
// sequences' states and compiles the copy that reads them back, charging the
// ledger for any growth. It runs outside a session (growSeat, LayersRows
// before submit), where Compile is allowed. Callers hold g.mu.
func (g *devTier) ensureRecNext(rows int) bool {
	fail := func(err error) bool { g.LastErr = "rows: " + err.Error(); return false }
	if g.recS == 0 {
		return true // a short convolution keeps no state beside its window
	}
	if bytes := uint64(rows * g.recS * 4); g.recNextBytes < bytes {
		b, err := g.dev.Alloc(int(bytes))
		if err != nil {
			return fail(err)
		}
		// Every recording, and every submission in flight, names the old one.
		g.dropGraph()
		if g.recNext != nil {
			g.recNext.Free()
			g.refund(g.recNextBytes)
		}
		g.recNext, g.recNextBytes = b, bytes
		g.charge(bytes)
	}
	return true
}

// freeRec releases a block's pool and every session's place in it, which is
// the same contract freeKV has and for the same reason: a release gives the
// block up entirely, so no sequence keeps a summary for it. The callers refund
// l.recBytes; a seat no block uses any more goes at recTidy.
func freeRec(l *layer) {
	if l.pool != nil {
		l.pool.free()
		l.pool = nil
	}
	clear(l.rec)
}

// addSessionKV gives session sid its own attention history for a block
// whose model-resident parts are already on the device. Callers hold g.mu.
func (g *devTier) addSessionKV(sid uint64, l *layer, p *nn.LayerPlan) bool {
	if p.NonCausal {
		if kvp := g.transientKV(p); kvp != nil {
			if l.kv == nil {
				l.kv = map[uint64]*kvPair{}
			}
			l.kv[sid] = kvp
			return true
		}
	}
	g.initKVCap(p)
	if g.pagedPlan(p) {
		// The pages are the sequence's, taken as its rows are written; the
		// block's layer is already in the pool.
		if l.kv == nil {
			l.kv = map[uint64]*kvPair{}
		}
		l.kv[sid] = &kvPair{paged: true}
		return true
	}
	cp := g.capped(p)
	cache, vcache := g.kvCacheBytes(p, cp.MaxSeq)
	need := uint64(cache + vcache)
	if !g.roomSeat(need, len(l.kv), need) && !g.stateFits(need, nil) {
		g.LastErr = "a second session's KV cache does not fit in this device's budget"
		g.SessionDeclines++
		return false
	}
	g.charge(need)
	kvp := &kvPair{}
	var err error
	// One allocation under MLA, where vcache is zero; see kvPair.value.
	if kvp.kc, err = g.dev.Alloc(cache); err == nil && vcache > 0 {
		kvp.vc, err = g.dev.Alloc(vcache)
	}
	if err != nil {
		if kvp.kc != nil {
			kvp.kc.Free()
		}
		g.refund(need)
		g.LastErr = err.Error()
		return false
	}
	if l.kv == nil {
		l.kv = map[uint64]*kvPair{}
	}
	kvp.bytes = need
	l.kv[sid] = kvp
	// l.kvBytes is what ReleaseLayers refunds, so it moves with every charge;
	// without it a block released after a second session refunds only the first
	// session's share and the budget leaks.
	l.kvBytes += need
	g.KVBytes += need
	return true
}

// freeKV releases every session's history for a block. A release gives the block
// up entirely, so no sequence keeps a cache for it.
func (g *devTier) freeKV(l *layer) {
	for sid, p := range l.kv {
		if p.shared {
			g.transientKVPut(l.geo)
			delete(l.kv, sid)
			continue
		}
		if p.kc != nil {
			p.kc.Free()
		}
		if p.vc != nil {
			p.vc.Free()
		}
		delete(l.kv, sid)
	}
}

// kvOf returns this session's cache for the block, or nil if it has none yet.
func (l *layer) kvOf(sid uint64) *kvPair {
	if l == nil || l.kv == nil {
		return nil
	}
	return l.kv[sid]
}

// blockScratch is the per-call staging every layer of a lane shares: a call
// holds its lane (devsess.go), so one block runs in it at a time.
type blockScratch struct {
	// headGraph is the head as ops from the model's front end (nn.Head.Graph):
	// what serving the head forms (served.go); headHidden is it writing its
	// normed rows too (nn.Head.HiddenGraph), a call reading them back's.
	headGraph, headHidden *graph.Graph
	// sampleGraph is the greedy token over the head's logits as ops from the
	// front end (nn.Head.Sample): what serving the argmax forms; sampleWalk
	// a top-k draw's token (nn.Head.SampleWalk), what serving SampleK forms;
	// pickGraph and pickHead the label pick over the logits and the head
	// projecting the picked rows alone (nn.Head.PickGraph, PickHead).
	sampleGraph, sampleWalk, pickGraph, pickHead *graph.Graph
	// headShared says head, mvHead, hNorm, hNormB, hBias and headCap are another
	// scratch's -- lane0's, borrowed by a clone (shareHead) -- and are not
	// freed or refunded with this one.
	headShared bool

	// walked is g.layerGen+1 at prepBatch's last successful layer walk for
	// this scratch: until a block is placed or released or the decode retune
	// rewrites their matvecs, the walk would find what it found then (0: never
	// walked).
	walked uint64
	// prompt is the lane's instance of each form the composer serves a
	// prompt chunk's blocks with at this scratch's rows, shared by the blocks
	// of the form (served.go): freed and refunded with the scratch.
	prompt map[*partForm]*partInst
	// arena is the device buffers those instances' passing buffers are laid
	// on, one set for every form, since the blocks run one after another
	// (partArena).
	arena *partArena
	p     nn.LayerPlan
	// rows is how many token positions this scratch runs at once: 1 for decode
	// and the prefill chunk width for the batched arm. Every buffer is sized
	// rows-wide.
	rows                int
	tok                 int
	qtile, ktile, atile int
	// bytes is what building this scratch allocated, by the device's count
	// (0 where it keeps none).
	bytes int64
	// pkv is the paged attention set; nil unless this scratch's history is
	// paged (devTier.pagedPlan), and then the staged, tiled and flash
	// kernels above that read a contiguous cache are not built.
	pkv *pagedScratch

	// The ragged-decode set (rows.go): attention whose rows are different
	// sequences, a per-row softmax, the head's batched twin and a per-row argmax.
	// ragBuilt says prepRowsHead built the set; the composer's sample
	// writes the tokens.
	ragBuilt          bool
	ragLogits, ragTok backend.Buf
	ragVocab          int // the rows of ragLogits a row holds
	// ragRows is the rows ragLogits and ragTok hold (headRows); headIn and
	// headReq a chunk's rows taken out of x and the take's request, built
	// where ragRows is short of the scratch's rows (headChunk).
	ragRows         int
	headIn, headReq backend.Buf
	sstride         int // the score row stride; padded at rows > 1
	// arows is how many query rows one pass of the attention kernels takes:
	// rows, except for a vision block whose score planes would be too large,
	// which runs its attention in query chunks (visionattn.go). qc and xc are a
	// chunk's queries and output, gathered from and scattered to q and xb at
	// qoffs[c], by qGather and xScatter.
	arows         int
	qoffs         []backend.Buf
	x, h          backend.Buf
	roff          backend.Buf // RoPE-q destination offsets, one per row
	hNorm, logits backend.Buf
	// embedGraph is the input rows as ops (nn.Head.Embed); embW and posW the
	// tables they read, the token embedding (the head where it is the
	// embedding) and a learned position table, resident for the model, and
	// embErr why PrepHead could not hold them (errEmbedRoom for room). embTok and embPos are the
	// rows' tokens and positions the composer's gather reads, embCast the
	// rows a promise no call takes are gathered into (materializeLocked),
	// every stream of them, embPLE their per-layer inputs, each as wide as
	// the widest call; castViews is embCast's streams at a call's rows.
	embedGraph     *graph.Graph
	embW, posW     *resident
	embErr         error
	embTok, embPos backend.Buf
	embCast        backend.Buf
	embPLE         backend.Buf
	castViews      map[int][]backend.Buf
	// embOuts is the buffers a step's input rows write (embedOuts), room
	// for every stream and the per-layer inputs.
	embOuts []backend.Buf
	// embSW and embSV are the tensors the input rows bind past the token
	// embedding and the position table (nn.Head.Streams), by role.
	embSW map[jlm.Role]*resident
	embSV map[jlm.Role]backend.Buf
	// collapse is the streams' collapse a call of the head alone runs
	// first (collapse.go), nil where the head reads one row.
	collapse *nn.RowsPart
	// foldReq is the request of the composer's row take that moves a batched
	// chunk's last residual row into this (decode) scratch's x, so the output
	// projection rides the chunk's own submission (layersOnce's folded head):
	// composer.RequestRows words and the row (servedRowTake).
	foldReq backend.Buf
	hNormB  backend.Buf // a LayerNorm head's bias (C6)
	// hBias is the output projection's bias (nn.Head.Bias), held where the
	// composer serves a head graph that binds it.
	hBias       backend.Buf
	head        *resident
	mvHead      mv
	hRaw        []byte
	hOut        []float32
	cs, n, zero backend.Buf
	// calls is a call's rows, one word: what a composed form of several
	// rows reads as the rows it runs (composer.CallRows); headCalls the rows a
	// call's head over rows projects.
	calls, headCalls backend.Buf
	// pics is a non-causal scratch's rows' descriptors (composer.RowWords
	// each) where the composer serves its blocks (towerStep), and picsW
	// the same rows each attending its window of the call; picStage and
	// picStageW are their bytes as a call stages them.
	pics, picsW         backend.Buf
	picStage, picStageW []byte
	// instNoRoom says a composed instance at the scratch's rows did not fit
	// the budget (promptInst): the reserved width narrows (fitReserved).
	instNoRoom bool
	// csSWA is the local layers' rotary table, allocated only for an
	// architecture that trains two bases (LayerPlan.SWAPeriod). nil elsewhere,
	// which is what keeps ropeTable a nil check for every other model.
	csSWA backend.Buf
	// ropeTab and ropeTabSWA are the per-model folded planes (nn.Rope.TabPlanes)
	// the composer's rotary tables read, uploaded once (initRopePlanes).
	ropeTab, ropeTabSWA backend.Buf
	// ropeTabAxes and rposAxes are what the composer's table over several
	// position axes reads (nn.LayerPlan.RopeTabAxes): its planes, and the
	// live row count then each row's four positions.
	ropeTabAxes, rposAxes backend.Buf
	// ropeRows is the call's rows as the composed rotary tables read them
	// (ropeRowsWords): two words a row past the count. rowPos is their
	// positions as a block reads them (rowPosWords).
	ropeRows, rowPos backend.Buf
	// Gemma 4's per-layer embedding (nn.LayerPlan.PLEDim): the rows' inputs,
	// staged per submission, a block's gate, its slice of the inputs, their
	// product and the projection back. nil elsewhere.
	pleIn backend.Buf
	// cut is a call's rows of a cut block's crossing values (layer.cut),
	// cutBytes its size, grown to the widest call.
	cut      backend.Buf
	cutBytes int
	// streams is, by a step's rows, each residual stream of x as a view at
	// its offset (x is stream-major), the inputs and the outputs a block of
	// several streams binds (servedStreams).
	streams map[int][2][]backend.Buf
	// streamsApart is streams with each stream's output a view of h at
	// x's offset, for a form that writes its streams apart (servedCopy).
	streamsApart map[int][2][]backend.Buf
	dRows        backend.Buf
	// cRows is the composer's recurrent descriptor in the shared form.
	cRows backend.Buf
	// gridGraph is a non-causal set's grid table (LayerPlan.GridPlanes), and
	// gridPos, gridReq and gridPlanes a call's cells, its request and the
	// table's planes (gridtable.go).
	gridGraph                    *graph.Graph
	gridPos, gridReq, gridPlanes backend.Buf
	// req is a call's request words (nn.RequestDevice), read where a block
	// reads composer.TableRequest.
	req             backend.Buf
	rec             nn.RecurrentPlan
	hx, hcs, hcsSWA []float32
	hraw            []byte
	hout            []float32
}

// PrepLayer is prepLayerFor for the zero session: a caller that never attached.
func (g *devTier) PrepLayer(li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	return g.prepLayerFor(0, li, p, w)
}

// prepLayerFor uploads one block and compiles the kernels it needs, taking it only
// if it fits.
func (g *devTier) prepLayerFor(sid uint64, li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	return g.prepLayer(sid, li, p, w, false)
}

// residScaled reports a plan whose block outputs are scaled before their
// residual adds (Granite's residual_scale). Such a block takes neither the
// projection's fused residual (mvres: the bias slot has no scale) nor the
// add-and-norm launches: the output lands in bs.mvOut as for a post-norm, and
// bs.addS adds s times it. A mixture's routed weights carry no part of it on
// the device, unlike the host's routed scale (model.Config.routedScale): the
// scaled add covers the routed sum and the shared expert at once.

// ungatedFFN reports whether the block's FFN has no gate matrix: up, the
// activation alone, down -- a vision tower's MLP, or a classic text block's
// (C6: phi-2, starcoder, falcon, nemotron).
func ungatedFFN(p *nn.LayerPlan) bool { return p.UngatedFFN }

// prepLayer is PrepLayer with the pager allowed. mayPage lets a block that does
// not fit evict another one rather than be declined. It is a second pass, not a
// mode: GPU.PrepLayer offers every device with mayPage false first, so a second
// card with room still gets the block and only when no device has room does
// anybody swap.
func (g *devTier) prepLayer(sid uint64, li int, p *nn.LayerPlan, w *nn.LayerWeights, mayPage bool) (placed bool) {
	fr, pin, ferr := g.offerFrames(li, p, w)
	if ferr != nil {
		pin.Release()
		g.LastErr = fmt.Sprintf("block %d: %v", li, ferr)
		return false
	}
	// Last to run, after the body's unlocks: the pin goes, then the card's
	// frame for the block once nothing on the card reads it there -- a
	// declined block's claim, and a placed block whose weights are all on
	// the card.
	defer func() {
		pin.Release()
		if fr != nil && (!placed || g.ReleasesHostPage(li)) {
			fr.DropPage(li)
		}
		if placed {
			g.sizeFrames(p.File)
		}
	}()
	// Every placement of a block forms its composed part, a move's and a
	// re-offer's too: a step would otherwise meet it first. Deferred first,
	// it runs after the body's own deferred unlocks.
	defer func() {
		if placed {
			g.mu.Lock()
			g.prewarmComposedTables()
			// A block whose composed part, or prompt part at the narrowest
			// width fitReserved reached, does not fit beside the blocks placed
			// is declined here rather than met by its first step or prompt; a
			// mixture's is offered again streamed.
			why := g.prewarmComposedBlock(li, true)
			if why == "" {
				why = g.unserved(li)
			}
			if why == "" {
				why = g.promptRoomless(li)
			}
			if why != "" && !g.sharedBeyond(li, sid) {
				g.dropGraph()
				g.releaseBlock(li)
				g.afterRelease()
				g.Declined++
				g.LastErr = why
				placed = false
			}
			g.mu.Unlock()
		}
	}()
	// A streamed block's expert counts are the offering session's: a placed
	// block is shared by every session that attaches, and each counts its own.
	// Recorded before anything can decline, so a block another session placed
	// still learns this one's.
	if w != nil && w.ExpertUses != nil {
		g.mu.Lock()
		if g.hostFns == nil {
			g.hostFns = map[uint64]map[int]hostSide{}
		}
		if g.hostFns[sid] == nil {
			g.hostFns[sid] = map[int]hostSide{}
		}
		g.hostFns[sid][li] = hostSide{uses: w.ExpertUses}
		g.mu.Unlock()
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	// A block the device could not take leaves nothing behind: the first
	// one built its scratch before the budget said no, and with no block on
	// the device that scratch is a ceiling nobody runs -- a tower's set's
	// too, with no tower block left.
	defer func() {
		if placed {
			return
		}
		if p.NonCausal {
			g.dropIdleTower()
		}
		g.dropIfEmpty()
	}()
	// The block's own geometry's scratch set, and the home set again on the
	// way out (geomsets.go): Gemma 4's sliding layers, and a non-causal
	// segment's blocks -- a vision tower's, whose residual, heads and FFN are
	// not the text model's. sp is the plan that set is built from.
	sp := scratchPlanOf(p)
	if k := geoOf(p); k != (geoKey{}) {
		g.geoUsed = true
		g.useGeom(k)
		defer g.useGeom(geoKey{})
	}

	// The KV ceiling is the largest context any session asked for, not the
	// first's, or a longer second session is refused at its next position.
	if !p.NonCausal && g.kvCap != 0 && p.MaxSeq > g.maxSeqAsked {
		g.maxSeqAsked = p.MaxSeq
	}
	if r := planConflict(g.bs, sp); r != "" {
		g.LastErr = fmt.Sprintf("block %d: %s", li, r)
		return false
	}
	if r := nonCausalConflict(g.bs, sp); r != "" {
		g.LastErr = fmt.Sprintf("block %d: %s", li, r)
		return false
	}

	// A block already on this device is shared and only the history is new: the
	// weights, norms, biases and router are the model's. Rebuilding would
	// re-upload them and replace the layer the first session is running against.
	if l := g.layers[li]; l != nil && l.ok {
		// Sharing is of one block: a model's block is the graph it was
		// placed with, and a session offering another at this index offers
		// another model's.
		if p.Graph != l.graph {
			g.LastErr = fmt.Sprintf("block %d: offered with a graph other than the one this device holds for it", li)
			return false
		}
		defer func() {
			if placed {
				l.join(sid)
			}
		}()
		// The new session's plan still has to reach the scratch: a longer session
		// may need a sliding window the first did not (model.windowFor), so take
		// the union as the fresh-block path does.
		if !l.nonCausal && g.bs != nil {
			if want := g.bs.p; planUnion(&want, sp) {
				g.dropGraph()
				if !g.rebuildScratch(&want) {
					return false
				}
			}
		}
		// A linear block's history is a recurrent pair, not a KV cache; a
		// block that attends as well keeps both.
		if l.linear {
			if l.recOf(sid) == nil && !g.addSessionRec(sid, l, p) {
				return false
			}
			if !l.withAttn {
				return true
			}
		}
		if l.kvOf(sid) != nil || l.kvSrc != li {
			return true // this session already has a history here, or reads its source's
		}
		if !g.addSessionKV(sid, l, p) {
			return false
		}
		return true
	}
	// The q/k norm is launched when the block carries its weights; the plan's
	// flag builds the kernels. The model offers the weights by the same
	// derivation that sets the flag (jlm.QKNormOn), so a block where the two
	// disagree is a contract violation, refused rather than run without its
	// norm. A block another session placed is offered without them, held
	// above.
	if attn := !p.NonCausal && (p.Recurrent.Conv == 0 || p.Recurrent.WithAttn); attn && p.QKNorm != (w.QNorm != nil) {
		g.LastErr = fmt.Sprintf("block %d: the plan's q/k norm is %v and the block offers %d norm "+
			"weights", li, p.QKNorm, len(w.QNorm))
		return false
	}
	g.dropGraph()
	// The KV capacity is decided before the scratch is built: bs.p is the capped
	// plan, and the scratch's score buffers and MigrateKV's staging are sized
	// from its MaxSeq, so building first would size them for the whole context.
	g.initKVCap(p)
	// A non-causal set is built for the rows of the pictures it runs, not for
	// the tower's largest grid (visrows.go): the first block sets where it
	// starts, ReserveRows moves it.
	if p.NonCausal {
		g.vs().max = max(g.vs().max, p.MaxSeq)
		if g.vs().cap == 0 {
			g.vs().cap = visRowsFor(min(p.MaxSeq, visStart), p.MaxSeq)
			g.vs().first = g.vs().cap
		}
	}
	if g.bs == nil && p.NonCausal {
		// A non-causal call is one whole run, so its set's scratch is built
		// for the run's rows -- the rows the set is built for (visCap), which
		// capped hands the plan as its MaxSeq: there is no decode width.
		if !g.buildTowerScratch(sp) {
			return false
		}
	} else if g.bs == nil {
		g.shapeChanged()
		if g.bs = g.initScratch(sp, 1); g.bs == nil {
			return false
		}
	} else if want := g.bs.p; p.NonCausal && (sp.NFFN > want.NFFN || sp.Recurrent.Conv > 0 && want.Recurrent.Conv == 0 ||
		sp.QKNorm && !want.QKNorm) {
		// A non-causal block with a wider MLP than the set was built for (a
		// decision head after its encoder): the set grows to it, and every
		// block runs its own width inside it (layer.nffn). An LFM2 encoder's
		// convolution blocks and attention blocks share a set too: it carries
		// the convolution's kernels and the q/k norm's, whichever came first.
		want.NFFN, want.MaxSeq = max(want.NFFN, sp.NFFN), g.vs().max
		if want.Recurrent.Conv == 0 {
			want.Recurrent = sp.Recurrent
		}
		want.QKNorm = want.QKNorm || sp.QKNorm
		if !g.towerAt(&want, g.vs().cap) {
			return false
		}
	} else if want := g.bs.p; !p.NonCausal && planUnion(&want, sp) {
		// g.bs is built from whichever block arrives first, and on a hybrid that
		// may be a linear block with no out-gate kernels. A hybrid's two kinds
		// alternate over one residual inside one submission, so they share one
		// scratch carrying both kernel sets, built from the union.
		if !g.rebuildScratch(&want) {
			return false
		}
	}
	if g.layers == nil {
		g.layers = map[int]*layer{}
	}
	l := &layer{nonCausal: p.NonCausal, windowed: p.Windowed, geo: geoOf(p), vFromK: p.VFromK, graph: p.Graph, cut: p.Cut}
	l.win, l.chunked = attendWindow(p.Graph)
	if p.NonCausal {
		l.nffn, l.ungated = p.NFFN, ungatedFFN(p)
		l.noRope = p.NRot == 0 || p.NoPosEnc
	}
	mkw := func(slot int, x nn.Weight) *resident {
		q, ok := quantOf(x.T)
		if !ok || x.Rows <= 0 || len(x.Data) == 0 {
			return nil
		}
		return g.resident(li, slot, nil, q, x.Data, x.Rows, x.K, x.Packed)
	}
	// An expert bank is one matrix of NExpert*Rows rows: a GGUF 3-D expert tensor
	// is expert-major and contiguous, so it packs, uploads and indexes as one
	// tall matrix, and the kernel does the row offset from Experts and Slots.
	moe := w.Router.Data != nil
	bank := 1
	slots := 0
	if moe {
		if p.NExpert < 1 || p.NExpertUsed < 1 || p.NExpertUsed > p.NExpert {
			return false
		}
		bank, slots = p.NExpert, p.NExpertUsed
	}
	// One list: layer.tensors() derives the free walk and pageSizeWhy the price
	// from this order, so a matrix uploaded outside it would be neither freed
	// nor charged. A linear block leaves 1..3 empty (Wq carries its fused q|k|v),
	// an attention block leaves 10..12 empty, and pageSizeWhy skips zero-length
	// weights.
	ws := append(weightList(w), l.takeExtra(w)...)
	// A streamed bank is priced at its compact size by scaling the three bank
	// rows from NExpert sheets to NExpertUsed; every size derives from
	// Weight.Rows, so the charge and the refund cannot disagree.
	_, auto := g.autoStream[li]
	stream := moe && (g.StreamExperts || auto) && slots < bank
	// cslots is the streamed bank's size in sheets, decided at the first bank
	// matrix (cacheSlotsFor): slots without an expert cache, more with one.
	cslots := -1
	// price is ws with the three banks removed for a streamed block: the
	// compact bank is one buffer for the whole device (sharedCompact), so a
	// block costs its base and nothing else.
	price := ws
	if stream {
		for i := 4; i <= 6; i++ {
			if ws[i].Rows%bank != 0 {
				g.LastErr = fmt.Sprintf("block %d: %s has %d rows, not a multiple of %d experts",
					li, weightRole(i), ws[i].Rows, bank)
				return false
			}
			ws[i].Rows = ws[i].Rows / bank * slots
		}
		price = append([]nn.Weight(nil), ws...)
		for i := 4; i <= 6; i++ {
			price[i].Data, price[i].Packed = nil, nil
		}
	}
	// Decide on the whole block before uploading any of it: declining halfway
	// leaves earlier tensors resident, counted and unreachable. Every decline
	// sets LastErr.
	full, why, ok := g.pageSizeWhy(price)
	if !ok {
		g.LastErr = fmt.Sprintf("block %d: %s", li, why)
		return false
	}
	// The slot width is the widest block, not this one; see slots().
	if full > g.widest {
		g.widest = full
	}
	need, _ := g.blockBytes(price)
	// Include what admitting it charges for as long as it is resident: its KV
	// cache (or recurrent pair) and norms. Leaving them out admitted blocks that
	// pushed the card over budget and made trim() page every token.
	perm := g.permBytes(p, w)
	need += perm
	// A streamed bank is spent with the block: its own of cslots sheets, or
	// the device's shared one where this block is the first to make it.
	// Priced apart from the block, the room the block was admitted for went
	// to the bank and its own tensors no longer fit.
	if stream {
		cslots = g.cacheSlotsFor(p, ws, g.StreamCacheSlots, slots, bank)
		bankNeed := g.compactNeed(ws, slots, cslots)
		if cslots > slots && !g.room(need+bankNeed) {
			cslots = slots
			bankNeed = g.compactNeed(ws, slots, cslots)
		}
		need += bankNeed
	}
	// Before the block's weights: the prompt's and the step's scratch are
	// budgeted first, with room left for this block, so a card that cannot
	// hold both takes a block fewer -- or a narrower chunk -- rather than
	// refusing the prompt later (scratch.go).
	g.reserveScratch(sp, need)
	// The KV pools' free pages go first: they hold nothing.
	g.kvCompact(need, nil)
	if !g.room(need) {
		// A block that does not fit pages another one out rather than falling to
		// the host. Below minSlots there is nowhere to swap through, so it
		// declines and LastErr says which case it is.
		// The slots are counted after this block's own permanent charge: a
		// streamed block whose norms and history take the last slot leaves
		// nothing to swap through, and the next page-in fails the device.
		if n := g.slotsIfPerm(perm); !mayPage || !g.stream[li] || n < minSlots {
			// Only the final pass counts as a decline: GPU.PrepLayer asks every
			// device with mayPage false first.
			if mayPage {
				g.Declined++
			}
			// Say which of the three it is: another device may still have room,
			// the block is placed resident, or the budget is too small to swap
			// through.
			why := "another device is asked first"
			if mayPage {
				why = "it is placed resident and does not fit, so it runs on the host " +
					"(a placement that streams it -- model.Place.Stream, -placement N=DEV~ -- " +
					"swaps it through the device instead)"
				if g.stream[li] {
					why = fmt.Sprintf("the budget holds %d block slot(s), below the %d a swap needs", n, minSlots)
				}
			}
			g.LastErr = fmt.Sprintf("block %d needs %d bytes, %d of %d used (pool %s: %d of %d); %s",
				li, need, g.used, g.limit, g.pool.Name(), g.pool.Used(), g.pool.Limit(), why)
			// Recorded on the final pass only, like Declined.
			if mayPage && g.DeclineWhy == "" {
				g.DeclineWhy = why
			}
			return false
		}
		// At load the scan ascends from here, so the victim is the block just
		// behind this one and a resident prefix remains: the set the run-time
		// policy settles on, so the first token starts warm. See victim().
		if !g.reclaim(need, li, 0, li+1) {
			g.Declined++
			g.LastErr = fmt.Sprintf("block %d needs %d bytes and nothing can be paged out (%d of %d used)",
				li, need, g.used, g.limit)
			if g.DeclineWhy == "" {
				g.DeclineWhy = "nothing can be paged out"
			}
			return false
		}
	}
	// The staging is sized after the block is admitted, from what is left
	// (need is what the block still has to spend). If that is not enough it is
	// not allocated and the covered tensors are packed on the host. It is
	// allocated once and survives into later uploads and page-ins.
	//
	// The bytes are fetched here, after the decision and before the first read:
	// nn.LayerWeights.Ensure makes the contents valid (a paged mixture sets it),
	// and calling it earlier would read a bank for every block the budget then
	// refuses.
	if w.Ensure != nil {
		if err := w.Ensure(w, fr); err != nil {
			g.LastErr = fmt.Sprintf("block %d: %v", li, err)
			return false
		}
	}
	rs := l.tensors()
	// The latent block's own slots: 0 (q) is the one-step query and is empty
	// where 13/14 are populated; 1 and 2 are always empty (MLA has no k or v
	// projection); 15..17 are always there. Each is an absent weight, not a
	// failed upload.
	l.mla = p.MLA()
	mlaSkip := func(i int) bool {
		if !l.mla {
			// No latent block, no latent matrices -- except Mamba-1's four,
			// which ride KDA's slots 18..21 (nn.SSMWeights), and a latent
			// query the container stores for a block of any attention: slot
			// 0 is then empty and 13/14 are the query.
			latentQ := len(ws) > 14 && len(ws[13].Data) != 0
			switch i {
			case 0:
				return latentQ && len(ws[0].Data) == 0
			case 13, 14:
				return !latentQ
			}
			return i >= 13 && !(i >= 18 && p.Recurrent.Scan)
		}
		// A linear block of an MLA model (Kimi-Linear) has none of them either:
		// l.mla is a property of the model, not the block.
		if p.Recurrent.Conv > 0 && i >= 13 && i < 18 {
			return true
		}
		switch i {
		case 0:
			// A linear block's q is its fused q|k|v whatever the model's
			// query rank (Kimi-K3 compresses its MLA query; Kimi-Linear
			// does not).
			return p.QLoraRank != 0 && p.Recurrent.Conv == 0
		case 1, 2:
			return true
		case 13, 14:
			return p.QLoraRank == 0
		}
		return false
	}
	for i, x := range ws {
		// A matrix past the shared structure's is uploaded as the file
		// stores it, by role.
		if i >= nNamed {
			r := mkw(i, x)
			if r == nil || !r.ok {
				if g.LastErr == "" {
					g.LastErr = fmt.Sprintf("block %d: %s (%s %dx%d) not resident", li, l.extra[i-nNamed].role, x.T, x.Rows, x.K)
				}
				return false
			}
			*rs[i] = r
			continue
		}
		// KDA's four low-rank gate matrices, checked before mlaSkip because they
		// sit past MLA's range and are not MLA's.
		if i >= 18 {
			switch {
			case !p.Recurrent.ChanDecay && !p.Recurrent.Scan, p.Recurrent.Conv == 0:
				// Not a KDA model, or an attention block of one: absent by
				// construction rather than by a missing tensor.
				continue
			case (i == 20 || i == 21) && p.Recurrent.ChanDecay && len(w.SSM.Gate.Data) != 0:
				// A full-rank output gate is slot 10's, and there is no
				// low-rank pair.
				continue
			case len(x.Data) == 0:
				g.LastErr = fmt.Sprintf("block %d: the plan is a channel-decay delta "+
					"block and %s is empty", li, weightRole(i))
				return false
			}
			// A present weight falls through to the upload; a continue here would
			// leave l.ssmGA nil and crash inside the recorded session.
		}
		if mlaSkip(i) {
			continue
		}
		if i >= 13 {
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan is a latent attention block "+
					"(KVLoraRank=%d) and %s is empty", li, p.KVLoraRank, weightRole(i))
				return false
			}
		}
		// An ungated block has no gate matrix: slot 4 is empty by construction.
		if ungatedFFN(p) && i == 4 {
			continue
		}
		// A half of a block cut in two holds what it was offered: its banks
		// and router are another device's, and a weight its graph binds and
		// the offer left out fails its composition by name.
		if p.Cut > 0 && len(x.Data) == 0 {
			continue
		}
		// A KV-sharing block has no k or v matrix (slots 1, 2): it attends to
		// its source's history.
		if p.KVShared && (i == 1 || i == 2) {
			continue
		}
		// Nor has a block whose v is k's projection (Gemma 4's global layers)
		// a v matrix: slot 2. The emit reads k's projection for v.
		if p.VFromK && i == 2 {
			if len(x.Data) != 0 {
				g.LastErr = fmt.Sprintf("block %d: v is k's projection and the block offers a v matrix", li)
				return false
			}
			continue
		}
		// Nor has a block whose graph reads no v matrix (its values are its
		// keys): slot 2 is empty by construction.
		if i == 2 && len(x.Data) == 0 && !bindsRole(p.Graph, jlm.RoleAttnV) {
			continue
		}
		// A block that is its mixer alone has no FFN at all.
		if p.NoFFN && i >= 4 && i <= 6 {
			continue
		}
		// A linear block has no k, v or o matrix: its one projection is slot 0
		// (Wq carries the fused q|k|v|gate), and offerRange clears the other three.
		if p.Recurrent.Conv > 0 && !p.Recurrent.WithAttn && i >= 1 && i <= 3 {
			continue
		}
		if i >= 10 && i < 13 {
			// A linear block's three; an attention block leaves them empty.
			if p.Recurrent.Conv == 0 {
				continue
			}
			// Kimi-Linear's KDA has no ssm_gate: its output gate is the
			// low-rank ssm_g_a/ssm_g_b pair. Kimi-K3's is one full-rank matrix,
			// here.
			if i == 10 && p.Recurrent.ChanDecay && len(w.SSM.GA.Data) != 0 {
				continue
			}
			// Mamba-1 has no ssm_ba: dt comes off its x_proj (slot 18).
			if i == 11 && p.Recurrent.Scan {
				continue
			}
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan is a linear block and %s is empty",
					li, weightRole(i))
				return false
			}
		}
		if i >= 7 && i < 10 {
			if p.NFFNShExp == 0 {
				continue // no shared expert in this architecture at all
			}
			// An ungated shared expert (Nemotron 3) has no gate matrix.
			if i == 7 && ungatedFFN(p) {
				continue
			}
			// A plan with a shared expert and an empty weight is a decline: a
			// skip would run the FFN on an unwritten buffer.
			if len(x.Data) == 0 {
				g.LastErr = fmt.Sprintf("block %d: the plan has a shared expert "+
					"(NFFNShExp=%d) and %s is empty", li, p.NFFNShExp,
					[...]string{"ShGate", "ShUp", "ShDown"}[i-7])
				return false
			}
		}
		e := 1
		// MLA's two absorb banks are one sheet per head, and the "expert" indexed
		// is the head: every head is selected on every token, so the selection is
		// the identity 0..NHead-1 and one indexed matvec replaces NHead, each slot
		// reading its own activation like the mixture's down projection.
		if l.mla && (i == 16 || i == 17) {
			e = p.NHead
		}
		// 4..6 only: the shared expert's matrices at 7..9 are dense.
		if moe && i >= 4 && i <= 6 { // gate, up and down are the banks
			e = bank
			if stream {
				// The kernel is told the bank is k experts and indexes it with
				// 0..k-1; engine.TestStreamedExpertBankMatchesTheResidentOne
				// holds it to the resident bank bit for bit. With an expert
				// cache it is told the cache's size and indexed by slot.
				if cslots < 0 {
					// An auto-streamed block is placed on the plain bank;
					// its cache comes after placement (sizeAutoCaches).
					cslots = g.cacheSlotsFor(p, ws, g.StreamCacheSlots, slots, bank)
				}
				e = cslots
			}
		}
		// Weight.Rows already counts the whole bank (model.get multiplies every
		// trailing dimension), which PackWeights and the upload want; the kernel is
		// built for one expert's rows and does the offset, so it is divided below.
		var r *resident
		if stream && i >= 4 && i <= 6 {
			q, qok := quantOf(x.T)
			if !qok {
				g.LastErr = fmt.Sprintf("block %d: %s is %s, which has no device kernel",
					li, weightRole(i), x.T)
				return false
			}
			if x.Packed == nil && (!layout.IsRaw(q) || len(x.Data) == 0) {
				// A streamed bank sends a sheet at a time: a packed bank's
				// planes, or a raw bank's own rows (its one plane,
				// expert-major) -- a float or a verbatim format's.
				g.LastErr = fmt.Sprintf("block %d: %s holds neither its packed planes nor its rows",
					li, weightRole(i))
				return false
			}
			var sh [3]int
			var ok bool
			if cslots > slots {
				// The block's own bank of cslots sheets, kept across tokens:
				// it is the expert cache, so it cannot be the device's one
				// shared compact bank. Its charge is the block's.
				r, sh, ok = g.residentCompact(q, x.Rows/slots, x.K, cslots)
			} else {
				r, sh, ok = g.sharedCompact(i, q, x.Rows/slots, x.K, slots)
			}
			if !ok {
				return false
			}
			if l.stream == nil {
				// w is seeded here so a host side that does not page (Ensure nil)
				// still has valid spans; otherwise fill() sends nothing and the FFN
				// contributes zeros.
				l.stream = &streamBank{sel: make([]uint32, slots), fr: fr,
					ensure: w.Ensure, sel2: w.EnsureExperts,
					pre: w.PrefetchExperts, spec: w.SpeculateExperts, w: *w, li: li}
				l.stream.nExpert = bank
				l.stream.initCache(cslots, bank)
				g.StreamBlocks++
			}
			l.stream.sh[i-4] = sh
		} else {
			r = mkw(i, x)
		}
		if r == nil || !r.ok {
			if g.LastErr == "" {
				g.LastErr = fmt.Sprintf("block %d tensor %d (%s %dx%d) not resident",
					li, i, x.T, x.Rows, x.K)
			}
			return false
		}
		*rs[i] = r
		if stream && i >= 4 && i <= 6 {
			// A streamed bank's rows were scaled to the selection (slots
			// sheets) above, whatever size its cache is.
			x.Rows /= slots
		} else if e > 1 {
			// e, not bank: they differ for a streamed bank, whose kernel is built for
			// k experts. This is rows per expert either way.
			if x.Rows%e != 0 {
				return false
			}
			x.Rows /= e
		}
	}
	if moe {
		// Resident in the device layout of the type the file stores, as every
		// other matrix, and permanent while the block is placed: a page-out
		// leaves it (permBytes charged it), and dropLayerLocal frees it.
		if w.Router.Rows != p.NExpert || w.Router.K != p.NEmbd {
			return false
		}
		r := mkw(slotRouter, w.Router)
		if r == nil || !r.ok {
			if g.LastErr == "" {
				g.LastErr = fmt.Sprintf("block %d: %s (%s %dx%d) not resident", li, jlm.RoleRouter, w.Router.T, w.Router.Rows, w.Router.K)
			}
			return false
		}
		l.router = r
	}
	var err error
	// Norms and biases arrive as f32, which is what the device reads.
	up := func(dst *backend.Buf, v []float32) {
		// A KV-sharing block's k norm is absent: it caches nothing.
		if err != nil || len(v) == 0 {
			return
		}
		f := v
		var b backend.Buf
		if b, err = g.dev.Alloc(len(f) * 4); err != nil {
			return
		}
		err = b.Write(f32b(f))
		*dst = b
	}
	// A block's pre-norms are the ones its file stores (none on a post-norm
	// block or an encoder's, no FFN norm on a parallel block sharing one);
	// the composed graph binds what is stored.
	if w.AttnNorm != nil {
		up(&l.nAttn, w.AttnNorm)
	}
	if w.FFNNorm != nil {
		up(&l.nFFN, w.FFNNorm)
	}
	if w.Clamp != nil {
		if len(w.Clamp) != 28 {
			g.LastErr = fmt.Sprintf("block %d: clipped-linear bounds are %d floats, want 28", li, len(w.Clamp))
			return false
		}
		up(&l.clampB, w.Clamp)
	}
	// Gemma 4's per-layer embedding: the block's slice offset.
	if p.PLEDim != 0 {
		if err == nil {
			if l.pleOff, err = g.dev.Alloc(4); err == nil {
				err = l.pleOff.Write(g.u32b(uint32(li * p.PLEDim)))
			}
		}
		l.auxBytes += 4
	}
	l.wide = p.NStreams > 1 || p.ExpLatent != 0
	// Gemma 4's output scalar, one float, charged with the norms.
	if p.OutScale != 0 {
		up(&l.outScale, []float32{p.OutScale})
		l.auxBytes += 4
	}
	if w.AttnNormB != nil {
		up(&l.nAttnB, w.AttnNormB)
	}
	if w.FFNNormB != nil {
		up(&l.nFFNB, w.FFNNormB)
	}
	if w.QNorm != nil {
		up(&l.nQ, w.QNorm)
		up(&l.nK, w.KNorm)
	}
	if r := p.Recurrent; r.Conv > 0 {
		if len(w.SSM.Conv1d) != r.Conv*r.Chans {
			g.LastErr = fmt.Sprintf("block %d: conv1d is %d floats, want %d x %d",
				li, len(w.SSM.Conv1d), r.Conv, r.Chans)
			return false
		}
		// The composed convolution reads its taps channel-major, the causal
		// one's and the centred one's (graph.OpRowConv) alike.
		taps := make([]float32, len(w.SSM.Conv1d))
		for u := range r.Conv {
			for c := range r.Chans {
				taps[c*r.Conv+u] = w.SSM.Conv1d[u*r.Chans+c]
			}
		}
		up(&l.ssmTaps.qs, taps)
		l.ssmTaps = resident{qs: l.ssmTaps.qs, nrows: r.Chans, k: r.Conv, t: layout.Float32, ok: true}
		l.auxBytes += uint64(len(taps) * 4)
	}
	for _, v := range w.Vectors {
		// A vector the block holds as a matrix too (a 2-D one) is a vector
		// still, to a graph that reads it as one -- but not one a field holds,
		// as a vector or as a weight laid out for it (a linear block's taps):
		// a block binds a role as a vector wherever the card holds one.
		if _, named := l.namedVector(v.Role); named || l.namedWeight(v.Role) != nil {
			continue
		}
		var b backend.Buf
		up(&b, v.V)
		if b != nil {
			l.vecs = append(l.vecs, roleBuf{role: v.Role, buf: b})
			l.auxBytes += uint64(4 * len(v.V))
		}
	}
	if err == nil {
		err = g.openNGram(l, p, fr)
	}
	// One extra KV position is a trash slot for the batched prefill: a ragged
	// last chunk runs the full grid width with surplus rows zeroed, and those rows
	// still write K and V. A chunk-width of slack costs too much VRAM and a real
	// row's slot would be clobbered, so they all write here and nothing reads it.
	//
	// A linear block takes no KV cache at all: it keeps a running summary, and
	// the plan's NKVHead/HeadDim are the model-wide attention shape.
	linear := p.Recurrent.Conv > 0
	l.linear = linear
	l.withAttn = p.Recurrent.WithAttn
	l.noFFN = p.NoFFN
	if linear {
		if !g.addSessionRec(sid, l, p) {
			return false
		}
	}
	// A block that runs attention keeps a history, beside a recurrent state
	// where it runs a mixer too.
	attends := !linear || l.withAttn
	var cache, vcache int
	var kvp *kvPair
	l.kvSrc = li
	if p.KVShared {
		// No history of its own: it reads its source's, which must be on
		// this device already (blocks are offered in order).
		src := g.layers[p.KVSource]
		if src == nil || !src.ok || p.KVSource >= li {
			g.LastErr = fmt.Sprintf("block %d attends to block %d's history, which is not on this device",
				li, p.KVSource)
			return false
		}
		l.kvSrc = p.KVSource
	}
	if p.KVShared {
		// Nothing to allocate.
	} else if attends && g.pagedPlan(p) {
		err = g.pagedLayer(sid, l, li, p)
	} else if attends {
		g.initKVCap(p)
		cp := g.capped(p)
		// The V cache is half the size when packed (KVF16), which can decide
		// whether a model fits on the card at all. Under MLA it is zero: the value
		// is the key row's own prefix (kvPair.value).
		cache, vcache = g.kvCacheBytes(p, cp.MaxSeq)
		// One cache per session. PrepLayer runs for a session (sid), so
		// this allocates for whichever sequence is placing blocks; a second
		// Instance placing the same block reuses the shared weights above and gets
		// its own history. A non-causal block takes the device's one shared
		// pair instead (transientkv.go), and charges nothing of its own.
		kvp = &kvPair{}
		if p.NonCausal {
			if sh := g.transientKV(p); sh != nil {
				kvp, cache, vcache = sh, 0, 0
			}
		}
		for i, dst := range []*backend.Buf{&kvp.kc, &kvp.vc} {
			if kvp.shared {
				break
			}
			n := cache
			if i == 1 {
				n = vcache
			}
			if err == nil && n > 0 {
				*dst, err = g.dev.Alloc(n)
			}
		}
		if l.kv == nil {
			l.kv = map[uint64]*kvPair{}
		}
		if old := l.kv[sid]; old != nil && old.shared {
			g.transientKVPut(l.geo)
		} else if old != nil {
			// Re-preparing a block this session already holds: its previous
			// history goes, or it leaks (see dropLayerLocal).
			if old.kc != nil {
				old.kc.Free()
			}
			if old.vc != nil {
				old.vc.Free()
			}
		}
		l.kv[sid] = kvp
	}
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	// WithPoisonKV fills the cache with NaN so an attention kernel reading a
	// position nothing wrote fails a gate instead of reading a lucky zero.
	if g.kb.poisonKV && kvp != nil && !kvp.shared {
		nan := make([]byte, cache)
		for i := 0; i+3 < len(nan); i += 4 {
			nan[i], nan[i+1], nan[i+2], nan[i+3] = 0x01, 0x00, 0xC0, 0x7F
		}
		kvp.kc.Write(nan)
		if kvp.vc != nil {
			kvp.vc.Write(nan[:vcache])
		}
	}
	// cache + vcache, not 2*cache: with KVF16 the V half is half the size.
	l.kvBytes = uint64(cache + vcache)
	if kvp != nil && !kvp.shared {
		kvp.bytes = l.kvBytes
	}
	// += because a linear block already charged its five vectors above, and
	// auxBytes is what ReleaseLayers refunds.
	l.auxBytes += uint64((2*len(w.AttnNorm) + len(w.Clamp)) * 4)
	g.charge(l.auxBytes + l.kvBytes)
	g.KVBytes += l.kvBytes
	l.ok = true
	// The page record is built for every block, including on a model that fits,
	// so a later demotion is expressible; the budget decides whether anything
	// moves. Its size is what the residents actually charged, so reclaim() asks
	// for exactly what a page-in will spend.
	var pb, pr uint64
	for _, rp := range l.tensors() {
		r := *rp
		pb += r.bytes()
		if n := r.bytes(); n > 0 {
			if f, err := g.planesFootprint(r.t, r.nrows, r.k); err == nil && f > n {
				pr += f - n
			}
		}
	}
	if stream {
		// A streamed block is pinned (pg == nil, which every pager path reads as
		// "resident, never a victim"): its base is small and its bank is not
		// resident, so there is nothing to relocate. Paging it would also be
		// wrong, since pageIn would refresh the bank at NExpert sheets against an
		// admitted shape of k.
		l.pg = nil
	} else {
		l.pg = &page{ws: append([]nn.Weight(nil), ws...), bytes: pb, round: pr, in: true,
			// Kept so a page-in can ask again: ws points into a reused frame.
			ensure: w.Ensure, fr: fr}
		g.pagesIn++
		// Inside the else because ReleaseLayers refunds pageBytes only when
		// l.pg.in was true.
		g.pageBytes += pb + pr
	}
	// Re-preparing a block frees the layer it replaces. Only the per-layer
	// buffers go: the weights live in g.res keyed by their bytes and may be
	// shared with the layer being installed.
	if old := g.layers[li]; old != nil && old != l {
		g.dropLayerLocal(old)
	}
	g.layers[li] = l
	l.join(sid)
	g.layerGen++
	g.placeGen.Add(1)
	// The reserved widths' scratches and the composer's prompt forms of the
	// block at those widths are the placement's (promptForms), so no prompt
	// or step builds them on a card the blocks have filled.
	if !p.NonCausal {
		was := g.prewarming
		g.prewarming = true
		g.formAhead(func() {
			for _, w := range []int{g.promptW, g.stepW} {
				if w > 0 {
					g.prepBatch(w)
				}
			}
		})
		g.prewarming = was
	}
	// Admitting a block charges its KV cache forever, so the slot count just
	// fell; give back whatever no longer fits under it.
	g.trim()
	return true
}

// poisonBuf fills a freshly allocated device buffer with NaN when the knob is
// on. It is the one place the two poison sites agree on the pattern.
func (g *devTier) poisonBuf(b backend.Buf, n int) {
	if g.kb.poisonScratch && b != nil && n > 0 {
		b.Write(nanFill(n))
	}
}

// nanFill is n bytes of float32 NaN, for poisoning a scratch allocation.
func nanFill(n int) []byte {
	b := make([]byte, n)
	for i := 0; i+4 <= n; i += 4 {
		// 0x7FC00001, a quiet NaN, in little-endian.
		b[i], b[i+1], b[i+2], b[i+3] = 0x01, 0x00, 0xC0, 0x7F
	}
	return b
}

// initRopePlanes uploads this scratch's folded rotary planes, once, for the
// composer's rotary tables (tableBinds), and zeroes the tables: a table
// writes only its live rows, so the padded tail must hold zeros already.
// Nothing else writes bs.cs, so this deliberately un-poisons it under
// WithPoisonScratch.
func (g *devTier) initRopePlanes(bs *blockScratch, p *nn.LayerPlan, rows int) error {
	if gridded(p) || p.NRot <= 0 || len(p.RopeTab) == 0 {
		return nil
	}
	up := func(dst *backend.Buf, v []float32) error {
		b, err := g.dev.Alloc(len(v) * 4)
		if err != nil {
			return err
		}
		*dst = b
		return b.Write(f32b(v))
	}
	if err := up(&bs.ropeTab, p.RopeTab); err != nil {
		return err
	}
	if p.SWAPeriod > 0 && len(p.RopeTabSWA) > 0 {
		if err := up(&bs.ropeTabSWA, p.RopeTabSWA); err != nil {
			return err
		}
	}
	z := make([]byte, rows*p.NRot*4)
	for _, t := range []backend.Buf{bs.cs, bs.csSWA} {
		if t == nil {
			continue
		}
		if err := t.Write(z); err != nil {
			return err
		}
	}
	return nil
}

// initRopeAxes holds what the composer's table over several position axes
// reads where the plan has one (RopeTabAxes): its planes, uploaded once, and
// the rows' positions, staged every call.
func (g *devTier) initRopeAxes(bs *blockScratch, p *nn.LayerPlan, rows int) error {
	if len(p.RopeTabAxes) == 0 || gridded(p) || p.NRot <= 0 || bs.cs == nil {
		return nil
	}
	b, err := g.dev.Alloc(len(p.RopeTabAxes) * 4)
	if err != nil {
		return err
	}
	if err := b.Write(f32b(p.RopeTabAxes)); err != nil {
		b.Free()
		return err
	}
	a, err := g.dev.Alloc((1 + 4*rows) * 4)
	if err != nil {
		b.Free()
		return err
	}
	bs.ropeTabAxes, bs.rposAxes = b, a
	return nil
}

// initRopeRows holds the rows the composed rotary tables read (ropeRowsWords),
// staged every call, and makes the device's marks arena (markArena).
func (g *devTier) initRopeRows(bs *blockScratch, p *nn.LayerPlan, rows int) error {
	if gridded(p) || p.NRot <= 0 || bs.cs == nil {
		return nil
	}
	b, err := g.dev.Alloc((1 + 2*rows) * 4)
	if err != nil {
		return err
	}
	bs.ropeRows = b
	return g.growMarks(1)
}

// initRowPos holds the rows' positions a block reads (rowPosWords), staged
// every call.
func (g *devTier) initRowPos(bs *blockScratch, rows int) error {
	b, err := g.dev.Alloc((1 + rows) * 4)
	if err != nil {
		return err
	}
	bs.rowPos = b
	return nil
}

// initScratch allocates what a call over the plan's blocks binds and no block
// owns: the residual and its twin, the rotary tables and the rows they are
// computed from, the rows' counts, positions and tokens, the descriptors a
// picture's rows or a paged history are read through, and the host's
// staging. Every block's intermediate values are its composed part's own.
func (g *devTier) initScratch(p *nn.LayerPlan, rows int) (out *blockScratch) {
	// Everything the build allocates is the scratch's, charged (scratch.go).
	defer g.scratchWin().close()
	if rows < 1 {
		rows = 1
	}
	// The capped plan: every bound below reads bs.p, so the scratch is built
	// for what the device holds, not the full context.
	cp := g.capped(p)
	p = &cp
	// A picture's rows go in query chunks where its blocks' planes do not fit
	// one (visionChunk), so the scratch is rounded up to whole chunks.
	if p.NonCausal && rows > 1 {
		_, rows = visionChunk(rows, p.NHead, scoreStride(p, rows), g.tiles.qtile, g.tiles.atile, g.planeBudget())
	}
	bs := &blockScratch{p: cp, rows: rows, qtile: 1, ktile: 1, atile: 1}
	// A build that fails part-way frees what it allocated: on a full card --
	// the case that refuses -- what it leaked was what the next build needed.
	defer func() {
		if out == nil {
			freeScratch(bs)
		}
	}()
	// What the build allocated, by the device's own count: a tower's set is
	// resized from it (visrows.go).
	defer func(a0 int64) {
		if out != nil {
			out.bytes = g.allocated() - a0
		}
	}(g.allocated())
	if rows > 1 {
		bs.qtile, bs.ktile, bs.atile = g.tiles.qtile, g.tiles.ktile, g.tiles.atile
		for rows%bs.qtile != 0 {
			bs.qtile /= 2
		}
		for rows%bs.atile != 0 {
			bs.atile /= 2
		}
	}
	var err error
	al := func(dst *backend.Buf, n int) {
		// cuMemAlloc(0) is an error, not an empty buffer; an absent buffer is
		// right, since nothing reads it.
		if n == 0 || err != nil {
			return
		}
		// The device refusing scratch is the card being full, the same as a
		// refused weight upload: counted as NoRoom.
		if *dst, err = g.dev.Alloc(n); err != nil {
			g.NoRoom++
		}
		// RULE 13: zero is the identity for a sum, so a form reading scratch it
		// did not write is right on the first session and wrong on the next.
		// WithPoisonScratch fills with NaN so the read shows in the first token.
		if err == nil && g.kb.poisonScratch && *dst != nil {
			(*dst).Write(nanFill(n))
		}
	}
	// Every stream, stream-major (altup.go); one stream elsewhere.
	al(&bs.x, rows*p.ResidW()*4)
	al(&bs.h, rows*p.ResidW()*4)
	// The score row a picture's chunks are sized by (visionChunk).
	bs.sstride = scoreStride(p, rows)
	bs.arows = rows
	if p.NonCausal {
		bs.arows, _ = visionChunk(rows, p.NHead, bs.sstride, bs.qtile, bs.atile, g.planeBudget())
	}
	al(&bs.cs, rows*p.RopeW()*4)
	if p.SWAPeriod > 0 {
		al(&bs.csSWA, rows*p.RopeW()*4)
	}
	if err == nil {
		err = g.initRopePlanes(bs, p, rows)
	}
	if err == nil {
		err = g.initRopeAxes(bs, p, rows)
	}
	if err == nil {
		err = g.initRopeRows(bs, p, rows)
	}
	if err == nil {
		err = g.initRowPos(bs, rows)
	}
	// A linear block's run descriptors (composer.RunDescWords): its rows and
	// each sequence's slots in the block's pool, and the composer's, where
	// the pool is one shared half its forms step in place.
	if r := cp.Recurrent; r.Conv > 0 {
		bs.rec = r
		al(&bs.dRows, 4*composer.RunDescWords(rows))
		al(&bs.cRows, 4*composer.RunDescWords(rows))
	}
	al(&bs.n, (rows+1)*4)
	al(&bs.calls, 4)
	al(&bs.headCalls, 4)
	if p.NonCausal && p.Graph != nil {
		al(&bs.pics, rows*composer.RowWords*4)
		al(&bs.picsW, rows*composer.RowWords*4)
		bs.picStage = make([]byte, rows*composer.RowWords*4)
		bs.picStageW = make([]byte, rows*composer.RowWords*4)
	}
	if gridded(p) && len(p.Tables) == 1 && p.Tables[0] != nil {
		bs.gridGraph = p.Tables[0]
		al(&bs.gridPos, (1+rows)*4)
		al(&bs.gridReq, (composer.RequestRows+rows)*4)
		al(&bs.gridPlanes, 4*len(p.GridPlanes))
	}
	al(&bs.roff, rows*4)
	al(&bs.zero, 4)
	al(&bs.req, (composer.RequestRows+rows)*4)
	// The per-layer inputs.
	if p.PLEWidth != 0 {
		al(&bs.pleIn, rows*p.PLEWidth*4)
	}
	if err != nil {
		g.LastErr = err.Error()
		return nil
	}
	if bs.gridPlanes != nil {
		if err = bs.gridPlanes.Write(f32b(p.GridPlanes)); err != nil {
			g.LastErr = err.Error()
			return nil
		}
	}
	if err = bs.zero.Write([]byte{0, 0, 0, 0}); err != nil {
		g.LastErr = err.Error()
		return nil
	}
	// Each row's offset into a picture's query rows, which the tower's
	// forms read (towerserve.go).
	qdim := p.NHead * p.HeadDim
	roff := make([]byte, 0, rows*4)
	for r := range rows {
		roff = append(roff, u32one(uint32(r*qdim))...)
	}
	if err = bs.roff.Write(roff); err != nil {
		g.LastErr = err.Error()
		return nil
	}
	if g.pagedPlan(p) {
		if err := g.initPagedScratch(bs); err != nil {
			g.LastErr = "paged attention: " + err.Error()
			return nil
		}
	}
	bs.hx = make([]float32, rows*p.ResidW())
	bs.hcs = make([]float32, rows*p.RopeW())
	if p.SWAPeriod > 0 {
		bs.hcsSWA = make([]float32, rows*p.RopeW())
	}
	bs.hraw = make([]byte, rows*p.ResidW()*4)
	bs.hout = unsafe.Slice((*float32)(unsafe.Pointer(&bs.hraw[0])), rows*p.ResidW())
	return bs
}

// pickLanes settles, once per device, whether the kernels whose arithmetic
// needs a 32-lane subgroup (the butterfly softmax, the per-head norm) may be
// used. It asks the device for a promise: CUDA's warp and Apple's simdgroup are
// 32 by architecture; a Vulkan device promises 32 when its min and max subgroup
// sizes are both 32 or it lets the width be pinned at pipeline creation. The
// reported default subgroupSize is not a promise (see backend.GuaranteedLanes).
//
// A refusal selects the thread-per-head twins, which run every model correctly,
// so this is a selection rather than a capability gate. WithLanes(1) forces the
// scalar twins for a paired measurement; WithLanes(32) forces the kernel but not
// the promise, so a refusing backend still declines the pipeline and the block
// runs on the host. Reintroducing the bug is vulkan subgroup "force", which
// scripts/three-way.sh --violate subgroup-width sets.
func (g *devTier) pickLanes() int {
	if g.Lanes != 0 {
		return g.Lanes
	}
	ok, why := backend.GuaranteedLanes(g.dev, ir.SubgroupLanes)
	g.Lanes, g.LanesWhy = 1, why
	if ok {
		g.Lanes, g.LanesWhy = ir.SubgroupLanes, ""
	}
	switch g.kb.lanes {
	case 1:
		g.Lanes, g.LanesWhy = 1, "WithLanes(1)"
	case ir.SubgroupLanes:
		g.Lanes, g.LanesWhy = ir.SubgroupLanes, "WithLanes(32)"
		if why != "" {
			g.LanesWhy += ", over the device's own answer: " + why
		}
	}
	return g.Lanes
}

// The softmax is picked by correctness, not speed. A load-time timing probe
// runs at pos=1, where the choice does not matter; the warp reduction's gain
// grows with context. `verify -ab softmax` at a long -n is how it is checked.

// headOut is the buffer the head's answer is read from: the capped logits
// where the model has a final softcap, the projection's own otherwise.
// servedStreams is a block of n residual streams' inputs and outputs at a
// step of rows: stream k of x, a view at k*rows rows of the stream-major
// residual, each stream's input and output the same view, the per-layer
// inputs after the inputs. Made once a step width; a warm step makes
// nothing.
func (bs *blockScratch) servedStreams(rows, n int) (in, out []backend.Buf, err error) {
	if v, ok := bs.streams[rows]; ok && len(v[1]) == n {
		return v[0], v[1], nil
	}
	w := rows * bs.p.NEmbd * 4
	out = make([]backend.Buf, n)
	for k := range out {
		if out[k], err = backend.View(bs.x, k*w, w); err != nil {
			return nil, nil, fmt.Errorf("tier: stream %d of a residual of %d rows: %w", k, rows, err)
		}
	}
	in = append(append([]backend.Buf(nil), out...), bs.pleIn)
	if bs.streams == nil {
		bs.streams = map[int][2][]backend.Buf{}
	}
	bs.streams[rows] = [2][]backend.Buf{in, out}
	return in, out, nil
}

// servedStreamsApart is servedStreams with each stream's output a view of h
// at x's offset, for a form that writes a stream before it last reads it:
// the copy after (servedCopy) moves them over x.
func (bs *blockScratch) servedStreamsApart(rows, n int) (in, out []backend.Buf, err error) {
	if v, ok := bs.streamsApart[rows]; ok && len(v[1]) == n {
		return v[0], v[1], nil
	}
	if in, _, err = bs.servedStreams(rows, n); err != nil {
		return nil, nil, err
	}
	w := rows * bs.p.NEmbd * 4
	out = make([]backend.Buf, n)
	for k := range out {
		if out[k], err = backend.View(bs.h, k*w, w); err != nil {
			return nil, nil, fmt.Errorf("tier: stream %d of a residual of %d rows apart from x: %w", k, rows, err)
		}
	}
	if bs.streamsApart == nil {
		bs.streamsApart = map[int][2][]backend.Buf{}
	}
	bs.streamsApart[rows] = [2][]backend.Buf{in, out}
	return in, out, nil
}

// headOut is the head's logits, every stage its graph states applied.
func (bs *blockScratch) headOut() backend.Buf { return bs.logits }

// ragOut is headOut for the per-row head.
func (bs *blockScratch) ragOut() backend.Buf { return bs.ragLogits }

// HeadResident reports whether the vocabulary projection is on the device.
//
// It is not derivable from the block count, which is the reason it exists:
// placeHead gives blocks up to make room for the projection, so the run with
// the better placement is the one showing fewer blocks.
func (g *devTier) HeadResident() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.headPlaced(g.bs)
}

// headScratch gives a device that holds no block the scratch the output
// projection runs in, built from the plan of the device that holds the last
// block (see GPU.PrepHead). A device that already has one keeps it.
func (g *devTier) headScratch(p *nn.LayerPlan) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.bs != nil {
		return true
	}
	g.initKVCap(p)
	g.bs = g.initScratch(p, 1)
	g.shapeChanged()
	return g.bs != nil
}

// dropHeadScratch frees what PrepHead allocated around the projection -- the
// norm, its bias, the logits and the softcap -- and refunds it, leaving the
// projection's resident to whoever holds it. Callers hold g.mu.
func (g *devTier) dropHeadScratch(bs *blockScratch) {
	g.shapeChanged()
	g.unstreamHead(bs)
	// So does a PrepHead after a seam gave its blocks back and grew again: the
	// norm, its bias and the logits are allocated there, and overwriting them
	// leaked all three (and their charge) on every round trip. bs.head is set
	// only where those charges were made.
	if bs.head != nil {
		n := uint64(bs.p.NEmbd*4 + bs.mvHead.rows*4)
		if bs.hNormB != nil {
			n += uint64(bs.p.NEmbd * 4)
		}
		if bs.hBias != nil {
			n += uint64(bs.mvHead.rows * 4)
		}
		g.refund(n)
	}
	for _, b := range []*backend.Buf{&bs.hNorm, &bs.hNormB, &bs.hBias, &bs.logits} {
		if *b != nil {
			(*b).Free()
			*b = nil
		}
	}
	bs.head = nil
	g.placeGen.Add(1)
	g.freeAltNorms()
}

// headGraphs records the head as ops from the model's front end, what the
// composer serves: the model's, so every lane's and every geometry set's. A
// later instance of the model, its graphs restated, offers them. Callers hold
// g.mu.
func (g *devTier) headGraphs(h *nn.Head) {
	g.eachLane(func(l *lane) {
		ds := []*blockScratch{l.bs}
		for _, set := range l.geoSets {
			ds = append(ds, set.bs)
		}
		for _, d := range ds {
			if d != nil {
				d.headGraph, d.headHidden, d.sampleGraph, d.embedGraph = h.Graph, h.HiddenGraph, h.Sample, h.Embed
				d.sampleWalk, d.pickGraph, d.pickHead = h.SampleWalk, h.PickGraph, h.PickHead
			}
		}
	})
}

// PrepHead uploads the output norm and the vocabulary projection.
func (g *devTier) PrepHead(h *nn.Head) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	defer g.placeGen.Add(1)
	g.dropGraph()
	bs := g.bs
	if bs == nil || h == nil {
		return false
	}
	g.headGraphs(h)
	h.GatherRow = h.Embed != nil
	g.prepCollapse(bs, h)
	// The head is the model's: a second instance on this tier finds it placed.
	// Building it again allocated a second norm and logits buffer over the
	// first ones, leaking both and charging the budget twice.
	if g.headPlaced(bs) && bs.mvHead.rows == h.W.Rows {
		return g.adoptHead(h)
	}
	g.dropHeadScratch(bs)
	q, ok := quantOf(h.W.T)
	if !ok || h.W.K != bs.p.NEmbd || h.W.Rows <= 0 || len(h.W.Data) == 0 {
		return false
	}
	// The reserved width's per-row logits are the head's (prepRowsReserved),
	// admitted with it: built after it without asking, they put the card past
	// its budget by a vocabulary a row, and a ragged step on a full card could
	// not build them.
	rowsNeed := func() uint64 {
		if bb := g.reservedFor(nn.ComposedBlock); bb != nil && g.rowsHeld() {
			return g.rowsHeadNeed(bb, h.W.Rows, h.Softcap)
		}
		return 0
	}
	rows := rowsNeed()
	foot, ferr := g.planesFootprint(q, h.W.Rows, h.W.K)
	if ferr != nil {
		foot = uint64(len(h.W.Data))
	}
	// A streamed projection is a page (headPage), unless the input rows read
	// it here and hold its bytes for every token.
	planes := foot
	stream := h.Stream && !(h.Embed != nil && h.Embeds)
	// The norm's bias and the output bias are the vectors the head's graph
	// names, held with the projection and admitted with it.
	spec, _ := headShape(h.Graph)
	if spec.hasNormBias && len(h.NormB) != len(h.Norm) {
		g.LastErr = fmt.Sprintf("a LayerNorm head with a %d-wide bias for a %d-wide norm", len(h.NormB), len(h.Norm))
		return false
	}
	if spec.hasBias {
		if len(h.Bias) != h.W.Rows {
			g.LastErr = fmt.Sprintf("an output bias of %d for a %d-row projection", len(h.Bias), h.W.Rows)
			return false
		}
		foot += uint64(h.W.Rows * 4)
	}
	// They shrink with the reserved width. A caller that did not list the CPU
	// has no other device for the head, so the width gives way first; with
	// the CPU listed the head goes home and the reservation holds.
	for h.NoCPU && rows > 0 && !g.room(foot+rows) && g.narrowReserved("the output projection's per-row logits") {
		rows = rowsNeed()
	}
	// Under paging the projection and a block slot compete for the same bytes,
	// and the head is permanent while a slot turns over every block. The head
	// runs once per token and a block NLayer times, so the projection is taken
	// only when it costs no slot.
	// A caller that did not list the CPU has no other device for it: the
	// projection is taken while a swap still has its slots, the streamed
	// blocks paging out to make the room.
	if stream {
		// What it holds for good is its bias and per-row logits; a slot holds
		// the projection as it passes, beside the block a run reads with it.
		was := g.widest
		g.widest = max(was, planes)
		n := g.slotsIfPerm(foot - planes + rows)
		g.widest = was
		if g.holdsPages() && n < minSlots+1 {
			g.LastErr = fmt.Sprintf("the output projection needs %d bytes and taking them "+
				"would leave %d block slot(s), below the %d a swap needs", foot+rows, n, minSlots+1)
			return false
		}
		if !g.room(foot+rows) && (len(g.stream) == 0 || !g.reclaim(foot+rows, -1, 0, 0)) {
			g.LastErr = fmt.Sprintf("the output projection needs %d bytes and no streamed block "+
				"can be paged out for it", foot+rows)
			return false
		}
	} else if want := foot + rows; len(g.stream) > 0 && h.NoCPU {
		after := g.slotsIfPerm(want)
		for after < minSlots && g.promptW > 0 && g.narrowReserved("the output projection and a block slot") {
			after = g.slotsIfPerm(want)
		}
		if after < minSlots {
			g.LastErr = fmt.Sprintf("the output projection needs %d bytes and taking them "+
				"would leave %d block slot(s), below the %d a swap needs", want, after, minSlots)
			return false
		}
		if !g.reclaim(want, -1, 0, 0) {
			g.LastErr = fmt.Sprintf("the output projection needs %d bytes and no streamed block "+
				"can be paged out for it", want)
			return false
		}
	} else if now, after := g.slots(), g.slotsIfPerm(want); len(g.stream) > 0 && after < now ||
		!h.NoCPU && g.holdsPages() && after < minSlots {
		// Paging is always on: a projection that leaves the blocks paged here
		// no slot to turn over in strands them.
		g.LastErr = fmt.Sprintf("the output projection needs %d bytes and "+
			"taking them would leave %d block slot(s) instead of %d for %d block(s) the card holds, so it "+
			"stays on the CPU: a slot turns over every block and the "+
			"projection runs once a token", want, after, now, g.pagesIn)
		return false
	}
	if rows > 0 && !g.room(foot+rows) {
		g.Declined++
		g.LastErr = fmt.Sprintf("the output projection needs %d bytes and its per-row logits %d more, %d of %d used",
			foot, rows, g.used, g.limit)
		return false
	}
	// -1: the output projection is not a block, so it is not released with
	// one. Read into the card's own frames unless a copy is held already.
	w := h.W
	if held := g.res[adhocKey(w.Entry, w.Data, q, w.Rows, w.K)]; held == nil || !held.ok {
		own, release, err := g.ownWeight(h.File, w)
		if err != nil {
			g.LastErr = "the output projection: " + err.Error()
			return false
		}
		defer release()
		w = own
	}
	r := g.resident(-1, 0, w.Entry, q, w.Data, w.Rows, w.K, w.Packed)
	if r == nil || !r.ok {
		return false
	}
	var err error
	// A head with no norm (Qwen3.8-Flash-Next's reads its mix as it is)
	// leaves hNorm nil, and the emit copies the row where the norm would
	// have written.
	if len(h.Norm) > 0 {
		f := make([]float32, len(h.Norm))
		for i, v := range h.Norm {
			f[i] = float32(v)
		}
		if bs.hNorm, err = g.dev.Alloc(len(f) * 4); err != nil {
			return false
		}
		if bs.hNorm.Write(f32b(f)) != nil {
			return false
		}
	}
	if spec.hasNormBias {
		if bs.hNormB, err = g.dev.Alloc(len(h.NormB) * 4); err != nil {
			return false
		}
		if bs.hNormB.Write(f32b(h.NormB)) != nil {
			return false
		}
		g.charge(uint64(len(h.NormB) * 4))
	}
	if bs.logits, err = g.dev.Alloc(h.W.Rows * 4); err != nil {
		return false
	}
	bs.hRaw = make([]byte, h.W.Rows*4)
	bs.hOut = unsafe.Slice((*float32)(unsafe.Pointer(&bs.hRaw[0])), h.W.Rows)
	bs.head, bs.mvHead = r, mv{rows: h.W.Rows, q: q, k: h.W.K}
	g.shapeChanged()
	g.headEntry, g.headSrc, g.headNormHost = h.W.Entry, &h.W.Data[0], slices.Clone(h.Norm)
	g.charge(uint64(len(h.Norm)*4 + h.W.Rows*4))
	g.prepEmbed(bs, h)
	// The input rows' tables are no slot's: a streamed projection with no
	// room beside them leaves the rows to the CPU's form of them, where the
	// caller listed it, rather than going home itself.
	if stream && !h.NoCPU && errors.Is(bs.embErr, errEmbedRoom) {
		h.Embed, h.GatherRow = nil, false
		g.headGraphs(h)
		bs.embW, bs.posW, bs.embErr = nil, nil, nil
	}
	// The input rows' tables are the head's: a head whose tables have no room
	// is given back, so placement buys the room as it does for the head
	// (placeHead), rather than met by its first step.
	if errors.Is(bs.embErr, errEmbedRoom) {
		g.giveBackHead(bs)
		g.Declined++
		g.LastErr = "the input rows' tables have no room beside the head and the blocks placed: " + g.LastErr
		return false
	}
	if spec.hasBias {
		b, e := g.dev.Alloc(h.W.Rows * 4)
		if e == nil {
			if e = b.Write(f32b(h.Bias)); e != nil {
				b.Free()
			}
		}
		if e != nil {
			g.giveBackHead(bs)
			g.LastErr = "the output bias: " + e.Error()
			return false
		}
		bs.hBias = b
		g.charge(uint64(h.W.Rows * 4))
	}
	// The greedy token on the device (nn.Head.ArgmaxOnly): the word the
	// composed sample writes. Without it the head still runs and every
	// caller gets its logits, so a failure is not one.
	if g.argmaxOut == nil && h.Sample != nil {
		w := g.scratchWin() // the device's own word (scratch.go)
		if b, e := g.dev.Alloc(4); e == nil {
			g.argmaxOut = b
		}
		w.close()
	}
	g.prewarmHeadKernels()
	// A head whose composed part has no room beside the blocks placed stays
	// home, as such a block does (prepLayer), rather than met by its first
	// step.
	if why := g.prewarmComposedHead(true); why != "" {
		g.giveBackHead(bs)
		g.Declined++
		g.LastErr = why
		return false
	}
	// A head graph binding what the tier holds no buffer for is the model's
	// error, named; it is never left to the tier's own kernels.
	if err := bs.headServes(bs.hNorm); err != nil {
		g.giveBackHead(bs)
		g.Declined++
		g.LastErr = "the composed head: " + err.Error()
		return false
	}
	if stream {
		g.streamHead(bs, h, q, planes)
	}
	g.prepRowsReserved()
	return true
}

// Layers is layersFor for the zero session: a caller that never attached.
func (g *devTier) Layers(lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	return g.layersFor(0, lo, hi, pos, n, x, cs, csSWA, head)
}

// layersFor runs blocks [lo, hi) at position pos as one submission.
//
// The read at the end is the only synchronise: consecutive launches on one
// stream are already ordered, so the whole prefix issues without the device
// draining. Those launches are captured once and replayed (a CUDA graph pays the
// per-launch cost once); see graphKey for why a decode token's sequence is
// stable enough to capture, and cuda/graph.go for the mechanism.
//
// It runs in session sid's view of the device, holding a lane for the call
// (devsess.go); a call reached from inside another keeps the lane it holds.
func (g *devTier) layersFor(sid uint64, lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	v := g.as(sid)
	if v == nil {
		return false
	}
	defer v.done()
	ok := v.layersCall(sid, lo, hi, pos, n, x, cs, csSWA, head)
	v.dropEmbed()
	return ok
}

// layersCall is Layers. The wrapper exists because an EmbedRows promise lives
// for exactly one Layers call: a call that refuses before submitting leaves the
// caller to fill x, and a surviving promise would be gathered into a later
// token's x.
func (g *devTier) layersCall(sid uint64, lo, hi, pos, n int, x, cs, csSWA []float32, head *nn.Head) bool {
	// Fault injection, present only in a `jitllmfault` build; see tier/fault.go.
	// In a release build injectedFail is `return false` and inlines away.
	if g.injectedFail() {
		return false
	}
	// A tower's range runs in lane0, whose tower set is the device's own
	// (visrows.go): a call holding a clone borrows lane0 for it.
	g.mu.Lock()
	if l := g.layers[lo]; lo < hi && l != nil && l.nonCausal && g.lane != g.lane0 {
		v0 := g.borrowLane0()
		g.mu.Unlock()
		ok := v0.layersCall(sid, lo, hi, pos, n, x, cs, csSWA, head)
		g.mu.Lock()
		g.returnLane0(v0)
		g.mu.Unlock()
		return ok
	}
	// Reset per call, because the caller reads it immediately after a false to
	// decide whether a host restart is sound.
	g.recSteps = 0
	// Gemma 4: the range's geometry's scratch set, and the caller's set again
	// on the way out (geomsets.go). GPU.runsInto hands one geometry per call.
	prev := g.geoCur
	if g.geoUsed {
		g.useGeom(g.geomFor(lo, hi))
	}
	g.mu.Unlock()
	defer g.restoreGeom(prev)
	// KV growth happens before anything is submitted, because it drops the
	// captured graph and dropGraph may not be called inside a backend Session
	// (a deadlock on CUDA). A non-causal range is decided first: its n is the
	// run's rows, not a position, and must not grow the causal capacity. It is
	// one whole run in its own geometry's scratch set, which the switch above
	// made current.
	// An empty range is the text head alone (PrepHead's call): its lo is the
	// block past the text model's last, which is a tower's first when a tower
	// follows, and says nothing about the call.
	if l := g.layers[lo]; lo < hi && l != nil && l.nonCausal {
		g.mu.Lock()
		bs := g.bs
		g.mu.Unlock()
		if bs == nil || n > bs.rows || head != nil {
			g.LastErr = fmt.Sprintf("non-causal n=%d scratch=%v head=%v", n, bs != nil, head != nil)
			return false
		}
		return g.submit(sid, bs, lo, hi, pos, x, cs, csSWA, nil)
	}
	if !g.ensureKVCap(pos + n) {
		return false
	}
	// Paged history takes the chunk's pages now, before any block is paged in
	// for the submission.
	g.pgRows = g.pagedRows(sid, g.pgRows, n, n, pos, nil)
	if !g.pagedAppendRows(sid, g.pgRows) {
		return false
	}
	// A prefill chunk, padded to a fixed width: every kernel bakes its row
	// count, so a ragged tail runs the full width. The surplus rows write K and V
	// at positions not yet reached, which decode overwrites, and no earlier row
	// attends to them because the scores mask past each row's causal count.
	if n != 1 {
		bw := batchWidth
		if n > bw {
			g.LastErr = fmt.Sprintf("batch n=%d width=%d head=%v", n, bw, head != nil)
			return false
		}
		// The width is the chunk's rounded up to the grain, so a short tail does
		// not run the full grid. A scratch per width costs VRAM, so the narrow one
		// is used only if it fits; otherwise the wide one pads.
		g.mu.Lock()
		grain := g.promptGrain() // f16GemmK probes once, under g.mu
		g.mu.Unlock()
		w := (n + grain - 1) / grain * grain
		if w > bw {
			return false
		}
		if g.NoTail {
			w = bw
		}
		g.mu.Lock()
		// The chunk's own width, or the width reserved at placement when the
		// narrow one does not fit (scratch.go); with no reservation, the full
		// width.
		// The chunk's own grain width first: batchFor rounds a width it has
		// not built to a power of two, which is right for a step loop whose
		// row count moves every step but ran a 415-token prompt over 512
		// rows, a quarter of its arithmetic on padding. prepBatch refuses a
		// width the budget has no room for, and the chunk then pads as before.
		//
		// Each exact width is budgeted beside the reservation. A few are
		// kept, the least recently used given back when the budget has no
		// room for the next or the lane holds maxExactWidths (keepExact);
		// a dropped width's kernels stay in the caches.
		//
		// A call the composer serves whole runs in the reserved width, its
		// rows the call's, and a longer one in chunks of that width: no
		// width is built, formed or compiled for a prompt's length.
		ww := 0
		composed := g.promptComposed(lo, hi)
		// A head over every row cannot be cut into the calls a streamed
		// block's pages hold.
		if composed && head != nil && head.RowLogits && n > g.pagedCall(lo, hi, n) {
			composed = false
		}
		switch {
		case composed && n <= g.promptW:
			ww = g.promptW
		case composed:
		default:
			if !g.NoExactPrompt && w != bw {
				if ww = g.keepExact(w); ww != 0 {
					g.ExactPrompts++
				}
			}
			if ww == 0 {
				ww = g.batchFor(w)
			}
			if ww == 0 && w != bw && g.prepBatch(bw) {
				ww = bw
			}
		}
		ok := ww > 0
		if ok {
			w = ww
		}
		bbs, dbs, sub := g.bbs[w], g.bs, max(g.promptW, nn.MaxPrefillChunk)
		if composed {
			sub = g.pagedCall(lo, hi, g.promptW)
		}
		// A head over every row (a speculative verification) runs only in
		// the one-submission path, with the per-row head built for it: every
		// fallback below splits the chunk and projects the last row alone.
		if ok && bbs != nil && head != nil && head.RowLogits {
			if dbs == nil || dbs.head == nil || !g.prepRowsHead(bbs, head, n) {
				ok = false
			} else if len(head.Tokens) < n {
				head.Tokens = make([]int32, n)
			}
		}
		// The fold copies the last row into the decode scratch and projects it
		// there (layersOnce), which a resident float head cannot take. Only a
		// resident head: the split runs the chunk before the head, and a head
		// that then failed would send a hybrid's advanced rows round again.
		// Never for a head over every row, which the per-row head above serves
		// and the split, projecting the last row alone, cannot.
		noFold := dbs != nil && dbs.head != nil && dbs.head.ok && bbs != nil &&
			!(head != nil && head.RowLogits) &&
			(layout.IsFloat(dbs.head.t) || dbs.p.NEmbd != bbs.p.NEmbd)
		g.mu.Unlock()
		if head != nil && head.RowLogits && (!ok || bbs == nil) {
			if g.LastErr == "" {
				g.LastErr = fmt.Sprintf("a %d-row chunk wanting every row's logits has no batched scratch", n)
			}
			return false
		}
		// A call the composer serves longer than the reserved width runs in
		// chunks of it, the last carrying the head, folded as a call of the
		// reserved width folds it; each chunk is its rows of every stream.
		// The head's own path below runs the blocks as a call of their own,
		// head-less: cut here like any other.
		if composed && n > sub && dbs != nil && !(head != nil && (head.RowLogits || noFold)) {
			nr, e, ns := dbs.p.RopeW(), dbs.p.NEmbd, dbs.p.Streams()
			g.mu.Lock()
			g.PromptSplits++
			g.mu.Unlock()
			side := g.side()
			defer g.sideAt(side, 0, n)
			if ns > 1 {
				g.realEmbed()
			}
			for off := 0; off < n; off += sub {
				m := min(sub, n-off)
				var rcs, rswa []float32
				if nr > 0 {
					rcs = cs[off*nr : (off+m)*nr]
					if len(csSWA) >= (off+m)*nr {
						rswa = csSWA[off*nr : (off+m)*nr]
					}
				}
				var h *nn.Head
				if off+m == n {
					h = head
				}
				xs := g.streamCut(x, ns, e, off, m, n)
				g.sideAt(side, off, n)
				if !g.layersFor(sid, lo, hi, pos+off, m, xs, rcs, rswa, h) {
					return false
				}
				streamPut(x, xs, ns, e, off, m, n)
			}
			return true
		}
		// Every fallback below cuts x by rows: each cut is its rows of every
		// stream and of the per-row inputs.
		ns := 1
		if dbs != nil {
			ns = dbs.p.Streams()
		}
		side := g.side()
		defer g.sideAt(side, 0, n)
		if ns > 1 && (head != nil && noFold || !ok || bbs == nil) {
			g.realEmbed()
		}
		// A head rides only the one-submission path. Every fallback splits the
		// chunk, so there the head runs on the last row as its own submission;
		// so does a head the fold cannot take, rather than the whole chunk
		// going a row at a time.
		if head != nil && (!ok || bbs == nil || noFold) {
			// A head-only call (lo == hi: the blocks ran in a submission of
			// their own, GPU.layersCall's tail) has no blocks to run first;
			// asking for the empty range without the head is a refusal, which
			// sent a chunk whose blocks had all run round again a row at a
			// time (MiniMax-M3's selecting blocks are a geometry of their own,
			// so its float head always came here).
			if dbs == nil || lo < hi && !g.layersFor(sid, lo, hi, pos, n, x, cs, csSWA, nil) {
				return false
			}
			e, nr := dbs.p.NEmbd, dbs.p.RopeW()
			var rcs, rswa []float32
			if nr > 0 {
				rcs = cs[(n-1)*nr : n*nr]
				if len(csSWA) >= n*nr {
					rswa = csSWA[(n-1)*nr : n*nr]
				}
			}
			// The head's call resets recSteps (layersCall), and the chunk's
			// blocks have advanced by then: a head that fails must still
			// report them, or the caller restarts a hybrid from the embeddings
			// and applies its linear blocks twice.
			g.mu.Lock()
			adv := g.recSteps
			g.mu.Unlock()
			xs := g.streamCut(x, ns, e, n-1, 1, n)
			g.sideAt(side, n-1, n)
			if !g.layersFor(sid, hi, hi, pos+n-1, 1, xs, rcs, rswa, head) {
				g.mu.Lock()
				g.recSteps += adv
				g.mu.Unlock()
				return false
			}
			return true
		}
		if (!ok || bbs == nil) && n > sub && dbs != nil {
			// A device that cannot hold the wide scratch takes the chunk in narrower
			// submissions -- the width it reserved, when it reserved one --
			// consecutive sub-chunks being what the caller's own chunk loop would
			// have issued.
			sw := sub
			nr := dbs.p.RopeW()
			g.mu.Lock()
			g.PromptSplits++
			g.mu.Unlock()
			for off := 0; off < n; off += sw {
				m := min(sw, n-off)
				var rcs, rswa []float32
				if nr > 0 {
					rcs = cs[off*nr : (off+m)*nr]
					if len(csSWA) >= (off+m)*nr {
						rswa = csSWA[off*nr : (off+m)*nr]
					}
				}
				e := dbs.p.NEmbd
				xs := g.streamCut(x, ns, e, off, m, n)
				g.sideAt(side, off, n)
				if !g.layersFor(sid, lo, hi, pos+off, m, xs, rcs, rswa, nil) {
					return false
				}
				streamPut(x, xs, ns, e, off, m, n)
			}
			return true
		}
		if (!ok || bbs == nil) && dbs == nil {
			return false
		}
		if (!ok || bbs == nil) && bidirCheck(g.bidir, pos, 1) != "" && g.attendsIn(lo, hi) {
			// One row at a time cuts a bidirectional run the chunk holds,
			// which attention cannot read row by row: refused before any
			// block runs (nothing has advanced), so the chunk restarts on
			// the host whole.
			g.LastErr = "a bidirectional chunk this device has no batched scratch for"
			return false
		}
		if !ok || bbs == nil {
			g.mu.Lock()
			g.PromptSplits++
			g.mu.Unlock()
			// A batch this device cannot prepare is not a device failure. prepBatch
			// refuses before submitting anything, so x is untouched and the rows go
			// one at a time. Returning false would restart the chunk on the host,
			// which on a hybrid re-runs linear blocks whose state already advanced.
			for i := 0; i < n; i++ {
				row := g.streamCut(x, ns, dbs.p.NEmbd, i, 1, n)
				g.sideAt(side, i, n)
				var rcs, rswa []float32
				if nr := dbs.p.RopeW(); nr > 0 {
					rcs = cs[i*nr : (i+1)*nr]
					if len(csSWA) >= (i+1)*nr {
						rswa = csSWA[i*nr : (i+1)*nr]
					}
				}
				// Each row is a token of the recurrence, and each linear block's
				// halves flip with it, so row 1's recording is keyed apart from
				// row 0's (rangeParity).
				if !g.submit(sid, dbs, lo, hi, pos+i, row, rcs, rswa, nil) {
					return false
				}
				streamPut(x, row, ns, dbs.p.NEmbd, i, 1, n)
			}
			return true
		}
		return g.submit(sid, bbs, lo, hi, pos, x, cs, csSWA, head)
	}
	if g.PerLayerSubmit && (hi-lo > 1 || head != nil) {
		// The other arm of the submission comparison: one Session and one Read
		// per block. The head becomes one more (empty-range) submission. submit
		// honours PerLayerSubmit itself, since it is also what pages a block in.
		if !g.submit(sid, g.bs, lo, hi, pos, x, cs, csSWA, nil) {
			return false
		}
		if head != nil {
			return g.layersOnce(sid, g.bs, hi, hi, pos, x, cs, csSWA, head)
		}
		return true
	}
	return g.submit(sid, g.bs, lo, hi, pos, x, cs, csSWA, head)
}

// parityMask is the halves a submission's linear blocks read, one bit per
// linear block of its range in order: what a recording of that range is keyed
// by, since a recording names one half as source and the other as
// destination. Four words cover 256 linear blocks; a range with more is never
// recorded (rangeParity).
type parityMask [4]uint64

// rangeParity is the halves the linear blocks in [lo, hi) read next -- the
// running step's (stepPar), so a step across sessions keys on the half its
// rows were aligned to. It is the range's own rather than the device's: a
// device holding two runs of one model (an explicit placement) submits twice a
// token, and a device-wide flip per submission would come back to the same
// parity every token while each block's halves alternate -- replaying a
// recording that reads the stale half. Every block, not only the first: a
// block placed later, or a session that stepped apart from the others, can
// read a half its neighbours do not. ok is false for a range too long to key.
// Callers hold g.mu.
func (g *devTier) rangeParity(sid uint64, lo, hi int) (m parityMask, ok bool) {
	bit := 0
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil || !l.linear {
			continue
		}
		if bit == 64*len(m) {
			return m, false
		}
		if par, has := g.stepPar(sid, l); has && par == 1 {
			m[bit/64] |= 1 << (bit % 64)
		}
		bit++
	}
	return m, true
}

// replayRecurrent is emitLinear's bookkeeping for a submission that replayed a
// recording instead of emitting. The flip lives in Go, not in the graph, so a
// replay must swap each linear block's halves by hand; otherwise the same
// recording is chosen every token and the recurrence freezes. A token that
// records ran emit during the capture and has already flipped.
func (g *devTier) replayRecurrent(sid uint64, lo, hi int) {
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil || !l.linear {
			continue
		}
		if _, has := g.stepPar(sid, l); has {
			g.stepFlip(sid, l)
			g.recSteps++
			if g.rag != nil && g.rag.sid != nil {
				g.SessionLinear++
			}
		}
	}
}

// hasRecurrent reports whether any block in [lo, hi) keeps a recurrent state.
func (g *devTier) hasRecurrent(lo, hi int) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	for li := lo; li < hi; li++ {
		if l := g.layers[li]; l != nil && l.rec != nil {
			return true
		}
	}
	return false
}

// graphKey is everything about a token that decides which launch sequence it
// issues. Two tokens with the same key issue byte-identical submissions, so one
// captured graph serves both.
//
// It is short because the sequence barely varies: every buffer a decode launch
// names is allocated once and never moves, and every per-token scalar (the
// position, the page descriptors, the rotary rows) reaches the device inside a
// buffer.
type graphKey struct {
	lo, hi int
	scoreN int // the rounded position count the call's forms were asked at
	rows   int // the batch width; 1 for decode
	head   bool
	// argmax says the head ends in the device argmax (nn.Head.ArgmaxOnly):
	// one more launch, so a recording without it cannot serve a call with it.
	argmax bool
	// sampleK is the composed walk's k (nn.Head.SampleK), 0 for none: its
	// launches follow the head, so a recording without them cannot serve a
	// call with them, nor one of another k.
	sampleK int
	// pick says the head ends in the label pick (nn.Head.Pick).
	pick bool
	// rag is a LayersRows step's sequence count, 0 for any other call: ragged
	// attention and a per-row head, which a recording names by address.
	rag int
	// ragRuns is how many runs a hybrid's ragged step groups its rows into.
	ragRuns int
	// The recurrent parity: a linear block's state deliberately moves between
	// two halves, so a recording names one as source and the other as
	// destination. Replayed at the other parity it would read the stale half
	// and the state would never advance. Two recordings alternate instead,
	// as a partial seam alternates two lo/hi keys. One bit per linear block
	// (rangeParity).
	recParity parityMask
	// keys is the paged history's key count, rounded to the recording grain
	// (pagedPrep): a form over passes launches by it.
	keys int
	// ragHead is how many of a ragged step's rows the head projects
	// (nn.Head.LogitRows), and a step of the same rows wanting fewer logits
	// differs only here.
	ragHead int
	// hnorm is which norm the head reads (headNormFor): a recording names its
	// buffer by address.
	hnorm int
	// persist says the recording holds composed persistent kernels
	// (Config.Persist); persistDeps that they are ordered by dependency
	// counters (Config.PersistDeps) rather than grid barriers.
	persist, persistDeps bool
	// kvF16 is the binary16 history (Config.KVF16), which the forms read.
	kvF16 bool
	// sid is the session the recording was made for: it bakes that session's
	// page descriptors and recurrent seats, so another session never replays
	// it. Keying on it, rather than dropping every recording when the current
	// session changes, is what lets two sessions that take turns -- a
	// speculator's trunk and draft, two requests stepped one after another --
	// each keep replaying instead of recording again at every turn.
	sid uint64
	// headSrv, tableSrv, sampleSrv and embedSrv say the composer formed the
	// head, the step's tables, its token and its input rows; pickHeadSrv and
	// pickSrv that the composer's head projects the picked rows alone, or its
	// gather picks.
	headSrv, tableSrv, gridSrv, sampleSrv, embedSrv bool
	pickHeadSrv, pickSrv                            bool
}

// canRecord reports whether this device's sessions can capture a launch
// sequence, cached because it costs an ownership hop to ask.
func (g *devTier) canRecord() bool {
	if g.recCap == 0 {
		g.recCap = -1
		g.dev.Session(func(s backend.Session) {
			if _, ok := s.(backend.Recorder); ok {
				g.recCap = 1
			}
		})
	}
	return g.recCap == 1
}

// scoreGrain rounds the score grid's position count up to a multiple of itself.
//
// The rounding keeps the sequence stable for a run of tokens, re-capturing when
// it rolls over, as llama.cpp pads its KV length. It costs a little arithmetic
// and no correctness: AttnScores clamps its thread index with n read from a
// buffer, so surplus groups recompute and store the same last score. It rounds
// on both arms of the graph A/B so they do the same device work.
const scoreGrain = 128

func (g *devTier) layersOnce(sid uint64, bs *blockScratch, lo, hi, pos int, x, cs, csSWA []float32, head *nn.Head) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	// Every wait comes first. The prologue below reads what other sessions
	// change -- this session's pages, whether any are in host memory, the blocks --
	// into the submission's descriptors, and a wait after it would let one of
	// them change under the submission (another session's page-in or
	// eviction). Alone on the device: wait for every submission in flight.
	// Beside the others: wait for a quiesce in progress to have its turn. From
	// here to the submission nothing lets g.mu go (prologueHeld).
	// The question is asked again after every wait: another session's
	// prologue may have sent this session's history to host memory meanwhile.
	var excl bool
	for {
		if excl = g.exclusiveRange(lo, hi, g.rag, bs) || g.evictedFor(sid, g.rag); excl {
			// Running alone is right whatever changes while it waits.
			g.quiesce()
			break
		}
		if g.quiet == 0 && g.alone == 0 {
			break
		}
		g.idle.Wait()
	}
	g.prologueHeld = true
	defer func() { g.prologueHeld = false }()
	// An empty range is legal only with a head: the projection under a partial
	// seam. Every refusal records why, since the instance's answer to false is a
	// silent restart on the host.
	refuse := func(f string, a ...any) bool {
		g.LastErr = fmt.Sprintf(f, a...)
		return false
	}
	switch {
	case bs == nil:
		return refuse("no scratch for this submission")
	case lo > hi || (lo == hi && head == nil):
		return refuse("an empty block range [%d,%d) with no head", lo, hi)
	case pos >= bs.p.MaxSeq:
		return refuse("position %d is past this device's capacity of %d (asked for %d)",
			pos, bs.p.MaxSeq, g.maxSeqAsked)
	case head != nil && lo < hi && bs.p.Streams() > 1:
		// AltUp's head reads the streams' mean and DeepSeek V4's their
		// collapse, a part of its own: the blocks and the head go as two
		// calls.
		return refuse("a head folded into %d blocks of %d streams", hi-lo, bs.p.Streams())
	}
	// The head reads one residual vector and writes one logit row. A ragged
	// step's head is the decode scratch's; the batched one has none.
	rag := g.rag
	hb := bs
	// A chunk projecting every row (nn.Head.RowLogits) runs the ragged step's
	// per-row head, which reads the resident head through the decode scratch.
	if rag != nil || bs.rows > 1 && head != nil && head.RowLogits {
		hb = g.bs
	}
	// A batched chunk may carry the head, which then runs on the last row:
	// prefill wants logits for its final position only. The row is copied into
	// the decode scratch on the card, its norm, quantize and projection run
	// there, and only the logits are read back -- no second session, no readback of
	// the chunk.
	// A chunk whose head wants every row's logits (nn.Head.RowLogits: a
	// speculative verification) runs the ragged step's per-row head over the
	// chunk instead of folding its last row; prepRowsHead built it.
	allRows := bs.rows > 1 && head != nil && rag == nil && head.RowLogits
	fold := bs.rows > 1 && head != nil && rag == nil && !allRows
	if allRows && (!bs.ragBuilt || bs.ragLogits == nil) {
		return refuse("a %d-row chunk asked for every row's logits with no per-row head built", bs.rows)
	}
	// The per-row head caps where decode's does: a step's rows end in an
	// argmax and a readback of their own, and uncapped they would be a
	// different function from the session's decode.
	// The head's norm: the one it was built with, or a prediction block's own
	// (headnorm.go). A norm the device was never given is a refusal, never the
	// resident one in its place.
	var hNorm backend.Buf
	hNormID := 0
	if head != nil && g.bs != nil {
		var ok bool
		if hNorm, hNormID, ok = g.headNormFor(head); !ok {
			return refuse("the head brings a norm this device was not given (PrepHead)")
		}
	}
	if fold {
		hb = g.bs
		switch {
		case hb == nil || hb.head == nil || !hb.head.ok:
			return refuse("the head is not resident on this device")
		case hb.p.NEmbd != bs.p.NEmbd:
			return refuse("a batched submission of %d rows %d wide with a head %d wide", bs.rows, bs.p.NEmbd, hb.p.NEmbd)
		}
		if e := g.foldReqBuf(hb); e != nil {
			return refuse("the folded head's row take: %v", e)
		}
	}
	if head != nil && (hb == nil || hb.head == nil || !hb.head.ok) {
		return refuse("the head is not resident on this device")
	}
	if rag != nil && len(rag.pos) != len(x)/bs.p.ResidW() {
		return refuse("%d ragged rows for %d rows of x", len(rag.pos), len(x)/bs.p.ResidW())
	}
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil || !l.ok {
			return refuse("block %d is not on this device", li)
		}
		// A paged-out block's weight buffers are freed, and launching against one
		// reads whatever now lives there without faulting. submit() pages the
		// range in first; this asserts it, and is why nothing else may call
		// layersOnce.
		if l.pg != nil && !l.pg.in {
			g.LastErr = fmt.Sprintf("block %d is paged out", li)
			return false
		}
		// A ragged step's rows are different sequences, or different positions
		// of one, each reading its own history through its descriptor: only the
		// paged latent cache has one (the contiguous cache is one sequence's).
		if l.mla && rag != nil && bs.pkv == nil {
			return refuse("block %d: a ragged step over a latent block needs the paged cache", li)
		}
	}
	g.Submits++
	p := &bs.p
	// The cache row; see initScratch.

	R := bs.rows
	// The caller's row count comes from the slice and the grid does not: one
	// scratch serves one width, so a ragged chunk runs the full width with the
	// surplus rows zeroed (harmless; see Layers).
	nrow := len(x) / p.ResidW()
	if g.roomFaulted("noroom-lie") && nrow > 1 {
		return refuse("a %d-row step's scratch passes the budget (roomFault)", nrow)
	}
	if (g.roomFaulted("narrow") || g.roomFaulted("narrow-lie")) && nrow > 64 {
		return refuse("%d rows on a device holding only 64-row scratch (roomFault)", nrow)
	}
	if nrow < 1 || nrow > R || rag == nil && pos+nrow > p.MaxSeq {
		return refuse("%d rows at position %d on a scratch of %d rows and capacity %d",
			nrow, pos, R, p.MaxSeq)
	}
	if rag != nil {
		for r := range nrow {
			if rag.pos[r] >= p.MaxSeq {
				return refuse("row %d at position %d past a capacity of %d", r, rag.pos[r], p.MaxSeq)
			}
		}
	}
	hx := bs.hx[:R*p.ResidW()]
	// A promised embedding is gathered into the scratch, not uploaded (see
	// EmbedRows); the gather writes all R rows of bs.x, padding as zeros. Its
	// last workgroup must not run past R rows; otherwise the promise is made real
	// on the host.
	pe := g.takeEmbedLocked(x)
	embedSrv := false
	// toks is the rows' tokens where the card does not gather the rows
	// itself: a block reading them (a mixture routed by them) still does.
	var toks []byte
	if pe != nil {
		// The composer's gather (nn.ComposedEmbed) at the call's rows; a call
		// it cannot write them for (a ragged step's rows, each its own
		// sequence's) takes them made real by the same form.
		if len(pe.ids) == nrow && nrow <= R && rag == nil && g.bs.embedServed() && g.embedBufs(g.bs) &&
			g.bs.embW.k == p.NEmbd {
			embedSrv = true
		} else {
			if g.bs.embTok != nil {
				toks, _ = g.embedWords(pe, R)
			}
			g.materializeLocked(pe)
			pe = nil
		}
	}
	if toks == nil && !embedSrv && g.bs.embTok != nil && len(g.stepToks) > 0 {
		toks = g.tokWords(g.stepToks, min(R, batchWidth))
	}
	if pe == nil {
		for i := range hx {
			hx[i] = 0
		}
		// AltUp's streams are stream-major over the caller's nrow rows and
		// over the scratch's R (altup.go); one stream is a prefix either way.
		streamRows(hx, x, p, R, nrow)
	}
	// pN carries a header at R > 1: element 0 is the uniform score width every
	// grid is sized for, elements 1..R each row's own causal key count. At R == 1
	// it is the single count.
	n := uint32(pos + R)
	if p.NonCausal && g.ncPhase == 2 {
		// A row-chunked block's second phase reads every key the first
		// phase wrote, whatever chunk its queries are.
		n = uint32(g.ncKeys)
	}
	if int(n) > p.MaxSeq {
		// The score row is MaxSeq wide, so the uniform width cannot exceed it.
		// Clamping is safe: a real row needs at most pos+nrow of it.
		n = uint32(p.MaxSeq)
	}
	// The staging bytes are the devTier's, reused: they are written to the
	// device before the call returns and read by nothing after.
	pn := append(g.subBytes.pn[:0], u32one(n)...)
	if R == 1 {
		n = uint32(pos + 1)
		pn = binary.LittleEndian.AppendUint32(pn[:0], n)
	}
	// Only a backend that can record pays the rounding; elsewhere it would be
	// surplus attention work for nothing.
	nCap := int(n)
	if g.canRecord() && R == 1 {
		nCap = (int(n) + scoreGrain - 1) / scoreGrain * scoreGrain
	}
	// A call with no attention block in it (the head alone, or a hybrid's
	// linear blocks, which scan their rows in order) cuts no run: a run is
	// a key count, and only attention reads one.
	if len(g.bidir) > 0 && !p.NonCausal && g.attendsIn(lo, hi) {
		if why := bidirCheck(g.bidir, pos, nrow); why != "" {
			return refuse("rows [%d,%d): %s", pos, pos+nrow, why)
		}
	}
	if R > 1 {
		// The per-row counts go in after the cap, which is what the rounding
		// above would otherwise invalidate -- so a batched call does not round.
		pn = pn[:4]
		binary.LittleEndian.PutUint32(pn, uint32(nCap))
		for r := 0; r < R; r++ {
			// A padded row takes the last real row's count. It must be clamped,
			// because AttnAcc walks each row's own count and an unclamped one runs
			// past the cache (a fault on one model, a neighbour's buffer on
			// another). It must not be smaller than its neighbours', because
			// AttnAccTiled walks the count of its tile's last row, so a short one
			// truncates every real row in the tile.
			n := pos + r + 1
			if r >= nrow {
				n = pos + nrow
			}
			if p.NonCausal {
				// The whole call is one run: the mask is a per-row key count, so
				// every row gets the same count and needs no second score kernel. A
				// padded row still takes nrow, for the tile reason above.
				n = nrow
				if g.ncPhase == 2 {
					n = g.ncKeys
				}
			} else if r < nrow {
				// A row of a bidirectional run counts keys to the run's end
				// (bidir.go); bidirCheck has kept the run inside these rows.
				if e := bidirCount(g.bidir, pos+r); e > 0 && !p.BidirOff {
					n = e
				}
			}
			pn = append(pn, u32one(uint32(n))...)
		}
	}

	// A ragged step's rows are different sequences (LayersRows), each at its
	// own position with its own pages (pagedRows); what is staged here is the
	// longest history, which sizes the recording's key.
	if rag != nil {
		pn = pn[:4]
		mc := 1
		for r := 0; r < R; r++ {
			cnt := 1
			if r < nrow {
				if rag.slot[r]-rag.pos[r] < 0 {
					return refuse("row %d: slot %d at position %d", r, rag.slot[r], rag.pos[r])
				}
				cnt = rag.pos[r] + 1
			}
			mc = max(mc, cnt)
			pn = append(pn, u32one(uint32(cnt))...)
		}
		// Rounded as decode's is, so a recording serves scoreGrain steps.
		nCap = mc
		if g.canRecord() {
			nCap = min((mc+scoreGrain-1)/scoreGrain*scoreGrain, p.MaxSeq)
		}
		binary.LittleEndian.PutUint32(pn, uint32(nCap))
	}

	// Paged history: every row's descriptor and the tables, staged with the
	// other per-call writes. The pages were taken before the submission
	// (pagedAppend); here nothing may allocate one, since growing the pool can
	// page weights out from under the range about to run.
	if pk := bs.pkv; pk != nil {
		pk.rows = g.pagedRows(sid, pk.rows, R, nrow, pos, rag)
		if e := g.pagedPrep(sid, bs, pk.rows); e != nil {
			return refuse("paged attention: %v", e)
		}
		pk.st = nil
		if pk.held {
			g.pagedStage(pk, p, pk.rows)
			// A call over evicted history streams (pagedstream.go).
			if e := g.pagedStreamPrep(sid, bs, pk.rows); e != nil {
				g.noteRoom(e)
				return refuse("paged attention: %v", e)
			}
		}
	}

	// A scratch with a local rotary table (two bases, gemma3) needs it for
	// every row the global one has: short, the local layers would rotate by
	// whatever the buffer last held -- a table of other positions -- and stay
	// fluent.
	if bs.csSWA != nil && len(csSWA) < len(cs) {
		return refuse("the local layers' rotary table has %d floats against the global one's %d",
			len(csSWA), len(cs))
	}
	hcs := cs
	if R > 1 && len(bs.hcs) < R*p.RopeW() {
		return refuse("the scratch stages %d rotary floats for %d rows of %d (its plan's %d rows of %d, rotary %d split %v; this plan's rotary %d split %v)",
			len(bs.hcs), R, p.RopeW(), bs.rows, bs.p.RopeW(), bs.p.NRot, bs.p.RopeSplit, p.NRot, p.RopeSplit)
	}
	if R > 1 {
		hcs = bs.hcs[:R*p.RopeW()]
		for i := range hcs {
			hcs[i] = 0
		}
		copy(hcs, cs)
	}
	// The local table gets the identical treatment, and its own staging buffer:
	// sharing hcs would have the second copy overwrite the first.
	hcsSWA := csSWA
	if R > 1 && csSWA != nil {
		hcsSWA = bs.hcsSWA[:R*p.RopeW()]
		for i := range hcsSWA {
			hcsSWA[i] = 0
		}
		copy(hcsSWA, csSWA)
	}

	// rrows is how many rows the caller filled. The rotary kernel is clamped to
	// it, so the padded rows of a ragged chunk (a short prompt is all ragged)
	// keep the zeros initRopeTable wrote rather than a real rotation. The
	// position bound is checked here, where the kernel is launched. NRot is zero
	// on a vision block, so the division is guarded.
	rrows := 0
	if p.NRot > 0 {
		rrows = R
		if v := len(cs) / p.RopeW(); v < rrows {
			rrows = v
		}
		if bs.csSWA != nil {
			if v := len(csSWA) / p.RopeW(); v < rrows {
				rrows = v
			}
		}
	}
	top := pos + rrows
	if rag != nil {
		top = 0
		for _, v := range rag.pos {
			top = max(top, v+1)
		}
	}
	// The call's tables the composer serves (nn.ComposedTable) at its rows,
	// which write cs and csSWA where the tier's own table would. A call of
	// the head alone (a model whose host collapses its streams between the
	// blocks and the head) turns nothing and is not asked for them, nor is a
	// non-causal segment's that turns by its picture's grid (servedGridTable).
	gridSrv := p.NonCausal && lo < hi && g.gridServes(bs)
	tableSrv := false
	var raxes []byte
	if bs.cs != nil && p.NRot > 0 && !gridSrv && lo < hi && rrows >= 1 {
		if err := g.tableServes(bs, p, pos, rrows, rag); err != nil {
			return refuse("the rotary tables: %v", err)
		}
		tableSrv = true
	}
	if gridSrv {
		if _, err := g.servedGridTable(bs, R); err != nil {
			g.Steps.Decline(nn.PartTable, partNotFormed)
			return refuse("the composed grid table: %v", err)
		}
	}
	switch {
	case tableSrv:
		// Each row's cache position and its sequence's marks: the tables
		// turn it by its positions on the axes they compute from them.
		raxes = g.ropeRowsWords(sid, pos, rrows, R, rag)
	case gridSrv:
	case bs.cs != nil:
		g.RopeTableUploads += R
	}

	// Retire recordings before the session opens: freeing one is a driver call
	// that cannot nest inside a session. See dropGraph.
	//
	// Retire on nCap, keep on lo/hi. nCap only increases, so a key differing
	// only there is dead once a longer one appears; lo/hi/head is a placement,
	// and a partial seam alternates two of them every token.
	//
	// The persistent path: decode only, and not over history streamed from
	// back, whose uploads sit between launches.
	persistOn := R == 1 && rag == nil && !(bs.pkv != nil && bs.pkv.st != nil) && g.ensurePersist()
	// The head the composer serves in this call, and so what the call leaves
	// to the caller after the logits (nn.Head.Applied): a call that serves the
	// head's graph applies all of its stages, any other the final softcap
	// alone. The token, the candidates and the picked labels are read before
	// the caller finishes the logits, so a request for them is declined while
	// a stage that bears on them is the caller's.
	var applied, left nn.HeadStage
	headWhy := partServes
	// Every head is the head scratch's graph at the rows it projects: a
	// decode step's or a prompt's one row, a chunk's folded last row, a
	// ragged step's wanted rows or every row of a chunk. A prompt's call,
	// its folded head too, asks only where the prompt is asked
	// (nn.ComposedPrompt).
	composedHead := head != nil
	if composedHead {
		// The tier holds the head its model placed (headGraphs), every
		// instance of the model offering the same graphs.
		if head.Graph != hb.headGraph || head.HiddenGraph != hb.headHidden {
			return refuse("the head offered is not the one its model placed on the tier")
		}
		if err := hb.headServes(hNorm); err != nil {
			return refuse("the composed head: %v", err)
		}
		applied |= head.Stages
	}
	if head != nil {
		left = head.Stages &^ applied
	}
	argmax := head != nil && head.ArgmaxOnly && g.argmaxOut != nil && R == 1 && left&nn.HeadBias == 0 &&
		rag == nil && !g.prompt && !fold && bs.sampleGraph != nil && head.Sample == bs.sampleGraph
	// A top-k draw's token (nn.Head.SampleK), the composer's walk written
	// where the argmax's is, where the front end states it.
	sampleK := 0
	walk := head != nil && !g.prompt && g.argmaxOut != nil && bs.sampleWalk != nil && head.SampleWalk == bs.sampleWalk
	if head != nil && !argmax && head.SampleK > 0 && R == 1 && rag == nil && !allRows && !fold &&
		left&(nn.HeadBias|nn.HeadScale) == 0 && walk && g.prepSample(head) {
		sampleK = head.SampleK
	}
	// The label pick: one row's logits, a decode's or a folded chunk's last.
	pickWant := head != nil && !argmax && sampleK == 0 && len(head.Pick) > 0 && rag == nil && !allRows &&
		(R == 1 || fold) && left&(nn.HeadBias|nn.HeadScale) == 0
	// The composer's pick, asked where the step's token is and a prompt's
	// where the prompt is: the head projecting the picked rows alone, where
	// it serves the head, no stage is left after the projection and the
	// picked rows' form is the head's own arithmetic at them, else the gather
	// over the logits. The projection's bias is the pick head's own.
	pickAsked := pickWant
	pickHeadSrv := pickAsked && composedHead && headWhy == partServes && hb.pickHead != nil && head.PickHead == hb.pickHead &&
		head.Stages&(nn.HeadScale|nn.HeadCap) == 0 && hb.pickRowsExact()
	pickSrv := pickAsked && !pickHeadSrv && hb.pickGraph != nil && head.PickGraph == hb.pickGraph
	pick := pickWant && g.prepPick(head, pickHeadSrv || pickSrv)
	pickHeadSrv, pickSrv = pickHeadSrv && pick, pickSrv && pick
	// The token the composer selects (nn.ComposedSample): a one-row step's
	// greedy token, written where the tier's argmax writes it, the sampler's
	// candidates, or the label pick.
	sampleSrv := false
	ragSample := head != nil && (rag != nil || allRows)
	if head != nil && (argmax || sampleK > 0 || pick || ragSample) {
		var serr string
		switch {
		case ragSample && (hb.sampleGraph == nil || head.Sample != hb.sampleGraph):
			serr = "a ragged step's sample graph is not the one placed"
		case ragSample:
			sampleSrv = true
		case pick && pickAsked && (pickSrv || pickHeadSrv):
			sampleSrv = true
		case pick && pickAsked:
			serr = "the label pick has neither its head over the picked rows nor its pick over the logits placed"
		case sampleK > 0:
			sampleSrv = true
		case bs.sampleGraph == nil || head.Sample != bs.sampleGraph:
			serr = "the step's sample graph is not the one placed"
		default:
			sampleSrv = true
		}
		if serr != "" {
			return refuse("the composer does not serve the step's token: %s", serr)
		}
	}
	// A pick that projects its labels' rows alone leaves no logits.
	pickOnly := pick && !fold && pickHeadSrv
	headSrv := composedHead && headWhy == partServes
	// The composer is the only path: a head it does not serve is refused.
	if head != nil && !headSrv {
		reason := "the composer was not asked for it"
		if composedHead {
			reason = DeclineName(int(headWhy))
		}
		return refuse("the composer does not serve the head: %s", reason)
	}
	// A head alone over every stream collapses them into the first before it
	// (collapse.go): the caller handed them uncollapsed.
	collapse := head != nil && lo == hi && hb.collapse != nil && p.Streams() > 1
	ragN, ragRuns, ragHead := 0, 0, 0
	if rag != nil {
		ragN, ragRuns = len(rag.pos), len(rag.runs)
		if head != nil {
			ragHead = head.WantedRows(ragN)
		}
	}
	// Counted here rather than in emit, which a replayed recording skips. The
	// composer's head runs the wanted rows' count (servedHead's rows.n), and
	// ragHead is in the recording's key.
	if ragHead == 1 && headSrv {
		g.RagHeadOne++
	}
	par, parOK := g.rangeParity(sid, lo, hi)
	if allRows {
		ragHead = head.WantedRows(nrow)
	}
	keys := 0
	if bs.pkv != nil {
		keys = bs.pkv.keys
	}
	key := graphKey{lo: lo, hi: hi, scoreN: nCap, rows: R, head: head != nil, argmax: argmax, sampleK: sampleK, pick: pick,
		rag: ragN, ragRuns: ragRuns, recParity: par, keys: keys, ragHead: ragHead, hnorm: hNormID,
		persist: persistOn, persistDeps: persistOn && g.PersistDeps, kvF16: g.KVF16, sid: sid,
		headSrv: headSrv, tableSrv: tableSrv, gridSrv: gridSrv, sampleSrv: sampleSrv, embedSrv: embedSrv, pickHeadSrv: pickHeadSrv, pickSrv: pickSrv}
	if lo == hi {
		// The head alone reads no history, so the capacity is not part of
		// what it records: keyed by it, prompts on either side of a growth
		// would retire each other's recording on every call.
		key.scoreN = 0
	}
	if _, live := g.recs[key]; !live {
		for k, r := range g.recs {
			// The same call at another position grain: its score count and
			// its history's key count are the positions'.
			older := k
			older.scoreN, older.keys = key.scoreN, key.keys
			if older != key {
				continue // a different placement, still live
			}
			g.stale = append(g.stale, r)
			delete(g.recs, k)
		}
	}
	// A backstop, not the policy above, and per session: several sessions
	// taking turns each hold their own few keys, and a session over the bound
	// loses its least recently used.
	if bound := g.graphBound(); len(g.recs) > bound {
		var mine []graphKey
		for k := range g.recs {
			if k.sid == sid {
				mine = append(mine, k)
			}
		}
		if len(mine) > bound {
			slices.SortFunc(mine, func(a, b graphKey) int { return cmp.Compare(g.recUsed[a], g.recUsed[b]) })
			for _, k := range mine[:len(mine)-bound] {
				g.stale = append(g.stale, g.recs[k])
				delete(g.recs, k)
			}
		}
		for k := range g.recUsed {
			if _, live := g.recs[k]; !live {
				delete(g.recUsed, k)
			}
		}
	}
	g.freeStale()

	g.subBytes.pn = pn
	t0 := time.Now()
	g.sub = submitArgs{
		bs: bs, hb: hb, head: head, lo: lo, hi: hi, pos: pos, x: x,
		hx: hx, hcs: hcs, hcsSWA: hcsSWA, pn: pn,
		p: p, rag: rag, pe: pe, key: key, R: R, nrow: nrow, toks: toks,
		nCap: nCap, rrows: rrows,
		tableSrv: tableSrv, gridSrv: gridSrv, sampleSrv: sampleSrv, embedSrv: embedSrv, raxes: raxes,
		fold: fold, argmax: argmax, sampleK: sampleK, pick: pick, ragHead: ragHead, parOK: parOK,
		allRows: allRows, hNorm: hNorm, headWhy: headWhy, pickOnly: pickOnly,
		pickHeadSrv: pickHeadSrv, pickSrv: pickSrv,
		collapse: collapse,
	}
	if g.subFn == nil {
		g.subFn = g.layersSession
	}
	// Read here, under g.mu: a page-in another session makes changes it.
	g.sub.paging = g.pagesIn < len(g.layers)
	g.sub.persist = persistOn
	g.sessQueue()
	// The prologue streams this call's history only when some of it was at
	// in host memory before the prologue began (evictedFor), which made the call one
	// to run alone; nothing can have gone to host memory since, so this is a check of
	// that reasoning and not a path.
	if !excl && g.exclusiveRange(lo, hi, rag, bs) {
		return refuse("a submission that must run alone was prepared to run beside others")
	}
	g.prologueHeld = false
	if excl && g.quiet == 0 && g.rowsAlone(lo, hi, rag, bs, sid) {
		// Alone, but with g.mu let go for the run: a step over several
		// sessions' rows writes the device's batched scratch and nothing else
		// of the device's, so no other submission starts (alone) while every
		// other session's bookkeeping goes on -- the next step's room and
		// inputs on this device, which would otherwise wait out this stage
		// and keep a second card's stage from starting beside it. What
		// changes what a submission reads quiesces, which waits for this one
		// as for any in flight (its ticket).
		g.alone++
		t := g.beginSub()
		g.inSub, g.Stats = true, &g.acc
		g.runUnlocked(g.subFn)
		g.inSub, g.Stats = false, &g.tot
		g.tot.add(g.acc)
		g.acc = Stats{}
		g.endSub(t)
		g.alone--
		g.idle.Broadcast()
		g.freeStale()
	} else if excl {
		// Alone on the device, as every submission was before sessions ran
		// at once: nothing else is in flight (the wait was before the
		// prologue), and nothing starts while g.mu is held across it.
		g.runSub(g.subFn)
	} else {
		// Beside any other session's: g.mu is let go for the submission, so
		// the bookkeeping of the others goes on meanwhile. What the
		// submission builds on the way takes g.mu for the build (subLock),
		// and its counters are its own until it completes.
		t := g.beginSub()
		g.inSub, g.Stats = true, &g.acc
		g.runUnlocked(g.subFn)
		g.inSub, g.Stats = false, &g.tot
		g.tot.add(g.acc)
		g.acc = Stats{}
		g.endSub(t)
		g.freeStale()
	}
	err, direct := g.sub.err, g.sub.direct
	g.sub = submitArgs{}
	if err != nil {
		g.LastErr = err.Error()
		if nn.IsBuildError(err) {
			g.noteCause(nn.FailBuild)
		} else {
			g.noteCause(nn.FailFault)
		}
		return false
	}
	if head != nil {
		head.Applied = applied
	}
	if head != nil && !direct {
		for i := range head.Logits {
			head.Logits[i] = hb.hOut[i]
		}
	} else if head == nil && !direct {
		streamRows(x, bs.hout, p, nrow, R)
	}
	g.TLayer += time.Since(t0)
	g.Blocks += hi - lo
	// The submission has completed (its readback waited for it), so pages
	// released before it are no longer read by anything in flight, and the
	// rows it wrote are history eviction may send to host memory.
	if pk := bs.pkv; pk != nil && pk.held && g.kvp != nil {
		g.kvp.wrote(pk.rows)
	}
	if g.kvp != nil {
		for _, l := range g.kvp.layers {
			l.fence()
		}
	}
	return true
}

// submitArgs is one layersOnce submission's state, handed to layersSession
// through the session's devSess rather than captured by a closure: the
// function a Session runs escapes, so a closure over the submission's locals
// was a heap object -- with every variable it shared with its own closures --
// on every token. A session runs one submission at a time on a device, so one
// set a session serves.
type submitArgs struct {
	// paging says a block of the device is paged out, read before the
	// submission: a recording names device addresses a page-in allocates
	// afresh.
	paging bool
	// persist says the submission composes its decode blocks (persist.go).
	persist            bool
	bs, hb             *blockScratch
	head               *nn.Head
	lo, hi, pos        int
	x, hx, hcs, hcsSWA []float32
	pn                 []byte
	p                  *nn.LayerPlan
	rag                *ragStep
	pe                 *embPend
	key                graphKey
	R, nrow            int
	nCap               int
	rrows              int
	ragHead            int
	fold, argmax       bool
	// tableSrv and sampleSrv say the composer serves the step's tables and
	// its token; raxes is a table over several axes' staged positions;
	// gridSrv says the set runs its grid table from the session's cells.
	tableSrv, sampleSrv, gridSrv bool
	// embedSrv says the composer gathers the decode row; toks is the rows'
	// tokens where it does not.
	embedSrv       bool
	toks           []byte
	raxes          []byte
	sampleK        int
	pick           bool
	parOK, allRows bool
	hNorm          backend.Buf
	// headWhy is why the composer is not asked for the head at this one-row
	// step, partServes where it is; pickOnly says the head projects the
	// picked labels alone.
	headWhy  partDecline
	pickOnly bool
	// collapse says the head reads the collapse of every stream (collapse.go).
	collapse bool
	// pickHeadSrv and pickSrv say the composer picks the labels: its head
	// over the picked rows alone, or its gather over the logits.
	pickHeadSrv, pickSrv bool
	// What the session hands back.
	err    error
	direct bool
}

// layersSession is the device side of layersOnce: the staging writes, the
// launches (or a graph replay) and the readback, in one Session. Its arguments
// are g.sub's; see submitArgs.
func (g *devTier) layersSession(s backend.Session) {
	// The persistent path (persist.go) runs the submission through psess,
	// which issues what the launcher holds before any other call on the
	// session. Entered again rather than s reassigned below: a reassigned
	// parameter that closures capture moves to the heap, five objects a token.
	if g.sub.persist && g.sub.R == 1 && g.sub.rag == nil && g.launchTo.ps != nil &&
		g.launchTo.ps.sync != nil && g.launchTo.psess.Session == nil {
		g.launchTo.psess = persistSession{Session: s, t: &g.launchTo, g: g}
		g.layersSession(&g.launchTo.psess)
		g.launchTo.psess = persistSession{}
		return
	}
	g.injectedPanic()
	a := &g.sub
	// The session the submission runs for, carried in its recording's key.
	sid := a.key.sid
	bs, hb, head, lo, hi, pos, x := a.bs, a.hb, a.head, a.lo, a.hi, a.pos, a.x
	hx, hcs, hcsSWA, pn := a.hx, a.hcs, a.hcsSWA, a.pn
	p, rag, pe, key := a.p, a.rag, a.pe, a.key
	R, nrow := a.R, a.nrow
	fold, argmax, sampleK, pick := a.fold, a.argmax, a.sampleK, a.pick
	tableSrv, sampleSrv, embedSrv, raxes := a.tableSrv, a.sampleSrv, a.embedSrv, a.raxes
	gridSrv := a.gridSrv
	ragHead, parOK, allRows, hNorm := a.ragHead, a.parOK, a.allRows, a.hNorm
	pickOnly := a.pickOnly
	pickHeadSrv := a.pickHeadSrv
	collapse := a.collapse
	// A call whose last block is a first half reads the cut back, not x.
	cutW := 0
	if hi > lo && hi <= len(g.layers) && g.layers[hi-1] != nil {
		cutW = g.layers[hi-1].cut
	}
	var err error
	direct := false
	defer func() { a.err, a.direct = err, direct }()
	w := func(b backend.Buf, p []byte) {
		if err == nil {
			t := time.Now()
			err = s.Write(b, p)
			g.TSubStage += time.Since(t)
		}
	}
	// lc is the submission's launch path; its la is the ordinary launch
	// (see launcher).
	g.launchTo.s = s
	if ps, ok := s.(*persistSession); ok {
		g.launchTo.s = ps.Session
	}
	persist := g.launchTo.psess.Session != nil
	defer func() { g.launchTo.s = nil }()
	lc := &launcher{to: &g.launchTo, err: &err, g: g}

	// Scanned rather than counted on the tier, so no release path can let a
	// counter drift.
	streamed := false
	for li := lo; li < hi && !streamed; li++ {
		// A batched mixture reads its selection back too.
		if l := g.layers[li]; l != nil && (l.stream != nil || (R > 1 && l.router != nil)) {
			streamed = true
		}
	}
	// emit issues the whole token's launches. It is a closure so the same code
	// both runs the sequence and records it: one source for both means a graph
	// cannot differ from the launches it replaces.
	emit := func() {
		// The rotary table first: every block reads it and nothing writes it again.
		// Both launches are pure functions of buffers, so a replay rebuilds it from
		// whatever bs.rpos holds this token.
		switch {
		case gridSrv:
			sv, serr := g.servedGridTable(bs, R)
			if serr == nil {
				serr = sv.run(g, lc, sv.inst, nil, nil)
			}
			if err == nil {
				err = serr
			}
		case tableSrv:
			for i := range bs.tableCount() {
				cs, planes, at := g.tableBinds(bs, i)
				sv, serr := g.servedTable(scratchTable(bs, p, i), R, p.RopeMarks, cs, planes, at)
				if serr == nil {
					serr = sv.run(g, lc, sv.inst, nil, nil)
				}
				if err == nil {
					err = serr
				}
			}
		}
		for li := lo; li < hi; li++ {
			l := g.layers[li]
			// The previous block's held launches go out; this block's are
			// held when it is one the path takes.
			lc.persistBlock(persist && !streamed && persistable(l, bs))
			// The block the composer serves whole: its graph from the model's
			// front end, formed as one unit and launched on the block's own
			// weights and history -- a decode step's, or a prompt's at its
			// rows (nn.ComposedPrompt): a chunk, or a one-row call of the prompt
			// (nn.PromptDevice), never counted as a step.
			// A ragged step's rows are a decode step of several sessions,
			// each row at its own position and history (its descriptor):
			// its blocks are asked as decode's, at the step's rows, and a
			// prompt's ragged rows as the prompt's.
			kind := g.rowsKind(R, rag != nil)
			// A non-causal call is a picture's rows, one run: its blocks are
			// asked whole, at the call's rows.
			if l.nonCausal {
				kind = nn.PartTower
			}
			if berr := g.blockServes(l, li, p, bs, kind == nn.PartPrompt); berr != nil {
				if err == nil {
					err = fmt.Errorf("tier: the composer does not serve block %d at %d rows: %w", li, R, berr)
				}
				continue
			}
			why := partServes
			if R > 1 && l.stream != nil && nrow > l.pagedRows() {
				// pagedCall holds every composed call to the rows a streamed
				// block's pages hold.
				if err == nil {
					err = fmt.Errorf("block %d: a call of %d rows reached a streamed block whose pages hold %d: it was not cut (pagedCall)",
						li, nrow, l.pagedRows())
				}
				continue
			}
			var sv, cp *servedPart
			var inst *partInst
			if why == partServes {
				var serr error
				g.keeping = l.linear && g.keepStep(rag, l)
				sv, inst, cp, serr = g.servedStep(bs, l, li, p, R, rag == nil || rag.oneSequence())
				if serr != nil {
					g.keeping = false
					if err == nil {
						err = serr
					}
					continue
				}
				bound := !l.linear || g.bindRec(sv, bs, sid, l)
				g.keeping = false
				if !bound {
					if err == nil {
						err = fmt.Errorf("tier: block %d at %d rows: its recurrent state is not bound to the step's slots", li, R)
					}
					continue
				}
				if inst == nil {
					why = partNoRoom
				}
				if R > 1 && sv != nil {
					sv.rows = nrow
					if cp != nil {
						cp.rows = nrow
					}
				}
			}
			if why != partServes {
				g.Steps.Decline(kind, why)
			}
			// The composer is the only path: a block it does not serve is
			// refused, never run on the tier's own kernels.
			if inst == nil {
				if err == nil {
					err = fmt.Errorf("tier: the composer does not serve block %d at %d rows: %s", li, R, DeclineName(int(why)))
					if why == partNoRoom && g.LastErr != "" {
						err = fmt.Errorf("%w (%s)", err, g.LastErr)
					}
				}
				continue
			}
			{
				out := bs.x
				if cp != nil {
					out = bs.h
				}
				// A second input is the rows' per-layer inputs (blockServes);
				// a block of several streams reads and writes each at its
				// offset in x.
				in := []backend.Buf{bs.x, bs.pleIn}
				outs := []backend.Buf{out}
				if l.cut > 0 {
					if cerr := g.cutScratch(bs, R*l.cut*4); cerr != nil {
						if err == nil {
							err = cerr
						}
						continue
					}
					outs[0] = bs.cut
				}
				if len(l.graph.Out) > 1 {
					var verr error
					streams := bs.servedStreams
					if cp != nil {
						streams = bs.servedStreamsApart
					}
					if in, outs, verr = streams(R, len(l.graph.Out)); verr != nil {
						if err == nil {
							err = verr
						}
						continue
					}
				}
				// A failure injected at this block (jitllmfault): it and the
				// blocks after it do not run, as a kernel failing here leaves
				// them.
				if ferr := g.blockFault(li); ferr != nil {
					if err == nil {
						err = ferr
					}
					continue
				}
				var serr error
				switch {
				case sv.f.rowSplit > 0:
					serr = sv.rowsRun(g, lc, s, inst, kind, l, in, outs)
				case sv.f.split > 0:
					serr = sv.pagedRun(g, lc, s, inst, kind, l, p.NExpert, g.hostFns[sid][li].uses, in, outs)
				case len(sv.f.sweeps) > 0:
					serr = sv.streamRun(g, lc, s, inst, kind, bs, g.kvp.layers[l.kvSrc], in, outs)
				default:
					serr = sv.runAs(g, lc, inst, kind, in, outs)
				}
				if err == nil {
					err = serr
				}
				if cp != nil {
					if serr := cp.run(g, lc, cp.inst, []backend.Buf{bs.h}, []backend.Buf{bs.x}); err == nil {
						err = serr
					}
				}
				// The form wrote the other half: the session reads it next,
				// as after emitLinear.
				if l.linear && err == nil {
					g.stepFlip(sid, l)
					g.recSteps++
					if rag != nil && rag.sid != nil {
						g.SessionLinear++
					}
				}
				continue
			}
		}
		if collapse && err == nil {
			err = g.runCollapse(lc, hb.collapse, bs, R)
		}
		// The head the composer serves: the step fails if it does not
		// form, every stage its graph states applied.
		if head != nil && (rag != nil || allRows) {
			// The wanted rows (they lead) at their own count.
			// The first of them the logits hold; the rest are headChunk's,
			// after the submission.
			out := bs.ragLogits
			rows := min(ragHead, bs.ragRows)
			if g.rowsFaulted("headrow") {
				rows = 1
			}
			// The head of the logits' capacity, its rows the wanted ones: one
			// form whatever the count.
			sv, serr := g.servedHead(hb, hNorm, bs.ragRows)
			if serr == nil {
				sv.rows, sv.sl.tables[tableKey(composer.TableCall)] = rows, bs.headCalls
				serr = sv.runHead(g, lc, bs.x, out, bs.h)
			}
			if err == nil {
				err = serr
			}
			if err == nil {
				err = g.ragTokens(lc, bs, hb, min(ragHead, bs.ragRows), sampleSrv)
			}
		} else if fold {
			// The chunk's last row, into the decode scratch, and decode's
			// head on it.
			if tk, terr := g.servedRowTake(hb, p.NEmbd, R); terr != nil {
				err = terr
			} else {
				g.rows.takeIn[0], g.rows.takeOut[0] = bs.x, hb.x
				err = tk.run(g, lc, tk.inst, g.rows.takeIn[:], g.rows.takeOut[:])
			}
			out := hb.logits
			in := hb.x
			if g.rowsFaulted("foldrow") {
				in = bs.x
			}
			if pickHeadSrv {
				if err == nil {
					err = g.runPickHead(lc, hb, hNorm, in)
				}
			} else {
				sv, serr := g.servedHead(hb, hNorm, 1)
				if serr == nil {
					serr = sv.runHead(g, lc, in, out, hb.h)
				}
				if err == nil {
					err = serr
				}
				if pick && err == nil {
					err = g.runPick(lc, hb)
				}
			}
			g.HeadFolds++
		} else if head != nil {
			// The output norm and the vocabulary projection, in the same
			// submission: the residual stream is never read back.
			if pickHeadSrv {
				if err == nil {
					err = g.runPickHead(lc, bs, hNorm, bs.x)
				}
			} else {
				out := bs.logits
				sv, serr := g.servedHead(bs, hNorm, 1)
				if serr == nil {
					serr = sv.runHead(g, lc, bs.x, out, bs.h)
				}
				if err == nil {
					err = serr
				}
			}
			switch {
			case argmax && sampleSrv && err == nil:
				sv, serr := g.servedSample(bs.sampleGraph)
				if serr == nil {
					serr = sv.run(g, lc, sv.inst, []backend.Buf{bs.headOut()}, []backend.Buf{g.argmaxOut})
				}
				err = serr
			}
			switch {
			case sampleK > 0 && sampleSrv && err == nil:
				sv, serr := g.servedSample(bs.sampleWalk)
				if serr == nil {
					serr = sv.run(g, lc, sv.inst, []backend.Buf{bs.headOut(), g.sampleWindow, g.samplePenalty, g.sampleWalk}, []backend.Buf{g.argmaxOut})
				}
				err = serr
			}
			if pick && !pickOnly && err == nil {
				err = g.runPick(lc, bs)
			}
		}
		// What the last block held goes out inside emit, so a recording
		// captures it.
		lc.persistFlush()
	}
	// submit gets the sequence onto the device, by replaying a captured graph
	// where the backend has one and by issuing the launches where it does not.
	// The staging writes and the readback stay outside it: a captured memcpy
	// would bake in a Go heap address.
	submit := func() {
		if err != nil {
			return
		}
		tl := time.Now()
		defer func() { g.TSubLaunch += time.Since(tl) }()
		rec, ok := g.recorderOf(s)
		if !ok {
			g.graphOff = true
		}
		// No recording when the key would not hold still or a replay cannot be
		// right:
		// - PerLayerSubmit changes the key every block, so -ab submit with a
		//   resident head compares two changes at once unless NoGraph is set on
		//   both arms.
		// - A prefill chunk's key steps with its position, so it would be
		//   captured every chunk and replayed never. A ragged step is recorded
		//   like decode.
		// - A paging device: a graph holds device addresses, and a page-in
		//   allocates the buffers afresh.
		// - A streamed block: the mid-FFN host readback and upload have no place
		//   in a replay.
		// - History streamed from host memory: its uploads and waits are per call.
		// - A failure injected at a block (jitllmfault): it lands between
		//   blocks only when they are issued one by one.
		// - A step keeping its rows: its keep slots are its session's.
		if (R > 1 && rag == nil) || !parOK || g.NoGraph || g.graphOff || g.PerLayerSubmit || g.keepArmed(rag) ||
			a.paging || streamed || bs.pkv != nil && bs.pkv.st != nil || g.blockFaultArmed() {
			te := time.Now()
			emit()
			g.TEmit += time.Since(te)
			return
		}
		cur, have := g.recs[key]
		if !have {
			tc := time.Now()
			parts := g.Steps.Served
			g.capturing = true
			r, e := backend.Record(rec, emit)
			g.capturing = false
			g.TCapture += time.Since(tc)
			if errors.Is(err, errServedCold) {
				// A served slice not formed yet: its compile and its
				// constants' upload have no place in a capture, so this step
				// is issued unrecorded, forming it, and the next is captured.
				if r != nil {
					g.stale = append(g.stale, r)
				}
				// The capture ran no part: the issue below counts them.
				err, g.Steps.Served = nil, parts
				emit()
				return
			}
			if e != nil || err != nil {
				// A capture executes nothing, so the work has still to be
				// issued -- the ordinary way, since a capture that failed
				// once will fail again. A recording that came back around a
				// failed launch is complete as far as the driver knows and
				// wrong as far as jitllm does, so it is retired unused.
				g.graphOff, err, g.Steps.Served = true, nil, parts
				if e != nil {
					g.LastErr = "graphs disabled: " + e.Error()
				}
				if r != nil {
					g.stale = append(g.stale, r)
				}
				emit()
				return
			}
			if g.recs == nil {
				g.recs = make(map[graphKey]backend.Recording)
			}
			g.recs[key], cur = r, r
			g.Captures++
			if g.recParts == nil {
				g.recParts = map[graphKey][nn.NumParts]int{}
			}
			for i := range parts {
				parts[i] = g.Steps.Served[i] - parts[i]
			}
			g.recParts[key] = parts
		}
		if e := rec.Replay(cur); e != nil {
			g.graphOff, g.LastErr = true, "graphs disabled: "+e.Error()
			g.stale = append(g.stale, cur)
			delete(g.recs, key)
			emit()
		} else {
			if g.recUsed == nil {
				g.recUsed = map[graphKey]uint64{}
			}
			g.recTick++
			g.recUsed[key] = g.recTick
			if have {
				g.replayRecurrent(sid, lo, hi)
				for i, n := range g.recParts[key] {
					g.PartsReplayed[i] += n
				}
			}
		}
	}

	// Position state is per token, not per block: written once here rather
	// than 54 times across 18 blocks.
	if fold {
		w(hb.foldReq, g.foldWords(nrow-1))
	}
	// The rows' tokens, to the set in use and to the set the blocks run on
	// (a prompt step's batch set), whose token table they bind.
	if !embedSrv && a.toks != nil {
		w(g.bs.embTok, a.toks)
		if bs != g.bs && bs.embTok != nil {
			w(bs.embTok, a.toks)
		}
	}
	if embedSrv {
		tok, at := g.embedWords(pe, R)
		w(g.bs.embTok, tok)
		if bs != g.bs && bs.embTok != nil {
			w(bs.embTok, tok)
		}
		if g.bs.posW != nil {
			w(g.bs.embPos, at)
		}
		if pe.kind == nn.PartEmbedRows {
			g.EmbedLaunches++
		}
		sv, serr := g.servedEmbed(g.bs, R)
		var outs []backend.Buf
		if serr == nil {
			outs, serr = bs.embedOuts(g.bs.embStreams(), R)
		}
		if serr == nil && err == nil {
			serr = sv.runAs(g, lc, sv.inst, pe.kind, nil, outs)
		}
		if err == nil {
			err = serr
		}
	} else {
		w(bs.x, f32b(hx))
	}
	// Served input rows write the per-layer inputs too (embedOuts).
	if bs.rowPos != nil {
		w(bs.rowPos, g.rowPosWords(pos, nrow, R, rag))
	}
	// A tower's tap rows are the caller's always: input rows served here
	// write per-layer embeddings alone.
	if bs.pleIn != nil && (!embedSrv || p.PLEDim == 0) {
		w(bs.pleIn, f32b(g.pleRows(R*p.PLEWidth)))
	}
	if bs.req != nil && len(g.reqHost) > 0 {
		w(bs.req, u32b(g.reqWords(composer.RequestRows+R)))
	}
	switch {
	case raxes != nil:
		w(bs.ropeRows, raxes)
	case gridSrv:
		w(bs.gridPos, g.gridCells)
		w(bs.gridReq, g.gridReq)
	default:
		if bs.cs != nil {
			// nil for a block with no rotary; nothing reads it and there is
			// no buffer to write into.
			w(bs.cs, f32b(hcs))
		}
		if bs.csSWA != nil {
			w(bs.csSWA, f32b(hcsSWA))
		}
	}
	w(bs.n, pn)
	if bs.calls != nil && R > 1 {
		binary.LittleEndian.PutUint32(g.subBytes.calls[:], uint32(nrow))
		w(bs.calls, g.subBytes.calls[:])
	}
	if bs.pics != nil {
		w(bs.pics, bs.picDesc(pos, nrow, g.ncPhase, g.ncKeys, g.rowsFaulted("piccausal")))
		w(bs.picsW, bs.picWinDesc(pos, nrow, g.ncPhase, g.ncKeys, g.winRuns, g.winPerRow, g.rowsFaulted("piccausal")))
	}
	if bs.headCalls != nil && R > 1 && ragHead > 0 {
		hr := ragHead
		if bs.ragRows > 0 {
			hr = min(hr, bs.ragRows)
		}
		if g.rowsFaulted("headrow") {
			hr = 1
		}
		binary.LittleEndian.PutUint32(g.subBytes.headCalls[:], uint32(hr))
		w(bs.headCalls, g.subBytes.headCalls[:])
	}
	switch {
	case sampleK > 0 && sampleSrv:
		// The composed sampler reads the penalties, the window where they
		// apply (penalties that are off leave the row whatever the window
		// holds) and the walk's words.
		for i, p := range head.SamplePenalty {
			binary.LittleEndian.PutUint32(g.subBytes.penalty[4*i:], math.Float32bits(p))
		}
		w(g.samplePenalty, g.subBytes.penalty[:])
		if !head.SamplePenalty.Off() {
			w(g.sampleWindow, f32b(head.SampleWindow))
		}
		w(g.sampleWalk, f32b(head.WalkArgs[:]))
	}
	if pick {
		// The ids change every question; the recording reads them.
		w(g.pickIDs, g.pickStage(head))
	}
	if bs.dRows != nil {
		// The recurrent descriptor (see emitLinear), staged here outside any
		// recording: written inside emit it was a memcpy mid-capture, which
		// invalidates a CUDA stream capture. Its row count is nrow, the
		// caller's rows, not the grid R: padding rows must not go through
		// the convolution's shift and the delta rule as tokens. Its slots
		// are where each sequence's state is in the pools, which a
		// recording therefore does not bake.
		w(bs.dRows, g.recDesc(sid, nrow, false))
		if g.recShared && bs.cRows != nil {
			w(bs.cRows, g.recDesc(sid, nrow, true))
		}
	}
	if pk := bs.pkv; pk != nil && pk.held {
		w(pk.pRow, u32view(pk.desc))
		if len(pk.descW) > 0 {
			w(pk.pRowW, u32view(pk.descW))
		}
		if pk.st != nil {
			pk.st.writes(w)
		}
	}

	submit()

	if err != nil {
		return
	}
	// Read straight into the caller's slice when the lengths agree: copying
	// out of bs.hRaw was a measurable cost for a large vocabulary, and
	// cuMemcpyDtoH takes any host pointer.
	tr := time.Now()
	defer func() { g.TSubRead += time.Since(tr) }()
	// The head's input, before the reads below consume the session's
	// attention: the normed rows a prediction block reads (nn.Head.Hidden).
	if head != nil && head.Hidden != nil && err == nil {
		src, rows := bs.h, 1
		switch {
		case rag != nil || allRows:
			rows = min(ragHead, bs.ragRows)
		case fold:
			src = hb.h
		}
		if len(head.Hidden) < rows*p.NEmbd {
			err = fmt.Errorf("a %d-float hidden destination for %d rows of %d", len(head.Hidden), rows, p.NEmbd)
		} else {
			err = s.Read(src, f32b(head.Hidden[:rows*p.NEmbd]))
		}
		if err == nil && key.headSrv {
			g.HiddenHeads++
			g.HiddenRows += rows
		}
	}
	switch {
	case (rag != nil || allRows) && head != nil:
		// The first chunk's tokens and logits, in ragTok and ragLogits
		// [ragRows][vocab]; the wanted rows lead.
		n := min(ragHead, bs.ragRows)
		if err = g.readHeadChunk(s, bs, head, 0, n, p.NEmbd, false); err == nil && head.RowLogits && len(head.Logits) < ragHead*g.bs.head.nrows {
			err = fmt.Errorf("rows: %d logits for %d rows of %d", len(head.Logits), ragHead, g.bs.head.nrows)
		}
		for k := n; err == nil && k < ragHead; k += bs.ragRows {
			err = g.headChunk(s, lc, bs, hb, hNorm, head, k, min(bs.ragRows, ragHead-k), p.NEmbd, sampleSrv)
		}
		direct = true
	case sampleK > 0:
		tok := g.subBytes.one[:]
		if err = s.Read(g.argmaxOut, tok); err == nil {
			head.Token = int32(binary.LittleEndian.Uint32(tok))
		}
		head.Sampled = err == nil
		// Counted here, not at the launches, which a replayed recording skips.
		g.SampleReads++
		direct = true
	case pick:
		err = s.Read(g.pickOut, f32b(head.PickVals[:len(head.Pick)]))
		head.Picked = err == nil
		// Counted here, not at the launch, which a replayed recording skips.
		g.PickReads++
		direct = true
	case argmax:
		tok := g.subBytes.one[:]
		if err = s.Read(g.argmaxOut, tok); err == nil {
			head.Token = int32(binary.LittleEndian.Uint32(tok))
		}
		direct = true
	case head != nil && len(head.Logits) == len(hb.hOut):
		err = s.Read(hb.headOut(), f32b(head.Logits))
		direct = true
	case head != nil:
		err = s.Read(hb.headOut(), hb.hRaw)
	case cutW > 0 && len(g.cutHost) < nrow*cutW:
		err = fmt.Errorf("tier: a cut call of %d rows of %d, where the caller holds %d floats", nrow, cutW, len(g.cutHost))
	case cutW > 0:
		err = s.Read(bs.cut, f32b(g.cutHost[:nrow*cutW]))
		direct = true
	case len(x) == nrow*p.ResidW() && (p.Streams() == 1 || nrow == R):
		err = s.Read(bs.x, f32b(x))
		direct = true
	case p.Streams() > 1:
		// Every stream is read back, stream-major at R rows (streamRows cuts it).
		err = s.Read(bs.x, bs.hraw[:R*p.ResidW()*4])
	default:
		err = s.Read(bs.x, bs.hraw[:nrow*p.NEmbd*4])
	}
	// Input rows gathered here wrote the per-layer inputs on this device
	// alone; blocks after the call's, on another device, upload them from
	// the session's own (pleRows), as from a gather of its own (readPLE).
	if w := nrow * p.PLEWidth; err == nil && embedSrv && head == nil && p.PLEDim != 0 && w > 0 && len(g.pleHost) >= w {
		err = s.Read(bs.pleIn, f32b(g.pleHost[:w]))
	}
}

// recorderOf is s as a backend.Recorder, asserted once per session object:
// the backends reuse their sessions, and an interface-to-interface assertion
// on every token caches its answer in a heap object at a random later call.
func (g *devTier) recorderOf(s backend.Session) (backend.Recorder, bool) {
	if g.recSess != s {
		g.recSess = s
		g.rec, _ = s.(backend.Recorder)
	}
	return g.rec, g.rec != nil
}

// stagingBytes is a devTier's host bytes for a submission's small writes,
// reused so a decode token allocates none of them.
type stagingBytes struct {
	pn               []byte
	tok              []byte // a ragged step's tokens, read back
	embIDs           []byte // a prompt chunk's ids for the gather on the card (embIDs)
	ropeRows         []byte // the rows as the rotary tables read them (ropeRowsWords)
	rowPos           []byte // the rows' positions a block reads (rowPosWords)
	embPos           []byte // the composer's gather's positions (embedWords)
	fold             []byte // the folded head's row take's request (foldWords)
	one              [4]byte
	calls, headCalls [4]byte  // a call's rows (blockScratch.calls, headCalls)
	penalty          [12]byte // the sampler's penalties (nn.Head.SamplePenalty)
}

// u32b is v as four little-endian bytes in the devTier's own scratch, for a
// write that completes before the next one: every backend's Write copies p
// before it returns.
func (g *devTier) u32b(v uint32) []byte {
	binary.LittleEndian.PutUint32(g.subBytes.one[:], v)
	return g.subBytes.one[:]
}

func u32one(v uint32) []byte {
	return []byte{byte(v), byte(v >> 8), byte(v >> 16), byte(v >> 24)}
}

// auxBufs is every per-layer device vector that a page-out and a release both
// drop. It is one list because separate copies drifted repeatedly, each time
// leaking a new kind of vector (vision MLP biases, post-norms, a linear block's
// five, gpt-oss's sinks and biases). Both callers are full teardowns, so
// nothing here is re-read without being uploaded again.
func (l *layer) auxBufs() []backend.Buf {
	vs := make([]backend.Buf, len(l.vecs))
	for i, v := range l.vecs {
		vs[i] = v.buf
	}
	return append(vs,
		l.nAttn, l.nFFN, l.nAttnB, l.nFFNB, l.nQ, l.nK,
		l.clampB,     // Gemma 4's tower: its clipped linears' bounds
		l.ssmTaps.qs, // a linear block's convolution
		l.outScale,   // Gemma 4's output scalar
		l.pleOff,     // Gemma 4's per-layer embedding
	)
}

// dropLayerLocal frees the buffers PrepLayer allocates fresh for one block and
// shares with nothing: the KV cache, the norms, the biases and the router. It
// does not touch the weights, which live in g.res keyed by their bytes and may
// be referenced elsewhere. Callers hold g.mu.
func (g *devTier) dropLayerLocal(l *layer) {
	g.closeNGram(l)
	if r := l.router; r != nil {
		for k, v := range g.res {
			if v == r {
				delete(g.res, k)
				break
			}
		}
		if n := r.bytes(); n > 0 {
			g.refund(n)
		}
		if r.ok {
			g.freeResident(r)
		}
		l.router = nil
	}
	for _, b := range l.auxBufs() {
		if b != nil {
			b.Free()
		}
	}
	g.freeKV(l)
	freeRec(l)
	if n := l.recBytes; n > 0 {
		g.refund(n)
		g.RecBytes -= n
		l.recBytes = 0
	}
	if n := l.kvBytes; n > 0 {
		g.refund(n)
		if g.KVBytes >= n {
			g.KVBytes -= n
		}
	}
	if n := l.auxBytes; n > 0 {
		g.refund(n)
	}
}

// ReleaseLayers hands blocks [lo, hi) back to the host and frees what they held
// on the device, which lets placement be undone (demote, park, re-adopt).
//
// It must drop the recording: a captured sequence holds device addresses, and
// replaying it after these buffers are freed reads memory that belongs to
// something else. Kernels are not closed, since g.kerns is shared by every layer
// of the same shape; only the per-layer router kernel is. The packed arena is
// not touched, so re-preparing the block is an upload rather than a repack;
// giving the host bytes back is GPU.DropArena.
//
// It takes the whole block, every session's history included: it is what a
// move between devices and a device's teardown use. A session giving blocks
// back goes through leaveLayers.
func (g *devTier) ReleaseLayers(lo, hi int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.dropGraph()
	for li := lo; li < hi; li++ {
		g.releaseBlock(li)
	}
	g.afterRelease()
}

// leaveLayers is ReleaseLayers for one session: session sid's history
// on blocks [lo, hi) goes, and a block goes with it only when no other
// session has history there -- the weights are the model's, and another
// session is still running against them. It returns the blocks it released.
func (g *devTier) leaveLayers(sid uint64, lo, hi int) []int {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.dropGraph()
	var gone []int
	for li := lo; li < hi; li++ {
		l := g.layers[li]
		if l == nil {
			continue
		}
		if g.sharedBeyond(li, sid) {
			g.dropHistory(li, l, sid)
			continue
		}
		g.releaseBlock(li)
		gone = append(gone, li)
	}
	g.afterRelease()
	return gone
}

// graphBeside says another block placed here runs l's graph.
func (g *devTier) graphBeside(l *layer) bool {
	for _, o := range g.layers {
		if o != nil && o != l && o.graph == l.graph {
			return true
		}
	}
	return false
}

// releaseBlock frees block li and every session's history on it. Callers
// hold g.mu and drop the graph.
func (g *devTier) releaseBlock(li int) {
	l := g.layers[li]
	g.freeRecSnaps(li)
	if l == nil {
		return
	}
	// Its parts go with it, unless another block placed here runs the same
	// graph: a part is keyed by its graph, not by its block.
	if l.graph != nil && !g.graphBeside(l) {
		g.dropServed(l.graph)
	}
	// The same free path as a page-out (layer.tensors()); a release also
	// gives up the KV cache and the block itself.
	freed := g.dropTensors(l)
	if l.pg != nil && l.pg.in {
		l.pg.in = false
		g.pagesIn--
		g.pageBytes -= min(g.pageBytes, freed+l.pg.round)
	}
	// The KV cache is charged to the budget at PrepLayer, so releasing has
	// to give it back or the card leaks budget every migration.
	if n := l.kvBytes; n > 0 {
		g.refund(n)
		if g.KVBytes >= n {
			g.KVBytes -= n
		}
	}
	if n := l.auxBytes; n > 0 {
		g.refund(n)
	}
	g.freeKV(l)
	freeRec(l)
	// The block's layer leaves the pool, and with it every session's pages
	// there.
	if g.kvp != nil {
		g.dropKVLayer(g.kvp, li)
		g.dropKVLayer(g.kvp, entKey(li))
	}
	if n := l.recBytes; n > 0 {
		g.refund(n)
		g.RecBytes -= n
		l.recBytes = 0
	}
	for _, b := range l.auxBufs() {
		if b != nil {
			b.Free()
		}
	}
	g.closeNGram(l)
	delete(g.layers, li)
	g.layerGen++
	g.placeGen.Add(1)
}

// dropIfEmpty empties a device left with no block and no head: it holds
// nothing of its own, so the composed parts (the tables', the row copy's),
// the scratches and the staging go, refunded -- and the marks arena, unless
// a session's marks are in it. The next placement forms, builds and
// reserves them again. Callers hold g.mu.
func (g *devTier) dropIfEmpty() {
	if len(g.layers) != 0 || g.bs != nil && g.bs.head != nil {
		return
	}
	g.dropGraph()
	g.partsClose()
	g.dropAllScratch()
	if !g.marks.marksPlaced() {
		g.freeMarks()
	}
	// What it held at no block's slot -- an input table, a picture's rows --
	// served its blocks and its head, and neither is left: a session that
	// brings its blocks back loads them again. The rows' cache of them goes
	// with them.
	g.releaseRows()
	for k, r := range g.res {
		if k.block >= 0 || r.shared {
			continue
		}
		delete(g.res, k)
		if r.ok {
			if n := r.bytes(); n > 0 {
				g.refund(n)
			}
			g.freeResident(r)
		}
	}
}

// afterRelease is what every release ends with. Callers hold g.mu.
func (g *devTier) afterRelease() {
	// A pool no session holds a place in, and the seats of sessions that hold
	// none, go back to the budget.
	g.recTidy()
	// The last tower block out takes its set's scratch with it, and a block
	// out takes its parts: the next placement forms the widths again.
	g.dropIdleTower()
	g.vs().warm = [2]int{}
	g.dropIfEmpty()
	// A decline is cached: resident() keeps a failed entry so a tensor is not
	// re-priced every matvec, and that entry says no forever. Giving budget
	// back is what clears those tombstones, or a seam grown after a shrink is
	// silently refused. Only failures are dropped; a live entry is another
	// block's weights.
	for k, r := range g.res {
		if r != nil && !r.ok {
			delete(g.res, k)
		}
	}
	g.freeStale()
}

// ReserveKV is reserveKV for the zero session: a caller that never attached.
func (g *devTier) ReserveKV(pos int) bool {
	return g.reserveKV(0, pos)
}

// reserveKV gives session sid's sequence the pages pos positions of
// its history need, from outside a submission: growing a layer drops the
// captured graph, which deadlocks CUDA inside a Session. A restored prefix
// arrives with no submission at all.
func (g *devTier) reserveKV(sid uint64, pos int) bool {
	// No scratch is read: the session's pages are all it touches, so it takes
	// no lane.
	g = g.bareView(sid)
	if pos <= 0 {
		return true
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.asking = true
	defer func() { g.asking = false }()
	// No paged history is no attention history here (a tower's keys live for
	// one encode): nothing to reserve.
	if !g.paged {
		return true
	}
	if err := g.pagedHold(sid, pos); err != nil {
		g.LastErr = err.Error()
		return false
	}
	return true
}

// TrimKV is trimKV for the zero session: a caller that never attached.
func (g *devTier) TrimKV(pos int) bool {
	return g.trimKV(0, pos)
}

// trimKV gives back session sid's pages past pos, and reports whether
// any went. The trimmed pages stay in the pool, for this session's next
// positions or anyone's; kvCompact gives them to the budget when something
// needs the room. They are room all the same, so it is announced.
func (g *devTier) trimKV(sid uint64, pos int) bool {
	g = g.bareView(sid)
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.kvp == nil || !g.kvp.trimSeq(seqID{sid, 0}, pos) {
		return false
	}
	g.roomGen.Add(1)
	return true
}

// dropSession frees everything session sid holds on this device: its history
// and its recurrent state on every block.
func (g *devTier) dropSession(sid uint64) {
	g.mu.Lock()
	defer g.mu.Unlock()
	delete(g.hostFns, sid)
	freed := false
	for li, l := range g.layers {
		freed = g.dropHistory(li, l, sid) || freed
	}
	if freed {
		// A captured launch sequence names the buffers just freed.
		g.dropGraph()
	}
	// Its own recordings name nothing that will be read again either, in
	// whichever lane they were made, and its state on this device goes.
	g.forgetSess(sid)
	// Its seat in the recurrent pools, and a pool it was the last to hold.
	g.recTidy()
	// Its pages are free now and stay in the pool for the next session;
	// kvCompact gives them back when something needs the room. The scratch's
	// bound stays: paged history sizes nothing by context, so resetting it
	// only rebuilt the scratch -- the batched ones with it -- at the next
	// session's first call, inside its prompt: every prompt after a process's
	// first ran about a fifth slower (gpu-kernels.md, "The prompt lost 22%").
}

// dropHistory frees session sid's history on block l -- its KV cache or its
// recurrent pair -- and reports whether it had any. Callers hold g.mu and drop
// the graph when it did.
func (g *devTier) dropHistory(li int, l *layer, sid uint64) bool {
	if l == nil {
		return false
	}
	delete(l.runs, sid)
	freed := false
	if kvp := l.kv[sid]; kvp != nil && kvp.paged {
		// Its pages in this layer go and the layer compacts: what the session
		// held here is room again, whoever keeps the block.
		delete(l.kv, sid)
		g.pagedLeave(li, sid, l)
		g.pagedLeave(entKey(li), sid, l)
		freed = true
	} else if kvp != nil && kvp.shared {
		delete(l.kv, sid)
		g.transientKVPut(l.geo)
		freed = true
	} else if kvp != nil {
		if kvp.kc != nil {
			kvp.kc.Free()
		}
		if kvp.vc != nil {
			kvp.vc.Free()
		}
		delete(l.kv, sid)
		g.refund(kvp.bytes)
		l.kvBytes -= min(l.kvBytes, kvp.bytes)
		g.KVBytes -= min(g.KVBytes, kvp.bytes)
		freed = true
	}
	if l.rec[sid] != nil {
		// Its slots in the block's pool are free; recTidy gives the pool, or
		// the slots past the last seat, back to the budget.
		delete(l.rec, sid)
		freed = true
	}
	return freed
}

// MigrateRec is migrateRec for the zero session: a caller that never attached.
func (g *devTier) MigrateRec(li int, conv, state []float32, toDevice bool) bool {
	return g.migrateRec(0, li, conv, state, toDevice)
}

// migrateRec moves block li's recurrent summary between host and device.
// engine.Instance.rconv/rstate are what every host consumer reads (including the
// -kv-cache seal), and a placed linear block never touches them, so without this
// a cached restore would resume from a summary that never saw the prompt. It
// reads half cur, the state as of the last completed token, at the session's
// seat in the block's pool.
func (g *devTier) migrateRec(sid uint64, li int, conv, state []float32, toDevice bool) bool {
	v := g.as(sid)
	if v == nil {
		return false
	}
	defer v.done()
	g = v
	g.mu.Lock()
	defer g.mu.Unlock()
	l := g.layers[li]
	if l == nil {
		return false
	}
	rp := l.recOf(sid)
	if rp == nil {
		// Not a linear block, or no session state: nothing to move is success,
		// exactly as MigrateKV treats a block with no cache.
		return true
	}
	if l.pool == nil || g.recC == 0 {
		return false
	}
	// A batch hands every row's state back to back, row r at r*ConvState and
	// r*StateLen: the seat's own layout, one slot a row.
	rows := len(conv) / g.recC
	if rows < 1 || len(conv) != rows*g.recC || len(state) != rows*g.recS {
		g.LastErr = fmt.Sprintf("MigrateRec: block %d wants a multiple of %d conv and %d state floats, "+
			"the caller offered %d and %d", li, g.recC, g.recS, len(conv), len(state))
		return false
	}
	st := g.recSeats[sid]
	switch {
	case st.rows == rows:
	case st.rows < rows && !toDevice:
		// The seat was never grown to the batch, and it grows at the batch's
		// first step (LayersRows) or before a state is sent up (below): no
		// step has run these rows here, so the host's copy is the current one
		// and there is nothing to read back.
		return true
	case st.rows < rows:
		// Sent up for a batch the seat does not hold yet: grow it first.
		// growSeat discards the session's states, which is right, since a
		// batch's rows have no history on this device before its first step
		// and every row is written next.
		if !g.growSeat(sid, rows) {
			return false
		}
		st = g.recSeats[sid]
	default:
		g.LastErr = fmt.Sprintf("MigrateRec: block %d holds %d row(s), the caller offered %d",
			li, st.rows, rows)
		return false
	}
	sh := rp.cur
	if g.recShared {
		// One persistent state (s[1] is nil): the step reads s[0] whichever
		// half of the conv window is current, as emitLinear does.
		sh = 0
	}
	if !toDevice {
		rd := func(b backend.Buf, v []float32, w int, name string) bool {
			// A buffer reads from its start, so a seat above slot 0 is first
			// copied to a buffer of its own on the device: reading the pool
			// directly would bring every other session's state below it to the CPU
			// too.
			err := func() error {
				if len(v) == 0 {
					// A block with no state of this kind (a short
					// convolution has no delta-rule half): nothing to read,
					// and a zero-byte buffer is no allocation a driver takes.
					return nil
				}
				if st.base == 0 {
					return b.Read(f32b(v))
				}
				tmp, err := g.dev.Alloc(len(v) * 4)
				if err != nil {
					return err
				}
				defer tmp.Free()
				if err := g.dev.Copy(tmp, 0, b, st.base*w*4, len(v)*4); err != nil {
					return err
				}
				return tmp.Read(f32b(v))
			}()
			if err != nil {
				g.LastErr = fmt.Sprintf("MigrateRec: block %d %s: %v", li, name, err)
				return false
			}
			return true
		}
		return rd(l.pool.c[rp.cur], conv, g.recC, "conv") && rd(l.pool.s[sh], state, g.recS, "state")
	}
	// Both halves. Seeding cur alone is already correct (the next submission
	// writes 1-cur before anything reads it); writing both is defence, so no
	// future caller depends on that parity argument.
	for i := 0; i < 2; i++ {
		if err := l.pool.c[i].WriteAt(st.base*g.recC*4, f32b(conv)); err != nil {
			g.LastErr = fmt.Sprintf("MigrateRec: block %d conv half %d: %v", li, i, err)
			return false
		}
		if l.pool.s[i] == nil || len(state) == 0 {
			continue // a shared pool's second half is the device's scratch
		}
		if err := l.pool.s[i].WriteAt(st.base*g.recS*4, f32b(state)); err != nil {
			g.LastErr = fmt.Sprintf("MigrateRec: block %d state half %d: %v", li, i, err)
			return false
		}
	}
	return true
}

// MigrateKV is migrateKV for the zero session: a caller that never attached.
func (g *devTier) MigrateKV(li int, k, v []float32, pos int, toDevice bool) bool {
	return g.migrateKV(0, li, k, v, pos, toDevice)
}

// migrateKV moves block li's attention history between host and device.
// Without it a block that changes sides mid-generation attends over whatever
// was in the pages it arrived at: fluent, wrong text.
//
// k and v are the host arrays for this block, laid out [pos*kvDim + i] in
// float32; only positions [0, pos) are moved (pagedMigrate). A block with no
// paged history here -- a tower, whose keys live for one encode -- has
// nothing to move.
func (g *devTier) migrateKV(sid uint64, li int, k, v []float32, pos int, toDevice bool) bool {
	sv := g.as(sid)
	if sv == nil {
		return false
	}
	defer sv.done()
	g = sv
	g.mu.Lock()
	defer g.mu.Unlock()
	l := g.layers[li]
	kvp := l.kvOf(sid)
	if kvp == nil || !kvp.paged || pos <= 0 {
		return l != nil
	}
	if err := g.pagedMigrate(sid, li, 0, k, v, pos, toDevice); err != nil {
		g.LastErr = fmt.Sprintf("MigrateKV: block %d: %v", li, err)
		return false
	}
	return true
}

// PrewarmLayer packs a whole block off the main loop. See GPU.Prewarm.
func (g *devTier) PrewarmLayer(li int, p *nn.LayerPlan, w *nn.LayerWeights) bool {
	// A lazy weight is not prewarmable: this must be safe beside a running
	// Layers, and nn.LayerWeights.Ensure reads the file into the pager's frames.
	// Packing without it would pack whatever the frame holds. The upload calls
	// Ensure itself, so only the prewarm is lost.
	if w.Ensure != nil {
		return false
	}
	any := false
	// weightList, not a second copy of it: a matrix left off would silently
	// never prewarm.
	for _, x := range weightList(w) {
		q, ok := quantOf(x.T)
		if !ok || x.Rows <= 0 || x.K <= 0 || len(x.Data) == 0 {
			continue
		}
		if g.prewarm(li, q, x.Data, x.Rows, x.K) {
			any = true
		}
	}
	return any
}

// weightList is a block's pageable matrices in layer.tensors() order.
//
// One list, for tensors()' reason: prepLayer builds it to decide and upload,
// and pageIn rebuilds it after Ensure re-points the slices. Two copies of the
// order could upload one matrix into another's slot with every size right.
func weightList(w *nn.LayerWeights) []nn.Weight {
	return []nn.Weight{w.Wq, w.Wk, w.Wv, w.Wo, w.Gate, w.Up, w.Down,
		w.ShGate, w.ShUp, w.ShDown,
		w.SSM.Gate, w.SSM.BA, w.SSM.Out,
		w.Wqa, w.Wqb, w.Wkva, w.Wkb, w.Wvb,
		w.SSM.FA, w.SSM.FB, w.SSM.GA, w.SSM.GB}
}

// nNamed is how many matrices weightList names; a block's extra follow.
const nNamed = 22

// slotRouter is a block's router in g.res: no page slot, since it is not
// paged with the block.
const slotRouter = -1

// roleRes is one of a block's extra matrices and its role.
type roleRes struct {
	role jlm.Role
	r    *resident
}

// takeExtra makes l.extra the matrices of w.Extra that l's graph binds, and
// is their weights in that order.
func (l *layer) takeExtra(w *nn.LayerWeights) []nn.Weight {
	binds := map[jlm.Role]bool{}
	if l.graph != nil {
		for _, o := range l.graph.Ops {
			for _, b := range o.Weight {
				binds[b.Role] = true
			}
		}
	}
	l.extra = l.extra[:0]
	var ws []nn.Weight
	for _, x := range w.Extra {
		if binds[x.Role] && len(x.W.Data) != 0 {
			l.extra = append(l.extra, roleRes{role: x.Role})
			ws = append(ws, x.W)
		}
	}
	return ws
}

// pageList is weightList and l's extra, as a page-in reads them again.
func (l *layer) pageList(w *nn.LayerWeights) []nn.Weight {
	ws := weightList(w)
	for _, e := range l.extra {
		var x nn.Weight
		for _, y := range w.Extra {
			if y.Role == e.role {
				x = y.W
				break
			}
		}
		ws = append(ws, x)
	}
	return ws
}

// BatchTok is how many token columns one thread of a batched matvec carries.
// A chunk of NTok tokens read one at a time reads the weights NTok times; with
// Tok columns per thread it reads them NTok/Tok times. Larger runs out of
// registers and occupancy. 16 won the sweep once attention was tiled; the best
// width moves with the kernel mix, so re-sweep after changing it.
// BatchShape.Tok overrides it.
const BatchTok = 16

// The batched prefill's tile geometry. Each is the fitted default; a tier's
// WithBatchShape and WithMMATile pins replace it for that tier (tilesFor).
//
// defBatchRowt is how many rows one thread carries. See MatVecShape.Rowt: Tok
// alone plateaus because the kernel issues one activation load per dot4, and a
// tile of rows shares those loads. 4 won by a wide margin in mvbench -ntok 128;
// 8 does not fit its accumulators.
//
// defMMAMT and defMMANT are the warp tile of the matrix-instruction matvec: MT
// blocks of 16 weight rows by NT blocks of 8 token columns. 4x4 beat narrower
// tiles and the dp4a tile in mvbench -mma -ntok 128 (see mmaTile).
//
// defAttnQTile and defAttnKTile are the query rows and key positions one thread
// of the batched scores kernel carries. maxKeyTile bounds the score-row
// padding, so it must not be smaller than the key tile. 4x2 won the end-to-end
// sweep: the query side is a broadcast (widening costs registers) while the key
// side is a strided walk (widening costs coalescing).
//
// defAttnATile is how many query rows one AttnAcc thread carries, so one V load
// serves all of them (see AttnAccTiled); 8 beat 1 end to end.
//
// defAttnNT is how many 8-position key tiles one warp of the tensor-core scores
// kernel carries, so one q fragment serves all of them; 2 beat 1, and 4 is no
// better.
//
// defBatchParts is the RMSNorm partial count for a batched chunk; see
// initScratch.
const (
	defBatchRowt               = 4
	defMMAMT, defMMANT         = 4, 4
	defAttnQTile, defAttnKTile = 4, 2
	defAttnATile               = 8
	defAttnNT                  = 2
	defBatchParts              = 64
)

// tiles is one tier's resolved batched geometry.
type tiles struct {
	rowt, mmaMT, mmaNT, qtile, ktile, atile, attnNT, parts int
}

// tilesFor applies a tier's pins to the defaults.
func tilesFor(kb knobs) tiles {
	t := tiles{rowt: defBatchRowt, mmaMT: defMMAMT, mmaNT: defMMANT, qtile: defAttnQTile,
		ktile: defAttnKTile, atile: defAttnATile, attnNT: defAttnNT, parts: defBatchParts}
	b := kb.batch
	if b.RowT >= 1 {
		t.rowt = b.RowT
	}
	for _, e := range []struct {
		v   int
		dst *int
	}{{b.QTile, &t.qtile}, {b.KTile, &t.ktile}, {b.ATile, &t.atile}, {b.Parts, &t.parts},
		{b.AttnNT, &t.attnNT}} {
		if e.v >= 1 && e.v <= 128 {
			*e.dst = e.v
		}
	}
	if kb.mmaMT > 0 && kb.mmaNT > 0 {
		t.mmaMT, t.mmaNT = kb.mmaMT, kb.mmaNT
	}
	return t
}

// maxKeyTile bounds how far past a row's causal width a kernel may write, so
// the score row is padded by it. The FMA tile writes up to ktile-1 past; the
// matrix kernel up to 8*attnNT-1; sm_70's AttnScoresMMA70 up to
// 32*volta70AttnMT-1.
//
// It must cover the widest tile: an overshoot writes -inf into the next row's
// scores, which softmaxes to NaN. volta70Attn refuses a tile wider than this.
const maxKeyTile = 64

// scoreStride is a block's score row: the context, padded for the batched
// arm -- AttnScoresTiled's last key tile can write up to maxKeyTile-1 past the
// causal width, which unpadded lands in the next row's scores -- and rounded
// to four floats, since AttnAccMMA70 reads a row's probabilities 16 bytes at a
// time and every batched kernel takes this one stride.
func scoreStride(p *nn.LayerPlan, rows int) int {
	if rows > 1 {
		return (p.MaxSeq + maxKeyTile + 3) &^ 3
	}
	return p.MaxSeq
}

// batchWidth is how many token positions one batched submission carries. It has
// to be a multiple of BatchTok, and larger only amortizes the launches -- the
// weight traffic is set by Tok alone.
const batchWidth = nn.MaxDevicePrefillChunk

// prepBatch builds the batched scratch and the composer's prompt forms at
// its width, on the first prompt that asks for one rather than at PrepLayer
// time.
//
// It is lazy because the scratch costs VRAM (mostly score buffers) that a
// decode-only process never uses, and separate because Alloc and Compile cannot
// run inside a Session.
func (g *devTier) prepBatch(width int) bool {
	// 8, not batchGrain: a ragged step on a device without a matrix
	// instruction runs at a multiple of 8 (see LayersRows).
	if g.bs == nil || width < 8 || width%8 != 0 {
		return false
	}
	if g.bbs == nil {
		g.bbs = map[int]*blockScratch{}
	}
	if g.bbs[width] == nil {
		if g.noRoomFor(width) {
			return false
		}
		tok := BatchTok
		if k := g.kb.batch.Tok; k >= 1 && batchWidth%k == 0 {
			tok = k
		}
		for tok > 1 && width%tok != 0 {
			tok /= 2
		}
		bs := g.initScratch(&g.bs.p, width)
		if bs == nil {
			return false
		}
		if g.lane != nil && g.lane.clone {
			g.LaneBuilds++
		} else {
			g.ScratchBuilds++
		}
		bs.tok = tok
		g.bbs[width] = bs
		// A width built after placement filled the card must not take what
		// the blocks were budgeted: it is refused, and the caller runs the
		// reserved width (batchFor) or narrower submissions. Without the
		// reservation the device asks the card, as it did before it. Nor
		// may it take the history admitted rows were promised (PromiseKV).
		// Free pool pages are not a reservation (PrewarmKV grows them into
		// whatever room placement left): they are given back first.
		if !g.NoScratchReserve && !g.kvCompact(g.promisedBytes(), nil) && width != g.promptW && width != g.stepW && !g.reserving {
			g.dropBatch(width)
			g.markNoRoom(width)
			g.ScratchRefused++
			g.LastErr = fmt.Sprintf("a %d-row batched scratch does not fit the budget (%d of %d bytes charged)", width, g.used, g.limit)
			return false
		}
	}
	defer g.scratchWin().close()
	// The layer walk is not guarded by the scratch: ReleaseLayers deletes
	// blocks, so a moved seam brings back blocks whose batched kernels are not
	// compiled, and finding that mid-submission (where Compile is not allowed)
	// would refuse every later batched prompt. The compiled kernels live in
	// g.kerns keyed by shape, which holds every width; batKern is then a map
	// lookup, safe inside a Session.
	bb := g.bbs[width]
	// The walk is ~25 kernel lookups a block, every prompt (0.9 ms of a
	// 512-row Qwen3-0.6B prompt on a Volta-class card); it only has something
	// to find after a block was placed or released.
	if bb.walked == g.layerGen+1 {
		return true
	}
	// The composer's prompt forms at this width and their instances
	// (nn.ComposedPrompt).
	g.promptForms(bb)
	if g.ahead == nil {
		bb.walked = g.layerGen + 1
	}
	return true
}

// batchGrain is the coarsest granularity every batched kernel divides: the
// matrix instruction covers 8 token columns and the shipped tile takes four of
// them, so a chunk width is a multiple of 32. A ragged tail is rounded up to
// this rather than to the full width, which is the difference between 67% and
// 33% waste on a 42-row tail.
const batchGrain = 8 * 4

// promptGrain is the grain a prompt chunk's own width is rounded to: the
// batched forms' batchGrain. Callers hold g.mu.
func (g *devTier) promptGrain() int { return batchGrain }

// attendsIn reports whether a block of [lo, hi) on this device attends: not
// a linear block's recurrence.
func (g *devTier) attendsIn(lo, hi int) bool {
	for li := lo; li < hi; li++ {
		if l := g.layers[li]; l != nil && !l.linear {
			return true
		}
	}
	return false
}

// capped is the plan with MaxSeq replaced by what this device is actually built
// for. Every kernel build, every allocation and every bound goes through it. It
// also caps the score scratch (rows*NHead*sstride*4 twice), which at a long
// context is larger than the KV cache.
func (g *devTier) capped(p *nn.LayerPlan) nn.LayerPlan {
	q := *p
	// A vision plan's MaxSeq is the largest grid its tower takes, which a
	// dynamic-resolution tower puts far past any picture (65536 patches on
	// HunyuanOCR): the set is built for the rows ReserveRows last sized it to
	// (visrows.go), and a call is one submission of at most that many.
	if q.NonCausal {
		// The plan's own set's rows, whichever set is in use.
		if v := g.visOf(geoOf(p)); v.cap > 0 {
			q.MaxSeq = min(q.MaxSeq, max(v.cap, v.keys))
		}
		return q
	}
	// The device's capacity, not the smaller of the two: every stride and slot
	// is kvCap wide, so a session with a shorter context still needs a cache
	// that wide.
	if g.kvCap > 0 {
		q.MaxSeq = g.kvCap
	}
	return q
}

// ensureKVCap makes the scratch cover the longest context a session has asked
// for (prepLayer raises maxSeqAsked). The history has no capacity -- it is
// paged -- so only the scratch's bound moves, and rebuilding it drops the
// graph, which cannot happen inside a Session (a CUDA deadlock): Layers calls
// this before entering one. Callers must NOT hold g.mu.
func (g *devTier) ensureKVCap(pos int) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	// Per lane: a lane built before the growth still has the old bound,
	// whichever lane grew first.
	if !g.paged || g.bs == nil || g.kvCap >= g.maxSeqAsked && g.bs.p.MaxSeq >= g.maxSeqAsked {
		return true
	}
	g.kvCap = g.maxSeqAsked
	// The recordings made in this lane name the scratch about to go; no
	// other lane's changes.
	g.dropLaneGraph()
	// Every scratch set (geomsets.go) is built at the capacity, not only the
	// one in use: a set left at the old bound refuses the next position.
	home := g.geoCur
	defer g.useGeom(home)
	for _, k := range append([]geoKey{home}, g.geoKeys()...) {
		g.useGeom(k)
		// A non-causal set is built for its run, not for a context.
		if g.bs == nil || g.bs.p.NonCausal {
			continue
		}
		plan := g.bs.p
		plan.MaxSeq = g.maxSeqAsked
		if !g.rebuildScratch(&plan) {
			return false
		}
	}
	// The composer's parts follow the scratch: formed here, at the bound the
	// rebuild took, as a placement forms them, not inside the step.
	g.useGeom(home)
	g.prewarmComposed()
	return true
}

// rebuildScratch replaces every scratch with one built for plan, carrying the
// output projection across.
//
// The head is what a rebuild would otherwise lose: its matvec and its
// hNorm/logits buffers do not depend on the KV capacity or the block shape, but
// freeScratch frees them and a fresh scratch forgets the resident head, leaving
// the model believing the head is placed.
func (g *devTier) rebuildScratch(plan *nn.LayerPlan) bool {
	prev := g.bs
	// Same shape at the same bound is nothing a rebuild would change.
	if prev != nil && sameShape(prev.p, g.capped(plan)) && prev.p.MaxSeq == g.kvCap {
		return true
	}
	fresh := g.initScratch(plan, 1)
	if fresh == nil {
		return false
	}
	moveHead(fresh, prev)
	g.dropScratch(prev)
	for w := range g.bbs {
		g.dropBatch(w)
	}
	g.bs = fresh
	g.shapeChanged()
	// The reserved widths went with the old scratches: built again at the new
	// shape, here, because a capacity growth rebuilds outside any placement.
	g.reserveScratch(plan, 0)
	return true
}

// planUnion widens a to cover b, and reports whether it had to.
//
// Sessions and block kinds can offer one tier different plans that one scratch
// must serve: a hybrid's linear blocks (Recurrent set, AttnOutGate cleared by
// model.offerRange), a session needing a sliding window, a dense lead block
// before mixtures. Each widened field only adds kernels and buffers that the
// narrower blocks never launch.
func planUnion(a *nn.LayerPlan, b *nn.LayerPlan) bool {
	grew := false
	// Sinks and mixture biases only add work a block without them skips (a
	// sinkless block passes a -inf sink; an unbiased one skips the adds), so a
	// scratch built for them serves every block.
	if b.AttnSinks && !a.AttnSinks {
		a.AttnSinks, grew = true, true
	}
	if b.MoEBias && !a.MoEBias {
		a.MoEBias, grew = true, true
	}
	if b.AttnOutGate && !a.AttnOutGate {
		a.AttnOutGate, grew = true, true
	}
	if b.NoPreNorm && !a.NoPreNorm {
		a.NoPreNorm, grew = true, true
	}
	if b.Recurrent.Conv > 0 && a.Recurrent.Conv == 0 {
		a.Recurrent, grew = b.Recurrent, true
	}
	// A window is a mask: a session that never reaches it computes the same
	// thing through the windowed kernels.
	if b.SWAWindow > 0 && a.SWAWindow == 0 {
		a.SWAWindow, grew = b.SWAWindow, true
	}
	// The mixture: DeepSeek's first blocks are dense, so the first block this
	// tier sees may allocate no MoE buffers. A mixture scratch serves a dense
	// block too, since the two use different buffers; the one non-additive
	// kernel is the activation (baked element count), hence bs.actMulE.
	if b.NExpert > 0 && a.NExpert == 0 {
		a.NExpert, a.NExpertUsed, a.NFFNExp = b.NExpert, b.NExpertUsed, b.NFFNExp
		a.ExpertSigmoid, a.ExpertSelBias = b.ExpertSigmoid, b.ExpertSelBias
		a.ExpertGroups, a.ExpertGroupsUsed = b.ExpertGroups, b.ExpertGroupsUsed
		a.ExpertScale, a.NoExpertNorm = b.ExpertScale, b.NoExpertNorm
		grew = true
	}
	// The shared expert and the dense FFN are widths, so the union is the max:
	// a DeepSeek mixture block carries no dense FFN at all and the dense lead
	// carries no experts, so neither alone sizes the scratch.
	if b.NFFNShExp > a.NFFNShExp {
		a.NFFNShExp, grew = b.NFFNShExp, true
	}
	if b.NFFN > a.NFFN {
		a.NFFN, grew = b.NFFN, true
	}
	// The streams are the model's; the experts' width is the mixture
	// blocks', which a dense lead does not carry.
	if b.NStreams > a.NStreams {
		a.NStreams, grew = b.NStreams, true
	}
	if b.ExpLatent > a.ExpLatent {
		a.ExpLatent, grew = b.ExpLatent, true
	}
	// A block that is its mixer alone says nothing about the FFN the others
	// run, gated or not.
	if a.NoFFN && !b.NoFFN {
		a.NoFFN, a.UngatedFFN, grew = false, b.UngatedFFN, true
	}
	return grew
}

// planConflict names a plan this tier's scratch cannot serve alongside the one
// it holds, or "". What a block computes is its composed part's, formed from
// its own graph (servedBlockAs over layer.graph): its norms, gating, scales,
// latent geometry and delta rule are the graph's, and its history's geometry
// its own plan's (kvGeomOf). The scratch holds only what a call binds for
// every block of its set (initScratch), and of a plan's attention it bakes
// one thing: the window its rows' windowed descriptors are staged for
// (pagedStage), so two different windows are the case.
func planConflict(bs *blockScratch, p *nn.LayerPlan) string {
	if bs == nil || p.NonCausal {
		return ""
	}
	if a, b := bs.p.SWAWindow, p.SWAWindow; a > 0 && b > 0 && a != b {
		return fmt.Sprintf("a sliding window of %d keys on a tier built for %d", b, a)
	}
	// A chunk is a window staged another way (pagedStage).
	if bs.p.SWAChunked != p.SWAChunked {
		return "a chunked attention of a different shape from the one this tier was built for"
	}
	return ""
}

// nonCausalConflict names a non-causal block the non-causal set, built from
// the first such block's plan, cannot run, or "". planConflict passes every
// non-causal plan, since a tower's blocks are one shape; an encoder with a
// decision head after it (Laya) is two. A block's own MLP width, gating and
// activation, and whether it rotates, are its own (layer.nffn, the set grown
// to the widest MLP); what the set's kernels bake for every block is the
// heads and their width, the norm kind, the rotary's width and pairing, a
// gate buffer and the copy where a norm is missing, so a block differing in
// one of those would run on the first one's kernels and is declined by name.
func nonCausalConflict(bs *blockScratch, p *nn.LayerPlan) string {
	if bs == nil || !p.NonCausal {
		return ""
	}
	a := &bs.p
	gated := map[bool]string{true: "ungated", false: "gated"}
	switch {
	case a.NEmbd != p.NEmbd || a.NHead != p.NHead || a.NKVHead != p.NKVHead || a.HeadDim != p.HeadDim:
		return fmt.Sprintf("a non-causal block of %d heads of %d (%d kv) over %d on a set built for %d of %d (%d kv) over %d",
			p.NHead, p.HeadDim, p.NKVHead, p.NEmbd, a.NHead, a.HeadDim, a.NKVHead, a.NEmbd)
	case !ungatedFFN(p) && ungatedFFN(a):
		// The set of an ungated MLP has no gate buffer; a block's own width,
		// gating and activation are otherwise its own (layer.nffn).
		return fmt.Sprintf("a non-causal block's MLP (%s, %d wide) on a set built for (%s, %d wide)",
			gated[ungatedFFN(p)], p.NFFN, gated[ungatedFFN(a)], a.NFFN)
	case p.LayerNorm != a.LayerNorm:
		return fmt.Sprintf("a non-causal block with LayerNorm %v on a set built with %v", p.LayerNorm, a.LayerNorm)
	case p.NRot > 0 && !p.NoPosEnc && (a.NRot == 0 || a.NoPosEnc):
		// A set that rotates nothing builds no rotary kernels; a block that
		// rotates nothing runs on one that does (layer.noRope).
		return fmt.Sprintf("a non-causal block rotating %d of each head on a set built to rotate none", p.NRot)
	case p.NRot > 0 && a.NRot > 0 && (p.NRot != a.NRot || p.RopeNeox != a.RopeNeox):
		return fmt.Sprintf("a non-causal block rotating %d (NEOX %v) on a set built for %d (NEOX %v)",
			p.NRot, p.RopeNeox, a.NRot, a.RopeNeox)
	case p.NoPreNorm && !a.NoPreNorm:
		return "a non-causal block with no pre-norm on a set built for blocks that have one"
	}
	return ""
}

// composedPageMin is the fewest positions a composed part's history page
// holds (composer.Step.Page): a part lays its keys out at that stride, so a
// history of fewer positions is allocated for that many.
const composedPageMin = 64

// kvCacheBytes is one session's K and V cache for one block at capacity c.
func (g *devTier) kvCacheBytes(p *nn.LayerPlan, c int) (cache, vcache int) {
	cache = max(c+1, composedPageMin) * p.KVRow() * 4
	// MLA has no V buffer: one row per position, the value its prefix.
	if p.MLA() {
		return cache, 0
	}
	vcache = cache
	if g.KVF16 {
		vcache = cache / 2
	}
	return cache, vcache
}

// restride copies rows rows of cols floats from src, srcStride apart, to dst,
// dstStride apart: one device copy where the run is contiguous, else a row a
// copy, run as one list. Every block of a growth has the same shape.
func (g *devTier) restride(src, dst backend.Buf, rows, cols, srcStride, dstStride int) bool {
	var err error
	if rows == 1 || cols == srcStride && cols == dstStride {
		err = g.dev.Copy(dst, 0, src, 0, rows*cols*4)
	} else {
		cs := g.gatherCopies[:0]
		for r := range rows {
			cs = append(cs, backend.CopyRange{Dst: dst, DstOff: r * dstStride * 4, Src: src, SrcOff: r * srcStride * 4, N: cols * 4})
		}
		g.gatherCopies = cs
		err = backend.CopyAll(g.dev, cs)
		clear(cs)
	}
	if err != nil {
		g.LastErr = err.Error()
		return false
	}
	return true
}

// initKVCap sets the starting capacity the first time this device sees a plan:
// one page, or the whole context when that is smaller.
func (g *devTier) initKVCap(p *nn.LayerPlan) {
	if g.kvCap != 0 || p.NonCausal {
		// A tower carries no growing history, so it must not be what sets the
		// ceiling the text model grows toward.
		return
	}
	g.maxSeqAsked = p.MaxSeq
	// Paged history has no capacity to grow: no kernel bakes one, so the
	// scratch is built for the whole context once and every session shares it.
	g.paged, g.kvCap = true, p.MaxSeq
}

// headChunk heads rows [k, k+n) of a ragged step's wanted rows, past the
// ones its logits hold (ragRows), after the submission: the rows are taken
// out of the residual into the staging set (headIn), headed and sampled there
// as the submission heads its first, and their tokens, logits and normed rows
// read into the call's at k.
func (g *devTier) headChunk(s backend.Session, lc *launcher, bs, hb *blockScratch, hNorm backend.Buf, head *nn.Head, k, n, nembd int, sampleSrv bool) error {
	if bs.headIn == nil || bs.headReq == nil {
		return fmt.Errorf("rows: %d wanted rows past the %d the logits hold, with no staging built", k+n, bs.ragRows)
	}
	b := g.subBytes.fold[:0]
	for range composer.RequestRows {
		b = binary.LittleEndian.AppendUint32(b, 0)
	}
	for r := range bs.ragRows {
		b = binary.LittleEndian.AppendUint32(b, uint32(k+min(r, n-1)))
	}
	g.subBytes.fold = b
	if err := s.Write(bs.headReq, b); err != nil {
		return err
	}
	binary.LittleEndian.PutUint32(g.subBytes.headCalls[:], uint32(n))
	if err := s.Write(bs.headCalls, g.subBytes.headCalls[:]); err != nil {
		return err
	}
	tk, err := g.servedRowTakeOf(bs.headReq, nembd, bs.rows, bs.ragRows)
	if err != nil {
		return err
	}
	g.rows.takeIn[0], g.rows.takeOut[0] = bs.x, bs.headIn
	if err := tk.run(g, lc, tk.inst, g.rows.takeIn[:], g.rows.takeOut[:]); err != nil {
		return err
	}
	sv, err := g.servedHead(hb, hNorm, bs.ragRows)
	if err != nil {
		return err
	}
	sv.rows, sv.sl.tables[tableKey(composer.TableCall)] = n, bs.headCalls
	if err := sv.runHead(g, lc, bs.headIn, bs.ragLogits, bs.h); err != nil {
		return err
	}
	if err := g.ragTokens(lc, bs, hb, n, sampleSrv); err != nil {
		return err
	}
	g.HeadChunks++
	return g.readHeadChunk(s, bs, head, k, n, nembd, true)
}

// readHeadChunk reads a head chunk's n rows into the call's at k: their
// tokens, their logits where the call asks for every row's, and their normed
// rows (nn.Head.Hidden) where hidden says the chunk wrote them.
func (g *devTier) readHeadChunk(s backend.Session, bs *blockScratch, head *nn.Head, k, n, nembd int, hidden bool) error {
	tb := slices.Grow(g.subBytes.tok[:0], 4*bs.ragRows)[:4*bs.ragRows]
	g.subBytes.tok = tb
	if err := s.Read(bs.ragTok, tb); err != nil {
		return err
	}
	for r := range n {
		head.Tokens[k+r] = int32(binary.LittleEndian.Uint32(tb[4*r:]))
	}
	vocab := g.bs.head.nrows
	if head.RowLogits {
		if len(head.Logits) < (k+n)*vocab {
			return fmt.Errorf("rows: %d logits for %d rows of %d", len(head.Logits), k+n, vocab)
		}
		if err := s.Read(bs.ragOut(), f32b(head.Logits[k*vocab:(k+n)*vocab])); err != nil {
			return err
		}
	}
	if hidden && head.Hidden != nil {
		if len(head.Hidden) < (k+n)*nembd {
			return fmt.Errorf("a %d-float hidden destination for %d rows of %d", len(head.Hidden), k+n, nembd)
		}
		return s.Read(bs.h, f32b(head.Hidden[k*nembd:(k+n)*nembd]))
	}
	return nil
}

// foldReqBuf holds the decode scratch's request for the folded head's row
// take (servedRowTake), once.
func (g *devTier) foldReqBuf(hb *blockScratch) error {
	if hb.foldReq != nil {
		return nil
	}
	w := g.scratchWin() // the scratch's own words (scratch.go)
	defer w.close()
	b, err := g.dev.Alloc(4 * (composer.RequestRows + 1))
	if err != nil {
		return err
	}
	hb.foldReq = b
	return nil
}

// foldWords is the folded head's request: the row a chunk's head reads.
func (g *devTier) foldWords(row int) []byte {
	b := g.subBytes.fold[:0]
	for range composer.RequestRows {
		b = binary.LittleEndian.AppendUint32(b, 0)
	}
	b = binary.LittleEndian.AppendUint32(b, uint32(row))
	g.subBytes.fold = b
	return b
}

// bindsRole reports whether graph g binds a weight of role r.
func bindsRole(g *graph.Graph, r jlm.Role) bool {
	if g == nil {
		return true
	}
	for _, o := range g.Ops {
		for _, b := range o.Weight {
			if b.Role == r {
				return true
			}
		}
	}
	return false
}

// gridded says plan p's rows turn by their picture's grid (servedGridTable)
// rather than by their positions.
func gridded(p *nn.LayerPlan) bool { return p.NonCausal && len(p.GridPlanes) > 0 }

// cutScratch makes bs's cut hold n bytes, a cut block's rows at a call
// (layer.cut): allocated at the first such call and grown at a wider one.
func (g *devTier) cutScratch(bs *blockScratch, n int) error {
	if bs.cutBytes >= n {
		return nil
	}
	if bs.cut != nil {
		bs.cut.Free()
		bs.cut, bs.cutBytes = nil, 0
	}
	b, err := g.dev.Alloc(n)
	if err != nil {
		g.NoRoom++
		return fmt.Errorf("tier: a cut block's rows: %w", err)
	}
	bs.cut, bs.cutBytes = b, n
	return nil
}
