package maintier

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	kernels "github.com/jitllm/jitllm/paritytest/mainkernels"
)

// Attention over a history whose oldest pages are at home (kvevict.go).
//
// The history goes in passes of w pages. A pass over evicted pages first
// uploads them into w of the layer's free ids and points the sequence's table
// at them; every pass then runs the staged decode kernels over its keys alone
// (its descriptors clipped to them) into the scratch's partials, which fold
// into a running partial (kernels.FlashAttentionMergeRun); the last fold is
// finished, sinks and all, into the layer's output. It is the chunk-and-merge
// contract prefill's passes already keep. A prompt chunk streams the same way:
// every row has its own descriptor, so a pass clipped to its keys serves each
// row's causal end, and the evicted prefix goes up once a layer for the whole
// chunk rather than once a row.
//
// Uploads into ids a pass is still reading would race it, so a pass that
// uploads waits for the one before (Session.Sync). The pass width is every id
// the layers have free, so a pool with room streams in few passes.
//
// A streamed call re-uploads its whole evicted prefix every call (a decode
// token, a prompt chunk). That is the ceiling of streaming from the host; a
// host-side partial over the evicted prefix, merged on the device, is the
// upgrade if long contexts past the card need to be fast rather than
// possible.

// streamSet is what a pass width compiles once: the staged variant for w
// pages of keys and the folds for its partials.
type streamSet struct {
	w                   int
	v                   *pagedVariant
	first, fold, finish backend.Kernel
	run                 [2]backend.Buf
	foldT, finT         int
}

func (ss *streamSet) close() {
	for _, k := range []backend.Kernel{ss.first, ss.fold, ss.finish} {
		if k != nil {
			closeKern(k)
		}
	}
	for _, b := range ss.run {
		if b != nil {
			b.Free()
		}
	}
}

// pagedStream is one call's streaming: the sequence, its evicted page count,
// and a descriptor set per pass for the full and the windowed history.
type pagedStream struct {
	g       *devTier
	set     *streamSet
	s       seqID
	evicted int
	passes  int
	desc    [2][]backend.Buf
	// words is each set's descriptors, pass after pass, for desc.
	words [2][]uint32
	ids   []uint32
}

// writes stages every pass's descriptors with the call's other writes.
func (st *pagedStream) writes(w func(backend.Buf, []byte)) {
	for i, bufs := range st.desc {
		if len(bufs) == 0 {
			continue
		}
		n := len(st.words[i]) / len(bufs)
		for j, b := range bufs {
			w(b, u32view(st.words[i][j*n:(j+1)*n]))
		}
	}
}

// pagedStreamPrep readies the call's streaming, or leaves pg.st nil when no
// row reads an evicted page and every row's keys fit the call's plan. The
// passes reach the furthest live row, so every row's keys are covered
// whichever sequence it belongs to; only one sequence has pages at home
// (evictVictim). A call whose staged plan was built for a pass
// (stagedPassKeys) and whose deepest row reads past it goes in passes too,
// over pages already on the card: nothing is uploaded and nothing waits.
// prefill says the call runs the prefill kernels, which pass over a long
// history on their own (pagedprefill.go). Callers hold g.mu, outside any
// submission; it compiles and allocates.
func (g *devTier) pagedStreamPrep(sid uint64, bs *blockScratch, rows []pagedRow, prefill bool) error {
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	pg := bs.pkv
	pg.st = nil
	kp := g.kvPages()
	var s seqID
	far, over := 0, false
	for _, r := range rows {
		if r.pad {
			continue
		}
		far = max(far, r.pos)
		if kp.evicted[r.s] > 0 {
			if over && r.s != s {
				return fmt.Errorf("two sequences with pages at home in one call: %v and %v", s, r.s)
			}
			s, over = r.s, true
		}
	}
	pass := g.stagedPassKeys(bs)
	deep := pass > 0 && far >= pass && !prefill && pg.cur != nil && pg.cur.plan.Path == kernels.PathStaged
	if !over && !deep {
		return nil
	}
	e, w := 0, pass/kp.p
	if over {
		e = kp.evicted[s]
		if !deep || e < w {
			w = e
		}
		if err := g.heldLayers(sid, func(li int, l *kvLayerPool) error {
			w = min(w, g.streamWidth(l, e))
			// Every layer holding s holds the same prefix at home (kvevict.go);
			// one that does not would have the passes read past its copies.
			if len(l.owned[s]) > 0 && len(l.home[s]) != e {
				return fmt.Errorf("block %d holds %d page(s) of sequence %v at home and the pool counts %d",
					li, len(l.home[s]), s, e)
			}
			return nil
		}); err != nil {
			return err
		}
		if w < 1 {
			return fmt.Errorf("%w: no free page to stream %d evicted page(s) through", ErrKVCapacity, e)
		}
	}
	// A power of two, so the widths a pool passes through compile a handful
	// of kernel sets rather than one per free-list length.
	for w&(w-1) != 0 {
		w &= w - 1
	}
	set, err := g.streamSetFor(bs, w)
	if err != nil {
		return err
	}
	st := &pg.stream
	*st = pagedStream{g: g, set: set, s: s, evicted: e, ids: st.ids[:0]}
	keys := w * kp.p
	st.passes = (far + keys) / keys
	for i, src := range [2][]uint32{pg.desc, pg.descW} {
		if len(src) == 0 {
			continue
		}
		for len(pg.streamDesc[i]) < st.passes {
			b, err := g.dev.Alloc(len(src) * 4)
			if err != nil {
				return err
			}
			pg.streamDesc[i] = append(pg.streamDesc[i], b)
		}
		all := slices.Grow(pg.streamD[i][:0], st.passes*len(src))[:st.passes*len(src)]
		pg.streamD[i] = all
		for j := 0; j < st.passes; j++ {
			lo, hi := uint32(j*keys), uint32((j+1)*keys)
			d := all[j*len(src) : (j+1)*len(src)]
			copy(d, src)
			for o := 0; o < len(d); o += kernels.PRowWords {
				d[o+kernels.PRowStart] = min(max(d[o+kernels.PRowStart], lo), hi)
				d[o+kernels.PRowEnd] = min(max(d[o+kernels.PRowEnd], lo), hi)
			}
		}
		st.desc[i], st.words[i] = pg.streamDesc[i][:st.passes], all
	}
	pg.st = st
	return nil
}

// streamSetFor is bs's pass set for w pages, compiled once.
func (g *devTier) streamSetFor(bs *blockScratch, w int) (*streamSet, error) {
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	pg := bs.pkv
	if ss := pg.streams[w]; ss != nil {
		return ss, nil
	}
	shape := kernels.PagedStagedPlan(pg.shape, w*g.kvPages().p, g.dev.Slots())
	v, err := g.pagedBuild(bs, kernels.DecodePlan{Path: kernels.PathStaged, Shape: shape})
	if err != nil {
		return nil, err
	}
	ss := &streamSet{w: w, v: v}
	// The folds carry no sink: it is added once, by the finish.
	ms := mergeShape(v.plan.Shape)
	one := ms
	one.Splits = 1
	ms.Sink = false
	comp := func(k *ir.Kernel, err error) backend.Kernel {
		if err == nil {
			var bk backend.Kernel
			if bk, err = g.dev.Compile(k); err == nil {
				return bk
			}
		}
		g.LastErr = err.Error()
		return nil
	}
	ss.first = comp(kernels.FlashAttentionMergeRun(ms, true))
	ss.fold = comp(kernels.FlashAttentionMergeRun(ms, false))
	ss.finish = comp(kernels.FlashAttentionMergeWide(one))
	if ss.first == nil || ss.fold == nil || ss.finish == nil {
		ss.close()
		return nil, fmt.Errorf("streamed attention's folds: %s", g.LastErr)
	}
	for i := range ss.run {
		b, err := g.dev.Alloc(kernels.FlashPartialFloats(one) * 4)
		if err != nil {
			ss.close()
			return nil, err
		}
		ss.run[i] = b
	}
	ss.foldT = ms.Rows * ms.Heads * ((ms.Dim + 31) / 32 * 32)
	ss.finT = ms.Rows * ms.Heads * ms.Dim
	if pg.streams == nil {
		pg.streams = map[int]*streamSet{}
	}
	pg.streams[w] = ss
	return ss, nil
}

// attend is one layer's decode attention for the call: streamed when a row
// reads evicted pages (pg.st), else pagedDecode. pl is the layer's pool; desc
// is pRow or pRowW, the windowed set.
func (pg *pagedScratch) attend(s backend.Session, lc *launcher, pl *kvLayerPool,
	q, k, v, n, out, tab, desc, sink backend.Buf) error {
	if pg.st != nil {
		return pg.streamDecode(s, lc, pl, q, k, v, n, out, tab, desc == pg.pRowW, sink)
	}
	return pg.pagedDecode(s, lc, q, k, v, n, out, tab, desc, sink)
}

// streamDecode is pagedDecode over a history with evicted pages, for layer
// pool l: win selects the windowed descriptors. Callers hold g.mu, inside the
// submission's session.
func (pg *pagedScratch) streamDecode(s backend.Session, lc *launcher,
	l *kvLayerPool, q, k, v, n, out, tab backend.Buf, win bool, sink backend.Buf) error {
	st := pg.st
	g, ss, vr := st.g, st.set, st.set.v
	kp := g.kvPages()
	di := 0
	if win {
		di = 1
	}
	kw, vw := l.geom.kWords(kp.p), l.geom.vWords(kp.p)
	off := kp.ranges[st.s].off
	home := l.home[st.s]
	for j := 0; j < st.passes; j++ {
		p0, p1 := j*ss.w, min((j+1)*ss.w, st.evicted)
		if p0 < p1 {
			// The ids the pass before read are this pass's: wait for it.
			if j > 0 {
				if err := s.Sync(); err != nil {
					return err
				}
			}
			st.ids = append(st.ids[:0], l.free[:p1-p0]...)
			for i, id := range st.ids {
				if pagedFaulted("stream") {
					break // the violation: the pass reads whatever the ids held
				}
				page := home[p0+i]
				if page == nil {
					// Released behind a windowed layer's window: every row's
					// window starts past it, so the pass reads none of it.
					continue
				}
				if err := s.WriteAt(k, int(id)*kw*4, f32b(page[:kw])); err != nil {
					return err
				}
				if vw > 0 {
					if err := s.WriteAt(v, int(id)*vw*4, f32b(page[kw:])); err != nil {
						return err
					}
				}
			}
			if err := s.WriteAt(tab, (off+p0)*4, u32view(st.ids)); err != nil {
				return err
			}
			g.KVStreamPasses++
		}
		g.StagedPasses++
		desc := st.desc[di][j]
		lc.la(vr.scores, vr.scoreT, q, k, n, pg.sc, tab, desc)
		if err := lc.launch(vr.soft, vr.softG, 64, pg.sc, n, pg.wt, pg.part, tab, desc); err != nil {
			return err
		}
		lc.la(vr.acc, vr.accT, pg.wt, v, n, pg.part, tab, desc)
		if j == 0 {
			lc.la(ss.first, ss.foldT, pg.part, ss.run[0])
		} else if !pagedFaulted("passfold") {
			lc.la(ss.fold, ss.foldT, pg.part, ss.run[(j+1)%2], ss.run[j%2])
		}
	}
	if sink != nil {
		lc.la(ss.finish, ss.finT, ss.run[(st.passes+1)%2], out, sink)
	} else {
		lc.la(ss.finish, ss.finT, ss.run[(st.passes+1)%2], out)
	}
	return nil
}

// devStagedPlane is the plane Config.StagedPassKeys' default is sized to:
// a scratch's score plane, rows x heads x keys floats, at most 64 MiB (its
// weight plane is the same again). A 512-row chunk of 32 heads passes 1024
// keys at a time, a 64-row step 8192; a plan for a 131072-position context
// was 8 GiB a plane at 256 rows.
const devStagedPlane = 64 << 20

// stagedPassKeys is the most keys bs's staged plan is built for, in whole
// pages, or 0 for no bound: a block whose attention reads the whole history
// through the table at once (the lightning indexer, MiniMax-M3's block
// selection, DeepSeek V4's compressed entries) does not go in passes.
// Callers hold g.mu.
func (g *devTier) stagedPassKeys(bs *blockScratch) int {
	pg := bs.pkv
	if pg == nil || pg.idx != nil || pg.msa != nil || pg.qsa != nil || bs.ds4 != nil {
		return 0
	}
	kp := g.kvPages()
	if kp == nil || kp.p <= 0 {
		return 0
	}
	n := g.StagedPassKeys
	if n <= 0 {
		n = devStagedPlane / 4 / max(pg.shape.Rows*pg.shape.Heads, 1)
	}
	return max(n/kp.p, 1) * kp.p
}
