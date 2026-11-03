package gpu

import (
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
)

// Attention over a history whose oldest pages are in host memory (kvevict.go).
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

// pagedStream is one call's streaming: the sequence, its evicted page count,
// and a descriptor set per pass for the full and the windowed history.
type pagedStream struct {
	g       *devTier
	s       seqID
	evicted int
	// passes is how many, each over w pages.
	passes, w int
	desc      [2][]backend.Buf
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
// row reads an evicted page. The passes reach the furthest live row, so every
// row's keys are covered whichever sequence it belongs to; only one sequence
// has pages in host memory (evictVictim). Callers hold g.mu, outside any
// submission; it allocates.
func (g *devTier) pagedStreamPrep(sid uint64, bs *blockScratch, rows []pagedRow) error {
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
				return fmt.Errorf("two sequences with pages in host memory in one call: %v and %v", s, r.s)
			}
			s, over = r.s, true
		}
	}
	if !over {
		return nil
	}
	e, w := kp.evicted[s], g.stagedPassKeys(bs)/kp.p
	// No pass wider than the pass width: the composer's form over passes
	// is formed for that many keys (passKeys).
	if w == 0 || e < w {
		w = e
	}
	if err := g.heldLayers(sid, func(li int, l *kvLayerPool) error {
		w = min(w, g.streamWidth(l, e))
		// Every layer holding s holds the same prefix in host memory (kvevict.go);
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
	st := &pg.stream
	*st = pagedStream{g: g, s: s, evicted: e, w: w, ids: st.ids[:0]}
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
			for o := 0; o < len(d); o += composer.RowWords {
				d[o+composer.RowStart] = min(max(d[o+composer.RowStart], lo), hi)
				d[o+composer.RowEnd] = min(max(d[o+composer.RowEnd], lo), hi)
			}
		}
		st.desc[i], st.words[i] = pg.streamDesc[i][:st.passes], all
	}
	pg.st = st
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
// selection) does not go in passes.
// Callers hold g.mu.
func (g *devTier) stagedPassKeys(bs *blockScratch) int {
	pg := bs.pkv
	if pg == nil || selectsHistory(&bs.p) {
		return 0
	}
	kp := g.kvPages()
	if kp == nil || kp.p <= 0 {
		return 0
	}
	n := g.StagedPassKeys
	if n <= 0 {
		n = devStagedPlane / 4 / max(bs.rows*bs.p.NHead, 1)
	}
	return max(n/kp.p, 1) * kp.p
}

// passKeys is the key bound of bs's form over passes, as of a history of
// maxSeq positions: a pass is at most the pass width (pagedStreamPrep), so
// the form's planes are its rows by heads by that, not by the whole history.
func (g *devTier) passKeys(bs *blockScratch, maxSeq int) int {
	if k := g.stagedPassKeys(bs); k > 0 && k < maxSeq {
		return k
	}
	return maxSeq
}

// inPasses says block l of this call reads a history with evicted pages, so
// the composer serves it with its form over passes (servedStep).
func (bs *blockScratch) inPasses(l *layer) bool {
	pk := bs.pkv
	return pk != nil && pk.st != nil && pk.st.evicted > 0 && (!l.linear || l.withAttn)
}

// runBufs is bs's running partial's buffers for block li's form over
// passes, grown to its rows' heads: the zeros the first pass folds into,
// then the folds' two sides. Callers hold g.mu, outside any submission.
func (g *devTier) runBufs(bs *blockScratch, l *layer, li int) ([]backend.Buf, error) {
	pg, n := bs.pkv, 0
	if l.graph == nil {
		return nil, fmt.Errorf("block %d has no graph to read its history in passes", li)
	}
	for _, o := range l.graph.Ops {
		if o.Kind == graph.OpAttend {
			n = max(n, bs.rows*o.Attr.NHead*(o.Attr.HeadDim+2))
		}
	}
	if n == 0 {
		return nil, fmt.Errorf("block %d reads a history it attends none of", li)
	}
	if pg.runN >= n {
		return pg.run[:2], nil
	}
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	for i, b := range pg.run {
		if b != nil {
			b.Free()
		}
		pg.run[i] = nil
	}
	pg.runN = 0
	for i := range pg.run {
		b, err := g.dev.Alloc(n * 4)
		if err != nil {
			return nil, err
		}
		pg.run[i] = b
	}
	if err := pg.run[0].Write(make([]byte, n*4)); err != nil {
		return nil, err
	}
	pg.runN = n
	return pg.run[:2], nil
}

// streamRun launches a form over a history read in passes: its codes before
// the passes, then each pass's codes over the keys the pass holds -- the
// evicted pages it covers first uploaded into the layer's free ids and the
// table pointed at them, as streamDecode stages them, the rows' descriptors
// clipped to the pass and the running partial folded from one side into the
// other -- then the codes after, which finish from the last fold. pl is the
// pool the form reads. Callers hold g.mu, inside the submission's session.
func (sv *servedPart) streamRun(g *devTier, lc *launcher, s backend.Session, inst *partInst, kind partKind, bs *blockScratch,
	pl *kvLayerPool, in, out []backend.Buf) error {
	f, pg := sv.f, bs.pkv
	st, t := pg.st, sv.sl.tables
	copy(sv.in[:sv.n], in)
	copy(sv.out[:sv.m], out)
	in, out = sv.in[:sv.n], sv.out[:sv.m]
	rows, rowsW := t[tableKey(composer.TableRows)], t[tableKey(composer.TableWindows)]
	defer func() {
		t[tableKey(composer.TableRows)], t[tableKey(composer.TableWindows)], t[tableKey(composer.TableRunIn)], t[tableKey(composer.TableRunOut)] = rows, rowsW, pg.run[0], pg.run[1]
	}()
	m0, err := sv.launchCodes(lc, inst, in, out, 0, f.passLo)
	if err != nil {
		return err
	}
	kp := g.kvPages()
	kw, vw := pl.geom.kWords(kp.p), pl.geom.vWords(kp.p)
	off := kp.ranges[st.s].off
	home := pl.home[st.s]
	for j := 0; j < st.passes; j++ {
		if p0, p1 := j*st.w, min((j+1)*st.w, st.evicted); p0 < p1 {
			// The ids the pass before read are this pass's: wait for it.
			if j > 0 {
				if err := s.Sync(); err != nil {
					return err
				}
			}
			st.ids = append(st.ids[:0], pl.free[:p1-p0]...)
			for i, id := range st.ids {
				if g.pagedFaulted("stream") {
					break // the violation: the pass reads whatever the ids held
				}
				page := home[p0+i]
				if page == nil {
					continue // released behind a window every row starts past
				}
				if err := s.WriteAt(pl.k, int(id)*kw*4, f32b(page[:kw])); err != nil {
					return err
				}
				if vw > 0 {
					if err := s.WriteAt(pl.v, int(id)*vw*4, f32b(page[kw:])); err != nil {
						return err
					}
				}
			}
			if err := s.WriteAt(pl.tab, (off+p0)*4, u32view(st.ids)); err != nil {
				return err
			}
			g.KVStreamPasses++
		}
		t[tableKey(composer.TableRows)] = st.desc[0][j]
		t[tableKey(composer.TableWindows)] = t[tableKey(composer.TableRows)]
		if rowsW == pg.pRowW && len(st.desc[1]) > 0 {
			t[tableKey(composer.TableWindows)] = st.desc[1][j]
		}
		t[tableKey(composer.TableRunIn)], t[tableKey(composer.TableRunOut)] = pg.run[0], pg.run[1+j%2]
		if j > 0 && !g.pagedFaulted("passfold") {
			t[tableKey(composer.TableRunIn)] = pg.run[1+(j+1)%2]
		}
		m, err := sv.launchCodes(lc, inst, in, out, f.passLo, f.passHi)
		if err != nil {
			return err
		}
		m0.matrix, m0.attend = m0.matrix+m.matrix, m0.attend+m.attend
		g.ComposedStreamPasses++
	}
	t[tableKey(composer.TableRows)], t[tableKey(composer.TableWindows)] = rows, rowsW
	m, err := sv.launchCodes(lc, inst, in, out, f.passHi, len(f.codes))
	if err != nil {
		return err
	}
	g.PartsServed[kind]++
	g.MatrixLaunches[kind] += m0.matrix + m.matrix
	g.AttendLaunches[kind] += m0.attend + m.attend
	return nil
}
