package gpu

import (
	"encoding/binary"
	"fmt"
	"slices"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
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

// pagedStream is one call's streaming: the sequences with pages in host
// memory, the most pages any has there, each pass's upload, and a descriptor
// set per pass for the full and the windowed history.
type pagedStream struct {
	g       *devTier
	seqs    []streamSeq
	evicted int
	// passes is how many: first a pass a run of at most w of one sequence's
	// evicted pages (up), then the resident history w pages at a time.
	passes, w int
	up        []streamUp
	desc      [2][]backend.Buf
	// words is each set's descriptors, pass after pass, for desc.
	words [2][]uint32
	ids   []uint32
}

// streamSeq is a sequence of a streamed call and its pages in host memory.
type streamSeq struct {
	s       seqID
	evicted int
}

// streamUp is a pass's upload: sequence seqs[k]'s evicted pages p0 to p1, or
// for k < 0 none -- a pass over the resident history from position p0.
type streamUp struct{ k, p0, p1 int }

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
// row reads an evicted page. Each sequence with pages in host memory streams
// them in passes of its own -- its rows' keys clipped to the pass, every
// other row attending nothing in it -- through the same free ids, one pass
// after another; then the resident history goes in passes reaching the
// furthest live row, each row's keys clipped past its sequence's evicted
// prefix. Callers hold g.mu, outside any submission; it allocates.
func (g *devTier) pagedStreamPrep(sid uint64, bs *blockScratch, rows []pagedRow) error {
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	pg := bs.pkv
	pg.st = nil
	kp := g.kvPages()
	st := &pg.stream
	seqs := st.seqs[:0]
	far, e := 0, 0
	for _, r := range rows {
		if r.pad {
			continue
		}
		far = max(far, r.pos)
		n := kp.evicted[r.s]
		if n == 0 {
			continue
		}
		known := false
		for _, q := range seqs {
			known = known || q.s == r.s
		}
		if !known {
			seqs, e = append(seqs, streamSeq{s: r.s, evicted: n}), max(e, n)
		}
	}
	st.seqs = seqs
	if len(seqs) == 0 {
		return nil
	}
	w := g.stagedPassKeys(bs) / kp.p
	// No pass wider than the pass width: the composer's form over passes
	// is formed for that many keys (passKeys).
	if w == 0 || e < w {
		w = e
	}
	if err := g.heldLayers(sid, func(li int, l *kvLayerPool) error {
		w = min(w, g.streamWidth(l, e))
		// Every layer holding s holds the same prefix in host memory (kvevict.go);
		// one that does not would have the passes read past its copies.
		for _, q := range seqs {
			if len(l.owned[q.s]) > 0 && len(l.home[q.s]) != q.evicted {
				return fmt.Errorf("block %d holds %d page(s) of sequence %v at home and the pool counts %d",
					li, len(l.home[q.s]), q.s, q.evicted)
			}
		}
		return nil
	}); err != nil {
		return err
	}
	if w < 1 {
		return fmt.Errorf("%w: no free page to stream %d evicted page(s) through", ErrKVCapacity, e)
	}
	keys := w * kp.p
	up := st.up[:0]
	for k, q := range seqs {
		for p0 := 0; p0 < q.evicted; p0 += w {
			up = append(up, streamUp{k: k, p0: p0, p1: min(p0+w, q.evicted)})
		}
	}
	// The resident passes start at the lowest position a row reads past its
	// sequence's evicted prefix.
	lo0 := far + 1
	for _, r := range rows {
		if !r.pad {
			lo0 = min(lo0, kp.evicted[r.s]*kp.p)
		}
	}
	for lo := lo0 / keys * keys; lo <= far; lo += keys {
		up = append(up, streamUp{k: -1, p0: lo})
	}
	st.g, st.evicted, st.w, st.up, st.passes, st.ids = g, e, w, up, len(up), st.ids[:0]
	if len(seqs) > 1 {
		g.KVStreamJoint++
	}
	for i, src := range [2][]uint32{pg.desc, pg.descW} {
		st.desc[i], st.words[i] = nil, nil
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
		for j, u := range up {
			d := all[j*len(src) : (j+1)*len(src)]
			copy(d, src)
			for o, ri := 0, 0; o < len(d); o, ri = o+composer.RowWords, ri+1 {
				var lo, hi uint32
				switch {
				case u.k >= 0:
					lo, hi = uint32(u.p0*kp.p), uint32(u.p1*kp.p)
					if ri >= len(rows) || rows[ri].pad || rows[ri].s != seqs[u.k].s {
						hi = lo // another sequence's row attends nothing here
					}
				default:
					lo, hi = uint32(u.p0), uint32(u.p0+keys)
					if ri < len(rows) && !rows[ri].pad {
						lo = min(max(lo, uint32(kp.evicted[rows[ri].s]*kp.p)), hi)
					}
				}
				s0 := d[o+composer.RowStart]
				s, e := min(max(s0, lo), hi), min(max(d[o+composer.RowEnd], lo), hi)
				if e <= s {
					// Nothing of the row in the pass: an empty range at its
					// own first position, so what a form reads for it,
					// clamped, is of its own pages.
					s, e = s0, s0
				}
				d[o+composer.RowStart], d[o+composer.RowEnd] = s, e
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
// pages, or 0 for no bound: a pooled block selection (MiniMax-M3's), which
// reads the whole history through the table at once, does not go in passes.
// Callers hold g.mu.
func (g *devTier) stagedPassKeys(bs *blockScratch) int {
	pg := bs.pkv
	if pg == nil || pooledSelects(bs.p.Graph) {
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
	if err := g.selBufs(bs, l); err != nil {
		return nil, err
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

// selBufs grows bs's running block scores to block l's selections over its
// whole history (composer.TableSelIn, TableSelOut): sel[0] the least floats
// the first pass folds into, sel[1] and sel[2] the folds' two sides. Callers
// hold g.mu, outside any submission.
func (g *devTier) selBufs(bs *blockScratch, l *layer) error {
	pg, n, whole := bs.pkv, 0, keyBound(bs.p.MaxSeq)
	for _, o := range l.graph.Ops {
		if o.Kind == graph.OpSelect && !o.Attr.Pooled && o.Attr.Span > 0 {
			n = max(n, bs.rows*o.Attr.NKVHead*((whole+o.Attr.Span-1)/o.Attr.Span))
		}
	}
	if n == 0 || pg.selN >= n {
		return nil
	}
	defer g.scratchWin().close() // the scratch's own buffers (scratch.go)
	for i, b := range pg.sel {
		if b != nil {
			b.Free()
		}
		pg.sel[i] = nil
	}
	pg.selN = 0
	for i := range pg.sel {
		b, err := g.dev.Alloc(n * 4)
		if err != nil {
			return err
		}
		pg.sel[i] = b
	}
	least := make([]byte, n*4)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint32(least[4*i:], 0xff7fffff)
	}
	if err := pg.sel[0].Write(least); err != nil {
		return err
	}
	pg.selN = n
	return nil
}

// streamRun launches a form over a history read in passes: its codes before
// the passes, then each sweep's (a selection's, then its attention's): each
// pass's codes over the keys the pass holds -- the evicted pages it covers
// first uploaded into the layer's free ids and the table pointed at them, as
// streamDecode stages them, the rows' descriptors clipped to the pass and
// the sweep's running partial folded from one side into the other -- and
// between two sweeps, and after the last, the codes that read the folds once
// over the whole history. pl is the pool the form reads. Callers hold g.mu,
// inside the submission's session.
func (sv *servedPart) streamRun(g *devTier, lc *launcher, s backend.Session, inst *partInst, kind nn.Part, bs *blockScratch,
	pl *kvLayerPool, in, out []backend.Buf) error {
	f, pg := sv.f, bs.pkv
	st, t := pg.st, sv.sl.tables
	copy(sv.in[:sv.n], in)
	copy(sv.out[:sv.m], out)
	in, out = sv.in[:sv.n], sv.out[:sv.m]
	rows, rowsW := t[tableKey(composer.TableRows)], t[tableKey(composer.TableWindows)]
	_, selects := t[tableKey(composer.TableSelIn)]
	defer func() {
		t[tableKey(composer.TableRows)], t[tableKey(composer.TableWindows)], t[tableKey(composer.TableRunIn)], t[tableKey(composer.TableRunOut)] = rows, rowsW, pg.run[0], pg.run[1]
		if selects {
			t[tableKey(composer.TableSelIn)], t[tableKey(composer.TableSelOut)] = pg.sel[0], pg.sel[1]
		}
	}()
	m0, err := sv.launchCodes(lc, inst, in, out, 0, f.sweeps[0].lo)
	if err != nil {
		return err
	}
	kp := g.kvPages()
	kw, vw := pl.geom.kWords(kp.p), pl.geom.vWords(kp.p)
	at := 0
	for si, sw := range f.sweeps {
		if si > 0 {
			t[tableKey(composer.TableRows)], t[tableKey(composer.TableWindows)] = rows, rowsW
			m, err := sv.launchCodes(lc, inst, in, out, at, sw.lo)
			if err != nil {
				return err
			}
			m0.matrix, m0.attend = m0.matrix+m.matrix, m0.attend+m.attend
		}
		side, inKey, outKey := pg.run, tableKey(composer.TableRunIn), tableKey(composer.TableRunOut)
		if sw.folds == composer.TableSelOut {
			side, inKey, outKey = pg.sel, tableKey(composer.TableSelIn), tableKey(composer.TableSelOut)
		}
		for j := 0; j < st.passes; j++ {
			if u := st.up[j]; u.k >= 0 {
				q, p0, p1 := st.seqs[u.k], u.p0, u.p1
				home := pl.home[q.s]
				// The ids the pass before read -- this sweep's, or the sweep
				// before's last -- are this pass's: wait for it.
				if j > 0 || si > 0 {
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
				if err := s.WriteAt(pl.tab, (kp.ranges[q.s].off+p0)*4, u32view(st.ids)); err != nil {
					return err
				}
				g.KVStreamPasses++
			}
			t[tableKey(composer.TableRows)] = st.desc[0][j]
			t[tableKey(composer.TableWindows)] = t[tableKey(composer.TableRows)]
			if rowsW == pg.pRowW && len(st.desc[1]) > 0 {
				t[tableKey(composer.TableWindows)] = st.desc[1][j]
			}
			t[inKey], t[outKey] = side[0], side[1+j%2]
			if j > 0 && !g.pagedFaulted("passfold") && !(sw.folds == composer.TableSelOut && g.pagedFaulted("selfold")) {
				t[inKey] = side[1+(j+1)%2]
			}
			m, err := sv.launchCodes(lc, inst, in, out, sw.lo, sw.hi)
			if err != nil {
				return err
			}
			m0.matrix, m0.attend = m0.matrix+m.matrix, m0.attend+m.attend
			g.ComposedStreamPasses++
		}
		// What reads the sweep's fold reads its last.
		last := side[1+(st.passes-1)%2]
		t[inKey], t[outKey] = last, last
		at = sw.hi
	}
	t[tableKey(composer.TableRows)], t[tableKey(composer.TableWindows)] = rows, rowsW
	m, err := sv.launchCodes(lc, inst, in, out, at, len(f.codes))
	if err != nil {
		return err
	}
	g.Steps.Serve(kind)
	g.MatrixLaunches[kind] += m0.matrix + m.matrix
	g.AttendLaunches[kind] += m0.attend + m.attend
	return nil
}
