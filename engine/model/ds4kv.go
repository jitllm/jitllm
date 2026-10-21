package model

import (
	"bytes"
	"errors"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// DeepSeek V4's compressed history (ds4.go): a compressed block keeps, beside
// the window's pages, a second paged history indexed by ENTRY -- entry w is
// the pool of positions [w*rate, (w+1)*rate) -- which grows one row every rate
// positions and is never released by the window, since every later query may
// read it.
//
// It is the block's own kvPages in kvCache.ent, built like a layer's: pages of
// entries (the kvPageTarget, or the model's pin), every sequence slot in a
// page, the value the key's memory (latent). Coordinates are explicit and
// never mixed: kvCache.layers is addressed by position, kvCache.ent by entry,
// and entry w exists once position (w+1)*rate-1 is written.

// initEnt gives every compressed block of kc its entry pages.
func (kc *kvCache) initEnt(c *Config, nseq int, l kvLayout, pin int) {
	if !c.DSV4() {
		return
	}
	kc.ent = make([]kvPages, len(kc.layers))
	kc.entRates = make([]int, len(kc.layers))
	p := align(kvPageTarget, kvKeyTile)
	if pin > 0 {
		p = pin
	}
	for li := kc.lo; li < len(kc.layers); li++ {
		w := c.EntRowAt(li)
		if w == 0 {
			continue
		}
		kc.entRates[li] = c.CompRateAt(li)
		e := &kc.ent[li]
		e.p, e.latent = p, true
		e.pp = nseq * p * w
	}
}

// entRate is how many positions one of block li's entries folds, zero where
// the block keeps none. Set by initEnt.
func (kc *kvCache) entRate(li int) int {
	if li >= len(kc.entRates) {
		return 0
	}
	return kc.entRates[li]
}

// entLayout is block li's entry layout: one row of EntRowAt floats per entry,
// always float32 (an entry is a pooled key, and the indexer selects over it).
func (s *State) entLayout(li int) kvLayout {
	l := s.kvl
	l.nKV, l.headDim, l.fmt, l.jit, l.headMajor = 1, s.m.Cfg.EntRowAt(li), cpu.KVF32, nil, false
	return l
}

// entWrite caches entry w of slot's history in block li.
func (s *State) entWrite(li, slot, w int, row []float32) {
	e := &s.kv.ent[li]
	e.write(s.entLayout(li), slot, w, row, row)
}

// entAt is entry w of slot's history in block li.
func (s *State) entAt(li, slot, w int) []float32 {
	k, _ := s.kv.ent[li].at(s.entLayout(li), slot, w, 0)
	return k
}

// The prefix cache over the entries (kvpage.go). Every block's rows slide, so
// the entries decide how much of a prompt restores: block li's entry page n
// holds entries [n*P, (n+1)*P), which positions [0, (n+1)*P*rate) wrote, and
// is named by those tokens (pageKeyP at P*rate positions); a partial page by
// its fill in positions, a multiple of the rate. seal stores whole pages as
// they fill, sealTail the partial one at every boundary of entChunk positions
// and at the prompt's end, and a restore takes the longest boundary every
// block's entries serve (entRestore). The rows' windows come back after it
// (restoreWin), as any windowed layer's do.

// entRestores says the entries decide a restore: a DeepSeek V4 cache, unless
// a gate walks the windowed rows from zero instead (walkWindow).
func (kc *kvCache) entRestores() bool { return kc.ent != nil && !kc.walkWindow }

// entStoreLayer is the store's layer for block li's entries, past the rows'
// (li*2), a recurrent summary's and the logits'.
func (kc *kvCache) entStoreLayer(li int) int { return 2 * (4*len(kc.layers) + li) }

// entChunk is a restore's boundary grain in positions: kvChunk, and a whole
// number of every block's entries.
func (kc *kvCache) entChunk() int {
	c := kvChunk
	for _, r := range kc.entRates {
		for r > 0 && c%r != 0 {
			c += kvChunk
		}
	}
	return c
}

// entSeal stores every entry page that the position has filled, once each.
func (kc *kvCache) entSeal(pos int) {
	for li := kc.lo; li < len(kc.ent); li++ {
		r := kc.entRate(li)
		if r == 0 {
			continue
		}
		e := &kc.ent[li]
		cur := min(e.written, pos/r) / e.p
		n := min(e.lastSealed, cur) // a rollback makes a page mutable again
		for ; n < cur; n++ {
			if n >= len(e.k) || e.k[n] == nil {
				break
			}
			key := kc.pageKeyP(e.p*r, n)
			if key == "" {
				break
			}
			if err := kc.store.Set(key, kc.entStoreLayer(li), n, bytes.NewReader(kvBytesOf(e.k[n]))); err != nil {
				kc.failed++
				break
			}
			kc.stored++
		}
		e.lastSealed = n
	}
}

// entSealTail stores each block's partial entry page at every entChunk
// boundary the position has passed inside it and at the position itself.
func (kc *kvCache) entSealTail(pos int) int {
	stored := 0
	c := kc.entChunk()
	for li := kc.lo; li < len(kc.ent); li++ {
		r := kc.entRate(li)
		if r == 0 {
			continue
		}
		e := &kc.ent[li]
		have := min(e.written, pos/r)
		cur := have / e.p
		if have -= cur * e.p; have <= 0 || cur >= len(e.k) || e.k[cur] == nil {
			continue
		}
		q, row := e.p*r, e.pp/e.p
		var fills []int
		for t := cur*q/c*c + c; t < pos; t += c {
			if f := t/r - cur*e.p; f > 0 && f < have {
				fills = append(fills, f)
			}
		}
		for _, f := range append(fills, have) {
			key := kc.pageKeyFillP(q, cur, f*r)
			if key == "" {
				continue
			}
			if err := kc.store.Set(key, kc.entStoreLayer(li), cur, bytes.NewReader(kvBytesOf(e.k[cur][:f*row]))); err != nil {
				kc.failed++
				continue
			}
			stored++
		}
	}
	return stored
}

// entFault brings block li's entry page n back from the store: whole, or its
// first fill entries.
func (kc *kvCache) entFault(li, n, fill int) error {
	e, r := &kc.ent[li], kc.entRate(li)
	if fill == 0 && e.resident(n) {
		return nil
	}
	key := kc.pageKeyP(e.p*r, n)
	if fill > 0 {
		key = kc.pageKeyFillP(e.p*r, n, fill*r)
	}
	if key == "" {
		return ErrNoPage
	}
	e.extend(n)
	e.drop(n)
	buf := make([]float32, e.pp)
	want := e.pp
	if fill > 0 {
		want = fill * (e.pp / e.p)
	}
	w := newSliceWriter(kvBytesOf(buf[:want]))
	if err := kc.store.Get(key, kc.entStoreLayer(li), n, w); err != nil {
		return kc.miss(key, err)
	}
	if w.n != len(w.b) {
		return kc.miss(key, shortPage(li, n, w.n, len(w.b)))
	}
	e.k[n], e.v[n] = buf, buf // the value is the key's memory (latent)
	kc.fetched++
	return nil
}

// entServe faults what every block's entries need for t positions -- their
// whole pages and the partial one -- and reports whether the store had all of
// it.
func (kc *kvCache) entServe(t int) (bool, error) {
	for li := kc.lo; li < len(kc.ent); li++ {
		r := kc.entRate(li)
		if r == 0 {
			continue
		}
		e := &kc.ent[li]
		full, fill := (t/r)/e.p, (t/r)%e.p
		for n := 0; n < full; n++ {
			if err := kc.entFault(li, n, 0); err != nil {
				return false, err
			}
		}
		if fill > 0 {
			if err := kc.entFault(li, full, fill); err != nil {
				return false, err
			}
		}
	}
	return true, nil
}

// entRestore is the longest boundary of this prompt every block's entries
// can serve -- its end, then each entChunk boundary down -- with their pages
// faulted, or 0.
func (kc *kvCache) entRestore() (int, error) {
	kc.entTruncate(0)
	l, c := len(kc.seq), kc.entChunk()
	ts := []int{l}
	for t := l / c * c; t >= c; t -= c {
		if t != l {
			ts = append(ts, t)
		}
	}
	for _, t := range ts {
		ok, err := kc.entServe(t)
		if errors.Is(err, ErrNoPage) {
			continue
		}
		if err != nil {
			return 0, err
		}
		if ok {
			kc.entTruncate(t)
			return t, nil
		}
	}
	kc.entTruncate(0)
	return 0, nil
}

// entTruncate keeps what n positions closed of every block's entries and
// drops the rest: whole pages are sealed, the page holding the last entry
// stays mutable.
func (kc *kvCache) entTruncate(n int) {
	for li := range kc.ent {
		r := kc.entRate(li)
		if r == 0 {
			continue
		}
		e := &kc.ent[li]
		have := n / r
		keep := (have + e.p - 1) / e.p
		for j := keep; j < len(e.k); j++ {
			e.drop(j)
		}
		e.lastSealed = have / e.p
		e.written = have
	}
}
