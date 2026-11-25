package gpu

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// A history's pages arrive and leave as floats (nn.LayerDevice.MigrateKV)
// wherever the pool holds other words: a typed history's keys and values (kvGeom.pool: q8 rows, GGUF blocks,
// bfloat16 halves). The card narrows and widens them itself, formed through
// the composer like any other part (composer.Pool.ConvOps): a page's floats
// staged and narrowed into the pool, a page's words widened before they are
// read back.

// convKey is one of the conversions: rows of w floats narrowed into pool's
// words, or the words widened where widen is set.
type convKey struct {
	pool  composer.Pool
	w     int
	widen bool
}

// convPage is the conversion of a page: rows rows of w floats in pool's
// words.
type convPage struct {
	pool    composer.Pool
	w, rows int
}

// convOf is the conversion geometry k's pages of p positions take, keys and
// values alike (a typed history's, its keys by position), ok false where its
// pages hold floats.
func (k kvGeom) convOf(p int) (c convPage, ok bool) {
	switch {
	case k.typed():
		w := k.kvRow
		if k.pool.Row > 0 {
			w = k.pool.Row
		}
		return convPage{k.pool, w, k.slots(p) * k.kvRow / w}, true
	}
	return convPage{}, false
}

// servedConv is c's rows narrowed into its pool's words, or the words
// widened back where widen is set.
func (g *devTier) servedConv(c convPage, widen bool) (*servedPart, error) {
	key := convKey{c.pool, c.w, widen}
	gr := g.convGraphs[key]
	if gr == nil {
		narrow, wide, at := c.pool.ConvOps()
		hw := c.pool.Words(c.w)
		b := graph.New()
		if widen {
			h := b.In("kv.h", hw)
			b.G.Values[h].Elem = graph.ElemU32
			b.Out(b.Emit(wide, []graph.ValueID{h}, []graph.Result{{Name: "kv.f", Width: c.w}}, nil, at, nil, nil)...)
		} else {
			b.Out(b.Emit(narrow, []graph.ValueID{b.In("kv.f", c.w)},
				[]graph.Result{{Name: "kv.h", Width: hw, Elem: graph.ElemU32}}, nil, at, nil, nil)...)
		}
		gr = b.G
		if g.convGraphs == nil {
			g.convGraphs = map[convKey]*graph.Graph{}
		}
		g.convGraphs[key] = gr
	}
	return g.servedFor(servedKey{kind: nn.PartHalves, graph: gr, rows: c.rows}, func() (composedPart, error) {
		return composedPart{kind: nn.PartHalves, graph: gr, apart: true}, nil
	})
}

// pageConv is a page's conversions: a page's floats and its words staged
// on the card, and the forms between them. words is a page's words read to
// host memory, and sv, in, out, err and fn a conversion's submission, kept
// so that one run a token allocates nothing.
type pageConv struct {
	g              *devTier
	f, h           backend.Buf
	narrow, widen  *servedPart
	fBytes, hBytes int
	words          []float32
	sv             *servedPart
	in, out        [1]backend.Buf
	err            error
	fn             func(backend.Session)
}

func newConvSizes(c convPage) *pageConv {
	return &pageConv{fBytes: c.rows * c.w * 4, hBytes: c.rows * c.pool.Words(c.w) * 4}
}

// stage makes the page's staging on the card.
func (c *pageConv) stage() error {
	var err error
	if c.f, err = c.g.alloc(c.fBytes); err != nil {
		return err
	}
	if c.h, err = c.g.alloc(c.hBytes); err != nil {
		c.f.Free()
		c.f = nil
		return err
	}
	return nil
}

func (c *pageConv) close() {
	if c.f != nil {
		c.f.Free()
		c.h.Free()
		c.f, c.h = nil, nil
	}
}

// formConv forms, at a placement, the page conversion c a history copy or
// a move runs (keptConv): the device's, held on a lane of its own and run under
// g.mu, its staging made at its first run. Callers hold g.mu.
func (g *devTier) formConv(c convPage) error {
	if g.convs[c] != nil {
		return nil
	}
	v := &devTier{devShared: g.devShared, devSess: g.dv.devSess, lane: &lane{}, Stats: g.dv.Stats}
	pc := newConvSizes(c)
	pc.g = v
	was := g.prewarming
	g.prewarming = true
	var err error
	if pc.narrow, err = v.servedConv(c, false); err == nil {
		pc.widen, err = v.servedConv(c, true)
	}
	g.prewarming = was
	if err != nil {
		v.dropLaneParts(v.lane)
		return err
	}
	if g.convs == nil {
		g.convs = map[convPage]*pageConv{}
	}
	g.convs[c] = pc
	return nil
}

// formGeomConv is formConv for the pages of geometry k, nothing for a
// geometry of floats. Callers hold g.mu.
func (g *devTier) formGeomConv(k kvGeom, p int) error {
	if c, ok := k.convOf(p); ok {
		return g.formConv(c)
	}
	return nil
}

// keptConv is the conversion formConv formed, staged at the first call
// and charged to the pool as the device's own buffers (scratchWin).
// Callers hold g.mu.
func (g *devTier) keptConv(c convPage) (*pageConv, error) {
	pc := g.convs[c]
	if pc == nil {
		return nil, fmt.Errorf("a history page's conversion of %d rows of %d floats into %+v was not formed at placement", c.rows, c.w, c.pool)
	}
	if pc.f == nil {
		win := g.scratchWin()
		err := pc.stage()
		win.close()
		if err != nil {
			return nil, err
		}
	}
	return pc, nil
}

// freeConvs frees what formConv formed, its charge given back. Callers
// hold g.mu.
func (g *devTier) freeConvs() {
	defer g.scratchWin().close()
	for key, c := range g.convs {
		c.close()
		c.g.dropLaneParts(c.g.lane)
		delete(g.convs, key)
	}
}

// run runs sv from in into out in a submission of its own.
func (c *pageConv) run(sv *servedPart, in, out backend.Buf) error {
	if c.fn == nil {
		c.fn = func(s backend.Session) {
			g := c.g
			g.convTo.s = s
			defer g.convTo.leave()
			lc := launcher{to: &g.convTo, err: &c.err, g: g}
			if e := c.sv.run(g, &lc, c.sv.inst, c.in[:], c.out[:]); e != nil && c.err == nil {
				c.err = e
			}
		}
	}
	c.sv, c.in[0], c.out[0], c.err = sv, in, out, nil
	c.g.dev.Session(c.fn)
	c.sv, c.in[0], c.out[0] = nil, nil, nil
	return c.err
}

// narrowPage narrows page's floats into pool at byte off, or, with pool nil,
// returns its words as the words a page held in host memory carries, valid
// until the next conversion.
func (c *pageConv) narrowPage(page []float32, pool backend.Buf, off int) ([]float32, error) {
	if err := c.f.Write(f32b(page)); err != nil {
		return nil, err
	}
	if err := c.run(c.narrow, c.f, c.h); err != nil {
		return nil, err
	}
	if pool != nil {
		return nil, c.g.dev.Copy(pool, off, c.h, 0, c.hBytes)
	}
	if c.words == nil {
		c.words = make([]float32, c.hBytes/4)
	}
	return c.words, c.h.Read(f32b(c.words))
}

// widenPage widens a page's words -- in pool at byte off, or, with pool nil,
// the words of a page held in host memory -- into out's floats.
func (c *pageConv) widenPage(pool backend.Buf, off int, words, out []float32) error {
	var err error
	if pool != nil {
		err = c.g.dev.Copy(c.h, 0, pool, off, c.hBytes)
	} else {
		err = c.h.Write(f32b(words))
	}
	if err != nil {
		return err
	}
	if err := c.run(c.widen, c.h, c.f); err != nil {
		return err
	}
	return c.f.Read(f32b(out))
}

// entryConv is the conversion an entries plane's pages of p positions take
// (convOf), nil where they hold floats. Callers hold g.mu.
func (g *devTier) entryConv(pl *kvLayerPool, p int) (*pageConv, error) {
	c, ok := pl.geom.convOf(p)
	if !ok {
		return nil, nil
	}
	return g.keptConv(c)
}
