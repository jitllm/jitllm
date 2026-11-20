package gpu

import (
	"fmt"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// A binary16 V pool's pages (Config.KVF16) arrive and leave as floats
// (nn.LayerDevice.MigrateKV). The card narrows and widens them itself,
// formed through the composer like any other part (graph.OpHalves,
// graph.OpWidenHalves): a page's floats staged and narrowed into the pool,
// a page's halves widened before they are read back.

// halvesKey is one of the conversions: rows of w floats, narrowed or widened.
type halvesKey struct {
	w     int
	widen bool
}

// servedHalves is rows rows of w floats narrowed to binary16, two a word, or
// the halves widened back where widen is set.
func (g *devTier) servedHalves(w, rows int, widen bool) (*servedPart, error) {
	key := halvesKey{w, widen}
	gr := g.halvesGraphs[key]
	if gr == nil {
		b := graph.New()
		if widen {
			h := b.In("kv.h", w/2)
			b.G.Values[h].Elem = graph.ElemU32
			b.Out(b.Emit(graph.OpWidenHalves, []graph.ValueID{h}, []graph.Result{{Name: "kv.f", Width: w}}, nil, graph.Attrs{}, nil, nil)...)
		} else {
			b.Out(b.Emit(graph.OpHalves, []graph.ValueID{b.In("kv.f", w)},
				[]graph.Result{{Name: "kv.h", Width: w / 2, Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)...)
		}
		gr = b.G
		if g.halvesGraphs == nil {
			g.halvesGraphs = map[halvesKey]*graph.Graph{}
		}
		g.halvesGraphs[key] = gr
	}
	return g.servedFor(servedKey{kind: nn.PartHalves, graph: gr, rows: rows}, func() (composedPart, error) {
		return composedPart{kind: nn.PartHalves, graph: gr, apart: true}, nil
	})
}

// pageHalves is a page's binary16 conversions: a page's floats and its
// halves staged on the card, and the forms between them, at a page's rows of
// positions. words is a page's halves read to host memory, and sv, in, out,
// err and fn a conversion's submission, kept so that one run a token
// allocates nothing.
type pageHalves struct {
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

// stage makes the page's staging on the card.
func (c *pageHalves) stage() error {
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

func (c *pageHalves) close() {
	if c.f != nil {
		c.f.Free()
		c.h.Free()
		c.f, c.h = nil, nil
	}
}

// formHalves forms, at a placement, the page conversions a history copy or
// a move runs at rows positions of w floats (keptHalves): the device's,
// held on a lane of their own and run under g.mu, their staging made at
// their first run. Callers hold g.mu.
func (g *devTier) formHalves(w, rows int) error {
	key := [2]int{w, rows}
	if g.halves[key] != nil {
		return nil
	}
	v := &devTier{devShared: g.devShared, devSess: g.dv.devSess, lane: &lane{}, Stats: g.dv.Stats}
	c := &pageHalves{g: v, fBytes: rows * w * 4, hBytes: rows * w * 2}
	was := g.prewarming
	g.prewarming = true
	var err error
	if c.narrow, err = v.servedHalves(w, rows, false); err == nil {
		c.widen, err = v.servedHalves(w, rows, true)
	}
	g.prewarming = was
	if err != nil {
		v.dropLaneParts(v.lane)
		return err
	}
	if g.halves == nil {
		g.halves = map[[2]int]*pageHalves{}
	}
	g.halves[key] = c
	return nil
}

// keptHalves is the conversions formHalves formed, staged at the first call
// and charged to the pool as the device's own buffers (scratchWin).
// Callers hold g.mu.
func (g *devTier) keptHalves(w, rows int) (*pageHalves, error) {
	c := g.halves[[2]int{w, rows}]
	if c == nil {
		return nil, fmt.Errorf("a history page's halves of %d floats at %d rows were not formed at placement", w, rows)
	}
	if c.f == nil {
		win := g.scratchWin()
		err := c.stage()
		win.close()
		if err != nil {
			return nil, err
		}
	}
	return c, nil
}

// freeHalves frees what formHalves formed, its charge given back. Callers
// hold g.mu.
func (g *devTier) freeHalves() {
	defer g.scratchWin().close()
	for key, c := range g.halves {
		c.close()
		c.g.dropLaneParts(c.g.lane)
		delete(g.halves, key)
	}
}

// run runs sv from in into out in a submission of its own.
func (c *pageHalves) run(sv *servedPart, in, out backend.Buf) error {
	if c.fn == nil {
		c.fn = func(s backend.Session) {
			g := c.g
			g.halvesTo.s = s
			defer g.halvesTo.leave()
			lc := launcher{to: &g.halvesTo, err: &c.err, g: g}
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
// returns its halves as the words a page held in host memory carries.
func (c *pageHalves) narrowPage(page []float32, pool backend.Buf, off int) ([]float32, error) {
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

// widenPage widens a page's halves -- in pool at byte off, or, with pool nil,
// the words of a page held in host memory -- into out's floats.
func (c *pageHalves) widenPage(pool backend.Buf, off int, words, out []float32) error {
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
