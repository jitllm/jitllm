package gpu

import (
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

// pageHalves is a migration's binary16 page conversions: a page's floats and
// its halves staged on the card, and the forms between them, at a page's rows
// of positions.
type pageHalves struct {
	g              *devTier
	f, h           backend.Buf
	narrow, widen  *servedPart
	fBytes, hBytes int
}

// newPageHalves stages a page of rows positions of w floats.
func (g *devTier) newPageHalves(w, rows int) (*pageHalves, error) {
	c := &pageHalves{g: g, fBytes: rows * w * 4, hBytes: rows * w * 2}
	var err error
	if c.narrow, err = g.servedHalves(w, rows, false); err != nil {
		return nil, err
	}
	if c.widen, err = g.servedHalves(w, rows, true); err != nil {
		return nil, err
	}
	if c.f, err = g.dev.Alloc(c.fBytes); err != nil {
		return nil, err
	}
	if c.h, err = g.dev.Alloc(c.hBytes); err != nil {
		c.f.Free()
		return nil, err
	}
	return c, nil
}

func (c *pageHalves) close() {
	c.f.Free()
	c.h.Free()
}

// run runs sv from in into out in a submission of its own.
func (c *pageHalves) run(sv *servedPart, in, out backend.Buf) error {
	g := c.g
	var err error
	g.dev.Session(func(s backend.Session) {
		g.halvesTo.s = s
		defer g.halvesTo.leave()
		lc := launcher{to: &g.halvesTo, err: &err, g: g}
		if e := sv.run(g, &lc, sv.inst, []backend.Buf{in}, []backend.Buf{out}); e != nil && err == nil {
			err = e
		}
	})
	return err
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
	words := make([]float32, c.hBytes/4)
	return words, c.h.Read(f32b(words))
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
