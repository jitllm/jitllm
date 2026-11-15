package jlm

import "fmt"

// BlockGraphs are the blocks' graphs where the converter states them: each
// distinct block's graph as the graph package stores one (graph.Encode),
// and Of each block's index into them, the trunk's blocks then the
// prediction blocks. The engine decodes a stated block's graph rather than
// building it. Empty where the converter states none. The twenty-eighth
// optional tail; see encodeConfig.
type BlockGraphs struct {
	Graphs []string
	Of     []uint16
}

// MaxBlockGraph is the most bytes one stated graph may hold.
const MaxBlockGraph = 1 << 24

func encodeBlockGraphs(w *wbuf, g *BlockGraphs) {
	w.u32(uint32(len(g.Graphs)))
	for _, s := range g.Graphs {
		w.str(s)
	}
	w.u32(uint32(len(g.Of)))
	for _, i := range g.Of {
		w.u16(i)
	}
}

func decodeBlockGraphs(r *rbuf) BlockGraphs {
	var g BlockGraphs
	// A graph is at least its length, an index two bytes.
	for range r.count(4) {
		g.Graphs = append(g.Graphs, r.str())
	}
	for range r.count(2) {
		g.Of = append(g.Of, r.u16())
	}
	return g
}

// valid refuses a statement that names a graph it does not hold, leaves a
// block out, or states a graph no block reads.
func (g *BlockGraphs) valid(blocks int) error {
	if len(g.Graphs) == 0 && len(g.Of) == 0 {
		return nil
	}
	if len(g.Of) != blocks {
		return fmt.Errorf("jlm: graphs stated for %d blocks of %d", len(g.Of), blocks)
	}
	read := make([]bool, len(g.Graphs))
	for li, i := range g.Of {
		if int(i) >= len(g.Graphs) {
			return fmt.Errorf("jlm: block %d's graph is %d of %d", li, i, len(g.Graphs))
		}
		read[i] = true
	}
	for i, s := range g.Graphs {
		switch {
		case !read[i]:
			return fmt.Errorf("jlm: stated graph %d is no block's", i)
		case len(s) == 0 || len(s) > MaxBlockGraph:
			return fmt.Errorf("jlm: stated graph %d is %d bytes", i, len(s))
		}
	}
	return nil
}

// Graph is block li's stated graph, empty where the converter states none.
func (g *BlockGraphs) Graph(li int) string {
	if li < 0 || li >= len(g.Of) {
		return ""
	}
	return g.Graphs[g.Of[li]]
}
