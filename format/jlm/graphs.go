package jlm

import "fmt"

// BlockGraphs are the blocks' graphs where the converter states them: each
// distinct block's graph as the graph package stores one (graph.Encode),
// and Of each block's index into them, the trunk's blocks, the head blocks
// (Config.HeadBlocks) then the prediction blocks. The engine decodes a stated block's graph rather than
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

// PredictionBlocks is how many prediction blocks follow the trunk: the
// blocks the graph statement states past the trunk's NLayer in a model with
// no head blocks, which follow the trunk instead. A prediction block is a
// block like any other; what it reads and produces is its graph's and its
// tensors', and the record keeps no count of them.
func (c *Config) PredictionBlocks() uint32 {
	if c.HeadBlocks() != 0 || len(c.Blocks.Of) <= int(c.NLayer) {
		return 0
	}
	return uint32(len(c.Blocks.Of)) - c.NLayer
}

// TextBlocks is the trunk's blocks and the prediction blocks after it.
func (c *Config) TextBlocks() uint32 { return c.NLayer + c.PredictionBlocks() }

// statedBlocks is how many blocks' graphs c states, and the count a
// container written before HeadGraphsRevision may state instead, its head
// blocks left to the config.
func (c *Config) statedBlocks() (blocks, short int) {
	blocks = int(c.NLayer + c.HeadBlocks() + c.PredictionBlocks())
	if c.Revision < HeadGraphsRevision {
		return blocks, int(c.TextBlocks())
	}
	return blocks, blocks
}

// valid refuses a statement that names a graph it does not hold, leaves a
// block out, or states a graph no block reads; short is the count an older
// container may state (Config.statedBlocks).
func (g *BlockGraphs) valid(blocks, short int) error {
	if len(g.Graphs) == 0 && len(g.Of) == 0 {
		return nil
	}
	if len(g.Of) != blocks && len(g.Of) != short {
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
