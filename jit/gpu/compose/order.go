package compose

import "github.com/jitllm/jitllm/jit/gpu/ir"

// Ordered effects: a write to a buffer is a node (a token), and what must
// come after it reads the token -- a barrier, a load through After, a later
// write -- so a loop's body may stage, wait and read again within one trip,
// in the order its tokens state.

// Write stores v into buffer b at index ix after token after (-1 none).
func (g *Row) Write(b, ix, v, after int) int {
	return g.Add(Node{Kind: NWrite, Buf: b, Ix: ix, A: v, B: -1, C: after})
}

// WriteVec stores vs, consecutive from ix, in one vector store.
func (g *Row) WriteVec(b, ix, after int, vs ...int) int {
	n := Node{Kind: NWrite, Buf: b, Ix: ix, A: vs[0], B: -1, C: after, Vec: uint8(len(vs))}
	for i, v := range vs[1:] {
		n.H[i] = int32(v)
	}
	return g.Add(n)
}

// WriteTile stores tile t (of shape tt) into buffer b from ix, its rows
// stride apart (major: its columns).
func (g *Row) WriteTile(b, ix, stride, t int, tt ir.TileType, major bool, after int) int {
	return g.Add(Node{Kind: NWrite, Buf: b, Ix: ix, A: t, B: stride, C: after, Tile: tt, Major: major})
}

// Barrier is the workgroup's barrier after token after (-1 none).
func (g *Row) Barrier(after int) int { return g.Add(Node{Kind: NBarrier, A: after}) }

// After is v, taken after token tok.
func (g *Row) After(v, tok int) int { return g.Add(Node{Kind: NHint, Hint: HintAfter, A: v, B: tok}) }

// TileLoad is a tile tt of buffer b from ix, its rows stride apart (major:
// its columns), after token after (-1 none).
func (g *Row) TileLoad(tt ir.TileType, b, ix, stride int, major bool, after int) int {
	return g.Add(Node{Kind: NTileLoad, Buf: b, Ix: ix, B: stride, C: after, Tile: tt, Major: major})
}

// TileSplat is a tile tt of every element v.
func (g *Row) TileSplat(tt ir.TileType, v int) int {
	return g.Add(Node{Kind: NTileSplat, A: v, Tile: tt})
}

// TileMMA is a times b plus c.
func (g *Row) TileMMA(a, b, c int) int { return g.Add(Node{Kind: NTileMMA, A: a, B: b, C: c}) }

// Subgroup is the thread's subgroup in its workgroup.
func (g *Row) Subgroup() int { return g.Add(Node{Kind: NSubgroup}) }
