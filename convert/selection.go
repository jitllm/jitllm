package convert

import (
	"math"

	"github.com/jitllm/jitllm/format/jlm"
)

// StateSelection states how the blocks' attention selects the history it
// reads (jlm.Selection), as each reference builds its indexer:
//
//   - DeepSeek V3.2's lightning indexer (an MLA model): a LayerNormed key
//     and a query on the query latent, both turned neox-style, a weight a
//     head; every position ranked by the relu'd heads' weighted sum, the
//     IdxTopK best read;
//   - MiniMax Sparse Attention (IdxBlock): a key and a query each RMS-normed
//     and turned, one query head per kv group; blocks of IdxBlock positions
//     ranked by their best, IdxTopK read, the IdxLocal ending at the query's
//     forced in, on every block past the dense lead;
//   - Qwen3.8-Flash-Next's QSA (QSARatio): the raw key cached, padded to a
//     head; the query's heads RMS-normed and turned; blocks of QSARatio
//     positions ranked by their keys pooled, normed and turned at the
//     block's first position, IdxTopK/QSARatio read, on every attention
//     block.
//
// A block selects where it stores the indexer's key.
func StateSelection(c *jlm.Config, stores func(r jlm.Role, li int32) bool) {
	if c.IdxHeads == 0 || c.NLayer == 0 || len(c.CompKinds) != 0 {
		return
	}
	n := c.NLayer + c.NMTP
	id := int32(c.IdxHeadDim)
	s := &jlm.Selection{Blocks: make([]bool, n), Heads: c.IdxHeads, HeadDim: c.IdxHeadDim, KeyWidth: c.IdxHeadDim,
		Groups: 1, Rank: jlm.RankBest, Block: 1, Keep: c.IdxTopK}
	prog := func(nin uint16, ops ...jlm.TowerOp) *jlm.Program {
		for i := range ops {
			if ops[i].In == nil {
				ops[i].In = []uint16{nin + uint16(i) - 1}
			}
		}
		return &jlm.Program{NIn: nin, Ops: ops, Out: nin + uint16(len(ops)) - 1}
	}
	on := func(v uint16) []uint16 { return []uint16{v} }
	linear := func(r jlm.Role) jlm.TowerOp { return jlm.TowerOp{Code: jlm.TOpLinear, Bind: []jlm.Role{r}} }
	rope := func(layout uint8) jlm.TowerOp { return jlm.TowerOp{Code: jlm.TOpRope, Span: id, Chain: layout} }
	headNorm := func(r jlm.Role) jlm.TowerOp {
		return jlm.TowerOp{Code: jlm.TOpRMSNorm, Bind: []jlm.Role{r}, Span: id, Eps: c.RMSEps}
	}
	at := func(li uint32) bool { return stores(jlm.RoleIdxK, int32(li)) }
	switch {
	case c.QSARatio != 0:
		for li := range n {
			s.Blocks[li] = at(li) && !(int(li) < len(c.LayerKinds) && c.LayerKinds[li].Recurrent())
		}
		s.Key = prog(1, jlm.TowerOp{Code: jlm.TOpLinear, In: on(0), Bind: []jlm.Role{jlm.RoleIdxK}})
		s.KeyWidth = c.HeadDim
		s.Query = prog(2, jlm.TowerOp{Code: jlm.TOpLinear, In: on(0), Bind: []jlm.Role{jlm.RoleIdxQ}},
			headNorm(jlm.RoleIdxQNorm), rope(jlm.RopeAsBlock))
		s.Rank, s.Block, s.Keep = jlm.RankPooled, c.QSARatio, c.IdxTopK/c.QSARatio
		s.PoolNorm, s.Eps = jlm.RoleIdxKNorm, c.RMSEps
	case c.IdxBlock != 0:
		for li := range n {
			s.Blocks[li] = at(li) && li >= c.NDenseLead
		}
		s.Key = prog(1, jlm.TowerOp{Code: jlm.TOpLinear, In: on(0), Bind: []jlm.Role{jlm.RoleIdxK}},
			jlm.TowerOp{Code: jlm.TOpRMSNorm, Bind: []jlm.Role{jlm.RoleIdxKNorm}, Eps: c.RMSEps}, rope(jlm.RopeAsBlock))
		s.Query = prog(2, jlm.TowerOp{Code: jlm.TOpLinear, In: on(0), Bind: []jlm.Role{jlm.RoleIdxQ}},
			headNorm(jlm.RoleIdxQNorm), rope(jlm.RopeAsBlock))
		s.Groups, s.Block, s.Local = c.IdxHeads, c.IdxBlock, c.IdxLocal
	case c.KVLoraRank != 0:
		for li := range n {
			s.Blocks[li] = at(li)
		}
		// The key's LayerNorm is nn.LayerNorm's default epsilon.
		s.Key = prog(1, jlm.TowerOp{Code: jlm.TOpLinear, In: on(0), Bind: []jlm.Role{jlm.RoleIdxK}},
			jlm.TowerOp{Code: jlm.TOpLayerNorm, Bind: []jlm.Role{jlm.RoleIdxKNorm, jlm.RoleIdxKNormB}, Eps: 1e-6},
			rope(jlm.RopeNeox))
		s.Query = prog(2, jlm.TowerOp{Code: jlm.TOpLinear, In: on(1), Bind: []jlm.Role{jlm.RoleIdxQB}}, rope(jlm.RopeNeox))
		var e jlm.ExprBuilder
		e.Fn(jlm.FnMul, jlm.Operand(0), jlm.Num(float32(1/math.Sqrt(float64(c.IdxHeadDim*c.IdxHeads)))))
		s.Weight = prog(1, linear(jlm.RoleIdxProj),
			jlm.TowerOp{Code: jlm.TOpStated, Width: int32(c.IdxHeads), Expr: &jlm.Expr{Terms: e.T}})
		s.Relu = true
	default:
		return
	}
	c.Selection = s
}
