package model

import (
	"fmt"
	"math"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// The BERT family's encoder block (BERT, nomic-bert) as a block graph: the
// same op language as a decoder's, stated post-norm (graph.Dense.PostResid:
// each residual add followed by a LayerNorm of the sum), bidirectional, with
// biased projections and an ungated GELU MLP (nomic-bert's a SwiGLU, and its
// rotary on q and k). Each block's graph is built from the container's
// inventory, verified, and matched against the encoder's own operations
// (Embedder.encode), which then run as the lowering states them.

// encLowering is one encoder block's verified graph and what its lowering
// matched.
type encLowering struct {
	g *graph.Graph
	// rope: q and k turned by the rotary (nomic-bert's); gated: the MLP is
	// silu(gate) * up (nomic-bert's), not up's GELU.
	rope, gated bool
}

// buildEncGraphs builds, verifies and lowers every block of a BERT-family
// encoder. A block the language does not state refuses the model.
func (m *Model) buildEncGraphs(c *jlm.File, enc *encoder) error {
	cfg := m.Cfg
	enc.graphs = make([]*encLowering, len(enc.blocks))
	for li := range enc.blocks {
		inv := map[jlm.Role]graph.Shape{}
		for i := range c.Entries() {
			e := &c.Entries()[i]
			if int(e.Block) != li {
				continue
			}
			s := graph.Shape{K: int(e.Dims[0])}
			if e.NDim >= 2 {
				s.Rows = int(e.Dims[1])
			}
			if e.NDim >= 3 {
				s.Sheets = int(e.Dims[2])
			}
			inv[e.Role] = s
		}
		has := func(r jlm.Role) bool { _, ok := inv[r]; return ok }
		act := graph.ActGELUTanh
		if has(jlm.RoleFFNGate) {
			act = graph.ActSiLU
		}
		d := graph.Dense{Block: li, NEmbd: cfg.NEmbd, NHead: cfg.NHead, NKVHead: cfg.NHead, HeadDim: cfg.HeadDim,
			NFFN: cfg.NFFN, Eps: cfg.RMSEps, Scale: 1 / math.Sqrt(float64(cfg.HeadDim)), Act: act,
			LayerNorm: true, QKVBias: has(jlm.RoleAttnQBias), OutBias: has(jlm.RoleAttnOutBias),
			UpBias: has(jlm.RoleFFNUpBias), DownBias: has(jlm.RoleFFNDownBias), Ungated: !has(jlm.RoleFFNGate),
			NoPreNorm: true, NoFFNNorm: true, NonCausal: true, PostResid: true, NoRope: enc.rope == nil}
		if r := enc.rope; r != nil {
			d.NRot, d.Base, d.Layout = r.NRot, r.Base, graph.RopeNorm
			if r.Neox {
				d.Layout = graph.RopeNeox
			}
		}
		g := d.Build()
		f := graph.Facts{Config: d, Inventory: inv}
		if u := graph.Unbound(g, f); len(u) > 0 {
			return fmt.Errorf("model: encoder block %d of %s is not covered by the block graph: %s (%v)",
				li, cfg.Arch, UncoveredUnboundTensor, u)
		}
		if err := graph.Verify(g, f); err != nil {
			return err
		}
		lw, err := lowerEncoder(g)
		if err != nil {
			return err
		}
		if lw.rope != (enc.rope != nil) || lw.gated != enc.swiglu {
			return fmt.Errorf("model: encoder block %d graph's rotary (%v) or gated MLP (%v) is not the encoder's",
				li, lw.rope, lw.gated)
		}
		enc.graphs[li] = lw
	}
	return nil
}

// lowerEncoder matches g's ops, in order, against the encoder block's
// operations: q, k and v with their biases, [the rotary on q and k,] the
// append and the bidirectional attention, o, the add and its LayerNorm; up
// and its activation [or gate, up and act_mul], down, the add and its
// LayerNorm. Anything else is refused.
func lowerEncoder(g *graph.Graph) (*encLowering, error) {
	ops, i := g.Ops, 0
	fail := func(want string) error {
		if i < len(ops) {
			return fmt.Errorf("model: encoder block %d graph op %d is %s; the encoder's lowering wants %s", g.Block, i, ops[i].Kind, want)
		}
		return fmt.Errorf("model: encoder block %d graph ends where the encoder's lowering wants %s", g.Block, want)
	}
	take := func(k graph.OpKind, role jlm.Role) bool {
		if i < len(ops) && ops[i].Kind == k && (role == 0 || len(ops[i].Weight) > 0 && ops[i].Weight[0].Role == role) {
			i++
			return true
		}
		return false
	}
	run := func(k graph.OpKind, role jlm.Role) error {
		if !take(k, role) {
			return fail(fmt.Sprintf("%s of %v", k, role))
		}
		return nil
	}
	lw := &encLowering{g: g}
	for _, r := range []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV} {
		if err := run(graph.OpMatVec, r); err != nil {
			return nil, err
		}
	}
	if take(graph.OpRope, 0) {
		lw.rope = true
		if err := run(graph.OpRope, 0); err != nil {
			return nil, err
		}
	}
	if err := run(graph.OpKVAppend, 0); err != nil {
		return nil, err
	}
	if i < len(ops) && ops[i].Kind == graph.OpAttend && !ops[i].Attr.Bidir {
		return nil, fail("a bidirectional attend")
	}
	steps := []struct {
		k    graph.OpKind
		role jlm.Role
	}{{graph.OpAttend, 0}, {graph.OpMatVec, jlm.RoleAttnOut}, {graph.OpAdd, 0}, {graph.OpLayerNorm, jlm.RoleAttnOutNorm}}
	for _, st := range steps {
		if err := run(st.k, st.role); err != nil {
			return nil, err
		}
	}
	if take(graph.OpMatVec, jlm.RoleFFNGate) {
		lw.gated = true
		if err := run(graph.OpMatVec, jlm.RoleFFNUp); err != nil {
			return nil, err
		}
		if err := run(graph.OpActMul, 0); err != nil {
			return nil, err
		}
	} else {
		if err := run(graph.OpMatVec, jlm.RoleFFNUp); err != nil {
			return nil, err
		}
		if err := run(graph.OpAct, 0); err != nil {
			return nil, err
		}
	}
	for _, st := range []struct {
		k    graph.OpKind
		role jlm.Role
	}{{graph.OpMatVec, jlm.RoleFFNDown}, {graph.OpAdd, 0}, {graph.OpLayerNorm, jlm.RoleLayerOutNorm}} {
		if err := run(st.k, st.role); err != nil {
			return nil, err
		}
	}
	if i != len(ops) {
		return nil, fail("the end of the block")
	}
	return lw, nil
}
