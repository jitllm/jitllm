package model

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
)

// The block graph's first slice (docs/design/composer.md, "Above the
// recipes"): each covered block's graph is built from what the container
// states, verified against the container's tensors and configuration, and
// lowered to the operations the hand-written block body runs. Under
// WithBlockGraph a host decode runs covered blocks through that lowering; the
// hand-written body is the control and stays the default until the graph
// covers decode, batch and the device for the dense family.

// WithBlockGraph runs the host decode of every block the block graph covers
// through the graph's lowering instead of the hand-written block body, and
// refuses to open a model with a text block the graph does not cover. The
// two compute the same values with the same kernels in the same order; the
// option exists so a gate can compare them.
func WithBlockGraph(on bool) Option { return func(l *loadOpts) { l.opt.blockGraph = on } }

// graphCovered is the frontend: the architectures whose classes are known to
// be the dense block graph.Dense describes. A trusted adapter by name, as the
// converter's are; the runtime below it never asks for an architecture.
func graphCovered(a string) bool { return a == "llama" || a == "qwen2" || a == "qwen3" }

// blockFacts is what the container states about text block li: the dense
// configuration and the tensors stored for the block by role.
func (m *Model) blockFacts(li int) (graph.Facts, error) {
	c := m.Cfg
	if !graphCovered(c.Arch) {
		return graph.Facts{}, fmt.Errorf("model: the block graph does not cover %s", c.Arch)
	}
	// Configuration the op language does not express yet: each is a missing
	// op or attribute, refused rather than dropped.
	w0, an := c.AttnWindow(li, 1<<20)
	switch {
	case c.LayerNorm || c.Parallel || c.AttnOutGate || c.MLA() || c.KVShared(li) || c.extraKVHead(li) ||
		c.Indexer() || c.streamHead() || c.PLEDim != 0 || c.ResidualScale != 1 || !c.RopeAt(li) ||
		c.AttnTemp(li, 1) != 1 || w0 != 0 || an != 1<<20+1 || c.LayerKind(li).Recurrent() || c.Act != nn.ActSiLU:
		return graph.Facts{}, fmt.Errorf("model: block %d of %s has configuration the block graph does not express", li, c.Arch)
	}
	layout := graph.RopeNorm
	if c.RopeNeox {
		layout = graph.RopeNeox
	}
	d := graph.Dense{Block: li, NEmbd: c.NEmbd, NHead: c.NHead, NKVHead: c.NKVHeadAt(li), HeadDim: c.HeadDimAt(li),
		NFFN: c.NFFNAt(li), Eps: c.RMSEps, NRot: c.NRot, Base: c.RopeBase, Layout: layout, RopeFreqs: m.rope.Freqs != nil,
		Scale: c.AttnScale, Softcap: float64(c.AttnSoftcap), Act: graph.ActSiLU, QKNorm: c.QKNormAt(li)}
	inv := map[jlm.Role]graph.Shape{}
	for i := range m.container.Entries() {
		e := &m.container.Entries()[i]
		if int(e.Block) != li {
			continue
		}
		s := graph.Shape{K: int(e.Dims[0])}
		if e.NDim >= 2 {
			s.Rows = int(e.Dims[1])
		}
		inv[e.Role] = s
	}
	_, d.QKVBias = inv[jlm.RoleAttnQBias]
	return graph.Facts{Config: d, Inventory: inv}, nil
}

// denseLowering is a verified graph lowered for the host decode: the dense
// block's op sequence matched against the operations the hand-written body
// runs. qk says the matched sequence holds the per-head q/k norms, which
// attnPrep applies with the rotary and the cache write.
type denseLowering struct {
	g  *graph.Graph
	qk bool
	// ffnChain is graph.Gate2In's verdict on the block (nn.LayerPlan.FFNChain),
	// and rewrites every rewrite's decision, for the plan explanation.
	ffnChain int8
	rewrites []graph.Decision
}

// lowerDense matches g's ops, in order, against the dense block's host
// operations, and refuses anything else: an op the lowering does not know is
// an op with no host lowering, never one skipped.
func lowerDense(g *graph.Graph) (*denseLowering, error) {
	ops, i := g.Ops, 0
	fail := func(want string) error {
		if i < len(ops) {
			return fmt.Errorf("model: block %d graph op %d is %s; the host lowering wants %s", g.Block, i, ops[i].Kind, want)
		}
		return fmt.Errorf("model: block %d graph ends where the host lowering wants %s", g.Block, want)
	}
	take := func(k graph.OpKind, role jlm.Role) bool {
		if i < len(ops) && ops[i].Kind == k && (role == 0 || (len(ops[i].Weight) > 0 && ops[i].Weight[0].Role == role)) {
			i++
			return true
		}
		return false
	}
	seq := []struct {
		k    graph.OpKind
		role jlm.Role
	}{{graph.OpNorm, jlm.RoleAttnNorm}, {graph.OpMatVec, jlm.RoleAttnQ}, {graph.OpMatVec, jlm.RoleAttnK}, {graph.OpMatVec, jlm.RoleAttnV}}
	for _, s := range seq {
		if !take(s.k, s.role) {
			return nil, fail(fmt.Sprintf("%s of %v", s.k, s.role))
		}
	}
	lw := &denseLowering{g: g}
	if take(graph.OpHeadNorm, jlm.RoleAttnQNorm) {
		if !take(graph.OpHeadNorm, jlm.RoleAttnKNorm) {
			return nil, fail("head_norm of attn_k_norm")
		}
		lw.qk = true
	}
	seq = []struct {
		k    graph.OpKind
		role jlm.Role
	}{{graph.OpRope, 0}, {graph.OpRope, 0}, {graph.OpKVAppend, 0}, {graph.OpAttend, 0},
		{graph.OpMatVec, jlm.RoleAttnOut}, {graph.OpAdd, 0}, {graph.OpNorm, jlm.RoleFFNNorm},
		{graph.OpMatVec, jlm.RoleFFNGate}, {graph.OpMatVec, jlm.RoleFFNUp}, {graph.OpActMul, 0},
		{graph.OpMatVec, jlm.RoleFFNDown}, {graph.OpAdd, 0}}
	for _, s := range seq {
		if !take(s.k, s.role) {
			return nil, fail(s.k.String())
		}
	}
	if i != len(ops) {
		return nil, fail("the end of the block")
	}
	return lw, nil
}

// buildGraphs builds, verifies and lowers every text block's graph
// (WithBlockGraph), refusing the model where any one fails.
func (m *Model) buildGraphs() error {
	m.graphs = make([]*denseLowering, m.Cfg.NLayer)
	for li := range m.graphs {
		f, err := m.blockFacts(li)
		if err != nil {
			return err
		}
		g := f.Config.Build()
		if err := graph.Verify(g, f); err != nil {
			return err
		}
		lw, err := lowerDense(g)
		if err != nil {
			return err
		}
		if lw.qk != m.Cfg.QKNormAt(li) {
			return fmt.Errorf("model: block %d graph's q/k norm is %v and attnPrep's %v", li, lw.qk, m.Cfg.QKNormAt(li))
		}
		lw.rewrites = graph.Apply(g, graph.Gate2In)
		lw.ffnChain = -1
		for _, d := range lw.rewrites {
			if d.Rewrite == graph.Gate2In.Name && d.Err == nil {
				lw.ffnChain = 1
			}
		}
		m.graphs[li] = lw
	}
	return nil
}

// BlockRewrites is every rewrite's decision on text block li's graph, legal
// or refused with its reason (nil without WithBlockGraph).
func (m *Model) BlockRewrites(li int) []graph.Decision {
	if li < 0 || li >= len(m.graphs) || m.graphs[li] == nil {
		return nil
	}
	return m.graphs[li].rewrites
}

// BlockGraph is text block li's verified graph, or nil where the model was
// opened without WithBlockGraph.
func (m *Model) BlockGraph(li int) *graph.Graph {
	if li < 0 || li >= len(m.graphs) || m.graphs[li] == nil {
		return nil
	}
	return m.graphs[li].g
}

// GraphBlocks counts the blocks a State ran through the block graph's
// lowering: the forced path's evidence (RULE 10).
func (s *State) GraphBlocks() int { return s.graphRuns }

// graphBlock runs block li's host decode through its lowering: the calls the
// hand-written body makes for a dense block, in its order, so the kernels and
// their order are the control's.
func (s *State) graphBlock(li int, l *layer, lw *denseLowering) error {
	c, m := s.c, s.m
	s.graphRuns++
	// norm(x) -> attn_in
	s.norm(s.h, s.x, l.attnNorm, l.attnNormB)
	s.jit.NewInput()
	m.trace(li, "attn_norm", s.h)
	// matvec q, k, v (one region inside a fused run, as the body groups them)
	kvd := c.KVDimAt(li)
	s.aheadOf(li, 0, [nn.MaxGroup]*tensor{&l.wo})
	grouped := s.groupOf(s.h, [nn.MaxGroup]*tensor{&l.wq, &l.wk, &l.wv}, [nn.MaxGroup][]float32{s.q, s.kc, s.vc}, 3)
	if grouped {
		s.addBiasRows(s.q, l.bq, 1)
	} else {
		if _, err := s.projectQ(l, s.q, s.ogate, s.h, 1, false); err != nil {
			return err
		}
		if err := s.mv(s.kc, l.wk, s.h); err != nil {
			return err
		}
	}
	s.addBias(s.kc, l.bk)
	if !grouped {
		if err := s.mv(s.vc, l.wv, s.h); err != nil {
			return err
		}
	}
	s.addBias(s.vc, l.bv)
	// [head_norm q, head_norm k,] rope q, rope k, kv_append: attnPrep
	s.attnPrep(l, s.q, s.kc[:kvd], s.vc[:kvd], s.ropeTable(li, s.cs, s.csSWA, 0), li, 0, s.pos)
	m.trace(li, "k", s.kc[:kvd])
	// attend
	s.attendRow(li, l)
	// matvec o, add into the residual (folded into o's rows where the run
	// fuses it)
	qd := c.QDimAt(li)
	src := s.xb
	plain := s.residFold && l.bo == nil
	if plain {
		s.grp.Resid, s.grp.Alpha = s.x, float32(c.ResidualScale)
	}
	s.aheadOf(li, 1, [nn.MaxGroup]*tensor{&l.gate, &l.up})
	attnResid := plain && s.groupOf(src[:qd], [nn.MaxGroup]*tensor{&l.wo}, [nn.MaxGroup][]float32{s.h}, 1)
	s.grp.Resid = nil
	if !attnResid && !s.groupOf(src[:qd], [nn.MaxGroup]*tensor{&l.wo}, [nn.MaxGroup][]float32{s.h}, 1) {
		if err := s.mv(s.h, l.wo, src[:qd]); err != nil {
			return err
		}
	}
	s.addBias(s.h, l.bo)
	if !attnResid {
		s.addInto(s.x, s.h)
	}
	m.trace(li, "attn_resid", s.x)
	// norm -> gate, up -> act_mul -> down, add into the residual
	s.norm(s.h, s.x, l.ffnNorm, l.ffnNormB)
	s.jit.NewInput()
	ff := c.NFFNAt(li)
	gate, up, act := s.gate[:ff], s.up[:ff], c.Act
	s.aheadOf(li, 2, [nn.MaxGroup]*tensor{&l.down})
	if !s.groupOf(s.h, [nn.MaxGroup]*tensor{&l.gate, &l.up}, [nn.MaxGroup][]float32{gate, up}, 2) {
		if err := s.mv(gate, l.gate, s.h); err != nil {
			return err
		}
		if err := s.mv(up, l.up, s.h); err != nil {
			return err
		}
	}
	s.gaussTopK(li, gate, ff)
	s.grp.ActUp, s.grp.Act = up, act
	if s.residFold {
		s.grp.Resid, s.grp.Alpha = s.x, float32(c.ResidualScale)
	}
	s.aheadNext(li)
	grouped = s.groupOf(gate, [nn.MaxGroup]*tensor{&l.down}, [nn.MaxGroup][]float32{s.h}, 1)
	ffnResid := grouped && s.grp.Resid != nil
	s.grp.ActUp, s.grp.Resid = nil, nil
	if !grouped {
		s.actmulAll(gate, up, act)
		s.jit.NewInput()
		if err := s.mv(s.h, l.down, gate); err != nil {
			return err
		}
	}
	if !ffnResid {
		s.addInto(s.x, s.h)
	}
	s.layerOutScale(l, s.x)
	m.trace(li, "ffn_resid", s.x)
	return nil
}
