package graph

import (
	"errors"
	"fmt"
	"sort"

	"github.com/jitllm/jitllm/format/jlm"
)

// Router is how a mixture's router turns its logits into the selection's
// weights.
type Router uint8

const (
	// RouterSoftmax is a softmax over every expert's logit, the top K kept.
	RouterSoftmax Router = iota + 1
	// RouterSigmoid scores each expert by its logit's sigmoid (DeepSeek-V3's
	// scoring), the top K kept.
	RouterSigmoid
	// RouterSparseMixer is phimoe's sparsemixer: each of the K picks is a
	// softmax over the experts within its jitter band of the running maximum.
	RouterSparseMixer
	// RouterSqrtSoftplus scores each expert by sqrt(softplus(logit))
	// (DeepSeek V4's), the top K kept; on a hash-routed block the K are the
	// table's for the row's token (RoleHashExperts, bound by the route).
	RouterSqrtSoftplus
)

// Mixture describes a routed mixture-of-experts sublayer: a router selects
// K of NExpert experts for the row, the row is gathered once per selected
// expert, each expert's gated FFN runs on its copy from the block's banks,
// and the K outputs are combined in selection order, weighted by the
// router's weights; a shared expert (an always-on FFN, its output optionally
// scaled by the sigmoid of a dot product with the row) is added after.
//
// The graph is the sublayer's: its input is the normed row the router and
// the experts read, its output the mixture's row before the residual takes
// it. A block's attention or recurrence, and a dense FFN beside the mixture,
// are outside it, and its Facts hold only the sublayer's tensors
// (MixtureRoles).
type Mixture struct {
	Block       int
	NEmbd       int
	NExpert, K  int
	NFF         int     // one expert's hidden width
	Act         Act     // the experts' (and the shared expert's) gate activation
	GLU         jlm.GLU // its bounds (Attrs.GLU)
	Router      Router
	Norm        bool    // the K weights are divided by their sum
	Scale       float64 // routed_scaling_factor; 0 and 1 are none
	Groups      int     // the router's expert groups (DeepSeek-V3); 0 or 1 is none
	GroupsUsed  int
	SharedFF    int // the shared expert's hidden width; 0 for none
	SharedGated bool
	// WeightIn scales each pair's input by its routed weight and sums the
	// experts unweighted (Attrs.WeightIn).
	WeightIn bool
	// Ungated experts (and shared expert) are down(Act(up(x))) with no gate
	// bank (Nemotron 3's): an OpAct between up and down in place of the
	// act_mul.
	Ungated bool
	// Latent, where not 0, runs the routed experts in a latent of that
	// width (Kimi-K3's): the row taken down by RoleFFNRoutedDown, the
	// experts and the combine at the latent width, the sum normed by
	// RoleFFNRoutedNorm where stored (at Eps) and taken back up by
	// RoleFFNRoutedUp; the router and the shared experts read the row.
	Latent int
	Eps    float64
	// Inv is the sublayer's stored tensors by role, as Facts.Inventory: the
	// graph binds what is there (a router bias, a selection bias, the banks'
	// biases, a per-expert output scale) and nothing else.
	Inv map[jlm.Role]Shape
}

// MixtureRoles are every tensor a mixture sublayer graph binds. A Facts
// inventory for a Mixture holds the block's tensors of these roles.
var MixtureRoles = []jlm.Role{jlm.RoleRouter, jlm.RoleRouterBias, jlm.RoleExpProbsB, jlm.RoleHashExperts,
	jlm.RoleExpGateBank, jlm.RoleExpUpBank, jlm.RoleExpDownBank,
	jlm.RoleExpGateBias, jlm.RoleExpUpBias, jlm.RoleExpDownBias, jlm.RoleExpScale,
	jlm.RoleShRouter, jlm.RoleShExpGate, jlm.RoleShExpUp, jlm.RoleShExpDown,
	jlm.RoleFFNRoutedDown, jlm.RoleFFNRoutedUp, jlm.RoleFFNRoutedNorm}

func (m *Mixture) bind(r jlm.Role) Binding {
	s := m.Inv[r]
	return Binding{Role: r, K: s.K, Rows: s.Rows, Sheets: s.Sheets}
}

func (m *Mixture) has(r jlm.Role) bool { _, ok := m.Inv[r]; return ok }

// Build is the sublayer's graph.
func (m *Mixture) Build() *Graph {
	g := &Graph{Version: Version, Block: m.Block}
	x := g.value("x", m.NEmbd)
	g.In = []ValueID{x}
	g.Out = []ValueID{m.emit(g, x)}
	return g
}

// emit writes the sublayer's ops into g over its input row x (the normed
// row the router and the experts read) and returns its output row: in a
// sublayer graph of its own (Build), or as a block's FFN (Dense.Mixture).
func (m *Mixture) emit(g *Graph, x ValueID) ValueID { return m.emitFrom(g, x, x) }

// emitFrom is emit with the router reading r and the experts (and a shared
// expert) reading x: Gemma 4's mixture routes a norm of the residual and
// feeds the experts another.
func (m *Mixture) emitFrom(g *Graph, r, x ValueID) ValueID {
	sel := g.value("sel", m.K)
	rw := []Binding{m.bind(jlm.RoleRouter)}
	for _, r := range []jlm.Role{jlm.RoleRouterBias, jlm.RoleExpProbsB, jlm.RoleHashExperts} {
		if m.has(r) {
			rw = append(rw, m.bind(r))
		}
	}
	g.op(Op{Kind: OpRoute, In: []ValueID{r}, Out: []ValueID{sel}, Weight: rw,
		Attr: Attrs{Experts: m.NExpert, TopK: m.K, Router: m.Router, RouteNorm: m.Norm, Scale: m.Scale,
			Groups: m.Groups, GroupsUsed: m.GroupsUsed}})
	// ew is the experts' width: the row's, or the latent's.
	ew, ex := m.NEmbd, x
	if m.Latent != 0 {
		ew, ex = m.Latent, g.value("latent", m.Latent)
		g.op(Op{Kind: OpMatVec, In: []ValueID{x}, Out: []ValueID{ex}, Weight: []Binding{m.bind(jlm.RoleFFNRoutedDown)}})
	}
	pairs := g.value("pairs", m.K*ew)
	g.op(Op{Kind: OpGather, In: []ValueID{ex, sel}, Out: []ValueID{pairs}, Attr: Attrs{WeightIn: m.WeightIn}})
	expert := func(name string, in ValueID, bank, bias jlm.Role) ValueID {
		w := []Binding{m.bind(bank)}
		if m.has(bias) {
			w = append(w, m.bind(bias))
		}
		out := g.value(name, m.K*m.Inv[bank].Rows)
		g.op(Op{Kind: OpExpertMatVec, In: []ValueID{in, sel}, Out: []ValueID{out}, Weight: w})
		return out
	}
	h := g.value("exp_act", m.K*m.NFF)
	if m.Ungated {
		up := expert("exp_up", pairs, jlm.RoleExpUpBank, jlm.RoleExpUpBias)
		g.op(Op{Kind: OpAct, In: []ValueID{up}, Out: []ValueID{h}, Attr: Attrs{Act: m.Act}})
	} else {
		gate := expert("exp_gate", pairs, jlm.RoleExpGateBank, jlm.RoleExpGateBias)
		up := expert("exp_up", pairs, jlm.RoleExpUpBank, jlm.RoleExpUpBias)
		g.op(Op{Kind: OpActMul, In: []ValueID{gate, up}, Out: []ValueID{h}, Attr: Attrs{Act: m.Act, GLU: m.GLU}})
	}
	down := expert("exp_down", h, jlm.RoleExpDownBank, jlm.RoleExpDownBias)
	y := g.value("routed", ew)
	var cw []Binding
	if m.has(jlm.RoleExpScale) {
		cw = []Binding{m.bind(jlm.RoleExpScale)}
	}
	g.op(Op{Kind: OpCombine, In: []ValueID{down, sel}, Out: []ValueID{y}, Weight: cw,
		Attr: Attrs{Experts: m.NExpert, TopK: m.K, WeightIn: m.WeightIn}})
	if m.Latent != 0 {
		if m.has(jlm.RoleFFNRoutedNorm) {
			yn := g.value("routed_normed", ew)
			g.op(Op{Kind: OpNorm, In: []ValueID{y}, Out: []ValueID{yn}, Weight: []Binding{m.bind(jlm.RoleFFNRoutedNorm)},
				Attr: Attrs{Eps: m.Eps}})
			y = yn
		}
		yu := g.value("routed_up", m.NEmbd)
		g.op(Op{Kind: OpMatVec, In: []ValueID{y}, Out: []ValueID{yu}, Weight: []Binding{m.bind(jlm.RoleFFNRoutedUp)}})
		y = yu
	}
	if m.SharedFF > 0 {
		mv := func(name string, in ValueID, r jlm.Role, w int) ValueID {
			out := g.value(name, w)
			g.op(Op{Kind: OpMatVec, In: []ValueID{in}, Out: []ValueID{out}, Weight: []Binding{m.bind(r)}})
			return out
		}
		sh := g.value("sh_act", m.SharedFF)
		if m.Ungated {
			su := mv("sh_up", x, jlm.RoleShExpUp, m.SharedFF)
			g.op(Op{Kind: OpAct, In: []ValueID{su}, Out: []ValueID{sh}, Attr: Attrs{Act: m.Act}})
		} else {
			sg := mv("sh_gate", x, jlm.RoleShExpGate, m.SharedFF)
			su := mv("sh_up", x, jlm.RoleShExpUp, m.SharedFF)
			g.op(Op{Kind: OpActMul, In: []ValueID{sg, su}, Out: []ValueID{sh}, Attr: Attrs{Act: m.Act, GLU: m.GLU}})
		}
		sd := mv("sh_down", sh, jlm.RoleShExpDown, m.NEmbd)
		if m.SharedGated {
			s := g.value("sh_scaled", m.NEmbd)
			g.op(Op{Kind: OpGateScalar, In: []ValueID{x, sd}, Out: []ValueID{s},
				Weight: []Binding{m.bind(jlm.RoleShRouter)}})
			sd = s
		}
		out := g.value("moe_out", m.NEmbd)
		g.op(Op{Kind: OpAdd, In: []ValueID{y, sd}, Out: []ValueID{out}})
		y = out
	}
	return y
}

// matchExperts is the routed experts' region in any graph that has one: a
// route, the gather of its row, the expert matvecs reading its selection
// with the act_mul between them, and the combine, in op order.
func matchExperts(g *Graph) []Region {
	use, _ := users(g)
	var out []Region
	for i, o := range g.Ops {
		if o.Kind != OpRoute || len(o.Out) != 1 {
			continue
		}
		r := Region{Ops: []int{i}}
		seen := map[int]bool{i: true}
		for _, u := range use[o.Out[0]] {
			if !seen[u] {
				seen[u] = true
				r.Ops = append(r.Ops, u)
			}
			// The act_mul between the expert matvecs reads no selection, and
			// a combine that reads another is still down's reader: both are
			// found by the experts' edges, and Legal judges them.
			if g.Ops[u].Kind == OpExpertMatVec {
				for _, a := range use[g.Ops[u].Out[0]] {
					if k := g.Ops[a].Kind; (k == OpActMul || k == OpCombine) && !seen[a] {
						seen[a] = true
						r.Ops = append(r.Ops, a)
					}
				}
			}
		}
		sort.Ints(r.Ops)
		out = append(out, r)
	}
	return out
}

// legalExperts holds a matched region to the grouped mixture's shape: one
// route, one gather of the row it routed, a gated expert FFN (gate and up on
// the gathered pairs into one act_mul, down on its product), every expert
// matvec reading that route's selection, and one combine of down's pairs
// under the same selection; no intermediate is read outside the region.
func legalExperts(g *Graph, r Region) error {
	use, _ := users(g)
	var route, gather, combine, act int = -1, -1, -1, -1
	var exps []int
	for _, i := range r.Ops {
		switch g.Ops[i].Kind {
		case OpRoute:
			route = i
		case OpGather:
			gather = i
		case OpCombine:
			combine = i
		case OpActMul:
			act = i
		case OpExpertMatVec:
			exps = append(exps, i)
		default:
			return fmt.Errorf("op %d (%s) is not part of a routed mixture", i, g.Ops[i].Kind)
		}
	}
	switch {
	case route < 0 || gather < 0 || combine < 0:
		return fmt.Errorf("the region lacks a route, a gather or a combine")
	case act < 0 || len(exps) != 3:
		return fmt.Errorf("the experts are not a gated FFN: %d expert matvecs", len(exps))
	}
	sel := g.Ops[route].Out[0]
	pairs := g.Ops[gather].Out[0]
	if g.Ops[gather].In[1] != sel || g.Ops[combine].In[1] != sel {
		return fmt.Errorf("the gather or the combine reads another selection than the route's")
	}
	gt, up, dn := g.Ops[exps[0]], g.Ops[exps[1]], g.Ops[exps[2]]
	am := g.Ops[act]
	switch {
	case gt.In[0] != pairs || up.In[0] != pairs:
		return fmt.Errorf("gate and up do not read the gathered pairs")
	case gt.In[1] != sel || up.In[1] != sel || dn.In[1] != sel:
		return fmt.Errorf("an expert matvec reads another selection than the route's")
	case am.In[0] != gt.Out[0] || am.In[1] != up.Out[0] || dn.In[0] != am.Out[0]:
		return fmt.Errorf("the experts are not act(gate)*up into down")
	case g.Ops[combine].In[0] != dn.Out[0]:
		return fmt.Errorf("the combine does not read down's pairs")
	case len(use[pairs]) != 2 || len(use[gt.Out[0]]) != 1 || len(use[up.Out[0]]) != 1 ||
		len(use[am.Out[0]]) != 1 || len(use[dn.Out[0]]) != 1:
		return fmt.Errorf("an expert intermediate has a reader outside the mixture")
	}
	return nil
}

// The guards' refusals are values: the waves' guard is asked once per
// block outside a token, but a value costs nothing on a warm path either.
var (
	errFewRows   = errors.New("the dp4a grouped matvec pays where the routed pairs are fewer than the bank's experts; this is not that row class")
	errOnHost    = errors.New("the block's experts run on the host: no grouped kernel runs on the card")
	errWaveRows  = errors.New("the waves run a prompt chunk's experts; a decode or a few-row step looks its experts up a token at a time")
	errWaveCache = errors.New("the waves admit experts through an expert cache on the card; this block's bank is not streamed through one")
	errWaveBias  = errors.New("a wave's slot ids index no per-expert bias or scale table: the block's experts carry one")
)

// MoEFewRow is the few-row mixture form (MTP lever 1): where the routed pairs
// are fewer than the bank's experts (a speculation's verification, a ragged
// step), most experts hold one column, and the staged f16 GEMM pads each to a
// 32-column block where the dp4a grouped matvec pads to its token group and
// reads the same weights. The executed row class decides it (Target.Rows,
// the scratch's pairs against the bank), and only where the experts run on
// the card.
var MoEFewRow = &Rewrite{
	Name:  "moe-few-row",
	Match: matchExperts,
	Legal: legalExperts,
	Pays: func(_ *Graph, _ Region, t Target) error {
		switch {
		case t.Rows != RowsFew:
			return errFewRows
		case t.Residency == ResidencyHost:
			return errOnHost
		}
		return nil
	},
	// Both forms are held to the row-by-row mixture at the same band: the
	// dp4a form reads int8 activations, the staged GEMM f16.
	Contract: map[string]Contract{
		"ptx": {Band: 1e-2, Gate: "model.TestBatchedMixturePrefillMatchesRowByRow (the narrow class forced; last-row logit NMSE)"},
		"msl": {Band: 1e-2, Gate: "model.TestBatchedMixturePrefillMatchesRowByRow (the narrow class forced; last-row logit NMSE)"},
	},
	Keys:    func(*Graph, Region) string { return "" },
	When:    WhenCompile,
	Chooser: "its guard at the executed row class and residency, or tier.WithRewrite",
}

// MoEWaves is a prompt chunk's routed experts run on the card in waves
// through the block's expert cache (streamBank.prefillWaves): the experts the
// cache holds first, then the rest admitted a cache-full at a time, each
// wave's grouped run accumulated into the block's output, where the host
// would read every one. It needs a prompt chunk's rows and a bank streamed
// through a cache, and is refused where the experts index a per-expert table
// by id (a wave's slot ids are not expert ids).
var MoEWaves = &Rewrite{
	Name:  "moe-waves",
	Match: matchExperts,
	Legal: func(g *Graph, r Region) error {
		if err := legalExperts(g, r); err != nil {
			return err
		}
		for _, i := range r.Ops {
			if o := g.Ops[i]; (o.Kind == OpExpertMatVec && len(o.Weight) > 1) || (o.Kind == OpCombine && len(o.Weight) > 0) {
				return errWaveBias
			}
		}
		return nil
	},
	Pays: func(_ *Graph, _ Region, t Target) error {
		switch {
		case t.Rows != RowsChunk:
			return errWaveRows
		case t.Residency != ResidencyCached:
			return errWaveCache
		}
		return nil
	},
	Contract: map[string]Contract{
		"ptx":   {Band: 3e-3, Gate: "model.TestHybridPrefillWavesMatchTheHost (prefill logit NMSE against the host's, the widest model band)"},
		"spirv": {Band: 3e-3, Gate: "model.TestHybridPrefillWavesMatchTheHost (prefill logit NMSE against the host's, the widest model band)"},
		"msl":   {Band: 3e-3, Gate: "model.TestHybridPrefillWavesMatchTheHost (prefill logit NMSE against the host's, the widest model band)"},
	},
	Keys:    func(*Graph, Region) string { return "" },
	When:    WhenCompile,
	Chooser: "its guard at the executed row class and residency, or tier.WithRewrite",
}
