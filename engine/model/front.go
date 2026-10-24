package model

import (
	"fmt"
	"math"
	"slices"
	"sync"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit"
)

// The model as the composer reads it (jit.Model): its container, and the
// graph of generic ops of each block, the head and the input rows, built from
// what the container states and nothing else: no graph here asks what family
// a model is. A block whose features the op language does not state yet has
// no graph, and Uncovered says why by name; main's runtime runs every model
// whatever its graphs are, so a missing graph refuses nothing at Open.

// Uncovered is why the front end states no graph for a block, the head or
// the input rows, Covered where it does: a feature the op language does not
// state yet -- shrinking that list is the front end's work list -- or a graph
// it built wrong (UncoveredUnbound, UncoveredUnverified), which a gate holds
// as a failure and UncoveredErr details.
type Uncovered uint8

const (
	Covered Uncovered = iota
	UncoveredKind
	UncoveredMixture
	UncoveredLatent
	UncoveredValueWidth
	UncoveredGeometry
	UncoveredSelect
	UncoveredStreams
	UncoveredRopeAxes
	UncoveredNonCausal
	UncoveredClassic
	UncoveredQKNormPlace
	UncoveredVNorm
	UncoveredKVShared
	UncoveredVFromK
	UncoveredClamp
	UncoveredTemp
	UncoveredPLE
	UncoveredCompressed
	UncoveredNGram
	UncoveredSparseAct
	UncoveredAct
	UncoveredTokenAct
	UncoveredPooling
	UncoveredEmbedRows
	UncoveredUnbound
	UncoveredUnverified
	numUncovered
)

var uncoveredNames = [numUncovered]string{
	Covered:              "covered",
	UncoveredKind:        "a layer that is not an attention block (a recurrent or state-space mixer)",
	UncoveredMixture:     "a mixture of experts",
	UncoveredLatent:      "latent attention (a compressed KV)",
	UncoveredValueWidth:  "a value head of another width than the key's",
	UncoveredGeometry:    "a layer of its own head width, head count or rotary width",
	UncoveredSelect:      "an attention indexer or sparse key selection",
	UncoveredStreams:     "several residual streams",
	UncoveredRopeAxes:    "a rotary over position axes its sections do not state (a pair's halves turned apart)",
	UncoveredNonCausal:   "non-causal attention",
	UncoveredClassic:     "LayerNorm or a parallel attention and FFN",
	UncoveredQKNormPlace: "a q/k norm without a weight, or one over the whole projection after the rotary",
	UncoveredVNorm:       "a norm over v",
	UncoveredKVShared:    "a block reading another block's history",
	UncoveredVFromK:      "v taken from k's projection",
	UncoveredClamp:       "clamped projections",
	UncoveredTemp:        "an attention temperature per position",
	UncoveredPLE:         "per-layer embeddings",
	UncoveredCompressed:  "compressed or grouped attention",
	UncoveredNGram:       "an n-gram embedding",
	UncoveredSparseAct:   "a gaussian top-k activation",
	UncoveredAct:         "an activation the op language does not state",
	UncoveredTokenAct:    "projection inputs quantized to int8 per token",
	UncoveredPooling:     "a pooled head (an embedding model's)",
	UncoveredEmbedRows:   "input rows beyond the token's row, a scale and a position table",
	UncoveredUnbound:     "stored tensors its graph does not bind",
	UncoveredUnverified:  "a graph that does not verify",
}

func (u Uncovered) String() string {
	if u < numUncovered {
		return uncoveredNames[u]
	}
	return fmt.Sprintf("uncovered(%d)", uint8(u))
}

// frontState is the model's graphs, built on first use and again for a
// model stated otherwise since: its config, its rotary or the tensors the
// container stores (restate), which only a test's violation changes after
// Open.
type frontState struct {
	mu  sync.Mutex
	set *frontSet
	// restate is a test's statement of the stored tensors in place of the
	// container's, nil otherwise; restated counts its changes.
	restate  func(block int32, inv map[jlm.Role]graph.Shape)
	restated int
}

// frontSet is the graphs one statement of the model builds, and the
// statement.
type frontSet struct {
	blocks      []*graph.Graph
	why         []Uncovered
	errs        []error
	head, embed *graph.Graph
	headWhy     Uncovered
	embedWhy    Uncovered
	headErr     error
	embedErr    error
	key         frontKey
}

// frontKey is what a set's graphs were built from beside the container.
type frontKey struct {
	cfg           *Config
	head          headKey
	rope, ropeSWA ropeKey
	restated      int
}

// headKey is what the head's graph reads of the config: a test that sets one
// of them in place states another head.
type headKey struct {
	softcap    float32
	scale, eps float64
	layerNorm  bool
}

func headKeyOf(c *Config) headKey {
	return headKey{softcap: c.FinalSoftcap, scale: c.LogitScale, eps: c.RMSEps, layerNorm: c.LayerNorm}
}

// ropeKey is what the front end reads of a rotary.
type ropeKey struct {
	nrot, runs       int
	base, scale      float64
	neox, freqs, set bool
}

func ropeKeyOf(r *nn.Rope) ropeKey {
	if r == nil {
		return ropeKey{}
	}
	return ropeKey{nrot: r.NRot, runs: len(r.Runs), base: r.Base, scale: r.Scale, neox: r.Neox, freqs: len(r.Freqs) > 0, set: true}
}

// composing is the model as the composer reads it.
type composing struct {
	*jlm.File
	m *Model
}

// Composing is the model as the composer reads it: its container and the
// graphs built from it.
func (m *Model) Composing() jit.Model { return composing{m.container, m} }

func (c composing) BlockGraph(li int) *graph.Graph {
	f := c.m.frontGraphs()
	if li < 0 || li >= len(f.blocks) {
		return nil
	}
	return f.blocks[li]
}

func (c composing) Blocks() int { return c.m.Cfg.NLayer }

func (c composing) SeamGraph(int) *graph.Graph { return nil }

func (c composing) HeadGraph() *graph.Graph { return c.m.frontGraphs().head }

func (c composing) EmbedGraph() *graph.Graph { return c.m.frontGraphs().embed }

// Rotaries is the model's rotary, then its sliding layers' where it trains
// two: the tables the blocks' rotaries turn by.
func (c composing) Rotaries() []graph.Rotary {
	out := []graph.Rotary{c.m.rotaryOf(c.m.rope)}
	if c.m.ropeSWA != nil {
		out = append(out, c.m.rotaryOf(*c.m.ropeSWA))
	}
	return out
}

func (m *Model) rotaryOf(r nn.Rope) graph.Rotary {
	g := graph.Rotary{NRot: r.NRot, Base: r.Base, Freqs: len(r.Freqs) > 0}
	g.Sections, g.Interleaved = m.ropeAxes(r)
	return g
}

func (c composing) Tower() bool { return false }

func (c composing) Segment() jit.Model { return nil }

// Uncovered is why block li has no graph, Covered for a block that has one;
// li of -1 is the head's, -2 the input rows'.
func (m *Model) Uncovered(li int) Uncovered {
	u, _ := m.uncoveredAt(li)
	return u
}

// UncoveredErr is what a graph built wrong got wrong (UncoveredUnbound,
// UncoveredUnverified), nil for any other reason.
func (m *Model) UncoveredErr(li int) error {
	_, err := m.uncoveredAt(li)
	return err
}

func (m *Model) uncoveredAt(li int) (Uncovered, error) {
	f := m.frontGraphs()
	switch {
	case li == -1:
		return f.headWhy, f.headErr
	case li == -2:
		return f.embedWhy, f.embedErr
	case li >= 0 && li < len(f.why):
		return f.why[li], f.errs[li]
	}
	return UncoveredUnverified, fmt.Errorf("no block %d", li)
}

func (m *Model) frontGraphs() *frontSet {
	f := &m.front
	f.mu.Lock()
	defer f.mu.Unlock()
	k := frontKey{cfg: m.Cfg, head: headKeyOf(m.Cfg), rope: ropeKeyOf(&m.rope), ropeSWA: ropeKeyOf(m.ropeSWA), restated: f.restated}
	if f.set != nil && f.set.key == k {
		return f.set
	}
	n := m.Cfg.NLayer
	s := &frontSet{key: k, blocks: make([]*graph.Graph, n), why: make([]Uncovered, n), errs: make([]error, n)}
	for li := range n {
		s.blocks[li], s.why[li], s.errs[li] = m.blockGraph(li)
	}
	s.head, s.headWhy, s.headErr = m.headGraph()
	s.embed, s.embedWhy, s.embedErr = m.embedGraph()
	f.set = s
	return s
}

// inventory is what the container stores for block (jlm.DenseBlock for the
// model-wide tensors), by role, at the shape a binding states: a vector's
// rows 0. A tensor stored per expert counts as a bank of its experts.
func (m *Model) inventory(block int32) (graph.Facts, bool) {
	inv := map[jlm.Role]graph.Shape{}
	perExpert := false
	for i := range m.container.Entries() {
		e := &m.container.Entries()[i]
		if e.Block != block || e.Role.Vision() {
			continue
		}
		if e.Index >= 0 {
			perExpert = true
			continue
		}
		s := graph.Shape{K: e.K()}
		if e.NDim >= 2 {
			s.Rows = e.Rows()
		}
		if e.NDim >= 3 {
			s.Sheets = int(e.Dims[2])
		}
		inv[e.Role] = s
	}
	if r := m.front.restate; r != nil {
		r(block, inv)
	}
	return graph.Facts{Inventory: inv}, perExpert
}

// graphActs is the activations the op language states.
var graphActs = map[nn.ActKind]graph.Act{
	nn.ActSiLU: graph.ActSiLU, nn.ActGELU: graph.ActGELUTanh, nn.ActGELUErf: graph.ActGELUErf,
	nn.ActQuickGELU: graph.ActQuickGELU, nn.ActReLU: graph.ActReLU, nn.ActReLU2: graph.ActReLU2,
}

// uncovered is why block li's features are not all ones this front end
// writes, Covered for none. It reads numbers the container states, never what the
// model is called.
func (m *Model) uncovered(li int) Uncovered {
	c := m.Cfg
	hd := c.HeadDimAt(li)
	switch {
	case c.LayerKind(li) != jlm.LayerFullAttn:
		return UncoveredKind
	case c.MoEAt(li):
		return UncoveredMixture
	case c.MLA() || c.QLoraRank != 0:
		return UncoveredLatent
	case c.HeadDimV != 0 && c.HeadDimV != hd:
		return UncoveredValueWidth
	case hd != c.HeadDim || c.NKVHeadAt(li) != c.NKVHead || c.NRotAt(li) != c.NRot && !(m.ropeSWA != nil && c.SWA(li)):
		return UncoveredGeometry
	case c.IdxHeads != 0:
		return UncoveredSelect
	case c.Streams() > 1:
		return UncoveredStreams
	case c.RopeAt(li) && !m.ropeAxesStated(m.ropeAt(li)):
		return UncoveredRopeAxes
	case c.NonCausal || c.BidirSWA:
		return UncoveredNonCausal
	case c.QKL2Norm || c.QKNormPost && c.QKNormWide:
		return UncoveredQKNormPlace
	case c.VNorm:
		return UncoveredVNorm
	case c.KVShared(li):
		return UncoveredKVShared
	case m.layers[li].vFromK:
		return UncoveredVFromK
	case c.TokenAct:
		// The norms inside a sublayer are stored only with this (model.go), so
		// until it is stated no container reaches the graph's statement of them.
		return UncoveredTokenAct
	case c.ClampKQV != 0:
		return UncoveredClamp
	case c.AttnTempScale != 0:
		return UncoveredTemp
	case c.PLEDim != 0:
		return UncoveredPLE
	case len(c.CompKinds) != 0 || c.OGroups != 0:
		return UncoveredCompressed
	case c.NGramAt(li):
		return UncoveredNGram
	case c.SparseStd != 0:
		return UncoveredSparseAct
	}
	// xIELU is stated for an ungated FFN, from the four numbers its block
	// stores; the gated activations are the ones graphActs lists.
	if _, ok := graphActs[c.Act]; !ok && !(c.Act == nn.ActXIELU && !m.stores(li, jlm.RoleFFNGate)) {
		return UncoveredAct
	}
	return Covered
}

// stores reports whether the container stores role for block li.
func (m *Model) stores(li int, role jlm.Role) bool {
	_, ok := m.container.Find(role, int32(li), -1)
	return ok
}

// ropeAt is the rotary layer li turns by: the sliding layers' where the
// model trains two and li slides, the model's own otherwise.
func (m *Model) ropeAt(li int) nn.Rope {
	if m.ropeSWA != nil && m.Cfg.SWA(li) {
		return *m.ropeSWA
	}
	return m.rope
}

// ropeAxes is the position axes rotary r turns its pairs by, as the op
// language states them (graph.Attrs.Sections, Interleaved): none for a
// rotary of one position.
func (m *Model) ropeAxes(r nn.Rope) (sections [4]int, interleaved bool) {
	if len(r.Runs) == 0 {
		return sections, false
	}
	return m.Cfg.RopeSections, m.Cfg.RopeInterleaved
}

// ropeAxesStated reports whether the sections ropeAxes states are rotary
// r's: every pair in a run of the sections' axes at its own frequency, so
// a pair turns by one angle. A pair whose two halves turn by different
// axes (RopeXD) is not.
func (m *Model) ropeAxesStated(r nn.Rope) bool {
	if len(r.Runs) == 0 {
		return true
	}
	sec, il := m.ropeAxes(r)
	want := nn.MRopeRuns(sec[:])
	if il {
		want = nn.IMRopeRuns(sec[:], r.NRot/2)
	}
	return !m.Cfg.RopeXD && nn.RopeRuns(r.Runs) == r.NRot/2 && slices.Equal(want, r.Runs)
}

// blockGraph is block li's graph: the attention block the container states
// -- a norm where it stores one (a LayerNorm with its bias where the model's
// norms are mean-centred), q, k and v with their stored biases (q split from
// its gate where the projection carries one), the q/k norm, the rotary, the
// history append, attention of the block's heads, scale, window, softcap and
// sinks, the output gate and the norm inside the sublayer where stored, the
// output projection, a norm over it where stored and the residual, scaled
// where the container states a scale; then the MLP the same way, gated where
// it stores a gate, its activation from the block's numbers where it states
// them, reading the residual after the attention or, in a parallel block, the
// block's input -- or why it has none.
func (m *Model) blockGraph(li int) (*graph.Graph, Uncovered, error) {
	if why := m.uncovered(li); why != Covered {
		return nil, why, nil
	}
	c := m.Cfg
	f, perExpert := m.inventory(int32(li))
	if perExpert {
		return nil, UncoveredMixture, nil
	}
	has := func(r jlm.Role) bool {
		_, ok := f.Inventory[r]
		return ok
	}
	stored := func(r jlm.Role) []jlm.Role {
		if has(r) {
			return []jlm.Role{r}
		}
		return nil
	}
	d, hd, nkv := c.NEmbd, c.HeadDimAt(li), c.NKVHeadAt(li)
	qdim, kvDim, nffn := c.NHead*hd, nkv*hd, c.NFFNAt(li)
	b := graph.NewBlock(li)
	norm := func(v graph.ValueID, r jlm.Role) graph.ValueID {
		if has(r) {
			return b.Norm(v, r, c.RMSEps)
		}
		return v
	}
	// preNorm is the norm before a sublayer: mean-centred, with its stored
	// bias, where the model's are; the post-norms and the q/k norms stay RMS.
	preNorm := func(v graph.ValueID, r, bias jlm.Role) graph.ValueID {
		if _, ok := f.Inventory[r]; !ok {
			return v
		}
		if c.LayerNorm {
			return b.LayerNorm(v, r, c.RMSEps, stored(bias)...)
		}
		return b.Norm(v, r, c.RMSEps)
	}
	resid := func(x, y graph.ValueID) graph.ValueID {
		if s := c.ResidualScale; s != 0 && s != 1 {
			return b.Emit(graph.OpAdd, []graph.ValueID{x, y}, []graph.Result{{Name: "sum", Width: d}}, nil, graph.Attrs{Alpha: s}, nil, nil)[0]
		}
		return b.Add(x, y)
	}
	x := b.In("x", d)
	n := preNorm(x, jlm.RoleAttnNorm, jlm.RoleAttnNormBias)
	// With an output gate the q projection is double width: each head's query,
	// then that head's gate.
	qrows := qdim
	if c.AttnOutGate {
		qrows = 2 * qdim
	}
	q := b.MatVec(n, jlm.RoleAttnQ, qrows, stored(jlm.RoleAttnQBias)...)
	k := b.MatVec(n, jlm.RoleAttnK, kvDim, stored(jlm.RoleAttnKBias)...)
	v := b.MatVec(n, jlm.RoleAttnV, kvDim, stored(jlm.RoleAttnVBias)...)
	var gate graph.ValueID
	if c.AttnOutGate {
		both := q
		half := func(name string, off int) graph.ValueID {
			return b.Emit(graph.OpSlice, []graph.ValueID{both}, []graph.Result{{Name: name, Width: qdim}}, nil,
				graph.Attrs{HeadDim: 2 * hd, Offset: off, Width: hd}, nil, nil)[0]
		}
		q, gate = half("q", 0), half("gate", hd)
	}
	// A q norm after the rotary keeps k's before it: the converter folded k's
	// weight into q's, and an unweighted norm commutes with the rotation.
	postQ := c.QKNormAt(li) && c.QKNormPost
	switch {
	case postQ:
		k = b.HeadNorm(k, jlm.RoleAttnKNorm, hd, c.RMSEps)
	case c.QKNormAt(li) && c.QKNormWide:
		q, k = b.Norm(q, jlm.RoleAttnQNorm, c.RMSEps), b.Norm(k, jlm.RoleAttnKNorm, c.RMSEps)
	case c.QKNormAt(li):
		q, k = b.HeadNorm(q, jlm.RoleAttnQNorm, hd, c.RMSEps), b.HeadNorm(k, jlm.RoleAttnKNorm, hd, c.RMSEps)
	}
	if c.RopeAt(li) {
		r := m.ropeAt(li)
		layout := graph.RopeNorm
		if c.RopeNeox {
			layout = graph.RopeNeox
		}
		a := graph.Attrs{HeadDim: hd, NRot: r.NRot, Base: r.Base, Layout: layout, Freqs: len(r.Freqs) > 0, Mscale: r.Scale}
		a.Sections, a.Interleaved = m.ropeAxes(r)
		q = b.Emit(graph.OpRope, []graph.ValueID{q}, []graph.Result{{Name: "q.rope", Width: qdim}}, nil, a, nil, nil)[0]
		k = b.Emit(graph.OpRope, []graph.ValueID{k}, []graph.Result{{Name: "k.rope", Width: kvDim}}, nil, a, nil, nil)[0]
	}
	if postQ {
		q = b.HeadNorm(q, jlm.RoleAttnQNorm, hd, c.RMSEps)
	}
	hist := b.KVAppend(k, v, 0)
	scale := c.AttnScale
	if scale == 0 {
		scale = 1 / math.Sqrt(float64(hd))
	}
	a := graph.Attrs{NHead: c.NHead, NKVHead: nkv, HeadDim: hd, Scale: scale, Softcap: float64(c.AttnSoftcap)}
	if c.SWA(li) {
		a.Window, a.Chunked = c.SWAWindow, c.SWAChunked
	}
	att := b.Attend(q, hist, a, stored(jlm.RoleAttnSinks)...)
	if c.AttnOutGate {
		att = b.ActMul(graph.ActSigmoid, gate, att)
	}
	att = norm(att, jlm.RoleAttnSubNorm)
	x2 := resid(x, norm(b.MatVec(att, jlm.RoleAttnOut, d, stored(jlm.RoleAttnOutBias)...), jlm.RolePostAttnNorm))
	// A parallel block's MLP reads the block's input: through its own norm
	// where it stores one, through the attention's where the two share one.
	ffnIn := x2
	if c.Parallel {
		ffnIn = x
	}
	n2 := preNorm(ffnIn, jlm.RoleFFNNorm, jlm.RoleFFNNormBias)
	if _, own := f.Inventory[jlm.RoleFFNNorm]; c.Parallel && !own {
		n2 = n
	}
	var h graph.ValueID
	switch {
	case has(jlm.RoleFFNGate):
		gt := b.MatVec(n2, jlm.RoleFFNGate, nffn)
		up := b.MatVec(n2, jlm.RoleFFNUp, nffn, stored(jlm.RoleFFNUpBias)...)
		h = b.ActMul(graphActs[c.Act], gt, up)
	case c.Act == nn.ActXIELU:
		h = xielu(b, b.MatVec(n2, jlm.RoleFFNUp, nffn, stored(jlm.RoleFFNUpBias)...), nffn)
	default:
		up := b.MatVec(n2, jlm.RoleFFNUp, nffn, stored(jlm.RoleFFNUpBias)...)
		h = b.Emit(graph.OpAct, []graph.ValueID{up}, []graph.Result{{Name: "act", Width: nffn}}, nil, graph.Attrs{Act: graphActs[c.Act]}, nil, nil)[0]
	}
	h = norm(h, jlm.RoleFFNSubNorm)
	down := b.MatVec(h, jlm.RoleFFNDown, d, stored(jlm.RoleFFNDownBias)...)
	b.Out(resid(x2, norm(down, jlm.RolePostFFNNorm)))
	if err := graph.Verify(b.G, f); err != nil {
		return nil, UncoveredUnverified, fmt.Errorf("the block's graph does not verify: %w", err)
	}
	if u := graph.Unbound(b.G, f); len(u) != 0 {
		return nil, UncoveredUnbound, fmt.Errorf("the container stores %v for the block, which its graph does not bind", u)
	}
	return b.G, Covered, nil
}

// xielu is the activation of x from the four numbers its block stores
// (jlm.RoleXIELU: alpha_p, alpha_n, beta, eps, each read as a scalar of the
// stored vector), as maps over x:
//
//	x > 0:   alpha_p*x*x + beta*x
//	x <= 0:  (expm1(min(x, eps)) - x)*alpha_n + beta*x
func xielu(b *graph.Builder, x graph.ValueID, width int) graph.ValueID {
	const (
		alphaP = iota
		alphaN
		beta
		eps
		numbers
	)
	// none is a map that reads no number: its operands are its inputs.
	const none = -1
	op := func(name string, fn graph.MapFn, number int, in ...graph.ValueID) graph.ValueID {
		var w []graph.Binding
		a := graph.Attrs{Fn: fn}
		if number != none {
			w, a.Offset, a.Width = []graph.Binding{{Role: jlm.RoleXIELU, K: numbers}}, number, 1
		}
		return b.Emit(graph.OpMap, in, []graph.Result{{Name: name, Width: width}}, w, a, nil, nil)[0]
	}
	diff := op("xielu.diff", graph.MapSub, none, op("xielu.expm1", graph.MapExpm1, none, op("xielu.min", graph.MapMin, eps, x)), x)
	neg := op("xielu.neg", graph.MapMul, alphaN, diff)
	pos := op("xielu.pos", graph.MapMul, none, op("xielu.ax", graph.MapMul, alphaP, x), x)
	pick := op("xielu.pick", graph.MapWhere, none, x, pos, neg)
	return op("xielu", graph.MapAdd, none, pick, op("xielu.bx", graph.MapMul, beta, x))
}

// headGraph is the head (graph.Head) of what the container stores: a
// LayerNorm with its stored bias where the model's norms are LayerNorms, an
// RMSNorm otherwise, the projection to the vocabulary (the embedding's where
// the model ties them) with its stored bias, then the final softcap and the
// logit scale, main's order over the logits.
func (m *Model) headGraph() (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	switch {
	case c.Pooling != jlm.PoolNone:
		return nil, UncoveredPooling, nil
	case c.Streams() > 1:
		return nil, UncoveredStreams, nil
	}
	f, _ := m.inventory(jlm.DenseBlock)
	stored := func(r jlm.Role) bool {
		_, ok := f.Inventory[r]
		return ok
	}
	h := graph.Head{NEmbd: c.NEmbd, NVocab: c.NVocab, Eps: c.RMSEps, LayerNorm: c.LayerNorm,
		NormBias: stored(jlm.RoleOutputNormBias), Output: jlm.RoleOutput, Bias: stored(jlm.RoleOutputBias),
		Softcap: float64(c.FinalSoftcap), Scale: c.LogitScale}
	if !stored(h.Output) {
		h.Output = jlm.RoleTokenEmbd
	}
	g := h.Build()
	if err := graph.Verify(g, f); err != nil {
		return nil, UncoveredUnverified, fmt.Errorf("the head's graph does not verify: %w", err)
	}
	return g, Covered, nil
}

// embedGraph is the step's input rows: each row's token's row of the
// embedding, times the stored scale, plus its position's row where the
// container stores a position table.
func (m *Model) embedGraph() (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	f, _ := m.inventory(jlm.DenseBlock)
	_, types := f.Inventory[jlm.RoleTokenTypes]
	switch {
	case c.Streams() > 1:
		return nil, UncoveredStreams, nil
	case c.PLEDim != 0:
		return nil, UncoveredPLE, nil
	case types || m.hardEmbd != nil:
		return nil, UncoveredEmbedRows, nil
	}
	b := graph.NewBlock(-1)
	x := b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "embd", Width: c.NEmbd}},
		[]graph.Binding{{Role: jlm.RoleTokenEmbd, K: c.NEmbd, Rows: c.NVocab}}, graph.Attrs{}, nil, nil)[0]
	if s := c.EmbdScale; s != 0 && s != 1 {
		x = b.Emit(graph.OpMap, []graph.ValueID{x}, []graph.Result{{Name: "scaled", Width: c.NEmbd}}, nil,
			graph.Attrs{Fn: graph.MapMul, Const: s}, nil, nil)[0]
	}
	if p, ok := f.Inventory[jlm.RolePosEmbd]; ok {
		pos := b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "pos", Width: c.NEmbd}},
			[]graph.Binding{{Role: jlm.RolePosEmbd, K: p.K, Rows: p.Rows}}, graph.Attrs{Positional: true}, nil, nil)[0]
		x = b.Add(x, pos)
	}
	b.Out(x)
	if err := graph.Verify(b.G, f); err != nil {
		return nil, UncoveredUnverified, fmt.Errorf("the input rows' graph does not verify: %w", err)
	}
	return b.G, Covered, nil
}
