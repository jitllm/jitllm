package model

import (
	"fmt"
	"math"
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
	UncoveredPartialRope
	UncoveredNonCausal
	UncoveredClassic
	UncoveredQKNormPlace
	UncoveredVNorm
	UncoveredKVShared
	UncoveredVFromK
	UncoveredOutGate
	UncoveredClamp
	UncoveredTemp
	UncoveredResidScale
	UncoveredPLE
	UncoveredCompressed
	UncoveredNGram
	UncoveredSparseAct
	UncoveredAct
	UncoveredPostNorm
	UncoveredSubNorm
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
	UncoveredRopeAxes:    "a rotary over several position axes or per element",
	UncoveredPartialRope: "a rotary over part of each head",
	UncoveredNonCausal:   "non-causal attention",
	UncoveredClassic:     "LayerNorm or a parallel attention and FFN",
	UncoveredQKNormPlace: "a q/k norm after the rotary or without a weight",
	UncoveredVNorm:       "a norm over v",
	UncoveredKVShared:    "a block reading another block's history",
	UncoveredVFromK:      "v taken from k's projection",
	UncoveredOutGate:     "a gate on the attention's output",
	UncoveredClamp:       "clamped projections",
	UncoveredTemp:        "an attention temperature per position",
	UncoveredResidScale:  "a scaled residual",
	UncoveredPLE:         "per-layer embeddings",
	UncoveredCompressed:  "compressed or grouped attention",
	UncoveredNGram:       "an n-gram embedding",
	UncoveredSparseAct:   "a gaussian top-k activation",
	UncoveredAct:         "an activation the op language does not state",
	UncoveredPostNorm:    "a norm over a sublayer's output before the residual",
	UncoveredSubNorm:     "a norm inside a sublayer before its output projection",
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
	rope, ropeSWA ropeKey
	restated      int
}

// ropeKey is what the front end reads of a rotary.
type ropeKey struct {
	nrot             int
	base, scale      float64
	neox, freqs, set bool
}

func ropeKeyOf(r *nn.Rope) ropeKey {
	if r == nil {
		return ropeKey{}
	}
	return ropeKey{nrot: r.NRot, base: r.Base, scale: r.Scale, neox: r.Neox, freqs: len(r.Freqs) > 0, set: true}
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
	out := []graph.Rotary{rotaryOf(c.m.rope)}
	if c.m.ropeSWA != nil {
		out = append(out, rotaryOf(*c.m.ropeSWA))
	}
	return out
}

func rotaryOf(r nn.Rope) graph.Rotary {
	return graph.Rotary{NRot: r.NRot, Base: r.Base, Freqs: len(r.Freqs) > 0}
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
	k := frontKey{cfg: m.Cfg, rope: ropeKeyOf(&m.rope), ropeSWA: ropeKeyOf(m.ropeSWA), restated: f.restated}
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
	case c.RopeSections != [4]int{} || c.RopeInterleaved || c.RopeXD:
		return UncoveredRopeAxes
	case c.NonCausal || c.BidirSWA:
		return UncoveredNonCausal
	case c.LayerNorm || c.Parallel:
		return UncoveredClassic
	case c.QKNormPost || c.QKL2Norm:
		return UncoveredQKNormPlace
	case c.VNorm:
		return UncoveredVNorm
	case c.KVShared(li):
		return UncoveredKVShared
	case m.layers[li].vFromK:
		return UncoveredVFromK
	case c.AttnOutGate:
		return UncoveredOutGate
	case c.ClampKQV != 0:
		return UncoveredClamp
	case c.AttnTempScale != 0:
		return UncoveredTemp
	case c.ResidualScale != 0 && c.ResidualScale != 1:
		return UncoveredResidScale
	case c.PLEDim != 0:
		return UncoveredPLE
	case len(c.CompKinds) != 0 || c.OGroups != 0:
		return UncoveredCompressed
	case c.NGramAt(li):
		return UncoveredNGram
	case c.SparseStd != 0:
		return UncoveredSparseAct
	}
	if _, ok := graphActs[c.Act]; !ok {
		return UncoveredAct
	}
	if m.stores(li, jlm.RolePostAttnNorm) || m.stores(li, jlm.RolePostFFNNorm) {
		return UncoveredPostNorm
	}
	if m.stores(li, jlm.RoleAttnSubNorm) || m.stores(li, jlm.RoleFFNSubNorm) {
		return UncoveredSubNorm
	}
	if r := m.ropeAt(li); c.RopeAt(li) && r.NRot != hd {
		return UncoveredPartialRope
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

// blockGraph is block li's graph: the pre-norm attention block the
// container states -- RMSNorm, q, k and v with their stored biases, the q/k
// norm, the rotary, the history append, attention of the block's heads,
// scale, window, softcap and sinks, the output projection and the residual,
// then the RMSNormed gated MLP and its residual -- or why it has none.
func (m *Model) blockGraph(li int) (*graph.Graph, Uncovered, error) {
	if why := m.uncovered(li); why != Covered {
		return nil, why, nil
	}
	c := m.Cfg
	f, perExpert := m.inventory(int32(li))
	if perExpert {
		return nil, UncoveredMixture, nil
	}
	stored := func(r jlm.Role) []jlm.Role {
		if _, ok := f.Inventory[r]; ok {
			return []jlm.Role{r}
		}
		return nil
	}
	d, hd, nkv := c.NEmbd, c.HeadDimAt(li), c.NKVHeadAt(li)
	qdim, kvDim, nffn := c.NHead*hd, nkv*hd, c.NFFNAt(li)
	b := graph.NewBlock(li)
	x := b.In("x", d)
	n := b.Norm(x, jlm.RoleAttnNorm, c.RMSEps)
	q := b.MatVec(n, jlm.RoleAttnQ, qdim, stored(jlm.RoleAttnQBias)...)
	k := b.MatVec(n, jlm.RoleAttnK, kvDim, stored(jlm.RoleAttnKBias)...)
	v := b.MatVec(n, jlm.RoleAttnV, kvDim, stored(jlm.RoleAttnVBias)...)
	if c.QKNormAt(li) {
		if c.QKNormWide {
			q, k = b.Norm(q, jlm.RoleAttnQNorm, c.RMSEps), b.Norm(k, jlm.RoleAttnKNorm, c.RMSEps)
		} else {
			q, k = b.HeadNorm(q, jlm.RoleAttnQNorm, hd, c.RMSEps), b.HeadNorm(k, jlm.RoleAttnKNorm, hd, c.RMSEps)
		}
	}
	if c.RopeAt(li) {
		r := m.ropeAt(li)
		layout := graph.RopeNorm
		if c.RopeNeox {
			layout = graph.RopeNeox
		}
		a := graph.Attrs{HeadDim: hd, NRot: r.NRot, Base: r.Base, Layout: layout, Freqs: len(r.Freqs) > 0, Mscale: r.Scale}
		q = b.Emit(graph.OpRope, []graph.ValueID{q}, []graph.Result{{Name: "q.rope", Width: qdim}}, nil, a, nil, nil)[0]
		k = b.Emit(graph.OpRope, []graph.ValueID{k}, []graph.Result{{Name: "k.rope", Width: kvDim}}, nil, a, nil, nil)[0]
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
	x2 := b.Add(x, b.MatVec(att, jlm.RoleAttnOut, d, stored(jlm.RoleAttnOutBias)...))
	n2 := b.Norm(x2, jlm.RoleFFNNorm, c.RMSEps)
	gt := b.MatVec(n2, jlm.RoleFFNGate, nffn)
	up := b.MatVec(n2, jlm.RoleFFNUp, nffn, stored(jlm.RoleFFNUpBias)...)
	down := b.MatVec(b.ActMul(graphActs[c.Act], gt, up), jlm.RoleFFNDown, d, stored(jlm.RoleFFNDownBias)...)
	b.Out(b.Add(x2, down))
	if err := graph.Verify(b.G, f); err != nil {
		return nil, UncoveredUnverified, fmt.Errorf("the block's graph does not verify: %w", err)
	}
	if u := graph.Unbound(b.G, f); len(u) != 0 {
		return nil, UncoveredUnbound, fmt.Errorf("the container stores %v for the block, which its graph does not bind", u)
	}
	return b.G, Covered, nil
}

// headGraph is the head: the RMSNormed row's projection to the vocabulary
// (the embedding's where the model ties them) with its stored bias, then the
// final softcap and the logit scale, main's order over the logits.
func (m *Model) headGraph() (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	switch {
	case c.Pooling != jlm.PoolNone:
		return nil, UncoveredPooling, nil
	case c.Streams() > 1:
		return nil, UncoveredStreams, nil
	case c.LayerNorm:
		return nil, UncoveredClassic, nil
	}
	f, _ := m.inventory(jlm.DenseBlock)
	out := jlm.RoleOutput
	if _, ok := f.Inventory[out]; !ok {
		out = jlm.RoleTokenEmbd
	}
	b := graph.NewBlock(-1)
	n := b.Norm(b.In("x", c.NEmbd), jlm.RoleOutputNorm, c.RMSEps)
	var bias []jlm.Role
	if _, ok := f.Inventory[jlm.RoleOutputBias]; ok {
		bias = []jlm.Role{jlm.RoleOutputBias}
	}
	l := b.MatVec(n, out, c.NVocab, bias...)
	mapc := func(name string, fn graph.MapFn, k float64, in ...graph.ValueID) graph.ValueID {
		return b.Emit(graph.OpMap, in, []graph.Result{{Name: name, Width: c.NVocab}}, nil, graph.Attrs{Fn: fn, Const: k}, nil, nil)[0]
	}
	if s := float64(c.FinalSoftcap); s != 0 {
		// s*tanh(l/s), tanh(y) as (e-1)/(e+1) with e = exp(2y).
		e := mapc("softcap.exp", graph.MapExp, 0, mapc("softcap.arg", graph.MapMul, 2/s, l))
		t := mapc("softcap.tanh", graph.MapDiv, 0, mapc("softcap.num", graph.MapSub, 1, e), mapc("softcap.den", graph.MapAdd, 1, e))
		l = mapc("capped", graph.MapMul, s, t)
	}
	if s := c.LogitScale; s != 0 && s != 1 {
		l = mapc("scaled", graph.MapMul, 1/s, l)
	}
	b.Out(l)
	if err := graph.Verify(b.G, f); err != nil {
		return nil, UncoveredUnverified, fmt.Errorf("the head's graph does not verify: %w", err)
	}
	return b.G, Covered, nil
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
