package model

import (
	"errors"
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
	UncoveredRouterScore
	UncoveredExperts
	UncoveredGeometry
	UncoveredSelect
	UncoveredStreams
	UncoveredRopeAxes
	UncoveredNonCausal
	UncoveredTemp
	UncoveredCompressed
	UncoveredNGram
	UncoveredAct
	UncoveredPooling
	UncoveredEmbedRows
	UncoveredClamp
	UncoveredUnbound
	UncoveredUnverified
	numUncovered
)

var uncoveredNames = [numUncovered]string{
	Covered:              "covered",
	UncoveredKind:        "a layer whose kind is not an attention block or a recurrent mixer the front end states",
	UncoveredRouterScore: "a router whose score the container does not state",
	UncoveredExperts:     "routed experts weighted at their input over biased banks, in a latent of their own, or stored one matrix each",
	UncoveredGeometry: "a value head of another width than the key's, a layer of its own rotary width turned by the model's rotary, " +
		"or a block reading another's history of another shape than its attention's",
	UncoveredSelect:     "an attention indexer or sparse key selection",
	UncoveredStreams:    "several residual streams",
	UncoveredRopeAxes:   "a rotary over position axes its sections do not state (a pair's halves turned apart)",
	UncoveredNonCausal:  "non-causal attention",
	UncoveredTemp:       "an attention temperature whose position offset is not a whole number",
	UncoveredCompressed: "compressed or grouped attention",
	UncoveredNGram:      "an n-gram embedding",
	UncoveredAct:        "an activation the container does not state, or not over the block's operands",
	UncoveredPooling:    "a pooled head (an embedding model's)",
	UncoveredEmbedRows:  "input rows beyond the token's row, a scale and a position table",
	UncoveredClamp:      "linears clamped to stored bounds",
	UncoveredUnbound:    "stored tensors its graph does not bind",
	UncoveredUnverified: "a graph that does not verify",
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
	blocks []*graph.Graph
	why    []Uncovered
	errs   []error
	// tower, towerWhy and towerErrs are the vision tower's blocks', vision
	// block i at i: block base+i of the model's block space.
	tower     []*graph.Graph
	towerWhy  []Uncovered
	towerErrs []error
	// towerEntry and towerProj are the tower's entry and projector
	// (towerEntryOf, towerProjOf), each with why the front end states none.
	towerEntry                  *graph.Graph
	towerProj                   towerProj
	towerEntryWhy, towerProjWhy Uncovered
	head, embed                 *graph.Graph
	// headHidden is head writing the norm's rows too (nn.Head.HiddenGraph).
	headHidden *graph.Graph
	// cpuHead and cpuHidden are head and headHidden over the row a model of
	// several streams collapses them to before its head, as the host's call
	// sites hand it to the CPU and to a card: the same graphs on a model of
	// one.
	cpuHead, cpuHidden *graph.Graph
	// expand and collapse are the input rows' streams past stream 0 and the
	// row the head reads from every stream (frontstreams.go), nil for a
	// model of one or of streams it does not predict; streamsErr is why a
	// graph of them did not verify.
	expand, collapse *graph.Graph
	streamsErr       error
	// cpuSample and cpuTopK are sample and topK over cpuHead's logits.
	cpuSample, cpuTopK *graph.Graph
	// tables are the step's rotary tables, the model's rotary's then its
	// sliding layers', as jit.Front states them (nil for one that turns
	// nothing); sample is the step's greedy sampling of the head's logits,
	// topK the sampler's candidates over them.
	tables []*graph.Graph
	sample *graph.Graph
	topK   *graph.Graph
	// pick is the label pick over the logits, pickHead the head projecting
	// the picked rows alone.
	pick, pickHead *graph.Graph
	// pool is an embedding model's readout (poolGraph), nil where the model
	// pools nothing or the front end states no graph (poolWhy).
	pool     *graph.Graph
	poolWhy  Uncovered
	poolErr  error
	headWhy  Uncovered
	embedWhy Uncovered
	headErr  error
	embedErr error
	key      frontKey
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
	pool       jlm.Pooling
}

func headKeyOf(c *Config) headKey {
	return headKey{softcap: c.FinalSoftcap, scale: c.LogitScale, eps: c.RMSEps, layerNorm: c.LayerNorm, pool: c.Pooling}
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
	if i := c.m.towerAt(li); i >= 0 {
		return f.tower[i]
	}
	if li < 0 || li >= len(f.blocks) {
		return nil
	}
	return f.blocks[li]
}

// towerAt is block li's index among the vision tower's blocks, -1 for a
// block that is not the tower's.
func (m *Model) towerAt(li int) int {
	t := m.tower
	if t == nil || li < t.base || li >= t.base+t.Cfg.NLayer {
		return -1
	}
	return li - t.base
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
	// A split rotary's table holds a pair an element (graph.RopeSplit).
	if m.Cfg.RopeXD && len(r.Runs) != 0 {
		g.NRot *= 2
	}
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
	case m.towerAt(li) >= 0:
		i := m.towerAt(li)
		return f.towerWhy[i], f.towerErrs[i]
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
	if t := m.tower; t != nil {
		nv := t.Cfg.NLayer
		s.tower, s.towerWhy, s.towerErrs = make([]*graph.Graph, nv), make([]Uncovered, nv), make([]error, nv)
		for i := range nv {
			s.tower[i], s.towerWhy[i], s.towerErrs[i] = m.towerGraph(t.base + i)
		}
		s.towerEntry, s.towerEntryWhy, _ = m.towerEntryOf()
		s.towerProj, s.towerProjWhy, _ = m.towerProjOf(false)
	}
	s.head, s.headWhy, s.headErr = m.headGraph(false)
	if s.head != nil {
		s.headHidden, _, _ = m.headGraph(true)
	}
	s.cpuHead, s.cpuHidden = s.head, s.headHidden
	if m.Cfg.Streams() > 1 {
		s.cpuHead, _, _ = m.headGraphOf(false, true)
		if s.cpuHead != nil {
			s.cpuHidden, _, _ = m.headGraphOf(true, true)
		}
	}
	var eerr, cerr error
	s.expand, eerr = m.streamExpandGraph()
	s.collapse, cerr = m.streamCollapseGraph()
	s.streamsErr = errors.Join(eerr, cerr)
	s.embed, s.embedWhy, s.embedErr = m.embedGraph()
	s.pool, s.poolWhy, s.poolErr = m.poolGraph()
	for i, r := range (composing{m.container, m}).Rotaries() {
		var t *graph.Graph
		if r.NRot > 0 {
			t = jit.TableGraph(r, i)
		}
		s.tables = append(s.tables, t)
	}
	if s.head != nil && len(s.head.Out) == 1 && s.head.Values[s.head.Out[0]].Width == m.Cfg.NVocab {
		sm := graph.Sample{NVocab: m.Cfg.NVocab}
		s.sample = sm.Build()
		tk := graph.SampleTopK{NVocab: m.Cfg.NVocab, K: nn.SampleMaxK, Hist: nn.SampleMaxHist}
		s.topK = tk.Build()
		pk := graph.Pick{NVocab: m.Cfg.NVocab, N: nn.PickMax}
		s.pick = pk.Build()
		s.pickHead, _ = m.pickHeadGraph()
	}
	s.cpuSample, s.cpuTopK = s.sample, s.topK
	if s.cpuSample == nil && s.cpuHead != nil && s.cpuHead.Values[s.cpuHead.Out[0]].Width == m.Cfg.NVocab {
		sm := graph.Sample{NVocab: m.Cfg.NVocab}
		s.cpuSample = sm.Build()
		tk := graph.SampleTopK{NVocab: m.Cfg.NVocab, K: nn.SampleMaxK, Hist: nn.SampleMaxHist}
		s.cpuTopK = tk.Build()
	}
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

// uncovered is why block li's features are not all ones this front end
// writes, Covered for none. It reads numbers the container states, never what the
// model is called.
func (m *Model) uncovered(li int) Uncovered {
	c := m.Cfg
	switch k := c.LayerKind(li); {
	case c.mixerAt(li) != mixNone && !k.Attends():
		return m.uncoveredRest(li)
	case k == jlm.LayerFullAttn || c.mixerAt(li) != mixNone:
	default:
		return UncoveredKind
	}
	switch {
	case c.NRotAt(li) != c.NRot && !(m.ropeSWA != nil && c.SWA(li)) || !c.MLA() && c.HeadDimV != c.HeadDim || !m.sharesGeometry(li):
		return UncoveredGeometry
	case c.IdxHeads != 0:
		return UncoveredSelect
	case c.Streams() > 1 && !c.statesStreams():
		return UncoveredStreams
	case c.RopeAt(li) && !m.ropeAxesStated(m.ropeAt(li)):
		return UncoveredRopeAxes
	case c.NonCausal && c.RopeAt(li) && m.ropeAt(li).NRot == 0:
		// A non-causal block turned by a rotary the model's does not hold
		// (an encoder's of its own bases).
		return UncoveredNonCausal
	case c.AttnTempScale != 0 && (c.AttnTempOffs < 0 || c.AttnTempOffs != math.Trunc(c.AttnTempOffs)):
		return UncoveredTemp
	case len(c.CompKinds) != 0 || c.OGroups != 0:
		return UncoveredCompressed
	}
	return m.uncoveredRest(li)
}

// uncoveredRest is uncovered's reasons that hold for every mixer: the
// residual, the sequence's direction, the input rows and the MLP.
func (m *Model) uncoveredRest(li int) Uncovered {
	c := m.Cfg
	switch {
	case c.Streams() > 1 && !c.statesStreams():
		return UncoveredStreams
	case c.NonCausal && c.mixerAt(li) != mixNone:
		// A recurrent or convolution mixer reading both ways.
		return UncoveredNonCausal
	case c.NGramAt(li):
		return UncoveredNGram
	case m.container.Config().ActAt(li) == nil:
		return UncoveredAct
	case !c.MoEAt(li):
	case len(m.container.Config().RouterScore.Terms) == 0:
		return UncoveredRouterScore
	case c.ExpertWeightIn && (m.stores(li, jlm.RoleExpGateBias) || m.stores(li, jlm.RoleExpUpBias)):
		return UncoveredExperts
	}
	return Covered
}

// sharesGeometry reports whether block li reads a history of its own
// attention's shape: its own, or the one its source block writes, whose
// key heads and head width are li's.
func (m *Model) sharesGeometry(li int) bool {
	c := m.Cfg
	if !c.KVShared(li) {
		return true
	}
	src := c.KVSource(li)
	return src >= 0 && src < li && !c.KVShared(src) && c.HeadDimAt(src) == c.HeadDimAt(li) && c.NKVHeadAt(src) == c.NKVHeadAt(li)
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
// rotary of one position. Under XD-RoPE an element of a head is a pair of
// the table (graph.RopeSplit), and section i is 2*sections[i] of them: the
// chunks of a head's elements its coordinates turn.
func (m *Model) ropeAxes(r nn.Rope) (sections [4]int, interleaved bool) {
	if len(r.Runs) == 0 {
		return sections, false
	}
	if m.Cfg.RopeXD {
		for i, n := range m.Cfg.RopeSections {
			sections[i] = 2 * n
		}
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
	if m.Cfg.RopeXD {
		// Each coordinate's chunk of elements, in order, covering the head's
		// rotated elements: the first halves' runs XDRopeRuns states.
		sum := 0
		for _, n := range sec {
			sum += n
		}
		a, _ := nn.XDRopeRuns(m.Cfg.RopeSections[:], r.NRot/2)
		return !m.Cfg.RopeInterleaved && sum == r.NRot && slices.Equal(a, r.Runs)
	}
	want := nn.MRopeRuns(sec[:])
	if il {
		want = nn.IMRopeRuns(sec[:], r.NRot/2)
	}
	return nn.RopeRuns(r.Runs) == r.NRot/2 && slices.Equal(want, r.Runs)
}

// blockGraph is block li's graph: the attention block the container states
// -- a norm where it stores one (a LayerNorm with its bias where the model's
// norms are mean-centred), q, k and v with their stored biases (q split from
// its gate where the projection carries one), the q/k norm, the rotary, the
// history append, attention of the block's heads, scale, window, softcap and
// sinks, the output gate and the norm inside the sublayer where stored, the
// output projection, a norm over it where stored and the residual, scaled
// where the container states a scale; then the MLP the same way, gated where
// it stores a gate, its activation the expression the container states for
// the block, over the gate and the up or the up alone, reading the residual after the attention or, in a parallel block, the
// block's input -- or why it has none.
func (m *Model) blockGraph(li int) (*graph.Graph, Uncovered, error) {
	if why := m.uncovered(li); why != Covered {
		return nil, why, nil
	}
	c := m.Cfg
	f, perExpert := m.inventory(int32(li))
	if perExpert {
		return nil, UncoveredExperts, nil
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
	qdim, kvDim := c.NHead*hd, nkv*hd
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
	// A projection's input is quantized to int8 per token where the model's
	// are (Config.TokenAct): the product is over the row's codes, then scaled
	// by the row's scale, the bias added after. Inputs shared by several
	// projections are quantized once.
	codes := map[graph.ValueID][2]graph.ValueID{}
	matvec := func(x graph.ValueID, r jlm.Role, rows int, bias ...jlm.Role) graph.ValueID {
		if !c.TokenAct {
			return b.MatVec(x, r, rows, bias...)
		}
		q, ok := codes[x]
		if !ok {
			q[0], q[1] = b.Int8Codes(x)
			codes[x] = q
		}
		y := mapRow(b, "scaled", graph.Attrs{Fn: graph.MapMul}, nil, rows, b.MatVec(q[0], r, rows), q[1])
		for _, br := range bias {
			y = mapRow(b, "biased", graph.Attrs{Fn: graph.MapAdd}, []graph.Binding{{Role: br, K: rows}}, rows, y)
		}
		return y
	}
	x := b.In("x", d)
	// A block over several residual streams reads each as an input of its
	// own and runs on stream 0's prediction (frontstreams.go).
	var st *streams
	if c.statesStreams() {
		st = &streams{in: []graph.ValueID{x}, mixed: c.mixesStreams(), gated: c.gatesStreams()}
		for k := 1; k < c.Streams(); k++ {
			st.in = append(st.in, b.In(fmt.Sprintf("x.%d", k), d))
		}
		if _, ok := c.resCheckpoint(li); ok && st.mixed {
			st.bank = streamCopy(b, x)
		}
	}
	// A model of per-layer embeddings gives each block the row's every
	// layer's input (the input rows' second output): the block reads its own.
	if c.PLEDim != 0 {
		b.In("ple", c.pleWidth())
	}
	attnIn := x
	switch {
	case st != nil && st.mixed:
		attnIn = m.bankMix(b, f, st.in[1:1+c.resBank(li, false)], x, jlm.RoleAttnResScore)
	case st != nil && st.gated:
		attnIn, st.w = m.gatedMix(b, f, st.in, jlm.RoleHCAttnNorm, jlm.RoleHCAttnDown, jlm.RoleHCAttnUp, jlm.RoleHCAttnInject)
	case st != nil:
		st.pred = m.predicted(b, f, st.in)
		x, attnIn = st.pred[0], st.pred[0]
	}
	n := preNorm(attnIn, jlm.RoleAttnNorm, jlm.RoleAttnNormBias)
	if st != nil && has(jlm.RoleLowRankIn) {
		st.side = m.lowRank(b, f, n)
	}
	var mix graph.ValueID
	var err error
	switch c.mixerAt(li) {
	case mixDelta:
		mix, err = m.deltaMixer(b, f, n, has, matvec)
	case mixSelective:
		mix, err = m.ssdMixer(b, f, n, has, matvec)
	case mixScan:
		mix, err = m.scanMixer(b, f, n, has, matvec)
	case mixShortConv:
		mix, err = m.shortConvMixer(b, n, matvec)
	}
	if err != nil {
		return nil, UncoveredUnverified, err
	}
	// A recurrent block's mixer is the block's own, or, where the block
	// attends as well, projected beside the attention's output.
	var side []graph.ValueID
	if c.mixerAt(li) != mixNone {
		if !c.LayerKind(li).Attends() {
			return m.blockRest(b, f, li, x, n, mix, jlm.RoleSSMOut, preNorm, norm, resid, matvec, stored, has, st)
		}
		side = append(side, matvec(mix, jlm.RoleSSMOut, d))
	}
	// With an output gate the q projection is double width: each head's query,
	// then that head's gate.
	qrows := qdim
	if c.AttnOutGate {
		qrows = 2 * qdim
	}
	// A query through a latent: down, normed, and up where the block stores
	// the two projections.
	qIn, qRole := n, jlm.RoleAttnQ
	if has(jlm.RoleAttnQA) {
		qa := f.Inventory[jlm.RoleAttnQA]
		qIn, qRole = b.Norm(matvec(n, jlm.RoleAttnQA, qa.Rows), jlm.RoleAttnQANorm, c.LatentNormEps), jlm.RoleAttnQB
	}
	q := matvec(qIn, qRole, qrows, stored(jlm.RoleAttnQBias)...)
	if has(jlm.RoleAttnKVA) {
		att, err := m.latentAttention(b, li, n, q, has, stored, matvec)
		if err != nil {
			return nil, UncoveredUnverified, err
		}
		return m.blockRest(b, f, li, x, n, att, jlm.RoleAttnOut, preNorm, norm, resid, matvec, stored, has, st, side...)
	}
	// A block that stores no v projection reads k's as its value, before
	// k's norm and rotary.
	// A block reading another's history (Config.KVShared) projects q alone:
	// its keys and values are the ones its source block wrote.
	shared := c.KVShared(li)
	vFromK := m.layers[li].vFromK
	var k, v graph.ValueID
	if !shared {
		k = matvec(n, jlm.RoleAttnK, kvDim, stored(jlm.RoleAttnKBias)...)
		v = k
		if !vFromK {
			v = matvec(n, jlm.RoleAttnV, kvDim, stored(jlm.RoleAttnVBias)...)
		}
	}
	var gate graph.ValueID
	if c.AttnOutGate {
		both := q
		half := func(name string, off int) graph.ValueID {
			return b.Emit(graph.OpSlice, []graph.ValueID{both}, []graph.Result{{Name: name, Width: qdim}}, nil,
				graph.Attrs{HeadDim: 2 * hd, Offset: off, Width: hd}, nil, nil)[0]
		}
		q, gate = half("q", 0), half("gate", hd)
	}
	if bound := c.ClampKQV; bound != 0 {
		clamp := func(x graph.ValueID, name string) graph.ValueID {
			return b.Emit(graph.OpClamp, []graph.ValueID{x}, []graph.Result{{Name: name, Width: b.G.Values[x].Width}}, nil,
				graph.Attrs{Bound: float64(bound)}, nil, nil)[0]
		}
		q = clamp(q, "q.clamp")
		switch {
		case shared:
		case vFromK:
			k = clamp(k, "k.clamp")
			v = k
		default:
			k, v = clamp(k, "k.clamp"), clamp(v, "v.clamp")
		}
	}
	// ones is an RMS norm over each head of x with no weight.
	ones := func(x graph.ValueID, name string) graph.ValueID {
		return b.Emit(graph.OpHeadNorm, []graph.ValueID{x}, []graph.Result{{Name: name, Width: b.G.Values[x].Width}}, nil,
			graph.Attrs{Eps: c.RMSEps, HeadDim: hd}, nil, nil)[0]
	}
	// A q norm after the rotary keeps k's before it: the converter folded k's
	// weight into q's, and an unweighted norm commutes with the rotation.
	// Under XD-RoPE k's norm follows the rotary too: a pair's elements
	// turned apart do not keep a head's RMS.
	postQ := c.QKNormAt(li) && c.QKNormPost
	postK := postQ && c.RopeXD && c.RopeAt(li) && !shared
	// onK is f over k where the block writes keys of its own.
	onK := func(f func(graph.ValueID) graph.ValueID) {
		if !shared {
			k = f(k)
		}
	}
	switch {
	case postK:
	case postQ:
		onK(func(k graph.ValueID) graph.ValueID { return b.HeadNorm(k, jlm.RoleAttnKNorm, hd, c.RMSEps) })
	case c.QKNormAt(li) && c.QKNormWide:
		q = b.Norm(q, jlm.RoleAttnQNorm, c.RMSEps)
		onK(func(k graph.ValueID) graph.ValueID { return b.Norm(k, jlm.RoleAttnKNorm, c.RMSEps) })
	case c.QKNormAt(li):
		q = b.HeadNorm(q, jlm.RoleAttnQNorm, hd, c.RMSEps)
		onK(func(k graph.ValueID) graph.ValueID { return b.HeadNorm(k, jlm.RoleAttnKNorm, hd, c.RMSEps) })
	}
	if c.RopeAt(li) {
		r := m.ropeAt(li)
		layout := graph.RopeNorm
		switch {
		case c.RopeXD && len(r.Runs) != 0:
			layout = graph.RopeSplit
		case c.RopeNeox:
			layout = graph.RopeNeox
		}
		a := graph.Attrs{HeadDim: hd, NRot: r.NRot, Base: r.Base, Layout: layout, Freqs: len(r.Freqs) > 0, Mscale: r.Scale}
		a.Sections, a.Interleaved = m.ropeAxes(r)
		q = b.Emit(graph.OpRope, []graph.ValueID{q}, []graph.Result{{Name: "q.rope", Width: qdim}}, nil, a, nil, nil)[0]
		onK(func(k graph.ValueID) graph.ValueID {
			return b.Emit(graph.OpRope, []graph.ValueID{k}, []graph.Result{{Name: "k.rope", Width: kvDim}}, nil, a, nil, nil)[0]
		})
		if c.QKL2Norm {
			q = ones(q, "q.l2")
			onK(func(k graph.ValueID) graph.ValueID { return ones(k, "k.l2") })
		}
	}
	if postQ {
		q = b.HeadNorm(q, jlm.RoleAttnQNorm, hd, c.RMSEps)
	}
	if postK {
		k = b.HeadNorm(k, jlm.RoleAttnKNorm, hd, c.RMSEps)
	}
	if c.VNorm && !shared {
		v = ones(v, "v.norm")
	}
	// The attention temperature, 1 + s ln(1 + floor((pos + o) / f)) at the
	// row's history position, multiplies q: the scores are linear in it.
	if s := c.AttnTempScale; s != 0 && !(c.NoPEGlobal && c.RopeAt(li)) {
		pos := b.Emit(graph.OpPosition, nil, []graph.Result{{Name: "pos", Width: 1}}, nil,
			graph.Attrs{Offset: int(c.AttnTempOffs), Span: int(c.AttnTempFloor)}, nil, nil)[0]
		temp := b.Expr("temp", 1, []graph.Term{
			{Fn: graph.MapAdd, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}, {Kind: graph.ArgConst, Const: 1}}},
			{Fn: graph.MapLog, Args: [3]graph.Arg{{Kind: graph.ArgTerm, Index: 0}}},
			{Fn: graph.MapFma, Args: [3]graph.Arg{{Kind: graph.ArgTerm, Index: 1}, {Kind: graph.ArgConst, Const: s}, {Kind: graph.ArgConst, Const: 1}}},
		}, nil, pos)
		q = mapRow(b, "q.temp", graph.Attrs{Fn: graph.MapMul}, nil, qdim, q, temp)
	}
	hist := graph.SharedHistory(c.KVSource(li))
	if !shared {
		hist = graph.StateRef{Name: "kv", Version: b.KVAppend(k, v, 0)}
	}
	scale := c.AttnScale
	if scale == 0 {
		scale = 1 / math.Sqrt(float64(hd))
	}
	a := graph.Attrs{NHead: c.NHead, NKVHead: nkv, HeadDim: hd, Scale: scale, Softcap: float64(c.AttnSoftcap), Bidir: c.NonCausal}
	if c.SWA(li) {
		a.Window, a.Chunked = c.SWAWindow, c.SWAChunked
	}
	att := b.AttendOf(q, hist, a, stored(jlm.RoleAttnSinks)...)
	if c.AttnOutGate {
		att = sigmoidMul(b, "gated", gate, att)
	}
	return m.blockRest(b, f, li, x, n, att, jlm.RoleAttnOut, preNorm, norm, resid, matvec, stored, has, st, side...)
}

// blockRest is block li's graph after its mixer's output att: the norm
// inside the sublayer where stored, the output projection out plus each
// side output (a second mixer's, projected), a norm over it where stored
// and the residual; then the MLP, where the block stores one.
func (m *Model) blockRest(b *graph.Builder, f graph.Facts, li int, x, n, att graph.ValueID, out jlm.Role,
	preNorm func(graph.ValueID, jlm.Role, jlm.Role) graph.ValueID, norm func(graph.ValueID, jlm.Role) graph.ValueID,
	resid func(graph.ValueID, graph.ValueID) graph.ValueID, matvec func(graph.ValueID, jlm.Role, int, ...jlm.Role) graph.ValueID,
	stored func(jlm.Role) []jlm.Role, has func(jlm.Role) bool, st *streams, side ...graph.ValueID) (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	d, nffn := c.NEmbd, c.NFFNAt(li)
	// A post-norm block normalises each sublayer's sum after its
	// residual add, where it stores the norm.
	postResid := func(v graph.ValueID, r, bias jlm.Role) graph.ValueID {
		if !has(r) {
			return v
		}
		if c.LayerNorm {
			return b.LayerNorm(v, r, c.RMSEps, stored(bias)...)
		}
		return b.Norm(v, r, c.RMSEps)
	}
	att = norm(att, jlm.RoleAttnSubNorm)
	proj := matvec(att, out, d, stored(jlm.RoleAttnOutBias)...)
	for _, s := range side {
		proj = b.Add(proj, s)
	}
	// Streams read through a gate take the sublayer's output each at its
	// weight; x2 is then none of them.
	var x2 graph.ValueID
	if st != nil && st.gated {
		st.cur = m.placed(b, st.in, st.w, norm(proj, jlm.RolePostAttnNorm))
	} else {
		x2 = postResid(resid(x, norm(proj, jlm.RolePostAttnNorm)), jlm.RoleAttnOutNorm, jlm.RoleAttnOutNormBias)
	}
	if _, ok := c.resCheckpoint(li); ok && st != nil && st.mixed {
		x2 = norm(proj, jlm.RolePostAttnNorm)
	}
	if st != nil && st.side != 0 {
		x2 = mapRow(b, "joined", graph.Attrs{Fn: graph.MapMul, Const: 1 / math.Sqrt2}, nil, d, b.Add(x2, st.side))
	}
	if !c.MoEAt(li) && !has(jlm.RoleFFNUp) && !has(jlm.RoleFFNDown) {
		if st != nil && st.gated {
			b.Out(st.cur...)
			return verified(b, f)
		}
		if st != nil && st.mixed {
			return m.banked(b, f, li, st, x2, has)
		}
		if st != nil {
			return m.corrected(b, f, li, st, x2, has)
		}
		y, why, err := m.perLayer(b, f, li, x2)
		if err != nil {
			return nil, why, err
		}
		b.Out(outScaled(b, has, y))
		return verified(b, f)
	}
	// A parallel block's MLP reads the block's input: through its own norm
	// where it stores one, through the attention's where the two share one.
	ffnIn := x2
	switch {
	case c.Parallel:
		ffnIn = x
	case st != nil && st.mixed:
		ffnIn = m.bankMix(b, f, st.bankAfter(c, li), x2, jlm.RoleFFNResScore)
	case st != nil && st.gated:
		ffnIn, st.w = m.gatedMix(b, f, st.cur, jlm.RoleHCFFNNorm, jlm.RoleHCFFNDown, jlm.RoleHCFFNUp, jlm.RoleHCFFNInject)
	}
	n2 := preNorm(ffnIn, jlm.RoleFFNNorm, jlm.RoleFFNNormBias)
	if _, own := f.Inventory[jlm.RoleFFNNorm]; c.Parallel && !own {
		n2 = n
	}
	var ffn graph.ValueID
	if c.MoEAt(li) {
		var why Uncovered
		var err error
		if ffn, why, err = m.mixture(b, f, li, ffnIn, n2, has, stored); err != nil {
			return nil, why, err
		}
	} else {
		var acts []graph.ValueID
		if has(jlm.RoleFFNGate) {
			acts = append(acts, matvec(n2, jlm.RoleFFNGate, nffn))
		}
		acts = append(acts, matvec(n2, jlm.RoleFFNUp, nffn, stored(jlm.RoleFFNUpBias)...))
		h, err := b.Stated("act", m.container.Config().ActAt(li), acts, nffn, f)
		if err != nil {
			return nil, UncoveredAct, fmt.Errorf("the block's stated activation: %w", err)
		}
		h = norm(h, jlm.RoleFFNSubNorm)
		ffn = matvec(h, jlm.RoleFFNDown, d, stored(jlm.RoleFFNDownBias)...)
	}
	if st != nil && st.gated {
		b.Out(m.placed(b, st.cur, st.w, norm(ffn, jlm.RolePostFFNNorm))...)
		return verified(b, f)
	}
	if st != nil && st.mixed {
		return m.banked(b, f, li, st, resid(x2, norm(ffn, jlm.RolePostFFNNorm)), has)
	}
	if st != nil {
		return m.corrected(b, f, li, st, resid(x2, norm(ffn, jlm.RolePostFFNNorm)), has)
	}
	y, why, err := m.perLayer(b, f, li, postResid(resid(x2, norm(ffn, jlm.RolePostFFNNorm)), jlm.RoleLayerOutNorm, jlm.RoleLayerOutNormBias))
	if err != nil {
		return nil, why, err
	}
	b.Out(outScaled(b, has, y))
	return verified(b, f)
}

// perLayer is block li's per-layer embedding over its residual y, where the
// graph reads the row's per-layer inputs (its second input): y plus the
// normed projection of the block's stated activation over its gate of y and
// the block's slice of the inputs. y itself where it reads none.
func (m *Model) perLayer(b *graph.Builder, f graph.Facts, li int, y graph.ValueID) (graph.ValueID, Uncovered, error) {
	if m.Cfg.PLEDim == 0 {
		return y, Covered, nil
	}
	o, why, err := m.perLayerOf(b, f, li, y)
	if err != nil {
		return 0, why, err
	}
	return b.Add(y, o), Covered, nil
}

// perLayerOf is the normed projection of block li's stated activation over
// its gate of g and the block's slice of the row's per-layer inputs (the
// graph's last input).
func (m *Model) perLayerOf(b *graph.Builder, f graph.Facts, li int, g graph.ValueID) (graph.ValueID, Uncovered, error) {
	c := m.Cfg
	p := c.PLEDim
	gate := b.MatVec(g, jlm.RolePLEGate, p)
	in := b.Emit(graph.OpSlice, []graph.ValueID{b.G.In[len(b.G.In)-1]}, []graph.Result{{Name: "ple.in", Width: p}}, nil,
		graph.Attrs{Offset: li * p, Width: p}, nil, nil)[0]
	h, err := b.Stated("ple.act", m.plainActAt(li), []graph.ValueID{gate, in}, p, f)
	if err != nil {
		return 0, UncoveredAct, fmt.Errorf("the per-layer input's stated activation: %w", err)
	}
	return b.Norm(b.MatVec(h, jlm.RolePLEProj, c.NEmbd), jlm.RolePLEPostNorm, c.RMSEps), Covered, nil
}

// outScaled is the block's output y times the scalar the block stores
// (jlm.RoleLayerOutScale), y where it stores none.
func outScaled(b *graph.Builder, has func(jlm.Role) bool, y graph.ValueID) graph.ValueID {
	if !has(jlm.RoleLayerOutScale) {
		return y
	}
	return mapRow(b, "out.scaled", graph.Attrs{Fn: graph.MapMul, Width: 1}, []graph.Binding{{Role: jlm.RoleLayerOutScale, K: 1}},
		b.G.Values[y].Width, y)
}

// verified is b's graph where it verifies against f and binds every tensor
// f stores.
func verified(b *graph.Builder, f graph.Facts) (*graph.Graph, Uncovered, error) {
	if err := graph.Verify(b.G, f); err != nil {
		return nil, UncoveredUnverified, fmt.Errorf("the block's graph does not verify: %w", err)
	}
	if u := graph.Unbound(b.G, f); len(u) != 0 {
		return nil, UncoveredUnbound, fmt.Errorf("the container stores %v for the block, which its graph does not bind", u)
	}
	return b.G, Covered, nil
}

// mixture is block li's mixture of experts over its normed row n: the
// router's logits (its matrix, with its stored bias), each expert's score
// the expression the container states over them; the selection ranks the
// scores plus the stored selection bias, within the best groups where the
// container states groups (a group scored by the sum of its best two, or
// its one, the rest of the experts out of reach); each selected expert's
// weight its unbiased score, renormalised by the weights' sum unless the
// container says not, and times the routed scale. Each selected expert's
// gate and up from its sheet of the banks, with their stored biases, the
// block's stated activation, and its down; the downs summed at their
// weights. Then the shared expert where the block stores one, scaled by the
// sigmoid of its gate's logit where it stores a gate.
// Where the block stores a router norm or an experts' pre-norm, the router
// or the experts read the block's unnormed x through it rather than n; a
// stored per-expert scale multiplies each selected weight last; and where
// it stores a post-norm of the shared expert's or of the routed sum's, each
// is normed before the two are added (a dense MLP beside the mixture).
func (m *Model) mixture(b *graph.Builder, f graph.Facts, li int, x, n graph.ValueID, has func(jlm.Role) bool,
	stored func(jlm.Role) []jlm.Role) (graph.ValueID, Uncovered, error) {
	c := m.Cfg
	d, E, K := c.NEmbd, c.NExpert, c.NExpertUsed
	normed := func(v graph.ValueID, r jlm.Role) graph.ValueID {
		if has(r) {
			return b.Norm(v, r, c.RMSEps)
		}
		return v
	}
	rin, ein := n, n
	if has(jlm.RoleRouterNorm) {
		rin = normed(x, jlm.RoleRouterNorm)
	}
	if has(jlm.RoleFFNNorm2) {
		ein = normed(x, jlm.RoleFFNNorm2)
	}
	if has(jlm.RoleFFNRoutedDown) {
		ein = b.MatVec(n, jlm.RoleFFNRoutedDown, f.Inventory[jlm.RoleFFNRoutedDown].Rows)
	}
	up, ok := f.Inventory[jlm.RoleExpUpBank]
	if !ok || up.Sheets != E || E < 1 || K < 1 || K > E {
		return 0, UncoveredUnverified, fmt.Errorf("a mixture choosing %d of %d experts over a bank of %d sheets", K, E, up.Sheets)
	}
	F := up.Rows
	// A latent mixture's routed experts run at its width: their input the
	// routed down projection of n, their weighted sum normed where stored
	// and taken back to n_embd.
	ew := d
	if lat, ok := f.Inventory[jlm.RoleFFNRoutedDown]; ok {
		ew = lat.Rows
	}
	mapc := func(name string, fn graph.MapFn, k float64, width int, w []graph.Binding, in ...graph.ValueID) graph.ValueID {
		return b.Emit(graph.OpMap, in, []graph.Result{{Name: name, Width: width}}, w, graph.Attrs{Fn: fn, Const: k}, nil, nil)[0]
	}
	reduce := func(name string, x graph.ValueID, span, width int) graph.ValueID {
		return b.Emit(graph.OpReduce, []graph.ValueID{x}, []graph.Result{{Name: name, Width: width}}, nil,
			graph.Attrs{Fn: graph.MapAdd, Span: span}, nil, nil)[0]
	}
	topk := func(name string, x graph.ValueID, k, span, width int) (sel, best graph.ValueID) {
		o := b.Emit(graph.OpTopK, []graph.ValueID{x}, []graph.Result{{Name: name + ".sel", Width: width, Elem: graph.ElemU32},
			{Name: name + ".best", Width: width}}, nil, graph.Attrs{TopK: k, Span: span}, nil, nil)
		return o[0], o[1]
	}
	// stored1 binds the block's vector r at the shape it is stored.
	stored1 := func(r jlm.Role) []graph.Binding {
		s := f.Inventory[r]
		return []graph.Binding{{Role: r, K: s.K, Rows: s.Rows}}
	}
	logits := b.MatVec(rin, jlm.RoleRouter, E, stored(jlm.RoleRouterBias)...)
	score, err := b.Stated("router.score", &m.container.Config().RouterScore, []graph.ValueID{logits}, E, f)
	if err != nil {
		return 0, UncoveredRouterScore, fmt.Errorf("the router's stated score: %w", err)
	}
	choose := score
	if has(jlm.RoleExpProbsB) {
		choose = mapc("router.biased", graph.MapAdd, 0, E, stored1(jlm.RoleExpProbsB), score)
	}
	if g, kept := c.NExpertGroup, c.NExpertGroupUsed; g > 1 {
		if E%g != 0 || kept < 1 || kept > g {
			return 0, UncoveredUnverified, fmt.Errorf("%d experts in %d groups, %d kept", E, g, kept)
		}
		take := min(2, E/g)
		_, best := topk("router.group", choose, take, E/g, g*take)
		rank := b.Emit(graph.OpRank, []graph.ValueID{reduce("router.group.score", best, take, g)},
			[]graph.Result{{Name: "router.group.rank", Width: g, Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)[0]
		choose = mapc("router.kept", graph.MapWhere, math.Inf(-1), E, nil, mapc("router.group.kept", graph.MapLess, float64(kept), g, nil, rank), choose)
	}
	sel, _ := topk("router", choose, K, 0, K)
	take := func(name string, v graph.ValueID) graph.ValueID {
		return b.Emit(graph.OpTake, []graph.ValueID{v, sel}, []graph.Result{{Name: name, Width: K}}, nil, graph.Attrs{}, nil, nil)[0]
	}
	var w graph.ValueID
	if rw := &m.container.Config().RouterWeight; len(rw.Terms) != 0 {
		// Each pick's weight is the stated row over the logits with the
		// earlier picks removed and the pick's logit, read at the pick.
		slice1 := func(name string, v graph.ValueID, k int) graph.ValueID {
			return b.Emit(graph.OpSlice, []graph.ValueID{v}, []graph.Result{{Name: name, Width: 1}}, nil, graph.Attrs{Offset: k, Width: 1}, nil, nil)[0]
		}
		rank := b.Emit(graph.OpRank, []graph.ValueID{choose}, []graph.Result{{Name: "router.rank", Width: E, Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)[0]
		picked := take("router.picked", logits)
		var ws []graph.ValueID
		for k := range K {
			rest := logits
			if k > 0 {
				earlier := mapc("router.earlier", graph.MapLess, float64(k), E, nil, rank)
				rest = b.Expr("router.rest", E, []graph.Term{{Fn: graph.MapWhere, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0},
					{Kind: graph.ArgConst, Const: math.Inf(-1)}, {Kind: graph.ArgIn, Index: 1}}}}, nil, earlier, logits)
			}
			row, err := b.Stated("router.weight", rw, []graph.ValueID{rest, slice1("router.pick", picked, k)}, E, f)
			if err != nil {
				return 0, UncoveredRouterScore, fmt.Errorf("the router's stated weight: %w", err)
			}
			ws = append(ws, slice1("router.w", take("router.row", row), k))
		}
		w = ws[0]
		if K > 1 {
			w = b.Emit(graph.OpConcat, ws, []graph.Result{{Name: "router.w", Width: K}}, nil, graph.Attrs{}, nil, nil)[0]
		}
	} else {
		w = take("router.w", score)
	}
	if !c.NoExpertNorm {
		w = mapc("router.wn", graph.MapDiv, 0, K, nil, w, mapc("router.sum", graph.MapAdd, 1e-20, 1, nil, reduce("router.wsum", w, 0, 1)))
	}
	if s := c.ExpertScale; s != 0 && s != 1 {
		w = mapc("router.scaled", graph.MapMul, s, K, nil, w)
	}
	if has(jlm.RoleExpScale) {
		es := b.Emit(graph.OpSlice, nil, []graph.Result{{Name: "router.expscale", Width: E}}, stored1(jlm.RoleExpScale),
			graph.Attrs{Width: E}, nil, nil)[0]
		es = b.Emit(graph.OpTake, []graph.ValueID{es, sel}, []graph.Result{{Name: "router.expscale.sel", Width: K}}, nil, graph.Attrs{}, nil, nil)[0]
		w = mapc("router.expscaled", graph.MapMul, 0, K, nil, w, es)
	}
	bank := func(name string, x graph.ValueID, r, bias jlm.Role, k, rows int) graph.ValueID {
		ws := []graph.Binding{{Role: r, K: k, Rows: rows, Sheets: E}}
		if has(bias) {
			ws = append(ws, graph.Binding{Role: bias, K: rows, Rows: E})
		}
		return b.Emit(graph.OpExpertMatVec, []graph.ValueID{x, sel}, []graph.Result{{Name: name, Width: K * rows}}, ws,
			graph.Attrs{}, nil, nil)[0]
	}
	in := b.Emit(graph.OpTile, []graph.ValueID{ein}, []graph.Result{{Name: "experts.in", Width: K * ew}}, nil, graph.Attrs{Span: ew}, nil, nil)[0]
	var acts []graph.ValueID
	if has(jlm.RoleExpGateBank) {
		acts = append(acts, bank("experts.gate", in, jlm.RoleExpGateBank, jlm.RoleExpGateBias, ew, F))
	}
	acts = append(acts, bank("experts.up", in, jlm.RoleExpUpBank, jlm.RoleExpUpBias, ew, F))
	// Weighted at its input, an expert's gate and up are its weight times
	// the projections (both linear), and its down is summed at one.
	if c.ExpertWeightIn {
		for i, a := range acts {
			acts[i] = mapc("experts.in.weighted", graph.MapMul, 0, K*F, nil, a, w)
		}
	}
	h, err := b.Stated("experts.act", m.container.Config().ActAt(li), acts, K*F, f)
	if err != nil {
		return 0, UncoveredAct, fmt.Errorf("the experts' stated activation: %w", err)
	}
	down := bank("experts.down", h, jlm.RoleExpDownBank, jlm.RoleExpDownBias, F, ew)
	if !c.ExpertWeightIn {
		down = mapc("experts.weighted", graph.MapMul, 0, K*ew, nil, down, w)
	}
	var out graph.ValueID
	for j := range K {
		y := b.Emit(graph.OpSlice, []graph.ValueID{down}, []graph.Result{{Name: "experts.out", Width: ew}}, nil,
			graph.Attrs{Offset: j * ew, Width: ew}, nil, nil)[0]
		if j == 0 {
			out = y
		} else {
			out = b.Add(out, y)
		}
	}
	out = normed(out, jlm.RolePostFFNNorm2)
	if has(jlm.RoleFFNRoutedUp) {
		out = b.MatVec(normed(out, jlm.RoleFFNRoutedNorm), jlm.RoleFFNRoutedUp, d)
	}
	sh, ok := f.Inventory[jlm.RoleShExpUp]
	if !ok {
		return out, Covered, nil
	}
	var sa []graph.ValueID
	if has(jlm.RoleShExpGate) {
		sa = append(sa, b.MatVec(n, jlm.RoleShExpGate, sh.Rows))
	}
	sa = append(sa, b.MatVec(n, jlm.RoleShExpUp, sh.Rows))
	sh1, err := b.Stated("shared.act", m.container.Config().ActAt(li), sa, sh.Rows, f)
	if err != nil {
		return 0, UncoveredAct, fmt.Errorf("the shared expert's stated activation: %w", err)
	}
	so := b.MatVec(sh1, jlm.RoleShExpDown, d)
	if has(jlm.RoleShRouter) {
		logit := reduce("shared.logit", mapc("shared.gate", graph.MapMul, 0, d, stored1(jlm.RoleShRouter), n), 0, 1)
		so = sigmoidMul(b, "shared.gated", logit, so)
	}
	return b.Add(out, normed(so, jlm.RolePostFFNNorm1)), Covered, nil
}

// latentAttention is the attention of a block that projects its history
// through a latent (the container stores RoleAttnKVA): the latent row, its
// first KVLoraRank normed and the rest a key turned by the rotary, cached
// once a position for every head; each head's query its head's first
// HeadDim-NRot through its sheet of RoleAttnKB, joined with the rest turned
// by the rotary, so a query row is the cached row's width; attention over
// the cached rows, each value the whole row, with the block's scale, window,
// softcap and sinks; each head's first KVLoraRank of the result through its
// sheet of RoleAttnVB, gated where the block stores a gate over the output.
func (m *Model) latentAttention(b *graph.Builder, li int, n, q graph.ValueID, has func(jlm.Role) bool, stored func(jlm.Role) []jlm.Role,
	matvec func(graph.ValueID, jlm.Role, int, ...jlm.Role) graph.ValueID) (graph.ValueID, error) {
	c := m.Cfg
	nh, hd, lat, rot, hdv := c.NHead, c.HeadDimAt(li), c.KVLoraRank, c.NRot, c.HeadDimV
	nope, row := hd-rot, lat+rot
	switch {
	case lat <= 0 || rot <= 0 || nope <= 0 || hdv <= 0:
		return 0, fmt.Errorf("a latent of %d under heads of %d, %d rotated, values of %d", lat, hd, rot, hdv)
	case c.AttnOutGate || c.QKNormAt(li) || c.TokenAct:
		return 0, fmt.Errorf("a latent attention with a gated query (%v), a q/k norm (%v) or int8 activations (%v), which this statement does not write",
			c.AttnOutGate, c.QKNormAt(li), c.TokenAct)
	}
	slice := func(v graph.ValueID, name string, headDim, off, width, out int) graph.ValueID {
		return b.Emit(graph.OpSlice, []graph.ValueID{v}, []graph.Result{{Name: name, Width: out}}, nil,
			graph.Attrs{HeadDim: headDim, Offset: off, Width: width}, nil, nil)[0]
	}
	sheets := func(v graph.ValueID, name string, r jlm.Role, k, rows int) graph.ValueID {
		return b.Emit(graph.OpExpertMatVec, []graph.ValueID{v}, []graph.Result{{Name: name, Width: nh * rows}},
			[]graph.Binding{{Role: r, K: k, Rows: rows, Sheets: nh}}, graph.Attrs{}, nil, nil)[0]
	}
	kv := matvec(n, jlm.RoleAttnKVA, row)
	latent := b.Norm(slice(kv, "latent", 0, 0, lat, lat), jlm.RoleAttnKVANorm, c.LatentNormEps)
	kpe := slice(kv, "k.pe", 0, lat, rot, rot)
	qpe := slice(q, "q.pe", hd, nope, rot, nh*rot)
	if c.RopeAt(li) {
		r := m.ropeAt(li)
		layout := graph.RopeNorm
		if c.RopeNeox {
			layout = graph.RopeNeox
		}
		a := graph.Attrs{HeadDim: rot, NRot: r.NRot, Base: r.Base, Layout: layout, Freqs: len(r.Freqs) > 0, Mscale: r.Scale}
		a.Sections, a.Interleaved = m.ropeAxes(r)
		qpe = b.Emit(graph.OpRope, []graph.ValueID{qpe}, []graph.Result{{Name: "q.pe.rope", Width: nh * rot}}, nil, a, nil, nil)[0]
		kpe = b.Emit(graph.OpRope, []graph.ValueID{kpe}, []graph.Result{{Name: "k.pe.rope", Width: rot}}, nil, a, nil, nil)[0]
	}
	abs := sheets(slice(q, "q.nope", hd, 0, nope, nh*nope), "q.abs", jlm.RoleAttnKB, nope, lat)
	qrow := b.Emit(graph.OpConcat, []graph.ValueID{abs, qpe}, []graph.Result{{Name: "q.row", Width: nh * row}}, nil,
		graph.Attrs{NHead: nh}, nil, nil)[0]
	krow := b.Emit(graph.OpConcat, []graph.ValueID{latent, kpe}, []graph.Result{{Name: "kv.row", Width: row}}, nil,
		graph.Attrs{}, nil, nil)[0]
	hist := b.KVAppend(krow, krow, 0)
	scale := c.AttnScale
	if scale == 0 {
		scale = 1 / math.Sqrt(float64(hd))
	}
	a := graph.Attrs{NHead: nh, NKVHead: 1, HeadDim: row, Scale: scale, Softcap: float64(c.AttnSoftcap)}
	if c.SWA(li) {
		a.Window, a.Chunked = c.SWAWindow, c.SWAChunked
	}
	att := b.Attend(qrow, hist, a, stored(jlm.RoleAttnSinks)...)
	o := sheets(slice(att, "att.latent", row, 0, lat, nh*lat), "att.v", jlm.RoleAttnVB, lat, hdv)
	if has(jlm.RoleAttnGate) {
		o = sigmoidMul(b, "att.gated", matvec(n, jlm.RoleAttnGate, nh*hdv), o)
	}
	return o, nil
}

// sigmoidMul is sigma(gate) * x, gate as wide as x or a scalar over it.
func sigmoidMul(b *graph.Builder, name string, gate, x graph.ValueID) graph.ValueID {
	return b.Expr(name, b.G.Values[x].Width, []graph.Term{
		{Fn: graph.MapSigmoid, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0}}},
		{Fn: graph.MapMul, Args: [3]graph.Arg{{Kind: graph.ArgTerm, Index: 0}, {Kind: graph.ArgIn, Index: 1}}},
	}, nil, gate, x)
}

// mapRow is one OpMap of in by a into a row of width, binding w.
func mapRow(b *graph.Builder, name string, a graph.Attrs, w []graph.Binding, width int, in ...graph.ValueID) graph.ValueID {
	return b.Emit(graph.OpMap, in, []graph.Result{{Name: name, Width: width}}, w, a, nil, nil)[0]
}

// headGraph is the head (graph.Head) of what the container stores: a
// LayerNorm with its stored bias where the model's norms are LayerNorms, an
// RMSNorm otherwise, the projection to the vocabulary (the embedding's where
// the model ties them) with its stored bias, then the final softcap and the
// logit scale, main's order over the logits. With hidden it writes the
// norm's rows too (graph.Head.Hidden).
func (m *Model) headGraph(hidden bool) (*graph.Graph, Uncovered, error) {
	return m.headGraphOf(hidden, false)
}

// headGraphOf is headGraph, with collapsed over the row a model of several
// streams collapses them to, which the head's graph does not read
// otherwise.
func (m *Model) headGraphOf(hidden, collapsed bool) (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	switch {
	case c.Pooling != jlm.PoolNone:
		return nil, UncoveredPooling, nil
	case c.Streams() > 1 && !collapsed:
		return nil, UncoveredStreams, nil
	}
	f, _ := m.inventory(jlm.DenseBlock)
	stored := func(r jlm.Role) bool {
		_, ok := f.Inventory[r]
		return ok
	}
	h := graph.Head{NEmbd: c.NEmbd, NVocab: c.NVocab, Eps: c.RMSEps, LayerNorm: c.LayerNorm,
		NormBias: stored(jlm.RoleOutputNormBias), Output: jlm.RoleOutput, Bias: stored(jlm.RoleOutputBias),
		Softcap: float64(c.FinalSoftcap), Scale: c.LogitScale, Hidden: hidden,
		// A collapse of the streams stands in for a norm the model does not store.
		NoNorm: collapsed && !stored(jlm.RoleOutputNorm)}
	if !stored(h.Output) {
		h.Output = jlm.RoleTokenEmbd
	}
	g := h.Build()
	if err := graph.Verify(g, f); err != nil {
		return nil, UncoveredUnverified, fmt.Errorf("the head's graph does not verify: %w", err)
	}
	return g, Covered, nil
}

// poolGraph is an embedding model's readout as ops: its last residual's
// rows through the output norm where the model stores one (a decoder's),
// the live rows pooled as the container says (graph.OpPool: the first row,
// the last, or their mean), projected by the dense heads where it stores
// them, then scaled to unit length -- an RMS norm of no weight over the row
// and 1/sqrt(width), sentence-transformers' 1e-12 floor on the norm taken
// per element as main's (Embedder.normalize).
func (m *Model) poolGraph() (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	if c.Pooling == jlm.PoolNone {
		return nil, Covered, nil
	}
	all, _ := m.inventory(jlm.DenseBlock)
	inv := map[jlm.Role]graph.Shape{}
	has := func(r jlm.Role) bool {
		s, ok := all.Inventory[r]
		if ok {
			inv[r] = s
		}
		return ok
	}
	d := c.NEmbd
	b := graph.NewBlock(int(jlm.DenseBlock))
	x := b.In("x", d)
	if has(jlm.RoleOutputNorm) {
		if c.LayerNorm {
			var bias []jlm.Role
			if has(jlm.RoleOutputNormBias) {
				bias = append(bias, jlm.RoleOutputNormBias)
			}
			x = b.LayerNorm(x, jlm.RoleOutputNorm, c.RMSEps, bias...)
		} else {
			x = b.Norm(x, jlm.RoleOutputNorm, c.RMSEps)
		}
	}
	p := b.Emit(graph.OpPool, []graph.ValueID{x}, []graph.Result{{Name: "pooled", Width: d}}, nil,
		graph.Attrs{Pooling: uint8(c.Pooling)}, nil, nil)[0]
	w := d
	for _, r := range [...]jlm.Role{jlm.RoleEmbdDense1, jlm.RoleEmbdDense2} {
		if has(r) {
			w = inv[r].Rows
			p = b.MatVec(p, r, w)
		}
	}
	n := b.Emit(graph.OpHeadNorm, []graph.ValueID{p}, []graph.Result{{Name: "rms", Width: w}}, nil,
		graph.Attrs{Eps: 1e-24 / float64(w), HeadDim: w}, nil, nil)[0]
	b.Out(mapRow(b, "unit", graph.Attrs{Fn: graph.MapMul, Const: 1 / math.Sqrt(float64(w))}, nil, w, n))
	return verified(b, graph.Facts{Inventory: inv})
}

// pickHeadGraph is the head projecting the rows of nn.PickMax ids alone,
// nil for a head with a bias or one the front end states no graph for.
func (m *Model) pickHeadGraph() (*graph.Graph, error) {
	g, why, err := m.headGraph(false)
	if g == nil || why != Covered || err != nil {
		return nil, err
	}
	c := m.Cfg
	f, _ := m.inventory(jlm.DenseBlock)
	stored := func(r jlm.Role) bool {
		_, ok := f.Inventory[r]
		return ok
	}
	if stored(jlm.RoleOutputBias) {
		return nil, nil
	}
	h := graph.Head{NEmbd: c.NEmbd, NVocab: c.NVocab, Eps: c.RMSEps, LayerNorm: c.LayerNorm,
		NormBias: stored(jlm.RoleOutputNormBias), Output: jlm.RoleOutput,
		Softcap: float64(c.FinalSoftcap), Scale: c.LogitScale, Pick: nn.PickMax}
	if !stored(h.Output) {
		h.Output = jlm.RoleTokenEmbd
	}
	pg := h.Build()
	if err := graph.Verify(pg, f); err != nil {
		return nil, fmt.Errorf("the head's pick graph does not verify: %w", err)
	}
	return pg, nil
}

// perLayerInputs is a row's every layer's per-layer input from its token and
// its scaled row x: the token's row of the per-layer table times sqrt of a
// layer's width, plus x's projection over sqrt of the row's width, normed
// over each layer's slice, the sum over sqrt 2.
func (m *Model) perLayerInputs(b *graph.Builder, f graph.Facts, x graph.ValueID) graph.ValueID {
	c := m.Cfg
	p, w := c.PLEDim, c.pleWidth()
	scale := func(name string, v graph.ValueID, k float64) graph.ValueID {
		return mapRow(b, name, graph.Attrs{Fn: graph.MapMul, Const: k}, nil, w, v)
	}
	t := b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "ple.tok", Width: w}},
		[]graph.Binding{{Role: jlm.RolePLETokEmbd, K: w, Rows: f.Inventory[jlm.RolePLETokEmbd].Rows}}, graph.Attrs{}, nil, nil)[0]
	t = scale("ple.tok.scaled", t, math.Sqrt(float64(p)))
	pr := scale("ple.proj.scaled", b.MatVec(x, jlm.RolePLEModelProj, w), 1/math.Sqrt(float64(c.NEmbd)))
	pr = b.HeadNorm(pr, jlm.RolePLEProjNorm, p, c.RMSEps)
	return scale("ple.in", b.Add(pr, t), 1/math.Sqrt2)
}

// embedGraph is the step's input rows: each row's token's row of the
// embedding, times the stored scale, plus its position's row where the
// container stores a position table; then, for a model of per-layer
// embeddings, each row's per-layer inputs, the blocks' second input.
func (m *Model) embedGraph() (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	f, _ := m.inventory(jlm.DenseBlock)
	_, types := f.Inventory[jlm.RoleTokenTypes]
	switch {
	case c.Streams() > 1:
		return nil, UncoveredStreams, nil
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
	if c.PLEDim != 0 {
		b.Out(m.perLayerInputs(b, f, x))
	}
	if err := graph.Verify(b.G, f); err != nil {
		return nil, UncoveredUnverified, fmt.Errorf("the input rows' graph does not verify: %w", err)
	}
	return b.G, Covered, nil
}
