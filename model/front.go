package model

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/nn"
)

// The model as the composer reads it (composer.Model): its container, and the
// graph of generic ops of each block, the head and the input rows, built from
// what the container states and nothing else: no graph here asks what family
// a model is. A block whose features the op language does not state yet has
// no graph, and Uncovered says why by name; the devices run only what the
// graphs state, so Open refuses a model with a part the front end states no
// graph for (Unstated).

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
	UncoveredStreams
	UncoveredAct
	UncoveredPooling
	UncoveredUnbound
	UncoveredUnverified
	numUncovered
)

// routerSumFloor guards a router's renormalising divide: the selected weights'
// sum is floored before it divides them. The oracle guards its own divide
// with the same floor (TestRouterFloorMatchesTheOracle).
const routerSumFloor = 1e-20

func (u Uncovered) String() string {
	switch u {
	case Covered:
		return "covered"
	case UncoveredKind:
		return "a layer whose kind is not an attention block or a recurrent mixer the front end states"
	case UncoveredRouterScore:
		return "a router whose score the container does not state"
	case UncoveredExperts:
		return "routed experts stored one matrix each (a container converted before its experts were stacked into banks)"
	case UncoveredStreams:
		return "several residual streams"
	case UncoveredAct:
		return "an activation the container does not state, or not over the block's operands"
	case UncoveredPooling:
		return "a readout over the rows in place of a head (an embedding model's pool, a marker head's scorer)"
	case UncoveredUnbound:
		return "stored tensors its graph does not bind"
	case UncoveredUnverified:
		return "a graph that does not verify"
	}
	return fmt.Sprintf("uncovered(%d)", uint8(u))
}

// frontState is the model's graphs, built on first use and again for a
// model stated otherwise since: its config, its rotary or the tensors the
// container stores (restate), which only a test's violation changes after
// Open.
type frontState struct {
	Mu  sync.Mutex
	Set *frontSet
	// restate is a test's statement of the stored tensors in place of the
	// container's, nil otherwise; restated counts its changes.
	Restate  func(block int32, inv map[jlm.Role]graph.Shape)
	Restated int
}

// frontSet is the graphs one statement of the model builds, and the
// statement.
type frontSet struct {
	Blocks []*graph.Graph
	why    []Uncovered
	errs   []error
	// mtp, mtpWhy and mtpErrs are the prediction blocks' bodies
	// (frontmtp.go), prediction block i at block NLayer+i.
	mtp     []*graph.Graph
	mtpWhy  []Uncovered
	mtpErrs []error
	// mtpEntry and mtpEntryWhy are each prediction block's entry
	// (mtpEntryGraph), mtpCollapse its own head mix over the streams where
	// it stores one (streamCollapseGraphAt).
	MtpEntry    []*graph.Graph
	mtpEntryWhy []Uncovered
	MtpCollapse []*graph.Graph
	Head, Embed *graph.Graph
	// cpuEmbed is embed without the streams past the first, which the CPU
	// expands through expand or a family's own expansion;
	// given is an embedding span's rows' input (givenGraph); outNorm the
	// output norm alone (outNormGraph). cpuInErr is why any of them did not
	// verify.
	CpuEmbed, Given, OutNorm *graph.Graph
	CpuInErr                 error
	// headHidden is head writing the norm's rows too (nn.Head.HiddenGraph).
	headHidden *graph.Graph
	// cpuHead and cpuHidden are head and headHidden over the row a model of
	// several streams collapses them to before its head, as the host's call
	// sites hand it to the CPU and to a card: the same graphs on a model of
	// one.
	CpuHead, CpuHidden *graph.Graph
	// expand and collapse are the input rows' streams past stream 0 and the
	// row the head reads from every stream (frontstreams.go), nil for a
	// model of one or of streams it does not predict; streamsErr is why a
	// graph of them did not verify.
	Expand, Collapse *graph.Graph
	StreamsErr       error
	// cpuSample and cpuWalk are sample and a top-k draw's walk over
	// cpuHead's logits; cpuRow and cpuCut the draw with no top-k, with no cut
	// and with one; cpuWalkProb and cpuCutProb the distributions the walk and
	// the cut draw from; cpuCutK and cpuCutKProb a top-k draw past the
	// walk's bound and its distribution; the four Typ graphs are the cut's
	// with typical-p (graph.SampleCut's Typical), apart so a draw without it
	// pays for none of it.
	CpuSample, CpuWalk, CpuRow, CpuCut *graph.Graph
	// CpuWalkInfill is a top-k draw's walk through llama.cpp's infill
	// sampler (graph.SampleWalk's Infill) at nn.InfillMaxK candidates, and
	// Infill the vocabulary's tables it reads.
	CpuWalkInfill              *graph.Graph
	Infill                     nn.InfillTables
	CpuWalkProb, CpuCutProb    *graph.Graph
	CpuCutK, CpuCutKProb       *graph.Graph
	CpuCutTyp, CpuCutTypProb   *graph.Graph
	CpuCutKTyp, CpuCutKTypProb *graph.Graph
	// CpuMiro and CpuMiroV1 are Mirostat's draws, version 2 and 1.
	CpuMiro, CpuMiroV1 *graph.Graph
	// CpuAdaptive is adaptive-p's transform of a draw's distribution
	// (graph.SampleAdaptive), its candidates a top-k draw's at most.
	CpuAdaptive *graph.Graph
	// CpuStage is one stage of a sampler chain in an order of its own
	// (graph.SampleStage), its candidates a top-k draw's at most.
	CpuStage *graph.Graph
	// cpuLogprobs is the logits' log-softmax and their MaxTopLogprobs most
	// likely tokens, cpuMask a constrained draw's masked logits.
	CpuLogprobs, CpuMask *graph.Graph
	// CpuLogprobsRow is the log-softmax with the whole row in order, the
	// likeliest first: the alternatives past MaxTopLogprobs.
	CpuLogprobsRow *graph.Graph
	// CpuDecide is a decision readout's distribution and answers
	// (graph.DecisionScores): over a label readout's picked logits, or a
	// marker head's scores, for a container that states either.
	CpuDecide *graph.Graph
	// tables are the step's rotary tables, the model's rotary's then its
	// sliding layers', as composer.Front states them (nil for one that turns
	// nothing).
	Tables []*graph.Graph
	// pick is the label pick over the logits, pickHead the head projecting
	// the picked rows alone.
	Pick, PickHead *graph.Graph
	// pool is an embedding model's readout (poolGraph), nil where the model
	// pools nothing or the front end states no graph (poolWhy).
	Pool     *graph.Graph
	poolWhy  Uncovered
	poolErr  error
	HeadWhy  Uncovered
	EmbedWhy Uncovered
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
type Composing struct {
	*jlm.File
	m *Model
}

// Composing is the model as the composer reads it: its container and the
// graphs built from it.
func (m *Model) Composing() composer.Model { return Composing{m.Container, m} }

func (c Composing) BlockGraph(li int) *graph.Graph {
	if t, b := c.m.Tower, c.m.TowerBase; t != nil && li >= b && li < b+len(t.Blocks) {
		return t.Blocks[li-b]
	}
	f := c.m.FrontGraphs()
	if i := li - len(f.Blocks); i >= 0 && i < len(f.mtp) {
		return f.mtp[i]
	}
	if li < 0 || li >= len(f.Blocks) {
		return nil
	}
	return f.Blocks[li]
}

func (c Composing) Blocks() int { return c.m.Cfg.NLayer }

func (c Composing) SeamGraph(int) *graph.Graph { return nil }

func (c Composing) HeadGraph() *graph.Graph { return c.m.FrontGraphs().Head }

func (c Composing) EmbedGraph() *graph.Graph { return c.m.FrontGraphs().Embed }

// Rotaries is the model's rotary, then its sliding layers' where it trains
// two: the tables the blocks' rotaries turn by.
func (c Composing) Rotaries() []graph.Rotary {
	out := []graph.Rotary{c.m.rotaryOf(c.m.Rope)}
	if c.m.RopeSWA != nil {
		out = append(out, c.m.rotaryOf(*c.m.RopeSWA))
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

func (c Composing) Tower() bool { return false }

func (c Composing) Segment() composer.Model { return nil }

// Uncovered is why block li has no graph, Covered for a block that has one;
// li of -1 is the head's, -2 the input rows'.
func (m *Model) Uncovered(li int) Uncovered {
	u, _ := m.UncoveredAt(li)
	return u
}

// UncoveredErr is what a graph built wrong got wrong (UncoveredUnbound,
// UncoveredUnverified), nil for any other reason.
func (m *Model) UncoveredErr(li int) error {
	_, err := m.UncoveredAt(li)
	return err
}

// Unstated is why the front end states no graph for a part the model has --
// a block, the head where it projects to its vocabulary, the input rows, the
// readout -- nil where it states each: the devices run only what the graphs
// state, so such a model runs nowhere and Open refuses it, naming the part.
func (m *Model) Unstated() error {
	f := m.FrontGraphs()
	refuse := func(what string, why Uncovered, err error) error {
		if err != nil {
			return fmt.Errorf("model: the front end states no graph for %s: %s: %w", what, why, err)
		}
		return fmt.Errorf("model: the front end states no graph for %s: %s", what, why)
	}
	for li := range m.Cfg.NLayer + m.Cfg.NMTP {
		if (Composing{m.Container, m}).BlockGraph(li) == nil {
			why, err := m.UncoveredAt(li)
			return refuse(fmt.Sprintf("block %d", li), why, err)
		}
	}
	switch {
	case f.CpuHead == nil && f.HeadWhy != UncoveredPooling:
		return refuse("the head", f.HeadWhy, f.headErr)
	case f.CpuEmbed == nil:
		return refuse("the input rows", f.EmbedWhy, f.CpuInErr)
	case f.Pool == nil && f.poolWhy != Covered:
		return refuse("the readout", f.poolWhy, f.poolErr)
	}
	return nil
}

func (m *Model) UncoveredAt(li int) (Uncovered, error) {
	f := m.FrontGraphs()
	switch {
	case li == -1:
		return f.HeadWhy, f.headErr
	case li == -2:
		return f.EmbedWhy, f.embedErr
	case li == -3:
		return f.poolWhy, f.poolErr
	case li >= 0 && li < len(f.why):
		return f.why[li], f.errs[li]
	case li >= len(f.why) && li < len(f.why)+len(f.mtp):
		return f.mtpWhy[li-len(f.why)], f.mtpErrs[li-len(f.why)]
	}
	return UncoveredUnverified, fmt.Errorf("no block %d", li)
}

func (m *Model) FrontGraphs() *frontSet {
	f := &m.Front
	f.Mu.Lock()
	defer f.Mu.Unlock()
	k := frontKey{cfg: m.Cfg, head: headKeyOf(m.Cfg), rope: ropeKeyOf(&m.Rope), ropeSWA: ropeKeyOf(m.RopeSWA), restated: f.Restated}
	if f.Set != nil && f.Set.key == k {
		return f.Set
	}
	n := m.Cfg.NLayer
	s := &frontSet{key: k, Blocks: make([]*graph.Graph, n), why: make([]Uncovered, n), errs: make([]error, n)}
	for li := range n {
		s.Blocks[li], s.why[li], s.errs[li] = m.blockGraph(li)
		if s.errs[li] == nil {
			s.errs[li] = m.shortVector(li, s.Blocks[li])
		}
	}
	if np := m.Cfg.NMTP; np > 0 {
		s.mtp, s.mtpWhy, s.mtpErrs = make([]*graph.Graph, np), make([]Uncovered, np), make([]error, np)
		s.MtpEntry, s.mtpEntryWhy, s.MtpCollapse = make([]*graph.Graph, np), make([]Uncovered, np), make([]*graph.Graph, np)
		for i := range np {
			s.mtp[i], s.mtpWhy[i], s.mtpErrs[i] = m.blockGraph(n + i)
			s.MtpEntry[i], s.mtpEntryWhy[i], _ = m.mtpEntryGraph(n + i)
			if m.Stores(n+i, jlm.RoleHCHeadDown) {
				s.MtpCollapse[i], _ = m.streamCollapseGraphAt(n + i)
			}
		}
	}
	s.Head, s.HeadWhy, s.headErr = m.headGraph(false)
	if s.Head != nil {
		s.headHidden, _, _ = m.headGraph(true)
	}
	s.CpuHead, s.CpuHidden = s.Head, s.headHidden
	if m.Cfg.Streams() > 1 {
		s.CpuHead, _, _ = m.headGraphOf(false, true)
		if s.CpuHead != nil {
			s.CpuHidden, _, _ = m.headGraphOf(true, true)
		}
	}
	var eerr, cerr error
	s.Expand, eerr = m.streamExpandGraph()
	s.Collapse, cerr = m.streamCollapseGraph()
	s.StreamsErr = errors.Join(eerr, cerr)
	s.Embed, s.EmbedWhy, s.embedErr = m.embedGraph(true)
	var ce, ge, ne error
	s.CpuEmbed, _, ce = m.embedGraph(false)
	s.Given, ge = m.givenGraph()
	s.OutNorm, ne = m.outNormGraph()
	s.CpuInErr = errors.Join(ce, ge, ne)
	s.Pool, s.poolWhy, s.poolErr = m.poolGraph()
	for i, r := range (Composing{m.Container, m}).Rotaries() {
		var t *graph.Graph
		if r.NRot > 0 {
			t = composer.TableGraph(r, i)
		}
		s.Tables = append(s.Tables, t)
	}
	vocab := func(h *graph.Graph) bool {
		return h != nil && len(h.Out) == 1 && h.Values[h.Out[0]].Width == m.Cfg.NVocab
	}
	if vocab(s.Head) {
		pk := graph.Pick{NVocab: m.Cfg.NVocab, N: nn.PickMax}
		s.Pick = pk.Build()
		s.PickHead, _ = m.pickHeadGraph()
	}
	if vocab(s.Head) || vocab(s.CpuHead) {
		sm := graph.Sample{NVocab: m.Cfg.NVocab}
		s.CpuSample = sm.Build()
		tk := graph.SampleTopK{NVocab: m.Cfg.NVocab, K: nn.SampleMaxK}
		wk := graph.SampleWalk{SampleTopK: tk}
		s.CpuWalk = wk.Build()
		if m.Vocab != nil {
			s.Infill = m.infillTables()
			wi := graph.SampleWalk{SampleTopK: graph.SampleTopK{NVocab: m.Cfg.NVocab, K: nn.InfillMaxK}, Infill: true,
				Depth: s.Infill.Depth, EOG: len(s.Infill.EOG)}
			s.CpuWalkInfill = wi.Build()
		}
		rw := graph.SampleRow{NVocab: m.Cfg.NVocab}
		s.CpuRow = rw.Build()
		ct := graph.SampleCut{NVocab: m.Cfg.NVocab}
		s.CpuCut = ct.Build()
		wp := graph.SampleWalk{SampleTopK: tk, Prob: true}
		s.CpuWalkProb = wp.Build()
		cp := graph.SampleCut{NVocab: m.Cfg.NVocab, Prob: true}
		s.CpuCutProb = cp.Build()
		ck := graph.SampleCut{NVocab: m.Cfg.NVocab, TopK: true}
		s.CpuCutK = ck.Build()
		ckp := graph.SampleCut{NVocab: m.Cfg.NVocab, TopK: true, Prob: true}
		s.CpuCutKProb = ckp.Build()
		typ := []**graph.Graph{&s.CpuCutTyp, &s.CpuCutTypProb, &s.CpuCutKTyp, &s.CpuCutKTypProb}
		for i, c := range []graph.SampleCut{{}, {Prob: true}, {TopK: true}, {TopK: true, Prob: true}} {
			c.NVocab, c.Typical = m.Cfg.NVocab, true
			*typ[i] = c.Build()
		}
		m2, m1 := graph.SampleMirostat{NVocab: m.Cfg.NVocab}, graph.SampleMirostat{NVocab: m.Cfg.NVocab, V1: true}
		s.CpuMiro, s.CpuMiroV1 = m2.Build(), m1.Build()
		ad := graph.SampleAdaptive{NVocab: m.Cfg.NVocab, K: nn.SampleMaxK}
		s.CpuAdaptive = ad.Build()
		st := graph.SampleStage{NVocab: m.Cfg.NVocab, K: nn.SampleMaxK}
		s.CpuStage = st.Build()
		lp := graph.Logprobs{NVocab: m.Cfg.NVocab, K: MaxTopLogprobs}
		s.CpuLogprobs = lp.Build()
		la := graph.Logprobs{NVocab: m.Cfg.NVocab, K: m.Cfg.NVocab}
		s.CpuLogprobsRow = la.Build()
		mk := graph.MaskRow{NVocab: m.Cfg.NVocab}
		s.CpuMask = mk.Build()
	}
	if cfg := m.Container.Config(); cfg.DecisionLabels != nil || cfg.MarkerPrompt != nil || cfg.JointHead != nil {
		ds := graph.DecisionScores{N: nn.DecisionOptions, G: nn.DecisionGroup}
		s.CpuDecide = ds.Build()
	}
	f.Set = s
	return s
}

// inventory is what the container stores for block (jlm.DenseBlock for the
// model-wide tensors), by role, at the shape a binding states: a vector's
// rows 0. A tensor stored per expert counts as a bank of its experts.
func (m *Model) Inventory(block int32) (graph.Facts, bool) {
	inv := map[jlm.Role]graph.Shape{}
	perExpert := false
	for i := range m.Container.Entries() {
		e := &m.Container.Entries()[i]
		if e.Block != block || e.Role.Vision() || nextnRole(e.Role) {
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
	// A row table is the model's, read by the block whose statement binds
	// it.
	for _, r := range m.Cfg.statedBinds(int(block)) {
		if e, ok := m.Container.Find(r, jlm.DenseBlock, -1); ok && jlm.RowTable(e) {
			inv[e.Role] = graph.Shape{K: e.K(), Rows: e.Rows()}
		}
	}
	if r := m.Front.Restate; r != nil {
		r(block, inv)
	}
	return graph.Facts{Inventory: inv}, perExpert
}

// uncovered is why block li's features are not all ones this front end
// writes, Covered for none. It reads numbers the container states, never what the
// model is called.
func (m *Model) uncovered(li int) Uncovered {
	c := m.Cfg.at(li)
	switch k := c.LayerKind(li); {
	case c.mixerAt(li) != mixNone && !k.Attends():
		return m.uncoveredRest(li)
	case k == jlm.LayerFullAttn || c.mixerAt(li) != mixNone:
	default:
		return UncoveredKind
	}
	switch {
	case c.Streams() > 1 && !c.StatesStreams():
		return UncoveredStreams
	}
	return m.uncoveredRest(li)
}

// uncoveredRest is uncovered's reasons that hold for every mixer: the
// residual, the sequence's direction, the input rows and the MLP.
func (m *Model) uncoveredRest(li int) Uncovered {
	c := m.Cfg.at(li)
	switch {
	case c.Streams() > 1 && !c.StatesStreams():
		return UncoveredStreams
	case m.Container.Config().ActAt(li) == nil:
		return UncoveredAct
	case !c.MoEAt(li):
	case len(m.Container.Config().RouterScore.Terms) == 0:
		return UncoveredRouterScore
	}
	return Covered
}

// stores reports whether the container stores role for block li.
func (m *Model) Stores(li int, role jlm.Role) bool {
	_, ok := m.Container.Find(role, int32(li), -1)
	return ok
}

// ropeAt is the rotary layer li turns by: the sliding layers' where the
// model trains two and li slides, the model's own otherwise.
func (m *Model) ropeAt(li int) nn.Rope {
	if m.RopeSWA != nil && m.Cfg.ropeLocal(li) {
		return *m.RopeSWA
	}
	return m.Rope
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
	// The runs say how the sections lie over the pairs: one after another
	// (Qwen2-VL's) or interleaved (Qwen3-VL's).
	sec, il := m.Cfg.RopeSections, m.Cfg.RopeInterleaved
	switch {
	case slices.Equal(r.Runs, nn.IMRopeRuns(sec[:], r.NRot/2)):
		il = true
	case slices.Equal(r.Runs, nn.MRopeRuns(sec[:])):
		il = false
	}
	return sec, il
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
// shortVector is block li's graph g binding a vector the file does not store
// for it, or past the end of one it does: a short one is read past its end,
// on every device.
func (m *Model) shortVector(li int, g *graph.Graph) error {
	if g == nil || li >= len(m.Layers) {
		return nil
	}
	for _, o := range g.Ops {
		for _, b := range o.Weight {
			if !b.Role.Expanded() {
				continue
			}
			v := nn.VectorOf(m.Layers[li].Vecs, b.Role)
			if v == nil {
				return fmt.Errorf("model: block %d reads its %v, which the file does not store", li, b.Role)
			}
			if len(v) < b.K*max(b.Rows, 1) {
				return fmt.Errorf("model: block %d reads %d floats of its %v, which holds %d", li,
					b.K*max(b.Rows, 1), b.Role, len(v))
			}
		}
	}
	return nil
}

func (m *Model) blockGraph(li int) (*graph.Graph, Uncovered, error) {
	if why := m.uncovered(li); why != Covered {
		return nil, why, nil
	}
	c := m.Cfg.at(li)
	f, perExpert := m.Inventory(int32(li))
	if perExpert {
		return nil, UncoveredExperts, nil
	}
	// A prediction block's own head mix is its collapse graph's
	// (streamCollapseGraphAt), not its body's.
	if li >= c.NLayer {
		for _, r := range []jlm.Role{jlm.RoleHCHeadNorm, jlm.RoleHCHeadDown, jlm.RoleHCHeadUp} {
			delete(f.Inventory, r)
		}
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
	// A marker head's first block adds its question type's row to every row
	// (jlm.Flag2HeadBlocks): the row of its table the step's request names.
	if tt, ok := f.Inventory[jlm.RoleTokenTypes]; ok {
		qt := b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "qtype", Width: d}},
			[]graph.Binding{{Role: jlm.RoleTokenTypes, K: tt.K, Rows: tt.Rows}}, graph.Attrs{Request: true}, nil, nil)[0]
		x = b.Add(x, qt)
	}
	// A block over several residual streams reads each as an input of its
	// own and runs on stream 0's prediction (frontstreams.go).
	var st *streams
	if c.StatesStreams() {
		st = &streams{in: []graph.ValueID{x}}
		for k := 1; k < c.Streams(); k++ {
			st.in = append(st.in, b.In(fmt.Sprintf("x.%d", k), d))
		}
		if c.stated.Blocks[li].Bank != 0 {
			st.bank = streamCopy(b, x)
		}
	}
	// A model of per-layer embeddings gives each block the row's every
	// layer's input (the input rows' second output): the block reads its own.
	// A tap's block reads the row's tap rows the same way.
	var pleIn graph.ValueID
	if c.PLEDim != 0 || li < c.DeepTaps {
		pleIn = b.In("ple", c.SideWidth())
	}
	// Streams with a stated addition before the attention take it first.
	if st != nil && c.stated.Blocks[li].Pre.Read != nil {
		pre := &c.stated.Blocks[li].Pre
		y, ex, err := m.streamRead(b, f, pre, st.in, x, x, pleIn, m.blockOps(b, li))
		if err == nil {
			st.in, err = m.streamWrites(b, f, pre, st.in, y, ex)
		}
		if err != nil {
			return nil, UncoveredStreams, fmt.Errorf("block %d's stated addition: %w", li, err)
		}
		x = st.in[0]
	}
	attnIn := x
	switch {
	case st != nil:
		var err error
		if attnIn, st.ex, err = m.streamRead(b, f, &c.stated.Blocks[li].Attn, st.in, x, x, pleIn, m.blockOps(b, li)); err != nil {
			return nil, UncoveredStreams, fmt.Errorf("block %d's stated attention read: %w", li, err)
		}
	}
	n := preNorm(attnIn, jlm.RoleAttnNorm, jlm.RoleAttnNormBias)
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
		mo := matvec(mix, jlm.RoleSSMOut, d)
		side = append(side, mo)
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
	if has(jlm.RoleAttnKVA) || c.KVShared(li) && c.MLA() {
		att, err := m.latentAttention(b, f, li, n, qIn, q, has, stored, matvec)
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
	vFromK := m.Layers[li].VFromK
	// A block storing neither v nor a reason to take k's projection for it
	// reads its key as attended, after the key's norm and rotary, as its
	// value.
	keyValue := !shared && !vFromK && !has(jlm.RoleAttnV)
	var k, v graph.ValueID
	if !shared {
		k = matvec(n, jlm.RoleAttnK, kvDim, stored(jlm.RoleAttnKBias)...)
		v = k
		if !vFromK && !keyValue {
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
	if act := c.QKVAct; act != nil {
		var err error
		through := func(x graph.ValueID, name string) graph.ValueID {
			if err != nil {
				return x
			}
			var y graph.ValueID
			y, err = b.Stated(name, act, []graph.ValueID{x}, b.G.Values[x].Width, f)
			return y
		}
		q = through(q, "q.act")
		switch {
		case shared:
		case vFromK:
			k = through(k, "k.act")
			v = k
		default:
			k, v = through(k, "k.act"), through(v, "v.act")
		}
		if err != nil {
			return nil, UncoveredUnverified, fmt.Errorf("block %d's stated q/k/v function: %w", li, err)
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
	postK := postQ && c.RopeXD && c.RopeAt(li) && !shared && !m.xdKNormFirst
	// A key norm stored beside no key latent is the key's own.
	if !shared && has(jlm.RoleAttnKVANorm) && !has(jlm.RoleAttnKVA) {
		k = b.Norm(k, jlm.RoleAttnKVANorm, c.RMSEps)
	}
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
	if c.QUnitNorm {
		q = ones(q, "q.unit")
	}
	var turn graph.Attrs
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
		if c.RopeTail {
			a.Offset = hd - r.NRot
		}
		turn = a
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
	if keyValue {
		v = k
	}
	// The attention temperature multiplies q: the scores are linear in it.
	if c.AttnTempScale != 0 && !(c.NoPEGlobal && c.RopeAt(li)) {
		q = mapRow(b, "q.temp", graph.Attrs{Fn: graph.MapMul}, nil, qdim, q, attnTemp(b, c))
	}
	// A selecting block caches its indexer's key as one more kv head of the
	// row, that head's value zero, and its indexer's query picks the
	// positions its heads attend.
	var ix indexed
	rowW := kvDim
	compKind, rate := c.CompAt(li)
	if c.SelectAt(li) && rate == 0 {
		if shared {
			return nil, UncoveredUnverified, fmt.Errorf("block %d selects its keys from a history it shares", li)
		}
		var err error
		if ix, err = m.indexer(b, f, li, n, qIn); err != nil {
			return nil, UncoveredUnverified, err
		}
		kw := int(c.sel.KeyWidth)
		rowW = kvDim + kw
		k = b.Emit(graph.OpConcat, []graph.ValueID{k, ix.key}, []graph.Result{{Name: "k.row", Width: rowW}}, nil, graph.Attrs{}, nil, nil)[0]
		v = b.Emit(graph.OpConcat, []graph.ValueID{v, zerosOf(b, "v.pad", ix.key)}, []graph.Result{{Name: "v.row", Width: rowW}}, nil,
			graph.Attrs{}, nil, nil)[0]
	}
	// A compressing block's row carries its compressor's values and gates
	// past the key; its entries, one a window of rate rows, are the
	// history 1 its attention reads beside them.
	compRow, cw, iw := 0, 0, 0
	if rate > 0 && !shared {
		cv := matvec(n, jlm.RoleCompKV, f.Inventory[jlm.RoleCompKV].Rows)
		cg := matvec(n, jlm.RoleCompGate, f.Inventory[jlm.RoleCompGate].Rows)
		parts := []graph.ValueID{k, cv, cg}
		cw = b.G.Values[cv].Width
		if c.SelectAt(li) {
			iv := matvec(n, jlm.RoleIdxCompKV, f.Inventory[jlm.RoleIdxCompKV].Rows)
			ig := matvec(n, jlm.RoleIdxCompGate, f.Inventory[jlm.RoleIdxCompGate].Rows)
			iw = b.G.Values[iv].Width
			parts = append(parts, iv, ig)
		}
		compRow = rowW + 2*cw + 2*iw
		k = b.Emit(graph.OpConcat, parts, []graph.Result{{Name: "k.row", Width: compRow}}, nil, graph.Attrs{}, nil, nil)[0]
		v = k
	}
	hist := graph.SharedHistory(c.KVSource(li))
	if !shared {
		hist = graph.StateRef{State: graph.StateKV, Version: b.KVAppend(k, v, 0)}
	}
	var entries graph.StateRef
	entW, entRow := 0, 0
	if compRow > 0 {
		// pool is the entry a compressor of w columns at off makes of each
		// window: over overlapping windows (CSA) the window before's first
		// half beside this one's second, normed and turned at the window's
		// first position.
		pool := func(off, w int, ape, norm jlm.Role) graph.ValueID {
			out, words := w, []uint64{0, uint64(off), uint64(off + w), 0}
			if compKind == jlm.CompCSA {
				out = w / 2
				words = []uint64{1, uint64(off), uint64(off + w), 0, 0, uint64(off + out), uint64(off + w + out), uint64(out)}
			}
			e := b.Norm(b.HistoryPool(hist, graph.Attrs{Span: rate, Width: compRow, Words: words}, ape, out), norm, c.RMSEps)
			if c.RopeAt(li) {
				t := turn
				t.HeadDim = out
				if c.RopeTail {
					t.Offset = out - t.NRot
				}
				e = b.Emit(graph.OpRope, []graph.ValueID{e, b.RopeTableAt(t, rate)}, []graph.Result{{Name: "ent.rope", Width: out}}, nil, t, nil, nil)[0]
			}
			return e
		}
		ent := pool(rowW, cw, jlm.RoleCompAPE, jlm.RoleCompNorm)
		entW = b.G.Values[ent].Width
		if iw > 0 {
			ie := pool(rowW+2*cw, iw, jlm.RoleIdxCompAPE, jlm.RoleIdxCompNorm)
			ent = b.Emit(graph.OpConcat, []graph.ValueID{ent, ie}, []graph.Result{{Name: "ent.row", Width: entW + b.G.Values[ie].Width}}, nil,
				graph.Attrs{}, nil, nil)[0]
		}
		entRow = b.G.Values[ent].Width
		entries = graph.StateRef{State: graph.StateKV, Hist: 1, Version: b.KVAppendEvery(1, ent, ent, 0, rate)}
	}
	scale := c.AttnScale
	if scale == 0 {
		scale = 1 / math.Sqrt(float64(hd))
	}
	a := graph.Attrs{NHead: c.NHead, NKVHead: nkv, HeadDim: hd, Scale: scale, Softcap: float64(c.AttnSoftcap), Bidir: c.NonCausal,
		RowWindows: c.NonCausal && c.RowWindowed}
	if c.SWA(li) {
		a.Window, a.Chunked = c.SWAWindow, c.SWAChunked
	}
	var att graph.ValueID
	if compRow > 0 {
		a.Width = compRow
		if c.SelectAt(li) {
			ix, err := m.indexer(b, f, li, n, qIn)
			if err != nil {
				return nil, UncoveredUnverified, err
			}
			att = b.AttendBothSelected(q, hist, entries, rate, entRow, m.selected(b, li, ix, entries, entW, entRow), a,
				stored(jlm.RoleAttnSinks)...)
		} else {
			att = b.AttendBoth(q, hist, entries, rate, entRow, a, stored(jlm.RoleAttnSinks)...)
		}
	} else if c.SelectAt(li) {
		a.Width = rowW
		att = b.AttendSelected(q, hist, m.selected(b, li, ix, hist, kvDim, rowW), a, stored(jlm.RoleAttnSinks)...)
	} else {
		att = b.AttendOf(q, hist, a, stored(jlm.RoleAttnSinks)...)
	}
	if c.OutUnrotated && c.RopeAt(li) {
		turn.Inverse = true
		att = b.Emit(graph.OpRope, []graph.ValueID{att}, []graph.Result{{Name: "att.unrope", Width: b.G.Values[att].Width}}, nil,
			turn, nil, nil)[0]
	}
	// A grouped output down-projection takes each group of heads through its
	// own sheet before the output projection mixes them.
	if oa, ok := f.Inventory[jlm.RoleAttnOutA]; ok {
		w, g := b.G.Values[att].Width, oa.Sheets
		if g < 1 || w%g != 0 || oa.K != w/g {
			return nil, UncoveredUnverified, fmt.Errorf("block %d's grouped output: %d sheets of %d over %d", li, g, oa.K, w)
		}
		att = b.Emit(graph.OpExpertMatVec, []graph.ValueID{att}, []graph.Result{{Name: "att.groups", Width: g * oa.Rows}},
			[]graph.Binding{{Role: jlm.RoleAttnOutA, K: oa.K, Rows: oa.Rows, Sheets: g}}, graph.Attrs{}, nil, nil)[0]
	}
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
	c := m.Cfg.at(li)
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
	// Streams the sublayer writes every one of take its output each as
	// stated; x2 is then none of them.
	var x2 graph.ValueID
	switch {
	case st != nil && len(c.stated.Blocks[li].Attn.Writes) != 0:
		var err error
		if st.cur, err = m.streamWrites(b, f, &c.stated.Blocks[li].Attn, st.in, norm(proj, jlm.RolePostAttnNorm), st.ex); err != nil {
			return nil, UncoveredStreams, fmt.Errorf("block %d's stated attention writes: %w", li, err)
		}
	default:
		x2 = postResid(resid(x, norm(proj, jlm.RolePostAttnNorm)), jlm.RoleAttnOutNorm, jlm.RoleAttnOutNormBias)
	}
	if st != nil && c.stated.Blocks[li].Attn.Join != nil {
		var err error
		if x2, err = m.streamJoin(b, f, &c.stated.Blocks[li].Attn, x, norm(proj, jlm.RolePostAttnNorm), resid); err != nil {
			return nil, UncoveredStreams, fmt.Errorf("block %d's stated attention join: %w", li, err)
		}
		x2 = postResid(x2, jlm.RoleAttnOutNorm, jlm.RoleAttnOutNormBias)
	}
	if !c.MoEAt(li) && !has(jlm.RoleFFNUp) && !has(jlm.RoleFFNDown) {
		if st != nil && st.cur != nil {
			b.Out(st.cur...)
			return verified(b, f)
		}
		if st != nil {
			return m.banked(b, f, li, st, x2, has)
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
	case st != nil:
		var err error
		cur := st.in
		if st.cur != nil {
			cur = st.cur
		}
		// The block's per-layer inputs are its graph's last input.
		var pleIn graph.ValueID
		if c.PLEDim != 0 || li < c.DeepTaps {
			pleIn = b.G.In[len(b.G.In)-1]
		}
		if ffnIn, st.ex, err = m.streamRead(b, f, &c.stated.Blocks[li].FFN, cur, x, x2, pleIn, m.blockOps(b, li)); err != nil {
			return nil, UncoveredStreams, fmt.Errorf("block %d's stated FFN read: %w", li, err)
		}
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
		h, err := b.Stated("act", m.Container.Config().ActAt(li), acts, nffn, f)
		if err != nil {
			return nil, UncoveredAct, fmt.Errorf("the block's stated activation: %w", err)
		}
		h = norm(h, jlm.RoleFFNSubNorm)
		ffn = matvec(h, jlm.RoleFFNDown, d, stored(jlm.RoleFFNDownBias)...)
	}
	if st != nil && len(c.stated.Blocks[li].FFN.Writes) != 0 {
		cur := st.in
		if st.cur != nil {
			cur = st.cur
		}
		out, err := m.streamWrites(b, f, &c.stated.Blocks[li].FFN, cur, norm(ffn, jlm.RolePostFFNNorm), st.ex)
		if err != nil {
			return nil, UncoveredStreams, fmt.Errorf("block %d's stated FFN writes: %w", li, err)
		}
		b.Out(out...)
		return verified(b, f)
	}
	if st != nil {
		y, err := m.streamJoin(b, f, &c.stated.Blocks[li].FFN, x2, norm(ffn, jlm.RolePostFFNNorm), resid)
		if err != nil {
			return nil, UncoveredStreams, fmt.Errorf("block %d's stated FFN join: %w", li, err)
		}
		return m.banked(b, f, li, st, y, has)
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
	if li < m.Cfg.DeepTaps {
		// Tap li's row, a picture's (Qwen3-VL's deepstack _deepstack_process),
		// zeros for a token's, after the block.
		E := m.Cfg.NEmbd
		in := b.Emit(graph.OpSlice, []graph.ValueID{b.G.In[len(b.G.In)-1]}, []graph.Result{{Name: "deep.in", Width: E}}, nil,
			graph.Attrs{Offset: li * E, Width: E}, nil, nil)[0]
		y = b.Add(y, in)
	}
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
	c := m.Cfg.at(li)
	p := c.PLEDim
	gate := b.MatVec(g, jlm.RolePLEGate, p)
	in := b.Emit(graph.OpSlice, []graph.ValueID{b.G.In[len(b.G.In)-1]}, []graph.Result{{Name: "ple.in", Width: p}}, nil,
		graph.Attrs{Offset: c.TapsWidth() + li*p, Width: p}, nil, nil)[0]
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
	c := m.Cfg.at(li)
	d, E, K := c.NEmbd, c.NExpert, c.NExpertUsed
	normed := func(v graph.ValueID, r jlm.Role) graph.ValueID {
		if has(r) {
			return b.Norm(v, r, c.RMSEps)
		}
		return v
	}
	lo := len(b.G.Ops)
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
	score, err := b.Stated("router.score", &m.Container.Config().RouterScore, []graph.ValueID{logits}, E, f)
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
	// A block storing a token table routes each row to its token's row of
	// it; the weights are still the router's scores at those experts.
	var sel graph.ValueID
	if s, ok := f.Inventory[jlm.RoleHashExperts]; ok {
		sel = b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "router.token", Width: K, Elem: graph.ElemU32}},
			[]graph.Binding{{Role: jlm.RoleHashExperts, K: s.K, Rows: s.Rows}}, graph.Attrs{}, nil, nil)[0]
	} else {
		sel, _ = topk("router", choose, K, 0, K)
	}
	take := func(name string, v graph.ValueID) graph.ValueID {
		return b.Emit(graph.OpTake, []graph.ValueID{v, sel}, []graph.Result{{Name: name, Width: K}}, nil, graph.Attrs{}, nil, nil)[0]
	}
	var w graph.ValueID
	if rw := &m.Container.Config().RouterWeight; len(rw.Terms) != 0 {
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
		w = mapc("router.wn", graph.MapDiv, 0, K, nil, w, mapc("router.sum", graph.MapAdd, routerSumFloor, 1, nil, reduce("router.wsum", w, 0, 1)))
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
	h, err := b.Stated("experts.act", m.Container.Config().ActAt(li), acts, K*F, f)
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
	b.G.Routed = graph.Region{Lo: lo, Hi: len(b.G.Ops)}
	out = normed(out, jlm.RolePostFFNNorm2)
	if has(jlm.RoleFFNRoutedUp) {
		out = normed(out, jlm.RoleFFNRoutedNorm)
		out = b.MatVec(out, jlm.RoleFFNRoutedUp, d)
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
	sh1, err := b.Stated("shared.act", m.Container.Config().ActAt(li), sa, sh.Rows, f)
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
// once a position for every head, an indexer's key after them where the
// block stores one (its heads' queries from qIn, the query's input, and
// their weights selecting the positions the heads attend); each head's query its head's first
// HeadDim-NRot through its sheet of RoleAttnKB, joined with the rest turned
// by the rotary, so a query row is the cached row's width; attention over
// the cached rows, each value the whole row, with the block's scale, window,
// softcap and sinks; each head's first KVLoraRank of the result through its
// sheet of RoleAttnVB, gated where the block stores a gate over the output.

func (m *Model) latentAttention(b *graph.Builder, f graph.Facts, li int, n, qIn, q graph.ValueID, has func(jlm.Role) bool, stored func(jlm.Role) []jlm.Role,
	matvec func(graph.ValueID, jlm.Role, int, ...jlm.Role) graph.ValueID) (graph.ValueID, error) {
	c := m.Cfg.at(li)
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
	// A block reading another's history (Config.KVShared) writes no latent
	// row: its keys and values are the rows its source block cached.
	shared := c.KVShared(li)
	var latent, kpe graph.ValueID
	if !shared {
		kv := matvec(n, jlm.RoleAttnKVA, row)
		latent = b.Norm(slice(kv, "latent", 0, 0, lat, lat), jlm.RoleAttnKVANorm, c.LatentNormEps)
		kpe = slice(kv, "k.pe", 0, lat, rot, rot)
	}
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
		if !shared {
			kpe = b.Emit(graph.OpRope, []graph.ValueID{kpe}, []graph.Result{{Name: "k.pe.rope", Width: rot}}, nil, a, nil, nil)[0]
		}
	}
	abs := sheets(slice(q, "q.nope", hd, 0, nope, nh*nope), "q.abs", jlm.RoleAttnKB, nope, lat)
	qrow := b.Emit(graph.OpConcat, []graph.ValueID{abs, qpe}, []graph.Result{{Name: "q.row", Width: nh * row}}, nil,
		graph.Attrs{NHead: nh}, nil, nil)[0]
	// A selection's key rides the cached row after the rotary key.
	parts, width := []graph.ValueID{latent, kpe}, row
	var ix indexed
	if c.SelectAt(li) {
		var err error
		if ix, err = m.indexer(b, f, li, n, qIn); err != nil {
			return 0, err
		}
		parts, width = append(parts, ix.key), row+int(c.sel.KeyWidth)
	}
	hist := graph.SharedHistory(c.KVSource(li))
	if !shared {
		krow := b.Emit(graph.OpConcat, parts, []graph.Result{{Name: "kv.row", Width: width}}, nil, graph.Attrs{}, nil, nil)[0]
		hist = graph.StateRef{State: graph.StateKV, Version: b.KVAppend(krow, krow, 0)}
	}
	scale := c.AttnScale
	if scale == 0 {
		scale = 1 / math.Sqrt(float64(hd))
	}
	a := graph.Attrs{NHead: nh, NKVHead: 1, HeadDim: row, Scale: scale, Softcap: float64(c.AttnSoftcap)}
	if c.SWA(li) {
		a.Window, a.Chunked = c.SWAWindow, c.SWAChunked
	}
	var att graph.ValueID
	if c.SelectAt(li) {
		a.Width = width
		att = b.AttendSelected(qrow, hist, m.selected(b, li, ix, hist, row, width), a, stored(jlm.RoleAttnSinks)...)
	} else {
		att = b.AttendOf(qrow, hist, a, stored(jlm.RoleAttnSinks)...)
	}
	o := sheets(slice(att, "att.latent", row, 0, lat, nh*lat), "att.v", jlm.RoleAttnVB, lat, hdv)
	if has(jlm.RoleAttnGate) {
		gate := matvec(n, jlm.RoleAttnGate, nh*hdv)
		o = sigmoidMul(b, "att.gated", gate, o)
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
	case c.Pooling != jlm.PoolNone || c.Head != nil:
		return nil, UncoveredPooling, nil
	case c.Streams() > 1 && !collapsed:
		return nil, UncoveredStreams, nil
	}
	f, _ := m.Inventory(jlm.DenseBlock)
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

// markGraph is a marker head's readout as ops (jlm.Flag2HeadBlocks): each
// row's request-named row of the last residual (OpTakeRows: the question's
// option markers, in their order, the rows past them repeating the last),
// through the scorer -- a LayerNorm, a d-wide projection, the erf GELU and a
// projection to one score -- into the row's score.
// jointGraph is a joint head's readout (jlm.JointHead; Clef's
// JointSchemaHead) as ops over the prompt's rows and the request's slots
// (nn.Joint*): the final rows normed, a memory of them, each question's and
// option's span averaged and each option's tokens' output embeddings too,
// the options routed over the memory, a summary of each question's options,
// the questions through the joint blocks, then each option's logit -- a
// prior from its tokens against its question, and a gated joint score from
// the cosine and a residual scorer of its question's field and its own row.
// Row i of its output is option i's logit.
func (m *Model) jointGraph() (*graph.Graph, error) {
	c := m.Cfg
	cfg := m.Container.Config()
	all, _ := m.Inventory(jlm.DenseBlock)
	shape := func(r jlm.Role) (graph.Shape, error) {
		s, ok := all.Inventory[r]
		if !ok {
			return s, fmt.Errorf("model: a joint head that stores no %v", r)
		}
		return s, nil
	}
	mem, err := shape(jlm.RoleDecProjMemory)
	if err != nil {
		return nil, err
	}
	out, err := shape(jlm.RoleOutput)
	if err != nil {
		out, err = shape(jlm.RoleTokenEmbd)
		if err != nil {
			return nil, err
		}
	}
	lexRole := jlm.RoleOutput
	if _, ok := all.Inventory[jlm.RoleOutput]; !ok {
		lexRole = jlm.RoleTokenEmbd
	}
	d, w, H := c.NEmbd, mem.Rows, int(cfg.DecisionHeads)
	R, J := int(cfg.DecisionRouting), int(cfg.DecisionBlocks)
	if H < 1 || w%H != 0 || J < 1 {
		return nil, fmt.Errorf("model: a joint head %d wide in %d heads with %d joint blocks", w, H, J)
	}
	eps := float64(cfg.DecisionEps)
	b := graph.NewBlock(int(jlm.DenseBlock))
	emit := func(k graph.OpKind, in []graph.ValueID, name string, width int, wb []graph.Binding, a graph.Attrs) graph.ValueID {
		return b.Emit(k, in, []graph.Result{{Name: name, Width: width}}, wb, a, nil, nil)[0]
	}
	fn := func(name string, width int, f graph.MapFn, k float64, in ...graph.ValueID) graph.ValueID {
		return emit(graph.OpMap, in, name, width, nil, graph.Attrs{Fn: f, Const: k})
	}
	sum := func(name string, v graph.ValueID) graph.ValueID {
		return emit(graph.OpReduce, []graph.ValueID{v}, name, 1, nil, graph.Attrs{Fn: graph.MapAdd})
	}
	ln := func(v graph.ValueID, role jlm.Role) graph.ValueID { return b.LayerNorm(v, role, eps, role+1) }
	gelu := func(name string, v graph.ValueID, width int) (graph.ValueID, error) {
		var e jlm.ExprBuilder
		e.GELUErf(jlm.Operand(0))
		act := jlm.Expr{Terms: e.T}
		return b.Stated(name, &act, []graph.ValueID{v}, width, all)
	}
	slots := func(k graph.OpKind, v graph.ValueID, name string, width, slot int) graph.ValueID {
		return emit(k, []graph.ValueID{v}, name, width, nil, graph.Attrs{Offset: slot})
	}
	// scale is v times the head's learned scalar i (decision.scales: the
	// prior's, the joint score's and the gate's, as used at inference).
	scale := func(name string, v graph.ValueID, i int) graph.ValueID {
		return emit(graph.OpMap, []graph.ValueID{v}, name, 1, []graph.Binding{{Role: jlm.RoleDecScales, K: 3}},
			graph.Attrs{Fn: graph.MapMul, Width: 1, Offset: i})
	}
	// norm2 is v over its length, the length floored at eps (F.normalize).
	length := func(name string, v graph.ValueID, width int) graph.ValueID {
		return fn(name+".len", 1, graph.MapSqrt, 0, sum(name+".sq", fn(name+".mul", width, graph.MapMul, 0, v, v)))
	}
	normalize := func(name string, v graph.ValueID, width int) graph.ValueID {
		return fn(name, width, graph.MapDiv, 0, v, fn(name+".floor", 1, graph.MapMax, 1e-12, length(name, v, width)))
	}
	attend := func(name string, q, k, v graph.ValueID, heads, width, slot int) graph.ValueID {
		hd := width / heads
		return emit(graph.OpRowsAttend, []graph.ValueID{q, k, v}, name, width, nil,
			graph.Attrs{NHead: heads, NKVHead: heads, HeadDim: hd, Scale: 1 / math.Sqrt(float64(hd)), Offset: slot})
	}
	// mha is torch's MultiheadAttention of block k: q from x, k and v from
	// kv, through the block's projections of kind q (self or cross).
	mha := func(k int, q0 jlm.Role, x, kv graph.ValueID, slot int) graph.ValueID {
		role := func(r jlm.Role) jlm.Role { return jlm.RoleDecBlock(k, r) }
		qr, kr, vr, or := q0, q0+1, q0+2, q0+3
		bias := func(r jlm.Role) jlm.Role { return r + 4 }
		q := b.MatVec(x, role(qr), w, role(bias(qr)))
		kk := b.MatVec(kv, role(kr), w, role(bias(kr)))
		v := b.MatVec(kv, role(vr), w, role(bias(vr)))
		a := attend("attn", q, kk, v, H, w, slot)
		return b.MatVec(a, role(or), w, role(bias(or)))
	}
	ffn := func(k int, x graph.ValueID) (graph.ValueID, error) {
		role := func(r jlm.Role) jlm.Role { return jlm.RoleDecBlock(k, r) }
		up, err := shape(role(jlm.RoleDecFFNUp))
		if err != nil {
			return 0, err
		}
		h := b.MatVec(ln(x, role(jlm.RoleDecFFNNorm)), role(jlm.RoleDecFFNUp), up.Rows, role(jlm.RoleDecFFNUpBias))
		g, err := gelu("ffn.act", h, up.Rows)
		if err != nil {
			return 0, err
		}
		return b.Add(x, b.MatVec(g, role(jlm.RoleDecFFNDown), w, role(jlm.RoleDecFFNDownBias))), nil
	}

	x := b.In("x", d)
	if _, ok := all.Inventory[jlm.RoleOutputNorm]; ok {
		x = b.Norm(x, jlm.RoleOutputNorm, c.RMSEps)
	}
	hn := ln(x, jlm.RoleDecHiddenNorm)
	memory := b.MatVec(hn, jlm.RoleDecProjMemory, w)
	qv := slots(graph.OpSpanPool, hn, "question", d, nn.JointQuestionSpans)
	ctx := slots(graph.OpSpanPool, hn, "option", d, nn.JointOptionSpans)
	tokens := emit(graph.OpEmbed, nil, "tokens", d, []graph.Binding{{Role: lexRole, K: d, Rows: out.Rows}},
		graph.Attrs{Request: true, Offset: nn.JointTokens})
	lex := slots(graph.OpSpanPool, tokens, "lexical", d, nn.JointOptionSpans)
	global := emit(graph.OpPool, []graph.ValueID{hn}, "global", d, nil, graph.Attrs{Pooling: uint8(jlm.PoolLast)})
	qOf := slots(graph.OpTakeRows, qv, "option.question", d, nn.JointOptionQuestion)
	opts := b.Add(b.Add(b.MatVec(ctx, jlm.RoleDecProjOptCtx, w), b.MatVec(lex, jlm.RoleDecProjOptLex, w)),
		b.MatVec(qOf, jlm.RoleDecProjOptQ, w))
	for k := range R {
		mn := ln(memory, jlm.RoleDecBlock(k, jlm.RoleDecCrossKVNorm))
		opts = b.Add(opts, mha(k, jlm.RoleDecCrossQ, ln(opts, jlm.RoleDecBlock(k, jlm.RoleDecCrossNorm)), mn, 0))
		if opts, err = ffn(k, opts); err != nil {
			return nil, err
		}
	}
	base := b.MatVec(qv, jlm.RoleDecProjQuestion, w)
	summary := attend("summary", base, opts, opts, 1, w, nn.JointQuestionOptions)
	types, err := shape(jlm.RoleTokenTypes)
	if err != nil {
		return nil, err
	}
	typed := emit(graph.OpEmbed, nil, "type", w, []graph.Binding{{Role: jlm.RoleTokenTypes, K: w, Rows: types.Rows}},
		graph.Attrs{Request: true, Offset: nn.JointQuestionTypes})
	fields := b.Add(b.Add(b.Add(base, ln(summary, jlm.RoleDecOptSumNorm)), b.MatVec(global, jlm.RoleDecProjGlobal, w)), typed)
	for k := R; k < R+J; k++ {
		fields = b.Add(fields, mha(k, jlm.RoleDecSelfQ, ln(fields, jlm.RoleDecBlock(k, jlm.RoleDecSelfNorm)),
			ln(fields, jlm.RoleDecBlock(k, jlm.RoleDecSelfNorm)), nn.JointFields))
		fields = b.Add(fields, mha(k, jlm.RoleDecCrossQ, ln(fields, jlm.RoleDecBlock(k, jlm.RoleDecCrossNorm)), memory, 0))
		if fields, err = ffn(k, fields); err != nil {
			return nil, err
		}
	}
	fields = ln(fields, jlm.RoleDecFieldNorm)
	field := slots(graph.OpTakeRows, fields, "option.field", w, nn.JointOptionQuestion)
	option := ln(opts, jlm.RoleDecOptNorm)
	// torch's cosine_similarity: x.y over the product of the lengths, the
	// product floored at 1e-8.
	dot := sum("cos.dot", fn("cos.mul", w, graph.MapMul, 0, field, option))
	lens := fn("cos.lens", 1, graph.MapMax, 1e-8, fn("cos.prod", 1, graph.MapMul, 0, length("cos.f", field, w), length("cos.o", option, w)))
	cosine := fn("cos", 1, graph.MapDiv, 0, dot, lens)
	features := emit(graph.OpConcat, []graph.ValueID{field, option, fn("feat.mul", w, graph.MapMul, 0, field, option),
		fn("feat.abs", w, graph.MapAbs, 0, fn("feat.sub", w, graph.MapSub, 0, field, option))}, "features", 4*w, nil, graph.Attrs{})
	hidden, err := gelu("scorer.act", b.MatVec(features, jlm.RoleScorer, w, jlm.RoleScorerBias), w)
	if err != nil {
		return nil, err
	}
	// The scorer's last projection is one row: stored as a matrix of one
	// row, or as the vector it is.
	so, err := shape(jlm.RoleScorerOut)
	if err != nil {
		return nil, err
	}
	var residual graph.ValueID
	if so.Rows < 1 {
		prod := emit(graph.OpMap, []graph.ValueID{hidden}, "scorer.mul", w, []graph.Binding{{Role: jlm.RoleScorerOut, K: w}},
			graph.Attrs{Fn: graph.MapMul})
		residual = emit(graph.OpMap, []graph.ValueID{sum("scorer.dot", prod)}, "scorer.out", 1,
			[]graph.Binding{{Role: jlm.RoleScorerOutBias, K: 1}}, graph.Attrs{Fn: graph.MapAdd, Width: 1})
	} else {
		residual = b.MatVec(hidden, jlm.RoleScorerOut, 1, jlm.RoleScorerOutBias)
	}
	joint := b.Add(scale("joint.scaled", cosine, 1), residual)
	anchor := normalize("anchor", b.Add(qOf, global), d)
	prior := scale("prior", sum("prior.dot", fn("prior.mul", d, graph.MapMul, 0, normalize("lexical.unit", lex, d), anchor)), 0)
	b.Out(b.Add(prior, scale("joint.gated", joint, 2)))
	return b.G, graph.Verify(b.G, all)
}

// pointerGraph is a pointer readout (jlm.Flag2MarkerPointer) as ops: the
// prompt's last row (graph.OpPool's) and the rows its request names, each
// through the output norm and RoleScorerOut's projection, an option's score
// the last row's first half dotted with its row's second half over the root
// of their width.
func (m *Model) pointerGraph() (*graph.Graph, error) {
	c := m.Cfg
	all, _ := m.Inventory(jlm.DenseBlock)
	out, ok := all.Inventory[jlm.RoleScorerOut]
	if !ok || out.Rows < 2 || out.Rows%2 != 0 {
		return nil, fmt.Errorf("model: a pointer readout whose %v is %d wide", jlm.RoleScorerOut, out.Rows)
	}
	if _, ok := all.Inventory[jlm.RoleOutputNorm]; !ok {
		return nil, fmt.Errorf("model: a pointer readout that stores no %v", jlm.RoleOutputNorm)
	}
	var bias []jlm.Role
	if _, ok := all.Inventory[jlm.RoleScorerOutBias]; ok {
		bias = []jlm.Role{jlm.RoleScorerOutBias}
	}
	d, w := c.NEmbd, out.Rows
	b := graph.NewBlock(int(jlm.DenseBlock))
	x := b.In("x", d)
	project := func(v graph.ValueID) graph.ValueID {
		return b.MatVec(b.Norm(v, jlm.RoleOutputNorm, c.RMSEps), jlm.RoleScorerOut, w, bias...)
	}
	last := b.Emit(graph.OpPool, []graph.ValueID{x}, []graph.Result{{Name: "last", Width: d}}, nil,
		graph.Attrs{Pooling: uint8(jlm.PoolLast)}, nil, nil)[0]
	marks := b.Emit(graph.OpTakeRows, []graph.ValueID{x}, []graph.Result{{Name: "marks", Width: d}}, nil, graph.Attrs{}, nil, nil)[0]
	q := b.Emit(graph.OpSlice, []graph.ValueID{project(last)}, []graph.Result{{Name: "query", Width: w / 2}}, nil,
		graph.Attrs{Width: w / 2}, nil, nil)[0]
	k := b.Emit(graph.OpSlice, []graph.ValueID{project(marks)}, []graph.Result{{Name: "key", Width: w / 2}}, nil,
		graph.Attrs{Offset: w / 2, Width: w / 2}, nil, nil)[0]
	qk := b.Emit(graph.OpMap, []graph.ValueID{q, k}, []graph.Result{{Name: "qk", Width: w / 2}}, nil, graph.Attrs{Fn: graph.MapMul}, nil, nil)[0]
	dot := b.Emit(graph.OpReduce, []graph.ValueID{qk}, []graph.Result{{Name: "dot", Width: 1}}, nil, graph.Attrs{Fn: graph.MapAdd}, nil, nil)[0]
	b.Out(b.Emit(graph.OpMap, []graph.ValueID{dot}, []graph.Result{{Name: "score", Width: 1}}, nil,
		graph.Attrs{Fn: graph.MapMul, Const: 1 / math.Sqrt(float64(w/2))}, nil, nil)[0])
	return b.G, graph.Verify(b.G, all)
}

func (m *Model) markGraph() (*graph.Graph, error) {
	c := m.Cfg
	all, _ := m.Inventory(jlm.DenseBlock)
	opt := func(r jlm.Role) []jlm.Role {
		if _, ok := all.Inventory[r]; ok {
			return []jlm.Role{r}
		}
		return nil
	}
	for _, r := range []jlm.Role{jlm.RoleScorerNorm, jlm.RoleScorer, jlm.RoleScorerOut} {
		if opt(r) == nil {
			return nil, fmt.Errorf("model: a marker head that stores no %v", r)
		}
	}
	d := c.NEmbd
	b := graph.NewBlock(int(jlm.DenseBlock))
	x := b.In("x", d)
	r := b.Emit(graph.OpTakeRows, []graph.ValueID{x}, []graph.Result{{Name: "marks", Width: d}}, nil, graph.Attrs{}, nil, nil)[0]
	n := b.LayerNorm(r, jlm.RoleScorerNorm, c.Head.RMSEps, opt(jlm.RoleScorerNormBias)...)
	h := b.MatVec(n, jlm.RoleScorer, d, opt(jlm.RoleScorerBias)...)
	var e jlm.ExprBuilder
	e.GELUErf(jlm.Operand(0))
	act := jlm.Expr{Terms: e.T}
	g, err := b.Stated("scorer.act", &act, []graph.ValueID{h}, d, all)
	if err != nil {
		return nil, fmt.Errorf("model: the scorer's activation: %w", err)
	}
	b.Out(b.MatVec(g, jlm.RoleScorerOut, 1, opt(jlm.RoleScorerOutBias)...))
	return b.G, graph.Verify(b.G, all)
}

// poolGraph is an embedding model's readout as ops: its last residual's
// rows through the output norm where the model stores one (a decoder's),
// the live rows pooled as the container says (graph.OpPool: the first row,
// the last, or their mean), projected by the dense heads where it stores
// them, then normalised as the input embed.norm asks (norm): at 2 scaled to
// unit length -- an RMS norm of no weight over the row and 1/sqrt(width),
// sentence-transformers' 1e-12 floor on the norm taken per element as
// main's (Embedder.normalize).
func (m *Model) poolGraph() (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	if c.Joint {
		g, err := m.jointGraph()
		if err != nil {
			return nil, UncoveredUnverified, err
		}
		return g, Covered, nil
	}
	if c.MarkPointer {
		g, err := m.pointerGraph()
		if err != nil {
			return nil, UncoveredUnverified, err
		}
		return g, Covered, nil
	}
	if c.Head != nil {
		g, err := m.markGraph()
		if err != nil {
			return nil, UncoveredUnverified, err
		}
		return g, Covered, nil
	}
	if c.Pooling == jlm.PoolNone {
		return nil, Covered, nil
	}
	all, _ := m.Inventory(jlm.DenseBlock)
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
	// unit is v at unit length: an RMSNorm of no weight is v/sqrt(mean(v^2)),
	// and 1/sqrt(w) turns the mean into the sum. The eps is
	// sentence-transformers' 1e-12 floor on the norm, squared and taken per
	// element.
	unit := func(v graph.ValueID, w int, name string) graph.ValueID {
		n := b.Emit(graph.OpHeadNorm, []graph.ValueID{v}, []graph.Result{{Name: name + ".rms", Width: w}}, nil,
			graph.Attrs{Eps: 1e-24 / float64(w), HeadDim: w}, nil, nil)[0]
		return mapRow(b, name, graph.Attrs{Fn: graph.MapMul, Const: 1 / math.Sqrt(float64(w))}, nil, w, n)
	}
	// norm is v normalised as llama.cpp's embd_normalize k asks
	// (common_embd_normalize): unit at 2, v as it is below 0, scaled into
	// int16's range by its largest magnitude at 0, else divided by its k-norm,
	// |v|^k as exp(k log|v|) and the sum's k-th root as exp(log(sum)/k). A
	// divisor of 0 leaves zeros.
	k := b.In("embed.norm", 1)
	norm := func(v graph.ValueID, w int, name string) graph.ValueID {
		m := func(n string, fn graph.MapFn, c float64, width int, in ...graph.ValueID) graph.ValueID {
			return mapRow(b, name+"."+n, graph.Attrs{Fn: fn, Const: c}, nil, width, in...)
		}
		over := func(n string, fn graph.MapFn, x graph.ValueID) graph.ValueID {
			return b.Emit(graph.OpReduce, []graph.ValueID{x}, []graph.Result{{Name: name + "." + n, Width: 1}}, nil,
				graph.Attrs{Fn: fn}, nil, nil)[0]
		}
		where := func(n string, width int, c, x, y graph.ValueID) graph.ValueID {
			return b.Expr(name+"."+n, width, []graph.Term{{Fn: graph.MapWhere, Args: [3]graph.Arg{{Kind: graph.ArgIn, Index: 0},
				{Kind: graph.ArgIn, Index: 1}, {Kind: graph.ArgIn, Index: 2}}}}, nil, c, x, y)
		}
		a := m("abs", graph.MapAbs, 0, w, v)
		pk := m("pow", graph.MapExp, 0, w, m("klog", graph.MapMul, 0, w, m("log", graph.MapLog, 0, w, a), k))
		root := m("root", graph.MapExp, 0, 1, m("lroot", graph.MapDiv, 0, 1, m("lsum", graph.MapLog, 0, 1, over("sum", graph.MapAdd, pk)), k))
		big := m("big", graph.MapMul, 1.0/32760, 1, over("max", graph.MapMax, a))
		div := where("div", 1, k, root, m("div0", graph.MapWhere, 1, 1, m("k0", graph.MapEqual, 0, 1, k), big))
		scale := m("scale", graph.MapWhere, 0, 1, div, m("recip", graph.MapRecip, 0, 1, div))
		return where("out", w, m("k2", graph.MapEqual, 2, 1, k), unit(v, w, name), m("scaled", graph.MapMul, 0, w, v, scale))
	}
	raw, w := p, d
	for _, r := range [...]jlm.Role{jlm.RoleEmbdDense1, jlm.RoleEmbdDense2} {
		if has(r) {
			w = inv[r].Rows
			p = b.MatVec(p, r, w)
		}
	}
	if p == raw {
		b.Out(norm(p, w, "unit"))
	} else {
		b.Out(norm(p, w, "unit"), norm(raw, d, "unit.raw"))
	}
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
	f, _ := m.Inventory(jlm.DenseBlock)
	stored := func(r jlm.Role) bool {
		_, ok := f.Inventory[r]
		return ok
	}
	h := graph.Head{NEmbd: c.NEmbd, NVocab: c.NVocab, Eps: c.RMSEps, LayerNorm: c.LayerNorm,
		NormBias: stored(jlm.RoleOutputNormBias), Bias: stored(jlm.RoleOutputBias), Output: jlm.RoleOutput,
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
	p, w := c.PLEDim, c.PleWidth()
	scale := func(name string, v graph.ValueID, k float64) graph.ValueID {
		return mapRow(b, name, graph.Attrs{Fn: graph.MapMul, Const: k}, nil, w, v)
	}
	t := b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "ple.tok", Width: w}},
		[]graph.Binding{{Role: jlm.RolePLETokEmbd, K: w, Rows: f.Inventory[jlm.RolePLETokEmbd].Rows}}, graph.Attrs{}, nil, nil)[0]
	t = scale("ple.tok.scaled", t, math.Sqrt(float64(p)))
	pr := scale("ple.proj.scaled", b.MatVec(x, jlm.RolePLEModelProj, w), 1/math.Sqrt(float64(c.NEmbd)))
	pr = b.HeadNorm(pr, jlm.RolePLEProjNorm, p, c.RMSEps)
	pl := scale("ple.in", b.Add(pr, t), 1/math.Sqrt2)
	if c.TapsWidth() == 0 {
		return pl
	}
	// A row's tap rows lead its side input, zeros until a picture's are
	// written over them.
	zeros := mapRow(b, "taps.zero", graph.Attrs{Fn: graph.MapMul, Const: 0}, nil, c.TapsWidth(), x)
	return b.Emit(graph.OpConcat, []graph.ValueID{zeros, pl}, []graph.Result{{Name: "side", Width: c.SideWidth()}}, nil,
		graph.Attrs{}, nil, nil)[0]
}

// embedGraph is the step's input rows: each row's token's row of the
// embedding, times the stored scale, plus its position's row where the
// container stores a position table; with streams, for a model of several
// streams the front end expands, every stream past the first from it
// (streamExpand); then, for a model of per-layer embeddings, each row's
// per-layer inputs, the blocks' last input. A bank of checkpoints writes
// the running residual's rows alone: a block writes each checkpoint before
// any block reads it.
func (m *Model) embedGraph(streams bool) (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	f, _ := m.Inventory(jlm.DenseBlock)
	// Token types the input rows add are a table of n_embd-wide rows; a
	// vector of them is a head's (clef's fields), which the rows never read.
	tt, types := f.Inventory[jlm.RoleTokenTypes]
	types = types && tt.K == c.NEmbd && tt.Rows > 0
	if streams && c.Streams() > 1 && !c.StatesStreams() && !m.ExpandsStreams() {
		return nil, UncoveredStreams, nil
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
	// A text is one segment: every row adds the token-type table's row 0, a
	// positional row at a span of one.
	if types {
		t0 := b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "types", Width: c.NEmbd}},
			[]graph.Binding{{Role: jlm.RoleTokenTypes, K: tt.K, Rows: tt.Rows}}, graph.Attrs{Positional: true, Span: 1}, nil, nil)[0]
		x = b.Add(x, t0)
	}
	if _, ok := f.Inventory[jlm.RoleTokenEmbdNorm]; ok {
		if c.LayerNorm {
			var bias []jlm.Role
			if _, ok := f.Inventory[jlm.RoleTokenEmbdNormBias]; ok {
				bias = append(bias, jlm.RoleTokenEmbdNormBias)
			}
			x = b.LayerNorm(x, jlm.RoleTokenEmbdNorm, c.RMSEps, bias...)
		} else {
			x = b.Norm(x, jlm.RoleTokenEmbdNorm, c.RMSEps)
		}
	}
	b.Out(x)
	if streams && m.ExpandsStreams() {
		xs, err := m.streamExpand(b, f, x)
		if err != nil {
			return nil, UncoveredStreams, err
		}
		b.Out(xs...)
	}
	if c.PLEDim != 0 {
		b.Out(m.perLayerInputs(b, f, x))
	}
	if err := graph.Verify(b.G, f); err != nil {
		return nil, UncoveredUnverified, fmt.Errorf("the input rows' graph does not verify: %w", err)
	}
	return b.G, Covered, nil
}

// outNormGraph is the output norm alone over a row: the trunk's hidden
// state a prediction block reads of a speculation's prompt rows. nil for a
// model that stores no output norm, whose hidden state is the row itself.
func (m *Model) outNormGraph() (*graph.Graph, error) {
	c := m.Cfg
	all, _ := m.Inventory(jlm.DenseBlock)
	inv := map[jlm.Role]graph.Shape{}
	has := func(r jlm.Role) bool {
		s, ok := all.Inventory[r]
		if ok {
			inv[r] = s
		}
		return ok
	}
	if !has(jlm.RoleOutputNorm) {
		return nil, nil
	}
	b := graph.NewBlock(int(jlm.DenseBlock))
	x := b.In("x", c.NEmbd)
	if c.LayerNorm {
		var bias []jlm.Role
		if has(jlm.RoleOutputNormBias) {
			bias = append(bias, jlm.RoleOutputNormBias)
		}
		x = b.LayerNorm(x, jlm.RoleOutputNorm, c.RMSEps, bias...)
	} else {
		x = b.Norm(x, jlm.RoleOutputNorm, c.RMSEps)
	}
	b.Out(x)
	g, _, err := verified(b, graph.Facts{Inventory: inv})
	return g, err
}

// givenGraph is an embedding span's rows' input, the rows the caller gives
// in place of a token's: each row plus its position's row where the
// container stores a position table, then, for a model of per-layer
// embeddings, each row's per-layer inputs from the row and the token it
// names (token 0's where it names none). Its outputs are the row where a
// position table moves it, then the per-layer inputs; nil for a model that
// adds nothing to a given row.
func (m *Model) givenGraph() (*graph.Graph, error) {
	c := m.Cfg
	f, _ := m.Inventory(jlm.DenseBlock)
	p, pos := f.Inventory[jlm.RolePosEmbd]
	if !pos && c.PLEDim == 0 {
		return nil, nil
	}
	b := graph.NewBlock(-1)
	x := b.In("x", c.NEmbd)
	if pos {
		pe := b.Emit(graph.OpEmbed, nil, []graph.Result{{Name: "pos", Width: c.NEmbd}},
			[]graph.Binding{{Role: jlm.RolePosEmbd, K: p.K, Rows: p.Rows}}, graph.Attrs{Positional: true}, nil, nil)[0]
		x = b.Add(x, pe)
		b.Out(x)
	}
	if c.PLEDim != 0 {
		b.Out(m.perLayerInputs(b, f, x))
	}
	if err := graph.Verify(b.G, f); err != nil {
		return nil, fmt.Errorf("the given rows' graph does not verify: %w", err)
	}
	return b.G, nil
}

// MaxTopLogprobs is the alternatives the log-softmax's own selection orders
// per token (CpuLogprobs), the OpenAI API's bound on top_logprobs; past it a
// Logprobs reads the whole row in order (CpuLogprobsRow).
const MaxTopLogprobs = 20

// indexed is a selecting block's indexer: its key as the row caches it, its
// query, and its heads' weight where it states one.
type indexed struct {
	key, query, weight graph.ValueID
	weighted           bool
}

// indexer is block li's stated indexer (jlm.Selection) over its normed input
// n and the attention's query latent qIn (n where it has none), its key
// padded with zeros to the width the row caches.
func (m *Model) indexer(b *graph.Builder, f graph.Facts, li int, n, qIn graph.ValueID) (indexed, error) {
	s := m.Cfg.sel
	turn := m.blockOps(b, li)
	var ix indexed
	// A selection with no key program selects among the block's entries,
	// whose key the block pools itself.
	if s.Key != nil {
		key, err := lowerValuesWith(b, nil, s.Key, []graph.ValueID{n}, 1, f, turn)
		if err != nil {
			return ix, fmt.Errorf("block %d's indexer key: %w", li, err)
		}
		ix.key = key[s.Key.Out]
		if w, kw := b.G.Values[ix.key].Width, int(s.KeyWidth); w < kw {
			ix.key = b.Emit(graph.OpConcat, []graph.ValueID{ix.key, zerosOf(b, "idx.k.pad", ix.key, kw-w)},
				[]graph.Result{{Name: "idx.k.row", Width: kw}}, nil, graph.Attrs{}, nil, nil)[0]
		}
	}
	q, err := lowerValuesWith(b, nil, s.Query, []graph.ValueID{n, qIn}, 1, f, turn)
	if err != nil {
		return ix, fmt.Errorf("block %d's indexer query: %w", li, err)
	}
	ix.query = q[s.Query.Out]
	if s.Weight != nil {
		w, err := lowerValuesWith(b, nil, s.Weight, []graph.ValueID{n}, 1, f, turn)
		if err != nil {
			return ix, fmt.Errorf("block %d's indexer weight: %w", li, err)
		}
		ix.weight, ix.weighted = w[s.Weight.Out], true
	}
	return ix, nil
}

// selected is the positions block li's indexer ix selects from hist, whose
// rows carry its key at off of width.
func (m *Model) selected(b *graph.Builder, li int, ix indexed, hist graph.StateRef, off, width int) graph.ValueID {
	c, s := m.Cfg, m.Cfg.sel
	a := graph.Attrs{NHead: int(s.Heads), NKVHead: int(s.Groups), HeadDim: int(s.HeadDim), Offset: off, Width: width,
		Span: int(s.Block), TopK: int(s.Keep), Local: int(s.Local)}
	if s.Rank == jlm.RankPooled {
		a.Fn, a.Eps = graph.MapMax, float64(s.Eps)
		if c.RopeAt(li) {
			r := m.ropeAt(li)
			a.NRot, a.Base, a.Layout, a.Freqs, a.Mscale = r.NRot, r.Base, ropeLayout(r.Neox), len(r.Freqs) > 0, r.Scale
			a.Sections, a.Interleaved = m.ropeAxes(r)
		}
		return b.SelectPooled(ix.query, hist, a, s.PoolNorm)
	}
	if s.Relu {
		a.Fn = graph.MapMax
	}
	if ix.weighted {
		return b.Select(ix.query, hist, a, ix.weight)
	}
	return b.Select(ix.query, hist, a)
}

// blockOps lowers the stated ops only a block has, block li's: a rotary
// (jlm.TOpRope), its attention's at the row's position in the stated
// layout, nothing on a block with none; and the ops over its recurrent tail
// (jlm.TOpNGramRows, jlm.TOpTailConv), each at its place in the tail after
// the delta rule's state, in the order they come.
func (m *Model) blockOps(b *graph.Builder, li int) func(ins []graph.ValueID, op *jlm.TowerOp) (graph.ValueID, error) {
	c := m.Cfg.at(li)
	cursor := 0
	return func(ins []graph.ValueID, op *jlm.TowerOp) (graph.ValueID, error) {
		switch op.Code {
		case jlm.TOpRope:
			v := ins[0]
			if !c.RopeAt(li) {
				return v, nil
			}
			r := m.ropeAt(li)
			layout := ropeLayout(r.Neox)
			switch op.Chain {
			case jlm.RopeNormal:
				layout = graph.RopeNorm
			case jlm.RopeNeox:
				layout = graph.RopeNeox
			}
			a := graph.Attrs{HeadDim: int(op.Span), NRot: r.NRot, Base: r.Base, Layout: layout, Freqs: len(r.Freqs) > 0, Mscale: r.Scale}
			a.Sections, a.Interleaved = m.ropeAxes(r)
			if c.RopeTail {
				a.Offset = int(op.Span) - r.NRot
			}
			return b.Emit(graph.OpRope, []graph.ValueID{v}, []graph.Result{{Name: "rope", Width: b.G.Values[v].Width}}, nil, a, nil, nil)[0], nil
		case jlm.TOpNGramRows, jlm.TOpTailConv:
			if !c.LayerKind(li).Recurrent() {
				return 0, fmt.Errorf("block %d keeps a recurrent tail and is no recurrent block", li)
			}
			g := c.Delta()
			base := g.VHeads * g.VDim * g.KDim
			at := b.G.StateVersion(graph.StateRecState)
			in, out := []graph.StateRef{{State: graph.StateRecState, Version: at}}, []graph.StateRef{{State: graph.StateRecState, Version: at + 1}}
			off := base + cursor
			if op.Code == jlm.TOpTailConv {
				w, taps, dil := int(op.Width), int(op.Span), int(op.Offset)
				cursor += (taps - 1) * dil * w
				return b.Emit(graph.OpStateConv, ins, []graph.Result{{Name: "tail.conv", Width: w}},
					[]graph.Binding{{Role: op.Bind[0], K: taps, Rows: w}},
					graph.Attrs{Taps: taps, Dil: dil, Offset: off, Span: stateSpan(c, base)}, in, out)[0], nil
			}
			hs := op.Hash
			n := int(op.Span)
			cursor += n - 1
			heads := (n - 1) * int(hs.HeadsPer)
			if heads <= 0 || int(op.Width)%heads != 0 || len(hs.Off) < heads || len(hs.Voc) < heads {
				return 0, fmt.Errorf("block %d gathers %d-grams %d wide over %d heads' ranges", li, n, op.Width, len(hs.Off))
			}
			rowW := int(op.Width) / heads
			tok := b.Emit(graph.OpToken, nil, []graph.Result{{Name: "ng.tok", Width: 1, Elem: graph.ElemU32}}, nil, graph.Attrs{}, nil, nil)[0]
			win := b.Emit(graph.OpNGramWindow, []graph.ValueID{tok}, []graph.Result{{Name: "ng.win", Width: n, Elem: graph.ElemU32}}, nil,
				graph.Attrs{Taps: n, Offset: off, Span: stateSpan(c, base), Words: []uint64{uint64(hs.Image)}}, in, out)[0]
			words := append([]uint64{uint64(hs.EOS)}, hs.Mul...)
			for h := range heads {
				words = append(words, hs.Off[h], hs.Voc[h])
			}
			hashed := b.Emit(graph.OpNGramHash, []graph.ValueID{win}, []graph.Result{
				{Name: "ng.rows", Width: heads, Elem: graph.ElemU32}, {Name: "ng.next", Width: n - 1, Elem: graph.ElemU32}}, nil,
				graph.Attrs{Span: int(hs.HeadsPer), Words: words, Reset: true}, nil, nil)
			rows := make([]graph.ValueID, heads)
			for h := range heads {
				col := b.Emit(graph.OpSlice, []graph.ValueID{hashed[0]}, []graph.Result{{Name: "ng.row", Width: 1, Elem: graph.ElemU32}}, nil,
					graph.Attrs{Offset: h, Width: 1}, nil, nil)[0]
				rows[h] = b.Emit(graph.OpEmbed, []graph.ValueID{col}, []graph.Result{{Name: "ng.emb", Width: rowW}},
					[]graph.Binding{{Role: op.Bind[0], K: rowW, Rows: int(hs.Voc[h]), From: int(hs.Off[h])}},
					graph.Attrs{}, nil, nil)[0]
			}
			return b.Emit(graph.OpConcat, rows, []graph.Result{{Name: "ngram", Width: int(op.Width)}}, nil, graph.Attrs{}, nil, nil)[0], nil
		}
		return 0, fmt.Errorf("op code %d is no block's", op.Code)
	}
}

// ropeLayout is a rotary's layout, neox-style or adjacent pairs.
func ropeLayout(neox bool) graph.RopeLayout {
	if neox {
		return graph.RopeNeox
	}
	return graph.RopeNorm
}

// zerosOf is a row of zeros as wide as x, or w wide (no wider than x).
func zerosOf(b *graph.Builder, name string, x graph.ValueID, w ...int) graph.ValueID {
	width := b.G.Values[x].Width
	if len(w) > 0 {
		cut := b.Emit(graph.OpSlice, []graph.ValueID{x}, []graph.Result{{Name: name + ".cut", Width: w[0]}}, nil,
			graph.Attrs{HeadDim: width, Width: w[0]}, nil, nil)[0]
		return mapRow(b, name, graph.Attrs{Fn: graph.MapMul}, nil, w[0], cut)
	}
	return mapRow(b, name, graph.Attrs{Fn: graph.MapMul}, nil, width, x)
}

// attnTemp is the attention temperature 1 + s ln(1 + floor((pos + o) / f))
// at the row's history position: the position op's whole-number offset and
// division where o is a whole number at or above zero, and otherwise the
// floor taken in the expression -- x rounded, less one where x lies below
// its rounding, exact for any x.
func attnTemp(b *graph.Builder, c *Config) graph.ValueID {
	s, o, f := c.AttnTempScale, c.AttnTempOffs, c.AttnTempFloor
	in := graph.Arg{Kind: graph.ArgIn, Index: 0}
	term := func(i int) graph.Arg { return graph.Arg{Kind: graph.ArgTerm, Index: i} }
	num := func(v float64) graph.Arg { return graph.Arg{Kind: graph.ArgConst, Const: v} }
	if o >= 0 && o == math.Trunc(o) {
		pos := b.Emit(graph.OpPosition, nil, []graph.Result{{Name: "pos", Width: 1}}, nil,
			graph.Attrs{Offset: int(o), Span: int(f)}, nil, nil)[0]
		return b.Expr("temp", 1, []graph.Term{
			{Fn: graph.MapAdd, Args: [3]graph.Arg{in, num(1)}},
			{Fn: graph.MapLog, Args: [3]graph.Arg{term(0)}},
			{Fn: graph.MapFma, Args: [3]graph.Arg{term(1), num(s), num(1)}},
		}, nil, pos)
	}
	pos := b.Emit(graph.OpPosition, nil, []graph.Result{{Name: "pos", Width: 1}}, nil, graph.Attrs{}, nil, nil)[0]
	return b.Expr("temp", 1, []graph.Term{
		{Fn: graph.MapAdd, Args: [3]graph.Arg{in, num(o)}},
		{Fn: graph.MapDiv, Args: [3]graph.Arg{term(0), num(f)}},
		{Fn: graph.MapRound, Args: [3]graph.Arg{term(1)}},
		{Fn: graph.MapLess, Args: [3]graph.Arg{term(1), term(2)}},
		{Fn: graph.MapSub, Args: [3]graph.Arg{term(2), term(3)}},
		{Fn: graph.MapAdd, Args: [3]graph.Arg{term(4), num(1)}},
		{Fn: graph.MapLog, Args: [3]graph.Arg{term(5)}},
		{Fn: graph.MapFma, Args: [3]graph.Arg{term(6), num(s), num(1)}},
	}, nil, pos)
}
