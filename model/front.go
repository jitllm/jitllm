package model

import (
	"errors"
	"fmt"
	"maps"
	"math"
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
	UncoveredExperts
	UncoveredStreams
	UncoveredPooling
	UncoveredUnbound
	UncoveredUnverified
	UncoveredUnstated
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
	case UncoveredExperts:
		return "routed experts stored one matrix each (a container converted before its experts were stacked into banks)"
	case UncoveredStreams:
		return "several residual streams"
	case UncoveredPooling:
		return "a readout over the rows in place of a head (an embedding model's pool, a marker head's scorer)"
	case UncoveredUnbound:
		return "stored tensors its graph does not bind"
	case UncoveredUnverified:
		return "a graph that does not verify"
	case UncoveredUnstated:
		return "a block the container states no graph for: it was converted before the converter stated its blocks, " +
			"and runs once its source is converted again (jitllm convert)"
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
	// derived is each block graph made over (DerivedBlockGraph), by the
	// graph it was made from and the caller's key.
	derived map[derivedKey]*graph.Graph
}

type derivedKey struct {
	from *graph.Graph
	key  int
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
	// (mtpEntryGraph); MtpHead and MtpHidden the head over its own mix of
	// the streams where it stores one (mtpHeadGraph), nil where it does
	// not.
	MtpEntry           []*graph.Graph
	mtpEntryWhy        []Uncovered
	MtpHead, MtpHidden []*graph.Graph
	Head, Embed        *graph.Graph
	// cpuEmbed is embed as the CPU composes it, every stream the input
	// rows expand into among its outputs; given is an embedding span's rows'
	// input (givenGraph); outNorm the
	// output norm alone (outNormGraph). cpuInErr is why any of them did not
	// verify.
	CpuEmbed, Given, OutNorm *graph.Graph
	CpuInErr                 error
	// headHidden is head writing the norm's rows too (nn.Head.HiddenGraph).
	headHidden *graph.Graph
	// cpuHead and cpuHidden are head and headHidden as the devices compose
	// them: over the mix of every residual stream the head reads first, one
	// input a stream, on a model of several; the same graphs on a model of
	// one.
	CpuHead, CpuHidden *graph.Graph
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
	// (graph.SampleStage), its candidates a top-k draw's at most; CpuKeep
	// min_keep over one of its cuts (graph.SampleKeep).
	CpuStage, CpuKeep *graph.Graph
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
	cfg      *Config
	head     headKey
	rope     ropeKey
	ropes    [nn.MaxRopeTables - 1]ropeKey
	restated int
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

// DerivedBlockGraph is block li's graph made over by derive, once for the
// graph the block states and key: every caller asking with one key is handed
// one graph, as a device shares a block only among the sessions that offer
// it the same graph. Nil where the block states none.
func (m *Model) DerivedBlockGraph(li, key int, derive func(*graph.Graph) *graph.Graph) *graph.Graph {
	g := (Composing{m.Container, m}).BlockGraph(li)
	if g == nil {
		return nil
	}
	f := &m.Front
	f.Mu.Lock()
	defer f.Mu.Unlock()
	k := derivedKey{g, key}
	if d, ok := f.derived[k]; ok {
		return d
	}
	if f.derived == nil {
		f.derived = map[derivedKey]*graph.Graph{}
	}
	d := derive(g)
	f.derived[k] = d
	return d
}

func (c Composing) SeamGraph(int) *graph.Graph { return nil }

func (c Composing) HeadGraph() *graph.Graph { return c.m.FrontGraphs().Head }

func (c Composing) EmbedGraph() *graph.Graph { return c.m.FrontGraphs().Embed }

// Rotaries is the model's rotary, then each further one (Model.Ropes): the
// tables the blocks' rotaries turn by.
func (c Composing) Rotaries() []graph.Rotary {
	out := []graph.Rotary{c.m.rotaryOf(c.m.Rope)}
	for _, r := range c.m.Ropes {
		out = append(out, c.m.rotaryOf(r))
	}
	return out
}

func (m *Model) rotaryOf(r nn.Rope) graph.Rotary {
	return graph.Rotary{NRot: 2 * r.TablePairs(), Base: r.Base, Freqs: len(r.Freqs) > 0, Axes: r.TableAxes()}
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
	k := frontKey{cfg: m.Cfg, head: headKeyOf(m.Cfg), rope: ropeKeyOf(&m.Rope), restated: f.Restated}
	for i := range min(len(m.Ropes), len(k.ropes)) {
		k.ropes[i] = ropeKeyOf(&m.Ropes[i])
	}
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
		s.MtpEntry, s.mtpEntryWhy = make([]*graph.Graph, np), make([]Uncovered, np)
		s.MtpHead, s.MtpHidden = make([]*graph.Graph, np), make([]*graph.Graph, np)
		for i := range np {
			s.mtp[i], s.mtpWhy[i], s.mtpErrs[i] = m.blockGraph(n + i)
			s.MtpEntry[i], s.mtpEntryWhy[i], _ = m.mtpEntryGraph(n + i)
			var he, hhe error
			s.MtpHead[i], he = m.mtpHeadGraph(false, n+i)
			s.MtpHidden[i], hhe = m.mtpHeadGraph(true, n+i)
			if err := errors.Join(he, hhe); err != nil {
				s.mtpWhy[i], s.mtpErrs[i] = UncoveredUnverified, errors.Join(s.mtpErrs[i], err)
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
	s.Embed, s.EmbedWhy, s.embedErr = m.embedGraph(true)
	var ce, ge, ne error
	s.CpuEmbed, _, ce = m.embedGraph(true)
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
		kp := graph.SampleKeep{NVocab: m.Cfg.NVocab, K: nn.SampleMaxK}
		s.CpuKeep = kp.Build()
		lp := graph.Logprobs{NVocab: m.Cfg.NVocab, K: MaxTopLogprobs}
		s.CpuLogprobs = lp.Build()
		la := graph.Logprobs{NVocab: m.Cfg.NVocab, K: m.Cfg.NVocab}
		s.CpuLogprobsRow = la.Build()
		mk := graph.MaskRow{NVocab: m.Cfg.NVocab}
		s.CpuMask = mk.Build()
	}
	if m.Container.Config().Decides() {
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
		if e, ok := m.Container.Find(r, jlm.DenseBlock, -1); ok && m.Container.RowTable(e) {
			inv[e.Role] = graph.Shape{K: e.K(), Rows: e.Rows()}
		}
	}
	if r := m.Front.Restate; r != nil {
		r(block, inv)
	}
	return graph.Facts{Inventory: inv}, perExpert
}

// stores reports whether the container stores role for block li.
func (m *Model) Stores(li int, role jlm.Role) bool {
	_, ok := m.Container.Find(role, int32(li), -1)
	return ok
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
	s := m.Container.Config().Blocks.Graph(li)
	if s == "" {
		return nil, UncoveredUnstated, nil
	}
	return m.statedBlockGraph(li, s)
}

// blockFacts is what the container stores for block li, as its graph binds
// it: a prediction block's own head mix is its head's graph's
// (mtpHeadGraph), not its body's.
func (m *Model) blockFacts(li int) (graph.Facts, Uncovered) {
	f, perExpert := m.Inventory(int32(li))
	if perExpert {
		return f, UncoveredExperts
	}
	if li >= m.Cfg.NLayer {
		for r := range m.headMixRoles() {
			delete(f.Inventory, r)
		}
	}
	return f, Covered
}

// headMixRoles is the tensors the head's stated read of the streams binds
// (streamMix), nil on a model that states none.
func (m *Model) headMixRoles() map[jlm.Role]bool {
	if !m.Cfg.StatesStreams() {
		return nil
	}
	f, _ := m.Inventory(jlm.DenseBlock)
	b := graph.NewBlock(-1)
	if _, err := m.streamMix(b, f); err != nil {
		return nil
	}
	rs := map[jlm.Role]bool{}
	for _, o := range b.G.Ops {
		for _, w := range o.Weight {
			rs[w.Role] = true
		}
	}
	return rs
}

// storesHeadMix says block li stores a tensor of the head's read of the
// streams: a mix of its own.
func (m *Model) storesHeadMix(li int) bool {
	for r := range m.headMixRoles() {
		if m.Stores(li, r) {
			return true
		}
	}
	return false
}

// statedBlockGraph is block li's graph as the converter states it (s,
// graph.Encode's), held to what the container stores for the block as a
// graph built here is.
func (m *Model) statedBlockGraph(li int, s string) (*graph.Graph, Uncovered, error) {
	f, why := m.blockFacts(li)
	if why != Covered {
		return nil, why, nil
	}
	g, err := graph.Decode([]byte(s), li, f)
	if err != nil {
		return nil, UncoveredUnverified, err
	}
	if u := graph.Unbound(g, f); len(u) != 0 {
		return nil, UncoveredUnbound, fmt.Errorf("the container stores %v for block %d, which its stated graph does not bind", u, li)
	}
	// A graph stated before its attention carried the key-run rule decodes
	// it as false, which on a layer the container keeps causal over a
	// picture would run it bidirectional.
	for _, o := range g.Ops {
		if o.Kind == graph.OpAttend && o.Attr.NoRuns == m.Cfg.BidirAt(li) {
			return nil, UncoveredUnverified, fmt.Errorf("block %d's stated graph honours a picture's key runs %v where the container's layers say %v: "+
				"convert the model again", li, !o.Attr.NoRuns, m.Cfg.BidirAt(li))
		}
	}
	return g, Covered, nil
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

// mapRow is one OpMap of in by a into a row of width, binding w.
func mapRow(b *graph.Builder, name string, a graph.Attrs, w []graph.Binding, width int, in ...graph.ValueID) graph.ValueID {
	return b.Map(name, a, w, width, in...)
}

// statedHead fills marker head h's attention and norms from its first
// block's stated graph: the attention's heads, their width and scale, and the
// LayerNorms' epsilon, which the scorer's norm shares (markGraph).
func (m *Model) statedHead(h *Config) error {
	c, b := m.Container.Config(), h.Trunk
	if b >= len(c.Blocks.Of) || int(c.Blocks.Of[b]) >= len(c.Blocks.Graphs) {
		return fmt.Errorf("model: the marker head's block %d states no graph: convert the model again", b)
	}
	f, _ := m.blockFacts(b)
	g, err := graph.Decode([]byte(c.Blocks.Graphs[c.Blocks.Of[b]]), b, f)
	if err != nil {
		return fmt.Errorf("model: the marker head's block %d: %w", b, err)
	}
	var att, norm *graph.Attrs
	for i := range g.Ops {
		switch o := &g.Ops[i]; {
		case o.Kind == graph.OpAttend && att == nil:
			att = &o.Attr
		case o.Kind == graph.OpLayerNorm && norm == nil:
			norm = &o.Attr
		}
	}
	if att == nil || norm == nil || att.NHead <= 0 || att.NHead*att.HeadDim != h.NEmbd || !(norm.Eps > 0) {
		return fmt.Errorf("model: the marker head's block %d states no attention over its %d-wide rows beside a LayerNorm", b, h.NEmbd)
	}
	h.NHead, h.NKVHead, h.HeadDim, h.HeadDimV, h.AttnScale = att.NHead, att.NKVHead, att.HeadDim, att.HeadDim, att.Scale
	h.RMSEps, h.LatentNormEps = norm.Eps, norm.Eps
	return nil
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
	if collapsed && c.StatesStreams() {
		var err error
		if g, err = m.streamHeadGraph(&h, -1); err != nil {
			return nil, UncoveredUnverified, err
		}
		return g, Covered, nil
	}
	if err := graph.Verify(g, f); err != nil {
		return nil, UncoveredUnverified, fmt.Errorf("the head's graph does not verify: %w", err)
	}
	return g, Covered, nil
}

// streamHeadGraph is head h over the mix of every residual stream the
// converter states as the head's read: one input a stream, block li's head
// roles where li is a prediction block's (its own mix), the model's for -1.
func (m *Model) streamHeadGraph(h *graph.Head, li int) (*graph.Graph, error) {
	f, _ := m.Inventory(jlm.DenseBlock)
	if li >= 0 {
		fb, _ := m.Inventory(int32(li))
		inv := maps.Clone(f.Inventory)
		maps.Copy(inv, fb.Inventory)
		f.Inventory = inv
	}
	b := graph.NewBlock(li)
	y, err := m.streamMix(b, f)
	if err != nil {
		return nil, err
	}
	h.Over(b, y)
	if err := graph.Verify(b.G, f); err != nil {
		return nil, fmt.Errorf("the head's graph over the streams does not verify: %w", err)
	}
	return b.G, nil
}

// mtpHeadGraph is the head (hidden, the norm's rows too) over prediction
// block li's own mix of the streams, nil where it stores none.
func (m *Model) mtpHeadGraph(hidden bool, li int) (*graph.Graph, error) {
	c := m.Cfg
	if !c.StatesStreams() || !m.storesHeadMix(li) {
		return nil, nil
	}
	f, _ := m.Inventory(jlm.DenseBlock)
	stored := func(r jlm.Role) bool {
		_, ok := f.Inventory[r]
		return ok
	}
	h := graph.Head{NEmbd: c.NEmbd, NVocab: c.NVocab, Eps: c.RMSEps, LayerNorm: c.LayerNorm,
		NormBias: stored(jlm.RoleOutputNormBias), Output: jlm.RoleOutput, Bias: stored(jlm.RoleOutputBias),
		Softcap: float64(c.FinalSoftcap), Scale: c.LogitScale, Hidden: hidden, NoNorm: !stored(jlm.RoleOutputNorm)}
	if !stored(h.Output) {
		h.Output = jlm.RoleTokenEmbd
	}
	g, err := m.streamHeadGraph(&h, li)
	if err != nil {
		return nil, fmt.Errorf("prediction block %d's own head: %w", li, err)
	}
	return g, nil
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
// position table moves it or the input rows expand it into streams, then
// those streams, then the per-layer inputs; nil for a model that
// adds nothing to a given row.
func (m *Model) givenGraph() (*graph.Graph, error) {
	c := m.Cfg
	f, _ := m.Inventory(jlm.DenseBlock)
	p, pos := f.Inventory[jlm.RolePosEmbd]
	expands := m.ExpandsStreams()
	if !pos && c.PLEDim == 0 && !expands {
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
	// The streams the input rows expand into follow the row, which the
	// graph then writes too.
	if expands {
		if !pos {
			b.Out(streamCopy(b, x))
		}
		xs, err := m.streamExpand(b, f, x)
		if err != nil {
			return nil, err
		}
		b.Out(xs...)
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
