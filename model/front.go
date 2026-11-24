package model

import (
	"errors"
	"fmt"
	"maps"
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
	UncoveredExperts
	UncoveredStreams
	UncoveredPooling
	UncoveredUnbound
	UncoveredUnverified
	UncoveredUnstated
	UncoveredUnstatedPart
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
	case UncoveredUnstatedPart:
		return "a part outside the blocks the container states no graph for: it was written before the converter " +
			"stated them, and runs once jitllm upgrade states them over what it holds"
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
	// cpuHeadWhy and cpuHeadErr are why cpuHead is nil.
	cpuHeadWhy Uncovered
	cpuHeadErr error
	// cpuSample and cpuWalk are sample and a top-k draw's walk over
	// cpuHead's logits; cpuRow and cpuCut the draw with no top-k, with no cut
	// and with one; cpuWalkProb and cpuCutProb the distributions the walk and
	// the cut draw from; cpuCutK and cpuCutKProb a top-k draw past the
	// walk's bound and its distribution; the four Typ graphs are the cut's
	// with typical-p (graph.SampleCut's Typical), apart so a draw without it
	// pays for none of it.
	CpuSample, CpuWalk, CpuRow, CpuCut *graph.Graph
	// CpuWalkInfill is a top-k draw's walk through the fill-in-the-middle
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

// statedRotaries states the model's further rotaries (Model.Ropes): each a
// width, base and magnitude a block's stated rotary op turns by that the
// model's rotary does not -- the sliding layers' own -- its pairs divided by
// the model's factors only where the op reads them. A block whose graph does
// not decode adds none; its error is FrontGraphs'.
func (m *Model) statedRotaries() error {
	m.Ropes = nil
	same := func(r *nn.Rope, a graph.Attrs) bool {
		return r.NRot == a.NRot && r.Base == a.Base && scaleOne(r.Scale) == scaleOne(a.Mscale) && (len(r.Freqs) > 0) == a.Freqs
	}
	bg := &m.Container.Config().Blocks
	seen := map[uint16]bool{}
	for li := range min(len(bg.Of), m.Cfg.NLayer+m.Cfg.NPredict) {
		if seen[bg.Of[li]] {
			continue
		}
		seen[bg.Of[li]] = true
		g, _, err := m.blockGraph(li)
		if err != nil || g == nil {
			continue
		}
		for _, o := range g.Ops {
			if o.Kind != graph.OpRope || same(&m.Rope, o.Attr) ||
				slices.ContainsFunc(m.Ropes, func(r nn.Rope) bool { return same(&r, o.Attr) }) {
				continue
			}
			if len(m.Rope.Runs) > 0 {
				return fmt.Errorf("model: block %d turns a rotary of its own beside the model's rotary sections", li)
			}
			r := m.Rope
			r.NRot, r.Base, r.Scale = o.Attr.NRot, o.Attr.Base, o.Attr.Mscale
			if !o.Attr.Freqs {
				r.Freqs = nil
			}
			m.Ropes = append(m.Ropes, r)
		}
	}
	return nil
}

// scaleOne is a rotary magnitude, 0 read as one.
func scaleOne(s float64) float64 {
	if s == 0 {
		return 1
	}
	return s
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
	for li := range m.Cfg.NLayer + m.Cfg.NPredict {
		if (Composing{m.Container, m}).BlockGraph(li) == nil {
			why, err := m.UncoveredAt(li)
			return refuse(fmt.Sprintf("block %d", li), why, err)
		}
	}
	switch {
	case f.CpuHead == nil && f.cpuHeadWhy != UncoveredPooling:
		return refuse("the head", f.cpuHeadWhy, f.cpuHeadErr)
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
	if np := m.Cfg.NPredict; np > 0 {
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
	s.cpuHeadWhy, s.cpuHeadErr = s.HeadWhy, s.headErr
	if m.Cfg.Streams() > 1 {
		s.CpuHead, s.cpuHeadWhy, s.cpuHeadErr = m.headGraphOf(false, true)
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
	pg := int32(m.Container.PageOf(int(block)))
	for i := range m.Container.Entries() {
		e := &m.Container.Entries()[i]
		if e.Block != pg || e.Role.Vision() || predictRole(e.Role) {
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
	_, ok := m.Container.Find(role, int32(m.Container.PageOf(li)), -1)
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

// statedPart is the graph the container states for the part named name,
// decoded as block li's over f, and why it has none: a part a container
// written before v34 lacks until jitllm upgrade states it.
func (m *Model) statedPart(name string, li int, f graph.Facts) (*graph.Graph, Uncovered, error) {
	s := m.Container.Config().Front.Graph(name)
	if s == "" {
		return nil, UncoveredUnstatedPart, fmt.Errorf("model: the container states no %s: jitllm upgrade states it", name)
	}
	g, err := graph.Decode([]byte(s), li, f)
	if err != nil {
		return nil, UncoveredUnverified, fmt.Errorf("model: the stated %s: %w", name, err)
	}
	return g, Covered, nil
}

// headPart is the name of the head's part: with hidden the norm's rows
// written too, collapsed over the mix of a model's streams.
func headPart(hidden, collapsed bool) string {
	switch {
	case collapsed && hidden:
		return jlm.FrontCollapsedHidden
	case collapsed:
		return jlm.FrontCollapsed
	case hidden:
		return jlm.FrontHeadHidden
	}
	return jlm.FrontHead
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

// statedHead fills marker head h's attention and norms from its first
// block's stated graph: the attention's heads, their width and scale, and the
// LayerNorms' epsilon, which the scorer's norm shares.
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

// headGraph is the head the container states: the row's norm, its
// projection to the vocabulary and what the logits take after it. With
// hidden it writes the norm's rows too.
func (m *Model) headGraph(hidden bool) (*graph.Graph, Uncovered, error) {
	return m.headGraphOf(hidden, false)
}

// headGraphOf is headGraph, with collapsed over the collapse of a model's
// several streams, which the head of one does not read.
func (m *Model) headGraphOf(hidden, collapsed bool) (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	switch {
	case c.Pooling != jlm.PoolNone || c.Head != nil:
		return nil, UncoveredPooling, nil
	case c.Streams() > 1 && !collapsed:
		return nil, UncoveredStreams, nil
	}
	f, _ := m.Inventory(jlm.DenseBlock)
	return m.statedPart(headPart(hidden, collapsed), -1, f)
}

// mtpHeadGraph is the head the container states over prediction block li's
// own mix of the streams (with hidden, the norm's rows too), nil where it
// stores none.
func (m *Model) mtpHeadGraph(hidden bool, li int) (*graph.Graph, error) {
	c := m.Cfg
	if !c.StatesStreams() || !m.storesHeadMix(li) {
		return nil, nil
	}
	f, _ := m.Inventory(jlm.DenseBlock)
	part := jlm.FrontPredict(jlm.FrontPredictHead, li-c.NLayer)
	if hidden {
		part = jlm.FrontPredict(jlm.FrontPredictHidden, li-c.NLayer)
	}
	fb, _ := m.Inventory(int32(li))
	inv := maps.Clone(f.Inventory)
	maps.Copy(inv, fb.Inventory)
	g, _, err := m.statedPart(part, li, graph.Facts{Inventory: inv})
	return g, err
}

// poolGraph is the readout the container states in place of a head: an
// embedding model's pool, a pointer's, a marker head's or a joint head's;
// nil for a model that reads none out.
func (m *Model) poolGraph() (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	if c.Pooling == jlm.PoolNone && !c.Joint && !c.MarkPointer && c.Head == nil {
		return nil, Covered, nil
	}
	all, _ := m.Inventory(jlm.DenseBlock)
	return m.statedPart(jlm.FrontPool, int(jlm.DenseBlock), all)
}

// pickHeadGraph is the stated head projecting the rows of nn.PickMax ids
// alone (pickOf), nil for a head the container states none of.
func (m *Model) pickHeadGraph() (*graph.Graph, error) {
	g, why, err := m.headGraph(false)
	if g == nil || why != Covered || err != nil {
		return nil, err
	}
	f, _ := m.Inventory(jlm.DenseBlock)
	pg, err := pickOf(g, m.Cfg.NVocab, nn.PickMax)
	if err != nil {
		return nil, err
	}
	if err := graph.Verify(pg, f); err != nil {
		return nil, fmt.Errorf("the head's pick graph does not verify: %w", err)
	}
	return pg, nil
}

// pickOf is head h projecting the rows of pick ids alone, the ids its second
// input: its one projection to the vocabulary reads the ids beside its row,
// and the logits and every row made from them are pick wide.
func pickOf(h *graph.Graph, nVocab, pick int) (*graph.Graph, error) {
	g := &graph.Graph{Version: h.Version, Block: h.Block, Values: slices.Clone(h.Values), Ops: slices.Clone(h.Ops),
		In: slices.Clone(h.In), Out: slices.Clone(h.Out), Routed: h.Routed}
	proj := -1
	for i, o := range g.Ops {
		if o.Kind == graph.OpMatVec && len(o.Out) == 1 && g.Values[o.Out[0]].Width == nVocab {
			if proj >= 0 {
				return nil, fmt.Errorf("model: the head projects to its vocabulary twice")
			}
			proj = i
		}
	}
	if proj < 0 || len(g.Out) != 1 {
		return nil, fmt.Errorf("model: a head of %d outputs with no one projection to its vocabulary", len(g.Out))
	}
	ids := graph.ValueID(len(g.Values))
	g.Values = append(g.Values, graph.Value{Name: "pick.ids", Width: pick, Elem: graph.ElemU32})
	g.In = append(g.In, ids)
	g.Ops[proj].In = append(slices.Clone(g.Ops[proj].In), ids)
	picked := map[graph.ValueID]bool{g.Ops[proj].Out[0]: true}
	g.Values[g.Ops[proj].Out[0]].Width = pick
	for _, o := range g.Ops[proj+1:] {
		if !slices.ContainsFunc(o.In, func(v graph.ValueID) bool { return picked[v] }) {
			continue
		}
		for _, v := range o.Out {
			if g.Values[v].Width == nVocab {
				g.Values[v].Width, picked[v] = pick, true
			}
		}
	}
	return g, nil
}

// embedGraph is the step's input rows as the container states them: each
// row's token's row and what the model adds to it, with streams the streams
// the input rows expand into, then the per-layer inputs.
func (m *Model) embedGraph(streams bool) (*graph.Graph, Uncovered, error) {
	c := m.Cfg
	if streams && c.Streams() > 1 && !c.StatesStreams() && !m.ExpandsStreams() {
		return nil, UncoveredStreams, nil
	}
	f, _ := m.Inventory(jlm.DenseBlock)
	return m.statedPart(jlm.FrontEmbed, -1, f)
}

// outNormGraph is the output norm alone over a row as the container states
// it: the trunk's hidden state a prediction block reads of a speculation's
// prompt rows. nil for a model that stores no output norm, whose hidden
// state is the row itself.
func (m *Model) outNormGraph() (*graph.Graph, error) {
	all, _ := m.Inventory(jlm.DenseBlock)
	if _, ok := all.Inventory[jlm.RoleOutputNorm]; !ok {
		return nil, nil
	}
	g, _, err := m.statedPart(jlm.FrontOutNorm, int(jlm.DenseBlock), all)
	return g, err
}

// givenGraph is an embedding span's rows' input as the container states it:
// what a position table, the streams the input rows expand into and the
// per-layer inputs add to a row the caller gives; nil for a model that adds
// nothing to a given row.
func (m *Model) givenGraph() (*graph.Graph, error) {
	f, _ := m.Inventory(jlm.DenseBlock)
	if _, pos := f.Inventory[jlm.RolePosEmbd]; !pos && m.Cfg.PLEDim == 0 && !m.ExpandsStreams() {
		return nil, nil
	}
	g, _, err := m.statedPart(jlm.FrontGiven, -1, f)
	return g, err
}

// MaxTopLogprobs is the alternatives the log-softmax's own selection orders
// per token (CpuLogprobs), the OpenAI API's bound on top_logprobs; past it a
// Logprobs reads the whole row in order (CpuLogprobsRow).
const MaxTopLogprobs = 20
