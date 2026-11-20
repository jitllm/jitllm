package convert

import (
	_ "embed"
	"encoding/json"
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

//go:embed front.json
var frontJSON []byte

// frontPart is how a part outside the blocks is stated: the steps it runs
// (front.json, by the part's name) and what its expressions read beside the
// model's facts.
type frontPart struct {
	hidden, collapsed bool
	// markerEps is a marker head's blocks' LayerNorm epsilon (markerEpsOf).
	markerEps float64
}

// markerEps is the part's marker head epsilon, 0 for a block's facts.
func (bf *blockFacts) markerEps() float64 {
	if bf.part == nil {
		return 0
	}
	return bf.part.markerEps
}

// markerEpsOf is the epsilon of the first LayerNorm of c's first head
// block's stated graph: the scorer's norm shares it.
func markerEpsOf(c *jlm.Config, s *jlm.Source, names tensorNames) (float64, error) {
	li := int(c.NLayer)
	g, err := graph.Decode([]byte(c.Blocks.Graph(li)), li, blockFactsOf(c, s, li, names).facts)
	if err != nil {
		return 0, fmt.Errorf("convert: the marker head's first block: %w", err)
	}
	for _, o := range g.Ops {
		if o.Kind == graph.OpLayerNorm && o.Attr.Eps > 0 {
			return o.Attr.Eps, nil
		}
	}
	return 0, fmt.Errorf("convert: the marker head's first block states no LayerNorm")
}

// frontFacts are what a part outside the blocks is stated over: the model's
// facts as a block's graph reads them, the tensors the container stores
// model-wide in place of a block's, the part's own (frontPart).
func frontFacts(c *jlm.Config, s *jlm.Source, names tensorNames, p frontPart) *blockFacts {
	bf := blockFactsOf(c, s, 0, names)
	bf.facts = graph.Facts{Inventory: map[jlm.Role]graph.Shape{}}
	for _, t := range s.Tensors {
		if t.Block != jlm.DenseBlock || t.Role.Vision() || predictRole(t.Role) || t.Index >= 0 {
			continue
		}
		sh := graph.Shape{K: int(t.Dims[0])}
		if t.NDim >= 2 {
			sh.Rows = int(t.Dims[1])
		}
		if t.NDim >= 3 {
			sh.Sheets = int(t.Dims[2])
		}
		bf.facts.Inventory[t.Role] = sh
	}
	bf.part = &p
	return bf
}

// predictFacts are frontFacts over prediction block li's own entry and head
// tensors in place of the model's, the trunk's token table standing in for
// one the block does not carry.
func predictFacts(c *jlm.Config, s *jlm.Source, names tensorNames, li int) *blockFacts {
	bf := frontFacts(c, s, names, frontPart{})
	bf.facts = graph.Facts{Inventory: map[jlm.Role]graph.Shape{}}
	for _, t := range s.Tensors {
		if int(t.Block) != li || !predictRole(t.Role) || t.Index >= 0 {
			continue
		}
		sh := graph.Shape{K: int(t.Dims[0])}
		if t.NDim >= 2 {
			sh.Rows = int(t.Dims[1])
		}
		bf.facts.Inventory[t.Role] = sh
	}
	if _, ok := bf.facts.Inventory[jlm.RolePredictEmbd]; !ok {
		bf.facts.Inventory[jlm.RolePredictEmbd] = graph.Shape{K: int(c.NEmbd), Rows: int(c.NVocab)}
	}
	return bf
}

// StateFront states the parts of c outside its blocks that front.json
// states, over the tensors s stores model-wide: the head, with and without
// the norm's rows, over one stream or the collapse of several, on a model
// that projects to its vocabulary; the input rows wherever the front end
// reads them; a given row's where a position table, the per-layer inputs or
// the streams the input rows expand into add to it; the readout of an
// embedding model, a pointer, a marker head or a joint head; each prediction
// block's entry, and its head over its own mix of the streams where it
// stores one; the output norm alone where the model stores one. Each is
// held to what the container stores (graph.Verify), each op it runs added to
// taken, by its part, where that is not nil.
func StateFront(c *jlm.Config, s *jlm.Source, taken map[string]bool) error {
	var parts map[string][]mjBlockOp
	if err := json.Unmarshal(frontJSON, &parts); err != nil {
		return fmt.Errorf("convert: front.json: %w", err)
	}
	names, err := readTensorNames()
	if err != nil {
		return err
	}
	g := &mjGraph{}
	for name, steps := range parts {
		if err := g.checkOps(steps, map[string]bool{}, nil, names); err != nil {
			return fmt.Errorf("convert: front.json's %s: %w", name, err)
		}
	}
	state := func(part, as string, p frontPart, li int) error {
		bf := frontFacts(c, s, names, p)
		if li >= 0 && p.collapsed {
			bf = predictHeadFacts(c, s, names, p, li)
		} else if li >= 0 {
			bf = predictFacts(c, s, names, li)
		}
		var ran map[string]bool
		if taken != nil {
			ran = map[string]bool{}
		}
		b := graph.NewBlock(li)
		if err := g.run(b, parts[part], bf, li, ran); err != nil {
			return fmt.Errorf("convert: the stated %s: %w", as, err)
		}
		for f := range ran {
			taken[frontFeature(part, f)] = true
		}
		if err := graph.Verify(b.G, bf.facts); err != nil {
			return fmt.Errorf("convert: the stated %s: %w", as, err)
		}
		e, err := graph.Encode(b.G)
		if err != nil {
			return err
		}
		c.Front.Names, c.Front.Graphs = append(c.Front.Names, as), append(c.Front.Graphs, string(e))
		return nil
	}
	c.Front = jlm.FrontGraphs{}
	streams := max(int(c.HCMult), int(c.Streams.N), 1)
	if c.Flags.Pooling() == jlm.PoolNone && !c.Flags2.Has(jlm.Flag2HeadBlocks) && streams == 1 {
		if err := state("head", jlm.FrontHead, frontPart{}, -1); err != nil {
			return err
		}
		if err := state("head", jlm.FrontHeadHidden, frontPart{hidden: true}, -1); err != nil {
			return err
		}
	}
	if c.Flags.Pooling() == jlm.PoolNone && !c.Flags2.Has(jlm.Flag2HeadBlocks) && streams > 1 {
		if err := state("head", jlm.FrontCollapsed, frontPart{collapsed: true}, -1); err != nil {
			return err
		}
		if err := state("head", jlm.FrontCollapsedHidden, frontPart{hidden: true, collapsed: true}, -1); err != nil {
			return err
		}
	}
	expands := len(c.Streams.Expand) != 0
	states := statedStreams(c) > 1 && streams == statedStreams(c)
	if streams == 1 || states || expands {
		if err := state("embed", jlm.FrontEmbed, frontPart{}, -1); err != nil {
			return err
		}
	}
	if storesDense(s, jlm.RolePosEmbd) || c.PLEDim != 0 || expands {
		if err := state("given", jlm.FrontGiven, frontPart{}, -1); err != nil {
			return err
		}
	}
	switch {
	case c.JointHead != nil:
		if err := state("joint", jlm.FrontPool, frontPart{}, int(jlm.DenseBlock)); err != nil {
			return err
		}
	case c.Flags2.Has(jlm.Flag2MarkerPointer):
		if err := state("pointer", jlm.FrontPool, frontPart{}, int(jlm.DenseBlock)); err != nil {
			return err
		}
	case c.Flags2.Has(jlm.Flag2HeadBlocks):
		eps, err := markerEpsOf(c, s, names)
		if err != nil {
			return err
		}
		if err := state("mark", jlm.FrontPool, frontPart{markerEps: eps}, int(jlm.DenseBlock)); err != nil {
			return err
		}
	case c.Flags.Pooling() != jlm.PoolNone:
		if err := state("pool", jlm.FrontPool, frontPart{}, int(jlm.DenseBlock)); err != nil {
			return err
		}
	}
	mix := headMixRoles(c, s, names)
	for i := range int(c.PredictionBlocks()) {
		li := int(c.NLayer) + i
		if err := state("predict_entry", jlm.FrontPredict(jlm.FrontPredictEntry, i), frontPart{}, li); err != nil {
			return err
		}
		if !states || !storesAny(s, li, mix) {
			continue
		}
		if err := state("head", jlm.FrontPredict(jlm.FrontPredictHead, i), frontPart{collapsed: true}, li); err != nil {
			return err
		}
		if err := state("head", jlm.FrontPredict(jlm.FrontPredictHidden, i), frontPart{hidden: true, collapsed: true}, li); err != nil {
			return err
		}
	}
	if storesDense(s, jlm.RoleOutputNorm) {
		if err := state("outnorm", jlm.FrontOutNorm, frontPart{}, int(jlm.DenseBlock)); err != nil {
			return err
		}
	}
	return nil
}

// predictHeadFacts are frontFacts over the model's tensors and prediction
// block li's own beside them, its entry's and head's aside: the head's read
// of the streams binds the block's own mix.
func predictHeadFacts(c *jlm.Config, s *jlm.Source, names tensorNames, p frontPart, li int) *blockFacts {
	bf := frontFacts(c, s, names, p)
	for _, t := range s.Tensors {
		if int(t.Block) != li || t.Role.Vision() || predictRole(t.Role) || t.Index >= 0 {
			continue
		}
		sh := graph.Shape{K: int(t.Dims[0])}
		if t.NDim >= 2 {
			sh.Rows = int(t.Dims[1])
		}
		if t.NDim >= 3 {
			sh.Sheets = int(t.Dims[2])
		}
		bf.facts.Inventory[t.Role] = sh
	}
	return bf
}

// headMixRoles are the tensors the head's stated read of the streams binds
// over the model's tensors, nil on a model that states no streams.
func headMixRoles(c *jlm.Config, s *jlm.Source, names tensorNames) map[jlm.Role]bool {
	n := statedStreams(c)
	if n < 2 {
		return nil
	}
	bf := frontFacts(c, s, names, frontPart{})
	b := graph.NewBlock(-1)
	xs := make([]graph.ValueID, n)
	for k := range xs {
		xs[k] = b.In(fmt.Sprintf("x.%d", k), int(c.NEmbd))
	}
	if _, _, err := graph.StreamRead(b, bf.facts, &c.Streams.Head, xs, xs[0], xs[0], 0, nil); err != nil {
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

// storesAny reports whether s stores one of roles for block li.
func storesAny(s *jlm.Source, li int, roles map[jlm.Role]bool) bool {
	for _, t := range s.Tensors {
		if int(t.Block) == li && t.Index < 0 && roles[t.Role] {
			return true
		}
	}
	return false
}

// storesDense reports whether s stores role model-wide.
func storesDense(s *jlm.Source, role jlm.Role) bool {
	for _, t := range s.Tensors {
		if t.Block == jlm.DenseBlock && t.Role == role && t.Index < 0 {
			return true
		}
	}
	return false
}
