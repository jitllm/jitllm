package nn

import (
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// What an instance hands a device to place its blocks and parts on.

// Tensors is where a form's weights are read: a block's copy, a tower's
// tensors or the model's own.
type Tensors interface {
	// Bytes is the matrix of role r: its bytes, its planes where it is
	// packed, and its rows; nil for one it does not hold.
	Bytes(r jlm.Role) ([]byte, *Packed, int)
	// Floats is the vector of role r, nil for one it does not hold.
	Floats(r jlm.Role) []float32
}

// Source is where an instance's blocks' weights are read in: Held says block
// li's page is in, so reading it reads nothing; Ensure reads it in where it
// is not and is the block's copy, bound at it, that its forms read.
type Source interface {
	Held(li int) bool
	Ensure(li int) (Tensors, error)
}

// ExpertPager is where the experts a block's selection picks are made resident
// and handed back: the instance's lease of its pages.
type ExpertPager interface {
	Experts(li int, sel []int32, count bool) error
	ReleaseExperts()
}

// Lease is what an execution's placement reads its blocks through: the
// experts a selection picks, paged in and handed back, and each block's
// weights, paged in.
type Lease interface {
	ExpertPager
	Source
}

// PartPlan is a part -- a graph apart from any block's residual: the head,
// a sampler, the input rows, a rotary table, the streams' mix, a readout --
// as the CPU device composes it: its weights' formats, the step's rows, the
// sequences and rotary marks a table reads and the activation window. None
// is why not, where the front end states no graph.
type PartPlan struct {
	Graph                        *graph.Graph
	Formats                      map[jlm.Role]jlm.Type
	Rows, Seqs, Marks, ActWindow int
	// Skip is a part the model has none of: not composed, not asked. A plan
	// with no graph is one.
	Skip bool
}

// PartsPlan is what the CPU device places an instance's parts from: each
// part's graphs as its plans, the weights each binds and the widths it runs
// at. Head is the head's and the samplers' plans in HeadParts' order;
// Streams, Pool and Entry say the instance has the residual streams' mix, a
// readout and a prediction block's entry; Tables is each rotary's table, a
// row's and a call's; KV is the history's conversions, nil for a history of
// floats.
type PartsPlan struct {
	Head     [20]PartPlan
	HeadNorm bool
	HeadW    PartWeights
	// D is a row's width, NVocab the vocabulary, TopLog the most log
	// probabilities a draw returns.
	D, NVocab, TopLog int

	Streams             bool
	StreamParts         [4]PartPlan
	StreamW, CollapseW  PartWeights
	NStreams            int
	Pool                bool
	PoolPart            PartPlan
	PoolW               PartWeights
	PoolRows, PoolWidth int
	PoolRaw             bool
	Entry               bool
	EntryParts          [2]PartPlan
	EntryW              PartWeights
	EntryWidth          int
	Inputs              [4]PartPlan
	Given, WritesX      bool
	InputsW             PartWeights
	PleW                int
	Tables              []TablePlan
	TablePlanes         [2][]float32
	TableWidths         [2]int
	TablesW             PartWeights
	// KV is each history format's conversions, by KVFmt (KVF32's are
	// none): the instance's own, KVFmt, and every one a session may bring.
	KV    [KVFormats][]KVConvPlan
	KVFmt KVFmt
}

// TablePlan is a rotary's table: a row's and a call's rows'.
type TablePlan struct{ One, Many PartPlan }

// KVConvPlan is a history's conversions at one row width: the floats FW
// wide into HW words and back.
type KVConvPlan struct {
	FW, HW        int
	Narrow, Widen PartPlan
}

// TowerPlans is what a tower's parts are composed from: each class's entry
// by sample format, the head and each tap.
type TowerPlans struct {
	Entry []map[uint8]PartPlan
	// Band is each class's band forms, nil where its resize runs whole
	// (graph.Attrs.PicPass); Acc the words each of the bands' accumulators
	// holds and BandOut the words a band form's own value takes.
	Band    []map[uint8]PartPlan
	Acc     int
	BandOut int
	// HAcc is the words each horizontal accumulator a picture's column
	// bands add into holds (graph.PicCols).
	HAcc int
	Head PartPlan
	Taps []PartPlan
	// Chain is a chained tower's blocks' plans in order (jlm.Tower.Chain),
	// each over its block's own grid, its weights the block's tensors.
	Chain []PartPlan
}

// TowerChain is a chained tower's blocks on a device: block i over the rows
// in into out, its weights block li's tensors w as the instance's pager
// holds them; and the head over the last block's rows and the earlier
// blocks' it reads beside them, into out.
type TowerChain interface {
	ChainBlock(i, li int, w Tensors, req, pos []uint32, in, out []float32) error
	ChainHead(req, pos []uint32, out []float32, in ...[]float32) error
}

// PartWeights is where a part's form reads its weights: block Block's
// tensors Of (-1 and the model's own for a part of no block), bound by the
// device where the form reads them.
type PartWeights struct {
	Block int
	Of    Tensors
}

// PlacedParts is an instance's parts on a device; a part the plan states
// none of is nil.
type PlacedParts struct {
	Head    HeadDevice
	Streams StreamsDevice
	Pool    PoolDevice
	Entry   EntryDevice
	Inputs  InputsDevice
	Tables  TablesDevice
	KVConv  [KVFormats]KVConvDevice
}

// LanePlacer readies a device for a placement of n blocks read through l,
// none composed yet: the experts and weights it pages in, the rotaries'
// planes, the experts' count and whether its calls are profiled.
type LanePlacer interface {
	PlaceLanes(n int, l Lease, planes [2][]float32, nExpert int, prof bool)
}

// PartsPlacer composes every part of a plan on a device and makes them the
// parts the instance runs there.
type PartsPlacer interface {
	PlaceParts(*PartsPlan) PlacedParts
}

// TowerPlacer composes a tower's parts on a device, their weights read in
// dense and each tap's in taps.
type TowerPlacer interface {
	PlaceTower(p TowerPlans, dense Tensors, taps []Tensors) (TowerParts, error)
}

// BlockRows says whether block li's rows run on the device as one call, and
// why not.
type BlockRows interface {
	RowsServed(li int) Decline
}

// BankViewer says whether block li's routed banks, at entries e, read as one
// view of its experts' pages on the device, of n blocks.
type BankViewer interface {
	BankViews(li, n int, e [3]*jlm.Entry) bool
}
