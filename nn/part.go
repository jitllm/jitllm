package nn

// A part of a model a device serves an instance from its composed forms: a
// whole block, the head, a step's rotary table, a picture's rows... Every
// device counts what it serves by part, the CPU as each card.

type Part uint8

const (
	PartHead Part = iota + 1
	// PartBlock is a whole block, its graph the model's front end's.
	PartBlock
	// PartPrompt is a whole block at a prompt chunk's rows: one sequence's
	// positions, each row its own descriptor.
	PartPrompt
	// PartTable is a step's rotary table (LayerPlan.Tables), written where
	// every block reads it.
	PartTable
	// PartSample is a step's greedy token over the head's logits
	// (Head.Sample), written where the device's argmax writes it.
	PartSample
	// PartEmbed is a decode step's input row (Head.Embed), gathered from
	// the tables the device holds; PartEmbedRows a prompt chunk's input
	// rows, its rows at the chunk's.
	PartEmbed
	PartEmbedRows
	// PartCopy is a block's output, written apart from x where its form
	// reads x in the kernel writing it, copied over x (graph.Copy).
	PartCopy
	// PartTower is a non-causal call's block whole: a picture's rows, one
	// run (the card's towerserve.go).
	PartTower
	// PartRows is a picture's rows apart from its blocks: a tower's entry
	// or projector (the card's picrows.go).
	PartRows
	// PartCollapse is the row a head reads from every residual stream
	// (the card's collapse.go).
	PartCollapse
	// PartConv is a convolutional tower's block whole, at its input grid's
	// rows.
	PartConv
	// PartHalves is a binary16 history page's conversion, its floats
	// narrowed or its halves widened, as a migration moves it (the card's
	// kvhalves.go).
	PartHalves
	// PartMTP is a prediction block's entry, PartStreams the residual
	// streams' expansion and collapse, PartPool an embedding model's
	// readout.
	PartMTP
	PartStreams
	PartPool
	NumParts
)

func (k Part) String() string {
	switch k {
	case PartHead:
		return "head"
	case PartBlock:
		return "block"
	case PartPrompt:
		return "prompt block"
	case PartTable:
		return "table"
	case PartSample:
		return "sample"
	case PartEmbed:
		return "input rows"
	case PartEmbedRows:
		return "prompt input rows"
	case PartCopy:
		return "copy"
	case PartTower:
		return "tower block"
	case PartRows:
		return "picture rows"
	case PartCollapse:
		return "streams' collapse"
	case PartConv:
		return "convolutional block"
	case PartHalves:
		return "history page's halves"
	case PartMTP:
		return "prediction entry"
	case PartStreams:
		return "streams"
	case PartPool:
		return "readout"
	}
	return ""
}

// Parts names the parts of a model the composer serves, each counted apart
// (Steps.Of).
type Parts uint32

const (
	ComposedHead Parts = 1 << PartHead
	// ComposedBlock serves a block whole from its graph (LayerPlan.Graph).
	ComposedBlock Parts = 1 << PartBlock
	// ComposedPrompt serves a prompt chunk's blocks whole at the chunk's
	// rows: a form a width, as the tier keeps a scratch a width.
	ComposedPrompt Parts = 1 << PartPrompt
	// ComposedTable serves a call's rotary tables from their graphs
	// (LayerPlan.Tables) at its rows' rotary positions, a step's, a
	// ragged step's or a prompt chunk's.
	ComposedTable Parts = 1 << PartTable
	// ComposedSample serves a one-row step's greedy token from its graph
	// (Head.Sample).
	ComposedSample Parts = 1 << PartSample
	// ComposedEmbed serves the input rows from their graph (Head.Embed),
	// a decode step's and a prompt chunk's, gathered on the device the
	// caller promises them to (EmbedDevice).
	ComposedEmbed Parts = 1 << PartEmbed
	// PromptEmbed counts what ComposedEmbed serves of a prompt chunk's input
	// rows (Stats.ServedPart), apart from a decode step's.
	PromptEmbed Parts = 1 << PartEmbedRows
	// CopiedOut selects in Stats.ServedPart the copies of block outputs
	// written apart from x (servedCopy): a block the composer serves asks
	// for one where it needs it.
	CopiedOut Parts = 1 << PartCopy
	// ComposedTower serves a non-causal call's blocks whole from their
	// graphs: a vision tower's, over a picture's rows (the card's towerserve.go).
	ComposedTower Parts = 1 << PartTower
	// ComposedRows serves a picture's rows apart from its blocks from their
	// graphs: a tower's entry and projector (RowsPart, picrows.go).
	ComposedRows Parts = 1 << PartRows
	// CollapsedHead counts the collapse of every stream a head reads, which
	// the composer serves where it serves the head (ComposedHead).
	CollapsedHead Parts = 1 << PartCollapse
	// ComposedConv serves a convolutional tower's blocks whole from their
	// graphs.
	ComposedConv Parts = 1 << PartConv
	// ConvertedHalves selects in Stats.ServedPart a binary16 history page's
	// conversions a migration ran (the card's kvhalves.go).
	ConvertedHalves Parts = 1 << PartHalves
	// ComposedMTP serves a prediction block's entry, ComposedStreams the
	// residual streams' expansion and collapse, ComposedPool an embedding
	// model's readout.
	ComposedMTP     Parts = 1 << PartMTP
	ComposedStreams Parts = 1 << PartStreams
	ComposedPool    Parts = 1 << PartPool
	// EveryPart is every part the composer serves, for a count over all of
	// them.
	EveryPart Parts = 1<<NumParts - 2
	// ComposedStep is every part of a step: its input rows, tables, blocks
	// (a prompt chunk's and a picture's among them), head and token.
	ComposedStep = ComposedEmbed | ComposedTable | ComposedBlock | ComposedPrompt | ComposedTower | ComposedRows | ComposedHead | ComposedSample
)

// Steps is what a device served an instance, part by part: the steps the
// composer's forms ran there, the steps it declined and why
// (DeclineName), and the steps of several rows its recurrent blocks
// ran whole on their rows' form or a row at a time. The CPU counts its input
// rows, streams and prediction entry by the row.
type Steps struct {
	Served   [NumParts]int
	Declined [NumParts][NumDeclines]int
	Whole    int
	Stepped  int
}

// Serve counts a step of part p served.
func (s *Steps) Serve(p Part) { s.Served[p]++ }

// Decline counts a step of part p declined for why.
func (s *Steps) Decline(p Part, why Decline) { s.Declined[p][why]++ }

// Of is how many steps of the parts of k were served, and how many were
// declined, by reason.
func (s Steps) Of(k Parts) (served int, declined [NumDeclines]int) {
	for p := range NumParts {
		if k&(1<<p) == 0 {
			continue
		}
		served += s.Served[p]
		for why, n := range s.Declined[p] {
			declined[why] += n
		}
	}
	return served, declined
}

// Add adds o's counts to s.
func (s *Steps) Add(o *Steps) {
	for p := range NumParts {
		s.Served[p] += o.Served[p]
		for why, n := range o.Declined[p] {
			s.Declined[p][why] += n
		}
	}
	s.Whole += o.Whole
	s.Stepped += o.Stepped
}

// Account is what a device kept of what it served an instance and formed
// for it.
type Account struct {
	Steps Steps
	// Forms is how many times each block was formed there: once at its
	// placement, and once more for each relocation back.
	Forms []int
	// Asks is how many forms were asked of the device's form cache, a cached
	// one among them: what an instance's creation, relocation or tuning
	// moves, and a prefill or a generated token must not.
	Asks int64
	// FormErr is the first part the device did not form, nil for none.
	FormErr error
}

// Ledger is a device that keeps an Account, by the name of each device it
// is: false where none of them is named dev.
type Ledger interface {
	AccountOn(dev string) (Account, bool)
}

// KVConvDevice is a device converting a history's rows: narrowing the
// floats f of head rows hd wide into the history's bytes h, or widening h
// into f, reporting whether it served them.
type KVConvDevice interface {
	Convert(narrow bool, hd int, f []float32, h []byte) bool
}

// DrawArgs is a draw's sampler: its top-k (0 for none), 1/T, top-p and
// min-p, and its penalties with the window they count (nil where they are
// off).
type DrawArgs struct {
	TopK                int
	InvTemp, TopP, MinP float32
	Penalty             Penalties
	Window              []float32
}

// Penalties are a draw's penalties over the tokens its window holds, as
// llama.cpp applies them: the repeat penalty once a token (1 is none), then
// presence once a token and frequency once a count subtracted (0 is none).
type Penalties [3]float32

// NoPenalties leaves every logit as it is.
func NoPenalties() Penalties { return Penalties{1, 0, 0} }

// Off reports whether p leaves every logit as it is, whatever the window
// holds.
func (p Penalties) Off() bool { return p == NoPenalties() }

// HeadDevice is a device running the head and the samplers: the head over a
// row (Head) or a call's rows (HeadRows), the output norm alone, a label
// pick, the greedy token, a draw at a uniform, a draw's distribution, a
// constrained draw's mask, the log-softmax with its most likely tokens and a
// label readout's distribution over a question's options (Decide:
// graph.DecisionScores' inputs and outputs, in their order).
type HeadDevice interface {
	Head(x, logits, hidden []float32) Decline
	HeadRows(x, logits, hidden []float32, n int) Decline
	OutNorm(dst, x []float32, n int) Decline
	Pick(x []float32, ids []int32, vals []float32) Decline
	Greedy(logits []float32) (int32, bool)
	Draw(logits []float32, a DrawArgs, u float32) (int32, bool)
	Dist(logits []float32, a DrawArgs, pall []float32) (dense bool, ids []uint32, vals []float32, ok bool)
	Mask(dst, logits, mask []float32) bool
	Logprobs(logits []float32) (row []float32, ids []uint32, ok bool)
	Decide(vals, keep, params, prior, probs, stats []float32) bool
}

// EntryDevice is a device writing a prediction block's entry: `rows` rows
// of z, the rows the block reads, from the tokens toks and the trunk's
// hidden rows x, reporting why it did not serve them.
type EntryDevice interface {
	Entry(z, x []float32, toks []int32, rows int) Decline
}

// TablesDevice is a device writing a call's rotary tables: row i at the
// cache position desc[2i], its sequence's marks from desc[2i+1] in marks,
// into cs and csSWA; SetGrid makes them a grid table's over a picture, its
// request req and its rows pos.
type TablesDevice interface {
	Tables(desc, marks []uint32, cs, csSWA []float32) error
	SetGrid(req, pos []uint32)
}

// InputsDevice is a device writing a call's input rows: the vocabulary rows
// of toks at positions pos into x and, where the model has them, their
// per-layer inputs into ple (Embed), or an embedding span's given rows with
// what the model adds to them (Given), each reporting why it did not serve
// them.
type InputsDevice interface {
	Embed(x, ple []float32, toks []int32, pos []int) Decline
	Given(x, ple, rows []float32, toks []int32, pos []int) Decline
}

// StreamsDevice is a device running a model's residual streams over a
// call's rows of the stream-major x: the expansion of stream 0 into the
// others and the collapse of every stream into the row the head reads, each
// reporting why it did not serve them.
type StreamsDevice interface {
	Expand(x []float32, rows int) Decline
	Collapse(x []float32, rows int) Decline
}

// PoolDevice is a device reading an embedding model's readout: the pooled,
// unit-length row of a call's n rows of x into out -- with raw, the row
// before the dense heads -- reporting whether it served it; or a marker
// head's, the scores of the rows the request words req name.
type PoolDevice interface {
	Pool(x []float32, n int, req []uint32, out []float32, raw bool) bool
}

// RequestDevice is a device whose blocks read the step's request words
// (composer.TableRequest): a marker head's question type. req holds for
// every call that follows until the next SetRequest.
type RequestDevice interface {
	SetRequest(req []uint32)
}
