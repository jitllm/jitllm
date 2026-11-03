package composer

import "fmt"

// Decline is why a device does not run what the composer forms for a part
// of a step -- a block, a head, a table, a sample, the input rows -- the one
// set of reasons every device's census counts under, a card's and the
// CPU's alike. DeclineServes is a part served.
type Decline uint8

const (
	DeclineServes Decline = iota
	DeclineNoGraph
	DeclinePersist
	DeclineSkip
	DeclineNoPool
	DeclinePrefill
	DeclineStream
	DeclineWindow
	DeclineHeadShape
	DeclineHeadNorm
	DeclinePick
	DeclineCaller
	DeclineNoRoom
	DeclineHeadBias
	DeclineHeadOther
	DeclineNoTable
	DeclineTableHeld
	DeclineNoSample
	DeclineNoEmbed
	DeclineEmbedFormat
	DeclineEmbedWeight
	DeclineEmbedToken
	DeclineFeeds
	DeclineRecurrent
	DeclineKVFormat
	DeclinePageOut
	DeclineNoWeight
	DeclineCutWeight
	DeclineTable
	DeclineNotFormed
	DeclineISA
	DeclineUnplaced
	DeclineExperts
	DeclineBidir
	DeclineSessions
	DeclineExpertsHost
	DeclineExpertsRows
	DeclineEmbedStreams
	NumDeclines
)

func (d Decline) String() string {
	switch d {
	case DeclineServes:
		return "served"
	case DeclineNoGraph:
		return "no graph from the front end"
	case DeclinePersist:
		return "inside the persistent block"
	case DeclineSkip:
		return "a fault probe skips blocks"
	case DeclineNoPool:
		return "no paged history on the device"
	case DeclinePrefill:
		return "the history is mid-prefill"
	case DeclineStream:
		return "the history streams"
	case DeclineWindow:
		return "a graph whose window is not the plan's"
	case DeclineHeadShape:
		return "a head graph other than a norm, its projection and elementwise ops after it"
	case DeclineHeadNorm:
		return "no norm buffer, a norm of another kind than the tier's, or a norm bias it holds none of"
	case DeclinePick:
		return "a head that picks labels only"
	case DeclineCaller:
		return "the caller keeps it on the tier's kernels"
	case DeclineNoRoom:
		return "its buffers do not fit the device's budget"
	case DeclineHeadBias:
		return "a head bias the tier holds no vector for"
	case DeclineHeadOther:
		return "a head graph of another State than the offered one"
	case DeclineNoTable:
		return "no table graph from the front end"
	case DeclineTableHeld:
		return "a table whose planes or positions the tier does not hold"
	case DeclineNoSample:
		return "no sample graph from the front end"
	case DeclineNoEmbed:
		return "no input rows graph from the front end"
	case DeclineEmbedFormat:
		return "input rows in a format with no container type"
	case DeclineEmbedWeight:
		return "input rows reading a weight other than the token embedding and a position table"
	case DeclineEmbedToken:
		return "a token or position outside its table's rows"
	case DeclineFeeds:
		return "a block graph reading more than the residual and per-layer inputs the device holds"
	case DeclineRecurrent:
		return "a recurrent state the form does not step (a tail or a variant it does not hold, none held, or the CPU's several rows)"
	case DeclineKVFormat:
		return "a history neither binary16 nor f32 (q8_0)"
	case DeclinePageOut:
		return "a history page outside the layer's arena"
	case DeclineNoWeight:
		return "a weight or vector the block does not hold"
	case DeclineCutWeight:
		return "a weight cut from a larger one"
	case DeclineTable:
		return "a table the device does not hold"
	case DeclineNotFormed:
		return "the composer did not form the block"
	case DeclineUnplaced:
		return "a block not composed for the device at its placement"
	case DeclineExperts:
		return "routed experts not held whole on the device (streamed, cached, run on another device or in expert pages)"
	case DeclineBidir:
		return "a run of rows attending both ways"
	case DeclineSessions:
		return "a row whose history the CPU's form does not bound (another model's session, or past its keys)"
	case DeclineExpertsHost:
		return "routed experts run on another device (a hybrid block's, on the host)"
	case DeclineExpertsRows:
		return "routed experts in pages at a step of several rows"
	case DeclineEmbedStreams:
		return "input rows over every residual stream into the rows of one"
	case DeclineISA:
		return "a CPU the composer's CPU lowering does not emit for (x86-64 below SSE4.1 and SSSE3)"
	}
	return fmt.Sprintf("decline %d", d)
}

// DeclineName is the reason a decline count is kept under, by its index in
// a census array.
func DeclineName(i int) string {
	return Decline(i).String()
}
