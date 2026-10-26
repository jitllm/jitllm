package jit

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
	DeclineLatent
	DeclineShared
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
	DeclineNoAxes
	DeclineNoSample
	DeclineNoEmbed
	DeclineEmbedFormat
	DeclineEmbedWeight
	DeclineEmbedToken
	DeclineFeeds
	DeclineRecurrent
	DeclineHistoryWindow
	DeclineKVFormat
	DeclineHeadMajor
	DeclineBatch
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
	NumDeclines
)

var declineNames = [NumDeclines]string{
	DeclineServes:        "served",
	DeclineNoGraph:       "no graph from the front end",
	DeclinePersist:       "inside the persistent block",
	DeclineSkip:          "a fault probe skips blocks",
	DeclineNoPool:        "no paged history on the device",
	DeclinePrefill:       "the history is mid-prefill",
	DeclineStream:        "the history streams",
	DeclineLatent:        "a latent attention's history",
	DeclineShared:        "the history of another block",
	DeclineWindow:        "a graph whose window is not the plan's",
	DeclineHeadShape:     "a head graph other than a norm, its projection and elementwise ops after it",
	DeclineHeadNorm:      "no norm buffer, a norm of another kind than the tier's, or a norm bias it holds none of",
	DeclinePick:          "a head that picks labels only",
	DeclineCaller:        "the caller keeps it on the tier's kernels",
	DeclineNoRoom:        "its buffers do not fit the device's budget",
	DeclineHeadBias:      "a head bias the tier holds no vector for",
	DeclineHeadOther:     "a head graph of another State than the offered one",
	DeclineNoTable:       "no table graph from the front end",
	DeclineTableHeld:     "a table whose planes or positions the tier does not hold",
	DeclineNoAxes:        "a table over several axes with no rotary positions for the row",
	DeclineNoSample:      "no sample graph from the front end",
	DeclineNoEmbed:       "no input rows graph from the front end",
	DeclineEmbedFormat:   "input rows in a format with no container type",
	DeclineEmbedWeight:   "input rows reading a weight other than the token embedding and a position table",
	DeclineEmbedToken:    "a token or position outside its table's rows",
	DeclineFeeds:         "a block graph reading more than the residual",
	DeclineRecurrent:     "a recurrent state the form does not step (a tail or a variant it does not hold, none held, or the CPU's several rows)",
	DeclineHistoryWindow: "a windowed layer's history",
	DeclineKVFormat:      "a history not binary16 (f32 or q8_0)",
	DeclineHeadMajor:     "a head-major history",
	DeclineBatch:         "a step of several rows",
	DeclinePageOut:       "a history page outside the layer's arena",
	DeclineNoWeight:      "a weight or vector the block does not hold",
	DeclineCutWeight:     "a weight cut from a larger one",
	DeclineTable:         "a table the device does not hold",
	DeclineNotFormed:     "the composer did not form the block",
	DeclineUnplaced:      "a block not composed for the device at its placement",
	DeclineExperts:       "routed experts not held whole on the device (streamed, cached, run on another device or in expert pages)",
	DeclineBidir:         "a run of rows attending both ways",
	DeclineSessions:      "a row whose history the CPU's form does not bound (another model's session, or past its keys)",
	DeclineISA:           "a CPU the composer's CPU lowering does not emit for (x86-64 below AVX2, FMA and F16C, or arm64)",
}

func (d Decline) String() string {
	if d >= NumDeclines {
		return fmt.Sprintf("decline %d", d)
	}
	return declineNames[d]
}

// DeclineName is the reason a decline count is kept under, by its index in
// a census array.
func DeclineName(i int) string {
	if i < 0 || i >= int(NumDeclines) {
		return fmt.Sprintf("decline %d", i)
	}
	return declineNames[i]
}
