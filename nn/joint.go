package nn

import "github.com/jitllm/jitllm/format/jlm"

// A joint head's readout (jlm.JointHead) reads its request in the slots the
// format states (jlm.JointQuestionSpans and on) after the marker rows
// (composer.RequestRows).
const (
	JointQuestionSpans   = jlm.JointQuestionSpans
	JointOptionSpans     = jlm.JointOptionSpans
	JointOptionQuestion  = jlm.JointOptionQuestion
	JointQuestionTypes   = jlm.JointQuestionTypes
	JointQuestionOptions = jlm.JointQuestionOptions
	JointFields          = jlm.JointFields
	JointTokens          = jlm.JointTokens
	JointSlots           = jlm.JointSlots
)
