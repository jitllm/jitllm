package nn

// A joint head's readout (jlm.JointHead) reads its request in slots after
// the marker rows (composer.RequestRows), each the call's live rows long:
// slot s, row r is word RequestRows + s*rows + r. A span is two slots, its
// first row then one past its last.
const (
	// JointQuestionSpans is each question's instructions' span.
	JointQuestionSpans = 1
	// JointOptionSpans is each option's span, the questions' options in
	// order.
	JointOptionSpans = 3
	// JointOptionQuestion is each option's question.
	JointOptionQuestion = 5
	// JointQuestionTypes is each question's row of the type table.
	JointQuestionTypes = 6
	// JointQuestionOptions is each question's options, a span of the option
	// rows.
	JointQuestionOptions = 7
	// JointFields is every question's span of the question rows: from 0 to
	// the question count.
	JointFields = 9
	// JointTokens is the prompt's token ids, a row's.
	JointTokens = 11
	// JointSlots is how many slots a joint request holds.
	JointSlots = 12
)
