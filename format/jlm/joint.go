package jlm

// JointHead is a joint head's readout: every question and option of a
// request read in one prompt, its spans' rows averaged, the head's
// Config.DecisionRouting routing blocks and Config.DecisionBlocks joint
// blocks (stored at RoleDecBlock) over them, one logit an option. Its prompt
// is the systemone template, rendered with markers that part a question's
// and an option's span. TypeRows is the type table's row of a choice, a
// score and a noul; SortKeys sorts every object's keys before the template
// renders it, SortChoices a choice's options by key, NoulTrueFirst shows a
// noul's true option first, and IDInstructions shows a question's id where
// its instructions are none or empty. Confidence is how an answer's
// confidence is read.
type JointHead struct {
	TypeRows                                             [3]uint8
	SortKeys, SortChoices, NoulTrueFirst, IDInstructions bool
	Confidence                                           PromptConfidence
}

func encodeJointHead(w *wbuf, j *JointHead) {
	for _, r := range j.TypeRows {
		w.u8(r)
	}
	w.bool(j.SortKeys)
	w.bool(j.SortChoices)
	w.bool(j.NoulTrueFirst)
	w.bool(j.IDInstructions)
	w.u8(uint8(j.Confidence))
}

func decodeJointHead(r *rbuf) *JointHead {
	j := &JointHead{}
	for i := range j.TypeRows {
		j.TypeRows[i] = r.u8()
	}
	j.SortKeys, j.SortChoices, j.NoulTrueFirst, j.IDInstructions = r.bool(), r.bool(), r.bool(), r.bool()
	j.Confidence = PromptConfidence(r.u8())
	return j
}
