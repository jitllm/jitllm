package engine

import "github.com/jitllm/jitllm/format/jlm"

// StatedSampler is the draw a model's container states it generates with
// when a caller asks nothing (jlm.Generation), over a context of nctx
// tokens: greedy where the reference does not sample, and its repetition
// penalty read over the whole sequence, as the reference's reads every token
// before the draw.
func StatedSampler(g jlm.Generation, nctx int) *Sampler {
	s := &Sampler{TopK: g.TopK, TopP: g.TopP, MinP: g.MinP, TypicalP: g.TypicalP, RepeatPen: g.RepetitionPenalty}
	if g.DoSample {
		s.Temp = g.Temperature
	}
	if s.RepeatPen != 1 {
		s.RepeatLastN = nctx
	}
	return s
}
