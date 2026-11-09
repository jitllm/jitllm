package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// A marker head's readout (jlm.Flag2HeadBlocks): a prompt's rows through the
// trunk and the head's blocks as one bidirectional chunk, the head's first
// block adding the question type's row its request names, then the scorer
// over the rows the request names, a score each (model.markGraph).

// MarkScores runs ids as one question of type qtype, its options' markers at
// rows marks, and writes each marker's score into out, len(marks) of them.
func (s *Instance) MarkScores(ids []int32, qtype int, marks []int, out []float32) error {
	n := len(ids)
	switch {
	case !s.c.Marks():
		return fmt.Errorf("model: the container reads out at no marker rows")
	case n == 0 || n > s.maxSeq:
		return fmt.Errorf("model: %d tokens, and this instance holds %d", n, s.maxSeq)
	case len(marks) == 0 || len(out) < len(marks):
		return fmt.Errorf("model: %d markers into %d scores", len(marks), len(out))
	}
	words := composer.RequestRows + n
	if cap(s.req) < words {
		s.req = make([]uint32, words)
	}
	req := s.req[:words]
	req[composer.RequestType], req[composer.RequestCount] = uint32(qtype), uint32(len(marks))
	for r := range n {
		m := marks[min(r, len(marks)-1)]
		if m < 0 || m >= n {
			return fmt.Errorf("model: a marker at row %d of %d", m, n)
		}
		req[composer.RequestRows+r] = uint32(m)
	}
	s.Reset()
	s.setRequest(req)
	defer s.setRequest(nil)
	// The windows' buffer is kept from the last question: a warm one
	// allocates nothing.
	s.mark = embedMode{bidir: s.c.NonCausal, whole: true, total: n, out: out[:len(marks)], req: req, wins: s.mark.wins}
	s.emb = &s.mark
	defer func() { s.emb = nil }()
	_, err := s.Prefill(ids)
	return err
}

// MarkScoresAfter is MarkScores with prefix's spans -- a request's pictures
// -- before ids, its markers at rows of ids. Where the model's media prefix
// reads itself alone (Model.Config.RowWindowed), the prefix's rows are
// windowed to the prefix and the text's to every row.
func (s *Instance) MarkScoresAfter(prefix []model.Span, ids []int32, qtype int, marks []int, out []float32) error {
	p := 0
	for i := range prefix {
		p += prefix[i].N(s.c.NEmbd)
	}
	n := p + len(ids)
	switch {
	case !s.c.Marks():
		return fmt.Errorf("model: the container reads out at no marker rows")
	case len(ids) == 0 || n > s.maxSeq:
		return fmt.Errorf("model: %d rows, and this instance holds %d", n, s.maxSeq)
	case len(marks) == 0 || len(out) < len(marks):
		return fmt.Errorf("model: %d markers into %d scores", len(marks), len(out))
	}
	words := composer.RequestRows + n
	if cap(s.req) < words {
		s.req = make([]uint32, words)
	}
	req := s.req[:words]
	clear(req)
	req[composer.RequestType], req[composer.RequestCount] = uint32(qtype), uint32(len(marks))
	for r := range n {
		m := marks[min(r, len(marks)-1)]
		if m < 0 || m >= len(ids) {
			return fmt.Errorf("model: a marker at row %d of %d", m, len(ids))
		}
		req[composer.RequestRows+r] = uint32(p + m)
	}
	var wins []nn.KeyRun
	if s.c.RowWindowed {
		wins = make([]nn.KeyRun, n)
		for r := range wins {
			wins[r] = nn.KeyRun{Lo: 0, Hi: n}
			if r < p {
				wins[r].Hi = p
			}
		}
	}
	spans := append(append([]model.Span(nil), prefix...), model.Span{Tokens: ids})
	for i := range spans {
		spans[i].Bidir = spans[i].Bidir || s.c.NonCausal
	}
	s.Reset()
	s.setRequest(req)
	defer s.setRequest(nil)
	s.mark = embedMode{bidir: s.c.NonCausal, whole: true, total: n, out: out[:len(marks)], req: req, wins: s.mark.wins, rowWins: wins}
	s.emb = &s.mark
	defer func() { s.emb = nil }()
	_, err := s.PrefillMixed(spans...)
	return err
}
