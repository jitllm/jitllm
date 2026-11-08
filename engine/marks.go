package engine

import (
	"fmt"

	"github.com/jitllm/jitllm/composer"
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
		return fmt.Errorf("model: %s reads out at no marker rows", s.c.Arch)
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
