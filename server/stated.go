package server

import (
	"fmt"
	"math/rand/v2"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/model"
)

// sessionSampler is a session's own draw: the one it was opened with, or the
// one lm's container states over its maxSeq tokens, seeded apart from every
// other session's. A container that states none gives no draw, and the
// error a generate asking for none is refused with.
func sessionSampler(own *engine.Sampler, lm *LoadedModel, maxSeq int, name string) (engine.Sampler, error) {
	if own != nil {
		return *own, nil
	}
	if lm.m.Cfg.Generation == nil {
		return engine.Sampler{}, fmt.Errorf("%w: %w", ErrInvalid, &model.NoStatedDrawError{Name: lm.called(name)})
	}
	s := *engine.StatedSampler(*lm.m.Cfg.Generation, maxSeq)
	s.Seed = rand.Int64()
	return s, nil
}

// draw is the sampler a generate on s runs: the request's own, else the
// session's, refused where the session has none to give (sessionSampler).
func (s *Session) draw(own *engine.Sampler) (engine.Sampler, error) {
	if own != nil {
		return *own, nil
	}
	return s.sampling, s.undrawn
}
