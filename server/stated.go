package server

import (
	"math/rand/v2"

	"github.com/jitllm/jitllm/engine"
)

// sessionSampler is a session's own draw: the one it was opened with, or the
// one lm's container states over its maxSeq tokens, seeded apart from every
// other session's.
func sessionSampler(own *engine.Sampler, lm *LoadedModel, maxSeq int) engine.Sampler {
	if own != nil {
		return *own
	}
	s := *engine.StatedSampler(lm.m.Cfg.Generation, maxSeq)
	s.Seed = rand.Int64()
	return s
}
