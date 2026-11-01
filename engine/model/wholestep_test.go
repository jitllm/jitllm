package model

import "github.com/jitllm/jitllm/jit/gpu/tier"

// wholeStep opens a tier with the composer serving every part: what a gate
// whose subject is a feature (batching, sessions, streamed banks, paged
// history) runs on, the composer being the only path.
var wholeStep = tier.WithConfig(func(c *tier.Config) { c.Composed = tier.ComposedStep })
