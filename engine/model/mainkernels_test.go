package model

import "github.com/jitllm/jitllm/jit/gpu/tier"

// mainKernels opens a tier on main's own kernels: the gates whose subject is
// one of them (the q/k/v grid, paired segments, the persistent decode, paged
// attention) while main's tier runs the blocks the composer does not.
var mainKernels = tier.WithConfig(func(c *tier.Config) { c.Composed = 0 })
