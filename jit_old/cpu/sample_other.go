//go:build !amd64 && !arm64

package cpu

// The sampler's kernels are owed on this architecture. sample.go has the AVX2
// ones, sample_a64.go the NEON ones and sse_sample.go the legacy-SSE ones; a
// host that is none of those cannot sample at all, because engine/model/sample.go has
// no Go arithmetic left to fall back to (RULE 8) and says so naming the op.
func EmitSampleSegMax(first, idsMem bool) ([]byte, error) {
	return nil, sampleShapeErr("segment maximum", 0)
}

func EmitSampleDraw() ([]byte, error) { return nil, sampleShapeErr("draw", 0) }

func EmitSamplePenalty() ([]byte, error) { return nil, sampleShapeErr("repeat penalty", 0) }
