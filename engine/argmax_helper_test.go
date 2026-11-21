package engine

import "math"

// argmaxID is the index of l's largest logit.
func argmaxID(l []float32) int32 {
	best, at := float32(math.Inf(-1)), 0
	for i, v := range l {
		if v > best {
			best, at = v, i
		}
	}
	return int32(at)
}
