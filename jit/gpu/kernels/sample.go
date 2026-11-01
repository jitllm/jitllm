package kernels

// SampleArgsWords is the sampler's argument block's length for a history of
// h tokens: the count, the penalty's bits, and the ids.
func SampleArgsWords(h int) int { return 2 + h }
