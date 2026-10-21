package cpu

// RowMode is which of the row-statistic kernels the layer norm's passes
// build: the layer norm itself, Gemma 3n's gaussian top-k (EmitGaussTopK),
// AltUp's magnitude match (EmitMagMatch) or a row standardised over its
// unbiased deviation (RowStd). They share the mean and variance passes and
// differ in what the statistics become.
type RowMode int

const (
	RowLayerNorm RowMode = iota
	RowGauss
	RowMag
	// RowStd is y = (x - mean) / (sqrt(sum((x-mean)^2) * Scr[3]) + eps):
	// NeMo's per-feature normalisation (d1-omni's audio front end), Scr[3]
	// 1/(n-1) (0 for one element, so its deviation is 0). No weight or bias.
	RowStd
)
