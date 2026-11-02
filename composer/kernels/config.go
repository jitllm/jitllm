package kernels

// Center is the weight centering a matvec is emitted with. Zero is the shipping
// default. It travels in MatVecShape, so every caller (a tier, a benchmark)
// chooses its own; tier fills it from WithCentering.
type Center struct {
	// NoCenter disables the weight centering, so a row can be measured with
	// and without it in one session.
	NoCenter bool
	// MinTok is the token-columns-per-thread at or above which centering is
	// emitted; 0 takes the default 8. It is a profitability threshold, not a
	// correctness one, derived from a static instruction count on sm_86 and
	// unvalidated on Metal and Vulkan. MinTok=1 forces centering everywhere,
	// decode included, so the exclusion can be measured per backend.
	MinTok int
}

func (c Center) minTok() int {
	if c.MinTok >= 1 {
		return c.MinTok
	}
	return 8
}
