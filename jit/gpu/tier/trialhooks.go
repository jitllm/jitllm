package tier

import (
	"github.com/jitllm/jitllm/jit/gpu/backend"
)

// trialHooks are the tuner's seams a gate replaces to count or fake a trial,
// per tier: a nil field runs the real one. Set only by the unexported
// options below, from this package's own tests.
type trialHooks struct {
	bg       func(g *devTier, kern, red backend.Kernel, r *resident, d, out, part backend.Buf, rows, split int, grp bool) (float64, bool)
	split    func(g *devTier, kern, red backend.Kernel, r *resident, d, out backend.Buf, rows, split int, grp bool) (float64, bool)
	binaryID func() string
}

func (g *devTier) bgTrial(kern, red backend.Kernel, r *resident, d, out, part backend.Buf, rows, split int, grp bool) (float64, bool) {
	if f := g.trial.bg; f != nil {
		return f(g, kern, red, r, d, out, part, rows, split, grp)
	}
	return g.timeSplitRound(kern, red, r, d, out, part, rows, split, grp)
}

func (g *devTier) splitTrial(kern, red backend.Kernel, r *resident, d, out backend.Buf, rows, split int, grp bool) (float64, bool) {
	if f := g.trial.split; f != nil {
		return f(g, kern, red, r, d, out, rows, split, grp)
	}
	return g.timeSplitMode(kern, red, r, d, out, rows, split, grp)
}

func (g *devTier) binaryID() string {
	if f := g.trial.binaryID; f != nil {
		return f()
	}
	return binaryIDOnce()
}

func withTrialHooks(h trialHooks) Option {
	return WithConfig(func(c *Config) { c.trial = h })
}
