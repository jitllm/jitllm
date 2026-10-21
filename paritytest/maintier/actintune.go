package maintier

import (
	"fmt"
	"os"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	kernels "github.com/jitllm/jitllm/paritytest/mainkernels"
)

// tuneActIn decides, on this card, which operand-prologue sites run: for each
// site key (a shape: the q/k/v launch, the attention output, down) it times
// the two-launch form against the prologue over a sequence of the placed
// blocks that share it, the way a token runs them, and keeps the prologue
// only where it is faster. The verdict is cached per binary, device and
// shape, as the split tuner's is. A caller that forced sites (ActInSites)
// skips it. Called from retuneDecode, holding g.mu, outside any Session.
func (g *devTier) tuneActIn() {
	if g.ActInSites != 0 || g.bs == nil {
		return
	}
	g.actInTrial = true
	for _, l := range g.layers {
		if l != nil && l.ok {
			g.fuseActIn(l)
		}
	}
	g.actInTrial = false
	type cand struct {
		site int
		ls   []*layer
	}
	by := map[string]*cand{}
	var keys []string
	for _, l := range g.layers {
		if l == nil || !l.ok || (l.pg != nil && !l.pg.in) {
			continue
		}
		for i, key := range l.actInKeys[:3] {
			if key == "" {
				continue
			}
			c := by[key]
			if c == nil {
				c = &cand{site: [3]int{actInQKV, actInO, actInDown}[i]}
				by[key], keys = c, append(keys, key)
			}
			if len(c.ls) < 16 {
				c.ls = append(c.ls, l)
			}
		}
	}
	if g.actInWin == nil {
		g.actInWin = map[string]bool{}
	}
	for _, key := range keys {
		c := by[key]
		ck := g.tuneKey("actin-" + key)
		if v, ok := lookupCache(g.kb.cacheDir, "gpuactin", ck); ok && g.kb.tune != TuneForce {
			g.actInWin[key] = v == "1"
			continue
		}
		r, ok := g.timeActIn(c.site, c.ls)
		win := ok && r < actInMargin
		g.actInWin[key] = win
		if backend.Verbose() {
			fmt.Fprintf(os.Stderr, "tier: operand prologue %s: %.4f of the two launches over %d blocks (resolved %v): %v\n",
				key, r, len(c.ls), ok, win)
		}
		if ok && ck != "" {
			v := "0"
			if win {
				v = "1"
			}
			saveCache(g.kb.cacheDir, "gpuactin", ck, v)
		}
	}
	// The verdicts of this tune, not a sum over retunes: a retune after the
	// seam grows decides the same keys again.
	g.ActInTimed, g.ActInKept = len(keys), 0
	for _, key := range keys {
		if g.actInWin[key] {
			g.ActInKept++
		}
	}
	for _, l := range g.layers {
		if l != nil && l.ok {
			g.fuseActIn(l)
		}
	}
	g.dropGraph()
}

// actInMargin is how much faster than the two launches the prologue must
// time to be kept: the trial resolves about 1%, and a tie keeps two launches.
const actInMargin = 0.99

// timeActIn times one site's two arms over the blocks ls (timeRecipes, the
// two launches the incumbent) and returns the median of the per-round ratios
// prologue/two-launch time. ok is false where the trial could not launch or
// did not resolve (IQR over splitTuneIQR).
func (g *devTier) timeActIn(site int, ls []*layer) (float64, bool) {
	bs := g.bs
	ax := []backend.Buf{bs.a, bs.ax}
	arm := func(s backend.Session, in bool) error {
		for _, l := range ls {
			switch site {
			case actInQKV:
				if !in {
					if bs.rmsQ == nil {
						return errNoBatch
					}
					if err := s.Launch(bs.rmsQ, 1, kernels.RMSNormGroup, bs.x, l.nAttn, bs.h, bs.a, bs.ax); err != nil {
						return err
					}
				}
				qkv := [3]struct {
					m   mv
					r   *resident
					out backend.Buf
				}{{l.mvq, l.wq, bs.q}, {l.mvk, l.wk, bs.k}, {l.mvv, l.wv, bs.v}}
				if l.qkv != nil {
					k, args := l.qkv, []backend.Buf{bs.a, bs.ax}
					if in {
						k, args = l.qkvIn, []backend.Buf{bs.x, l.nAttn}
					}
					for i, sg := range qkv {
						args = append(args, sg.r.qs, sg.r.d, sg.r.sc, sg.out)
						if l.qkvBias[i] {
							args = append(args, sg.m.bias)
						}
					}
					if err := s.Launch(k, l.qkvBlocks, l.qkvW, args...); err != nil {
						return err
					}
					continue
				}
				for _, p := range qkv {
					k, pre := p.m.kern, ax
					if in {
						k, pre = p.m.kernIn, []backend.Buf{bs.x, l.nAttn}
					}
					var extra []backend.Buf
					if p.m.bias != nil {
						extra = append(extra, p.m.bias)
					}
					if err := g.mvLaunch(s, p.m, k, p.r, pre, p.out, extra...); err != nil {
						return err
					}
				}
			case actInO, actInDown:
				m, r, q, src, dst, resid := l.mvo, l.wo, bs.quantQ, bs.xb, bs.x2, bs.x
				if site == actInDown {
					m, r, q, src, dst, resid = l.mvd, l.down, bs.quantF, bs.act, bs.x, bs.x2
				}
				if in {
					if err := g.mvLaunch(s, m, m.resIn, r, []backend.Buf{src}, dst, resid); err != nil {
						return err
					}
					continue
				}
				if q == nil {
					return errNoBatch
				}
				if err := s.Launch(q, (kernels.QuantizeThreads(m.k/32)+127)/128, 128, src, bs.a, bs.ax); err != nil {
					return err
				}
				if err := g.mvLaunch(s, m, m.res, r, ax, dst, resid); err != nil {
					return err
				}
			}
		}
		return nil
	}
	med, ok := g.timeRecipes([]func(backend.Session) error{
		func(s backend.Session) error { return arm(s, false) },
		func(s backend.Session) error { return arm(s, true) },
	})
	return med[1], ok[1]
}
