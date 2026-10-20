package maintier

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"time"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	kernels "github.com/jitllm/jitllm/paritytest/mainkernels"
)

// A site is a place in a block's graph where more than one generated form
// computes the same values, and a recipe is one of those forms: the kernels
// it launches and over which buffers. The composer's state is the chosen
// recipe per site and per shape key, timed on this card (docs/design/composer.md).
// The operand prologue's three sites (tuneActIn) and the dense gated FFN's
// gate/up site (tuneFFN) are decided by the one trial here, timeRecipes.

// FFNForm is a recipe for a dense gated FFN's gate and up (with the norm
// before them, or down after them, where the recipe takes those in).
type FFNForm int

const (
	// FFNTuned takes the per-shape verdict (tuneFFN), Gate2 where none.
	FFNTuned FFNForm = iota
	// FFNTwo: the gate's matvec, then up's gated matvec writing act(gate)*up.
	FFNTwo
	// FFNGate2: gate and up summed in one launch (kernels.MatVecShape.Gate2).
	FFNGate2
	// FFNGate2In: Gate2 normalising and quantizing its own operand, so the
	// FFN norm's launch goes too (behind a residual the attention output's
	// launch already added).
	FFNGate2In
	// FFNGU: gate and up as one segment launch, down forming act(gate)*up
	// in its prologue (Config.GateUpFuse's form).
	FFNGU
)

var ffnFormNames = [...]string{"tuned", "two", "gate2", "gate2in", "gu"}

func (f FFNForm) String() string {
	if f >= 0 && int(f) < len(ffnFormNames) {
		return ffnFormNames[f]
	}
	return strconv.Itoa(int(f))
}

// ParseFFNForm reads an FFNForm by its name ("" is FFNTuned).
func ParseFFNForm(s string) (FFNForm, error) {
	if s == "" {
		return FFNTuned, nil
	}
	for i, n := range ffnFormNames {
		if n == s {
			return FFNForm(i), nil
		}
	}
	return 0, fmt.Errorf("tier: FFN recipe %q is not one of %v", s, ffnFormNames)
}

// ffnFormOf is the recipe block l's FFN asks for: the caller's, else the
// block's verdict, else Gate2. The launch still checks that the block can
// express it, and falls down the chain where it cannot.
func (g *devTier) ffnFormOf(l *layer) FFNForm {
	if g.FFNForm != FFNTuned {
		return g.FFNForm
	}
	if g.GateUp2Force {
		return FFNGate2
	}
	if l.ffnForm != FFNTuned {
		return l.ffnForm
	}
	return FFNGate2
}

// recipeMargin is how much faster than the incumbent a challenger must time
// to replace it: the trial resolves about 1%, and a tie keeps the incumbent.
const recipeMargin = 0.99

// recipeRounds is the trial's paired rounds per recipe, after one discarded
// for first touch.
const recipeRounds = 20

// timeRecipes times arms (arms[0] the incumbent) over one Session each,
// interleaved: every round runs every arm once, the order rotated by the
// round, and an arm's sample is three repetitions then a Sync. It returns,
// per challenger, the median of its per-round ratios to the incumbent, and
// whether that median resolved (IQR/median within splitTuneIQR). An arm that
// fails to launch resolves as false.
func (g *devTier) timeRecipes(arms []func(s backend.Session) error) (med []float64, ok []bool) {
	tuneTrials.Add(1)
	n := len(arms)
	med, ok = make([]float64, n), make([]bool, n)
	failed := make([]bool, n)
	timed := func(i int) float64 {
		const reps = 3
		t0 := time.Now()
		var err error
		g.dev.Session(func(s backend.Session) {
			backend.MarkTrial(s) // a profile tells a tuner's trial from the work
			for r := 0; r < reps && err == nil; r++ {
				err = arms[i](s)
			}
			if err == nil {
				err = s.Sync()
			}
		})
		if err != nil {
			failed[i] = true
		}
		return time.Since(t0).Seconds()
	}
	ratios := make([][]float64, n)
	t := make([]float64, n)
	for round := 0; round <= recipeRounds; round++ {
		for j := 0; j < n; j++ {
			i := (j + round) % n
			t[i] = timed(i)
		}
		if failed[0] {
			return med, ok
		}
		if round == 0 || t[0] <= 0 {
			continue // the first round pays first touch
		}
		for i := 1; i < n; i++ {
			ratios[i] = append(ratios[i], t[i]/t[0])
		}
	}
	for i := 1; i < n; i++ {
		r := ratios[i]
		if failed[i] || len(r) == 0 {
			continue
		}
		sort.Float64s(r)
		m := r[len(r)/2]
		iqr := r[len(r)*3/4] - r[len(r)/4]
		med[i], ok[i] = m, m > 0 && iqr/m <= splitTuneIQR
	}
	return med, ok
}

// mvLaunch launches matvec m's kernel k over weight r: pre is the operand (a
// and ax, or a prologue's float row and its second input), extra the bias,
// residual or gate (Gate2's gate weight, three buffers). A split writing partials reduces after. It is the
// launch encodeLayer's mvrun, mvres and mvIn make, for the tuners' arms.
func (g *devTier) mvLaunch(s backend.Session, m mv, k backend.Kernel, r *resident, pre []backend.Buf, out backend.Buf, extra ...backend.Buf) error {
	w := 128
	if m.group {
		w = kernels.GroupSplitWidth(m.split)
	} else if m.width > 0 && k == m.kern {
		w = m.width
	}
	n := m.rows * max(m.split, 1)
	if m.rowt > 1 && k != m.gated && k != m.gatedIn && k != m.gate2 && k != m.gate2In {
		n /= m.rowt
	}
	dst := out
	if m.split > 1 && !m.group {
		dst = g.partBuf
	}
	var arr [12]backend.Buf
	args := append(append(arr[:0], r.qs, r.d, r.sc), pre...)
	args = append(append(args, dst), extra...)
	if err := s.Launch(k, (n+w-1)/w, w, args...); err != nil {
		return err
	}
	if m.split > 1 && !m.group {
		return s.Launch(m.red, (m.rows+127)/128, 128, g.partBuf, out)
	}
	return nil
}

// ffnKey is block l's FFN site key: what decides the recipes' cost (every
// matvec's format, width, rows, split and reduction form, the activation and
// the norm) and nothing else; "" where the block offers fewer than two
// recipes or its FFN is not the plain dense gated one the arms run.
func (g *devTier) ffnKey(l *layer) string {
	p := &g.bs.p
	if l.router != nil || l.mla || l.linear || l.nonCausal || l.sparse || l.clampB != nil || l.nSubFFN != nil ||
		l.nPostFFN != nil || l.nFFNB != nil || l.nResid != nil || l.k3 != nil || l.q4e != nil || l.ds4 != nil ||
		l.altRouter != nil || p.Parallel || ungatedFFN(p) || lnBlock(p) || l.nFFN == nil ||
		l.gate == nil || l.up == nil || l.down == nil || l.mvd.res == nil || l.mvu.gated == nil ||
		g.bs.rmsQ == nil || g.bs.quantF == nil || l.mvd.k != p.NFFN || !g.GateUp2 || !g.gateUp2Shape(l) {
		return ""
	}
	// gateUp2Shape holds, so gate and up are one shape at one split.
	mg, md := l.mvg, l.mvd
	return fmt.Sprintf("ffn/%v/%d/%d/%d/%v/%v/%d/%d/%d/%v/act%v/w%d/warp%v/in%v",
		mg.q, mg.k, mg.rows, mg.split, mg.group, md.q, md.rows, md.split, md.rowt, md.group,
		kernels.ActKind(p.Act), max(p.ActWin, 32), g.warpNorm(), md.resIn != nil)
}

// tuneFFN decides, on this card, the recipe each dense gated FFN shape runs:
// two launches, Gate2, Gate2 with the norm's prologue, or gate and up as one
// segment launch with down forming the product. Every recipe the shape can
// express is timed over a sequence of the placed blocks sharing the key,
// norm through down, the way a token runs them; the incumbent is Gate2 and a
// challenger replaces it only where it is recipeMargin faster. Verdicts are
// cached per binary, device and key. Called after tuneActIn, so down runs in
// the form the prologue tuner chose. A forced Config.FFNForm or GateUpFuse
// skips it; called holding g.mu, outside any Session.
func (g *devTier) tuneFFN() {
	if g.FFNForm != FFNTuned || g.GateUpFuse || !g.GateUp2 || g.GateUp2Force || g.bs == nil || !g.actInOn() {
		return
	}
	if g.gu2Win == nil {
		g.gu2Win = map[splitKey]bool{}
	}
	type cand struct{ ls []*layer }
	by := map[string]*cand{}
	var keys []string
	for _, l := range g.layers {
		if l == nil || !l.ok {
			continue
		}
		l.ffnKey, l.ffnForm = "", FFNTuned
		if l.pg != nil && !l.pg.in {
			continue
		}
		key := g.ffnKey(l)
		if key == "" {
			continue
		}
		l.ffnKey = key
		c := by[key]
		if c == nil {
			c = &cand{}
			by[key], keys = c, append(keys, key)
		}
		if len(c.ls) < 16 {
			c.ls = append(c.ls, l)
		}
	}
	if g.ffnWin == nil {
		g.ffnWin = map[string]FFNForm{}
	}
	g.FFNTimed, g.FFNKept = 0, 0
	for _, key := range keys {
		g.FFNTimed++
		ck := g.tuneKey("ffn-" + key)
		if v, ok := lookupCache(g.kb.cacheDir, "gpuffn", ck); ok && g.kb.tune != TuneForce {
			f, err := ParseFFNForm(v)
			if err == nil {
				g.ffnWin[key] = f
				g.gu2Win[splitKey{by[key].ls[0].mvu.q, by[key].ls[0].mvu.rows, by[key].ls[0].mvu.k}] = f == FFNGate2 || f == FFNGate2In
				continue
			}
		}
		ls := by[key].ls
		sk := splitKey{ls[0].mvu.q, ls[0].mvu.rows, ls[0].mvu.k}
		// The incumbent is the device's default (gateUp2Pick with no
		// verdict): Gate2 where it writes the two launches' bits, else two.
		forms := []FFNForm{FFNGate2, FFNTwo}
		if !kernels.Gate2Exact(g.dev.API()) {
			forms = []FFNForm{FFNTwo, FFNGate2}
		}
		if ls[0].mvu.gate2In != nil {
			forms = append(forms, FFNGate2In)
		}
		if ls[0].gu != nil && ls[0].guDown != nil {
			forms = append(forms, FFNGU)
		}
		arms := make([]func(backend.Session) error, len(forms))
		for i, f := range forms {
			arms[i] = func(s backend.Session) error { return g.ffnArm(s, f, ls) }
		}
		med, ok := g.timeRecipes(arms)
		win, best := forms[0], recipeMargin
		for i := 1; i < len(forms); i++ {
			if ok[i] && med[i] < best {
				win, best = forms[i], med[i]
			}
		}
		g.ffnWin[key] = win
		g.gu2Win[sk] = win == FFNGate2 || win == FFNGate2In
		if backend.Verbose() {
			fmt.Fprintf(os.Stderr, "tier: FFN recipes %s over %d blocks:", key, len(ls))
			for i, f := range forms[1:] {
				fmt.Fprintf(os.Stderr, " %v %.4f (resolved %v)", f, med[i+1], ok[i+1])
			}
			fmt.Fprintf(os.Stderr, " of %v: %v\n", forms[0], win)
		}
		if ck != "" {
			saveCache(g.kb.cacheDir, "gpuffn", ck, win.String())
		}
	}
	for _, l := range g.layers {
		if l != nil && l.ok && l.ffnKey != "" {
			l.ffnForm = g.ffnWin[l.ffnKey]
		}
	}
	def := FFNTwo
	if kernels.Gate2Exact(g.dev.API()) {
		def = FFNGate2
	}
	for _, key := range keys {
		if g.ffnWin[key] != def {
			g.FFNKept++
		}
	}
	g.dropGraph()
}

// ffnArm is recipe f over the blocks ls: the FFN norm (where the recipe does
// not take it in), gate and up, and down with its residual, the launches
// encodeLayer makes for that recipe on a one-token decode.
func (g *devTier) ffnArm(s backend.Session, f FFNForm, ls []*layer) error {
	bs := g.bs
	ax := []backend.Buf{bs.a, bs.ax}
	for _, l := range ls {
		if f != FFNGate2In {
			if err := s.Launch(bs.rmsQ, 1, kernels.RMSNormGroup, bs.x2, l.nFFN, bs.h, bs.a, bs.ax); err != nil {
				return err
			}
		}
		switch f {
		case FFNGU:
			args := []backend.Buf{bs.a, bs.ax, l.gate.qs, l.gate.d, l.gate.sc, bs.g, l.up.qs, l.up.d, l.up.sc, bs.u}
			if err := s.Launch(l.gu.kern, l.gu.blocks, l.gu.w, args...); err != nil {
				return err
			}
			if err := g.mvLaunch(s, l.mvd, l.guDown, l.down, []backend.Buf{bs.g, bs.u}, bs.x, bs.x2); err != nil {
				return err
			}
			continue
		case FFNTwo:
			if err := g.mvLaunch(s, l.mvg, l.mvg.kern, l.gate, ax, bs.g); err != nil {
				return err
			}
			if err := g.mvLaunch(s, l.mvu, l.mvu.gated, l.up, ax, bs.act, bs.g); err != nil {
				return err
			}
		case FFNGate2:
			if err := g.mvLaunch(s, l.mvu, l.mvu.gate2, l.up, ax, bs.act, l.gate.qs, l.gate.d, l.gate.sc); err != nil {
				return err
			}
		case FFNGate2In:
			if err := g.mvLaunch(s, l.mvu, l.mvu.gate2In, l.up, []backend.Buf{bs.x2, l.nFFN}, bs.act,
				l.gate.qs, l.gate.d, l.gate.sc); err != nil {
				return err
			}
		}
		if l.mvd.resIn != nil {
			if err := g.mvLaunch(s, l.mvd, l.mvd.resIn, l.down, []backend.Buf{bs.act}, bs.x, bs.x2); err != nil {
				return err
			}
			continue
		}
		if err := s.Launch(bs.quantF, (kernels.QuantizeThreads(l.mvd.k/32)+127)/128, 128, bs.act, bs.a, bs.ax); err != nil {
			return err
		}
		if err := g.mvLaunch(s, l.mvd, l.mvd.res, l.down, ax, bs.x, bs.x2); err != nil {
			return err
		}
	}
	return nil
}
