package server

import "time"

// stepBudget is how many prompt tokens a step of the loop carries beside its
// decoding rows: the chunked prefill's token budget, measured rather than
// set.
//
// A prompt admitted beside decoding rows costs each of them its chunk's time
// at their next token. Fed at the step's full width (a device or host chunk,
// 512 rows) that is a whole prefill chunk between two tokens, and the decoding
// rows' inter-token tail grows by it at every admission. So the loop feeds a
// prompt in pieces sized so a step costs at most cost decode steps
// (Config.StepCost). It fits the steps it runs anyway, a step's time against
// the prompt tokens it carried beside decoding rows, as a line: the intercept
// is a decode step's time, the slope a prompt token's. The budget is the
// tokens whose slope comes to cost-1 intercepts. A step that carried no prompt
// token measures the intercept itself, and once there is one it stands in for
// the fitted one: an intercept extrapolated across a few noisy widths can come
// out near zero (0.02 ms against 2.7 ms decode-only steps, stories15M on a
// V100), and every step then reads as over its cost and the budget stays at
// its floor. Steps under arrivals nearly always carry some prompt, so the
// intercept cannot wait for decode-only steps; and until the steps' widths
// spread enough to give a slope, or while
// prompt tokens measure free (under the compute crossover the step is bound
// by the weight read either way), the budget doubles, which spreads them.
// Once a fit has given a budget, steps that settle on it keep it.
//
// The factor is the policy between two axes: a decoding row's worst
// inter-token gap is about cost decode steps, and a waiting prompt's time to
// its first token falls as the factor rises. Prompts are fed oldest first,
// beside the fairness level's even share (promptUnits, batchfair.go), so a
// long one still finishes, a budget a step.
//
// Config.StepPromptTokens fixes the budget instead. Only the loop goroutine
// touches it.
type stepBudget struct {
	fixed, ceiling int
	// cost is how many decode steps' time a step carrying prompt tokens may
	// take.
	cost float64
	// cur is the measured budget.
	cur int
	// The line's moving sums: weight, x (prompt tokens), y (step time), xx
	// and xy, each decayed by budgetAlpha a step.
	n, x, y, xx, xy float64
	// fitted says the fit has given a budget, and icept is its last decode
	// step's time; dec and decN are the moving sums of the steps that carried
	// no prompt token, the decode step measured (decodeStep).
	fitted    bool
	icept     float64
	dec, decN float64
}

const (
	// budgetStart is the budget before anything is measured: small, so the
	// first admission beside decoding rows is cheap.
	budgetStart = 32
	// budgetFloor keeps a prompt moving however slow a token measures.
	budgetFloor = 8
	// budgetAlpha is how fast the fit forgets: the decoding rows and their
	// contexts move under it.
	budgetAlpha = 0.1
	// DefaultStepCost is Config.StepCost's default.
	DefaultStepCost = 4.0
)

func newStepBudget(fixed, ceiling int, cost float64) *stepBudget {
	ceiling = max(ceiling, 1)
	if cost <= 1 {
		cost = DefaultStepCost
	}
	return &stepBudget{fixed: fixed, ceiling: ceiling, cost: cost, cur: min(budgetStart, ceiling)}
}

// tokens is the prompt tokens the next step beside decoding rows may carry.
func (b *stepBudget) tokens() int {
	if b.fixed > 0 {
		return b.fixed
	}
	return b.cur
}

// observe takes a step's time: the decoding rows and prompt tokens it carried.
func (b *stepBudget) observe(decoding, prompt int, d time.Duration) {
	if decoding == 0 || b.fixed > 0 {
		return
	}
	x, y := float64(prompt), float64(d)
	k := 1 - budgetAlpha
	b.n, b.x, b.y = k*b.n+1, k*b.x+x, k*b.y+y
	b.xx, b.xy = k*b.xx+x*x, k*b.xy+x*y
	if prompt == 0 {
		b.dec, b.decN = k*b.dec+y, k*b.decN+1
	}
	if prompt < b.cur {
		// A step the budget did not limit can lower it, never raise it.
		if w, ok := b.want(); ok && w < b.cur {
			b.cur = max(w, budgetFloor)
		}
		return
	}
	w, ok := b.want()
	ic := b.decodeStep()
	switch {
	case ok:
		b.fitted = true
	case b.fitted && y <= b.cost*ic:
		// The widths settled on the budget and stopped spreading, and the
		// step is within its cost: the last fit stands -- unless the step
		// spent under half of what its prompt may cost, when the fit is
		// stale and the budget doubles, which spreads the widths again. A
		// fit taken across an outlier (a step that compiled kernels or grew
		// the KV pool) otherwise held the budget at its floor for good: on a
		// V100 at 64 requests it read 60 ms a prompt token, fed every prompt
		// 8 tokens a step, and the first token waited minutes.
		if y-ic >= (b.cost-1)*ic/2 {
			return
		}
		w = 2 * b.cur
	case b.fitted:
		// Settled, and over its cost: the prompt tokens cost more than
		// the fit said. Against the last decode step's time, the tokens
		// that would have cost cost-1 of them.
		w = int(x * (b.cost - 1) * ic / (y - ic))
	default:
		w = 2 * b.cur
	}
	b.cur = min(max(min(w, 2*b.cur), budgetFloor), b.ceiling)
}

// want is the budget the fit gives, and whether it gives one: the spread of
// the widths is wide enough for a slope, and the slope is positive.
func (b *stepBudget) want() (int, bool) {
	mx, my := b.x/b.n, b.y/b.n
	vx := b.xx/b.n - mx*mx
	if vx < 4 {
		return 0, false
	}
	slope := (b.xy/b.n - mx*my) / vx
	icept := my - slope*mx
	if slope <= 0 || icept <= 0 && b.decN == 0 {
		return 0, false
	}
	if icept > 0 {
		b.icept = icept
	}
	return int((b.cost - 1) * b.decodeStep() / slope), true
}

// decodeStep is a decode step's time: the steps that carried no prompt token
// measure it where there are any, the fit's intercept otherwise.
func (b *stepBudget) decodeStep() float64 {
	if b.decN > 0 {
		return b.dec / b.decN
	}
	return b.icept
}
