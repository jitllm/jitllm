package tier

import (
	"fmt"
	"os"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/gpu/backend"
)

// The gated output norm's recipes (graph.NormGate): a gated delta rule's
// output, RMS-normalised per value head and gated by SiLU or sigmoid of z,
// as two launches (outNorm writing the normalised row, then outActMul or
// outSigMul reading it) or as one (kernels.HeadNormGated), the normalised
// row never written. The rewrite decides where the fused launch is offered
// at all (the sublayer graph's edges and the backend's numerical contract);
// a forced form or the per-card trial decides where it runs.

// normGateBroken (the wrong-answer switch of a gate, Config.hook; WithViolation) hands the fused launch z as the rule's output and the
// output as z: a test's violation (BreakNormGateForTest).

// NormGateForm is the recipe a gated output norm runs (Config.NormGate).
type NormGateForm int

const (
	// NormGateTuned takes tuneNormGate's verdict at decode, and the two
	// launches at a prompt chunk's rows, where nothing was timed.
	NormGateTuned NormGateForm = iota
	// NormGateTwo is the reference: the norm, then the gate.
	NormGateTwo
	// NormGateFused is the norm and its gate in one launch, at every row
	// class the rewrite is offered at.
	NormGateFused
)

var normGateNames = [...]string{"tuned", "two", "fused"}

func (f NormGateForm) String() string {
	if f >= 0 && int(f) < len(normGateNames) {
		return normGateNames[f]
	}
	return fmt.Sprintf("normgate%d", int(f))
}

// ParseNormGateForm reads a NormGateForm by name.
func ParseNormGateForm(s string) (NormGateForm, error) {
	for i, n := range normGateNames {
		if s == n {
			return NormGateForm(i), nil
		}
	}
	return 0, fmt.Errorf("tier: gated norm recipe %q: want tuned, two or fused", s)
}

// normGateOn is whether block l runs its gated output norm as one launch at
// decode (one row a sequence) or at a prompt chunk's rows: offered there by
// the block's graph, and forced or kept by the trial. Read inside a token, so
// it only reads what prepNormGate and tuneNormGate decided.
func (g *devTier) normGateOn(l *layer, decode bool) bool {
	i := 1
	if decode {
		i = 0
	}
	if !l.normGateOffer[i] {
		return false
	}
	switch g.Rewrites[graph.NormGate.Name] {
	case graph.ChoiceTake:
		return true
	case graph.ChoiceRefuse:
		return false
	}
	switch g.NormGate {
	case NormGateFused:
		return true
	case NormGateTwo:
		return false
	}
	return decode && g.normGateWin[l.normGateKey]
}

// prepNormGate asks the block's graph, at this device's Target for each row
// class, whether graph.NormGate is offered on block l, and keys the decode
// verdict by the shape and the Target. A block whose plan carries no sublayer
// graph, or one where the rewrite did not match, is not offered it. Outside a
// token.
func (g *devTier) prepNormGate(l *layer) {
	l.normGateOffer, l.normGateKey, l.normGateWhy = [2]bool{}, "", nil
	o, found := graph.Find(l.rewrites, graph.NormGate.Name)
	if !found {
		return
	}
	// The shape the trial's cost depends on, read off the matched region:
	// the row's width, the head width, the epsilon and the gate.
	hn, am := o.Graph.Ops[o.Region.Ops[0]], o.Graph.Ops[o.Region.Ops[1]]
	shape := fmt.Sprintf("normgate/%d/%d/eps%g/act%d", o.Graph.Values[hn.Out[0]].Width, hn.Attr.HeadDim,
		hn.Attr.Eps, am.Attr.Act)
	for i, rows := range []graph.RowClass{graph.RowsDecode, graph.RowsChunk} {
		_, tkey, why := g.offer(l, graph.NormGate.Name, rows)
		l.normGateOffer[i] = why == nil
		if i == 0 {
			l.normGateWhy = why
			if why == nil {
				l.normGateKey = shape + targetSuffix(tkey)
			}
		}
	}
}

// tuneNormGate decides, on this card, where the gated output norm runs as
// one launch at decode: per key it times the two launches against the fused
// one over a sequence of the placed blocks sharing it (timeRecipes, the two
// launches the incumbent), and keeps the fused launch only where it is
// recipeMargin faster. The verdict is cached per binary, device and key. A
// forced Config.NormGate skips it; called from retuneDecode, holding g.mu,
// outside any Session.
func (g *devTier) tuneNormGate() {
	if g.bs == nil {
		return
	}
	g.explainDrop("normgate")
	g.NormGateTimed, g.NormGateKept = 0, 0
	if g.NormGate != NormGateTuned {
		g.explainSet(RecipeChoice{Site: "normgate", Key: "*", Recipe: g.NormGate.String(), Source: "forced",
			Why: "Config.NormGate (JITLLM_GPU_NORMGATE) names it"})
		return
	}
	by := map[string][]*layer{}
	var keys []string
	for _, l := range g.layers {
		if l == nil || !l.ok {
			continue
		}
		if l.normGateWhy != nil {
			g.explainSet(RecipeChoice{Site: "normgate", Key: "refused", Recipe: NormGateTwo.String(),
				Source: "refused", Why: l.normGateWhy.Error()})
		}
		if l.normGateKey == "" || (l.pg != nil && !l.pg.in) {
			continue
		}
		if by[l.normGateKey] == nil {
			keys = append(keys, l.normGateKey)
		}
		if len(by[l.normGateKey]) < 16 {
			by[l.normGateKey] = append(by[l.normGateKey], l)
		}
	}
	if g.normGateWin == nil {
		g.normGateWin = map[string]bool{}
	}
	for _, key := range keys {
		ls := by[key]
		g.NormGateTimed++
		ck := g.tuneKey("normgate-" + key)
		if v, ok := lookupCache(g.kb.cacheDir, "gpunormgate", ck); ok && g.kb.tune != TuneForce {
			g.normGateWin[key] = v == "1"
			g.explainSet(RecipeChoice{Site: "normgate", Key: key, Recipe: normGateRecipe(v == "1"), Source: "cache",
				Why: "a verdict timed earlier on this binary and device", Blocks: len(ls)})
		} else {
			med, ok := g.timeRecipes([]func(backend.Session) error{
				func(s backend.Session) error { return g.normGateArm(s, false, ls) },
				func(s backend.Session) error { return g.normGateArm(s, true, ls) },
			})
			win := ok[1] && med[1] < recipeMargin
			g.normGateWin[key] = win
			why := "the fused launch did not time 1% faster than the two"
			if win {
				why = "the fused launch timed 1% faster than the two"
			} else if !ok[1] {
				why = "the trial did not resolve; the two launches stay"
			}
			g.explainSet(RecipeChoice{Site: "normgate", Key: key, Recipe: normGateRecipe(win), Source: "timed",
				Why: why, Blocks: len(ls), Trial: []Arm{{Recipe: "fused", Ratio: med[1], Resolved: ok[1]}}})
			if backend.Verbose() {
				fmt.Fprintf(os.Stderr, "tier: gated norm %s: %.4f of the two launches over %d blocks (resolved %v): %v\n",
					key, med[1], len(ls), ok[1], win)
			}
			if ok[1] && ck != "" {
				v := "0"
				if win {
					v = "1"
				}
				saveCache(g.kb.cacheDir, "gpunormgate", ck, v)
			}
		}
		if g.normGateWin[key] {
			g.NormGateKept++
		}
	}
	g.dropGraph()
}

func normGateRecipe(fused bool) string {
	if fused {
		return NormGateFused.String()
	}
	return NormGateTwo.String()
}

// normGateArm is one recipe over the blocks ls on a one-row decode: the
// launches emitLinear makes for it, on the decode scratch.
func (g *devTier) normGateArm(s backend.Session, fused bool, ls []*layer) error {
	bs := g.bs
	r := &bs.rec
	if bs.outNorm == nil || bs.outNormGate == nil || bs.dOut == nil || bs.dZ == nil || bs.dOutN == nil || bs.dAct == nil {
		return errNoBatch
	}
	inner := r.VHeads * r.VDim
	vg, vwid := headNormGeom(bs.outNormL, r.VHeads)
	mul := bs.outActMul
	if r.ChanDecay || r.SigmoidGate {
		mul = bs.outSigMul
	}
	for _, l := range ls {
		if fused {
			if err := s.Launch(bs.outNormGate, vg, vwid, bs.dOut, l.ssmNorm, bs.dZ, bs.dAct); err != nil {
				return err
			}
			continue
		}
		if err := s.Launch(bs.outNorm, vg, vwid, bs.dOut, l.ssmNorm, bs.dOutN); err != nil {
			return err
		}
		if err := s.Launch(mul, (inner+127)/128, 128, bs.dZ, bs.dOutN, bs.dAct); err != nil {
			return err
		}
	}
	return nil
}
