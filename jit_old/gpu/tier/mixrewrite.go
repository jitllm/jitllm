package tier

import (
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/engine/graph"
)

// mixOffer is a mixture block's rewrites decided outside a token
// (prepMixture): found where the block's plan carries a mixture graph
// (nn.LayerPlan.Rewrites), and then the waves' decision at every row class
// and residency, so the site picks an entry and allocates nothing.
type mixOffer struct {
	found bool
	waves [graph.RowsChunk + 1][graph.ResidencyHost + 1]error
}

// prepMixture asks graph.MoEWaves on block l at every row class and
// residency on this device. A block whose plan has no mixture graph keeps the
// tier's own facts (wavesRun). Outside a token.
func (g *devTier) prepMixture(l *layer) {
	l.mix = mixOffer{}
	o, found := graph.Find(l.rewrites, graph.MoEWaves.Name)
	if !found {
		return
	}
	l.mix.found = true
	for rows := graph.RowsDecode; rows <= graph.RowsChunk; rows++ {
		t := g.target(rows)
		for res := graph.ResidencyCard; res <= graph.ResidencyHost; res++ {
			t.Residency = res
			l.mix.waves[rows][res] = o.Offer(t)
			if g.Rewrites[graph.MoEWaves.Name] == graph.ChoiceRefuse && l.mix.waves[rows][res] == nil {
				l.mix.waves[rows][res] = errRefusedByCaller
			}
		}
	}
}

// residency is where a streamed block's experts run: through its expert
// cache on the card, or on the host.
func (st *streamBank) residency() graph.Residency {
	if st.cached() {
		return graph.ResidencyCached
	}
	return graph.ResidencyHost
}

// errRefusedByCaller is a rewrite's decision where the caller's choice
// (Config.Rewrites) refuses it.
var errRefusedByCaller = errors.New("the caller refused it (tier.WithRewrite)")

// prepChoices checks the caller's Takes (Config.Rewrites) against block l's
// graph on this device: a Take whose rewrite matched the block but is
// offered at no row class and residency here is unmet, counted and
// explained, so a choice the device cannot honour is never dropped
// silently. A run-time alternative (graph.WhenRun) is taken nowhere else:
// with no executor for it the Take is unmet by construction. Outside a token.
func (g *devTier) prepChoices(l *layer) {
	for _, o := range l.rewrites {
		if g.Rewrites[o.Rewrite.Name] != graph.ChoiceTake {
			continue
		}
		var why error
		met := false
		for rows := graph.RowsDecode; rows <= graph.RowsChunk && !met; rows++ {
			t := g.target(rows)
			for res := graph.ResidencyCard; res <= graph.ResidencyHost && !met; res++ {
				t.Residency = res
				var taken bool
				taken, met = o.Chosen(t, graph.ChoiceTake)
				if !taken {
					why = o.Offer(t)
				}
			}
		}
		// Taking a run-time alternative needs its executor, which this tier
		// does not have for any of them yet.
		if met && o.Rewrite.When == graph.WhenRun {
			met, why = false, errors.New("no executor for a run-time alternative on this tier")
		}
		if !met {
			g.RewritesUnmet++
			g.RewritesUnmetWhy = fmt.Sprintf("%s asked for: %v", o.Rewrite.Name, why)
			g.explainSet(RecipeChoice{Site: o.Rewrite.Name, Key: "unmet", Recipe: "reference", Source: "asked",
				Why: g.RewritesUnmetWhy})
		}
	}
}
