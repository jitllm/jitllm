package convert

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// decisionLabelsOf is the label readout's rules a kind's reference
// hardcodes, nil for a kind that reads another mechanism.
//
//   - lev (interfaze-ai/lev) and vllm-d3 (vllm-sr/d3): codes A..Z then
//     AA..ZZ, those that are one token, up to 255 options; lev reads a noul
//     at its nine rating levels, asks a choice again in reversed order and
//     was trained on sorted keys; d3 asks its own default question where the
//     instructions are absent, and gives a choice's and a score's confidence
//     as one minus the normalised entropy, and a score's legend with each
//     description that is not text as sorted-key JSON.
//   - lfm2-d1 (LiquidAI/d1-3B): a choice's own keys where each is one letter,
//     else A..Z, or two digits past 26 options, falling back through a pool
//     of letters, digits and pairs; each code with its " "+code token; a
//     noul's yes and no forms, true shown first; a score read at its keys'
//     own tokens.
//   - openjev (openjev/openjev): one letter per option, A..Z then a..z, which
//     its template writes itself; a noul shown true first.
func decisionLabelsOf(kind jlm.DecisionKind) *jlm.DecisionLabels {
	letters := func(from, to rune) []string {
		var out []string
		for c := from; c <= to; c++ {
			out = append(out, string(c))
		}
		return out
	}
	pairs := func() []string {
		var out []string
		for a := 'A'; a <= 'Z'; a++ {
			for b := 'A'; b <= 'Z'; b++ {
				out = append(out, string([]rune{a, b}))
			}
		}
		return out
	}
	digits := func(n int, f string) []string {
		out := make([]string, n)
		for i := range out {
			out[i] = fmt.Sprintf(f, i)
		}
		return out
	}
	switch kind {
	case jlm.DecisionLev, jlm.DecisionVllmD3:
		l := &jlm.DecisionLabels{Codes: append(letters('A', 'Z'), pairs()...), MaxOptions: 255, ShowCodes: true}
		if kind == jlm.DecisionLev {
			l.NoulRatings, l.Reversed, l.SortKeys, l.IDInstructions = 9, true, true, true
		} else {
			l.EntropyConfidence, l.SortedLegend = true, true
		}
		return l
	case jlm.DecisionLFM2D1:
		pool := append(letters('A', 'Z'), digits(100, "%02d")...)
		pool = append(pool, letters('a', 'z')...)
		pool = append(pool, digits(200, "#%d")...)
		return &jlm.DecisionLabels{Codes: letters('A', 'Z'), WideCodes: digits(255, "%02d"), Pool: append(pool, pairs()...),
			MaxOptions: 255, KeyCodes: true, Spaced: true, ShowCodes: true,
			NoulTrue: []string{"yes", "Yes", "YES"}, NoulFalse: []string{"no", "No", "NO"}, NoulTrueFirst: true,
			ScoreByKey: true, IDInstructions: true}
	case jlm.DecisionOpenJev:
		codes := append(letters('A', 'Z'), letters('a', 'z')...)
		return &jlm.DecisionLabels{Codes: codes, MaxOptions: uint32(len(codes)), NoulTrueFirst: true, IDInstructions: true}
	}
	return nil
}

// StatesDecisionLabels says the converter states a label readout for kind.
func StatesDecisionLabels(kind jlm.DecisionKind) bool { return decisionLabelsOf(kind) != nil }
