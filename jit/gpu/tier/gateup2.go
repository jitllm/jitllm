package tier

import (
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// gateUp2On says block l's gated FFN runs gate and up as one Gate2 launch
// (Config.GateUp2): up's kernel carries the gate's weight as a second stream
// and writes act(gate)*up where up's gated launch wrote it. It holds where
// the two launches it replaces would run -- a one-token decode, a plain dense
// gated FFN whose up has a gated form -- where gate and up are one shape at
// one split, and where this device chose it for the shape (gateUp2Pick).
func (g *devTier) gateUp2On(l *layer, R int, ragged, voltaOnly, ungated, noActMul bool) bool {
	mg, mu := &l.mvg, &l.mvu
	return g.GateUp2 && g.gateUp2Shape(l) && R == 1 && !ragged && !voltaOnly && !ungated && !noActMul &&
		(g.GateUp2Force || g.FFNForm == FFNGate2 || g.FFNForm == FFNGate2In || g.gateUp2Pick(splitKey{mu.q, mu.rows, mu.k})) &&
		!(g.TableSplit && (mg.alt != nil || mu.alt != nil))
}

// gateUp2Shape says block l's FFN has a Gate2 form: a plain dense gated FFN
// whose gate and up are one shape at one final-row split.
func (g *devTier) gateUp2Shape(l *layer) bool {
	mg, mu := &l.mvg, &l.mvu
	return !g.Persist && mu.gate2 != nil && mu.gated != nil && l.clampB == nil && l.nSubFFN == nil && !l.sparse &&
		l.router == nil && l.gate != nil && l.up != nil && mg.kern != nil && mg.bias == nil && mu.bias == nil &&
		mg.q == mu.q && mg.rows == mu.rows && mg.k == mu.k && mg.split == mu.split && mg.group == mu.group &&
		mg.rowt <= 1 && mg.slots == 0 && mg.experts <= 1
}

// gateUp2Pick is this device's choice for a shape: the tuner's verdict
// (tuneFFN, recipe.go) where it has one; otherwise Gate2 where it writes the two
// launches' bits (oldkernels.Gate2Exact), which measured faster on every shape
// a V100 was given, and the two launches elsewhere until a trial says.
func (g *devTier) gateUp2Pick(key splitKey) bool {
	if v, ok := g.gu2Win[key]; ok {
		return v
	}
	return oldkernels.Gate2Exact(g.dev.API())
}

// gateUp2Swapped (the wrong-answer switch of a gate, Config.hook; WithViolation) hands the Gate2 launch up's weight as the gate's and the
// gate's as up's: the violation the gate/up gate must see
// (SwapGateUp2ForTest, jitllmtest builds only).
