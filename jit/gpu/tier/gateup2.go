package tier

// gateUp2On says block l's gated FFN runs gate and up as one Gate2 launch
// (Config.GateUp2): up's kernel carries the gate's weight as a second stream
// and writes act(gate)*up where up's gated launch wrote it. It holds where
// the two launches it replaces would run -- a one-token decode, a plain dense
// gated FFN whose up has a gated form -- and where gate and up are one shape
// at one split, so each row's sum is the one the gate's own launch makes.
func (g *devTier) gateUp2On(l *layer, R int, ragged, voltaOnly, ungated, noActMul bool) bool {
	mg, mu := &l.mvg, &l.mvu
	return g.GateUp2 && !g.Persist && mu.gate2 != nil && mu.gated != nil && R == 1 && !ragged && !voltaOnly &&
		!ungated && !noActMul && l.clampB == nil && l.nSubFFN == nil && !l.sparse && l.router == nil &&
		l.gate != nil && l.up != nil && mg.kern != nil && mg.bias == nil && mu.bias == nil &&
		mg.q == mu.q && mg.rows == mu.rows && mg.k == mu.k && mg.split == mu.split && mg.group == mu.group &&
		mg.rowt <= 1 && mg.slots == 0 && mg.experts <= 1 &&
		!(g.TableSplit && (mg.alt != nil || mu.alt != nil))
}

// gateUp2Swapped hands the Gate2 launch up's weight as the gate's and the
// gate's as up's: the violation the gate/up gate must see
// (SwapGateUp2ForTest, jitllmtest builds only).
var gateUp2Swapped bool
