package tier

// gateUp2Shape says block l's FFN has a Gate2 form: a plain dense gated FFN
// whose gate and up are one shape at one final-row split.
func (g *devTier) gateUp2Shape(l *layer) bool {
	mg, mu := &l.mvg, &l.mvu
	return !g.Persist && mu.gate2 != nil && mu.gated != nil && l.clampB == nil && l.nSubFFN == nil && !l.sparse &&
		l.router == nil && l.gate != nil && l.up != nil && mg.kern != nil && mg.bias == nil && mu.bias == nil &&
		mg.q == mu.q && mg.rows == mu.rows && mg.k == mu.k && mg.split == mu.split && mg.group == mu.group &&
		mg.rowt <= 1 && mg.slots == 0 && mg.experts <= 1
}

// gateUp2Swapped hands the Gate2 launch up's weight as the gate's and the
// gate's as up's: the violation the gate/up gate must see
// (SwapGateUp2ForTest, jitllmtest builds only).
var gateUp2Swapped bool
