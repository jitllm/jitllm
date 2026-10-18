package tier

// The dense FFN's gate and up on one block-step, as one named choice made
// before any launch (ffnGateUp) rather than inline at each: the op span
// gate, up [and the FFN norm before them] lowered to one of these forms.
// emit switches on it; the launches are the ones each form always made.

// ffnGateUpForm is how a dense FFN's gate and up projections run.
type ffnGateUpForm uint8

const (
	// gateUpUngated: no gate (an ungated FFN); up runs with the activation.
	gateUpUngated ffnGateUpForm = iota
	// gateUpClamped: Gemma 4's clipped linears, each reading its own
	// clamped copy of the normed rows.
	gateUpClamped
	// gateUpGate2In: the norm, gate and up in one launch (the gate2in recipe).
	gateUpGate2In
	// gateUpIn: gate (and up) normalising and quantizing the stream
	// themselves (the actin prologue).
	gateUpIn
	// gateUpGU: gate and up in one launch over one operand, down forming
	// act(gate)*up as it quantizes.
	gateUpGU
	// gateUpGate2: gate and up in one launch writing the product (Gate2).
	gateUpGate2
	// gateUpTwo: gate and up as two matvecs.
	gateUpTwo
)

var gateUpNames = [...]string{"ungated", "clamped", "gate2in", "actin", "gu", "gate2", "two"}

func (f ffnGateUpForm) String() string { return gateUpNames[f] }

// ffnGateUp decides block l's gate/up form on a step of R rows (ragged where
// rag), from what the block holds and the step's conditions. It launches
// nothing.
func (g *devTier) ffnGateUp(l *layer, R int, rag, voltaOnly, ungated, resScaled bool, skip map[string]bool,
	g2in, ffnIn bool, nffn int) ffnGateUpForm {
	switch {
	case l.clampB != nil && !ungated:
		return gateUpClamped
	case g2in:
		return gateUpGate2In
	case ffnIn:
		return gateUpIn
	case (g.GateUpFuse || g.ffnFormOf(l) == FFNGU) && l.gu != nil && l.guDown != nil && R == 1 && !rag && !voltaOnly && !ungated &&
		l.nSubFFN == nil && l.nPostFFN == nil && !resScaled && !skip["add"] && !skip["actmul"] && !skip["quantf"] &&
		!g.TableSplit && l.mvd.k == nffn && l.k3 == nil && l.q4e == nil && !l.nonCausal && !l.sparse:
		return gateUpGU
	case g.ffnFormOf(l) != FFNTwo && g.gateUp2On(l, R, rag, voltaOnly, ungated, skip["actmul"]):
		return gateUpGate2
	case !ungated:
		return gateUpTwo
	}
	return gateUpUngated
}

// ffnDownForm is how a dense FFN's down projection reads its operand.
type ffnDownForm uint8

const (
	// downFormPlain: down reads the quantized act(gate)*up the step wrote; the
	// residual add rides down's rows where the step allows (mvres).
	downFormPlain ffnDownForm = iota
	// downFormIn: down quantizes the product up formed in its own store.
	downFormIn
	// downFormAct: down forms act(gate)*up from gate and up as it quantizes.
	downFormAct
)

func (f ffnDownForm) String() string { return [...]string{"plain", "actin", "actin-act"}[f] }

// ffnDown decides block l's down form on a step, once gate and up have run:
// gated is whether up formed the product in its own store.
func (g *devTier) ffnDown(l *layer, R int, rag, voltaOnly, resScaled bool, skip map[string]bool,
	gated, ffnIn bool, nffn int) ffnDownForm {
	ok := R == 1 && !rag && g.actInOn() && !voltaOnly && l.clampB == nil && l.nPostFFN == nil && l.nSubFFN == nil &&
		!resScaled && !skip["add"] && !skip["quantf"] && !skip["actmul"] && l.k3 == nil && l.q4e == nil && !l.nonCausal &&
		!(g.TableSplit && l.mvd.alt != nil) && l.mvd.k == nffn
	switch {
	case ok && gated && l.mvd.resIn != nil:
		return downFormIn
	case ok && ffnIn && !gated && l.mvd.resActIn != nil:
		return downFormAct
	}
	return downFormPlain
}
