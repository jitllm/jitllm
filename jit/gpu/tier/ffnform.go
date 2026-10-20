package tier

import "sync/atomic"

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

// qkvForm is how a block's q, k and v projections run on a step.
type qkvForm uint8

const (
	// qkvQOnly: q alone; k and v are a KV-sharing block's source's.
	qkvQOnly qkvForm = iota
	// qkvQK: q and k, v being k's projection (Gemma 4's global layers).
	qkvQK
	// qkvGrouped: the three in one grouped launch (a ragged step's segments).
	qkvGrouped
	// qkvClamped: Gemma 4's clipped linears, each over its clamped input.
	qkvClamped
	// qkvSeparate: three launches normalising and quantizing the block input
	// themselves (the actin prologue).
	qkvSeparate
	// qkvGEMM: one binary16 grid over the three (GemmVoltaSegments).
	qkvGEMM
	// qkvThree: three matvecs.
	qkvThree
)

var qkvNames = [...]string{"q-only", "q-k", "grouped", "clamped", "actin", "gemm", "three"}

func (f qkvForm) String() string { return qkvNames[f] }

// The span a recorded form is of.
const (
	formQKV = iota
	formGateUp
	formDown
	formBlock
	numForms
)

// blockForm is whether a block-step ran as its device program (devprog.go)
// or through the block encoder.
type blockForm uint8

const (
	blockEncoder blockForm = iota
	blockProgram
)

func (f blockForm) String() string { return [...]string{"encoder", "program"}[f] }

// blockForms is the forms a block's last step took, one byte a span (one
// past the form, zero for none yet), packed so a reader never races a step.
type blockForms struct{ v atomic.Uint32 }

func (b *blockForms) note(span int, form uint8) {
	for {
		old := b.v.Load()
		sh := uint(8 * span)
		nv := old&^(0xff<<sh) | uint32(form+1)<<sh
		if old == nv || b.v.CompareAndSwap(old, nv) {
			return
		}
	}
}

// BlockForms is the lowering block li's last step took on this device, by
// op span ("qkv", "gate_up", "down"), each the form's name; nil where the
// device does not hold the block. A span the block has not run is absent.
func (g *GPU) BlockForms(li int) map[string]string {
	g.mu.Lock()
	d := g.own[li]
	g.mu.Unlock()
	if d == nil {
		return nil
	}
	d.mu.Lock()
	l := d.layers[li]
	d.mu.Unlock()
	if l == nil {
		return nil
	}
	v := l.ran.v.Load()
	out := map[string]string{}
	for span, name := range [numForms]string{"qkv", "gate_up", "down", "block"} {
		f := uint8(v >> (8 * span))
		if f == 0 {
			continue
		}
		switch span {
		case formQKV:
			out[name] = qkvForm(f - 1).String()
		case formGateUp:
			out[name] = ffnGateUpForm(f - 1).String()
		case formDown:
			out[name] = ffnDownForm(f - 1).String()
		case formBlock:
			out[name] = blockForm(f - 1).String()
		}
	}
	return out
}
