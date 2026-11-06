package jlm

import "fmt"

// DeltaDecaySoftplus is the delta rule's decay exp(A * softplus(alpha + dt)):
// its logits shifted by the stored dt bias, through softplus, times the
// stored A, exponentiated.
func DeltaDecaySoftplus() *Program {
	return &Program{NIn: 1, Out: 4, Ops: []TowerOp{
		{Code: TOpShift, In: []uint16{0}, Bind: []Role{RoleSSMDtBias}},
		{Code: TOpStated, In: []uint16{1}, Expr: &Expr{Terms: []Term{{Fn: FnSoftplus, Args: [3]Arg{{Kind: ArgIn}}}}}},
		{Code: TOpScale, In: []uint16{2}, Bind: []Role{RoleSSMA}},
		{Code: TOpStated, In: []uint16{3}, Expr: &Expr{Terms: []Term{{Fn: FnExp, Args: [3]Arg{{Kind: ArgIn}}}}}},
	}}
}

// DeltaDecayBounded is the decay under a lower bound lb (negative):
// exp(lb * sigmoid(-A * (alpha + dt))).
func DeltaDecayBounded(lb float32) *Program {
	return &Program{NIn: 1, Out: 3, Ops: []TowerOp{
		{Code: TOpShift, In: []uint16{0}, Bind: []Role{RoleSSMDtBias}},
		{Code: TOpScale, In: []uint16{1}, Bind: []Role{RoleSSMA}},
		{Code: TOpStated, In: []uint16{2}, Expr: &Expr{Terms: []Term{
			{Fn: FnMul, Args: [3]Arg{{Kind: ArgIn}, {Kind: ArgConst, Value: -1}}},
			{Fn: FnSigmoid, Args: [3]Arg{{Kind: ArgTerm, Index: 0}}},
			{Fn: FnMul, Args: [3]Arg{{Kind: ArgTerm, Index: 1}, {Kind: ArgConst, Value: lb}}},
			{Fn: FnExp, Args: [3]Arg{{Kind: ArgTerm, Index: 2}}},
		}}},
	}}
}

// DeltaGateSiLU is the output gate silu(z) * o, z the gate's projection and
// o the normed output.
func DeltaGateSiLU() *Program {
	return &Program{NIn: 2, Out: 2, Ops: []TowerOp{{Code: TOpStated, In: []uint16{0, 1}, Expr: &Expr{Terms: []Term{
		{Fn: FnSigmoid, Args: [3]Arg{{Kind: ArgIn}}},
		{Fn: FnMul, Args: [3]Arg{{Kind: ArgIn}, {Kind: ArgTerm, Index: 0}}},
		{Fn: FnMul, Args: [3]Arg{{Kind: ArgTerm, Index: 1}, {Kind: ArgIn, Index: 1}}},
	}}}}}
}

// DeltaGateSigmoid is the output gate sigmoid(z) * o.
func DeltaGateSigmoid() *Program {
	return &Program{NIn: 2, Out: 2, Ops: []TowerOp{{Code: TOpStated, In: []uint16{0, 1}, Expr: &Expr{Terms: []Term{
		{Fn: FnSigmoid, Args: [3]Arg{{Kind: ArgIn}}},
		{Fn: FnMul, Args: [3]Arg{{Kind: ArgTerm, Index: 0}, {Kind: ArgIn, Index: 1}}},
	}}}}}
}

// linear reports a model with a delta rule's blocks.
func (c *Config) linear() bool {
	for _, k := range c.LayerKinds {
		if k == LayerLinearAttn {
			return true
		}
	}
	return false
}

// olderDelta states, for a container an older writer made, the decay and
// gate it carried as fields: the lower bound (KDALowerBound) and the sigmoid
// gate (Flag2DeltaSigmoidGate), the per-channel decay's gate a sigmoid (its
// two architectures' own), every other one silu.
func (c *Config) olderDelta() error {
	if c.DeltaDecay != nil || c.DeltaGate != nil || !c.linear() {
		return nil
	}
	if !(c.KDALowerBound <= 0) {
		return fmt.Errorf("jlm: a decay bound of %g; a bound is negative", c.KDALowerBound)
	}
	c.DeltaDecay, c.DeltaGate = DeltaDecaySoftplus(), DeltaGateSiLU()
	if c.KDALowerBound != 0 {
		c.DeltaDecay = DeltaDecayBounded(c.KDALowerBound)
	}
	if c.Flags2.Has(Flag2DeltaSigmoidGate) || c.Arch == ArchKimiLinear || c.Arch == ArchKimiK3 {
		c.DeltaGate = DeltaGateSigmoid()
	}
	return nil
}

// deltaValid refuses a statement no delta rule reads: both programs or
// neither, a decay of one input and a gate of two, and none on a model with
// no delta rule.
func (c *Config) deltaValid() error {
	if c.DeltaDecay == nil && c.DeltaGate == nil {
		return nil
	}
	switch {
	case c.DeltaDecay == nil || c.DeltaGate == nil:
		return fmt.Errorf("jlm: a delta rule's decay without its gate, or its gate without its decay")
	case !c.linear():
		return fmt.Errorf("jlm: a delta rule's statement on a model with no delta rule")
	case c.DeltaDecay.NIn != 1 || c.DeltaGate.NIn != 2:
		return fmt.Errorf("jlm: a delta rule's decay over %d inputs and gate over %d", c.DeltaDecay.NIn, c.DeltaGate.NIn)
	}
	if err := c.DeltaDecay.valid("delta decay"); err != nil {
		return err
	}
	return c.DeltaGate.valid("delta gate")
}
