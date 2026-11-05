package engine

import "github.com/jitllm/jitllm/nn"

// What a device served an instance and formed for it is read by the
// device's name, the CPU's as a card's: the CPU's the instance keeps as it
// runs there, a card's the card keeps for every instance it serves.

// Account is what device dev kept of what it served the instance and formed
// for it (nn.Account): nothing where no device of the instance is named dev.
func (s *Instance) Account(dev string) nn.Account {
	if dev == nn.CPU {
		return nn.Account{Steps: s.cpu.steps, Forms: s.cpu.composed, Asks: s.eng.cpu.Forms(s.m.Id).Asks(), FormErr: s.cpu.formErr}
	}
	for _, d := range [2]nn.LayerDevice{s.ld, s.ldCand} {
		if l, ok := d.(nn.Ledger); ok {
			if a, ok := l.AccountOn(dev); ok {
				return a
			}
		}
	}
	return nn.Account{}
}
