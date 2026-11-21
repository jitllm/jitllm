package engine

import "github.com/jitllm/jitllm/nn"

// What a device served an instance and formed for it is read by the
// device's name, the CPU's as a card's: the CPU's the instance keeps as it
// runs there, a card's the card keeps for every instance it serves.

// Account is what device dev kept of what it served the instance and formed
// for it (nn.Account): nothing where no device of the instance is named dev.
func (s *Instance) Account(dev string) nn.Account {
	for _, d := range s.devices() {
		if l, ok := d.(nn.Ledger); ok {
			if a, ok := l.AccountOn(dev); ok {
				return a
			}
		}
	}
	return nn.Account{}
}

// devices is every device the instance holds: its execution on the lanes'
// device, the card's session view and the card.
func (s *Instance) devices() [3]any { return [3]any{s.places[routeLanes], s.ld, s.ldCand} }
