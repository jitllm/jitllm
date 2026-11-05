package gpu

import "github.com/jitllm/jitllm/nn"

// AccountOn is what the card named dev kept (nn.Ledger): its own counters,
// which every instance it serves writes.
func (g *GPU) AccountOn(dev string) (nn.Account, bool) {
	for _, d := range g.devs {
		if d.name == dev {
			s := d.stats()
			return nn.Account{Steps: s.Steps, Forms: s.BlockForms, Asks: int64(s.FormAsks), FormErr: s.FormErr}, true
		}
	}
	return nn.Account{}, false
}

// AccountOn is the device's (GPU.AccountOn).
func (s *gpuSession) AccountOn(dev string) (nn.Account, bool) { return s.g.AccountOn(dev) }
