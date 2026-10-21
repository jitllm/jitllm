package jit

type Device uint8

const (
	DeviceGPU Device = iota
	DeviceCPU
)

// Target is the device kernels are formed for.
type Target struct {
	Device Device
	// Slots is the device's resident threads.
	Slots int
	Lanes int
}

// Pass transforms one nest into the nests it becomes; applied false where
// the nest is not its to change.
type Pass interface {
	Name() string
	Apply(n *Nest, t Target) (out []*Nest, applied bool, err error)
}

// Profile is the passes in the order they run, each switched on or off.
type Profile struct {
	Passes []Pass
	Off    map[string]bool
}

type formed struct {
	nest   *Nest
	passes []string
}

func (p Profile) run(n *Nest, t Target) ([]formed, error) {
	cur := []formed{{nest: n}}
	for _, ps := range p.Passes {
		if p.Off[ps.Name()] {
			continue
		}
		var next []formed
		for _, f := range cur {
			out, applied, err := ps.Apply(f.nest, t)
			if err != nil {
				return nil, err
			}
			if !applied {
				next = append(next, f)
				continue
			}
			for _, o := range out {
				next = append(next, formed{nest: o, passes: append(append([]string(nil), f.passes...), ps.Name())})
			}
		}
		cur = next
	}
	return cur, nil
}

// ProfileFor is the passes target t runs, set to main's choices.
func ProfileFor(t Target) Profile {
	return Profile{}
}
