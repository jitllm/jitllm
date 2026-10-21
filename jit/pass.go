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

// Pass transforms the program's nests, naming itself in each nest it
// changes.
type Pass interface {
	Name() string
	Apply(prog []*Nest, t Target) ([]*Nest, error)
}

// Profile is the passes in the order they run, each switched on or off.
type Profile struct {
	Passes []Pass
	Off    map[string]bool
}

func (p Profile) run(prog []*Nest, t Target) ([]*Nest, error) {
	for _, ps := range p.Passes {
		if p.Off[ps.Name()] {
			continue
		}
		var err error
		if prog, err = ps.Apply(prog, t); err != nil {
			return nil, err
		}
	}
	return prog, nil
}

// eachNest is a pass over one nest at a time: f's nests in its place where
// it applied.
func eachNest(name string, prog []*Nest, f func(*Nest) ([]*Nest, bool, error)) ([]*Nest, error) {
	var out []*Nest
	for _, n := range prog {
		ns, applied, err := f(n)
		if err != nil {
			return nil, err
		}
		if !applied {
			out = append(out, n)
			continue
		}
		for _, m := range ns {
			m.Passes = append(append([]string(nil), n.Passes...), name)
			out = append(out, m)
		}
	}
	return out, nil
}

// ProfileFor is the passes target t runs at a step of rows, set to main's
// choices.
func ProfileFor(t Target, rows int) Profile {
	var pairs []Pair
	if rows == 1 {
		pairs = append(pairs, Pair{From: PrimContract, To: PrimMap})
	}
	return Profile{Passes: []Pass{
		Fuse{Pairs: pairs},
		Split{Slots: 30720, Max: 32, Block: 32},
		Blocks{Size: 32},
	}}
}
