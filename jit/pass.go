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

// Profile is the passes in the order they run, each switched on or off,
// and the choices the decomposition takes from it. QuantizeActivations
// gives a contract's activation its int8 form whatever the weight's format,
// as main does where a row-local op of the block makes it.
type Profile struct {
	Passes              []Pass
	Off                 map[string]bool
	QuantizeActivations bool
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
	pairs := []Pair{{From: PrimMap, To: PrimMap, AtStores: true}}
	if rows == 1 {
		pairs = append(pairs, Pair{From: PrimContract, To: PrimMap, Additive: true})
	}
	contract := []PrimKind{PrimContract}
	return Profile{QuantizeActivations: true, Passes: []Pass{
		Fuse{Pairs: pairs},
		WorkRow{Group: 1024},
		Coarsen{Unit: 4},
		Parts{Kinds: []PrimKind{PrimMap}},
		SplitKeys{Align: 32, Pad: 32, Copies: 32},
		Stage{Unroll: 4, Chains: 4, Runs: true, Mask: true},
		Split{Slots: 30720, Max: 32, Block: 32, Kinds: contract},
		Blocks{Size: 32, Kinds: contract},
	}}
}

// covers reports whether a pass's kinds include k.
func covers(kinds []PrimKind, k PrimKind) bool {
	for _, x := range kinds {
		if x == k {
			return true
		}
	}
	return false
}
