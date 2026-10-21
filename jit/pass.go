package jit

import "fmt"

// Pass is one optimization: IR in, rewritten IR out. It decides from the
// IR's shape and structure and the device, never from names.
type Pass interface {
	Name() string
	Apply(ks []*Kernel, d Device) ([]*Kernel, error)
}

// Profile is a device's choices: what its parts' decompositions carry, and
// its passes in order, any of them turned off by name.
type Profile struct {
	Int8   *Int8Form
	Passes []Pass
	Off    map[string]bool
}

// Int8Form is the int8 form a device keeps of a block's contraction inputs
// produced by a row's op: windows of Window by an exact scale amax/127,
// codes rounded half away from zero packed Word to a word, the codes' sums
// a Sums.
type Int8Form struct {
	Window, Word, Sums int
}

func (p Profile) run(ks []*Kernel, d Device) ([]*Kernel, error) {
	for _, ps := range p.Passes {
		if p.Off[ps.Name()] {
			continue
		}
		out, err := ps.Apply(ks, d)
		if err != nil {
			return nil, fmt.Errorf("jit: pass %s: %w", ps.Name(), err)
		}
		ks = out
	}
	return ks, nil
}

// eachKernel applies f to every kernel, recording the pass on those f
// changes.
func eachKernel(name string, ks []*Kernel, f func(k *Kernel) ([]*Kernel, bool, error)) ([]*Kernel, error) {
	var out []*Kernel
	for _, k := range ks {
		ns, changed, err := f(k)
		if err != nil {
			return nil, err
		}
		if !changed {
			out = append(out, k)
			continue
		}
		for _, n := range ns {
			n.Passes = append(append([]string(nil), k.Passes...), name)
		}
		out = append(out, ns...)
	}
	return out, nil
}
