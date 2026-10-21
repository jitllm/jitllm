package jit

import "fmt"

// Pass is one optimization: IR in, rewritten IR out. It decides from the
// IR's shape and structure and the device, never from names.
type Pass interface {
	Name() string
	Apply(ks []*Kernel, d Device) ([]*Kernel, error)
}

// Profile is a device's passes in order, any of them turned off by name.
type Profile struct {
	Passes []Pass
	Off    map[string]bool
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
