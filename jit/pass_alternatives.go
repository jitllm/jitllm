package jit

// Alternatives takes, for each value a primitive computes two ways, the
// second way: its readers read that one.
type Alternatives struct{}

func (Alternatives) Name() string { return "alternatives" }

func (p Alternatives) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if len(k.Alts) == 0 {
			return nil, false, nil
		}
		m := k.clone()
		for _, a := range m.Alts {
			m.redirect(a[0], a[1], -1)
		}
		m.Alts = nil
		m.compact()
		return []*Kernel{m}, true, nil
	})
}
