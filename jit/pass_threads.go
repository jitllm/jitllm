package jit

// Threads binds a kernel's elements to threads, Group a workgroup, where
// nothing has bound them yet.
type Threads struct {
	Group int
}

func (Threads) Name() string { return "threads" }

func (p Threads) Apply(ks []*Kernel, d Device) ([]*Kernel, error) {
	return eachKernel(p.Name(), ks, func(k *Kernel) ([]*Kernel, bool, error) {
		if k.Group > 0 {
			return nil, false, nil
		}
		m := k.clone()
		m.Group = p.Group
		return []*Kernel{m}, true, nil
	})
}
