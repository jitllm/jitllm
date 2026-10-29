//go:build jitllmfault

package jit

import "slices"

// FlashOneProduct makes r's GPU flash form stage no remainders: one binary16
// product a block of dots and one of weighted sums.
func (r *Registry) FlashOneProduct() {
	p := r.Profiles[DeviceGPU]
	p.Passes = slices.Clone(p.Passes)
	for i, ps := range p.Passes {
		if f, ok := ps.(Flash); ok {
			f.one = true
			p.Passes[i] = f
		}
	}
	r.Profiles[DeviceGPU] = p
}
