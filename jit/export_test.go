package jit

// What the external tests, which compose a model the engine opens, reach of
// the package's internals.

func (r *Registry) Decompose(f Model, p Part, i int, s Step, d Device) ([]Prim, error) {
	return r.decompose(f, p, i, s, d)
}

func (k *Kernel) TopOrder() []int { return k.topOrder() }

var (
	Generate             = generate
	EntryName            = entryName
	GPULowering Lowering = gpuLowering{}
)
