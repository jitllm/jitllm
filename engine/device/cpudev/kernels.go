package cpudev

import (
	"slices"
	"strings"
)

// Kernel is one composed kernel's register facts on the CPU, the worst over
// the forms holding it, and how many do.
type Kernel struct {
	Name  string `json:"name"`
	Vec   int    `json:"vec_used"`
	File  int    `json:"file"`
	Stack int    `json:"stack_refs"`
	Bytes int    `json:"bytes"`
	Forms int    `json:"forms"`
}

// Kernels is the register facts of every kernel model's composed forms on c
// hold, by name.
func (c *CPU) Kernels(model uint64) []Kernel {
	by := map[string]*Kernel{}
	for _, f := range c.Forms(model).All() {
		for i := range f.Codes {
			cc := &f.Codes[i]
			r := cc.Regs()
			k := by[cc.Name()]
			if k == nil {
				k = &Kernel{Name: cc.Name(), Vec: r.Vec, File: r.File}
				by[cc.Name()] = k
			}
			k.Vec = max(k.Vec, r.Vec)
			k.Stack = max(k.Stack, r.Stack)
			k.Bytes = max(k.Bytes, cc.Size())
			k.Forms++
		}
	}
	out := make([]Kernel, 0, len(by))
	for _, k := range by {
		out = append(out, *k)
	}
	slices.SortFunc(out, func(a, b Kernel) int { return strings.Compare(a.Name, b.Name) })
	return out
}
