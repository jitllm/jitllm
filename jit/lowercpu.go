package jit

import "fmt"

// cpuLowering lowers a kernel to x86 through jit/x86.
type cpuLowering struct{}

func (cpuLowering) Lower(n *Kernel, d Device) (Lowered, error) {
	return Lowered{}, fmt.Errorf("the x86 lowering of %s is not written", n.Name)
}
