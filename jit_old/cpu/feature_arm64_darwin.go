//go:build arm64 && darwin

package cpu

import "syscall"

// probeDotProd asks the kernel (sysctl), which is how Darwin reports optional
// ARM features. Every Apple silicon chip has FEAT_DotProd; asking means a VM
// that hides it degrades instead of crashing. The syscall is tagged, the
// decision is not.
func probeDotProd() bool {
	v, err := syscall.SysctlUint32("hw.optional.arm.FEAT_DotProd")
	return err == nil && v != 0
}
