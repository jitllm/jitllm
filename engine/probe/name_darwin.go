//go:build darwin

package probe

import "syscall"

// HostName is the host CPU's model, as Darwin states it.
func HostName() (string, error) { return syscall.Sysctl("machdep.cpu.brand_string") }
