package backend

import "testing"

// TestDevicesImplementTheirInterfaces holds every device to the optional
// interfaces it is reached through by type assertion, where a misspelled
// method would otherwise fail silently at run time. It is checked when the
// test compiles.
func TestDevicesImplementTheirInterfaces(t *testing.T) {
	var (
		_ HostImport       = (*cudaDev)(nil)
		_ Unified          = (*cudaDev)(nil)
		_ ResourceReporter = (*cudaDev)(nil)
		_ Identified       = (*cudaDev)(nil)
		_ KernelClock      = (*cudaDev)(nil)

		_ BufferLimited = (*vkDev)(nil)
		_ HostImport    = (*vkDev)(nil)
		_ Identified    = (*vkDev)(nil)

		_ BufferLimited = (*mtlDev)(nil)
		_ HostImport    = (*mtlDev)(nil)
		_ Unified       = (*mtlDev)(nil)
		_ Identified    = (*mtlDev)(nil)
		_ KernelClock   = (*mtlDev)(nil)

		_ Queued        = (*hipDev)(nil)
		_ Identified    = (*hipDev)(nil)
		_ Unified       = (*hipDev)(nil)
		_ Lanes         = (*hipDev)(nil)
		_ OrdinalDevice = (*hipDev)(nil)
	)
}
