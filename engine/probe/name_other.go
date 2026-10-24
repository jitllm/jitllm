//go:build !linux && !darwin

package probe

import (
	"fmt"
	"runtime"
)

// HostName is the host CPU's model; only Linux and Darwin say it yet.
func HostName() (string, error) {
	return "", fmt.Errorf("probe: no host name on %s", runtime.GOOS)
}
