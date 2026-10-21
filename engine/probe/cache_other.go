//go:build !linux && !darwin

package probe

import (
	"fmt"
	"runtime"
)

// cacheSizes is the data caches; only Linux and Darwin say them yet.
func cacheSizes() ([]cache, error) {
	return nil, fmt.Errorf("probe: no cache sizes on %s", runtime.GOOS)
}
