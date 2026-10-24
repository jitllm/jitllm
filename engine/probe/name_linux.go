//go:build linux

package probe

import (
	"fmt"
	"os"
	"strings"
)

// HostName is the host CPU's model, as Linux states it.
func HostName() (string, error) {
	b, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return "", err
	}
	for _, line := range strings.Split(string(b), "\n") {
		if k, v, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(k) == "model name" {
			return strings.TrimSpace(v), nil
		}
	}
	return "", fmt.Errorf("probe: /proc/cpuinfo names no model")
}
