package probe

import (
	"strings"

	"github.com/jitllm/jitllm/backend"
)

// GPUName is device d's hardware model, as the hardware table names it: the
// backend's name without the ordinal before it ("#0 ") or the architecture
// after it (" (sm_70)").
func GPUName(d backend.Device) string {
	name := strings.TrimSpace(d.Name())
	if rest, ok := strings.CutPrefix(name, "#"); ok {
		if _, after, ok := strings.Cut(rest, " "); ok {
			name = after
		}
	}
	if i := strings.LastIndex(name, " ("); i > 0 && strings.HasSuffix(name, ")") {
		name = name[:i]
	}
	return strings.TrimSpace(name)
}
