package cpu

import (
	"fmt"
	"strings"
)

// SpecOptions is what a cpu device spec's variants state about its Target
// (cpu:VARIANT[+VARIANT...]): sse its tier (AtTier(TierSSE)), novnni and
// nodotprod a feature taken away (WithoutVNNI, WithoutDotProd), perfmap the
// perf map (WithPerfMap). An empty selector states nothing.
func SpecOptions(sel string) ([]Option, error) {
	if sel == "" {
		return nil, nil
	}
	var o []Option
	for _, v := range strings.Split(sel, "+") {
		switch v {
		case "sse":
			o = append(o, AtTier(TierSSE))
		case "novnni":
			o = append(o, WithoutVNNI())
		case "nodotprod":
			o = append(o, WithoutDotProd())
		case "perfmap":
			o = append(o, WithPerfMap())
		default:
			return nil, fmt.Errorf("cpu:%s: %q is no variant of the CPU device; want sse, novnni, nodotprod or perfmap, joined by +", sel, v)
		}
	}
	return o, nil
}
