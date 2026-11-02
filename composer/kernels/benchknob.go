//go:build jitllmbench

package kernels

// benchKnobs select an alternative emission of a kernel for a same-pass A/B:
// the form a change replaced, or a value a sweep varies. Compiled in only
// under the jitllmbench tag, so no release build can generate anything but
// the shipping form; a knob at 0 is the shipping form.
//
//	"flash-interleaved"  FlashAttention's contiguous V accumulate as it was
//	                     before grouping: each load beside its FMA
//	"flash-vgroup"       FlashAttention's keys a V load group (0: 64/dims)
//	"kv-constdesc"       paged FlashDecodeKV with its one row's descriptor
//	                     baked: table at 0, keys [0, n)
var benchKnobs = map[string]int{}

// SetBenchKnob sets one knob for the kernels generated after it; 0 clears it.
func SetBenchKnob(name string, v int) {
	if v == 0 {
		delete(benchKnobs, name)
		return
	}
	benchKnobs[name] = v
}

func benchKnob(name string) int { return benchKnobs[name] }
