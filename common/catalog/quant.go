package catalog

import (
	"regexp"
	"sort"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
)

// quantTag is a publisher's quantization label in a file name: Q4_K_M, Q8_0,
// IQ3_XS, F16, BF16, int4.
var quantTag = regexp.MustCompile(`(?i)(?:^|[-_.])((?:IQ|Q)\d(?:_[A-Z0-9]+)*|BF16|F16|F32|INT[48])(?:[-_.]|$)`)

// QuantFromName is the quantization a file's name says, or "".
//
// The name first: "Q4_K_M" is a recipe (Q4_K for most weights, Q6_K for a
// few), and the container keeps only each tensor's own type.
func QuantFromName(name string) string {
	m := quantTag.FindStringSubmatch(strings.TrimSuffix(name, ".jlm"))
	if m == nil {
		return ""
	}
	return strings.ToUpper(m[1])
}

// familiar is the GGUF name of each container type, which is the vocabulary
// the rest of the world uses for them.
var familiar = map[jlm.Type]string{
	jlm.TypeF32: "F32", jlm.TypeF16: "F16", jlm.TypeBF16: "BF16",
	jlm.TypeQ4: "Q4_0", jlm.TypeQ8: "Q8_0", jlm.TypeQ5: "Q5_0", jlm.TypeQ51: "Q5_1", jlm.TypeMX4: "MXFP4",
	jlm.TypeQ3S: "Q3_K", jlm.TypeQ4S: "Q4_K", jlm.TypeQ5S: "Q5_K", jlm.TypeQ6S: "Q6_K",
	jlm.TypeTQ2: "TQ2_0", jlm.TypeTQ2F: "TQ2_F32",
}

// QuantFromTensors is the container's dominant weight type by element count,
// naming a second when it holds a fifth or more: "Q4_K" or "Q4_K + Q6_K".
func QuantFromTensors(es []jlm.Entry) string {
	count := map[jlm.Type]uint64{}
	for _, e := range es {
		if e.NDim < 2 {
			continue // norms and biases: a rounding error, and never quantized
		}
		count[e.Type] += uint64(e.Rows()) * uint64(e.K())
	}
	type tc struct {
		t jlm.Type
		n uint64
	}
	var all []tc
	var total uint64
	for t, n := range count {
		all = append(all, tc{t, n})
		total += n
	}
	if total == 0 {
		return ""
	}
	sort.Slice(all, func(i, j int) bool { return all[i].n > all[j].n || all[i].n == all[j].n && all[i].t < all[j].t })
	name := func(t jlm.Type) string {
		if s, ok := familiar[t]; ok {
			return s
		}
		return t.String()
	}
	out := name(all[0].t)
	if len(all) > 1 && all[1].n*5 >= total {
		out += " + " + name(all[1].t)
	}
	return out
}
