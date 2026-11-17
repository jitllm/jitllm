package convert

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestLayaReadoutNeedsAnEncoder holds the laya readout's refusal to what it
// reads, whatever the architecture: an encoder's rows (a non-causal model)
// and head blocks after them. A causal trunk and a file with no head blocks
// are the violations; an encoder with head blocks states the marker head.
func TestLayaReadoutNeedsAnEncoder(t *testing.T) {
	file := func(blocks uint64) *meta.File {
		k := baseKeys(4)
		k["decision.type"] = meta.MakeString("laya")
		k["decision.block_count"] = meta.MakeUint(blocks)
		k["decision.max_head_tokens"] = meta.MakeUint(512)
		f := headerOnly("bert", k)
		f.KV["tokenizer.chat_template.systemone"] = meta.MakeString("{{ question }}")
		return f
	}
	for _, tc := range []struct {
		name    string
		flags   jlm.Flags
		blocks  uint64
		refusal string
	}{
		{"causal", 0, 2, "causal"},
		{"no head blocks", jlm.FlagNonCausal, 0, "states none"},
		{"encoder", jlm.FlagNonCausal, 2, ""},
	} {
		c := &Config{Config: &jlm.Config{}}
		c.NEmbd, c.Flags = 64, tc.flags
		err := decisionOf(file(tc.blocks), c)
		switch {
		case tc.refusal == "" && err != nil:
			t.Errorf("%s: %v", tc.name, err)
		case tc.refusal == "" && !c.Flags2.Has(jlm.Flag2HeadBlocks):
			t.Errorf("%s: converted with no marker head stated", tc.name)
		case tc.refusal != "" && (err == nil || !strings.Contains(err.Error(), tc.refusal)):
			t.Errorf("%s: got %v, want a refusal naming %q", tc.name, err, tc.refusal)
		}
	}
}
