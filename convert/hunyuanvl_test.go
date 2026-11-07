package convert

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestHunyuanVLTurnsTheWholeHead: transformers' XD-RoPE turns a whole head
// with NEOX pairs, and the converter states it so -- a file whose rotary
// covers part of the head is refused here, where there is a file in front of
// a person, rather than run as a rotary the reference never runs.
func TestHunyuanVLTurnsTheWholeHead(t *testing.T) {
	file := func(nrot uint64, secs []int32) *meta.File {
		k := baseKeys(2)
		k["rope.dimension_sections"] = meta.MakeInt32s(secs)
		if nrot != 0 {
			k["rope.dimension_count"] = meta.MakeUint(nrot)
		}
		return headerOnly("hunyuan_vl", k)
	}
	c, err := configOf(file(0, []int32{2, 2, 2, 2}))
	if err != nil {
		t.Fatal(err)
	}
	if c.Arch != jlm.ArchHunyuanVL || c.NRot != c.HeadDim || !c.Flags.Has(jlm.FlagRopeNeox) {
		t.Fatalf("arch %v, %d of %d dimensions turned, NEOX %v", c.Arch, c.NRot, c.HeadDim, c.Flags.Has(jlm.FlagRopeNeox))
	}
	if _, err := configOf(file(8, []int32{1, 1, 1, 1})); err == nil || !strings.Contains(err.Error(), "the reference turns the whole head") {
		t.Fatalf("a rotary over half the head converted (%v)", err)
	}
}
