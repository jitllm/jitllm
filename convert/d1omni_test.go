package convert

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestD1OmniConfigIsTheTrunks reads d1-omni's own GGUF: the trunk is sixteen
// LFM2 blocks made an encoder, its FFN width the trunk's (the head's 4096 is
// not it), its layer kinds read off the trunk's head counts alone, and the
// head's two blocks after it. The same file with attention.causal true is
// refused: it is not the graph ArchLFM2Enc runs.
func TestD1OmniConfigIsTheTrunks(t *testing.T) {
	src := testmodels.Path("d1-omni/d1-omni-600M-F16.gguf")
	if _, err := os.Stat(src); err != nil {
		testmodels.Missing(t, "%s (LiquidAI/d1-omni-600M's GGUF)", src)
	}
	f, err := gguf.Open(src)
	if err != nil {
		t.Fatal(err)
	}
	c, err := configOf(f)
	if err != nil {
		t.Fatal(err)
	}
	if c.Arch != jlm.ArchLFM2Enc || !c.Flags.Has(jlm.FlagNonCausal) || c.NLayer != 16 || c.DecisionBlocks != 2 ||
		c.NFFN != 4608 || c.NKVHead != 8 || c.Decision != jlm.DecisionD1Omni || c.DecisionHeads != 16 {
		t.Fatalf("arch %v flags %v layers %d+%d ffn %d kv heads %d readout %v head heads %d",
			c.Arch, c.Flags, c.NLayer, c.DecisionBlocks, c.NFFN, c.NKVHead, c.Decision, c.DecisionHeads)
	}
	attn := 0
	for _, k := range c.LayerKinds {
		if k == jlm.LayerFullAttn {
			attn++
		}
	}
	if attn != 6 || len(c.LayerKinds) != 16 {
		t.Errorf("%d attention blocks in the trunk, the checkpoint's layer_types has 6", attn)
	}
	k := f.Arch() + ".attention.causal"
	v := f.KV[k]
	v.Raw = []byte{1}
	f.KV[k] = v
	if _, err := configOf(f); err == nil {
		t.Errorf("a causal d1-omni trunk converted")
	}
}
