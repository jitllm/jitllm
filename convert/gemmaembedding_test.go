package convert

import (
	"maps"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestTheGemmaEmbeddingLabelIsStated holds what a file under the
// "gemma-embedding" label leaves to its label to what the library states for
// it: every position attends to every position with no attention.causal key,
// the readout is a mean with no pooling_type, the score's scale is
// 1/sqrt(key_length) whatever attention.scale says, and the window's period
// is six with no pattern. The same file under the gemma3 label is causal,
// which is what makes the statement the thing under test. The widths a file
// states for its dense heads are held to the heads' tensors.
func TestTheGemmaEmbeddingLabelIsStated(t *testing.T) {
	path := testmodels.Path("embed/synth-gemma-embedding.gguf")
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (scripts/embedfixture.py) -- this gate proved nothing", err)
	}
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if f.Arch() != "gemma-embedding" {
		t.Fatalf("%s is %q", path, f.Arch())
	}
	const a = "gemma-embedding."
	conv := func(kv map[string]meta.Value, ts []meta.Tensor) (*Config, error) {
		return configOf(meta.New(f.Path, kv, ts))
	}
	for _, k := range []string{"attention.causal", "attention.sliding_window_pattern", "attention.scale"} {
		if _, ok := f.KV[a+k]; ok {
			t.Fatalf("the fixture states %s: it cannot show what the label states", k)
		}
	}

	c, err := conv(f.KV, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	if !c.Flags.Has(jlm.FlagNonCausal) || c.Flags.Pooling() != jlm.PoolMean || c.AttnScale != 0 ||
		c.SWAWindow != 8 || c.SWAPeriod != 6 || c.RopeBaseSWA != 10000 {
		t.Errorf("stated: non-causal %v pooling %v scale %g window %d period %d swa base %g, want true mean 0 8 6 10000",
			c.Flags.Has(jlm.FlagNonCausal), c.Flags.Pooling(), c.AttnScale, c.SWAWindow, c.SWAPeriod, c.RopeBaseSWA)
	}

	holds := func(what string, kv map[string]meta.Value, ok func(*Config) bool) {
		t.Helper()
		c, err := conv(kv, f.Tensors)
		switch {
		case err != nil:
			t.Errorf("%s: %v", what, err)
		case !ok(c):
			t.Errorf("%s: non-causal %v pooling %v scale %g", what, c.Flags.Has(jlm.FlagNonCausal), c.Flags.Pooling(), c.AttnScale)
		}
	}
	kv := maps.Clone(f.KV)
	delete(kv, a+"pooling_type")
	holds("no pooling_type, want the label's mean", kv, func(c *Config) bool { return c.Flags.Pooling() == jlm.PoolMean })
	kv = maps.Clone(f.KV)
	kv[a+"pooling_type"] = meta.MakeUint(3)
	holds("pooling_type 3, want the file's last position", kv, func(c *Config) bool { return c.Flags.Pooling() == jlm.PoolLast })
	kv = maps.Clone(f.KV)
	kv[a+"attention.scale"] = meta.MakeFloat(0.5)
	holds("attention.scale 0.5, want 1/sqrt(key_length) (0)", kv, func(c *Config) bool { return c.AttnScale == 0 })

	kv = map[string]meta.Value{}
	for k, v := range f.KV {
		kv[strings.Replace(k, a, "gemma3.", 1)] = v
	}
	kv["general.architecture"] = meta.MakeString("gemma3")
	holds("the same file as gemma3, want causal: the label states what the other does not", kv,
		func(c *Config) bool { return !c.Flags.Has(jlm.FlagNonCausal) })

	kv = maps.Clone(f.KV)
	kv[a+"dense_2_feat_out"] = meta.MakeUint(95)
	if _, err := conv(kv, f.Tensors); err == nil || !strings.Contains(err.Error(), "dense_2_feat_out") {
		t.Errorf("a dense_2_feat_out its tensor contradicts: %v, want the refusal", err)
	}
	var heads []meta.Tensor
	for _, x := range f.Tensors {
		if x.Name != "dense_3.weight" {
			heads = append(heads, x)
		}
	}
	if _, err := conv(f.KV, heads); err == nil || !strings.Contains(err.Error(), "dense_3") {
		t.Errorf("dense_3's widths without dense_3.weight: %v, want the refusal", err)
	}
	kv = maps.Clone(f.KV)
	for _, k := range []string{"dense_2_feat_in", "dense_2_feat_out", "dense_3_feat_in", "dense_3_feat_out"} {
		delete(kv, a+k)
	}
	if _, err := conv(kv, f.Tensors); err != nil {
		t.Errorf("dense heads with no widths stated: %v", err)
	}
}
