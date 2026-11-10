package convert

import (
	"errors"
	"math"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/quant"
)

// headerOnly is a GGUF with keys and one embedding tensor, which is all
// configOf reads.
func headerOnly(arch string, keys map[string]meta.Value) *meta.File {
	kv := map[string]meta.Value{"general.architecture": meta.MakeString(arch)}
	for k, v := range keys {
		kv[arch+"."+k] = v
	}
	ts := []meta.Tensor{{Name: "token_embd.weight", Dims: []uint64{64, 100}, Type: quant.F32, NBytes: 4 * 64 * 100}}
	return meta.New("synth-"+arch+".gguf", kv, ts)
}

func baseKeys(layers uint64) map[string]meta.Value {
	return map[string]meta.Value{
		"block_count":          meta.MakeUint(layers),
		"embedding_length":     meta.MakeUint(64),
		"feed_forward_length":  meta.MakeUint(128),
		"attention.head_count": meta.MakeUint(4),
	}
}

func uint32s(n int, v int32) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// TestGraniteCarriesItsFourScales is the converter half of the Granite gate:
// the four constants reach the container, a per-layer head_count_kv array reads
// as its value rather than the MHA default, and a dense Granite's shared-expert
// width (granite-4.0-micro carries one beside expert_count 0) is dropped. The
// model half is TestGreedyMatchesLlamaCpp on the real files.
func TestGraniteCarriesItsFourScales(t *testing.T) {
	k := baseKeys(4)
	k["attention.head_count_kv"] = meta.MakeInt32s(uint32s(4, 2))
	k["attention.scale"] = meta.MakeFloat(0.015625)
	k["embedding_scale"] = meta.MakeFloat(12)
	k["residual_scale"] = meta.MakeFloat(0.22)
	k["logit_scale"] = meta.MakeFloat(8)
	k["expert_count"] = meta.MakeUint(0)
	k["expert_shared_feed_forward_length"] = meta.MakeUint(8192)
	c, err := configOf(headerOnly("granite", k))
	if err != nil {
		t.Fatal(err)
	}
	if c.Arch != ArchGranite {
		t.Errorf("arch %v, want granite", c.Arch)
	}
	if c.NKVHead != 2 {
		t.Errorf("head_count_kv array read as %d, want 2", c.NKVHead)
	}
	if c.AttnScale != 0.015625 || c.EmbdScale != 12 || c.ResidualScale != 0.22 || c.LogitScale != 8 {
		t.Errorf("scales attn %g embd %g resid %g logit %g", c.AttnScale, c.EmbdScale, c.ResidualScale, c.LogitScale)
	}
	if c.NFFNShExp != 0 {
		t.Errorf("a dense model kept a shared-expert width of %d", c.NFFNShExp)
	}

	// The mixture name is the same code, and routes by the file.
	k["expert_count"] = meta.MakeUint(8)
	k["expert_used_count"] = meta.MakeUint(2)
	k["expert_feed_forward_length"] = meta.MakeUint(32)
	delete(k, "expert_shared_feed_forward_length")
	if c, err = configOf(headerOnly("granitemoe", k)); err != nil || c.Arch != ArchGranite || c.NExpert != 8 {
		t.Errorf("granitemoe: %v %+v", err, c)
	}

	// A head count that differs by layer is a geometry the container cannot
	// say, and no logit_scale is not a Granite.
	k["attention.head_count_kv"] = meta.MakeInt32s([]int32{2, 2, 4, 2})
	if _, err := configOf(headerOnly("granite", k)); !errors.Is(err, ErrNotImplemented) {
		t.Errorf("a per-layer head_count_kv converted: %v", err)
	}
	k["attention.head_count_kv"] = meta.MakeUint(2)
	delete(k, "logit_scale")
	if _, err := configOf(headerOnly("granite", k)); err == nil {
		t.Error("a Granite with no logit_scale converted")
	}
}

// TestGemmaWindowAndScaleFollowTheReference covers the gemma2/gemma3 keys: no
// window key on gemma3 means no sliding layers, the pattern key overrides the
// period in either spelling, gemma3's final softcap is read, gemma2's absent
// window is its 4096, and the 27B's score scale is 1/sqrt(n_embd/n_head).
func TestGemmaWindowAndScaleFollowTheReference(t *testing.T) {
	c, err := configOf(headerOnly("gemma3", baseKeys(6)))
	if err != nil {
		t.Fatal(err)
	}
	if c.SWAWindow != 0 || c.SWAPeriod != 0 {
		t.Errorf("a gemma3 with no window key slides: window %d period %d", c.SWAWindow, c.SWAPeriod)
	}
	k := baseKeys(6)
	k["attention.sliding_window"] = meta.MakeUint(512)
	k["final_logit_softcapping"] = meta.MakeFloat(30)
	if c, err = configOf(headerOnly("gemma3", k)); err != nil || c.SWAWindow != 512 || c.SWAPeriod != 6 ||
		c.RopeBaseSWA != 10000 || c.FinalSoftcap != 30 {
		t.Errorf("gemma3 window: %v %+v", err, c)
	}
	k["attention.sliding_window_pattern"] = meta.MakeUint(4)
	k["rope.freq_base_swa"] = meta.MakeFloat(20000)
	if c, err = configOf(headerOnly("gemma3", k)); err != nil || c.SWAPeriod != 4 || c.RopeBaseSWA != 20000 {
		t.Errorf("gemma3 scalar pattern: %v %+v", err, c)
	}
	bools := func(v ...byte) meta.Value {
		return meta.Value{Type: meta.Array, Elem: meta.Bool, N: uint64(len(v)), Raw: v}
	}
	k["attention.sliding_window_pattern"] = bools(1, 1, 0, 1, 1, 0)
	if c, err = configOf(headerOnly("gemma3", k)); err != nil || c.SWAPeriod != 3 {
		t.Errorf("gemma3 array pattern: %v %+v", err, c)
	}
	k["attention.sliding_window_pattern"] = bools(1, 0, 0, 1, 1, 0)
	if _, err = configOf(headerOnly("gemma3", k)); !errors.Is(err, ErrNotImplemented) {
		t.Errorf("an aperiodic pattern converted: %v", err)
	}

	if c, err = configOf(headerOnly("gemma2", baseKeys(4))); err != nil || c.SWAWindow != 4096 || c.SWAPeriod != 2 {
		t.Errorf("gemma2 with no window key: %v %+v", err, c)
	}

	// The 27Bs: 32 heads over 4608 and 5376, heads 128 wide.
	for _, tc := range []struct {
		arch         string
		layers, embd uint64
	}{{"gemma2", 46, 4608}, {"gemma3", 62, 5376}} {
		k := baseKeys(tc.layers)
		k["embedding_length"] = meta.MakeUint(tc.embd)
		k["attention.head_count"] = meta.MakeUint(32)
		k["attention.key_length"] = meta.MakeUint(128)
		c, err := configOf(headerOnly(tc.arch, k))
		want := float32(1 / math.Sqrt(float64(tc.embd/32)))
		if err != nil || c.AttnScale != want {
			t.Errorf("%s 27B: %v, score scale %g want %g", tc.arch, err, c.AttnScale, want)
		}
	}
	if c, err = configOf(headerOnly("gemma3", baseKeys(26))); err != nil || c.AttnScale != 0 {
		t.Errorf("gemma3 1B took a score scale: %v %g", err, c.AttnScale)
	}
}
