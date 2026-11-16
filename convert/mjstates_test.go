package convert

import (
	"errors"
	"maps"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestTheLibraryStatesTheContainer holds each GGUF architecture whose
// container the library states to what configOfFile set for it by name
// before the files did: the flags, the activation, the norm epsilon read
// off its key with its default, the embedding scale off the width, the
// window and its pattern, the windowed layers' rotary base and the caps.
func TestTheLibraryStatesTheContainer(t *testing.T) {
	neox, qk := jlm.FlagRopeNeox, jlm.FlagRopeNeox|jlm.FlagQKNorm
	noSections := map[string]meta.Value{"rope.dimension_sections": meta.MakeInt32s([]int32{})}
	ln := jlm.FlagLayerNorm | jlm.FlagRopeNeox | jlm.FlagGELU
	gemma := map[string]meta.Value{"embedding_length": meta.MakeUint(2048)}
	with := func(m map[string]meta.Value, k string, v meta.Value) map[string]meta.Value {
		out := map[string]meta.Value{k: v}
		for k, v := range m {
			out[k] = v
		}
		return out
	}
	for _, tc := range []struct {
		arch        string
		keys        map[string]meta.Value
		flags       jlm.Flags
		eps, scale  float32
		win, period uint32
		ropeSWA     float32
		acap, fcap  float32
		logit       float32
		actq        jlm.ActQuant
	}{
		{arch: "qwen2", flags: neox},
		{arch: "qwen2vl", keys: noSections, flags: neox},
		{arch: "qwen3", flags: qk},
		{arch: "qwen3vl", keys: noSections, flags: qk},
		{arch: "qwen3moe", flags: qk},
		{arch: "qwen3vlmoe", keys: noSections, flags: qk},
		{arch: "bert", flags: jlm.FlagLayerNorm | jlm.FlagNonCausal | jlm.FlagNoPosEnc | jlm.FlagGELU, eps: 1e-12},
		{arch: "nomic-bert", flags: jlm.FlagLayerNorm | jlm.FlagRopeNeox | jlm.FlagNonCausal, eps: 1e-12},
		{arch: "modern-bert", flags: jlm.FlagNonCausal | jlm.FlagRopeNeox | jlm.FlagLayerNorm, eps: 1e-5},
		{arch: "olmoe", flags: qk | jlm.FlagQKNormWide | jlm.FlagNoExpertNorm},
		{arch: "bitnet-b1.58", flags: neox | jlm.FlagReLU2, actq: jlm.ActQuantRowInt8},
		{arch: "gemma", keys: map[string]meta.Value{"embedding_length": meta.MakeUint(2048)},
			flags: neox | jlm.FlagGELU, scale: float32(math.Sqrt(2048))},
		{arch: "starcoder2", flags: ln, eps: 1e-5},
		{arch: "starcoder2", keys: map[string]meta.Value{"attention.layer_norm_epsilon": meta.MakeFloat(1e-6)},
			flags: ln, eps: 1e-6},
		{arch: "falcon", keys: map[string]meta.Value{"attention.layer_norm_epsilon": meta.MakeFloat(3e-7)},
			flags: ln | jlm.FlagParallel, eps: 3e-7},
		{arch: "nemotron", flags: jlm.FlagLayerNorm | neox | jlm.FlagReLU2, eps: 1e-5},
		{arch: "phi2", flags: ln | jlm.FlagParallel, eps: 1e-5},
		{arch: "command-r", keys: map[string]meta.Value{"logit_scale": meta.MakeFloat(0.0625)},
			flags: jlm.FlagLayerNorm | jlm.FlagParallel, eps: 1e-5, logit: 16},
		{arch: "command-r", keys: map[string]meta.Value{"logit_scale": meta.MakeFloat(1)},
			flags: jlm.FlagLayerNorm | jlm.FlagParallel, eps: 1e-5},
		{arch: "cohere2", keys: map[string]meta.Value{"attention.sliding_window": meta.MakeUint(4096), "logit_scale": meta.MakeFloat(0.25)},
			flags: jlm.FlagLayerNorm | jlm.FlagParallel | jlm.FlagNoPEGlobal, eps: 1e-5, logit: 4, win: 4096, period: 4},
		{arch: "cohere2", keys: map[string]meta.Value{"attention.sliding_window": meta.MakeUint(4096), "rope.freq_base_swa": meta.MakeFloat(1e4),
			"rope.freq_base": meta.MakeFloat(5e4)},
			flags: jlm.FlagLayerNorm | jlm.FlagParallel | jlm.FlagNoPEGlobal, eps: 1e-5, win: 4096, period: 4, ropeSWA: 1e4},
		{arch: "gemma2", keys: gemma, flags: neox | jlm.FlagGELU, scale: float32(math.Sqrt(2048)),
			win: 4096, period: 2, ropeSWA: 10000},
		{arch: "gemma2", keys: with(with(with(gemma, "rope.freq_base", meta.MakeFloat(5e5)), "attn_logit_softcapping",
			meta.MakeFloat(50)), "final_logit_softcapping", meta.MakeFloat(30)),
			flags: neox | jlm.FlagGELU, scale: float32(math.Sqrt(2048)), win: 4096, period: 2, ropeSWA: 5e5, acap: 50, fcap: 30},
		{arch: "gemma3", keys: gemma, flags: qk | jlm.FlagGELU, scale: float32(math.Sqrt(2048))},
		{arch: "gemma3", keys: with(with(gemma, "attention.sliding_window", meta.MakeUint(512)),
			"attention.sliding_window_pattern", meta.MakeUint(4)),
			flags: qk | jlm.FlagGELU, scale: float32(math.Sqrt(2048)), win: 512, period: 4, ropeSWA: 10000},
		{arch: "deepseek4", keys: map[string]meta.Value{"expert_count": meta.MakeUint(4), "expert_shared_count": meta.MakeUint(1), "attention.compress_rope_freq_base": meta.MakeFloat(16e4), "hyper_connection.count": meta.MakeUint(4), "hyper_connection.sinkhorn_iterations": meta.MakeUint(20), "hyper_connection.epsilon": meta.MakeFloat(1e-6), "attention.output_group_count": meta.MakeUint(2), "attention.output_lora_rank": meta.MakeUint(8), "attention.q_lora_rank": meta.MakeUint(96), "attention.sliding_window": meta.MakeUint(128), "block_count": meta.MakeUint(4)},
			win: 128, period: 5, ropeSWA: 10000},
		{arch: "gpt-oss", keys: map[string]meta.Value{"attention.sliding_window": meta.MakeUint(128)},
			flags: neox | jlm.FlagSwiGLUOAI | jlm.FlagYarnExact, win: 128, period: 2},
		{arch: "phi3", keys: map[string]meta.Value{"attention.sliding_window": meta.MakeUint(2047), "block_count": meta.MakeUint(32)},
			flags: neox, win: 2047, period: 33},
		{arch: "phi3", keys: map[string]meta.Value{"attention.sliding_window": meta.MakeUint(0), "block_count": meta.MakeUint(32)},
			flags: neox},
		{arch: "smollm3", flags: jlm.FlagNoPEGlobal, period: 4},
		{arch: "seed_oss", flags: neox},
		{arch: "arcee", flags: jlm.FlagReLU2},
		{arch: "olmo2", flags: qk | jlm.FlagQKNormWide},
		{arch: "olmo2", keys: map[string]meta.Value{"attention.sliding_window": meta.MakeUint(4096), "rope.freq_base": meta.MakeFloat(5e5)},
			flags: qk | jlm.FlagQKNormWide, win: 4096, period: 4, ropeSWA: 5e5},
		{arch: "exaone4", keys: map[string]meta.Value{"attention.sliding_window": meta.MakeUint(0)}, flags: qk},
		{arch: "exaone4", keys: map[string]meta.Value{"attention.sliding_window": meta.MakeUint(4096), "rope.freq_base": meta.MakeFloat(1e6)},
			flags: qk | jlm.FlagNoPEGlobal, win: 4096, period: 4},
		{arch: "exaone4", keys: map[string]meta.Value{"attention.sliding_window": meta.MakeUint(4096), "rope.freq_base_swa": meta.MakeFloat(1e4),
			"rope.freq_base": meta.MakeFloat(1e6)},
			flags: qk | jlm.FlagNoPEGlobal, win: 4096, period: 4, ropeSWA: 1e4},
		{arch: "mistral3"},
	} {
		kv := map[string]meta.Value{"general.architecture": meta.MakeString(tc.arch)}
		for k, v := range tc.keys {
			kv[tc.arch+"."+k] = v
		}
		c := &Config{Config: &jlm.Config{}}
		if err := stateContainer(meta.New(tc.arch+".gguf", kv, nil), c); err != nil {
			t.Fatalf("%s: %v", tc.arch, err)
		}
		if c.Flags != tc.flags || c.RMSEps != tc.eps || c.EmbdScale != tc.scale {
			t.Errorf("%s: flags %v, eps %g, scale %g; want %v, %g, %g", tc.arch, c.Flags, c.RMSEps, c.EmbdScale,
				tc.flags, tc.eps, tc.scale)
		}
		if c.LogitScale != tc.logit {
			t.Errorf("%s: logits divided by %g, want %g", tc.arch, c.LogitScale, tc.logit)
		}
		if c.Stated.actQuant != tc.actq {
			t.Errorf("%s: activation quantization %v, want %v", tc.arch, c.Stated.actQuant, tc.actq)
		}
		if c.SWAWindow != tc.win || c.SWAPeriod != tc.period || c.RopeBaseSWA != tc.ropeSWA ||
			c.AttnSoftcap != tc.acap || c.FinalSoftcap != tc.fcap {
			t.Errorf("%s: window %d every %d at base %g, caps %g %g; want %d every %d at %g, %g %g", tc.arch,
				c.SWAWindow, c.SWAPeriod, c.RopeBaseSWA, c.AttnSoftcap, c.FinalSoftcap,
				tc.win, tc.period, tc.ropeSWA, tc.acap, tc.fcap)
		}
	}
	for arch, want := range map[string]string{"qwen3vl": "interleaved", "qwen3vlmoe": "interleaved",
		"qwen35": "interleaved", "qwen35moe": "interleaved", "qwen3next": "interleaved", "qwen4exp": "interleaved",
		"clef": "interleaved", "hunyuan_vl": "xd", "qwen2vl": "", "qwen3": ""} {
		if st, _, err := ggufStates(arch); err != nil || st.RopeLayout != want {
			t.Errorf("%s: rotary layout %q (%v), want %q", arch, st.RopeLayout, err, want)
		}
	}
	u := meta.MakeUint
	for _, tc := range []struct {
		arch               string
		keys               map[string]meta.Value
		kv, q, heads, w, k uint32
	}{
		{arch: "deepseek2", keys: map[string]meta.Value{"attention.value_length": u(512)}, kv: 512},
		{arch: "deepseek2", keys: map[string]meta.Value{"attention.kv_lora_rank": u(64), "attention.value_length": u(80),
			"attention.q_lora_rank": u(96)}, kv: 64, q: 96},
		{arch: "kimi-k3", keys: map[string]meta.Value{"attention.kv_lora_rank": u(64), "activation.situ_beta": meta.MakeFloat(3),
			"activation.situ_linear_beta": meta.MakeFloat(5), "attention.head_count": u(4), "kda.head_dim": u(16), "ssm.conv_kernel": u(4),
			"block_count": u(2), "attention.head_count_kv": meta.MakeInt32s([]int32{0, 4})}, kv: 64},
		{arch: "deepseek4", keys: map[string]meta.Value{"expert_count": meta.MakeUint(4), "expert_shared_count": meta.MakeUint(1), "attention.compress_rope_freq_base": meta.MakeFloat(16e4), "hyper_connection.count": meta.MakeUint(4), "hyper_connection.sinkhorn_iterations": meta.MakeUint(20), "hyper_connection.epsilon": meta.MakeFloat(1e-6), "attention.output_group_count": meta.MakeUint(2), "attention.output_lora_rank": meta.MakeUint(8), "attention.q_lora_rank": u(96), "attention.sliding_window": u(128), "block_count": u(4)}, q: 96},
		{arch: "deepseek4", keys: map[string]meta.Value{"expert_count": meta.MakeUint(4), "expert_shared_count": meta.MakeUint(1), "attention.compress_rope_freq_base": meta.MakeFloat(16e4), "hyper_connection.count": meta.MakeUint(4), "hyper_connection.sinkhorn_iterations": meta.MakeUint(20), "hyper_connection.epsilon": meta.MakeFloat(1e-6), "attention.output_group_count": meta.MakeUint(2), "attention.output_lora_rank": meta.MakeUint(8), "attention.q_lora_rank": u(96), "attention.sliding_window": u(128), "block_count": u(4), "attention.indexer.head_count": u(4),
			"attention.indexer.key_length": u(32), "attention.indexer.top_k": u(8)}, q: 96, heads: 4, w: 32, k: 8},
		{arch: "deepseek32", keys: map[string]meta.Value{"attention.kv_lora_rank": u(64), "attention.q_lora_rank": u(96),
			"attention.indexer.head_count": u(4), "attention.indexer.key_length": u(32), "attention.indexer.top_k": u(8)},
			kv: 64, q: 96, heads: 4, w: 32, k: 8},
	} {
		kv := map[string]meta.Value{"general.architecture": meta.MakeString(tc.arch)}
		for k, v := range tc.keys {
			kv[tc.arch+"."+k] = v
		}
		f := meta.New(tc.arch+".gguf", kv, nil)
		c := &Config{Config: &jlm.Config{NLayer: uint32(f.UintKey("block_count", 0))}}
		if err := stateContainer(f, c); err != nil {
			t.Fatalf("%s: %v", tc.arch, err)
		}
		if c.KVLoraRank != tc.kv || c.QLoraRank != tc.q || c.IdxHeads != tc.heads || c.IdxHeadDim != tc.w || c.IdxTopK != tc.k {
			t.Errorf("%s %v: latents %d %d, indexer %d of %d top %d; want %d %d, %d of %d top %d", tc.arch, tc.keys,
				c.KVLoraRank, c.QLoraRank, c.IdxHeads, c.IdxHeadDim, c.IdxTopK, tc.kv, tc.q, tc.heads, tc.w, tc.k)
		}
	}
	ds32 := map[string]meta.Value{"attention.kv_lora_rank": u(64), "attention.q_lora_rank": u(96),
		"attention.indexer.head_count": u(4), "attention.indexer.key_length": u(64), "attention.indexer.top_k": u(8)}
	ds4 := map[string]meta.Value{"expert_count": meta.MakeUint(4), "expert_shared_count": meta.MakeUint(1), "attention.compress_rope_freq_base": meta.MakeFloat(16e4), "hyper_connection.count": meta.MakeUint(4), "hyper_connection.sinkhorn_iterations": meta.MakeUint(20), "hyper_connection.epsilon": meta.MakeFloat(1e-6), "attention.output_group_count": meta.MakeUint(2), "attention.output_lora_rank": meta.MakeUint(8), "attention.q_lora_rank": u(96), "attention.sliding_window": u(128), "block_count": u(4),
		"attention.indexer.head_count": u(4), "attention.indexer.key_length": u(64), "attention.indexer.top_k": u(8)}
	without := func(m map[string]meta.Value, k string) map[string]meta.Value {
		n := maps.Clone(m)
		delete(n, k)
		return n
	}
	set := func(m map[string]meta.Value, k string, v meta.Value) map[string]meta.Value {
		n := maps.Clone(m)
		n[k] = v
		return n
	}
	for _, tc := range []struct {
		arch, want string
		keys       map[string]meta.Value
	}{
		{"deepseek32", "attention.indexer.head_count", without(ds32, "attention.indexer.head_count")},
		{"deepseek32", "attention.indexer.top_k", without(ds32, "attention.indexer.top_k")},
		{"deepseek4", "an indexer of 0 heads", without(ds4, "attention.indexer.head_count")},
		{"deepseek32", "no key-value rank", map[string]meta.Value{"attention.q_lora_rank": u(96)}},
		{"deepseek32", "no query latent", without(ds32, "attention.q_lora_rank")},
		{"deepseek32", "not whole groups of eight", set(ds32, "attention.indexer.key_length", u(68))},
		{"deepseek4", "not whole groups of eight", set(ds4, "attention.indexer.key_length", u(68))},
		{"deepseek32", "with a 64-wide rotary", set(ds32, "attention.indexer.key_length", u(32))},
		{"deepseek4", "with a 64-wide rotary", set(ds4, "attention.indexer.key_length", u(32))},
		{"deepseek4", "attention.sliding_window", without(ds4, "attention.sliding_window")},
		{"deepseek4", "a window period over a window of 0", set(ds4, "attention.sliding_window", u(0))},
		{"deepseek4", "attention.compress_rope_freq_base", without(ds4, "attention.compress_rope_freq_base")},
		{"deepseek4", "rope_base is 0", set(ds4, "attention.compress_rope_freq_base", meta.MakeFloat(0))},
		{"deepseek4", "1 residual streams", set(ds4, "hyper_connection.count", u(1))},
		{"deepseek4", "0 residual streams", without(ds4, "hyper_connection.count")},
		{"deepseek4", "20 Sinkhorn rounds at 0", without(ds4, "hyper_connection.epsilon")},
		{"deepseek4", "0 Sinkhorn rounds", set(ds4, "hyper_connection.sinkhorn_iterations", u(0))},
		{"deepseek4", "3 output groups of rank 8 over 4 heads", set(ds4, "attention.output_group_count", u(3))},
		{"deepseek4", "2 output groups of rank 0", without(ds4, "attention.output_lora_rank")},
		{"deepseek4", "5 hash-routed blocks of 4", set(ds4, "hash_layer_count", u(5))},
	} {
		kv := map[string]meta.Value{"general.architecture": meta.MakeString(tc.arch)}
		for k, v := range tc.keys {
			kv[tc.arch+"."+k] = v
		}
		err := stateContainer(meta.New(tc.arch+".gguf", kv, nil), &Config{Config: &jlm.Config{NRot: 64, NHead: 4, NLayer: 4}})
		if err == nil || !strings.Contains(err.Error(), tc.want) || errors.Is(err, ErrNotImplemented) {
			t.Errorf("%s with %v: %v, want a malformed-file error naming %q", tc.arch, tc.keys, err, tc.want)
		}
	}
	k3 := map[string]meta.Value{"attention.kv_lora_rank": u(64), "activation.situ_beta": meta.MakeFloat(3),
		"activation.situ_linear_beta": meta.MakeFloat(5), "attn_res.block_size": u(2), "expert_latent_length": u(16),
		"attention.head_count": u(4), "kda.head_dim": u(16), "ssm.conv_kernel": u(4), "block_count": u(2),
		"attention.head_count_kv": meta.MakeInt32s([]int32{0, 4})}
	for _, tc := range []struct {
		want string
		keys map[string]meta.Value
	}{
		{"situ bounds 0 and 5", without(k3, "activation.situ_beta")},
		{"situ bounds 3 and -1", set(k3, "activation.situ_linear_beta", meta.MakeFloat(-1))},
	} {
		kv := map[string]meta.Value{"general.architecture": meta.MakeString("kimi-k3")}
		for k, v := range tc.keys {
			kv["kimi-k3."+k] = v
		}
		err := stateContainer(meta.New("kimi-k3.gguf", kv, nil), &Config{Config: &jlm.Config{NLayer: 2}})
		if err == nil || !strings.Contains(err.Error(), tc.want) || errors.Is(err, ErrNotImplemented) {
			t.Errorf("kimi-k3 with %v: %v, want a malformed-file error naming %q", tc.keys, err, tc.want)
		}
	}
	for _, width := range []uint32{32, 16} {
		kv := map[string]meta.Value{"general.architecture": meta.MakeString("kimi-k3")}
		for k, v := range k3 {
			kv["kimi-k3."+k] = v
		}
		c := &Config{Config: &jlm.Config{NEmbd: width, NLayer: 2}}
		if err := stateContainer(meta.New("kimi-k3.gguf", kv, nil), c); err != nil {
			t.Fatalf("kimi-k3: %v", err)
		}
		want := map[uint32]uint32{32: 16, 16: 0}[width]
		if c.AttnResBlock != 2 || c.ExpertLatent != want || c.SituBeta != 3 || c.SituLinearBeta != 5 {
			t.Errorf("kimi-k3 at width %d: residual-attention block %d, expert latent %d, situ %g %g; want 2, %d, 3 5",
				width, c.AttnResBlock, c.ExpertLatent, c.SituBeta, c.SituLinearBeta, want)
		}
	}
	for _, arch := range []string{"deepseek32", "deepseek4"} {
		keys := map[string]map[string]meta.Value{"deepseek32": ds32, "deepseek4": ds4}[arch]
		kv := map[string]meta.Value{"general.architecture": meta.MakeString(arch)}
		for k, v := range keys {
			kv[arch+"."+k] = v
		}
		c := &Config{Config: &jlm.Config{NRot: 64, NHead: 4, NLayer: 4, RopeBase: 1e4}}
		if arch == "deepseek4" {
			kv[arch+".hash_layer_count"] = u(3)
		}
		if err := stateContainer(meta.New(arch+".gguf", kv, nil), c); err != nil {
			t.Errorf("%s: the base of the refusals above is refused: %v", arch, err)
		}
		if want := map[string]float32{"deepseek32": 1e4, "deepseek4": 16e4}[arch]; c.RopeBase != want {
			t.Errorf("%s: rotary base %g, want %g", arch, c.RopeBase, want)
		}
		got := [6]float32{float32(c.HCMult), float32(c.HCIters), c.HCEps, float32(c.OGroups), float32(c.OLoraRank),
			float32(c.NHashLayers)}
		if want := map[string][6]float32{"deepseek4": {4, 20, 1e-6, 2, 8, 3}}[arch]; got != want {
			t.Errorf("%s: streams, rounds, eps, output groups, rank, hash blocks %v; want %v", arch, got, want)
		}
	}
	kv := map[string]meta.Value{"general.architecture": meta.MakeString("command-r"), "command-r.logit_scale": meta.MakeFloat(-2)}
	if err := stateContainer(meta.New("command-r.gguf", kv, nil), &Config{Config: &jlm.Config{}}); err == nil ||
		!strings.Contains(err.Error(), "logit_div is -0.5") {
		t.Errorf("a negative logit scale: %v, want it refused", err)
	}
	for want, keys := range map[string]map[string]meta.Value{
		"is undefined":                       {},
		"a window period over a window of 0": {"attention.sliding_window": meta.MakeUint(0)},
	} {
		kv := map[string]meta.Value{"general.architecture": meta.MakeString("gpt-oss")}
		for k, v := range keys {
			kv["gpt-oss."+k] = v
		}
		err := stateContainer(meta.New("gpt-oss.gguf", kv, nil), &Config{Config: &jlm.Config{}})
		if err == nil || !strings.Contains(err.Error(), want) || errors.Is(err, ErrNotImplemented) {
			t.Errorf("gpt-oss with %v: %v, want a malformed-file error naming %q", keys, err, want)
		}
	}
}

// TestTheLibraryStatesTheRopeSections holds the M-RoPE sections the library
// states to the ones the qwen3next family's case read: the file's own, none
// where it writes none, and a split short of the head's pairs refused.
func TestTheLibraryStatesTheRopeSections(t *testing.T) {
	for _, arch := range []string{"qwen3next", "qwen35", "qwen35moe", "qwen4exp", "clef"} {
		keys := map[string]meta.Value{"ssm.conv_kernel": meta.MakeUint(4), "ssm.group_count": meta.MakeUint(2),
			"ssm.inner_size": meta.MakeUint(64), "ssm.state_size": meta.MakeUint(16), "ssm.time_step_rank": meta.MakeUint(4),
			"hyper_connection.count": meta.MakeUint(2), "hyper_connection.low_rank": meta.MakeUint(4)}
		for _, tc := range []struct {
			secs []int32
			want [4]uint32
			fail bool
		}{
			{nil, [4]uint32{}, false},
			{[]int32{11, 11, 10, 0}, [4]uint32{11, 11, 10, 0}, false},
			{[]int32{11, 11, 9}, [4]uint32{}, true},
		} {
			if tc.secs != nil {
				keys["rope.dimension_sections"] = meta.MakeInt32s(tc.secs)
			}
			c := &Config{Name: "qwen3next", Config: &jlm.Config{NRot: 64}}
			err := stateContainer(layerFile(arch, keys, nil), c)
			if tc.fail != (err != nil) || !tc.fail && c.RopeSections != tc.want {
				t.Errorf("%s %v: sections %v, %v; want %v, refused %v", arch, tc.secs, c.RopeSections, err, tc.want, tc.fail)
			}
		}
	}
}

// TestContainerStatementsAreChecked refuses a flag no reader states, an
// activation that is not the container's, and an expression reading
// anything but the keys; a statement of the wrong type fails where it is
// read. Which containers a file states is the library's
// (TestLibraryContainersAreChecked).
func TestContainerStatementsAreChecked(t *testing.T) {
	file := func(states string) string {
		return `{"gguf": ["x"], "container": {` + states + `}}`
	}
	for _, tc := range []struct{ states, want string }{
		{`"x": {"flags": {"rope_sideways": "true"}}`, "is no graph feature"},
		{`"x": {"activation": "gelu_fast"}`, "is not one the container states"},
		{`"x": {"activation": "xielu"}`, "carries its parameters"},
		{`"x": {"norm_eps": "eps | default(1e-5)"}`, "is not the keys"},
		{`"x": {"norm_eps": "src[["}`, "jinja expression"},
		{`"x": {"window": {"period": "2"}}`, "states its size"},
		{`"x": {"window": {"size": "8", "period": "2", "pattern": "2"}}`, "not both"},
		{`"x": {"rope_base": "y"}`, "is not the keys"},
		{`"x": {"streams": {"count": "4", "eps": "1e-6"}}`, "sinkhorn_iterations with eps"},
		{`"x": {"output_groups": {"groups": "2"}}`, "their groups and rank"},
		{`"x": {"hash_layers": "n"}`, "is not the keys"},
		{`"x": {"situ": {"beta": "3"}}`, "its beta and linear_beta"},
		{`"x": {"expert_latent": "n"}`, "is not the keys"},
		{`"x": {"latent": {"kv": "64"}}`, "states its query rank"},
		{`"x": {"indexer": {"heads": "4", "top_k": "8"}}`, "states its heads, key_width and top_k"},
		{`"x": {"indexer": {"heads": "n", "key_width": "32", "top_k": "8"}}`, "is not the keys"},
		{`"x": {"revision": {"act_quant": "int4"}}`, "activation quantization \"int4\" is not one"},
		{`"x": {"revision": {"rope_scale_swa": "scale"}}`, "is not the keys"},
		{`"x": {"pre_tokenizer": "o200k-ish"}`, "pre-tokenizer \"o200k-ish\" is not one"},
		{`"x": {"rope_layout": "pairs"}`, "rotary layout \"pairs\" is not one"},
	} {
		_, err := parseModelJSON([]byte(file(tc.states)), "test", testNames(t))
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want an error naming %q", tc.states, err, tc.want)
		}
	}
	f := meta.New("x.gguf", map[string]meta.Value{"general.architecture": meta.MakeString("x"),
		"x.heads": meta.MakeUint(8)}, nil)
	for _, tc := range []struct {
		st   mjStates
		want string
	}{
		{mjStates{Flags: map[string]string{"qk_norm": "src['heads']"}}, "not true or false"},
		{mjStates{NormEps: "'small'"}, "not a number"},
		{mjStates{EmbdScale: "src['width']"}, "is undefined"},
		{mjStates{Flags: map[string]string{"post_norm": "true"}}, "not implemented"},
		{mjStates{RopeSections: "src['heads']"}, "not a list"},
		{mjStates{RopeSections: "[4, -1]"}, "not a count"},
		{mjStates{RopeSections: "[8, 8]"}, "sum to 16"},
		{mjStates{Revision: &mjStatesRevision{SWAGlobalFirst: "src['heads']"}}, "swa_global_first is 8, not true or false"},
		{mjStates{Revision: &mjStatesRevision{RopeScaleSWA: "'one'"}}, "rope_scale_swa is"},
		{mjStates{Revision: &mjStatesRevision{PictureBidirSliding: "src['heads'] + 1"}}, "picture_bidir_sliding is 9"},
		{mjStates{NoRopeEvery: "1"}, "every layer skips the rotary"},
		{mjStates{NoRopeEvery: "4", Window: &mjStatesWindow{Size: "8", Period: "2"}}, "beside a sliding window"},
		{mjStates{Window: &mjStatesWindow{Size: "0", Period: "4"}}, "a window period over a window of 0"},
		{mjStates{AttnTemp: &mjStatesTemp{Scale: "0.1", Floor: "8", Offset: "0.5"}}, "not a whole count of positions"},
	} {
		srcs := []string{tc.st.Flags["qk_norm"], tc.st.Flags["post_norm"], tc.st.NormEps, tc.st.EmbdScale, tc.st.NoRopeEvery,
			tc.st.RopeSections}
		if r := tc.st.Revision; r != nil {
			srcs = append(srcs, r.SWAGlobalFirst, r.RopeScaleSWA, r.PictureBidirSliding)
		}
		if w := tc.st.Window; w != nil {
			srcs = append(srcs, w.Size, w.Period)
		}
		if a := tc.st.AttnTemp; a != nil {
			srcs = append(srcs, a.Scale, a.Floor, a.Offset)
		}
		if err := tc.st.compile(srcs); err != nil {
			t.Fatal(err)
		}
		err := tc.st.apply("test", f, &Config{Config: &jlm.Config{}})
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%+v: %v, want an error naming %q", tc.st, err, tc.want)
		}
	}
}

// TestTheLibraryStatesWhatAFileMustSay holds the statements a GGUF must
// carry the key of (an absent one is refused naming -config: no guess from
// the model's size), the windowed layers' unscaled rotary and the q
// temperature to what configOfFile set by name before the files did.
func TestTheLibraryStatesWhatAFileMustSay(t *testing.T) {
	state := func(arch string, keys map[string]meta.Value) (*Config, error) {
		kv := map[string]meta.Value{"general.architecture": meta.MakeString(arch)}
		for k, v := range keys {
			kv[arch+"."+k] = v
		}
		c := &Config{Config: &jlm.Config{}}
		return c, stateContainer(meta.New(arch+".gguf", kv, nil), c)
	}
	stored := func(arch string, names ...string) (*Config, error) {
		ts := make([]meta.Tensor, len(names))
		for i, n := range names {
			ts[i] = meta.Tensor{Name: n, Dims: []uint64{8, 8}}
		}
		c := &Config{Config: &jlm.Config{}}
		return c, stateContainer(meta.New(arch+".gguf", map[string]meta.Value{"general.architecture": meta.MakeString(arch)}, ts), c)
	}
	ln := jlm.FlagLayerNorm | jlm.FlagRopeNeox
	for _, tc := range []struct {
		arch    string
		tensors []string
		flags   jlm.Flags
		eps     float32
		refused string
	}{
		{arch: "stablelm", tensors: []string{"blk.0.ffn_norm.weight"}, flags: ln, eps: 1e-5},
		{arch: "stablelm", flags: ln | jlm.FlagParallel, eps: 1e-5},
		{arch: "starcoder", tensors: []string{"position_embd.weight"}, flags: jlm.FlagLayerNorm | jlm.FlagNoPosEnc | jlm.FlagGELU, eps: 1e-5},
		{arch: "starcoder", refused: "no position_embd.weight"},
		{arch: "bert", flags: jlm.FlagLayerNorm | jlm.FlagNonCausal | jlm.FlagNoPosEnc | jlm.FlagGELU, eps: 1e-12},
		{arch: "nomic-bert", flags: ln | jlm.FlagNonCausal, eps: 1e-12},
		{arch: "arcee", flags: jlm.FlagReLU2},
		{arch: "arcee", tensors: []string{"blk.0.ffn_gate.weight"}, refused: "its FFN is ungated"},
	} {
		c, err := stored(tc.arch, tc.tensors...)
		switch {
		case tc.refused != "":
			if err == nil || !strings.Contains(err.Error(), tc.refused) {
				t.Errorf("%s storing %v: %v, want it refused naming %q", tc.arch, tc.tensors, err, tc.refused)
			}
		case err != nil:
			t.Errorf("%s storing %v: %v", tc.arch, tc.tensors, err)
		case c.Flags != tc.flags || c.RMSEps != tc.eps:
			t.Errorf("%s storing %v: flags %v eps %g, want %v %g", tc.arch, tc.tensors, c.Flags, c.RMSEps, tc.flags, tc.eps)
		}
	}
	for _, arch := range []string{"phi3", "exaone4"} {
		if _, err := state(arch, map[string]meta.Value{"block_count": meta.MakeUint(64), "context_length": meta.MakeUint(4096)}); err == nil ||
			!strings.Contains(err.Error(), "-config") {
			t.Errorf("%s with no attention.sliding_window: %v, want it refused naming -config", arch, err)
		}
	}
	c, err := state("olmo2", map[string]meta.Value{"attention.sliding_window": meta.MakeUint(4096)})
	if err != nil || c.Stated.ropeScaleSWA != 1 || c.RopeBaseSWA != 10000 {
		t.Errorf("olmo2 with a window: %v, sliding rotary at %g scaled %g; want 10000 unscaled", err, c.RopeBaseSWA, c.Stated.ropeScaleSWA)
	}
	if c, err := state("olmo2", nil); err != nil || c.Stated.ropeScaleSWA != 0 {
		t.Errorf("olmo2 with no window: %v, sliding rotary scaled %g; want none stated", err, c.Stated.ropeScaleSWA)
	}
	c, err = state("mistral3", map[string]meta.Value{"attention.temperature_scale": meta.MakeFloat(0.125),
		"rope.scaling.original_context_length": meta.MakeUint(16384)})
	if err != nil || c.AttnTempScale != 0.125 || c.AttnTempFloor != 16384 || c.AttnTempOffset != 0 {
		t.Errorf("mistral3's q temperature: %v, %g floor %d offset %g", err, c.AttnTempScale, c.AttnTempFloor, c.AttnTempOffset)
	}
	if _, err := state("mistral3", map[string]meta.Value{"attention.temperature_scale": meta.MakeFloat(0.125)}); err == nil ||
		!strings.Contains(err.Error(), "no count to floor positions by") {
		t.Errorf("a q temperature with no original context: %v, want it refused", err)
	}
	for states, want := range map[string]string{
		`{"attn_temp": {"scale": "1"}}`:  "states its scale and its floor",
		`{"learned_positions": "names"}`: "is not the keys",
	} {
		if _, err := parseModelJSON([]byte(`{"gguf": ["x"], "container": {"x": `+states+`}}`), "test", testNames(t)); err == nil ||
			!strings.Contains(err.Error(), want) {
			t.Errorf("%s: %v, want it refused naming %q", states, err, want)
		}
	}
}

// TestLibraryContainersAreChecked: a file states the containers of the GGUF
// architectures it names and of a class's own graph (a name no GGUF
// architecture has, which no GGUF reaches); a container of another file's
// architecture, one no class names, and a class whose graph no container
// states are refused.
func TestLibraryContainersAreChecked(t *testing.T) {
	file := func(name string, gguf []string, arch string, containers ...string) *modelJSON {
		mj := &modelJSON{name: name, GGUF: gguf, Container: map[string]*mjStates{}}
		if arch != "" {
			mj.Classes, mj.Config.Arch = []string{name + "Class"}, arch
		}
		for _, c := range containers {
			mj.Container[c] = &mjStates{}
		}
		return mj
	}
	for _, tc := range []struct {
		name  string
		files []*modelJSON
		want  string
	}{
		{"a class's own container", []*modelJSON{file("a", []string{"x"}, "", "x", "own"), file("b", nil, "own")}, ""},
		{"a container no class names", []*modelJSON{file("a", []string{"x"}, "", "x", "y")}, "no GGUF architecture and no class names"},
		{"another file's architecture", []*modelJSON{file("a", []string{"x"}, "", "z"), file("b", []string{"z"}, "")}, "a GGUF architecture it does not name"},
		{"a class's graph no container states", []*modelJSON{file("a", []string{"x"}, ""), file("b", nil, "w")}, "no container the library states"},
	} {
		err := libraryContainers(tc.files)
		if tc.want == "" && err != nil || tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)) {
			t.Errorf("%s: %v, want an error naming %q", tc.name, err, tc.want)
		}
	}
}
