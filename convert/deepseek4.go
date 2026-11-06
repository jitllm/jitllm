package convert

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

func deepseek4RoleOf(name string) (jlm.Role, bool) {
	switch name {
	// DeepSeek V4's attention: one key that is also the value, a grouped
	// output projection in two halves, sinks (gpt-oss's role) and the
	// compressors.
	case "attn_kv.weight":
		return jlm.RoleAttnK, true
	case "attn_output_a.weight":
		return jlm.RoleAttnOutA, true
	case "attn_output_b.weight":
		return jlm.RoleAttnOut, true
	case "attn_compressor_kv.weight":
		return jlm.RoleCompKV, true
	case "attn_compressor_gate.weight":
		return jlm.RoleCompGate, true
	case "attn_compressor_ape.weight":
		return jlm.RoleCompAPE, true
	case "attn_compressor_norm.weight":
		return jlm.RoleCompNorm, true
	case "indexer_compressor_kv.weight":
		return jlm.RoleIdxCompKV, true
	case "indexer_compressor_gate.weight":
		return jlm.RoleIdxCompGate, true
	case "indexer_compressor_ape.weight":
		return jlm.RoleIdxCompAPE, true
	case "indexer_compressor_norm.weight":
		return jlm.RoleIdxCompNorm, true
	// The hyper-connections, per block and at the head.
	case "hc_attn_fn.weight":
		return jlm.RoleHCAttnFn, true
	case "hc_attn_base.weight":
		return jlm.RoleHCAttnBase, true
	case "hc_attn_scale.weight":
		return jlm.RoleHCAttnScale, true
	case "hc_ffn_fn.weight":
		return jlm.RoleHCFFNFn, true
	case "hc_ffn_base.weight":
		return jlm.RoleHCFFNBase, true
	case "hc_ffn_scale.weight":
		return jlm.RoleHCFFNScale, true
	case "output_hc_fn.weight":
		return jlm.RoleHCHeadFn, true
	case "output_hc_base.weight":
		return jlm.RoleHCHeadBase, true
	case "output_hc_scale.weight":
		return jlm.RoleHCHeadScale, true
	// A hash-routed block's token-id table (I32 in the file; ds4HashTable).
	case "ffn_gate_tid2eid.weight":
		return jlm.RoleHashExperts, true
	}
	return 0, false
}

// ds4SwiGLULimit is DeepSeek V4's SwiGLU clamp: transformers' default
// swiglu_limit, the published config's, and the value llama.cpp's converter
// writes per layer from it. jlm.Flag2SwiGLUClamp bakes it.
const ds4SwiGLULimit = 10

// deepseek4Config is llama.cpp's deepseek4.cpp and transformers'
// DeepseekV4ForCausalLM, text only: the hyper-connection streams (HCMult,
// HCIters, HCEps), a q LoRA, one kv head that is also the value, the rotary on
// each head's last NRot dimensions in adjacent pairs at two bases (YaRN on the
// compressed blocks), a window and sinks on every block, compressed blocks
// (CompKinds, CompRateCSA, CompRateHCA) with the indexer, a grouped low-rank
// output, the sqrt(softplus) mixture with hash-routed lead blocks, and a
// SwiGLU clamped at ds4SwiGLULimit.
//
// Refused: any gate but sqrt-softplus, the renormalisation off, another
// clamp, and a window that does not cover the HCA rate and two CSA windows
// (the pending compressor inputs live in the window's cached rows).
//
// The reference-to-container table and the reasons in full:
// docs/engineering-history/model-correctness.md,
// "convert/deepseek4.go: deepseek4Config".
func deepseek4Config(f *meta.File, c *jlm.Config) error {
	c.Flags2 |= jlm.Flag2SwiGLUClamp | jlm.Flag2ExpertSqrtSoftplus | jlm.Flag2RopeTail | jlm.Flag2KeyValue |
		jlm.Flag2QUnitNorm | jlm.Flag2OutUnrotated
	if c.NKVHead != 1 {
		return fmt.Errorf("%d kv heads: DeepSeek V4's attention is one shared key and value: %w",
			c.NKVHead, ErrNotImplemented)
	}
	if c.HeadDimV != 0 {
		return fmt.Errorf("value width %d beside key width %d: DeepSeek V4's value is its key: %w",
			c.HeadDimV, c.HeadDim, ErrNotImplemented)
	}
	if err := partialRotary(c); err != nil {
		return err
	}
	c.QLoraRank = uint32(f.UintKey("attention.q_lora_rank", 0))
	c.SWAWindow = uint32(f.UintKey("attention.sliding_window", 0))
	if c.QLoraRank == 0 || c.SWAWindow == 0 {
		return fmt.Errorf("q_lora_rank %d, sliding_window %d: DeepSeek V4 has both", c.QLoraRank, c.SWAWindow)
	}
	c.SWAPeriod = allLocal(c.NLayer)
	// The two rotaries: the plain one at rope.freq_base on the sliding
	// blocks, the compressed blocks' at their own base, where the YaRN keys
	// read below apply (Config's global rotary carries them).
	comp, ok := f.Key("attention.compress_rope_freq_base")
	if !ok {
		return fmt.Errorf("no attention.compress_rope_freq_base")
	}
	cb, _ := comp.Float()
	if !(cb > 0) {
		return fmt.Errorf("attention.compress_rope_freq_base %g", cb)
	}
	c.RopeBaseSWA, c.RopeBase = c.RopeBase, float32(cb)

	c.HCMult = uint32(f.UintKey("hyper_connection.count", 0))
	c.HCIters = uint32(f.UintKey("hyper_connection.sinkhorn_iterations", 0))
	c.HCEps = float32(f.FloatKey("hyper_connection.epsilon", 0))
	if c.HCMult < 2 || c.HCIters == 0 || !(c.HCEps > 0) {
		return fmt.Errorf("hyper-connections %d streams, %d Sinkhorn rounds at %g", c.HCMult, c.HCIters, c.HCEps)
	}
	c.OGroups = uint32(f.UintKey("attention.output_group_count", 0))
	c.OLoraRank = uint32(f.UintKey("attention.output_lora_rank", 0))
	if c.OGroups == 0 || c.OLoraRank == 0 || c.NHead%c.OGroups != 0 {
		return fmt.Errorf("%d output groups of rank %d over %d heads", c.OGroups, c.OLoraRank, c.NHead)
	}
	c.IdxHeads = uint32(f.UintKey("attention.indexer.head_count", 0))
	c.IdxHeadDim = uint32(f.UintKey("attention.indexer.key_length", 0))
	c.IdxTopK = uint32(f.UintKey("attention.indexer.top_k", 0))
	c.NHashLayers = uint32(f.UintKey("hash_layer_count", 0))
	if c.NHashLayers > c.NLayer {
		return fmt.Errorf("%d hash-routed blocks of %d", c.NHashLayers, c.NLayer)
	}
	if err := ds4Clamp(f); err != nil {
		return err
	}
	if kv, ok := f.Key("expert_gating_func"); ok {
		if v, _ := kv.Uint(); v != 4 {
			return fmt.Errorf("expert_gating_func %d, and DeepSeek V4's router is sqrt-softplus: %w",
				v, ErrNotImplemented)
		}
	}
	if kv, ok := f.Key("expert_weights_norm"); ok {
		if v, _ := kv.Uint(); v == 0 {
			return fmt.Errorf("expert_weights_norm false, and DeepSeek V4's routers always "+
				"renormalise: %w", ErrNotImplemented)
		}
	}
	if f.UintKey("expert_count", 0) == 0 || f.UintKey("expert_shared_count", 0) != 1 {
		return fmt.Errorf("expert_count %d with %d shared experts: DeepSeek V4 is a mixture with "+
			"one shared expert in every block: %w", f.UintKey("expert_count", 0),
			f.UintKey("expert_shared_count", 0), ErrNotImplemented)
	}
	return ds4Kinds(f, c)
}

// ds4Clamp refuses a clamp other than the architecture's: every block's
// routed and shared limit (llama.cpp's converter writes both arrays from
// swiglu_limit).
func ds4Clamp(f *meta.File) error {
	for _, key := range []string{"swiglu_clamp_exp", "swiglu_clamp_shexp"} {
		kv, ok := f.Key(key)
		if !ok {
			continue
		}
		vs, err := kv.Float32s()
		if err != nil {
			v, ok := kv.Float()
			if !ok {
				return fmt.Errorf("%s: %w", key, err)
			}
			vs = []float32{float32(v)}
		}
		for i, v := range vs {
			if v != ds4SwiGLULimit {
				return fmt.Errorf("%s[%d] is %g, and DeepSeek V4's clamp is %d: %w", key, i, v,
					ds4SwiGLULimit, ErrNotImplemented)
			}
		}
	}
	return nil
}

// ds4Kinds reads each block's compression off compress_ratios and the
// tensors: a block with the indexer is CSA (its compressor two heads wide,
// the overlapping windows), one with a compressor alone is HCA, one with
// neither a sliding window alone, and its ratio must say the same. llama.cpp
// tells the two kinds apart by the ratio (4 against 128); transformers by the
// layer type, each type at one rate -- which is what the container carries.
func ds4Kinds(f *meta.File, c *jlm.Config) error {
	kv, ok := f.Key("attention.compress_ratios")
	if !ok {
		return fmt.Errorf("no attention.compress_ratios")
	}
	ratios, err := kv.Int32s()
	if err != nil {
		return fmt.Errorf("attention.compress_ratios: %w", err)
	}
	if len(ratios) < int(c.NLayer) {
		return fmt.Errorf("%d compress ratios for %d blocks", len(ratios), c.NLayer)
	}
	c.CompKinds = make([]jlm.CompKind, c.NLayer)
	for i := range c.CompKinds {
		r := ratios[i]
		_, idx := f.Get("blk." + itoa(i) + ".indexer.proj.weight")
		ckv, cmp := f.Get("blk." + itoa(i) + ".attn_compressor_kv.weight")
		rows := uint64(0)
		if cmp && len(ckv.Dims) == 2 {
			rows = ckv.Dims[1]
		}
		switch {
		case r == 0 && !cmp && !idx:
			c.CompKinds[i] = jlm.CompNone
		case r > 0 && cmp && idx && rows == 2*uint64(c.HeadDim):
			c.CompKinds[i] = jlm.CompCSA
			if c.CompRateCSA != 0 && c.CompRateCSA != uint32(r) {
				return fmt.Errorf("block %d compresses every %d where an earlier CSA block does every %d: %w",
					i, r, c.CompRateCSA, ErrNotImplemented)
			}
			c.CompRateCSA = uint32(r)
		case r > 0 && cmp && !idx && rows == uint64(c.HeadDim):
			c.CompKinds[i] = jlm.CompHCA
			if c.CompRateHCA != 0 && c.CompRateHCA != uint32(r) {
				return fmt.Errorf("block %d compresses every %d where an earlier HCA block does every %d: %w",
					i, r, c.CompRateHCA, ErrNotImplemented)
			}
			c.CompRateHCA = uint32(r)
		default:
			return fmt.Errorf("block %d: ratio %d with a compressor %v (%d rows on %d-wide heads) and an "+
				"indexer %v, which is neither CSA, HCA nor a sliding block", i, r, cmp, rows, c.HeadDim, idx)
		}
	}
	if c.CompRateCSA != 0 {
		switch {
		case c.IdxHeads == 0 || c.IdxHeadDim == 0 || c.IdxTopK == 0:
			return fmt.Errorf("CSA blocks with an indexer of %d heads of %d, top %d", c.IdxHeads, c.IdxHeadDim,
				c.IdxTopK)
		case c.NRot > c.IdxHeadDim:
			return fmt.Errorf("indexer heads of %d with a %d-wide rotary", c.IdxHeadDim, c.NRot)
		}
	}
	if w := c.SWAWindow; w < c.CompRateHCA || w < 2*c.CompRateCSA {
		return fmt.Errorf("a window of %d under HCA every %d and CSA every %d: the engine keeps the "+
			"compressor's pending inputs in the window's rows: %w", w, c.CompRateHCA, c.CompRateCSA,
			ErrNotImplemented)
	}
	return nil
}

// ds4Tensors finishes DeepSeek V4's tensors. The grouped output projection's
// first half is OGroups matrices of OLoraRank rows over one group's heads,
// stacked in the file as one {group width, OGroups*OLoraRank} matrix (llama.cpp
// reshapes it to three dimensions at load); it is carried as the bank it is,
// {group width, OLoraRank, OGroups}, the same bytes, so the container packs
// each group as its own sheet and a reader cuts a group the way it cuts an
// expert.
func ds4Tensors(s *jlm.Source) error {
	c := s.Config
	if c.Arch != jlm.ArchDeepseek4 {
		return nil
	}
	gw := uint64(c.NHead/c.OGroups) * uint64(c.HeadDim)
	for i := range s.Tensors {
		e := &s.Tensors[i]
		if e.Role != jlm.RoleAttnOutA {
			continue
		}
		if e.NDim != 2 || e.Dims[0] != gw || e.Dims[1] != uint64(c.OGroups)*uint64(c.OLoraRank) {
			return fmt.Errorf("convert: %s is %v, want {%d, %d}: %d groups of rank %d over %d heads of %d",
				e.Name, e.Dims[:e.NDim], gw, c.OGroups*c.OLoraRank, c.OGroups, c.OLoraRank, c.NHead, c.HeadDim)
		}
		e.NDim = 3
		e.Dims[1], e.Dims[2] = uint64(c.OLoraRank), uint64(c.OGroups)
	}
	return nil
}

// ds4HashTable is a hash-routed block's token-id table, I32 in the file, as
// the F32 the container stores (ids are exact well past any expert count).
// Every id is checked against the expert count, since it indexes the banks.
func ds4HashTable(t *meta.Tensor, data []byte, c *jlm.Config) ([]byte, error) {
	if t.Type != quant.Type(26) {
		return nil, fmt.Errorf("convert: %s is %s, want I32", t.Name, t.Type)
	}
	if len(t.Dims) != 2 || t.Dims[0] != uint64(c.NExpertUsed) || t.Dims[1] != uint64(c.NVocab) ||
		uint64(len(data)) != 4*t.Dims[0]*t.Dims[1] {
		return nil, fmt.Errorf("convert: %s is %v (%d bytes), want %d ids for each of %d tokens", t.Name,
			t.Dims, len(data), c.NExpertUsed, c.NVocab)
	}
	// A token's ids must be distinct: the engine selects them as the top-k of
	// a mask (model.ds4HashBias), which cannot select one expert twice, where
	// both references would run it twice.
	out := make([]byte, len(data))
	k := int(c.NExpertUsed)
	for i := 0; i < len(data); i += 4 {
		id := int32(binary.LittleEndian.Uint32(data[i:]))
		tok, j := i/4/k, i/4%k
		if id < 0 || uint32(id) >= c.NExpert {
			return nil, fmt.Errorf("convert: %s routes token %d to expert %d of %d", t.Name, tok, id, c.NExpert)
		}
		for p := 0; p < j; p++ {
			if int32(binary.LittleEndian.Uint32(data[(tok*k+p)*4:])) == id {
				return nil, fmt.Errorf("convert: %s routes token %d to expert %d twice: %w", t.Name, tok, id,
					ErrNotImplemented)
			}
		}
		binary.LittleEndian.PutUint32(out[i:], math.Float32bits(float32(id)))
	}
	return out, nil
}
