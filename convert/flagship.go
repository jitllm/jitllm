package convert

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// The flagship architectures (ArchMiniMaxM2 and the group around it).
// docs/design/model-coverage.md section 9 is the survey they were chosen
// from, with what each needs and what it costs.

func flagshipRoleOf(name string) (jlm.Role, bool) {
	switch name {
	// Gemma 4's per-block output scalar (layer_scalar), [1].
	case "layer_output_scale.weight":
		return jlm.RoleLayerOutScale, true
	// Gemma 4's mixture block (jlm.Flag2DenseMoE).
	case "pre_ffw_norm_2.weight":
		return jlm.RoleFFNNorm2, true
	case "post_ffw_norm_1.weight":
		return jlm.RolePostFFNNorm1, true
	case "post_ffw_norm_2.weight":
		return jlm.RolePostFFNNorm2, true
	case "ffn_gate_inp.scale":
		return jlm.RoleRouterNorm, true
	case "ffn_down_exps.scale":
		return jlm.RoleExpScale, true
	// The gate and up banks in one tensor, gate first (llama.cpp's
	// build_moe_ffn views the first n_ff rows of each expert as the gate);
	// unfuse splits it into the two banks.
	case "ffn_gate_up_exps.weight":
		return jlm.RoleExpUpBank, true
	// Gemma 4's per-layer embeddings (E2B/E4B).
	case "per_layer_token_embd.weight":
		return jlm.RolePLETokEmbd, true
	case "per_layer_model_proj.weight":
		return jlm.RolePLEModelProj, true
	case "per_layer_proj_norm.weight":
		return jlm.RolePLEProjNorm, true
	case "inp_gate.weight":
		return jlm.RolePLEGate, true
	case "proj.weight":
		return jlm.RolePLEProj, true
	case "post_norm.weight":
		return jlm.RolePLEPostNorm, true
	// DeepSeek V3.2's lightning indexer.
	case "indexer.attn_q_b.weight":
		return jlm.RoleIdxQB, true
	case "indexer.attn_k.weight":
		return jlm.RoleIdxK, true
	case "indexer.k_norm.weight":
		return jlm.RoleIdxKNorm, true
	case "indexer.k_norm.bias":
		return jlm.RoleIdxKNormB, true
	case "indexer.proj.weight":
		return jlm.RoleIdxProj, true
	// MiniMax-M3's indexer: the query from the block input, and the key under
	// the names its own tensor table uses.
	case "indexer.q_proj.weight":
		return jlm.RoleIdxQ, true
	case "indexer.q_norm.weight":
		return jlm.RoleIdxQNorm, true
	case "indexer.k_proj.weight":
		return jlm.RoleIdxK, true
	}
	return 0, false
}

// deepseek32Config is llama.cpp's deepseek32.cpp and transformers'
// DeepseekV32ForCausalLM: deepseek2's MLA and V3 router, read by the shared
// path, plus DeepSeek Sparse Attention's lightning indexer:
//
//	attention.indexer.head_count  -> IdxHeads   heads of the indexer's query
//	attention.indexer.key_length  -> IdxHeadDim one key per position, cached
//	attention.indexer.top_k       -> IdxTopK    the positions attention reads
//
// The indexer's query is projected from the query latent, so a model without
// q_lora_rank has nothing to project it from and is refused; its rotary is
// the model's on the first rope.dimension_count dimensions of each head.
func deepseek32Config(f *meta.File, c *Config) error {
	c.IdxHeads = uint32(f.UintKey("attention.indexer.head_count", 0))
	c.IdxHeadDim = uint32(f.UintKey("attention.indexer.key_length", 0))
	c.IdxTopK = uint32(f.UintKey("attention.indexer.top_k", 0))
	switch {
	case c.IdxHeads == 0 || c.IdxHeadDim == 0 || c.IdxTopK == 0:
		return fmt.Errorf("indexer %d heads of %d, top %d: deepseek32 has an indexer",
			c.IdxHeads, c.IdxHeadDim, c.IdxTopK)
	case c.KVLoraRank == 0 || c.QLoraRank == 0:
		return fmt.Errorf("an indexer with no query latent (q_lora_rank %d, kv_lora_rank %d)",
			c.QLoraRank, c.KVLoraRank)
	case c.IdxHeadDim%8 != 0 || c.NRot > c.IdxHeadDim:
		return fmt.Errorf("indexer heads of %d with a %d-wide rotary", c.IdxHeadDim, c.NRot)
	}
	return nil
}

// gemma4Config is llama.cpp's gemma4.cpp and transformers' Gemma4ForCausalLM,
// text only: gemma3's block with two attention geometries (the global layers'
// HeadDim, NKVHead and NRot, the sliding layers' their own, each block's
// stated in jlm.Config.Geoms), the global layers' "proportional" rotary as
// RoleRopeFreqs, a weightless norm on v (Flag2VNorm), k's projection as v on
// a global layer with no v_proj, an attention scale of one and a per-block
// layer_scalar (RoleLayerOutScale). The 26B runs a dense MLP beside a
// mixture (jlm.Flag2DenseMoE); the E2B/E4B add per-layer embeddings
// (PLEDim), KV sharing (NKVShared) and a double-width FFN on the sharing
// layers, each block's width in its geometry.
//
// The reference-to-container table: docs/engineering-history/
// model-correctness.md, "convert/flagship.go: gemma4Config".
func gemma4Config(f *meta.File, c *Config) error {
	c.EmbdScale = float32(math.Sqrt(float64(c.NEmbd)))
	c.Flags |= jlm.FlagGELU | jlm.FlagRopeNeox | jlm.FlagQKNorm
	c.Flags2 |= jlm.Flag2VNorm
	c.AttnScale = 1
	c.FinalSoftcap = float32(f.FloatKey("final_logit_softcapping", 0))
	c.NKVShared = uint32(f.UintKey("attention.shared_kv_layers", 0))
	if c.NKVShared >= c.NLayer {
		return fmt.Errorf("%d KV-sharing layers of %d", c.NKVShared, c.NLayer)
	}
	c.PLEDim = uint32(f.UintKey("embedding_length_per_layer_input", 0))
	if c.PLEDim != 0 {
		if _, ok := f.Get("per_layer_token_embd.weight"); !ok {
			return fmt.Errorf("per-layer embeddings %d wide and no per_layer_token_embd", c.PLEDim)
		}
	}
	// One width, or the base width and twice it on the KV-sharing layers.
	var ffns []int32
	if c.NFFN == 0 {
		kv, ok := f.Key("feed_forward_length")
		if !ok {
			return fmt.Errorf("no feed_forward_length")
		}
		ws, err := kv.Int32s()
		if err != nil || len(ws) != int(c.NLayer) || ws[0] <= 0 {
			return fmt.Errorf("feed_forward_length: %d values for %d layers (%v)", len(ws), c.NLayer, err)
		}
		c.NFFN = uint32(ws[0])
		for i, w := range ws {
			if w != ws[0] && (!c.KVShared(i) || w != 2*ws[0]) {
				return fmt.Errorf("feed_forward_length %v: layer %d is %d, and only the %d KV-sharing "+
					"layers may be twice %d", ws, i, w, c.NKVShared, c.NFFN)
			}
		}
		ffns = ws
	}
	c.SWAWindow = uint32(f.UintKey("attention.sliding_window", 0))
	if c.SWAWindow == 0 {
		return fmt.Errorf("no attention.sliding_window")
	}
	if err := swaPatternOf(f, c, 0); err != nil {
		return err
	}
	if c.SWAPeriod == 0 || c.SWAPeriod > c.NLayer {
		return fmt.Errorf("no sliding pattern with a global layer in it (period %d of %d layers)",
			c.SWAPeriod, c.NLayer)
	}
	c.RopeBaseSWA = float32(f.FloatKey("rope.freq_base_swa", 10000))
	// The global layers' geometry is the model's (key_length and
	// rope.dimension_count are the global layers'); the sliding layers' is
	// its own.
	var swa jlm.Geom
	swa.HeadDim = uint32(f.UintKey("attention.key_length_swa", 0))
	if v := uint32(f.UintKey("attention.value_length_swa", uint64(swa.HeadDim))); v != swa.HeadDim {
		return fmt.Errorf("sliding key width %d and value width %d", swa.HeadDim, v)
	}
	swa.NRot = uint32(f.UintKey("rope.dimension_count_swa", uint64(swa.HeadDim)))
	if swa.HeadDim == 0 || swa.NRot == 0 || swa.NRot%2 != 0 || swa.NRot > swa.HeadDim {
		return fmt.Errorf("sliding head %d with rotary %d", swa.HeadDim, swa.NRot)
	}
	if c.HeadDimV != 0 {
		return fmt.Errorf("global key width %d and value width %d", c.HeadDim, c.HeadDimV)
	}
	// The proportional rotary turns all of the global head, with frequencies
	// rope_freqs gives; a rotary narrower than the head would pair (i,
	// i+NRot/2) and scale every exponent by head_dim/NRot.
	if c.NRot != c.HeadDim {
		return fmt.Errorf("global rotary %d on a %d-wide head, where the proportional rotary "+
			"turns the whole head", c.NRot, c.HeadDim)
	}
	if _, ok := f.Get("rope_freqs.weight"); !ok {
		return fmt.Errorf("no rope_freqs: the global layers' proportional rotary is carried there")
	}
	// The kv heads, one per layer: every sliding layer agrees with every
	// other and every global layer with every other.
	kv, ok := f.Key("attention.head_count_kv")
	if !ok {
		c.NKVHead = c.NHead
	} else if v, ok := kv.Uint(); ok {
		c.NKVHead = uint32(v)
	} else {
		vs, err := kv.Int32s()
		if err != nil || len(vs) != int(c.NLayer) {
			return fmt.Errorf("attention.head_count_kv: %d values for %d layers (%v)", len(vs), c.NLayer, err)
		}
		for i, v := range vs {
			dst := &c.NKVHead
			if sliding(c, i) {
				dst = &swa.NKVHead
			}
			if v <= 0 || (*dst != 0 && *dst != uint32(v)) {
				return fmt.Errorf("attention.head_count_kv %v does not split by the sliding pattern", vs)
			}
			*dst = uint32(v)
		}
	}
	if c.NKVHead == 0 || c.NHead%c.NKVHead != 0 {
		return fmt.Errorf("%d global kv heads for %d query heads", c.NKVHead, c.NHead)
	}
	if swa.NKVHead == 0 {
		swa.NKVHead = c.NKVHead
	}
	c.Geoms = blockGeoms(c, swa, ffns)
	return nil
}

// minimaxM2Config is llama.cpp's minimax-m2.cpp and transformers'
// MiniMaxM2ForCausalLM:
//
//	q and k RMSNormed over the whole projection  -> FlagQKNorm|FlagQKNormWide
//	  ({n_head*head_dim} and {n_kv*head_dim}, before the reshape into heads)
//	NEOX rotary on rope.dimension_count of head  -> FlagRopeNeox, NRot (64 of 128)
//	sigmoid router, selection-only bias,          -> FlagExpertSigmoid, RoleExpProbsB
//	  renormalised, no routed scale, no shared
//	  expert, every block a mixture
//
// The gate is SIGMOID whatever the file says (RULE 7m): MiniMaxM2TopKRouter
// hardcodes nn.functional.sigmoid and reads no gating key, the published
// config states scoring_func "sigmoid", and llama.cpp's builder passes the
// key through. A file that states softmax is refused rather than obeyed.
//
// The routed weights are renormalised whatever the file says: the class
// divides by the top-k sum unconditionally and llama.cpp's builder passes
// true as a literal, so a file writing expert_weights_norm false describes a
// model nothing runs and is refused.
func minimaxM2Config(f *meta.File, c *Config) error {
	c.Flags |= jlm.FlagRopeNeox | jlm.FlagQKNorm | jlm.FlagQKNormWide | jlm.FlagExpertSigmoid
	if err := partialRotary(c); err != nil {
		return err
	}
	if kv, ok := f.Key("expert_gating_func"); ok {
		if v, _ := kv.Uint(); v != 2 {
			return fmt.Errorf("expert_gating_func %d, and MiniMaxM2TopKRouter is sigmoid: %w",
				v, ErrNotImplemented)
		}
	}
	if kv, ok := f.Key("expert_weights_norm"); ok {
		if v, _ := kv.Uint(); v == 0 {
			return fmt.Errorf("expert_weights_norm false, and MiniMaxM2TopKRouter always "+
				"renormalises: %w", ErrNotImplemented)
		}
	}
	if f.UintKey("expert_count", 0) == 0 {
		return fmt.Errorf("no expert_count: minimax-m2 is a mixture")
	}
	if f.UintKey("expert_shared_count", 0) != 0 || f.UintKey("leading_dense_block_count", 0) != 0 {
		return fmt.Errorf("a shared expert or a dense lead, which minimax-m2 does not have: %w",
			ErrNotImplemented)
	}
	for i := 0; i < int(c.NLayer); i++ {
		if _, ok := f.Get("blk." + itoa(i) + ".exp_probs_b.bias"); !ok {
			return fmt.Errorf("block %d has no selection bias (use_routing_bias false): %w",
				i, ErrNotImplemented)
		}
	}
	return nil
}

// minimaxM3Config is llama.cpp's minimax-m3.cpp and transformers'
// MiniMaxM3VLForCausalLM: a per-head q/k norm, NEOX rotary on part of each
// head, a dense lead and then V3's sigmoid router with a shared expert,
// gpt-oss's clamped SwiGLU in every FFN (FlagSwiGLUOAI), and MiniMax Sparse
// Attention (the Idx fields) on every block past the lead.
//
// Refused by name: a gate that is not sigmoid or weights not renormalised
// (MiniMaxM3VLTopKRouter hardcodes both; RULE 7m), an indexer key of another
// width than the head (it is cached as one more kv head), other than one
// indexer head per kv group, and no local block (every group must read the
// same number of positions). The swigluoai constants are literals.
//
// The reference-to-container table and the reasons in full:
// docs/engineering-history/model-correctness.md,
// "convert/flagship.go: minimaxM3Config".
func minimaxM3Config(f *meta.File, c *Config) error {
	c.Flags |= jlm.FlagRopeNeox | jlm.FlagQKNorm | jlm.FlagExpertSigmoid | jlm.FlagSwiGLUOAI
	if err := partialRotary(c); err != nil {
		return err
	}
	if kv, ok := f.Key("expert_gating_func"); ok {
		if v, _ := kv.Uint(); v != 2 {
			return fmt.Errorf("expert_gating_func %d, and MiniMaxM3VLTopKRouter is sigmoid: %w",
				v, ErrNotImplemented)
		}
	}
	if kv, ok := f.Key("expert_weights_norm"); ok {
		if v, _ := kv.Uint(); v == 0 {
			return fmt.Errorf("expert_weights_norm false, and MiniMaxM3VLTopKRouter always "+
				"renormalises: %w", ErrNotImplemented)
		}
	}
	if f.UintKey("expert_count", 0) == 0 || f.UintKey("expert_shared_count", 0) != 1 {
		return fmt.Errorf("expert_count %d with %d shared experts: minimax-m3 is a mixture "+
			"with one shared expert: %w", f.UintKey("expert_count", 0),
			f.UintKey("expert_shared_count", 0), ErrNotImplemented)
	}
	c.IdxHeads = uint32(f.UintKey("attention.indexer.head_count", 0))
	c.IdxHeadDim = uint32(f.UintKey("attention.indexer.key_length", 0))
	c.IdxTopK = uint32(f.UintKey("attention.indexer.top_k", 0))
	c.IdxBlock = uint32(f.UintKey("attention.indexer.block_size", 0))
	c.IdxLocal = uint32(f.UintKey("attention.indexer.local_blocks", 0))
	switch {
	case c.IdxHeads == 0 || c.IdxHeadDim == 0 || c.IdxTopK == 0 || c.IdxBlock == 0:
		return fmt.Errorf("indexer %d heads of %d, top %d blocks of %d: minimax-m3 has an indexer",
			c.IdxHeads, c.IdxHeadDim, c.IdxTopK, c.IdxBlock)
	case c.IdxHeads != c.NKVHead:
		return fmt.Errorf("%d indexer heads for %d kv groups, and MiniMax Sparse Attention "+
			"selects once per group: %w", c.IdxHeads, c.NKVHead, ErrNotImplemented)
	case c.IdxHeadDim != c.HeadDim:
		return fmt.Errorf("indexer keys of %d on %d-wide heads: the engine caches the key as "+
			"one more kv head: %w", c.IdxHeadDim, c.HeadDim, ErrNotImplemented)
	case c.IdxLocal == 0 || c.IdxLocal > c.IdxTopK:
		return fmt.Errorf("%d local blocks of %d selected: a query's own block forced in, "+
			"inside the top-k, is implemented: %w", c.IdxLocal, c.IdxTopK, ErrNotImplemented)
	}
	return nil
}

// minimaxM3Blocks checks the indexer sits exactly where the selection runs:
// on every block past the dense lead (llama.cpp's is_sparse, and the
// published sparse_attention_freq), never inside it. Read after the dense
// lead, which configOf reads with the mixture.
func minimaxM3Blocks(f *meta.File, c *Config) error {
	for i := 0; i < int(c.NLayer); i++ {
		_, ok := f.Get("blk." + itoa(i) + ".indexer.q_proj.weight")
		if want := i >= int(c.NDenseLead); ok != want {
			return fmt.Errorf("block %d (dense lead %d) has an indexer: %v; sparse attention "+
				"runs on exactly the blocks past the lead: %w", i, c.NDenseLead, ok, ErrNotImplemented)
		}
	}
	return nil
}

// splitGateUpBank cuts a fused gate|up expert bank ({k, 2*n_ff, n_expert},
// each expert's gate rows then its up rows) into the gate bank and the up
// bank. The bytes are still the source's row-major blocks, so each expert's
// half is a byte range; the banks are new buffers because an expert's two
// halves interleave with the next expert's.
func splitGateUpBank(e *jlm.Tensor, c *Config) ([]jlm.Tensor, bool, error) {
	src, ok := sourceType(e.Type)
	if !ok {
		return nil, false, fmt.Errorf("convert: %s is %v, which has no source type", e.Name, e.Type)
	}
	be, bb := uint64(src.BlockElems()), uint64(src.BlockBytes())
	if be == 0 || e.Dims[0]%be != 0 {
		return nil, false, fmt.Errorf("convert: %s rows are %d elements, not a multiple of %s's %d",
			e.Name, e.Dims[0], src, be)
	}
	ff, n := uint64(c.NFFNExp), e.Dims[2]
	if n != uint64(c.NExpert) {
		return nil, false, fmt.Errorf("convert: %s holds %d experts, want %d", e.Name, n, c.NExpert)
	}
	half := ff * (e.Dims[0] / be * bb)
	if uint64(len(e.Data)) != 2*half*n {
		return nil, false, fmt.Errorf("convert: %s holds %d bytes, want %d", e.Name, len(e.Data), 2*half*n)
	}
	gate, up := make([]byte, half*n), make([]byte, half*n)
	for x := uint64(0); x < n; x++ {
		copy(gate[x*half:(x+1)*half], e.Data[2*x*half:(2*x+1)*half])
		copy(up[x*half:(x+1)*half], e.Data[(2*x+1)*half:(2*x+2)*half])
	}
	out := make([]jlm.Tensor, 0, 2)
	for _, p := range []struct {
		role jlm.Role
		data []byte
	}{{jlm.RoleExpGateBank, gate}, {jlm.RoleExpUpBank, up}} {
		t := *e
		t.Role, t.Data, t.Name = p.role, p.data, e.Name+"/"+p.role.String()
		t.Dims[1] = ff
		out = append(out, t)
	}
	return out, true, nil
}

// gemma4Tensors finishes Gemma 4's blocks. A KV-sharing block's k and v
// projections and k norm are dropped: transformers never builds them and
// llama.cpp loads and never reads them, and a file may carry them anyway.
// A mixture block's dense MLP (jlm.Flag2DenseMoE) moves to the shared expert's
// roles, since its gate/up/down roles are its banks, and the router's scale
// becomes its input's norm weight, router.scale * n_embd^-1/2.
func gemma4Tensors(c *Config, s *jlm.Source) error {
	if c == nil || c.Arch != ArchGemma4 {
		return nil
	}
	if c.NKVShared != 0 {
		kept := s.Tensors[:0]
		for _, t := range s.Tensors {
			if t.Block >= 0 && !t.Role.Vision() && c.KVShared(int(t.Block)) &&
				(t.Role == jlm.RoleAttnK || t.Role == jlm.RoleAttnV || t.Role == jlm.RoleAttnKNorm) {
				continue
			}
			kept = append(kept, t)
		}
		s.Tensors = kept
	}
	if !c.Flags2.Has(jlm.Flag2DenseMoE) {
		return nil
	}
	inv := float32(1 / math.Sqrt(float64(c.NEmbd)))
	for i := range s.Tensors {
		t := &s.Tensors[i]
		if t.Block < 0 || t.Role.Vision() {
			continue
		}
		switch t.Role {
		case jlm.RoleFFNGate:
			t.Role = jlm.RoleShExpGate
		case jlm.RoleFFNUp:
			t.Role = jlm.RoleShExpUp
		case jlm.RoleFFNDown:
			t.Role = jlm.RoleShExpDown
		case jlm.RoleRouterNorm:
			if t.Type != jlm.TypeF32 || t.NDim != 1 || t.Dims[0] != uint64(c.NEmbd) ||
				len(t.Data) != 4*int(c.NEmbd) {
				return fmt.Errorf("convert: %s is %v %v, want an F32 vector of %d", t.Name, t.Type,
					t.Dims[:t.NDim], c.NEmbd)
			}
			d := make([]byte, len(t.Data))
			for j := 0; j < int(c.NEmbd); j++ {
				v := math.Float32frombits(binary.LittleEndian.Uint32(t.Data[4*j:]))
				binary.LittleEndian.PutUint32(d[4*j:], math.Float32bits(v*inv))
			}
			t.Data = d
		}
	}
	return nil
}

// gemma4MoEConfig is the 26B's mixture (jlm.Flag2DenseMoE), read after the
// generic mixture keys: softmax, renormalised whatever the file says --
// Gemma4TextRouter divides by the top-k sum and llama.cpp's builder passes
// true -- with no routed scale or groups, and the dense MLP carried as an
// ungated shared expert at the dense width.
func gemma4MoEConfig(f *meta.File, c *Config) error {
	if c.NExpert == 0 {
		return nil
	}
	c.Flags2 |= jlm.Flag2DenseMoE
	if c.Flags.Has(jlm.FlagExpertSigmoid) {
		return fmt.Errorf("a sigmoid gate, and Gemma4TextRouter is a softmax: %w", ErrNotImplemented)
	}
	c.Flags &^= jlm.FlagNoExpertNorm
	if (c.ExpertScale != 0 && c.ExpertScale != 1) || c.NExpertGroup != 0 {
		return fmt.Errorf("a routed scale %g or expert groups, which Gemma 4 does not have: %w",
			c.ExpertScale, ErrNotImplemented)
	}
	if f.UintKey("expert_shared_count", 0) != 0 {
		return fmt.Errorf("a shared expert beside the dense MLP: %w", ErrNotImplemented)
	}
	c.NFFNShExp = c.NFFN
	return nil
}

// sliding reports that block i of c's pattern slides.
func sliding(c *Config, i int) bool { return uint32(i)%c.SWAPeriod < c.SWAPeriod-1 }

// blockGeoms is each block's geometry: the sliding layers' swa where the
// pattern slides, the model's elsewhere, and ffns' feed-forward width where it
// is stated per layer; none at all when every block is the model's.
func blockGeoms(c *Config, swa jlm.Geom, ffns []int32) []jlm.Geom {
	model := jlm.Geom{NKVHead: c.NKVHead, HeadDim: c.HeadDim, NRot: c.NRot, NFFN: c.NFFN}
	gs := make([]jlm.Geom, c.NLayer)
	differ := false
	for i := range gs {
		g := model
		if sliding(c, i) {
			g.NKVHead, g.HeadDim, g.NRot = swa.NKVHead, swa.HeadDim, swa.NRot
		}
		if ffns != nil {
			g.NFFN = uint32(ffns[i])
		}
		gs[i] = g
		differ = differ || g != model
	}
	if !differ {
		return nil
	}
	return gs
}
