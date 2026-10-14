package convert

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// roleOf is the source format's tensor-name vocabulary, and the only place it
// appears: a container identifies a tensor by (Role, Block, Index), so this
// table is read once per conversion. An unrecognised name is a refusal, not a
// skip, because a tensor dropped at conversion is a model that loads and is
// wrong.
var roleOf = map[string]jlm.Role{
	// Model-level.
	"token_embd.weight":  jlm.RoleTokenEmbd,
	"output.weight":      jlm.RoleOutput,
	"output_norm.weight": jlm.RoleOutputNorm,
	// The classic block's LayerNorm biases and phi-2's biased head (C6).
	"output_norm.bias":          jlm.RoleOutputNormBias,
	"output.bias":               jlm.RoleOutputBias,
	"rope_factors_long.weight":  jlm.RoleRopeFactorsLong,
	"rope_factors_short.weight": jlm.RoleRopeFactorsShort,
	"rope_freqs.weight":         jlm.RoleRopeFreqs,
	// An encoder's embedding stage (bert, nomic-bert).
	"token_types.weight":     jlm.RoleTokenTypes,
	"position_embd.weight":   jlm.RolePosEmbd,
	"token_embd_norm.weight": jlm.RoleTokenEmbdNorm,
	"token_embd_norm.bias":   jlm.RoleTokenEmbdNormBias,
	// A decision model's scorer, as llama.cpp names Laya's (conversion/bert.py).
	"cls.norm.weight":   jlm.RoleScorerNorm,
	"cls.norm.bias":     jlm.RoleScorerNormBias,
	"cls.weight":        jlm.RoleScorer,
	"cls.bias":          jlm.RoleScorerBias,
	"cls.output.weight": jlm.RoleScorerOut,
	"cls.output.bias":   jlm.RoleScorerOutBias,
	// Clef's scorer (conversion/clef.py), the same four positions as Laya's
	// at another width; the rest of Clef's head is named in init below.
	"decision.scorer.weight":     jlm.RoleScorer,
	"decision.scorer.bias":       jlm.RoleScorerBias,
	"decision.scorer_out.weight": jlm.RoleScorerOut,
	"decision.scorer_out.bias":   jlm.RoleScorerOutBias,
	// EmbeddingGemma's sentence-transformers Dense layers, which llama.cpp
	// names after the module index they occupy in the pipeline (2 and 3).
	"dense_2.weight": jlm.RoleEmbdDense1,
	"dense_3.weight": jlm.RoleEmbdDense2,
	// Ollama's embeddinggemma spells the same two matrices dense.0/dense.1.
	"dense.0.weight": jlm.RoleEmbdDense1,
	"dense.1.weight": jlm.RoleEmbdDense2,

	// Per block.
	"attn_norm.weight":           jlm.RoleAttnNorm,
	"attn_q.weight":              jlm.RoleAttnQ,
	"attn_k.weight":              jlm.RoleAttnK,
	"attn_v.weight":              jlm.RoleAttnV,
	"attn_output.weight":         jlm.RoleAttnOut,
	"attn_qkv.weight":            jlm.RoleAttnQKV,
	"attn_qkv.bias":              jlm.RoleAttnQKVBias,
	"attn_norm.bias":             jlm.RoleAttnNormBias,
	"attn_norm_2.weight":         jlm.RoleAttnNorm2,
	"attn_norm_2.bias":           jlm.RoleAttnNorm2Bias,
	"ffn_norm.bias":              jlm.RoleFFNNormBias,
	"attn_q.bias":                jlm.RoleAttnQBias,
	"attn_k.bias":                jlm.RoleAttnKBias,
	"attn_v.bias":                jlm.RoleAttnVBias,
	"attn_output.bias":           jlm.RoleAttnOutBias,
	"attn_q_norm.weight":         jlm.RoleAttnQNorm,
	"attn_k_norm.weight":         jlm.RoleAttnKNorm,
	"attn_gate.weight":           jlm.RoleAttnGate,
	"attn_sinks.weight":          jlm.RoleAttnSinks,
	"post_attention_norm.weight": jlm.RolePostAttnNorm,
	"post_ffw_norm.weight":       jlm.RolePostFFNNorm,
	"ffn_norm.weight":            jlm.RoleFFNNorm,
	"ffn_gate.weight":            jlm.RoleFFNGate,
	"ffn_up.weight":              jlm.RoleFFNUp,
	"ffn_down.weight":            jlm.RoleFFNDown,
	// A post-norm encoder block's two LayerNorms and its MLP biases.
	"attn_output_norm.weight":  jlm.RoleAttnOutNorm,
	"attn_output_norm.bias":    jlm.RoleAttnOutNormBias,
	"layer_output_norm.weight": jlm.RoleLayerOutNorm,
	"layer_output_norm.bias":   jlm.RoleLayerOutNormBias,
	"ffn_up.bias":              jlm.RoleFFNUpBias,
	"ffn_down.bias":            jlm.RoleFFNDownBias,
	"ffn_gate_inp.weight":      jlm.RoleRouter,
	"ffn_gate_inp.bias":        jlm.RoleRouterBias,
	// DeepSeek's selection bias, which llama.cpp keeps as its own tensor
	// rather than as the router's bias (see RoleExpProbsB). The ".bias" suffix
	// is appended at llama.cpp's load call site, not in its name constant; no
	// GGUF contains the bare "exp_probs_b".
	"exp_probs_b.bias":     jlm.RoleExpProbsB,
	"ffn_gate_exps.bias":   jlm.RoleExpGateBias,
	"ffn_up_exps.bias":     jlm.RoleExpUpBias,
	"ffn_down_exps.bias":   jlm.RoleExpDownBias,
	"ffn_gate_exps.weight": jlm.RoleExpGateBank,
	"ffn_up_exps.weight":   jlm.RoleExpUpBank,
	"ffn_down_exps.weight": jlm.RoleExpDownBank,
	// Multi-head Latent Attention. attn_kv_b is the un-absorbed pair that older
	// deepseek2 GGUFs ship; unfuseMLA splits it into attn_k_b/attn_v_b, so no
	// container carries one.
	"attn_q_a.weight":       jlm.RoleAttnQA,
	"attn_q_a_norm.weight":  jlm.RoleAttnQANorm,
	"attn_q_b.weight":       jlm.RoleAttnQB,
	"attn_kv_a_mqa.weight":  jlm.RoleAttnKVA,
	"attn_kv_a_norm.weight": jlm.RoleAttnKVANorm,
	"attn_kv_b.weight":      jlm.RoleAttnKVB,
	"attn_k_b.weight":       jlm.RoleAttnKB,
	"attn_v_b.weight":       jlm.RoleAttnVB,

	// A multi-token-prediction block's own tensors (jlm.Config.NMTP).
	"nextn.eh_proj.weight":          jlm.RoleNextnEHProj,
	"nextn.enorm.weight":            jlm.RoleNextnENorm,
	"nextn.hnorm.weight":            jlm.RoleNextnHNorm,
	"nextn.shared_head_norm.weight": jlm.RoleNextnHeadNorm,
	"nextn.shared_head_head.weight": jlm.RoleNextnHead,
	"nextn.embed_tokens.weight":     jlm.RoleNextnEmbd,

	"ffn_gate_inp_shexp.weight": jlm.RoleShRouter,
	"ffn_gate_shexp.weight":     jlm.RoleShExpGate,
	"ffn_up_shexp.weight":       jlm.RoleShExpUp,
	"ffn_down_shexp.weight":     jlm.RoleShExpDown,

	// The linear-attention block. The source spells the family "ssm_"; what
	// these drive is a gated delta rule over a recurrent state. attn_qkv and
	// attn_gate are not listed again: the layer kind (Config.LayerKinds) says
	// which graph consumes them, so one role code serves both.
	"ssm_ba.weight":     jlm.RoleSSMBA,
	"ssm_conv1d.weight": jlm.RoleSSMConv1d,
	"ssm_a":             jlm.RoleSSMA,
	"ssm_dt.bias":       jlm.RoleSSMDtBias,
	"ssm_norm.weight":   jlm.RoleSSMNorm,
	"ssm_out.weight":    jlm.RoleSSMOut,
	// Mamba-2's: in_proj is split at conversion (unfuseSSD), so no container
	// carries RoleSSMInProj; D and the convolution's bias are vectors.
	"ssm_in.weight":   jlm.RoleSSMInProj,
	"ssm_d":           jlm.RoleSSMD,
	"ssm_conv1d.bias": jlm.RoleSSMConvBias,
	// Mamba-1's: x_proj is split at conversion (unfuseMamba1) and arrives
	// whole under its first part's role; dt_proj and the three norms.
	"ssm_x.weight":       jlm.RoleSSMXDt,
	"ssm_dt.weight":      jlm.RoleSSMDtProj,
	"ssm_dt_norm.weight": jlm.RoleSSMDtNorm,
	"ssm_b_norm.weight":  jlm.RoleSSMBNorm,
	"ssm_c_norm.weight":  jlm.RoleSSMCNorm,
	// Falcon-H1's FFN norm, which llama.cpp names with no ".weight".
	"ffn_norm": jlm.RoleFFNNorm,
	// LFM2's short convolution: in_proj is split at conversion (unfuseLFM2),
	// the convolution is a window per channel like Mamba's.
	"shortconv.in_proj.weight":  jlm.RoleSSMInProj,
	"shortconv.conv.weight":     jlm.RoleSSMConv1d,
	"shortconv.out_proj.weight": jlm.RoleSSMOut,

	// Vision tower. ln1 is the pre-attention norm and ln2 the pre-MLP one;
	// the tower's "ffn_up"/"ffn_down" are its two MLP matrices.
	"v.patch_embd.weight":    jlm.RoleVPatchEmbd,
	"v.patch_embd.bias":      jlm.RoleVPatchBias,
	"v.position_embd.weight": jlm.RoleVPosEmbd,
	"v.post_ln.weight":       jlm.RoleVPostNorm,
	"v.post_ln.bias":         jlm.RoleVPostNormBias,
	"mm.model.fc.weight":     jlm.RoleVProj,
	"mm.model.fc.bias":       jlm.RoleVProjBias,
	// gemma3's projector: an RMSNorm, then one matrix (no bias). The norm
	// weight already carries Gemma's +1 (llama.cpp's converter adds it).
	"mm.input_projection.weight": jlm.RoleVProj,
	"mm.soft_emb_norm.weight":    jlm.RoleVProjNorm,
	// CLIP. The class embedding is one row prepended to the patch sequence;
	// pre_ln is a LayerNorm over the whole sequence before block 0; and the
	// MLP projector's two matrices are mm.0 and mm.2 (the odd index is the GELU
	// between them). mm.0 is RoleVProj because it is the same graph position as
	// idefics3's single fc.
	"v.class_embd":      jlm.RoleVClassEmbd,
	"v.pre_ln.weight":   jlm.RoleVPreNorm,
	"v.pre_ln.bias":     jlm.RoleVPreNormBias,
	"mm.0.weight":       jlm.RoleVProj,
	"mm.0.bias":         jlm.RoleVProjBias,
	"mm.2.weight":       jlm.RoleVProj2,
	"mm.2.bias":         jlm.RoleVProj2Bias,
	"v.ln1.weight":      jlm.RoleVAttnNorm,
	"v.ln1.bias":        jlm.RoleVAttnNormBias,
	"v.ln2.weight":      jlm.RoleVFFNNorm,
	"v.ln2.bias":        jlm.RoleVFFNNormBias,
	"v.attn_q.weight":   jlm.RoleVAttnQ,
	"v.attn_k.weight":   jlm.RoleVAttnK,
	"v.attn_v.weight":   jlm.RoleVAttnV,
	"v.attn_out.weight": jlm.RoleVAttnOut,
	"v.attn_q.bias":     jlm.RoleVAttnQBias,
	"v.attn_k.bias":     jlm.RoleVAttnKBias,
	"v.attn_v.bias":     jlm.RoleVAttnVBias,
	"v.attn_out.bias":   jlm.RoleVAttnOutBias,
	"v.ffn_up.weight":   jlm.RoleVFC1,
	"v.ffn_up.bias":     jlm.RoleVFC1Bias,
	"v.ffn_down.weight": jlm.RoleVFC2,
	"v.ffn_down.bias":   jlm.RoleVFC2Bias,
	// A gated tower MLP's gate (Qwen2.5-VL).
	"v.ffn_gate.weight": jlm.RoleVFFNGate,
	"v.ffn_gate.bias":   jlm.RoleVFFNGateBias,

	// MiniCPM-V's resampler (llama.cpp's TN_MINICPMV_*). The query is a matrix
	// of learned rows, read once; the five matrices are weights.
	"resampler.query":           jlm.RoleVRsQuery,
	"resampler.kv.weight":       jlm.RoleVRsKV,
	"resampler.ln_q.weight":     jlm.RoleVRsLnQ,
	"resampler.ln_q.bias":       jlm.RoleVRsLnQBias,
	"resampler.ln_kv.weight":    jlm.RoleVRsLnKV,
	"resampler.ln_kv.bias":      jlm.RoleVRsLnKVBias,
	"resampler.attn.q.weight":   jlm.RoleVRsQ,
	"resampler.attn.q.bias":     jlm.RoleVRsQBias,
	"resampler.attn.k.weight":   jlm.RoleVRsK,
	"resampler.attn.k.bias":     jlm.RoleVRsKBias,
	"resampler.attn.v.weight":   jlm.RoleVRsV,
	"resampler.attn.v.bias":     jlm.RoleVRsVBias,
	"resampler.attn.out.weight": jlm.RoleVRsOut,
	"resampler.attn.out.bias":   jlm.RoleVRsOutBias,
	"resampler.ln_post.weight":  jlm.RoleVRsLnPost,
	"resampler.ln_post.bias":    jlm.RoleVRsLnPostBias,
	"resampler.proj.weight":     jlm.RoleVRsProj,
}

// towerName is a tower tensor's name in the vocabulary identify reads, where
// a projector spells one differently: Janus-Pro's second aligner matrix is
// mm.1, the graph position llava's mm.2 holds. v is nil for a text file.
func towerName(v *jlm.Vision, name string) string {
	if v != nil && v.Projector == jlm.ProjJanus && strings.HasPrefix(name, "mm.1.") {
		return "mm.2." + strings.TrimPrefix(name, "mm.1.")
	}
	// Llama 4's post_attention_layernorm is its pre-MLP norm (ln2), and
	// llama.cpp's tensor map names it attn_post_norm for a transformers 5
	// checkpoint (vision_model.model.layers.N.post_attention_layernorm,
	// which it lists for Gemma 4's genuine post-attention norm).
	if v != nil && v.Projector == jlm.ProjLlama4 && strings.Contains(name, ".attn_post_norm.") {
		return strings.Replace(name, ".attn_post_norm.", ".ln2.", 1)
	}
	// Kimi-VL's and LFM2-VL's are mm.1 and mm.2: llava's mm.0 and mm.2
	// positions.
	if v != nil && (v.Projector == jlm.ProjKimiVL || v.Projector == jlm.ProjLFM2) && strings.HasPrefix(name, "mm.1.") {
		return "mm.0." + strings.TrimPrefix(name, "mm.1.")
	}
	return name
}

// unreadTower names the tower tensors a file carries and no graph reads, each
// with the reason. Every other unknown name is refused (identify): a tensor
// dropped at conversion is a model that loads and is wrong, so the exceptions
// are a list, never a pattern.
var unreadTower = map[string]string{
	// The resampler's keys take a 2-D sin/cos position computed for the
	// slice's own grid (resampler.py's get_2d_sincos_pos_embed, llama.cpp's
	// minicpmv.cpp); the converter's 70x70 table of it is loaded by llama.cpp
	// and read by no node of its graph.
	"resampler.pos_embed_k": "the resampler's position is computed per grid",
}

// indexedRole is the per-expert form: "ffn_gate.3.weight" is expert 3's gate.
var indexedRole = map[string]jlm.Role{
	"ffn_gate": jlm.RoleExpGate,
	"ffn_up":   jlm.RoleExpUp,
	"ffn_down": jlm.RoleExpDown,
}

// retarget maps a role whose name means different things in different
// architectures onto what the tensor does. gemma2/gemma3's post_attention_norm
// is applied to the attention output; qwen3next's, gpt-oss's and glm4moe's tensor of that
// name is the FFN's input norm (ffn_norm by position, which they do not ship
// under that name). Resolving it here keeps the source's ambiguity out of the
// container.
func retarget(arch jlm.Arch, role jlm.Role) jlm.Role {
	if (arch == jlm.ArchQwen3Next || arch == jlm.ArchGPTOSS || arch == jlm.ArchGLM4MoE || arch == jlm.ArchGLM4VMoE ||
		arch == jlm.ArchSeedOSS) &&
		role == jlm.RolePostAttnNorm {
		return jlm.RoleFFNNorm
	}
	// DBRX's attn_output_norm is its FFN pre-norm (llama.cpp's dbrx.cpp),
	// where BERT's is applied after the residual add.
	if arch == jlm.ArchDBRX && role == jlm.RoleAttnOutNorm {
		return jlm.RoleFFNNorm
	}
	// LFM2's final norm is named token_embd_norm (llama.cpp's
	// LLM_TENSOR_OUTPUT_NORM_LFM2, "fix for wrong tensor name"); it is applied
	// after the last block, not to the embeddings.
	if arch == jlm.ArchLFM2 && role == jlm.RoleTokenEmbdNorm {
		return jlm.RoleOutputNorm
	}
	// Qwen3.8-Flash-Next's n-gram table carries Gemma's per-layer table name
	// and is another object: a row table of hashed n-grams.
	if arch == jlm.ArchQwen4Exp && role == jlm.RolePLETokEmbd {
		return jlm.RoleNGramTable
	}
	return role
}

// identify turns a source tensor name into the triple a container stores.
// Index is -1 unless the name carries one.
func identify(name string) (role jlm.Role, block int32, index int32, err error) {
	block, index = jlm.DenseBlock, -1
	rest := name
	// "v." marks the vision tower; its blocks are "v.blk.N.".
	vision := strings.HasPrefix(rest, "v.")
	// A decision head's blocks are "dec.blk.N." (Clef's): model-level tensors
	// with the head block in Index, so they never meet a backbone block.
	if r, ok := strings.CutPrefix(rest, "dec.blk."); ok {
		i := strings.IndexByte(r, '.')
		n, e := strconv.ParseInt(r[:max(i, 0)], 10, 32)
		if i <= 0 || e != nil || n < 0 {
			return 0, 0, 0, fmt.Errorf("convert: %q: no head block index", name)
		}
		if role, ok := roleOf["dec."+r[i+1:]]; ok {
			return role, jlm.DenseBlock, int32(n), nil
		}
		return 0, 0, 0, fmt.Errorf("convert: tensor %q has no role in this container's vocabulary", name)
	}
	for _, p := range []string{"v.blk.", "blk."} {
		if strings.HasPrefix(rest, p) {
			r := rest[len(p):]
			i := strings.IndexByte(r, '.')
			if i <= 0 {
				return 0, 0, 0, fmt.Errorf("convert: %q: no block index", name)
			}
			n, e := strconv.Atoi(r[:i])
			if e != nil || n < 0 {
				return 0, 0, 0, fmt.Errorf("convert: %q: block index %q", name, r[:i])
			}
			block, rest = int32(n), r[i+1:]
			if vision {
				rest = "v." + rest
			}
			break
		}
	}
	if r, ok := roleOf[rest]; ok {
		return r, block, index, nil
	}
	// The second temporal plane of Qwen2-VL's conv3d patch embedding, which the
	// GGUF spells as a second 4-D tensor. It is carried as index 1 and folded
	// into index -1 by prepareTower, so nothing indexed reaches the container.
	if rest == "v.patch_embd.weight.1" {
		return jlm.RoleVPatchEmbd, block, 1, nil
	}
	// The indexed per-expert form.
	if f := strings.Split(rest, "."); len(f) == 3 && f[2] == "weight" {
		if r, ok := indexedRole[f[0]]; ok {
			n, e := strconv.Atoi(f[1])
			if e == nil && n >= 0 {
				return r, block, int32(n), nil
			}
		}
	}
	return 0, 0, 0, fmt.Errorf("convert: tensor %q has no role in this container's vocabulary; "+
		"a tensor dropped at conversion is a model that loads and is wrong, so this is a refusal", name)
}

// typeOf maps a source weight type onto the container's own code.
//
// A type the container does not define is a refusal rather than a stored
// source number.
func typeOf(t quant.Type) (jlm.Type, error) {
	ty, ok := jlm.TypeOf(t)
	if !ok {
		return 0, fmt.Errorf("convert: weight type %s has no code in this container's vocabulary", t)
	}
	return ty, nil
}

// sourceType is typeOf's inverse, for the gate below.
func sourceType(t jlm.Type) (quant.Type, bool) { return jlm.SourceType(t) }
