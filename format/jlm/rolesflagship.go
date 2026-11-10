package jlm

// The flagship group's roles, numbered from 300: 200 and 201 are Qwen2.5-VL's
// tower gate (RoleVFFNGate), and a concurrent addition counting up from the
// last core role does not reach this far.
const (
	// RoleLayerOutScale is Gemma 4's per-block output scalar (layer_scalar):
	// one float the block's whole output row is multiplied by after its last
	// residual add. A vector of one, read once at load.
	RoleLayerOutScale Role = 300

	// Gemma 4's mixture block (Flag2DenseMoE). RoleFFNNorm2 is the experts'
	// pre-norm (pre_feedforward_layernorm_2), RolePostFFNNorm1 the dense
	// MLP's post-norm and RolePostFFNNorm2 the experts'. RoleRouterNorm is the
	// router input's RMSNorm weight: router.scale over sqrt(n_embd), folded at
	// conversion, since the router reads rmsnorm(x) * scale * n_embd^-1/2.
	// RoleExpScale is one factor per expert multiplying its renormalised
	// weight (router.per_expert_scale, llama.cpp's ffn_down_exps.scale).
	RoleFFNNorm2     Role = 301
	RolePostFFNNorm1 Role = 302
	RolePostFFNNorm2 Role = 303
	RoleRouterNorm   Role = 304
	RoleExpScale     Role = 305

	// Gemma 4's per-layer embeddings (E2B/E4B; Config.PLEDim). RolePLETokEmbd
	// is the second token table, NLayer*PLEDim wide per token; RolePLEModelProj
	// projects the scaled embedding to the same width and RolePLEProjNorm
	// RMSNorms each layer's PLEDim slice of it. Per block, RolePLEGate
	// (n_embd -> PLEDim) gates the slice, RolePLEProj (PLEDim -> n_embd)
	// projects it back and RolePLEPostNorm norms it before the residual add.
	RolePLETokEmbd   Role = 306
	RolePLEModelProj Role = 307
	RolePLEProjNorm  Role = 308
	RolePLEGate      Role = 309
	RolePLEProj      Role = 310
	RolePLEPostNorm  Role = 311

	// DeepSeek V3.2's lightning indexer (ArchDeepseek32), per block.
	// RoleIdxQB projects the query latent (the normed q_a output) to
	// IdxHeads*IdxHeadDim; RoleIdxK projects the block input to one
	// IdxHeadDim key, LayerNormed by RoleIdxKNorm/RoleIdxKNormB; RoleIdxProj
	// projects the block input to one weight per indexer head.
	RoleIdxQB     Role = 312
	RoleIdxK      Role = 313
	RoleIdxKNorm  Role = 314
	RoleIdxKNormB Role = 315
	RoleIdxProj   Role = 316

	// Predicted residual streams (Config.PredStreams, Gemma 3n's AltUp) and a
	// low-rank side branch (its LAuReL). RoleStreamExpand projects the scaled
	// embedding to the other PredStreams-1 streams, their rows stacked
	// (stream k's at (k-1)*n_embd); RoleStreamCollapse1..3 project stream k
	// back before the streams are averaged for the head.
	//
	// Per block: RoleStreamRouter (PredStreams rows of n_embd) reads
	// rmsnorm(x0) * RoleStreamRouterNorm / n_embd into one weight a stream;
	// RoleStreamPredictCoef holds the prediction coefficients, PredStreams^2
	// rows of PredStreams the router's weights multiply; RoleStreamCorrectCoef
	// the correction's, PredStreams rows of PredStreams;
	// RoleStreamCorrectScale the active stream's scale before the per-layer
	// input gate. RoleLowRankIn (n_embd -> rank) and RoleLowRankOut (rank ->
	// n_embd) are the side branch and RoleLowRankNorm its norm.
	RoleStreamExpand       Role = 317
	RoleStreamCollapse1    Role = 318
	RoleStreamCollapse2    Role = 319
	RoleStreamCollapse3    Role = 320
	RoleStreamRouter       Role = 321
	RoleStreamRouterNorm   Role = 322
	RoleStreamPredictCoef  Role = 323
	RoleStreamCorrectCoef  Role = 324
	RoleStreamCorrectScale Role = 325
	RoleLowRankIn          Role = 326
	RoleLowRankOut         Role = 327
	RoleLowRankNorm        Role = 328

	// MiniMax Sparse Attention's indexer (ArchMiniMaxM3), per sparse block.
	// RoleIdxQ projects the block's normed input to IdxHeads*IdxHeadDim (one
	// head per kv group) and RoleIdxQNorm RMSNorms each head; the key is
	// RoleIdxK with RoleIdxKNorm, the roles DeepSeek V3.2's indexer key uses,
	// here an RMSNorm with no bias.
	RoleIdxQ     Role = 329
	RoleIdxQNorm Role = 330

	// DeepSeek V4 (ArchDeepseek4). Each block's two hyper-connections, at the
	// attention and at the FFN: RoleHC*Fn projects the RMS-normed
	// concatenation of the HCMult streams to (2+HCMult)*HCMult mixes, and
	// RoleHC*Base/RoleHC*Scale (that many, and three) turn them into the
	// streams' collapse weights, the block output's placement and the
	// Sinkhorn-normalised stream mixer. RoleHCHead* is the model-level
	// collapse before the output norm (HCMult mixes, one scale).
	RoleHCAttnFn    Role = 331
	RoleHCAttnBase  Role = 332
	RoleHCAttnScale Role = 333
	RoleHCFFNFn     Role = 334
	RoleHCFFNBase   Role = 335
	RoleHCFFNScale  Role = 336
	RoleHCHeadFn    Role = 337
	RoleHCHeadBase  Role = 338
	RoleHCHeadScale Role = 339
	// RoleAttnOutA is the grouped output projection's first half: OGroups
	// rows of OLoraRank, group g reading heads g*NHead/OGroups.. of the
	// attention output; RoleAttnOut takes the OGroups*OLoraRank result to
	// n_embd.
	RoleAttnOutA Role = 340
	// A compressed block's compressor: RoleCompKV and RoleCompGate project the
	// block input to one HeadDim (HCA) or two (CSA: the next window's half,
	// then this window's) per position, RoleCompAPE is the per-slot position
	// bias added to the gate (the compression rate's rows), and RoleCompNorm
	// RMSNorms each entry. RoleIdxComp* are a CSA block's indexer compressor,
	// at IdxHeadDim; the indexer's query is RoleIdxQB on the query latent and
	// its head weights RoleIdxProj, DeepSeek V3.2's roles.
	RoleCompKV      Role = 341
	RoleCompGate    Role = 342
	RoleCompAPE     Role = 343
	RoleCompNorm    Role = 344
	RoleIdxCompKV   Role = 345
	RoleIdxCompGate Role = 346
	RoleIdxCompAPE  Role = 347
	RoleIdxCompNorm Role = 348
	// RoleHashExperts is a hash-routed block's frozen token-id table: NVocab
	// rows of NExpertUsed expert ids, stored as F32 (exact; the source's I32
	// is not a weight type here).
	RoleHashExperts Role = 349

	// Kimi-K3 (ArchKimiK3). RoleAttnResScore and RoleFFNResScore are a
	// block's two residual-attention score vectors (n_embd): each checkpoint
	// in the bank and the running residual scores sum(w * rmsnorm(v)), and
	// the sublayer reads their softmax mix. Each is the reference's
	// <x>_res_norm.weight times <x>_res_proj.weight, which llama.cpp's
	// converter writes as one product. RoleOutputResScore is the head's.
	RoleAttnResScore   Role = 350
	RoleFFNResScore    Role = 351
	RoleOutputResScore Role = 352
	// The latent mixture: RoleFFNRoutedDown takes the FFN input to
	// ExpertLatent, where the routed experts run; their weighted sum is
	// RMSNormed by RoleFFNRoutedNorm and taken back to n_embd by
	// RoleFFNRoutedUp. The router and the shared experts read the input at
	// n_embd.
	RoleFFNRoutedDown Role = 353
	RoleFFNRoutedUp   Role = 354
	RoleFFNRoutedNorm Role = 355

	// Qwen3.8-Flash-Next (ArchQwen4Exp). Each block's two hyper-connections,
	// at the attention and at the FFN: RoleHC*Norm is the grouped RMSNorm's
	// weight (HCMult rows of n_embd, the +1 folded), RoleHC*Down
	// (HCMult*n_embd -> HCLowRank) and RoleHC*Up (HCLowRank -> HCMult*n_embd)
	// make the streams' gate, and RoleHC*Inject (HCMult*n_embd -> HCMult) the
	// sublayer output's placement weights. RoleHCHead* is the head's mix,
	// which has no placement and stands in for the output norm.
	RoleHCAttnNorm   Role = 356
	RoleHCAttnDown   Role = 357
	RoleHCAttnUp     Role = 358
	RoleHCAttnInject Role = 359
	RoleHCFFNNorm    Role = 360
	RoleHCFFNDown    Role = 361
	RoleHCFFNUp      Role = 362
	RoleHCFFNInject  Role = 363
	RoleHCHeadNorm   Role = 364
	RoleHCHeadDown   Role = 365
	RoleHCHeadUp     Role = 366
	// The n-gram embedding. RoleNGramTable is the table, NGramRowDim wide,
	// stored row-major in the source's own block format and read a row at a
	// time (never resident whole). In its block: RoleNGramKey
	// (NGramHeads*NGramRowDim -> HCMult*n_embd) and RoleNGramValue (-> n_embd)
	// project the gathered rows; RoleNGramNormKey, RoleNGramNormQuery and
	// RoleNGramNormConv are grouped RMSNorm weights (HCMult rows of n_embd,
	// the +1 folded); RoleNGramConv is the depthwise convolution, HCMult*n_embd
	// channels of NGramConv taps.
	RoleNGramTable     Role = 367
	RoleNGramKey       Role = 368
	RoleNGramValue     Role = 369
	RoleNGramNormKey   Role = 370
	RoleNGramNormQuery Role = 371
	RoleNGramNormConv  Role = 372
	RoleNGramConv      Role = 373
)

func flagshipRoleName(r Role) (string, bool) {
	switch r {
	case RoleLayerOutScale:
		return "layer_output_scale", true
	case RoleFFNNorm2:
		return "pre_ffw_norm_2", true
	case RolePostFFNNorm1:
		return "post_ffw_norm_1", true
	case RolePostFFNNorm2:
		return "post_ffw_norm_2", true
	case RoleRouterNorm:
		return "router_norm", true
	case RoleExpScale:
		return "ffn_down_exps_scale", true
	case RolePLETokEmbd:
		return "per_layer_token_embd", true
	case RolePLEModelProj:
		return "per_layer_model_proj", true
	case RolePLEProjNorm:
		return "per_layer_proj_norm", true
	case RolePLEGate:
		return "inp_gate", true
	case RolePLEProj:
		return "proj", true
	case RolePLEPostNorm:
		return "post_norm", true
	case RoleIdxQB:
		return "indexer.attn_q_b", true
	case RoleIdxK:
		return "indexer.attn_k", true
	case RoleIdxKNorm:
		return "indexer.k_norm", true
	case RoleIdxKNormB:
		return "indexer.k_norm_bias", true
	case RoleIdxProj:
		return "indexer.proj", true
	case RoleStreamExpand:
		return "stream_expand", true
	case RoleStreamCollapse1:
		return "stream_collapse.1", true
	case RoleStreamCollapse2:
		return "stream_collapse.2", true
	case RoleStreamCollapse3:
		return "stream_collapse.3", true
	case RoleStreamRouter:
		return "stream_router", true
	case RoleStreamRouterNorm:
		return "stream_router_norm", true
	case RoleStreamPredictCoef:
		return "stream_predict_coef", true
	case RoleStreamCorrectCoef:
		return "stream_correct_coef", true
	case RoleStreamCorrectScale:
		return "stream_correct_scale", true
	case RoleLowRankIn:
		return "lowrank_in", true
	case RoleLowRankOut:
		return "lowrank_out", true
	case RoleLowRankNorm:
		return "lowrank_post_norm", true
	case RoleIdxQ:
		return "indexer.q_proj", true
	case RoleIdxQNorm:
		return "indexer.q_norm", true
	case RoleHCAttnFn:
		return "hc_attn_fn", true
	case RoleHCAttnBase:
		return "hc_attn_base", true
	case RoleHCAttnScale:
		return "hc_attn_scale", true
	case RoleHCFFNFn:
		return "hc_ffn_fn", true
	case RoleHCFFNBase:
		return "hc_ffn_base", true
	case RoleHCFFNScale:
		return "hc_ffn_scale", true
	case RoleHCHeadFn:
		return "output_hc_fn", true
	case RoleHCHeadBase:
		return "output_hc_base", true
	case RoleHCHeadScale:
		return "output_hc_scale", true
	case RoleAttnOutA:
		return "attn_output_a", true
	case RoleCompKV:
		return "attn_compressor_kv", true
	case RoleCompGate:
		return "attn_compressor_gate", true
	case RoleCompAPE:
		return "attn_compressor_ape", true
	case RoleCompNorm:
		return "attn_compressor_norm", true
	case RoleIdxCompKV:
		return "indexer_compressor_kv", true
	case RoleIdxCompGate:
		return "indexer_compressor_gate", true
	case RoleIdxCompAPE:
		return "indexer_compressor_ape", true
	case RoleIdxCompNorm:
		return "indexer_compressor_norm", true
	case RoleHashExperts:
		return "ffn_gate_tid2eid", true
	case RoleAttnResScore:
		return "attn_res_score", true
	case RoleFFNResScore:
		return "ffn_res_score", true
	case RoleOutputResScore:
		return "output_res_score", true
	case RoleFFNRoutedDown:
		return "ffn_routed_down", true
	case RoleFFNRoutedUp:
		return "ffn_routed_up", true
	case RoleFFNRoutedNorm:
		return "ffn_routed_norm", true
	case RoleHCAttnNorm:
		return "hc_attn_norm", true
	case RoleHCAttnDown:
		return "hc_attn_down", true
	case RoleHCAttnUp:
		return "hc_attn_up", true
	case RoleHCAttnInject:
		return "hc_attn_inject", true
	case RoleHCFFNNorm:
		return "hc_ffn_norm", true
	case RoleHCFFNDown:
		return "hc_ffn_down", true
	case RoleHCFFNUp:
		return "hc_ffn_up", true
	case RoleHCFFNInject:
		return "hc_ffn_inject", true
	case RoleHCHeadNorm:
		return "output_hc_norm", true
	case RoleHCHeadDown:
		return "output_hc_down", true
	case RoleHCHeadUp:
		return "output_hc_up", true
	case RoleNGramTable:
		return "ngram_token_embd", true
	case RoleNGramKey:
		return "ple_key", true
	case RoleNGramValue:
		return "ple_value", true
	case RoleNGramNormKey:
		return "ple_norm_key", true
	case RoleNGramNormQuery:
		return "ple_norm_query", true
	case RoleNGramNormConv:
		return "ple_norm_conv", true
	case RoleNGramConv:
		return "ple_conv1d", true
	}
	return "", false
}

// flagshipExpanded is Role.Expanded for the roles above: the per-block vectors
// a reader copies out at load.
func flagshipExpanded(r Role) bool {
	switch r {
	case RoleLayerOutScale, RoleFFNNorm2, RolePostFFNNorm1, RolePostFFNNorm2,
		RoleRouterNorm, RoleExpScale, RolePLEProjNorm, RolePLEPostNorm, RoleIdxKNorm, RoleIdxKNormB,
		RoleStreamRouterNorm, RoleStreamPredictCoef, RoleStreamCorrectCoef, RoleStreamCorrectScale, RoleLowRankNorm,
		RoleIdxQNorm, RoleHCAttnBase, RoleHCAttnScale, RoleHCFFNBase, RoleHCFFNScale, RoleCompAPE,
		RoleCompNorm, RoleIdxCompAPE, RoleIdxCompNorm, RoleHashExperts, RoleAttnResScore,
		RoleFFNResScore, RoleOutputResScore, RoleFFNRoutedNorm, RoleHCAttnNorm, RoleHCFFNNorm,
		RoleHCHeadNorm, RoleHCHeadBase, RoleHCHeadScale, RoleNGramNormKey, RoleNGramNormQuery, RoleNGramNormConv, RoleNGramConv:
		return true
	}
	return false
}
