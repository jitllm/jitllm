//go:build amd64 || arm64

package cpu

import (
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// Emitters is every kernel emitter one ISA tier offers, as one table.
//
// One table because a tier is one answer: nn asks EmittersFor(tier) once and
// every emission goes through it, so an op cannot be taken from one tier and
// its neighbour from another. The primary table wraps the existing emitters
// without changing their bytes (emitters_test.go compares them).
//
// Every field returns ([]byte, error). nn treats an error as it treats a failed
// Map: mustMap panics with the op's name for a kernel a token cannot run
// without, and the packed/float builders skip a kernel that is optional.
//
// The ABI of each field is the AVX2 emitter's of the same name (its comment is
// the contract): Args is unchanged, cpu.ElemLanes stays 8 on every tier (an
// SSE body runs each unit as two XMM halves -- elemLoopSSE), and
// PackedFusedGroupOf, PackedRows, PackedTail, the PackedScratch layout and
// PackedRowConsts are shared by both tiers.
type Emitters struct {
	// Tier is the tier these emitters generate for.
	Tier Tier

	// ---- family 1: elementwise and softmax (act.go, axpy.go, softmax.go) ----
	// Args.K = n/ElemLanes whole units, Args.Rows = n%ElemLanes tail elements,
	// Scr = ActConsts() (or &alpha for Axpy/Scale, {2/c, c} in W for Softcap).
	Axpy    func() ([]byte, error)
	Scale   func() ([]byte, error)
	Softcap func() ([]byte, error)
	// Clamp is x = min(max(x, lo), hi) with Scr = {lo, hi}: DBRX's clip_qkv.
	Clamp func() ([]byte, error)
	// PrefetchLines hints Args.K 64-byte lines from Args.Out into the outer
	// caches: a fused region's lookahead (prefetchlines.go).
	PrefetchLines func() ([]byte, error)
	Softmax       func() ([]byte, error)
	// LogSoftmax is Softmax's log form, the same Args and Scr = ActConsts():
	// x - max - ln(sum exp(x - max)), in place (EmitLogSoftmax).
	LogSoftmax func() ([]byte, error)
	ActMul     func(k ActKind) ([]byte, error) // every kind in Gated
	Act        func(k ActKind) ([]byte, error) // every kind in Ungated
	SigmoidMul func() ([]byte, error)
	// Ln is x = ln(x) in place over positive finite x, Scr = ActConsts()
	// (EmitLn): a log-mel feature's log.
	Ln func() ([]byte, error)
	// XIELU is Apertus's ungated activation over a runtime length, its four
	// numbers a buffer (xielu_const.go has the contract).
	XIELU func() ([]byte, error)

	// ---- the image path (resample.go, imgproc_const.go). Not optional: a
	// picture has no other way through. ----
	//
	// Resample is one output row of the vertical resampling pass at
	// precision prec: uint8 samples, Cols taps, RowStr bytes between the
	// source rows, Scr = ResampleConsts(prec). ResampleH is the horizontal
	// pass, PixLUT the per-channel table lookup, Copy32 the strided copy, and
	// PixelRow one row of a decoded picture to what the processors read.
	Resample  func(prec int) ([]byte, error)
	ResampleH func(prec int) ([]byte, error)
	PixLUT    func() ([]byte, error)
	Copy32    func() ([]byte, error)
	PixelRow  func(f PixFmt, sub int, o PixOut) ([]byte, error)
	// AxisTaps and LerpGrid are a resampled position table's per-axis taps and
	// weights and their 2-D outer product (imgproc_const.go).
	AxisTaps func(f AxisFilter) ([]byte, error)
	LerpGrid func() ([]byte, error)
	// SinCosTab is MiniCPM-V's resampler position table along one axis: a
	// power, a sine and a cosine per entry, in float64 (imgproc_const.go).
	SinCosTab func() ([]byte, error)

	// ---- family 2: norms, rotary, embedding row (norm.go, layernorm.go,
	// rope.go, rowpacked.go) ----
	RMSNorm   func(n int) ([]byte, error)
	LayerNorm func(n int, bias bool) ([]byte, error)
	// RowStat is the layer norm's passes turned to Gemma 3n's two row ops
	// (mode RowGauss, RowMag or RowStd; see EmitGaussTopK, EmitMagMatch and
	// EmitRowStd).
	RowStat func(n int, mode RowMode) ([]byte, error)
	RoPE    func(hd, nrot int, neox bool) ([]byte, error)
	// RoPESplit is XD-RoPE's rotation (HunyuanVL): NEOX pairs whose halves
	// turn by tables of their own (EmitRoPESplit).
	RoPESplit func(hd, nrot int) ([]byte, error)
	PackedRow func(q kernels.Quant) ([]byte, error)
	Widen     func(bf16 bool) ([]byte, error)
	// NarrowF16 is Widen's inverse for binary16, the f16 KV cache's store:
	// Args.K whole units of ElemLanes and Args.Rows singles of float32 (W)
	// rounded to binary16 (Out) exactly as quant.EncodeHalf rounds.
	NarrowF16 func() ([]byte, error)
	// RopeTable is the rotary table -- the {cos, sin} pair per rotary pair of
	// one position -- where RoPE above is its application to a run of heads.
	// npairs is baked; Out is the table, AScale the per-model plane block,
	// Scr = RopeTabConsts() and K the position. ropetab.go has the contract.
	// Not optional: there is no other way to build the table, so every tier
	// generates it.
	RopeTable func(npairs int) ([]byte, error)
	// HCMix and ColPool are DeepSeek V4's hyper-connection mixer and its
	// compressor's pool over positions; hc_const.go has both contracts.
	HCMix func(iters int, head bool) ([]byte, error)
	// NGramHash is Qwen3.8-Flash-Next's n-gram table rows for a position,
	// integer code one emitter serves on both amd64 tiers (ngram_const.go has
	// the contract).
	NGramHash func(n, per int, reset bool) ([]byte, error)
	// RowIQ4NL widens IQ4_NL blocks to floats: the n-gram table's rows
	// (iq4nl_x86.go has the contract).
	RowIQ4NL func() ([]byte, error)
	ColPool  func() ([]byte, error)

	// ---- family 3: attention (attn.go). All six non-tiled kernels are
	// mandatory: nn.AddAttn maps every one of them. ----
	AttnScores      func(hd, kvStride int, kv KVFmt) ([]byte, error)
	AttnAcc         func(hd, kvStride int, kv KVFmt) ([]byte, error)
	AttnAccInto     func(hd, kvStride int, kv KVFmt) ([]byte, error)
	AttnScores2     func(hd, kvStride int, kv KVFmt) ([]byte, error)
	AttnAcc2        func(hd, kvStride int, kv KVFmt) ([]byte, error)
	AttnAcc2Into    func(hd, kvStride int, kv KVFmt) ([]byte, error)
	AttnScoresTiled func(hd, kvStride, qStride, scoreStride, qt int, kv KVFmt) ([]byte, error)
	// KVWiden is the q8 cache's dequantization to float32 rows (EmitKVWiden).
	KVWiden func(hd int) ([]byte, error)

	// ---- family 4: matvec (matvec.go's float path, packed.go,
	// packedfused.go). ----
	//
	// RowMajorSupported/RowMajor are the row-major family: the float matvec
	// (F32/F16/BF16 routers and verbatim float matrices) on every tier, and the
	// quantized GGUF kernels on the AVX2 tier only. nn builds Spec{Rows: 1};
	// an emitter that serves no interleaved form returns an error for
	// Rows != 1 and nn simply has no f.code4 for that type.
	RowMajorSupported func(t quant.Type) bool
	RowMajor          func(s Spec) ([]byte, error)
	// FloatMatMul is the float GEMM over a tile of 1..MaxFloatTokens tokens
	// (EmitFloatMatMul): F32, F16 or BF16 rows read once for the tile, Args
	// as the float matvec's plus OutStr between the tokens' outputs.
	FloatMatMul func(t quant.Type, nt int) ([]byte, error)
	// RowMajorGGUF reports whether the rest of the GGUF row-major machinery
	// exists on this tier: nn.MatMul's token-tiled GEMM and AddShape's
	// interleaved pack-width kernels. It does not on the SSE tier (a container
	// never decodes through either), and nn declines both there without
	// emitting anything.
	RowMajorGGUF bool
	// PackedSupported is "can this tier run the packed kernel for t" (the
	// container decode path); PackedMatVec emits the 64-row tile or the 8-row
	// tail (rows = PackedRows or PackedTail), PackedFused the fused kernel
	// (Rows = groups of PackedFusedGroupOf(t)). PackedWide and PackedTiled are
	// the wide and token-tiled kernels, which the SSE tier refuses:
	// MatMulPacked then runs the fused kernel per token, the same answer.
	PackedSupported func(t quant.Type) bool
	PackedMatVec    func(t quant.Type, rows int) ([]byte, error)
	PackedWide      func(t quant.Type) ([]byte, error)
	PackedFused     func(t quant.Type) ([]byte, error)
	// PackedFusedAhead is PackedFused with a software prefetch `ahead` row
	// groups in front; a tier with no such form refuses, and nn does not tune.
	PackedFusedAhead func(t quant.Type, ahead int) ([]byte, error)
	// PackedFusedWin is PackedFused at activation window win, with the
	// integer super-block fold where that window allows it (arm64). nil, or
	// refusing, means the float kernel is the tier's only form.
	PackedFusedWin func(t quant.Type, win int) ([]byte, error)
	PackedTiled    func(t quant.Type, k, nrows, tok int) ([]byte, error)
	MaxTiledTokens func(t quant.Type) int
	// PackedGEMM is the weight-stationary prefill GEMM (packedgemm.go): one
	// kernel per (t, k, nrows) serving any number of tokens, Args.Cols of them
	// a call, with Args.Rows counted in eight-row groups; win is the
	// activation window the JIT quantizes with. A tier without one
	// refuses and MatMulPacked keeps the tiled/fused path.
	PackedGEMM func(t quant.Type, k, nrows, win int) ([]byte, error)
	// QuantAct is the activation side of that same path: one amax window of
	// float32 in, int8 plus the {scale, -sum*biasC/8} pairs every packed kernel
	// reads out. half bakes whether the format also needs per-16 sums
	// (NeedsHalfSums). Its absence is an ordinary error, not ErrNoSSEKernel.
	QuantAct func(half bool) ([]byte, error)
	// QuantActNarrow is the same op at a window of one block, a different
	// kernel rather than a flag: for Q4_0 and Q8_0 (a scale every 32 elements)
	// the wide form's fold and divisions would dominate. It quantizes eight
	// blocks an iteration and divides once.
	QuantActNarrow func(half bool) ([]byte, error)
	// Argmax is the greedy sampler's, over a whole logits vector: the index of
	// the largest element with the lowest index winning a tie. Optional on the
	// same terms as QuantAct -- model.Greedy's Go scan is the fallback.
	Argmax func() ([]byte, error)
	// MoETopK is the mixture router's whole selection: the k largest of n
	// probabilities in selection order with the lowest id winning a tie, the
	// same ids ascending, both orders' renormalised weights, and the divisor.
	// n, k and whether the architecture renormalises are baked. topk.go has the
	// contract. Not optional: there is no other top-k, so a tier without it
	// returns an error naming the router (not a Baseline refusal, since a dense
	// model on that host runs fine).
	MoETopK func(n, k int, norm bool) ([]byte, error)
	// MoERoute is that same kernel generalised to the DeepSeek-V3 router
	// (R1, Kimi-K2 and GLM-4-MoE share it), adding a selection bias plane,
	// expert groups and a scaling factor. cpu.MoEGate is the configuration and
	// topk.go has the contract. MoETopK is this function at
	// MoEGate{Norm: norm}: one body behind both fields.
	MoERoute func(n, k int, g MoEGate) ([]byte, error)
	// SparseMixer is Phi-3.5-MoE's router weights for n experts, run after
	// MoERoute has selected the top two of the raw logits:
	// sparsemixer_const.go has the contract. Not optional, for MoETopK's
	// reason.
	SparseMixer func(n int) ([]byte, error)
	// SampleSegMax, SampleDraw and SamplePenalty are the sampler's three
	// kernels: the ordering's segmented eligible maximum, the min-p/top-p cut
	// with the inverse-CDF walk, and the repeat penalty's scatter.
	// sample.go has the contract. Not optional, for MoETopK's reason.
	SampleSegMax  func(first, idsMem bool) ([]byte, error)
	SampleDraw    func() ([]byte, error)
	SamplePenalty func() ([]byte, error)

	// ---- family 5: the hybrid's recurrence (delta.go, conv.go, gate.go) ----
	GatedDelta func(n int) ([]byte, error)
	// GatedDeltaChan is GatedDelta with a per-channel decay in AScale2 --
	// Kimi-Linear's delta rule, where qwen3next's is the scalar one above.
	GatedDeltaChan func(n int) ([]byte, error)
	Conv1d         func(taps, chans int) ([]byte, error)
	DeltaGate      func() ([]byte, error) // Scr = DeltaGateConsts()
	// DeltaDecayBound is Kimi-K3's KDA decay under gate_lower_bound,
	// exp(lb*sigma(-A*(a+dt))), in place of DeltaGate's softplus decay.
	DeltaDecayBound func() ([]byte, error) // Scr = DeltaGateConsts()
	// GatedSSD is Mamba-2's selective state update, the delta rule with the
	// key dot removed and a D skip; SSDGate its dt and decay (delta.go and
	// gate.go have the contracts).
	GatedSSD func(n int) ([]byte, error)
	SSDGate  func() ([]byte, error) // Scr = DeltaGateConsts()
	// SelScan is Mamba-1's selective scan: GatedSSD per channel, with the
	// decay a vector over the state (delta.go has the contract).
	SelScan func(n int) ([]byte, error)

	// ---- family 6: the convolutional tower (dwconv.go) ----
	// DWConv is one output row of a depthwise convolution over channel-last
	// rows, its whole shape baked (DWShape).
	DWConv func(s DWShape) ([]byte, error)
}

// ErrNoSSEKernel is what an SSE-tier stub returns for an op that has no SSE
// kernel. It is not a refusal: the integration gate requires that
// no field returns it.
var ErrNoSSEKernel = errors.New("jit: this op has no SSE-tier kernel yet")

// errNoSSE wraps ErrNoSSEKernel with the op's name, for the stubs.
func errNoSSE(op string) error { return fmt.Errorf("%w (%s)", ErrNoSSEKernel, op) }

// EmittersFor returns the emitter table for tier t. TierSSE is the legacy-SSE
// table; TierAVX2 (amd64) and TierNEON (arm64) are the primary table, whose
// names resolve to that architecture's emitters. TierNone gets the primary
// table too: on such a host Map refuses every one of its kernels by name
// (ISAError), which is the refusal that host is owed.
func EmittersFor(t Tier) *Emitters {
	if t == TierSSE {
		return &sseEmitters
	}
	return &primaryEmitters
}

// EmitOpts are the emission choices that belong to one JIT rather than to the
// process. Zero is the shipping default.
type EmitOpts struct {
	// Prefetch is the PREFETCHT0 distance in super-blocks on the amd64 GGUF
	// k-quant kernels; 0 emits none, which is the default until it is measured.
	Prefetch int
	// A64Prefetch is how many bytes ahead of each payload line the arm64
	// packed matvec issues a PRFM: 0 takes the default, one tile (rows*4
	// bytes), and a negative value emits none. See EmitA64PackedMatVecPF.
	A64Prefetch int
	// Probes is arm64 measurement scaffolding: it injects dead operations into
	// the k-quant decode kernel so one class can be priced against another.
	// Never set outside a measurement; it makes the kernel deliberately slower.
	Probes Probes
}

// Probes counts the dead scalar adds, 128-bit loads and vector adds injected
// per block into the arm64 k-quant decode matvec. 0 disables each.
type Probes struct{ Scalar, Load, Vector int }

// EmittersWith is EmittersFor(t) with o applied to the emitters it reaches: a
// copy of the table, so two JITs in one process emit with their own choices.
func EmittersWith(t Tier, o EmitOpts) *Emitters {
	base := EmittersFor(t)
	if o == (EmitOpts{}) || t == TierSSE {
		return base
	}
	e := *base
	e.RowMajor = func(s Spec) ([]byte, error) { return rowMajorPF(s, o) }
	e.PackedMatVec = func(q quant.Type, rows int) ([]byte, error) { return packedMatVecPF(q, rows, o.A64Prefetch) }
	return &e
}

// Native is EmittersFor(HostTier()): the table a caller about to EXECUTE uses.
func Native() *Emitters { return EmittersFor(HostTier()) }

// must adapts an emitter that cannot fail. An empty result is still an error:
// an architecture stub (delta_stub.go) returns nil, and mapping nothing is
// what Map refused before this table existed.
func must(op string, b []byte) ([]byte, error) {
	if len(b) == 0 {
		return nil, fmt.Errorf("jit: %s: the emitter produced no code on this architecture", op)
	}
	return b, nil
}

// primaryEmitters is the table this architecture has always had: the AVX2
// tier on amd64, NEON on arm64. Every field is the existing function.
var primaryEmitters = Emitters{
	Tier: primaryTier,

	Axpy:    func() ([]byte, error) { return must("axpy", EmitAxpy()) },
	Scale:   func() ([]byte, error) { return must("scale", EmitScale()) },
	Softcap: func() ([]byte, error) { return must("softcap", EmitSoftcap()) },
	Clamp:   func() ([]byte, error) { return must("clamp", EmitClamp()) },
	PrefetchLines: func() ([]byte, error) {
		return must("prefetch_lines", EmitPrefetchLines())
	},
	Resample:   func(prec int) ([]byte, error) { return must("resample", EmitResample(prec)) },
	ResampleH:  func(prec int) ([]byte, error) { return must("resample_h", EmitResampleH(prec)) },
	PixLUT:     func() ([]byte, error) { return must("pixlut", EmitPixLUT()) },
	Copy32:     func() ([]byte, error) { return must("copy32", EmitCopy32()) },
	PixelRow:   EmitPixelRow,
	AxisTaps:   func(f AxisFilter) ([]byte, error) { return must("axistaps", EmitAxisTaps(f)) },
	LerpGrid:   func() ([]byte, error) { return must("lerpgrid", EmitLerpGrid()) },
	SinCosTab:  func() ([]byte, error) { return must("sincostab", EmitSinCosTab()) },
	Softmax:    func() ([]byte, error) { return must("softmax", EmitSoftmax()) },
	LogSoftmax: func() ([]byte, error) { return must("logsoftmax", EmitLogSoftmax()) },
	ActMul:     func(k ActKind) ([]byte, error) { return must("actmul", EmitActMul(k)) },
	Act:        func(k ActKind) ([]byte, error) { return must("act", EmitAct(k)) },
	SigmoidMul: func() ([]byte, error) { return must("sigmoidmul", EmitSigmoidMul()) },
	Ln:         func() ([]byte, error) { return must("ln", EmitLn()) },
	XIELU:      func() ([]byte, error) { return must("xielu", EmitXIELU()) },

	RMSNorm:   func(n int) ([]byte, error) { return must("rmsnorm", EmitRMSNorm(n)) },
	LayerNorm: func(n int, bias bool) ([]byte, error) { return must("layernorm", EmitLayerNorm(n, bias)) },
	RowStat: func(n int, mode RowMode) ([]byte, error) {
		switch mode {
		case RowGauss:
			return must("gausstopk", EmitGaussTopK(n))
		case RowMag:
			return must("magmatch", EmitMagMatch(n))
		case RowStd:
			return must("rowstd", EmitRowStd(n))
		}
		return nil, fmt.Errorf("jit: row statistic %d", mode)
	},
	RoPE:      EmitRoPE,
	RoPESplit: EmitRoPESplit,
	PackedRow: EmitPackedRow,
	Widen:     func(bf16 bool) ([]byte, error) { return must("widen", EmitWiden(bf16)) },
	NarrowF16: func() ([]byte, error) { return must("narrow_f16", EmitNarrowF16()) },
	RopeTable: EmitRopeTable,
	HCMix:     EmitHCMix,
	NGramHash: EmitNGramHash,
	RowIQ4NL:  func() ([]byte, error) { return must("row_iq4nl", EmitRowIQ4NL()) },
	ColPool:   func() ([]byte, error) { return must("colpool", EmitColPool()) },

	AttnScores:      EmitAttnScores,
	AttnAcc:         EmitAttnAcc,
	AttnAccInto:     EmitAttnAccInto,
	AttnScores2:     EmitAttnScores2,
	AttnAcc2:        EmitAttnAcc2,
	AttnAcc2Into:    EmitAttnAcc2Into,
	AttnScoresTiled: EmitAttnScoresTiled,
	KVWiden:         EmitKVWiden,

	RowMajorSupported: primaryRowMajorSupported,
	RowMajor:          EmitNative,
	FloatMatMul:       EmitFloatMatMul,
	RowMajorGGUF:      true,
	PackedSupported:   primaryPackedSupported,
	// The packed emitters take the host's dot sequence at EMISSION time, as
	// engine/nn/jit.go always passed it -- a closure rather than a captured value, so
	// a re-probe (ForceNoVNNIForTest) is seen by the next emission.
	PackedMatVec: func(t quant.Type, rows int) ([]byte, error) { return EmitPackedMatVec(t, rows, HostDotKind()) },
	PackedWide:   func(t quant.Type) ([]byte, error) { return EmitPackedMatVecWide(t, HostDotKind()) },
	PackedFused:  func(t quant.Type) ([]byte, error) { return EmitPackedMatVecFused(t, HostDotKind()) },
	PackedFusedAhead: func(t quant.Type, ahead int) ([]byte, error) {
		return EmitPackedMatVecFusedAhead(t, HostDotKind(), ahead)
	},
	PackedFusedWin: EmitPackedMatVecFusedWin,
	PackedTiled: func(t quant.Type, k, nrows, tok int) ([]byte, error) {
		return EmitPackedMatMulTiled(t, k, nrows, tok, HostDotKind())
	},
	MaxTiledTokens: MaxTiledTokens,
	PackedGEMM: func(t quant.Type, k, nrows, win int) ([]byte, error) {
		return EmitPackedMatMulStationary(t, k, nrows, win, HostDotKind())
	},
	QuantAct:       func(half bool) ([]byte, error) { return must("quantact", EmitQuantAct(half)) },
	QuantActNarrow: func(half bool) ([]byte, error) { return must("quantactnarrow", EmitQuantActNarrow(half)) },
	Argmax:         func() ([]byte, error) { return must("argmax", EmitArgmax()) },
	MoETopK:        EmitMoETopK,
	MoERoute:       EmitMoERoute,
	SparseMixer:    EmitSparseMixer,
	SampleSegMax:   EmitSampleSegMax,
	SampleDraw:     EmitSampleDraw,
	SamplePenalty:  EmitSamplePenalty,

	GatedDelta:     EmitGatedDelta,
	GatedDeltaChan: EmitGatedDeltaChan,
	Conv1d:         EmitConv1d,
	DeltaGate:      func() ([]byte, error) { return must("delta_gate", EmitDeltaGate()) },
	DeltaDecayBound: func() ([]byte, error) {
		return must("delta_decay_bound", EmitDeltaDecayBound())
	},
	GatedSSD: EmitGatedSSD,
	SSDGate:  func() ([]byte, error) { return must("ssd_gate", EmitSSDGate()) },
	SelScan:  EmitSelScan,

	DWConv: EmitDWConv,
}

// sseEmitters is the legacy-SSE tier. The matvec family's wide and tiled
// refusals live in sse_refuse.go.
var sseEmitters = Emitters{
	Tier: TierSSE,

	Axpy:          EmitAxpySSE,
	Scale:         EmitScaleSSE,
	Softcap:       EmitSoftcapSSE,
	Clamp:         EmitClampSSE,
	PrefetchLines: EmitPrefetchLinesSSE,
	Resample:      EmitResampleSSE,
	ResampleH:     EmitResampleHSSE,
	PixLUT:        EmitPixLUTSSE,
	Copy32:        EmitCopy32SSE,
	PixelRow:      EmitPixelRowSSE,
	AxisTaps:      EmitAxisTapsSSE,
	LerpGrid:      EmitLerpGridSSE,
	SinCosTab:     EmitSinCosTabSSE,
	Softmax:       EmitSoftmaxSSE,
	LogSoftmax:    EmitLogSoftmaxSSE,
	ActMul:        EmitActMulSSE,
	Act:           EmitActSSE,
	SigmoidMul:    EmitSigmoidMulSSE,
	Ln:            EmitLnSSE,
	XIELU:         EmitXIELUSSE,

	RMSNorm:   EmitRMSNormSSE,
	LayerNorm: EmitLayerNormSSE,
	RowStat: func(n int, mode RowMode) ([]byte, error) {
		switch mode {
		case RowGauss:
			return EmitGaussTopKSSE(n)
		case RowMag:
			return EmitMagMatchSSE(n)
		case RowStd:
			return EmitRowStdSSE(n)
		}
		return nil, fmt.Errorf("jit: row statistic %d", mode)
	},
	RoPE:      EmitRoPESSE,
	RoPESplit: EmitRoPESplitSSE,
	PackedRow: EmitPackedRowSSE,
	Widen:     EmitWidenSSE,
	NarrowF16: EmitNarrowF16SSE,
	RopeTable: EmitRopeTableSSE,
	HCMix:     EmitHCMixSSE,
	NGramHash: EmitNGramHashSSE,
	RowIQ4NL:  EmitRowIQ4NLSSE,
	ColPool:   EmitColPoolSSE,

	AttnScores:      EmitAttnScoresSSE,
	AttnAcc:         EmitAttnAccSSE,
	AttnAccInto:     EmitAttnAccIntoSSE,
	AttnScores2:     EmitAttnScores2SSE,
	AttnAcc2:        EmitAttnAcc2SSE,
	AttnAcc2Into:    EmitAttnAcc2IntoSSE,
	AttnScoresTiled: EmitAttnScoresTiledSSE,
	KVWiden:         EmitKVWidenSSE,

	RowMajorSupported: SupportedRowMajorSSE,
	RowMajor:          EmitRowMajorSSE,
	FloatMatMul:       EmitFloatMatMulSSE,
	RowMajorGGUF:      false,
	PackedSupported:   SupportedPackedSSE,
	PackedMatVec:      EmitPackedMatVecSSE,
	PackedWide:        EmitPackedMatVecWideSSE,
	PackedFused:       EmitPackedMatVecFusedSSE,
	PackedFusedAhead:  EmitPackedMatVecFusedAheadSSE,
	PackedFusedWin:    EmitPackedMatVecFusedWinSSE,
	PackedTiled:       EmitPackedMatMulTiledSSE,
	MaxTiledTokens:    MaxTiledTokensSSE,
	PackedGEMM:        EmitPackedGEMMSSE,
	QuantAct:          EmitQuantActSSE,
	QuantActNarrow:    EmitQuantActNarrowSSE,
	Argmax:            EmitArgmaxSSE,
	MoETopK:           EmitMoETopKSSE,
	MoERoute:          EmitMoERouteSSE,
	SparseMixer:       EmitSparseMixerSSE,
	SampleSegMax:      EmitSampleSegMaxSSE,
	SampleDraw:        EmitSampleDrawSSE,
	SamplePenalty:     EmitSamplePenaltySSE,

	GatedDelta:      EmitGatedDeltaSSE,
	GatedDeltaChan:  EmitGatedDeltaChanSSE,
	Conv1d:          EmitConv1dSSE,
	DeltaGate:       EmitDeltaGateSSE,
	DeltaDecayBound: EmitDeltaDecayBoundSSE,
	GatedSSD:        EmitGatedSSDSSE,
	SSDGate:         EmitSSDGateSSE,
	SelScan:         EmitSelScanSSE,

	DWConv: EmitDWConvSSE,
}
