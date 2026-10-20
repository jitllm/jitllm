// Package lowertest runs every kernel jitllm ships through all three
// lowerers, and through the vendors' own validators (ptxas, spirv-val) where
// they are installed. Unit tests of the lowerers miss what an assembler
// rejects (a div.f32 with no rounding modifier once reached hardware as a bare
// CUresult=218). Without the validators the lowering itself is still
// exercised.
package lowertest

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/gpu/msl"
	"github.com/jitllm/jitllm/jit/gpu/ptx"
	"github.com/jitllm/jitllm/jit/gpu/spirv"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

func all(t *testing.T) map[string]*ir.Kernel {
	t.Helper()
	out := map[string]*ir.Kernel{}
	add := func(name string, f func() (*ir.Kernel, error)) {
		k, err := f()
		if err != nil {
			t.Fatalf("build %s: %v", name, err)
		}
		out[name] = k
	}
	for _, splits := range []int{1, 8} {
		shape := mainkernels.FlashShape{Heads: 4, KVHeads: 2, Dim: 64, Rows: 3, KStride: 4096, Scale: .125, Window: 97, F16: true, Sink: true, Softcap: 3, Splits: splits}
		add(fmt.Sprintf("flash/split%d", splits), func() (*ir.Kernel, error) { return mainkernels.FlashAttention(shape) })
		if splits > 1 {
			add("flash/merge", func() (*ir.Kernel, error) { return mainkernels.FlashAttentionMerge(shape) })
		}
	}
	// MiniMax Sparse Attention's block selection (kernels/msa.go): two rows,
	// two kv groups of 64, blocks of 16, the top 4 over 256 positions, the
	// indexer's key the pool row's third head.
	add("msa/scores", func() (*ir.Kernel, error) { return mainkernels.MSAScoresPaged(2, 2, 64, 256, 64, 192, 128) })
	add("msa/blockmax", func() (*ir.Kernel, error) { return mainkernels.MSABlockMax(2, 2, 256, 16, 1) })
	add("msa/select", func() (*ir.Kernel, error) { return mainkernels.MSASelect(2, 2, 256, 16, 4) })
	add("msa/compact", func() (*ir.Kernel, error) { return mainkernels.MSACompact(2, 2, 256, 16, 4) })
	add("msa/gatherk", func() (*ir.Kernel, error) { return mainkernels.MSAGatherK(2, 2, 64, 16, 4, 64, 192) })
	add("msa/gatherv-f16", func() (*ir.Kernel, error) { return mainkernels.MSAGatherV(2, 2, 64, 16, 4, 64, 192, true) })
	add("msa/desc", func() (*ir.Kernel, error) { return mainkernels.MSADesc(2, 16, 4, 64) })
	// DeepSeek V4 (kernels/ds4.go): two rows of 64-wide streams, a CSA row of
	// 32-wide heads (key, two kv, two gate, two of each indexer head) on
	// 64-position pages at rate 4, a window of 8 and the indexer's top 3.
	add("ds4/hcflat", func() (*ir.Kernel, error) { return mainkernels.HCFlat(64, 2) })
	add("ds4/hcmix", func() (*ir.Kernel, error) { return mainkernels.HCMix(2, 20, 1e-6) })
	add("ds4/hccollapse", func() (*ir.Kernel, error) { return mainkernels.HCCollapse(64, 2) })
	add("ds4/hcpost", func() (*ir.Kernel, error) { return mainkernels.HCPost(64, 2, false) })
	add("ds4/ropetail-inv", func() (*ir.Kernel, error) { return mainkernels.RoPETail(4, 32, 16, 2, true) })
	add("ds4/apeadd", func() (*ir.Kernel, error) { return mainkernels.DS4ApeAdd(64, 4, 2) })
	add("ds4/pool-csa", func() (*ir.Kernel, error) { return mainkernels.DS4Pool(2, 64, 288, 32, 96, 32, 4, true, false) })
	add("ds4/entwrite", func() (*ir.Kernel, error) { return mainkernels.DS4EntWrite(32, 2, 64, 64, 32, 4) })
	add("ds4/entdesc", func() (*ir.Kernel, error) { return mainkernels.DS4EntDesc(2, 4) })
	csa := mainkernels.DS4Keys{Window: 8, Rate: 4, TopK: 3}
	add("ds4/gather-csa", func() (*ir.Kernel, error) { return mainkernels.DS4Gather(2, 11, 32, 64, 288, 64, csa) })
	add("ds4/desc-csa", func() (*ir.Kernel, error) { return mainkernels.DS4Desc(2, 11, 64, csa) })
	add("ds4/routebias-hash", func() (*ir.Kernel, error) { return mainkernels.DS4RouteBias(8, 2, 100, 2, true) })
	// Kimi-K3's residual attention (kernels/k3.go): two rows of 64, three
	// checkpoints banked of four streams, the checkpoint at stream 2.
	add("k3/resscores", func() (*ir.Kernel, error) { return mainkernels.K3ResScores(64, 2, 3, 1e-6, false) })
	add("k3/resmix", func() (*ir.Kernel, error) { return mainkernels.K3ResMix(64, 2, 3) })
	add("k3/post-restart", func() (*ir.Kernel, error) { return mainkernels.K3Post(64, 2, 4, 2, true) })
	add("k3/decaybound", func() (*ir.Kernel, error) { return mainkernels.DeltaDecayBoundRows(32, 2, -5) })
	// Clef's decision head (kernels/dechead.go): a linear with its bias, two
	// heads' scores and their values, a LayerNorm and an RMSNorm, and the
	// head's row ops, at toy widths.
	add("dh/matabt-bias", func() (*ir.Kernel, error) { return mainkernels.DHMatABt(1, 3, 8, 16, 16, 16, 8, 0, 0, 0, 1, true) })
	add("dh/matabt-heads", func() (*ir.Kernel, error) { return mainkernels.DHMatABt(2, 3, 5, 4, 8, 8, 5, 4, 4, 15, .5, false) })
	add("dh/matab-heads", func() (*ir.Kernel, error) { return mainkernels.DHMatAB(2, 3, 4, 5, 5, 8, 8, 15, 4, 4) })
	add("dh/layernorm", func() (*ir.Kernel, error) { return mainkernels.DHNormRows(16, 3, 1e-5, false, true) })
	add("dh/rmsnorm", func() (*ir.Kernel, error) { return mainkernels.DHNormRows(16, 3, 1e-6, true, false) })
	add("dh/gatheradd", func() (*ir.Kernel, error) { return mainkernels.DHGatherAdd(16, 3) })
	add("dh/l2", func() (*ir.Kernel, error) { return mainkernels.DHL2Rows(16, 3) })
	add("dh/rowdot", func() (*ir.Kernel, error) { return mainkernels.DHRowDot(16, 3) })
	add("dh/features", func() (*ir.Kernel, error) { return mainkernels.DHFeatures(16, 3) })
	add("dh/combine", func() (*ir.Kernel, error) { return mainkernels.DHCombine(5, 1.5, 2, .25) })
	// The grid barrier's gate kernel (kernels/gridprobe.go), so ptxas and
	// spirv-val see the atomics and the spin loop.
	add("grid/probe", func() (*ir.Kernel, error) { return mainkernels.GridProbe(4, 128), nil })
	// A kernel that uses shared memory and a barrier, so the vendors'
	// assemblers see that surface. The barrier is load-bearing: each thread
	// stages one element and reads its neighbour's.
	add("shared/roundtrip", func() (*ir.Kernel, error) {
		b := ir.New("shared_roundtrip", [3]int{64, 1, 1})
		pIn := b.Param("pIn", ir.F32)
		pOut := b.Param("pOut", ir.F32)
		sh := b.Shared("tile", ir.F32, 64)
		tid := b.TID()
		b.Store(sh, tid, b.Load(ir.F32, pIn, tid, 0), 0)
		b.Barrier()
		other := b.Rem(ir.U32, b.Add(ir.U32, tid, b.Const(ir.U32, 1)), b.Const(ir.U32, 64))
		b.Store(pOut, tid, b.Load(ir.F32, sh, other, 0), 0)
		return b.Done(), nil
	})

	// Every subgroup collective at every scalar type, so the assemblers see
	// the reductions, the indexed shuffle and the lane builtins.
	for _, ty := range []ir.Type{ir.F32, ir.I32, ir.U32} {
		add("subgroup/"+ty.String(), func() (*ir.Kernel, error) { return mainkernels.SubgroupProbe(ty) })
	}

	for _, q := range []kernels.Quant{kernels.Q4_0, kernels.Q8_0, kernels.Q4_K,
		kernels.Q3_K, kernels.Q5_K, kernels.Q6_K} {
		for _, split := range []int{1, 4} {
			q, split := q, split
			add(fmt.Sprintf("matvec/%s/split%d", q, split), func() (*ir.Kernel, error) {
				return mainkernels.MatVec(mainkernels.MatVecShape{T: q, K: 2048, Rows: 64, Split: split})
			})
		}
	}
	// The indexed matvec (MUL_MAT_ID). Slots=1 is the top-1 router shape, and
	// the one that once crashed the Vulkan driver while multi-slot shapes
	// assembled fine.
	for _, c := range []struct{ experts, slots, split int }{
		{8, 2, 1}, {8, 1, 1}, {8, 2, 4}, {4, 4, 1},
	} {
		c := c
		add(fmt.Sprintf("matvec-moe/e%ds%d/split%d", c.experts, c.slots, c.split),
			func() (*ir.Kernel, error) {
				return mainkernels.MatVec(mainkernels.MatVecShape{T: kernels.Q4_K, K: 2048,
					Rows: 64, Split: c.split, Experts: c.experts, Slots: c.slots})
			})
	}
	// The router. ExpertRank uses OpSelect, so OpLt's Pred reaches an
	// assembler.
	for _, c := range []struct{ n, k int }{{4, 2}, {8, 2}, {128, 8}, {64, 1}} {
		c := c
		add(fmt.Sprintf("expertrank/n%dk%d", c.n, c.k), func() (*ir.Kernel, error) {
			return mainkernels.ExpertRank(mainkernels.MoERoute{NExpert: c.n, K: c.k})
		})
	}
	add("routermatvec", func() (*ir.Kernel, error) { return mainkernels.RouterMatVecOf(128, 2048, false) })
	// q, k and v in one launch, with v in another format.
	add("matvecseg/q4k-q4k-q6k/split16g", func() (*ir.Kernel, error) {
		return mainkernels.MatVecSegments([]mainkernels.MatVecShape{
			{T: kernels.Q4_K, K: 2048, Rows: 256}, {T: kernels.Q4_K, K: 2048, Rows: 64},
			{T: kernels.Q6_K, K: 2048, Rows: 64, Bias: true}}, 16, true)
	})
	// The gated up projection, on both final-row paths.
	add("matvec-gate/silu/split1", func() (*ir.Kernel, error) {
		return mainkernels.MatVec(mainkernels.MatVecShape{T: kernels.Q4_K, K: 2048, Rows: 64, Split: 1,
			Gate: true, GateAct: kernels.ActSiLU})
	})
	add("matvec-gate/gelu/split4g", func() (*ir.Kernel, error) {
		return mainkernels.MatVec(mainkernels.MatVecShape{T: kernels.Q4_K, K: 2048, Rows: 64, Split: 4,
			GroupSplit: true, Gate: true, GateAct: kernels.ActGELU})
	})
	add("expertweights", func() (*ir.Kernel, error) {
		return mainkernels.ExpertWeights(mainkernels.MoERoute{NExpert: 64, K: 8, Norm: true})
	})
	add("expertweights_nonorm", func() (*ir.Kernel, error) { return mainkernels.ExpertWeights(mainkernels.MoERoute{NExpert: 64, K: 8}) })
	// The sigmoid router divides by the sum of the selected sigmoids, which
	// can be zero, and carries the divisor's floor; the softmax path's divisor
	// is at least 1. Both are pinned.
	add("expertweights_sigmoid", func() (*ir.Kernel, error) {
		return mainkernels.ExpertWeights(mainkernels.MoERoute{NExpert: 64, K: 8, Norm: true, Sigmoid: true})
	})
	add("expertweights_sigmoid_scaled", func() (*ir.Kernel, error) {
		return mainkernels.ExpertWeights(mainkernels.MoERoute{
			NExpert: 64, K: 8, Norm: true, Sigmoid: true, Bias: true,
			NGroup: 8, NGroupUsed: 4, Scale: 2.5,
		})
	})
	// DeepSeek V4's router: sqrt(softplus) gates, a selection bias per row
	// (MoERoute.RowBias), over a batch of three rows.
	ds4Route := mainkernels.MoERoute{NExpert: 256, K: 6, Norm: true, SqrtSoftplus: true, Bias: true, RowBias: true,
		Scale: 1.5, Rows: 3}
	add("expertrank/sqrtsoftplus-rowbias", func() (*ir.Kernel, error) { return mainkernels.ExpertRank(ds4Route) })
	add("expertweights_sqrtsoftplus", func() (*ir.Kernel, error) { return mainkernels.ExpertWeights(ds4Route) })
	add("expertcombine", func() (*ir.Kernel, error) { return mainkernels.ExpertCombine(2048, 8) })
	add("reduce", func() (*ir.Kernel, error) { return mainkernels.Reduce(64, 4) })
	add("quantize", func() (*ir.Kernel, error) { return mainkernels.Quantize(2048, 32) })
	add("rmsnormrows", func() (*ir.Kernel, error) { return mainkernels.RMSNormRows(4096, 1, 1e-5, false, false) })
	add("restride", func() (*ir.Kernel, error) { return mainkernels.Restride(1024, 257, 257, 513) })
	add("rmsnormrows-add", func() (*ir.Kernel, error) { return mainkernels.RMSNormRows(4096, 1, 1e-5, false, true) })
	add("rmsnormquant-add/w256", func() (*ir.Kernel, error) {
		return mainkernels.RMSNormQuantRows(4096, 1, 1e-5, false, true, 256)
	})
	add("normpart", func() (*ir.Kernel, error) { return mainkernels.NormPart(2048, 64) })
	add("normapply", func() (*ir.Kernel, error) { return mainkernels.NormApply(2048, 64, 1e-6, false) })
	add("normapply1", func() (*ir.Kernel, error) { return mainkernels.NormApply(2048, 64, 1e-6, true) })
	add("silumul", func() (*ir.Kernel, error) { return mainkernels.ActMul(16384, kernels.ActSiLU) })
	add("sigmoidmul", func() (*ir.Kernel, error) { return mainkernels.SigmoidMul(16384) })
	add("splitheadgate", func() (*ir.Kernel, error) { return mainkernels.SplitHeadGate(16, 256, 1) })
	add("sharedrouterlogits", func() (*ir.Kernel, error) { return mainkernels.SharedRouterLogits(1, 2048) })
	add("sharedexpertadd", func() (*ir.Kernel, error) { return mainkernels.SharedExpertAdd(2048, 1) })
	add("add", func() (*ir.Kernel, error) { return mainkernels.Add(2048) })
	add("scale", func() (*ir.Kernel, error) { return mainkernels.Scale(2048) })
	add("actmul-gelu", func() (*ir.Kernel, error) { return mainkernels.ActMul(16384, kernels.ActGELU) })
	add("actmul-swiglu-oai", func() (*ir.Kernel, error) { return mainkernels.ActMul(16384, kernels.ActSwiGLUOAI) })
	add("actmul-situ", func() (*ir.Kernel, error) { return mainkernels.ActMul(16384, kernels.ActSitu) })
	add("act-silu", func() (*ir.Kernel, error) { return mainkernels.Act(16384, kernels.ActSiLU) })
	add("act-gelu", func() (*ir.Kernel, error) { return mainkernels.Act(16384, kernels.ActGELU) })
	add("act-quickgelu", func() (*ir.Kernel, error) { return mainkernels.Act(16384, kernels.ActQuickGELU) })
	add("xielu", func() (*ir.Kernel, error) { return mainkernels.XIELU(16384) })
	add("expertweights_sparsemixer", func() (*ir.Kernel, error) {
		return mainkernels.ExpertWeights(mainkernels.MoERoute{NExpert: 16, K: 2, SparseMixer: 0.01})
	})
	add("softcap", func() (*ir.Kernel, error) { return mainkernels.Softcap(8*2048, 50) })
	add("softmax-sink-scalar", func() (*ir.Kernel, error) { return mainkernels.SoftmaxRowsSink(64, 4096, 1, 1, 1, true) })
	add("softmax-sink-warp", func() (*ir.Kernel, error) { return mainkernels.SoftmaxRowsSink(64, 4096, 32, 1, 1, true) })
	add("softmax-sink-rows", func() (*ir.Kernel, error) { return mainkernels.SoftmaxRowsSink(64, 4096, 32, 4, 1, true) })
	add("indexedbiasadd", func() (*ir.Kernel, error) { return mainkernels.IndexedBiasAdd(4, 2880) })
	add("layernormpart", func() (*ir.Kernel, error) { return mainkernels.LayerNormPartRows(768, 64, 1) })
	add("layernormvar", func() (*ir.Kernel, error) { return mainkernels.LayerNormVarRows(768, 64, 1) })
	add("layernormapply", func() (*ir.Kernel, error) { return mainkernels.LayerNormApplyRows(768, 64, 1e-6, false, 1) })
	add("layernormapply-bias", func() (*ir.Kernel, error) { return mainkernels.LayerNormApplyRows(768, 64, 1e-6, true, 1) })
	// The batched LayerNorm a vision block runs on the device; the per-row
	// partial base is where a batched norm goes wrong.
	add("layernormpart-rows", func() (*ir.Kernel, error) { return mainkernels.LayerNormPartRows(768, 64, 8) })
	add("layernormvar-rows", func() (*ir.Kernel, error) { return mainkernels.LayerNormVarRows(768, 64, 8) })
	add("layernormapply-rows", func() (*ir.Kernel, error) {
		return mainkernels.LayerNormApplyRows(768, 64, 1e-6, true, 8)
	})
	add("rope-neox", func() (*ir.Kernel, error) { return mainkernels.RoPERows(8, 256, 256, true, 1) })
	add("rope", func() (*ir.Kernel, error) { return mainkernels.RoPERows(8, 256, 256, false, 1) })
	// The rotary table, in both shapes. With no Metal device on the Linux
	// box, msl.Emit through this list is the only check that its MSL
	// lowering compiles.
	add("ropetable", func() (*ir.Kernel, error) { return mainkernels.RopeTable(128, 1) })
	add("ropetable-rows", func() (*ir.Kernel, error) { return mainkernels.RopeTable(128, 8) })
	add("copyat", func() (*ir.Kernel, error) { return mainkernels.CopyAt(256) })
	add("attnscores", func() (*ir.Kernel, error) {
		return mainkernels.AttnScoresTiled(8, 256, 256, 8, 2048, 0.0625, 1, 1, 1, 0)
	})
	// Both softmax arms: the warp reduction ships where the device has a
	// subgroup shuffle and the scalar one is the fallback, so both must
	// assemble everywhere.
	add("softmax", func() (*ir.Kernel, error) { return mainkernels.SoftmaxRows(8, 2048, 32, 1, 1) })
	add("softmax-scalar", func() (*ir.Kernel, error) { return mainkernels.SoftmaxRows(8, 2048, 1, 1, 1) })
	// The head norm is the other subgroup-width kernel; both variants for the
	// softmax pair's reason.
	add("headnorm", func() (*ir.Kernel, error) { return mainkernels.HeadNorm(16, 128, 1e-5, 32) })
	add("headnorm-scalar", func() (*ir.Kernel, error) { return mainkernels.HeadNorm(16, 128, 1e-5, 1) })
	add("attnacc", func() (*ir.Kernel, error) { return mainkernels.AttnAcc(8, 256, 256, 8, 2048) })
	// The recurrent pair a hybrid block needs. They are shaped kernels, so
	// they live here rather than in the elementwise inventory.
	add("conv1drows", func() (*ir.Kernel, error) { return mainkernels.Conv1dRows(4, 128, 32) })
	add("conv1dshift", func() (*ir.Kernel, error) { return mainkernels.Conv1dShift(4, 128, 32) })
	// Their run forms, a ragged step's rows grouped by sequence through the
	// run descriptor: the order list and each row's run are loads the chunk
	// forms do not have.
	add("conv1drows-runs", func() (*ir.Kernel, error) { return mainkernels.Conv1dRowsRuns(4, 128, 8) })
	add("conv1dshift-runs", func() (*ir.Kernel, error) { return mainkernels.Conv1dShiftRuns(4, 128, 8) })
	// The fixture's geometry and the 80B's, so a lowering that only copes
	// with a small kDim is caught here.
	add("gateddelta/small", func() (*ir.Kernel, error) { return mainkernels.GatedDeltaStep(4, 4, 8, 2) })
	add("gateddelta/80b", func() (*ir.Kernel, error) { return mainkernels.GatedDeltaStep(32, 128, 128, 4) })
	// Kimi-Linear's per-channel decay. rep=2 pins the decay being indexed by
	// the value head where k and q use the key head.
	add("gateddeltachan/small", func() (*ir.Kernel, error) { return mainkernels.GatedDeltaStepChan(4, 4, 8, 2) })
	add("gateddeltachan/kda", func() (*ir.Kernel, error) { return mainkernels.GatedDeltaStepChan(32, 128, 128, 1) })
	// The recurrent pool's slot copy: the shared form's way home and the move
	// between a pool's halves.
	add("copyslots", func() (*ir.Kernel, error) { return mainkernels.CopySlots(32*128*128, 4) })
	add("deltagate", func() (*ir.Kernel, error) { return mainkernels.DeltaGateRows(32, 1) })
	add("splitdeltagates", func() (*ir.Kernel, error) { return mainkernels.SplitDeltaGatesRows(16, 2, 1) })
	add("slicerows", func() (*ir.Kernel, error) { return mainkernels.SliceRows(2048, 8192, 2048, 1) })
	// A convolutional tower's (kernels/conv.go), at a stride-2 3x3 over 13
	// channels and attention with fewer keys than queries.
	add("conv/pad", func() (*ir.Kernel, error) { return mainkernels.PadRows(7, 9, 13, 9, 11, 1, 1) })
	add("conv/dw", func() (*ir.Kernel, error) { return mainkernels.DWConv(3, 2, 11, 13, 4, 5) })
	add("conv/im2col", func() (*ir.Kernel, error) { return mainkernels.Im2col(3, 2, 11, 13, 5, 128, 20, 9*11*13) })
	add("conv/mqascores", func() (*ir.Kernel, error) { return mainkernels.MQAScores(12, 4, 3, 8) })
	add("conv/softmax", func() (*ir.Kernel, error) { return mainkernels.RowSoftmax(36, 4) })
	add("conv/mqaacc", func() (*ir.Kernel, error) { return mainkernels.MQAAcc(12, 4, 3, 8) })
	// The GGUF unpack at both ends of the stride range: at the 70B's widest
	// tensor the Load and Store immediates are large, which a 256-row shape
	// would not exercise.
	add("unpack-q4k/256x2048", func() (*ir.Kernel, error) {
		return mainkernels.Unpack(kernels.Q4_K, 256, 2048)
	})
	add("unpack-q4k/28672x8192", func() (*ir.Kernel, error) {
		return mainkernels.Unpack(kernels.Q4_K, 28672, 8192)
	})
	return out
}

// widthDependent is the inventory of kernels whose arithmetic needs a subgroup
// of a particular width. The sweep asks every shipped kernel what it declared
// (ir.Validate already refuses an undeclared width-dependent op); this list is
// the other half, so a kernel that loses its shuffle by accident is also red.
var widthDependent = map[string]int{
	"flash/split1":       32,
	"flash/split8":       32,
	"softmax":            32, // the butterfly max and sum
	"softmax-sink-warp":  32, // the same, with the sink folded in after
	"softmax-sink-rows":  32,
	"headnorm":           32, // the butterfly sum of squares
	"headnormgated-silu": 32, // the same, with the gate as its epilogue
	"subgroup/f32":       32, // every subgroup collective
	"subgroup/i32":       32,
	"subgroup/u32":       32,
	// The matvec-mma and mmaprobe entries are added by the sweep itself: they
	// are built by mmaKernels and are PTX-only.
}

// TestEveryWidthDependentKernelDeclaresItsWidth is the sweep. A kernel that
// shuffles without declaring its width is correct on any 32-wide device and
// wrong elsewhere; a declaration with no shuffle costs a device that could run
// it.
func TestEveryWidthDependentKernelDeclaresItsWidth(t *testing.T) {
	ks := all(t)
	for n, k := range mmaKernels(t) {
		ks[n] = k
		widthDependent[n] = ir.SubgroupLanes
	}
	if len(widthDependent) < 3 {
		t.Fatal("the inventory is empty; this test would pass and prove nothing")
	}
	seen := 0
	for name, k := range ks {
		var uses ir.Kind
		found := false
		for _, o := range k.Ops {
			if ir.WidthDependent(o.Kind) {
				uses, found = o.Kind, true
				break
			}
		}
		want := widthDependent[name]
		switch {
		case found && k.Lanes != ir.SubgroupLanes:
			t.Errorf("%s emits %s and declares Lanes=%d: it would be offered to a device "+
				"that guarantees nothing", name, uses, k.Lanes)
		case !found && k.Lanes != 0:
			t.Errorf("%s declares Lanes=%d and has no width-dependent op", name, k.Lanes)
		case want != 0 && k.Lanes != want:
			t.Errorf("%s is in the width-dependent inventory at %d lanes and declares %d",
				name, want, k.Lanes)
		case want == 0 && k.Lanes != 0:
			t.Errorf("%s declares %d lanes and is not in the inventory; add it there or "+
				"nothing states that it must keep needing one", name, k.Lanes)
		}
		if k.Lanes != 0 {
			seen++
		}
	}
	if seen != len(widthDependent) {
		t.Errorf("%d kernels declare a width, the inventory names %d", seen, len(widthDependent))
	}
	t.Logf("%d of %d shipped kernels depend on the subgroup width", seen, len(ks))
}

func TestLowersEverywhere(t *testing.T) {
	ks := all(t)
	dir := t.TempDir()
	ptxas, _ := exec.LookPath("ptxas")
	val, _ := exec.LookPath("spirv-val")
	if val == "" {
		// spirv-val may be installed but not on $PATH (a snap); a validator
		// that is present and unused logs the same as an absent one.
		for _, p := range []string{
			"/snap/kf6-core24/current/usr/bin/spirv-val",
			"/snap/kf6-core24/64/usr/bin/spirv-val",
			"/usr/lib/x86_64-linux-gnu/spirv-tools/spirv-val",
		} {
			if _, err := os.Stat(p); err == nil {
				val = p
				break
			}
		}
	}
	if ptxas == "" {
		t.Log("ptxas not installed: PTX is lowered but not assembled")
	}
	if val == "" {
		t.Log("spirv-val not installed: SPIR-V is lowered but not validated")
	}
	// The matrix kernels are PTX-only, and the other two backends must refuse
	// them rather than quietly succeed.
	for name, k := range mmaKernels(t) {
		t.Run(name, func(t *testing.T) {
			src, err := ptx.Lower(k, "sm_86")
			if err != nil {
				t.Fatalf("ptx: %v", err)
			}
			if ptxas != "" {
				f := filepath.Join(dir, "m.ptx")
				if err := os.WriteFile(f, []byte(src), 0o600); err != nil {
					t.Fatal(err)
				}
				if out, err := exec.Command(ptxas, "--gpu-name", "sm_86", f,
					"-o", filepath.Join(dir, "m.cubin")).CombinedOutput(); err != nil {
					t.Fatalf("ptxas: %v\n%s", err, out)
				}
			}
			if _, err := spirv.Emit(k); err == nil {
				t.Error("spirv accepted an MMA kernel; it has no integer matrix instruction")
			}
			if _, err := msl.Emit(k); err == nil {
				t.Error("msl accepted an MMA kernel; simdgroup_matrix is float only")
			}
		})
	}

	for name, k := range ks {
		t.Run(name, func(t *testing.T) {
			src, err := ptx.Lower(k, "sm_86")
			if err != nil {
				t.Fatalf("ptx: %v", err)
			}
			if ptxas != "" {
				f := filepath.Join(dir, "k.ptx")
				if err := os.WriteFile(f, []byte(src), 0o600); err != nil {
					t.Fatal(err)
				}
				if out, err := exec.Command(ptxas, "--gpu-name", "sm_86", f,
					"-o", filepath.Join(dir, "k.cubin")).CombinedOutput(); err != nil {
					t.Fatalf("ptxas: %v\n%s", err, out)
				}
			}
			words, err := spirv.Emit(k)
			if err != nil {
				t.Fatalf("spirv: %v", err)
			}
			if val != "" {
				f := filepath.Join(dir, "k.spv")
				if err := os.WriteFile(f, words, 0o600); err != nil {
					t.Fatal(err)
				}
				// Under Vulkan's environment rules, not the universal ones:
				// only they check what a Vulkan driver requires of the
				// push-constant block every module carries (spirv.Emit's
				// group base). vulkan1.1 is the environment of the SPIR-V
				// 1.3 the modules declare.
				if out, err := exec.Command(val, "--target-env", "vulkan1.1", f).CombinedOutput(); err != nil {
					t.Fatalf("spirv-val: %v\n%s", err, out)
				}
			}
			if _, err := msl.Emit(k); err != nil {
				t.Fatalf("msl: %v", err)
			}
		})
	}
}

// mmaKernels are the warp-matrix kernels, which only one backend can lower.
func mmaKernels(t *testing.T) map[string]*ir.Kernel {
	t.Helper()
	out := map[string]*ir.Kernel{}
	for _, sh := range []ir.MMAShape{{M: 16, N: 8, K: 16}, {M: 16, N: 8, K: 32}} {
		k, err := mainkernels.MMAProbe(sh)
		if err != nil {
			t.Fatal(err)
		}
		out[fmt.Sprintf("mmaprobe/m%dn%dk%d", sh.M, sh.N, sh.K)] = k
	}
	for _, q := range []kernels.Quant{kernels.Q4_0, kernels.Q8_0, kernels.Q4_K,
		kernels.Q3_K, kernels.Q5_K, kernels.Q6_K} {
		for _, tl := range [][2]int{{1, 1}, {4, 4}} {
			k, err := mainkernels.MatVecMMA(mainkernels.MatVecShape{T: q, K: 2048, Rows: 128,
				NTok: 128, MT: tl[0], NT: tl[1]})
			if err != nil {
				t.Fatalf("%s %dx%d: %v", q, tl[0], tl[1], err)
			}
			out[fmt.Sprintf("matvec-mma/%s/%dx%d", q, tl[0], tl[1])] = k
		}
	}
	return out
}
