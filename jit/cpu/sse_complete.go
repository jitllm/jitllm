//go:build amd64 || arm64

package cpu

import (
	"errors"
	"fmt"
	"strings"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// SSEPending lists the SSE-tier table's ops that have no kernel yet, by field
// name: every emitter of EmittersFor(TierSSE) that answers ErrNoSSEKernel at
// an engine shape. Empty means the tier is complete.
//
// Baseline asks this table rather than a constant, so a host on the SSE tier
// passes exactly when every op a token runs has an SSE kernel and is refused
// naming the ones that do not.
//
// The probes are engine shapes, not zero values: a landed emitter handed hd=0
// or n=0 may divide by zero or loop on a zero stride, and this runs inside
// Baseline. TestSSEProbesCoverEveryField fails when a table field has no probe.
// An emitter that panics at an engine shape is reported as its own entry.
func SSEPending() []string {
	em := EmittersFor(TierSSE)
	var out []string
	for _, p := range sseProbes {
		if msg := p.pending(em); msg != "" {
			out = append(out, msg)
		}
	}
	return out
}

// sseProbe is one representative call into one table field.
type sseProbe struct {
	field string
	call  func(em *Emitters) error
}

// pending returns "" for a field that has its kernel, the field's name for a
// pending one, and the name with the panic for one that faults at the shape.
func (p sseProbe) pending(em *Emitters) (msg string) {
	defer func() {
		if r := recover(); r != nil {
			msg = fmt.Sprintf("%s (panicked: %v)", p.field, r)
		}
	}()
	if err := p.call(em); errors.Is(err, ErrNoSSEKernel) {
		return p.field
	}
	return ""
}

func probe0(f func() ([]byte, error)) error { _, err := f(); return err }

// sseProbes covers every error-returning field of Emitters. The shapes are the
// ones emitters_test.go and the engine use: a 64-wide head over a 256-float
// KV stride, a 64-float norm, Q4_K for the packed family.
var sseProbes = []sseProbe{
	{"Axpy", func(em *Emitters) error { return probe0(em.Axpy) }},
	{"Scale", func(em *Emitters) error { return probe0(em.Scale) }},
	{"Softcap", func(em *Emitters) error { return probe0(em.Softcap) }},
	{"Clamp", func(em *Emitters) error { return probe0(em.Clamp) }},
	{"PrefetchLines", func(em *Emitters) error { return probe0(em.PrefetchLines) }},
	{"Resample", func(em *Emitters) error { _, err := em.Resample(22); return err }},
	{"ResampleH", func(em *Emitters) error { _, err := em.ResampleH(22); return err }},
	{"PixLUT", func(em *Emitters) error { return probe0(em.PixLUT) }},
	{"Copy32", func(em *Emitters) error { return probe0(em.Copy32) }},
	{"PixelRow", func(em *Emitters) error { _, err := em.PixelRow(PixYCbCr, 1, PixOver8); return err }},
	{"AxisTaps", func(em *Emitters) error { _, err := em.AxisTaps(AxisCubic); return err }},
	{"LerpGrid", func(em *Emitters) error { return probe0(em.LerpGrid) }},
	{"SinCosTab", func(em *Emitters) error { return probe0(em.SinCosTab) }},
	{"Softmax", func(em *Emitters) error { return probe0(em.Softmax) }},
	{"LogSoftmax", func(em *Emitters) error { return probe0(em.LogSoftmax) }},
	{"SigmoidMul", func(em *Emitters) error { return probe0(em.SigmoidMul) }},
	{"Ln", func(em *Emitters) error { return probe0(em.Ln) }},
	{"XIELU", func(em *Emitters) error { return probe0(em.XIELU) }},
	{"ActMul", func(em *Emitters) error {
		for _, k := range Gated {
			if _, err := em.ActMul(k); err != nil {
				return err
			}
		}
		return nil
	}},
	{"Act", func(em *Emitters) error {
		for _, k := range Ungated {
			if _, err := em.Act(k); err != nil {
				return err
			}
		}
		return nil
	}},
	{"RMSNorm", func(em *Emitters) error { _, err := em.RMSNorm(64); return err }},
	{"LayerNorm", func(em *Emitters) error { _, err := em.LayerNorm(64, true); return err }},
	{"RowStat", func(em *Emitters) error {
		for _, m := range []RowMode{RowGauss, RowMag} {
			if _, err := em.RowStat(64, m); err != nil {
				return err
			}
		}
		return nil
	}},
	{"RoPE", func(em *Emitters) error { _, err := em.RoPE(64, 64, true); return err }},
	{"RoPESplit", func(em *Emitters) error { _, err := em.RoPESplit(64, 64); return err }},
	{"RopeTable", func(em *Emitters) error { _, err := em.RopeTable(32); return err }},
	{"HCMix", func(em *Emitters) error { _, err := em.HCMix(20, false); return err }},
	{"ColPool", func(em *Emitters) error { return probe0(em.ColPool) }},
	{"NGramHash", func(em *Emitters) error { _, err := em.NGramHash(3, 8, true); return err }},
	{"RowIQ4NL", func(em *Emitters) error { return probe0(em.RowIQ4NL) }},
	{"PackedRow", func(em *Emitters) error {
		q, _ := kernels.QuantOf(quant.Q4_K)
		_, err := em.PackedRow(q)
		return err
	}},
	{"Widen", func(em *Emitters) error { _, err := em.Widen(false); return err }},
	{"NarrowF16", func(em *Emitters) error { _, err := em.NarrowF16(); return err }},
	{"AttnScores", func(em *Emitters) error { _, err := em.AttnScores(64, 256, KVF32); return err }},
	{"AttnAcc", func(em *Emitters) error { _, err := em.AttnAcc(64, 256, KVF32); return err }},
	{"AttnAccInto", func(em *Emitters) error { _, err := em.AttnAccInto(64, 256, KVF32); return err }},
	{"AttnScores2", func(em *Emitters) error { _, err := em.AttnScores2(64, 256, KVF32); return err }},
	{"AttnAcc2", func(em *Emitters) error { _, err := em.AttnAcc2(64, 256, KVF32); return err }},
	{"AttnAcc2Into", func(em *Emitters) error { _, err := em.AttnAcc2Into(64, 256, KVF32); return err }},
	{"AttnScoresTiled", func(em *Emitters) error {
		_, err := em.AttnScoresTiled(64, 64, 64, 1024, 8, KVF32)
		return err
	}},
	{"KVWiden", func(em *Emitters) error { _, err := em.KVWiden(64); return err }},
	{"SampleSegMax", func(em *Emitters) error {
		for _, f := range []bool{false, true} {
			for _, m := range []bool{false, true} {
				if _, err := em.SampleSegMax(f, m); err != nil {
					return err
				}
			}
		}
		return nil
	}},
	{"SampleDraw", func(em *Emitters) error { return probe0(em.SampleDraw) }},
	{"SamplePenalty", func(em *Emitters) error { return probe0(em.SamplePenalty) }},
	{"RowMajor", func(em *Emitters) error {
		_, err := em.RowMajor(Spec{W: quant.F32, Rows: 1, Cols: 1, Accs: 1})
		return err
	}},
	{"FloatMatMul", func(em *Emitters) error {
		for _, t := range []quant.Type{quant.F32, quant.F16, quant.BF16} {
			for nt := 1; nt <= MaxFloatTokens; nt++ {
				if _, err := em.FloatMatMul(t, nt); err != nil {
					return err
				}
			}
		}
		return nil
	}},
	{"PackedMatVec", func(em *Emitters) error { _, err := em.PackedMatVec(quant.Q4_K, PackedRows); return err }},
	{"PackedWide", func(em *Emitters) error { _, err := em.PackedWide(quant.Q4_K); return err }},
	{"PackedFused", func(em *Emitters) error { _, err := em.PackedFused(quant.Q4_K); return err }},
	{"PackedTiled", func(em *Emitters) error { _, err := em.PackedTiled(quant.Q4_K, 2048, 512, 2); return err }},
	{"PackedGEMM", func(em *Emitters) error { _, err := em.PackedGEMM(quant.Q4_K, 2048, 512, 256); return err }},
	{"PackedFusedAhead", func(em *Emitters) error { _, err := em.PackedFusedAhead(quant.Q4_K, 4); return err }},
	{"PackedFusedWin", func(em *Emitters) error { _, err := em.PackedFusedWin(quant.Q4_K, 256); return err }},
	// A tier without QuantAct answers errQuantActOwed, not ErrNoSSEKernel,
	// so this reports "" and Baseline does not refuse the host; the probe
	// keeps the field covered should a stub ever answer ErrNoSSEKernel.
	{"QuantAct", func(em *Emitters) error { _, err := em.QuantAct(false); return err }},
	{"QuantActNarrow", func(em *Emitters) error { _, err := em.QuantActNarrow(false); return err }},
	{"Argmax", func(em *Emitters) error { return probe0(em.Argmax) }},
	// The 30B's router shape. Like QuantAct this reports "" when the kernel is
	// absent -- topk_other.go answers an ordinary shape error, not
	// ErrNoSSEKernel -- because a host without it still runs every dense model.
	{"MoETopK", func(em *Emitters) error { _, err := em.MoETopK(128, 8, true); return err }},
	// DeepSeek-V3's own shape: 256 experts in 8 groups, 4 groups used, 8
	// experts, a selection bias and routed_scaling_factor 2.5. The narrow
	// probe above cannot reach the bias, group or scale phases at all.
	{"MoERoute", func(em *Emitters) error {
		_, err := em.MoERoute(256, 8, MoEGate{Norm: true, Bias: true, NGroup: 8, NGroupUsed: 4, Scale: true})
		return err
	}},
	// Phi-3.5-MoE's 16 experts.
	{"SparseMixer", func(em *Emitters) error { _, err := em.SparseMixer(16); return err }},
	{"GatedDelta", func(em *Emitters) error { _, err := em.GatedDelta(128); return err }},
	{"GatedDeltaChan", func(em *Emitters) error { _, err := em.GatedDeltaChan(128); return err }},
	{"Conv1d", func(em *Emitters) error { _, err := em.Conv1d(4, 64); return err }},
	{"DWConv", func(em *Emitters) error {
		_, err := em.DWConv(DWShape{K: 3, Stride: 2, WP: 9, Chans: 13, WOut: 4})
		return err
	}},
	{"DeltaGate", func(em *Emitters) error { return probe0(em.DeltaGate) }},
	{"DeltaDecayBound", func(em *Emitters) error { return probe0(em.DeltaDecayBound) }},
	{"GatedSSD", func(em *Emitters) error { _, err := em.GatedSSD(128); return err }},
	{"SSDGate", func(em *Emitters) error { return probe0(em.SSDGate) }},
	{"SelScan", func(em *Emitters) error { _, err := em.SelScan(16); return err }},
}

// TierReport is one line saying which kernel tier this host runs, why, and --
// on the SSE tier -- whether the engine can run here, for `jitllm hardware`
// and the desktop's machine page.
//
// It names the tier the probe chose, the host extensions it chose from, and
// on the SSE tier the ops that still have no kernel (SSEPending), since those
// are why a model would be refused.
func TierReport() string {
	f := CPU()
	switch HostTier() {
	case TierAVX2:
		return fmt.Sprintf("avx2 -- VEX-encoded 256-bit kernels with FMA and F16C (host: %s)", f.ISA())
	case TierSSE:
		s := fmt.Sprintf("sse -- legacy-encoded 128-bit kernels, no FMA, f16 converted in software "+
			"(host: %s; no usable %s)", f.ISA(), missingNames(f, ISATierAVX2))
		if p := SSEPending(); len(p) > 0 {
			return s + fmt.Sprintf("; %d ops have no SSE-tier kernel yet, so a model will not load: %s",
				len(p), strings.Join(p, ", "))
		}
		return s + "; every op is generated for it"
	case TierNEON:
		return "neon -- A64 kernels"
	}
	return fmt.Sprintf("none -- below the amd64 floor (missing %s): no model will load",
		missingNames(f, ISATierSSE))
}

// HostDotLabel names the int8 dot sequence the packed kernels run on this host:
// HostDotKind's on the AVX2 tier (where it is a real choice), and the tier
// itself elsewhere, because HostDotKind answers DotVEX on an SSE host -- a VEX
// sequence that host cannot execute -- and printing it there reports a kernel
// no run will use.
func HostDotLabel() string {
	switch HostTier() {
	case TierAVX2, TierNEON:
		return HostDotKind().String()
	case TierSSE:
		return "sse"
	}
	return "none"
}
