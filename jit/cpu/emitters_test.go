//go:build amd64

package cpu

import (
	"bytes"
	"errors"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestEmittersTablesAreComplete requires every function field of both amd64
// tables to be set: a new op cannot land on one tier and be forgotten on the
// other, because the table literal will not compile without the name and this
// fails on a nil left in its place.
func TestEmittersTablesAreComplete(t *testing.T) {
	for _, tier := range []Tier{TierAVX2, TierSSE} {
		em := EmittersFor(tier)
		if em.Tier != tier {
			t.Errorf("EmittersFor(%v).Tier = %v", tier, em.Tier)
		}
		v := reflect.ValueOf(*em)
		n := 0
		for i := 0; i < v.NumField(); i++ {
			f := v.Field(i)
			if f.Kind() != reflect.Func {
				continue
			}
			n++
			if f.IsNil() {
				t.Errorf("%v table: %s is nil", tier, v.Type().Field(i).Name)
			}
		}
		if n < 30 {
			t.Errorf("%v table has %d function fields; the reflection walked the wrong thing", tier, n)
		}
	}
	if EmittersFor(TierNone) != EmittersFor(TierAVX2) {
		t.Error("TierNone must get the primary table, whose kernels Map refuses by name")
	}
}

// emitterCall is one representative call into a table field, with the direct
// call it must reproduce on the primary table.
type emitterCall struct {
	name   string
	table  func(em *Emitters) ([]byte, error)
	direct func() ([]byte, error)
}

func emitted(b []byte) func() ([]byte, error) { return func() ([]byte, error) { return b, nil } }

// emitterCalls covers every field at the shapes the engine uses, including
// ragged widths and both KV widths.
func emitterCalls() []emitterCall {
	var cs []emitterCall
	add := func(name string, table func(em *Emitters) ([]byte, error), direct func() ([]byte, error)) {
		cs = append(cs, emitterCall{name, table, direct})
	}
	add("axpy", func(em *Emitters) ([]byte, error) { return em.Axpy() }, emitted(EmitAxpy()))
	add("prefetch_lines", func(em *Emitters) ([]byte, error) { return em.PrefetchLines() }, emitted(EmitPrefetchLines()))
	add("scale", func(em *Emitters) ([]byte, error) { return em.Scale() }, emitted(EmitScale()))
	add("softcap", func(em *Emitters) ([]byte, error) { return em.Softcap() }, emitted(EmitSoftcap()))
	add("softmax", func(em *Emitters) ([]byte, error) { return em.Softmax() }, emitted(EmitSoftmax()))
	add("sigmoidmul", func(em *Emitters) ([]byte, error) { return em.SigmoidMul() }, emitted(EmitSigmoidMul()))
	for _, k := range Gated {
		add("actmul/"+k.String(), func(em *Emitters) ([]byte, error) { return em.ActMul(k) }, emitted(EmitActMul(k)))
	}
	for _, k := range Ungated {
		add("act/"+k.String(), func(em *Emitters) ([]byte, error) { return em.Act(k) }, emitted(EmitAct(k)))
	}
	for _, n := range []int{1, 7, 64, 771, 2048} {
		add("rmsnorm/"+shapeName(n), func(em *Emitters) ([]byte, error) { return em.RMSNorm(n) }, emitted(EmitRMSNorm(n)))
		for _, b := range []bool{false, true} {
			add("layernorm/"+shapeName(n)+map[bool]string{true: "+bias"}[b], func(em *Emitters) ([]byte, error) { return em.LayerNorm(n, b) }, emitted(EmitLayerNorm(n, b)))
		}
	}
	for _, r := range []struct {
		hd, nrot int
		neox     bool
	}{{64, 64, false}, {128, 128, true}, {256, 64, true}, {80, 32, false}} {
		add("rope/"+shapeName(r.hd), func(em *Emitters) ([]byte, error) { return em.RoPE(r.hd, r.nrot, r.neox) },
			func() ([]byte, error) { return EmitRoPE(r.hd, r.nrot, r.neox) })
	}
	for _, qt := range quant.PackedTypes {
		q, _ := kernels.QuantOf(qt)
		add("packed_row/"+qt.String(), func(em *Emitters) ([]byte, error) { return em.PackedRow(q) },
			func() ([]byte, error) { return EmitPackedRow(q) })
	}
	for _, b := range []bool{false, true} {
		add("widen"+map[bool]string{true: "/bf16"}[b], func(em *Emitters) ([]byte, error) { return em.Widen(b) }, emitted(EmitWiden(b)))
	}
	// The rotary table at every real rotary width plus two ragged ones, so
	// the second body the tail emits is covered as well as the loop's.
	for _, np := range []int{32, 48, 64, 128, 26, 9, 4, 3, 1} {
		add("rope_table/"+shapeName(np), func(em *Emitters) ([]byte, error) { return em.RopeTable(np) },
			func() ([]byte, error) { return EmitRopeTable(np) })
	}
	for _, it := range []int{0, 1, 20} {
		for _, head := range []bool{false, true} {
			add("hcmix/"+shapeName(it)+map[bool]string{true: "/head"}[head],
				func(em *Emitters) ([]byte, error) { return em.HCMix(it, head) },
				func() ([]byte, error) { return EmitHCMix(it, head) })
		}
	}
	add("colpool", func(em *Emitters) ([]byte, error) { return em.ColPool() }, emitted(EmitColPool()))
	add("row_iq4nl", func(em *Emitters) ([]byte, error) { return em.RowIQ4NL() }, emitted(EmitRowIQ4NL()))
	for _, rs := range []bool{false, true} {
		add("ngramhash/3x8"+map[bool]string{false: "/noreset"}[rs],
			func(em *Emitters) ([]byte, error) { return em.NGramHash(3, 8, rs) },
			func() ([]byte, error) { return EmitNGramHash(3, 8, rs) })
	}
	// The image path's kernels (imgproc_const.go).
	for _, prec := range []int{13, 22} {
		add("resample/"+shapeName(prec), func(em *Emitters) ([]byte, error) { return em.Resample(prec) },
			emitted(EmitResample(prec)))
		add("resample_h/"+shapeName(prec), func(em *Emitters) ([]byte, error) { return em.ResampleH(prec) },
			emitted(EmitResampleH(prec)))
	}
	add("pixlut", func(em *Emitters) ([]byte, error) { return em.PixLUT() }, emitted(EmitPixLUT()))
	add("copy32", func(em *Emitters) ([]byte, error) { return em.Copy32() }, emitted(EmitCopy32()))
	for _, f := range []AxisFilter{AxisBilinear, AxisLinear, AxisCubic} {
		add("axistaps/"+shapeName(int(f)), func(em *Emitters) ([]byte, error) { return em.AxisTaps(f) },
			emitted(EmitAxisTaps(f)))
	}
	add("lerpgrid", func(em *Emitters) ([]byte, error) { return em.LerpGrid() }, emitted(EmitLerpGrid()))
	add("sincostab", func(em *Emitters) ([]byte, error) { return em.SinCosTab() }, emitted(EmitSinCosTab()))
	for _, f := range []PixFmt{PixRGBA, PixNRGBA, PixGray, PixRGBA64, PixYCbCr} {
		for _, o := range []PixOut{PixOver8, PixPlanes} {
			sub := 0
			if f == PixYCbCr {
				sub = 1
			}
			add("pixelrow/"+shapeName(int(f))+"/"+shapeName(int(o)),
				func(em *Emitters) ([]byte, error) { return em.PixelRow(f, sub, o) },
				func() ([]byte, error) { return EmitPixelRow(f, sub, o) })
		}
	}
	for _, hd := range []int{64, 80, 128} {
		for _, fm := range []KVFmt{KVF32, KVF16, KVQ8} {
			kv := 4 * hd
			attn := []struct {
				name   string
				table  func(em *Emitters) func(int, int, KVFmt) ([]byte, error)
				direct func(int, int, KVFmt) ([]byte, error)
			}{
				{"attn_scores", func(em *Emitters) func(int, int, KVFmt) ([]byte, error) { return em.AttnScores }, EmitAttnScores},
				{"attn_acc", func(em *Emitters) func(int, int, KVFmt) ([]byte, error) { return em.AttnAcc }, EmitAttnAcc},
				{"attn_acc_into", func(em *Emitters) func(int, int, KVFmt) ([]byte, error) { return em.AttnAccInto }, EmitAttnAccInto},
				{"attn_scores2", func(em *Emitters) func(int, int, KVFmt) ([]byte, error) { return em.AttnScores2 }, EmitAttnScores2},
				{"attn_acc2", func(em *Emitters) func(int, int, KVFmt) ([]byte, error) { return em.AttnAcc2 }, EmitAttnAcc2},
				{"attn_acc2_into", func(em *Emitters) func(int, int, KVFmt) ([]byte, error) { return em.AttnAcc2Into }, EmitAttnAcc2Into},
			}
			for _, a := range attn {
				add(a.name+"/"+shapeName(hd)+"/"+fm.String(), func(em *Emitters) ([]byte, error) { return a.table(em)(hd, kv, fm) },
					func() ([]byte, error) { return a.direct(hd, kv, fm) })
			}
		}
	}
	add("attn_scores_tiled", func(em *Emitters) ([]byte, error) { return em.AttnScoresTiled(64, 64, 64, 1024, 8, KVF32) },
		func() ([]byte, error) { return EmitAttnScoresTiled(64, 64, 64, 1024, 8, KVF32) })
	for _, ft := range []quant.Type{quant.F32, quant.F16, quant.BF16} {
		s := Spec{W: ft, Rows: 1, Cols: 1, Accs: 1}
		add("row_major/"+ft.String(), func(em *Emitters) ([]byte, error) { return em.RowMajor(s) },
			func() ([]byte, error) { return EmitNative(s) })
	}
	for _, qt := range quant.PackedTypes {
		for _, rows := range []int{PackedRows, PackedTail} {
			add("packed/"+qt.String()+"/"+shapeName(rows), func(em *Emitters) ([]byte, error) { return em.PackedMatVec(qt, rows) },
				func() ([]byte, error) { return EmitPackedMatVec(qt, rows, HostDotKind()) })
		}
		add("packed_fused/"+qt.String(), func(em *Emitters) ([]byte, error) { return em.PackedFused(qt) },
			func() ([]byte, error) { return EmitPackedMatVecFused(qt, HostDotKind()) })
		add("packed_wide/"+qt.String(), func(em *Emitters) ([]byte, error) { return em.PackedWide(qt) },
			func() ([]byte, error) { return EmitPackedMatVecWide(qt, HostDotKind()) })
		add("packed_tiled/"+qt.String(), func(em *Emitters) ([]byte, error) { return em.PackedTiled(qt, 2048, 512, 2) },
			func() ([]byte, error) { return EmitPackedMatMulTiled(qt, 2048, 512, 2, HostDotKind()) })
	}
	add("gated_delta", func(em *Emitters) ([]byte, error) { return em.GatedDelta(128) },
		func() ([]byte, error) { return EmitGatedDelta(128) })
	add("conv1d", func(em *Emitters) ([]byte, error) { return em.Conv1d(4, 64) },
		func() ([]byte, error) { return EmitConv1d(4, 64) })
	dw := DWShape{K: 3, Stride: 2, WP: 9, Chans: 13, WOut: 4}
	add("dwconv", func(em *Emitters) ([]byte, error) { return em.DWConv(dw) },
		func() ([]byte, error) { return EmitDWConv(dw) })
	add("delta_gate", func(em *Emitters) ([]byte, error) { return em.DeltaGate() }, emitted(EmitDeltaGate()))
	add("delta_decay_bound", func(em *Emitters) ([]byte, error) { return em.DeltaDecayBound() },
		emitted(EmitDeltaDecayBound()))
	return cs
}

func shapeName(n int) string { return strconv.Itoa(n) }

// TestPrimaryTableIsTodaysEmitters is the "AVX2 bytes do not move" gate: every
// field of the primary table returns exactly what the function it wraps
// returns, and none of it carries an ISA declaration.
//
// It catches a miswired field: AttnAcc pointed at EmitAttnAccInto compiles and
// computes a plausible attention that is wrong on the first token of every
// page.
func TestPrimaryTableIsTodaysEmitters(t *testing.T) {
	em := EmittersFor(TierAVX2)
	for _, c := range emitterCalls() {
		got, gerr := c.table(em)
		want, werr := c.direct()
		if (gerr == nil) != (werr == nil) {
			t.Errorf("%s: table error %v, direct error %v", c.name, gerr, werr)
			continue
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%s: the primary table emits %d bytes that differ from the direct call's %d", c.name, len(got), len(want))
		}
		if _, declared := KernelISA(got); declared {
			t.Errorf("%s: an AVX2-tier kernel carries an ISA declaration", c.name)
		}
	}
}

// TestSSETableKernelsPassTheVEXGate runs every SSE-tier kernel the table can
// emit through the VEX-leak gate, and lists the ops that have no SSE
// kernel.
//
// A family that emits a kernel for a field gets it objdump-checked at every
// shape above; undeclared (AVX2-looking) bytes from an SSE field fail here.
func TestSSETableKernelsPassTheVEXGate(t *testing.T) {
	em := EmittersFor(TierSSE)
	var pending []string
	checked, refused := 0, 0
	for _, c := range emitterCalls() {
		code, err := c.table(em)
		switch {
		case errors.Is(err, ErrNoSSEKernel):
			pending = append(pending, c.name)
			continue
		case err != nil:
			refused++ // a decided refusal (wide, tiled) or an unsupported shape
			continue
		}
		if KernelTier(code) != TierSSE {
			t.Errorf("%s: the SSE table returned a kernel that is not declared SSE-tier", c.name)
		}
		requireSSEKernel(t, strings.ReplaceAll(c.name, "/", "_"), code)
		checked++
	}
	sort.Strings(pending)
	t.Logf("%d SSE-tier kernels passed the VEX-leak gate, %d refused by decision, %d pending a family: %s",
		checked, refused, len(pending), strings.Join(pending, " "))
}
