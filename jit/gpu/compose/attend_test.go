package compose_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// attendPlans are the paged decode plans the attention former is held to:
// the shapes TestAttnIRUnchanged pins (every schedule knob one at a time),
// binary16 K, a latent row and its value-tail violation, and every plan
// kernels.ChooseDecodePlan picks over key counts on each API.
func attendPlans() []kernels.DecodePlan {
	var out []kernels.DecodePlan
	base := []kernels.FlashShape{
		{Heads: 8, KVHeads: 8, Dim: 64, Rows: 1, Page: 64, Scale: 0.125},
		{Heads: 16, KVHeads: 4, Dim: 128, Rows: 1, Page: 128, Scale: 0.0884},
		{Heads: 32, KVHeads: 8, Dim: 256, Rows: 1, Page: 64, Scale: 0.0625},
		{Heads: 16, KVHeads: 2, Dim: 64, Rows: 4, Page: 64, Scale: 0.125},
		{Heads: 4, KVHeads: 2, Dim: 34, Rows: 1, Page: 64, Scale: 0.17},
		{Heads: 8, KVHeads: 2, Dim: 80, Rows: 3, Page: 192, Scale: 0.11},
	}
	knobs := []func(s kernels.FlashShape) kernels.FlashShape{
		func(s kernels.FlashShape) kernels.FlashShape { return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.Splits = 4; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.F16 = true; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.F16K = true; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.F16K, s.KImm = true, true; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.Sink = true; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.Sink, s.Splits = true, 3; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.Softcap = 30; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.Group = 1; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.Warps = 4; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.Warps = 2; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.KImm = true; return s },
		func(s kernels.FlashShape) kernels.FlashShape { s.VecV = true; return s },
	}
	for _, s := range base {
		for _, f := range knobs {
			sh := f(s)
			out = append(out, kernels.DecodePlan{Path: kernels.PathFlashKV, Shape: sh})
			for _, keys := range []int{100, 512, 4096} {
				out = append(out, kernels.DecodePlan{Path: kernels.PathStaged, Shape: kernels.PagedStagedAt(sh, keys, 8)},
					kernels.DecodePlan{Path: kernels.PathStaged, Shape: kernels.PagedStagedPlan(sh, keys, 0)})
			}
		}
		for _, api := range []string{"ptx", "spirv", "msl"} {
			for _, keys := range []int{32, 512, 4096, 16384, 65536} {
				for _, slots := range []int{0, 2560, 81920} {
					sh := s
					sh.KImm, sh.VecV = api == "msl", api == "ptx"
					out = append(out, kernels.ChooseDecodePlan(api, sh, keys, slots))
				}
			}
		}
	}
	for _, mla := range []kernels.FlashShape{
		{Heads: 16, KVHeads: 1, Dim: 576, Rows: 1, Page: 64, MLA: 512, Scale: 0.07},
		{Heads: 32, KVHeads: 1, Dim: 576, Rows: 3, Page: 256, MLA: 512, Scale: 0.07, MLAValueTail: true},
		{Heads: 8, KVHeads: 1, Dim: 192, Rows: 1, Page: 64, MLA: 128, Scale: 0.09, Softcap: 20},
	} {
		for _, keys := range []int{64, 2048, 20000} {
			out = append(out, kernels.ChooseDecodePlan("ptx", mla, keys, 0))
		}
	}
	return out
}

// sameIR reports that two kernels are one program: their parameters, ops,
// workgroup and lane width, whatever they are named.
func sameIR(a, b *ir.Kernel) bool {
	return reflect.DeepEqual(a.Params, b.Params) && reflect.DeepEqual(a.Ops, b.Ops) && a.Group == b.Group && a.Lanes == b.Lanes
}

// TestAttendIsTheMenusIR holds every formed kernel of each plan to the menu
// kernel it replaces, op for op: FlashDecodeKV, the staged scores (one lane
// and a warp a slot), softmax (a thread and a warp a split) and accumulate,
// and both merges. The same ops are the same bits on every device. A formed
// attention whose plan differs from the menu's by one split, and a former
// told the wrong lanes, must not match (the comparison discriminates).
func TestAttendIsTheMenusIR(t *testing.T) {
	type pair struct {
		name       string
		menu, form func() (*ir.Kernel, error)
	}
	n, refused := 0, 0
	for i, p := range attendPlans() {
		s := p.Shape
		var pairs []pair
		if p.Path == kernels.PathFlashKV {
			a := &compose.Attend{S: s}
			pairs = append(pairs, pair{"flashkv", func() (*ir.Kernel, error) { return kernels.FlashDecodeKV(s) }, a.Emit})
			if a.Width() != kernels.FlashKVWidth(s) || a.Groups() != kernels.FlashKVGroups(s) {
				t.Errorf("plan %d: the former launches %dx%d, the menu %dx%d", i, a.Groups(), a.Width(),
					kernels.FlashKVGroups(s), kernels.FlashKVWidth(s))
			}
		} else {
			for _, lanes := range []int{1, 32} {
				sc, sm := &compose.StagedScores{S: s, Lanes: lanes}, &compose.StagedSoftmax{S: s, Lanes: lanes}
				pairs = append(pairs,
					pair{fmt.Sprintf("scores/%d", lanes), func() (*ir.Kernel, error) { return kernels.PagedStagedScores(s, lanes) }, sc.Emit},
					pair{fmt.Sprintf("softmax/%d", lanes), func() (*ir.Kernel, error) { return kernels.PagedStagedSoftmax(s, lanes) }, sm.Emit})
			}
			ac := &compose.StagedAcc{S: s}
			pairs = append(pairs, pair{"acc", func() (*ir.Kernel, error) { return kernels.PagedStagedAcc(s) }, ac.Emit})
		}
		ms := s
		if ms.MLA > 0 {
			ms.Dim, ms.MLA, ms.MLAValueTail = ms.MLA, 0, false
		}
		ms.Splits = max(ms.Splits, 1)
		wide, items := &compose.Merge{S: ms, Wide: true}, &compose.Merge{S: ms}
		pairs = append(pairs,
			pair{"mergewide", func() (*ir.Kernel, error) { return kernels.FlashAttentionMergeWide(ms) }, wide.Emit},
			pair{"merge", func() (*ir.Kernel, error) { return kernels.FlashAttentionMerge(ms) }, items.Emit})
		for _, c := range pairs {
			want, werr := c.menu()
			got, gerr := c.form()
			if (werr != nil) != (gerr != nil) {
				t.Errorf("plan %d %s (%+v): the menu says %v, the former %v", i, c.name, s, werr, gerr)
				continue
			}
			if werr != nil {
				refused++
				continue
			}
			n++
			if !sameIR(got, want) {
				t.Errorf("plan %d %s (%+v): the formed kernel is not the menu's (%d ops, the menu %d)",
					i, c.name, s, len(got.Ops), len(want.Ops))
			}
		}
	}
	if n < 500 {
		t.Fatalf("only %d kernels compared: the sweep does not reach the formers", n)
	}
	t.Logf("%d formed kernels are the menu's, %d shapes refused by both", n, refused)

	// The violations: a plan one split off, and a softmax at the other lanes.
	s := kernels.FlashShape{Heads: 16, KVHeads: 4, Dim: 128, Rows: 1, Page: 128, Scale: 0.0884, Splits: 4}
	want, err := kernels.FlashDecodeKV(s)
	if err != nil {
		t.Fatal(err)
	}
	off := s
	off.Splits = 5
	got, err := (&compose.Attend{S: off}).Emit()
	if err != nil {
		t.Fatal(err)
	}
	if sameIR(got, want) {
		t.Error("an attention formed at 5 splits matched the menu's at 4")
	}
	st := kernels.PagedStagedAt(s, 4096, 8)
	want, err = kernels.PagedStagedSoftmax(st, 32)
	if err != nil {
		t.Fatal(err)
	}
	if got, err = (&compose.StagedSoftmax{S: st, Lanes: 1}).Emit(); err != nil || sameIR(got, want) {
		t.Errorf("a softmax formed a thread a split matched the warp's: %v", err)
	}
}
