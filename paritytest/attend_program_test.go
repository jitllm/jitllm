package paritytest

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// noLanes is a target that promises no subgroup width.
var noLanes = compose.Target{API: compose.APISPIRV}

// softmaxProgram is a staged softmax of shape s as a program of one prim.
func softmaxProgram(s mainkernels.FlashShape) *compose.Program {
	p := &compose.Program{Rows: s.Rows, Vals: []compose.Value{{Buf: -1}, {Elem: compose.ValU32, Buf: -1}, {Buf: -1}, {Buf: -1}}}
	p.Prims = []compose.Prim{{Kind: compose.PrimSoftmax, In: []int{0, 1}, Out: []int{2, 3}, Paged: pagedOf(s)}}
	return p
}

// TestSoftmaxPrimIsMains holds the softmax prim, lowered naively and planned
// for the target, to main's staged softmax: at one lane where the target
// promises no subgroup, over 32 where it does.
func TestSoftmaxPrimIsMains(t *testing.T) {
	for _, s := range stagedShapes {
		for _, lanes := range []int{1, 32} {
			name := fmt.Sprintf("r%d_h%d_kv%d_d%d_s%d_c%d_l%d", s.Rows, s.Heads, s.KVHeads, s.Dim, s.Splits, s.Chunk, lanes)
			t.Run(name, func(t *testing.T) {
				mk, err := mainkernels.PagedStagedSoftmax(s, lanes)
				if err != nil {
					t.Fatal(err)
				}
				gs, err := softmaxProgram(s).Lower()
				if err != nil {
					t.Fatal(err)
				}
				f := compose.Facts{Target: noLanes}
				if lanes > 1 {
					f = compose.Facts{Target: sm70, RowGroup: 64, GroupRows: true}
				}
				same(t, name, []*ir.Kernel{mk}, planned(t, &gs[0].Row, f))
			})
		}
	}
}

// TestMergePrimIsMains holds the merge prim, lowered alone, to main's
// FlashAttentionMergeWide, with and without sinks.
func TestMergePrimIsMains(t *testing.T) {
	for _, s := range stagedShapes {
		for _, sink := range []bool{false, true} {
			s.Sink = sink
			name := fmt.Sprintf("r%d_h%d_d%d_s%d_sink%v", s.Rows, s.Heads, s.Dim, s.Splits, sink)
			t.Run(name, func(t *testing.T) {
				ms := s
				if s.MLA > 0 {
					ms.Dim, ms.MLA = s.MLA, 0
				}
				mk, err := mainkernels.FlashAttentionMergeWide(ms)
				if err != nil {
					t.Fatal(err)
				}
				p := &compose.Program{Rows: s.Rows, Vals: []compose.Value{{Buf: -1}, {Buf: -1}, {Buf: -1}}}
				in := []int{0}
				if sink {
					in = append(in, 2)
				}
				p.Prims = []compose.Prim{{Kind: compose.PrimMerge, In: in, Out: []int{1}, Paged: pagedOf(s)}}
				gs, err := p.Lower()
				if err != nil {
					t.Fatal(err)
				}
				same(t, name, []*ir.Kernel{mk}, planned(t, &gs[0].Row, compose.Facts{Target: sm70}))
			})
		}
	}
}

// TestAccPrimIsMains holds the accumulation prim, lowered naively and
// through the passes, to main's staged accumulation at lanes 1.
func TestAccPrimIsMains(t *testing.T) {
	for _, s := range stagedShapes {
		name := fmt.Sprintf("r%d_h%d_kv%d_d%d_m%d_s%d_c%d", s.Rows, s.Heads, s.KVHeads, s.Dim, s.MLA, s.Splits, s.Chunk)
		t.Run(name, func(t *testing.T) {
			mk, err := mainkernels.PagedStagedAcc(s)
			if err != nil {
				t.Fatal(err)
			}
			p := &compose.Program{Rows: s.Rows, Vals: []compose.Value{{Buf: -1}, {Buf: -1}, {Elem: compose.ValU32, Buf: -1}, {Elem: compose.ValU32, Buf: -1}, {Buf: -1}}}
			p.Prims = []compose.Prim{{Kind: compose.PrimAcc, In: []int{0, 1, 2, 3}, Out: []int{4}, Paged: pagedOf(s)}}
			gs, err := p.Lower()
			if err != nil {
				t.Fatal(err)
			}
			ls, err := gs[0].Launches(sm70)
			if err != nil || len(ls) != 1 {
				t.Fatalf("%d launches: %v", len(ls), err)
			}
			k, err := ls[0].Emit()
			if err != nil {
				t.Fatal(err)
			}
			same(t, name, []*ir.Kernel{mk}, []*ir.Kernel{k})
		})
	}
}

// TestScoresPrimIsMains holds the scores prim, lowered naively and through
// the passes, to main's staged scores at lanes 1 over f32 K.
func TestScoresPrimIsMains(t *testing.T) {
	for _, s := range stagedShapes {
		name := fmt.Sprintf("r%d_h%d_kv%d_d%d_m%d_s%d_c%d", s.Rows, s.Heads, s.KVHeads, s.Dim, s.MLA, s.Splits, s.Chunk)
		t.Run(name, func(t *testing.T) {
			mk, err := mainkernels.PagedStagedScores(s, 1)
			if err != nil {
				t.Fatal(err)
			}
			p := &compose.Program{Rows: s.Rows, Vals: []compose.Value{{Buf: -1}, {Buf: -1}, {Elem: compose.ValU32, Buf: -1}, {Elem: compose.ValU32, Buf: -1}, {Buf: -1}}}
			p.Prims = []compose.Prim{{Kind: compose.PrimScores, In: []int{0, 1, 2, 3}, Out: []int{4}, Paged: pagedOf(s), Scale: s.Scale, Softcap: s.Softcap}}
			gs, err := p.Lower()
			if err != nil {
				t.Fatal(err)
			}
			ls, err := gs[0].Launches(sm70)
			if err != nil || len(ls) != 1 {
				t.Fatalf("%d launches: %v", len(ls), err)
			}
			k, err := ls[0].Emit()
			if err != nil {
				t.Fatal(err)
			}
			same(t, name, []*ir.Kernel{mk}, []*ir.Kernel{k})
		})
	}
}
