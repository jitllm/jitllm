//go:build jitllmfault

package backend_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestPagedGatesDiscriminate arms each generation fault kernels.SetPagedFault
// knows and demands that the paged gates FAIL on it (RULE 10), on every
// device, for both decode kernels, and that they pass again disarmed:
//
//	go test -tags jitllmfault ./backend/ -run TestPagedGatesDiscriminate
func TestPagedGatesDiscriminate(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	defer kernels.SetPagedFault("")
	const page = 64
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API()+"/"+d.Name(), func(t *testing.T) {
			for _, c := range []struct {
				fault string
				rows  string
				shape kernels.FlashShape
			}{
				{"(t-1)/P", "one-long", kernels.FlashShape{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125}},
				{"unaligned", "rows", kernels.FlashShape{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125}},
				{"normpart", "one-long", kernels.FlashShape{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Splits: 3}},
				{"sinkchunk", "one-long", kernels.FlashShape{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Splits: 3, Sink: true}},
				{"vgroup", "one-long", kernels.FlashShape{Heads: 4, KVHeads: 2, Dim: 256, Scale: .0625}},
				{"vecv", "rows", kernels.FlashShape{Heads: 8, KVHeads: 2, Dim: 128, Scale: .088, VecV: true}},
				{"f16k", "rows", kernels.FlashShape{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, F16K: true}},
				{"f16k", "one-long", kernels.FlashShape{Heads: 4, KVHeads: 2, Dim: 34, Scale: .17, Splits: 2, F16K: true}},
			} {
				for _, kv := range []bool{true, false} {
					if (c.fault == "vgroup" && kv) || (c.fault == "vecv" && (!kv || d.API() == "spirv")) {
						continue // one kernel's V accumulate; vectors where lowered
					}
					t.Run(fmt.Sprintf("%s/kv%v", c.fault, kv), func(t *testing.T) {
						rows := pagedCases(page)[c.rows]
						s := c.shape
						s.Page, s.Rows = page, len(rows)
						build := kernels.FlashAttention
						if kv {
							build = kernels.FlashDecodeKV
						}
						k, err := build(s)
						if err != nil {
							t.Fatal(err)
						}
						if !laneOK(t, d, k) {
							t.Skip("no 32-lane guarantee")
						}
						p := newPagedPool(rand.New(rand.NewSource(5)), page, s.KVHeads*s.Dim, s.F16, rows)
						if s.F16K {
							p.withF16K()
						}
						kernels.SetPagedFault(c.fault)
						bad, berr := pagedAttn(t, d, s, kv, p, p.tab, rows)
						kernels.SetPagedFault("")
						if berr == nil && bad < 1e-6 {
							t.Fatalf("violation %q passed the gate: NMSE %.3g", c.fault, bad)
						}
						good, gerr := pagedAttn(t, d, s, kv, p, p.tab, rows)
						if gerr != nil || good > 1e-10 {
							t.Fatalf("disarmed: NMSE %.3g, %v", good, gerr)
						}
						t.Logf("violation %q: NMSE %.3g %v; clean %.3g", c.fault, bad, berr, good)
					})
				}
			}
			// The staged path shares the page lookup and the partition (its
			// accumulate reads a page id per block of keys), and its softmax
			// writes the partials the merge rescales.
			for _, fault := range []string{"(t-1)/P", "unaligned", "normpart", "f16k"} {
				t.Run(fault+"/staged", func(t *testing.T) {
					rows := pagedCases(page)["one-long"]
					if fault == "unaligned" {
						rows = pagedCases(page)["rows"]
					}
					s := kernels.FlashShape{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Page: page, Chunk: 128, Splits: 4, Sink: true, F16K: fault == "f16k"}
					p := newPagedPool(rand.New(rand.NewSource(5)), page, s.KVHeads*s.Dim, s.F16, rows)
					if s.F16K {
						p.withF16K()
					}
					kernels.SetPagedFault(fault)
					bad, berr := pagedStaged(t, d, s, 1, 1, p, p.tab, rows)
					kernels.SetPagedFault("")
					if berr == nil && bad < 1e-6 {
						t.Fatalf("violation %q passed the staged gate: NMSE %.3g", fault, bad)
					}
					good, gerr := pagedStaged(t, d, s, 1, 1, p, p.tab, rows)
					if gerr != nil || good > 1e-10 {
						t.Fatalf("disarmed: NMSE %.3g, %v", good, gerr)
					}
					t.Logf("staged violation %q: NMSE %.3g %v; clean %.3g", fault, bad, berr, good)
				})
			}
			// The contiguous FlashAttention groups its V loads as the paged one
			// does: a group that weights every key by its first key's
			// probability must fail its gate too.
			for _, dim := range []int{64, 256} {
				t.Run(fmt.Sprintf("vgroup/contiguous/d%d", dim), func(t *testing.T) {
					s := kernels.FlashShape{Heads: 4, KVHeads: 2, Dim: dim, Rows: 3, KStride: 128, Scale: .125, F16: dim == 256}
					if k, err := kernels.FlashAttention(s); err != nil || !laneOK(t, d, k) {
						t.Skip("no 32-lane guarantee")
					}
					kernels.SetPagedFault("vgroup")
					bad, berr := flashDecodeNMSE(t, d, s, 97, false)
					kernels.SetPagedFault("")
					if berr == nil && bad < 1e-6 {
						t.Fatalf("grouped V weighting every key by its group's first passed: NMSE %.3g", bad)
					}
					good, gerr := flashDecodeNMSE(t, d, s, 97, false)
					if gerr != nil || good > 1e-10 {
						t.Fatalf("disarmed: NMSE %.3g, %v", good, gerr)
					}
					t.Logf("contiguous vgroup violation: NMSE %.3g %v; clean %.3g", bad, berr, good)
				})
			}
			// The writers share the page lookup: (t-1)/P misplaces the first slot
			// of every page.
			t.Run("f16krope/writers", func(t *testing.T) {
				kernels.SetPagedFault("f16krope")
				err := pagedWriteCheck(t, d, page, 2, 64, 64, true, false, true, []pagedRow{{0, 0, page}, {0, 0, 2*page + 3}, {5, 5, 5}})
				kernels.SetPagedFault("")
				if err == nil {
					t.Fatal("the binary16 K writers passed with a word's halves swapped")
				}
				t.Logf("writers under f16krope: %v", err)
				if err := pagedWriteCheck(t, d, page, 2, 64, 64, true, false, true, []pagedRow{{0, 0, page}, {0, 0, 2*page + 3}, {5, 5, 5}}); err != nil {
					t.Fatalf("disarmed: %v", err)
				}
			})
			t.Run("(t-1)/P/writers", func(t *testing.T) {
				kernels.SetPagedFault("(t-1)/P")
				err := pagedWriteCheck(t, d, page, 2, 64, 64, true, false, false, []pagedRow{{0, 0, page}, {0, 0, 2*page + 3}})
				kernels.SetPagedFault("")
				if err == nil {
					t.Fatal("the writers passed with page index (t-1)/P")
				}
				t.Logf("writers under (t-1)/P: %v", err)
			})
		})
	}
}
