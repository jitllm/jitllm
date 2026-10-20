//go:build jitllmbench

package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/jitllm/jitllm/dev/bench"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestFlashAttentionAB measures the actual staged sequence, including its MMA
// scores where available. Both arms run in sessions and synchronize equally.
func TestFlashAttentionAB(t *testing.T) {
	gpuLock(t)
	ds := backend.Open()
	if len(ds) == 0 {
		t.Fatal("no GPU")
	}
	for _, d := range ds {
		defer d.Close()
		if api := os.Getenv("JITLLM_FLASH_API"); api != "" && api != d.API() {
			continue
		}
		for _, rows := range []int{1, 64} {
			depths := []int{128, 2048}
			if dd := os.Getenv("JITLLM_FLASH_DEPTH"); dd != "" {
				var x int
				fmt.Sscanf(dd, "%d", &x)
				depths = []int{x}
			}
			for _, depth := range depths {
				t.Run(fmt.Sprintf("%s/r%d/n%d", d.API(), rows, depth), func(t *testing.T) {
					heads, dim, nkv := 32, 64, 4
					// JITLLM_FLASH_SHAPE=heads,kvheads,dim (Phi-4-mini is 24,8,128).
					if sh := os.Getenv("JITLLM_FLASH_SHAPE"); sh != "" {
						fmt.Sscanf(sh, "%d,%d,%d", &heads, &nkv, &dim)
					}
					g := newGPU(t, d)
					defer g.free()
					compile := func(k *ir.Kernel, e error) backend.Kernel {
						if e != nil {
							t.Fatal(e)
						}
						c, e := d.Compile(k)
						if e != nil {
							t.Fatal(e)
						}
						t.Cleanup(c.Close)
						return c
					}
					shape := mainkernels.FlashShape{Heads: heads, KVHeads: nkv, Dim: dim, Rows: rows, KStride: depth, Scale: .125}
					if rows == 1 && depth >= 512 {
						shape.Splits = 8
					}
					// JITLLM_FLASH_KV=1 times mainkernels.FlashDecodeKV as the fused arm.
					kv := os.Getenv("JITLLM_FLASH_KV") == "1" && rows == 1
					fGroups, fWidth := rows*heads*max(1, shape.Splits), 32
					build := mainkernels.FlashAttention
					if kv {
						// JITLLM_FLASH_GROUP / JITLLM_FLASH_SPLITS pin the KV kernel's shape.
						fmt.Sscanf(os.Getenv("JITLLM_FLASH_GROUP"), "%d", &shape.Group)
						fmt.Sscanf(os.Getenv("JITLLM_FLASH_SPLITS"), "%d", &shape.Splits)
						c := mainkernels.FlashKVChunk(shape)
						shape.Splits = max(shape.Splits, (depth+c-1)/c)
						build = mainkernels.FlashDecodeKV
						fGroups, fWidth = mainkernels.FlashKVGroups(shape), mainkernels.FlashKVWidth(shape)
					}
					f := compile(build(shape))
					var merge backend.Kernel
					if shape.Splits > 1 {
						merge = compile(mainkernels.FlashAttentionMerge(shape))
					}
					qt, kt, at := 1, 1, 1
					if rows > 1 {
						qt, kt, at = 4, 2, 4
					}
					score := compile(mainkernels.AttnScoresTiled(heads, dim, nkv*dim, heads/nkv, depth, .125, rows, qt, kt, depth))
					scoreGroups, scoreWidth := ((rows/qt)*heads*((depth+kt-1)/kt)+127)/128, 128
					if rows > 1 {
						k, e := mainkernels.AttnScoresMMA(heads, dim, nkv*dim, heads/nkv, depth, .125, rows, depth, 4)
						if e == nil {
							if c, e := d.Compile(k); e == nil {
								t.Cleanup(c.Close)
								score = c
								scoreGroups = ((rows/16)*heads*((depth+31)/32) + 3) / 4
								scoreWidth = 128
								t.Log("baseline uses tensor-core scores")
							}
						}
					}
					soft := compile(mainkernels.SoftmaxRows(heads, depth, 32, rows, at))
					acc := compile(mainkernels.AttnAccTiled(heads, dim, nkv*dim, heads/nkv, depth, rows, at))
					accSplit := 1
					var reduce backend.Kernel
					if rows == 1 {
						slots := d.Slots()
						if slots <= 0 {
							slots = 16384
						}
						for accSplit < 32 && heads*dim*accSplit*2 <= slots {
							accSplit *= 2
						}
						acc = compile(mainkernels.AttnAccSplit(heads, dim, nkv*dim, heads/nkv, depth, accSplit))
						reduce = compile(mainkernels.Reduce(heads*dim, accSplit))
					}
					t.Logf("fused splits=%d; staged accumulator splits=%d", max(1, shape.Splits), accSplit)
					rng := rand.New(rand.NewSource(83))
					data := func(n int) backend.Buf {
						a := make([]float32, n)
						for i := range a {
							a[i] = float32(rng.NormFloat64())
						}
						return g.up(f32bytes(a))
					}
					q, k, v := data(rows*heads*dim), data(depth*nkv*dim), data(depth*nkv*dim)
					ns := []uint32{uint32(depth)}
					for r := 0; r < rows; r++ {
						ns = append(ns, uint32(depth-rows+1+r))
					}
					nb := g.up(u32bytes(ns))
					scores, probs, out := data(rows*heads*depth), data(rows*heads*depth), data(rows*heads*dim)
					partial := data(mainkernels.FlashPartialFloats(shape))
					accPart := data(heads * dim * accSplit)
					run := func(fused bool) func(int) {
						return func(iters int) {
							d.Session(func(s backend.Session) {
								launch := func(c backend.Kernel, groups, width int, args ...backend.Buf) {
									if e := s.Launch(c, groups, width, args...); e != nil {
										panic(e)
									}
								}
								for i := 0; i < iters; i++ {
									if fused {
										dst := out
										if merge != nil {
											dst = partial
										}
										launch(f, fGroups, fWidth, q, k, v, nb, dst)
										if merge != nil {
											launch(merge, rows*heads, 32, partial, out)
										}
									} else {
										launch(score, scoreGroups, scoreWidth, q, k, nb, scores)
										launch(soft, rows*heads, 32, scores, nb, probs)
										if reduce != nil {
											launch(acc, (heads*dim*accSplit+127)/128, 128, probs, v, nb, accPart)
											launch(reduce, (heads*dim+127)/128, 128, accPart, out)
										} else {
											launch(acc, (rows*heads*dim/at+127)/128, 128, probs, v, nb, out)
										}
									}
								}
								if e := s.Sync(); e != nil {
									panic(e)
								}
							})
						}
					}
					// Validate the measured launch sequences before timing either arm.
					read := func() []float32 {
						raw := make([]byte, rows*heads*dim*4)
						if e := out.Read(raw); e != nil {
							t.Fatal(e)
						}
						vals := make([]float32, rows*heads*dim)
						for i := range vals {
							vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
						}
						return vals
					}
					run(false)(1)
					want := read()
					run(true)(1)
					got := read()
					var se, ss float64
					for i, a := range want {
						b := got[i]
						if (math.IsNaN(float64(a)) || math.IsNaN(float64(b)) || math.IsInf(float64(a), 0) || math.IsInf(float64(b), 0)) && os.Getenv("JITLLM_FLASH_NOCHECK") != "1" {
							t.Fatal("nonfinite benchmark output")
						}
						delta := float64(a - b)
						se += delta * delta
						ss += float64(a) * float64(a)
					}
					// JITLLM_FLASH_NOCHECK=1 times a phase probe that is wrong on purpose.
					if nmse := se / max(ss, 1e-30); nmse > 1e-5 && os.Getenv("JITLLM_FLASH_NOCHECK") != "1" {
						t.Fatalf("benchmark sequences differ: NMSE %g", nmse)
					}
					a := bench.Case{Name: "staged", Fn: run(false)}
					b := bench.Case{Name: "fused", Fn: run(true)}
					for until := time.Now().Add(2 * time.Second); time.Now().Before(until); {
						a.Fn(100)
						b.Fn(100)
					}
					aa := bench.AB(a, a, 20, 100)
					t.Logf("control %s", aa)
					if !aa.Stable() || math.Abs(aa.Median-1) > .05 {
						t.Skip("unstable A/A control; reject this measurement")
					}
					for pass := 0; pass < 2; pass++ {
						r := bench.AB(a, b, 20, 100)
						t.Logf("pass%d %s; staged score/prob scratch=%d bytes; fused partials=%d bytes; shared=128 bytes/group", pass, r, rows*heads*depth*8, func() int {
							if shape.Splits > 1 {
								return mainkernels.FlashPartialFloats(shape) * 4
							}
							return 0
						}())
					}
				})
			}
		}
	}
}
