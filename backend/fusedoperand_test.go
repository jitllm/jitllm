package backend_test

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestFusedOperandsAreTheirTwoKernels holds each of a prompt chunk's fused
// operand kernels to the pair of launches it replaces, bit for bit, every
// output NaN-poisoned first:
//
//   - QuantizeAct against ActMul then Quantize: the float row, the int8
//     words, the scales and the per-16 sums, at windows 32 and 256, over the
//     gated activations, at widths of one block, a ragged count of groups and
//     several;
//   - ActMulF16T against ActMul then ActF16T, for every binary16 layout
//     (16- and 32-element sub-blocks, a secondary plane, a flat byte), at
//     token counts 1, 3 and 32;
//   - RMSNormF16TRows against RMSNormRows then ActF16T, with and without the
//     residual add, both reductions on CUDA, rows 1, 5 and 16.
func TestFusedOperandsAreTheirTwoKernels(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(17))
	ran := 0
	for _, d := range devs {
		defer d.Close()
		compile := func(k *kernelOrErr) backend.Kernel {
			t.Helper()
			if k.err != nil {
				t.Fatal(k.err)
			}
			c, err := d.Compile(k.k)
			if err != nil {
				t.Fatalf("%s: %v", d.API(), err)
			}
			return c
		}
		read := func(b backend.Buf, n int) []byte {
			p := make([]byte, n)
			if err := b.Read(p); err != nil {
				t.Fatal(err)
			}
			return p
		}
		poison := func(g *gpu, n int) backend.Buf { return g.up(f32bytes(nanFill(n))) }
		same := func(name string, a, b []byte) {
			t.Helper()
			if !bytes.Equal(a, b) {
				t.Fatalf("%s: the fused kernel's bytes differ from the two launches'", name)
			}
		}
		for _, kind := range []kernels.ActKind{kernels.ActSiLU, kernels.ActGELU, kernels.ActSwiGLUOAI} {
			for _, n := range []int{32, 4096 + 32*3, 8192 * 3} {
				for _, win := range []int{32, 256} {
					if n%win != 0 {
						continue
					}
					name := fmt.Sprintf("%s/quantact/%v/n%d/w%d", d.API(), kind, n, win)
					g := newGPU(t, d)
					gb, ub := g.up(f32bytes(randActs(n, rng))), g.up(f32bytes(randActs(n, rng)))
					fk := compile(kOf(kernels.QuantizeAct(n, win, kind)))
					ak := compile(kOf(kernels.ActMul(n, kind)))
					qk := compile(kOf(kernels.Quantize(n, win)))
					nb := n / 32
					o1, a1, x1 := poison(g, n), poison(g, n/4), poison(g, 3*nb)
					o2, a2, x2 := poison(g, n), poison(g, n/4), poison(g, 3*nb)
					launchFused(t, fk, kernels.QuantizeThreads(nb), gb, ub, o1, a1, x1)
					launchFused(t, ak, n, gb, ub, o2)
					launchFused(t, qk, kernels.QuantizeThreads(nb), o2, a2, x2)
					same(name+"/float", read(o1, 4*n), read(o2, 4*n))
					same(name+"/int8", read(a1, n), read(a2, n))
					same(name+"/scales", read(x1, 12*nb), read(x2, 12*nb))
					fk.Close()
					ak.Close()
					qk.Close()
					g.free()
					ran++
				}
			}
		}
		// The binary16 operand is the binary16 GEMMs' (PTX's and Metal's); a
		// backend that does not lower ActF16T itself has no operand to fuse.
		if probe, err := kernels.ActF16T(kernels.Q4_K, 1, 256); err == nil {
			c, err := d.Compile(probe)
			if err != nil {
				t.Logf("%s: no binary16 operand here: %v", d.API(), err)
				continue
			}
			c.Close()
		}
		for _, q := range []kernels.Quant{kernels.Q4_K, kernels.Q6_K, kernels.Q8_0, kernels.Q5_K, kernels.Q4_0} {
			for _, ntok := range []int{1, 3, 32} {
				const k = 2048
				n := ntok * k
				name := fmt.Sprintf("%s/actmulf16t/%v/t%d", d.API(), q, ntok)
				g := newGPU(t, d)
				gb, ub := g.up(f32bytes(randActs(n, rng))), g.up(f32bytes(randActs(n, rng)))
				sub, _, _, _ := kernels.Layout(q)
				threads := ntok * k / sub
				fk := compile(kOf(kernels.ActMulF16T(q, ntok, k, kernels.ActSiLU)))
				ak := compile(kOf(kernels.ActMul(n, kernels.ActSiLU)))
				ck := compile(kOf(kernels.ActF16T(q, ntok, k)))
				b1, f2, b2 := poison(g, n/2), poison(g, n), poison(g, n/2)
				launchFused(t, fk, kernels.ActMulF16TThreads(ntok, k), gb, ub, b1)
				launchFused(t, ak, n, gb, ub, f2)
				launchFused(t, ck, threads, f2, b2)
				same(name, read(b1, 2*n), read(b2, 2*n))
				fk.Close()
				ak.Close()
				ck.Close()
				g.free()
				ran++
				for _, add := range []bool{false, true} {
					for _, warp := range []bool{false, true} {
						if warp && d.API() != "ptx" {
							continue
						}
						// The last shape is Qwen2.5-VL's tower FFN norm: a picture's
						// rows at its width, where the V100 read a wrong operand.
						for _, sh := range [][2]int{{k, 1}, {k, 5}, {k, 16}, {1280, 1024}} {
							k, rows := sh[0], sh[1]
							nr := rows * k
							name := fmt.Sprintf("%s/rmsf16t/%v/k%d/r%d/add%v/warp%v", d.API(), q, k, rows, add, warp)
							g := newGPU(t, d)
							mkn := kernels.RMSNormRows
							if warp {
								mkn = kernels.RMSNormRowsWarp
							}
							xb, yb, wb := g.up(f32bytes(randActs(nr, rng))), g.up(f32bytes(randActs(nr, rng))), g.up(f32bytes(randActs(k, rng)))
							nk := compile(kOf(kernels.RMSNormF16TRows(k, rows, 1e-5, false, add, q, warp)))
							pk := compile(kOf(mkn(k, rows, 1e-5, false, add)))
							ck := compile(kOf(kernels.ActF16T(q, rows, k)))
							s1, o1, b1 := poison(g, nr), poison(g, nr), poison(g, nr/2)
							s2, o2, b2 := poison(g, nr), poison(g, nr), poison(g, nr/2)
							args1 := []backend.Buf{xb, wb, o1, b1}
							args2 := []backend.Buf{xb, wb, o2}
							if add {
								args1 = []backend.Buf{xb, yb, wb, s1, o1, b1}
								args2 = []backend.Buf{xb, yb, wb, s2, o2}
							}
							if err := nk.Launch(rows, kernels.RMSNormGroup, args1...); err != nil {
								t.Fatal(err)
							}
							if err := pk.Launch(rows, kernels.RMSNormGroup, args2...); err != nil {
								t.Fatal(err)
							}
							launchFused(t, ck, rows*k/sub, o2, b2)
							same(name+"/norm", read(o1, 4*nr), read(o2, 4*nr))
							same(name+"/operand", read(b1, 2*nr), read(b2, 2*nr))
							if add {
								same(name+"/sum", read(s1, 4*nr), read(s2, 4*nr))
							}
							nk.Close()
							pk.Close()
							ck.Close()
							g.free()
							ran++
						}
					}
				}
			}
		}
	}
	if ran == 0 {
		t.Skip("no device ran a case")
	}
}

// kernelOrErr carries a generator's two results to compile.
type kernelOrErr struct {
	k   *ir.Kernel
	err error
}

func kOf(k *ir.Kernel, err error) *kernelOrErr { return &kernelOrErr{k, err} }

// launchFused runs k over threads in groups of 128.
func launchFused(t *testing.T, k backend.Kernel, threads int, bufs ...backend.Buf) {
	t.Helper()
	if err := k.Launch((threads+127)/128, 128, bufs...); err != nil {
		t.Fatal(err)
	}
}
