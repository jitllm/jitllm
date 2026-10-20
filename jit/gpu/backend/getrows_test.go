package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// TestGetRowsPackedMatchesDequant holds the device embedding lookup to the
// GGUF dequantizer on the source bytes -- not to a reader of the packed layout,
// so the packer's indexing and the kernel's cannot agree on a shared mistake.
//
// Every packed format the device reads, a row count that is not a multiple of
// four (the d plane pairs rows two and four to a word, so an odd count has a
// half-filled last word), ids that include the first and last rows and repeat
// one, and an embedding scale -- gemma's sqrt(n_embd) is how a real model
// reaches it. The comparison is to float32 rounding: the kernel computes
// scale*q - bias, as the host reader does, and may fuse it.
func TestGetRowsPackedMatchesDequant(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		for _, q := range []kernels.Quant{kernels.Q4_0, kernels.Q5_0, kernels.Q5_1, kernels.Q8_0, kernels.Q3_K,
			kernels.Q4_K, kernels.Q5_K, kernels.Q6_K, kernels.TQ2, kernels.TQ2F} {
			for _, c := range []struct {
				rows, k int
				ids     []uint32
				scale   float32
			}{
				{37, 512, []uint32{0, 36, 5, 5, 17, 1, 37, 3}, 1}, // 37: a padding row
				{130, 256, []uint32{129, 0, 64, 65, 66, 67, 128}, 45.25},
			} {
				name := fmt.Sprintf("%s/%v/%dx%d/scale%g", d.API(), q, c.rows, c.k, c.scale)
				t.Run(name, func(t *testing.T) {
					getRowsCase(t, d, q, c.rows, c.k, c.ids, c.scale, false)
				})
				ran++
			}
		}
	}
	if ran == 0 {
		t.Skip("no device ran")
	}
}

// TestGetRowsPackedIsGated runs the comparison against a violation and requires
// it to fail (RULE 10): the ids are handed to the kernel shifted by one row, so
// every gathered row is its neighbour -- the same scales and planes, the wrong
// weights, and on a row count of 37 a row whose d slot sits in the other half
// of its word.
func TestGetRowsPackedIsGated(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		for _, q := range []kernels.Quant{kernels.Q8_0, kernels.Q4_K, kernels.Q6_K} {
			t.Run(fmt.Sprintf("%s/%v", d.API(), q), func(t *testing.T) {
				getRowsCase(t, d, q, 37, 512, []uint32{0, 7, 20, 35}, 1, true)
			})
		}
	}
}

func getRowsCase(t *testing.T, d backend.Device, q kernels.Quant, rows, k int, ids []uint32, scale float32, violate bool) {
	t.Helper()
	kk, err := oldkernels.GetRowsPacked(q, rows, k, len(ids), scale)
	if err != nil {
		t.Fatal(err)
	}
	kern, err := d.Compile(kk)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer kern.Close()
	rng := rand.New(rand.NewSource(int64(rows*31 + k + int(q))))
	raw := rawWeights(q, rows, k, rng)
	gg, ok := ggufOf[q]
	if q == kernels.Q3_K {
		// rawWeights plants no Q3_K scale; a random f16 there is NaN often.
		gg, ok = quant.Q3_K, quant.PlantScales(quant.Q3_K, raw, rows+k)
	}
	if !ok {
		t.Fatalf("%v: no GGUF twin to dequantize the source with", q)
	}
	qs, dw, scw, err := kernels.PackWeights(q, raw, rows, k)
	if err != nil {
		t.Fatal(err)
	}
	wf := make([]float32, rows*k)
	if err := quant.Dequant32(gg, raw, wf); err != nil {
		t.Fatal(err)
	}
	g := newGPU(t, d)
	defer g.free()
	send := append([]uint32(nil), ids...)
	if violate {
		for i := range send {
			send[i] = (send[i] + 1) % uint32(rows)
		}
	}
	out := g.up(f32bytes(nanFill(len(ids) * k)))
	if err := kern.Launch((oldkernels.GetRowsThreads(q, k, len(ids))+127)/128, 128,
		g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw)), g.up(u32bytes(send)), out); err != nil {
		t.Fatal(err)
	}
	rb := make([]byte, len(ids)*k*4)
	if err := out.Read(rb); err != nil {
		t.Fatal(err)
	}
	var sse, sy2 float64
	for i, id := range ids {
		for e := 0; e < k; e++ {
			got := float64(math.Float32frombits(binary.LittleEndian.Uint32(rb[(i*k+e)*4:])))
			want := 0.0
			if int(id) < rows {
				want = float64(wf[int(id)*k+e] * scale)
			}
			if math.IsNaN(got) || math.IsInf(got, 0) {
				if violate {
					return
				}
				t.Fatalf("row %d (id %d) elem %d: %v", i, id, e, got)
			}
			sse += (got - want) * (got - want)
			sy2 += want * want
		}
	}
	if sy2 == 0 {
		t.Fatal("the reference is all zero; the oracle is degenerate")
	}
	nmse := sse / sy2
	if violate {
		if nmse < 1e-3 {
			t.Fatalf("the violation (ids shifted by one row) read NMSE %.3e: the gate cannot see a wrong row", nmse)
		}
		t.Logf("violation NMSE %.3e (fails the 1e-12 bound, as it must)", nmse)
		return
	}
	t.Logf("NMSE %.3e", nmse)
	if nmse > 1e-12 {
		t.Fatalf("NMSE %.3e against the GGUF dequantizer", nmse)
	}
}
