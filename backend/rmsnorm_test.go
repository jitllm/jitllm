package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestRMSNormRowsMatchesFloat64: the one-launch norm against a float64
// RMSNorm -- widths the tier uses, ragged ones below and above a group, one
// and three rows, with and without the residual add (whose SUM is checked
// too) and the gemma +1 weight. Outputs are poisoned with NaN with a guard
// word after each, and a non-finite value fails before any tolerance.
func TestRMSNormRowsMatchesFloat64(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(11))
	for _, d := range devs {
		defer d.Close()
		for _, k := range []int{4096, 2048, 5120, 100, 7, 1500} {
			for _, rows := range []int{1, 3} {
				for _, add := range []bool{false, true} {
					for _, addOne := range []bool{false, true} {
						name := fmt.Sprintf("%s/k%d/r%d/add%v/one%v", d.API(), k, rows, add, addOne)
						rmsCase(t, d, rng, name, k, rows, add, addOne, 1e-5, false)
						if d.API() == "ptx" { // the warp form needs contiguous 32-thread subgroups
							rmsCase(t, d, rng, name+"/warp", k, rows, add, addOne, 1e-5, true)
						}
					}
				}
			}
		}
	}
}

func rmsCase(t *testing.T, d backend.Device, rng *rand.Rand, name string, k, rows int, add, addOne bool, eps float32, warp bool) {
	x, y, w := make([]float32, k*rows), make([]float32, k*rows), make([]float32, k)
	for i := range x {
		x[i], y[i] = float32(rng.NormFloat64()*3), float32(rng.NormFloat64())
	}
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	mk := kernels.RMSNormRows
	if warp {
		mk = kernels.RMSNormRowsWarp
	}
	kern, err := mk(k, rows, eps, addOne, add)
	if err != nil {
		t.Fatal(err)
	}
	if err := kern.Validate(); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	kk, err := d.Compile(kern)
	if err != nil {
		t.Fatalf("%s: compile: %v", name, err)
	}
	defer kk.Close()
	g := newGPU(t, d)
	defer g.free()
	const guard = 0x5EED5EED
	poisoned := func() backend.Buf {
		p := make([]float32, k*rows+1)
		for i := range p {
			p[i] = float32(math.NaN())
		}
		b := f32bytes(p)
		binary.LittleEndian.PutUint32(b[k*rows*4:], guard)
		return g.up(b)
	}
	bx, by, bw := g.up(f32bytes(x)), g.up(f32bytes(y)), g.up(f32bytes(w))
	bout, bsum := poisoned(), poisoned()
	bufs := []backend.Buf{bx, bw, bout}
	if add {
		bufs = []backend.Buf{bx, by, bw, bsum, bout}
	}
	if err := kk.Launch(rows, kernels.RMSNormGroup, bufs...); err != nil {
		t.Fatalf("%s: launch: %v", name, err)
	}
	read := func(b backend.Buf) []float32 {
		raw := make([]byte, (k*rows+1)*4)
		if err := b.Read(raw); err != nil {
			t.Fatal(err)
		}
		if gw := binary.LittleEndian.Uint32(raw[k*rows*4:]); gw != guard {
			t.Fatalf("%s: wrote past the output (guard %#x)", name, gw)
		}
		v := make([]float32, k*rows)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return v
	}
	out := read(bout)
	var sum []float32
	if add {
		sum = read(bsum)
	}
	for r := 0; r < rows; r++ {
		var ss float64
		v := make([]float64, k)
		for i := 0; i < k; i++ {
			v[i] = float64(x[r*k+i])
			if add {
				v[i] += float64(y[r*k+i])
				if s := sum[r*k+i]; s != float32(v[i]) {
					t.Fatalf("%s: row %d sum[%d] = %v, want %v", name, r, i, s, float32(v[i]))
				}
			}
			ss += v[i] * v[i]
		}
		scale := 1 / math.Sqrt(ss/float64(k)+float64(eps))
		for i := 0; i < k; i++ {
			wi := float64(w[i])
			if addOne {
				wi++
			}
			want, got := v[i]*scale*wi, float64(out[r*k+i])
			if math.IsNaN(got) || math.IsInf(got, 0) {
				t.Fatalf("%s: row %d [%d] = %v", name, r, i, got)
			}
			if math.Abs(got-want) > 1e-4*(1+math.Abs(want)) {
				t.Fatalf("%s: row %d [%d] = %v, want %v", name, r, i, got, want)
			}
		}
	}
}
