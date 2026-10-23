package backend_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestSharedExpertKernelsMatchTheHost gates qwen3next's always-on expert: the
// one-logit gate and the fold that adds it to the routed sum.
//
// Both kernels are per row: a single logit for the whole chunk (RouterMatVec
// with rows=1 counts experts, not tokens) or a fold reading pLogit[0] would
// apply the first row's gate to every row. rows > 1 makes that visible.
func TestSharedExpertKernelsMatchTheHost(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	const (
		k    = 96 // model dimension
		n    = 64 // the block's output width
		rows = 5  // ragged on purpose: not a whole workgroup
	)
	w := make([]float32, k)
	x := make([]float32, rows*k)
	acc := make([]float32, rows*n)
	v := make([]float32, rows*n)
	for i := range w {
		w[i] = float32((i%13)-6) / 8
	}
	for i := range x {
		x[i] = float32((i%17)-8) / 16
	}
	for i := range acc {
		acc[i] = float32(i%11) - 5
		v[i] = float32(i%7) - 3
	}
	// The host's own arithmetic, from engine/model/moe.go: one dot per row, then
	// out += sigma(logit) * shOut.
	wantL := make([]float32, rows)
	wantD := make([]float32, rows*n)
	for r := 0; r < rows; r++ {
		var l float32
		for i := 0; i < k; i++ {
			l += w[i] * x[r*k+i]
		}
		wantL[r] = l
		sig := float32(1 / (1 + math.Exp(-float64(l))))
		for i := 0; i < n; i++ {
			wantD[r*n+i] = acc[r*n+i] + sig*v[r*n+i]
		}
	}

	for _, d := range devs {
		// the gate
		kl, err := kernels.SharedRouterLogits(rows, k)
		if err != nil {
			t.Fatal(err)
		}
		gotL := launch3(t, d, kl, kernels.SharedRouterThreads(rows), f32le(w), f32le(x), rows*4)
		for r := range wantL {
			if math.Abs(float64(gotL[r]-wantL[r])) > 1e-4 {
				t.Fatalf("%s logit[%d] = %v, want %v", d.API(), r, gotL[r], wantL[r])
			}
		}
		// the fold
		kf, err := kernels.SharedExpertAdd(n, rows)
		if err != nil {
			t.Fatal(err)
		}
		gotD := launch4(t, d, kf, rows*n, f32le(acc), f32le(v), f32le(wantL), rows*n*4)
		for i := range wantD {
			if math.Abs(float64(gotD[i]-wantD[i])) > 1e-4 {
				t.Fatalf("%s dst[%d] = %v, want %v (row %d)", d.API(), i, gotD[i], wantD[i], i/n)
			}
		}
	}
}

// launch3 runs a two-input one-output kernel and reads the output as floats.
func launch3(t *testing.T, d backend.Device, k *ir.Kernel, grid int, a, b []byte, outBytes int) []float32 {
	t.Helper()
	return launchN(t, d, k, grid, [][]byte{a, b}, outBytes)
}

func launch4(t *testing.T, d backend.Device, k *ir.Kernel, grid int, a, b, c []byte, outBytes int) []float32 {
	t.Helper()
	return launchN(t, d, k, grid, [][]byte{a, b, c}, outBytes)
}

func launchN(t *testing.T, d backend.Device, k *ir.Kernel, grid int, in [][]byte, outBytes int) []float32 {
	t.Helper()
	if err := k.Validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	kern, err := d.Compile(k)
	if err != nil {
		t.Fatalf("%s: compile: %v", d.API(), err)
	}
	defer kern.Close()
	bufs := make([]backend.Buf, 0, len(in)+1)
	for _, b := range in {
		buf, err := d.Alloc(len(b))
		if err != nil {
			t.Fatalf("alloc: %v", err)
		}
		defer buf.Free()
		if err := buf.Write(b); err != nil {
			t.Fatalf("write: %v", err)
		}
		bufs = append(bufs, buf)
	}
	out, err := d.Alloc(outBytes)
	if err != nil {
		t.Fatalf("alloc out: %v", err)
	}
	defer out.Free()
	bufs = append(bufs, out)
	if err := kern.Launch((grid+127)/128, 128, bufs...); err != nil {
		t.Fatalf("launch: %v", err)
	}
	raw := make([]byte, outBytes)
	if err := out.Read(raw); err != nil {
		t.Fatalf("read: %v", err)
	}
	v := make([]float32, outBytes/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return v
}
