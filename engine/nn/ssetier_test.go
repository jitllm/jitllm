//go:build (amd64 || arm64) && jitllmtest

package nn

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestForcedSSETierSelectsSSEKernels is the selection gate the SSE families
// build on: under cpu.ForceTierForTest(TierSSE) every nn entry point must take
// its kernel from the SSE tier's table, and none may reuse an AVX2 kernel an
// earlier call cached.
//
// For each op, either the SSE table has no kernel (ErrNoSSEKernel) and the
// call must panic naming it, or it has one and the call must run. The failure
// is the third outcome: the call runs with no SSE kernel, meaning a cache keyed
// without the tier handed over an AVX2 kernel warmed below. And no AVX2 kernel
// may be mapped while the force is on (cpu.MappedByTier).
func TestForcedSSETierSelectsSSEKernels(t *testing.T) {
	const n = 67 // ragged: eight whole units and a tail of three
	x, y, w, b := vec(n, 1), vec(n, 2), vec(n, 3), vec(n, 4)
	cs := vec(32, 5) // a rotary table for hd 64 rotating 32 dims (partial)
	halves := make([]byte, 2*n)
	gate := [6][]float32{vec(n, 6), vec(n, 7), vec(n, 8), vec(n, 9), vec(n, 10), vec(n, 11)}
	pk := packOne(t, quant.Q4_K, 8, 256)
	sse := cpu.EmittersFor(cpu.TierSSE)
	ops := []struct {
		name string
		run  func()
		have func() error // the SSE table's answer for the kernel run needs
	}{
		{"softmax", func() { Softmax32JIT(append([]float32(nil), x...), n) }, errOf(sse.Softmax)},
		{"axpy", func() { Axpy32JIT(append([]float32(nil), x...), y, 0.5) }, errOf(sse.Axpy)},
		{"scale", func() { Scale32JIT(append([]float32(nil), x...), 0.5) }, errOf(sse.Scale)},
		{"softcap", func() { Softcap32JIT(append([]float32(nil), x...), 30) }, errOf(sse.Softcap)},
		{"sigmoidmul", func() { SigmoidMul32JIT(append([]float32(nil), x...), y) }, errOf(sse.SigmoidMul)},
		{"actmul/silu", func() { ActMul32JIT(append([]float32(nil), x...), y, ActSiLU) },
			func() error { return errOfK(sse.ActMul, ActSiLU) }},
		{"act/gelu", func() { Act32JIT(append([]float32(nil), x...), ActGELU) },
			func() error { return errOfK(sse.Act, ActGELU) }},
		{"rmsnorm", func() { RMSNorm32JIT(make([]float32, n), x, w, 1e-6) },
			func() error { _, err := sse.RMSNorm(n); return err }},
		{"layernorm", func() { LayerNorm32JIT(make([]float32, n), x, w, b, 1e-6) },
			func() error { _, err := sse.LayerNorm(n, true); return err }},
		{"rope", func() { RoPE32JIT(append([]float32(nil), x[:64]...), 64, cs, true) },
			func() error { _, err := sse.RoPE(64, 32, true); return err }},
		{"widen/f16", func() { Row32JIT(make([]float32, n), quant.F16, halves, 0, n) },
			func() error { _, err := sse.Widen(false); return err }},
		{"packed_row/Q4_K", func() { RowPacked32JIT(make([]float32, 256), quant.Q4_K, pk, 3, 8, 256) },
			func() error { q, _ := kernels.QuantOf(quant.Q4_K); _, err := sse.PackedRow(q); return err }},
		{"delta_gate", func() {
			DeltaGate32JIT(make([]float32, n), make([]float32, n), gate[0], gate[1], gate[2], gate[3])
		}, errOf(sse.DeltaGate)},
	}

	// Warm every package-level cache on the host's own tier first. A cache
	// keyed without the tier would hand the forced run exactly these. On a
	// host that is itself SSE-tier (an Atom) there is nothing to warm, and a
	// pending family panics here exactly as it does below.
	for _, op := range ops {
		if msg := panicked(op.run); msg != "" && cpu.HostTier() != cpu.TierSSE {
			t.Fatalf("%s on the host's own tier (%v): %s", op.name, cpu.HostTier(), msg)
		}
	}

	old := cpu.ForceTierForTest(cpu.TierSSE)
	defer cpu.ForceTierForTest(old)
	if cpu.HostTier() != cpu.TierSSE {
		t.Skipf("the tier could not be forced to SSE on this build (host tier %v)", cpu.HostTier())
	}
	avx2Before := cpu.MappedByTier()[cpu.TierAVX2]

	for _, op := range ops {
		missing := op.have()
		msg := panicked(op.run)
		switch {
		case missing != nil && !errors.Is(missing, cpu.ErrNoSSEKernel):
			t.Errorf("%s: the SSE table refused with %v, which is not a pending family", op.name, missing)
		case missing != nil && msg == "":
			t.Errorf("%s: RAN on the forced SSE tier although the SSE table has no kernel for it -- "+
				"a cache keyed without the tier served the AVX2 kernel", op.name)
		case missing != nil && !strings.Contains(msg, "SSE-tier"):
			t.Errorf("%s: panicked with %q, which does not name the missing SSE-tier kernel", op.name, msg)
		case missing == nil && msg != "":
			t.Errorf("%s: the SSE table has a kernel and the call panicked: %s", op.name, msg)
		}
		// A second call must say the same thing, not dereference a half-filled
		// cache (tierOnce's reason).
		if missing != nil && msg != "" {
			if again := panicked(op.run); !strings.Contains(again, "SSE-tier") {
				t.Errorf("%s: the second call panicked with %q -- the first failure latched", op.name, again)
			}
		}
	}

	// The JIT records the forced tier and builds from its table.
	f := NewJIT(256, 64, []quant.Type{quant.Q4_K, quant.F32})
	defer f.Close()
	if f.tier != cpu.TierSSE || f.em != sse {
		t.Errorf("a JIT built under the force records tier %v", f.tier)
	}
	if _, err := sse.PackedMatVec(quant.Q4_K, cpu.PackedRows); errors.Is(err, cpu.ErrNoSSEKernel) && f.packed[quant.Q4_K] != nil {
		t.Error("the JIT built a Q4_K packed kernel on the SSE tier, which has none yet -- it came from another tier")
	}
	if _, err := sse.RowMajor(cpu.Spec{W: quant.F32, Rows: 1, Cols: 1, Accs: 1}); errors.Is(err, cpu.ErrNoSSEKernel) && f.code[quant.F32] != nil {
		t.Error("the JIT built an F32 matvec on the SSE tier, which has none yet -- it came from another tier")
	}
	if _, err := sse.AttnScores(64, 256, cpu.KVF32); errors.Is(err, cpu.ErrNoSSEKernel) {
		if msg := panicked(func() { f.AddAttn(64, 256, cpu.KVF32) }); !strings.Contains(msg, "SSE-tier") {
			t.Errorf("AddAttn under the force: %q, want the missing SSE-tier kernel named", msg)
		}
	}
	if f.MatMul(make([]float32, 64*2), quant.F32, make([]byte, 64*256*4), make([]float32, 2*256), 64, 256, 2) {
		t.Error("nn.MatMul ran on the SSE tier, which has no GGUF GEMM")
	}

	if d := cpu.MappedByTier()[cpu.TierAVX2] - avx2Before; d != 0 {
		t.Errorf("%d AVX2 kernels were mapped while the SSE tier was forced", d)
	}
}

func vec(n int, seed float32) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(math.Sin(float64(seed) + float64(i)*0.37))
	}
	return v
}

func errOf(f func() ([]byte, error)) func() error {
	return func() error { _, err := f(); return err }
}

func errOfK(f func(cpu.ActKind) ([]byte, error), k cpu.ActKind) error {
	_, err := f(k)
	return err
}

// panicked runs f and returns the panic's text, or "" if it returned.
func panicked(f func()) (msg string) {
	defer func() {
		if r := recover(); r != nil {
			msg = fmt.Sprint(r)
			if msg == "" {
				msg = "(empty panic)"
			}
		}
	}()
	f()
	return ""
}
