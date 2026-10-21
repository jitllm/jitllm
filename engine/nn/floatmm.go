//go:build amd64 || arm64

package nn

import (
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// The float GEMM: a batch of ntok activation rows through an F32, F16 or BF16
// matrix with each weight row read once per tile of tokens
// (cpu.EmitFloatMatMul), where MatVec per token reads the whole matrix once
// per token. out is [ntok][nrows].

type floatMMKey struct {
	t  quant.Type
	nt int
}

// floatMMFor is the kernel for a tile of nt tokens, emitted on first use and
// kept until Close; nil when this tier has none (a cached refusal).
func (f *JIT) floatMMFor(t quant.Type, nt int) *cpu.Code {
	if f.em == nil || f.em.FloatMatMul == nil {
		return nil
	}
	key := floatMMKey{t, nt}
	f.tiledMu.Lock()
	defer f.tiledMu.Unlock()
	if c, ok := f.floatMM[key]; ok {
		return c
	}
	if f.floatMM == nil {
		f.floatMM = make(map[floatMMKey]*cpu.Code)
	}
	var code *cpu.Code
	if b, err := f.em.FloatMatMul(t, nt); err == nil {
		if c, err := cpu.MapNamed(b, t.String()+"_float_gemm"); err == nil {
			code = c
		}
	}
	f.floatMM[key] = code
	return code
}

// MatMulFloat is out[j*nrows+r] = dot(row r of w, x[j*k:(j+1)*k]) for ntok
// tokens, on the pool over rows, a tile of up to cpu.MaxFloatTokens tokens per
// pass over the weights. It reports false for a type that is not a float
// format or a tier with no kernel, and the caller runs MatVec per token.
func (f *JIT) MatMulFloat(out []float32, t quant.Type, w []byte, x []float32, nrows, k, ntok int) bool {
	if f == nil || ntok < 1 || nrows < 1 || k < 1 {
		return false
	}
	if t != quant.F32 && t != quant.F16 && t != quant.BF16 {
		return false
	}
	es := int(t.BlockBytes())
	if len(w) < nrows*k*es || len(x) < ntok*k || len(out) < ntok*nrows {
		return false
	}
	// Every tile's kernel first, so a refusal leaves out untouched.
	full := f.floatMMFor(t, min(ntok, cpu.MaxFloatTokens))
	if full == nil {
		return false
	}
	var tail *cpu.Code
	if r := ntok % cpu.MaxFloatTokens; ntok > cpu.MaxFloatTokens && r != 0 {
		if tail = f.floatMMFor(t, r); tail == nil {
			return false
		}
	}
	chunk := max(1, nrows/(4*f.pool.N()))
	j := &f.hot.floatMM
	j.w, j.k, j.rowBytes, j.nrows = w, k, k*es, nrows
	for j0 := 0; j0 < ntok; j0 += cpu.MaxFloatTokens {
		nt := min(cpu.MaxFloatTokens, ntok-j0)
		j.code = full
		if nt < min(ntok, cpu.MaxFloatTokens) {
			j.code = tail
		}
		j.x, j.out = x[j0*k:], out[j0*nrows:]
		f.pool.DoRange(nrows, chunk, f.hot.floatMMFn())
	}
	j.code, j.w, j.x, j.out = nil, nil, nil, nil
	f.floatMMCalls.Add(1)
	return true
}

// FloatMatMulCalls is how many MatMulFloat calls ran the GEMM, so a gate can
// assert it was selected.
func (f *JIT) FloatMatMulCalls() int64 {
	if f == nil {
		return 0
	}
	return f.floatMMCalls.Load()
}
