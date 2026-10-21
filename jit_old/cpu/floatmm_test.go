//go:build amd64 || arm64

package cpu_test

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// floatMMTiers is every tier whose float GEMM runs on this host: the primary
// one, and on amd64 the legacy-SSE tier as well (any x86-64 runs it).
func floatMMTiers() []*cpu.Emitters {
	ts := []*cpu.Emitters{cpu.Native()}
	if runtime.GOARCH == "amd64" && cpu.Native().Tier != cpu.TierSSE {
		ts = append(ts, cpu.EmittersFor(cpu.TierSSE))
	}
	return ts
}

// TestFloatMatMulMatchesTheSums runs the float GEMM at every token tile and at
// k that exercises each of its loops alone and together, against a float64
// sum over the weights exactly as stored. Every token's output row carries a
// guard, and the rows past Rows must not move.
func TestFloatMatMulMatchesTheSums(t *testing.T) {
	if n := floatMMSweep(t, false); n != 0 {
		t.Fatalf("%d mismatches", n)
	}
}

// TestFloatMatMulGateDiscriminates hands the sweep a kernel whose tile is one
// token short of what it is called with; the sweep must see the missing
// token's row as untouched guard.
func TestFloatMatMulGateDiscriminates(t *testing.T) {
	if n := floatMMSweep(t, true); n == 0 {
		t.Fatal("a kernel one token short passed the sweep: the gate does not see a token")
	}
}

func floatMMSweep(t *testing.T, short bool) int {
	t.Helper()
	bad := 0
	for _, em := range floatMMTiers() {
		for _, typ := range []quant.Type{quant.F32, quant.F16, quant.BF16} {
			es := int(typ.BlockBytes())
			for nt := 1; nt <= cpu.MaxFloatTokens; nt++ {
				emitNT := nt
				if short {
					if nt == 1 {
						continue
					}
					emitNT = nt - 1
				}
				code, err := em.FloatMatMul(typ, emitNT)
				if err != nil {
					t.Fatalf("%v %s nt=%d: %v", em.Tier, typ, nt, err)
				}
				c, err := cpu.Map(code)
				if err != nil {
					t.Fatal(err)
				}
				for _, k := range []int{1, 2, 3, 4, 7, 8, 9, 13, 45, 100, 255, 1024, 2047} {
					const rows, guard = 5, 3
					rng := rand.New(rand.NewSource(int64(k*31 + nt)))
					w := make([]byte, rows*k*es)
					wv := make([]float64, rows*k)
					for i := range wv {
						v := float32(rng.NormFloat64())
						switch typ {
						case quant.F32:
							*(*float32)(unsafe.Pointer(&w[4*i])) = v
							wv[i] = float64(v)
						case quant.F16:
							h := quant.EncodeHalf(v)
							*(*uint16)(unsafe.Pointer(&w[2*i])) = h
							wv[i] = quant.DecodeHalf(h)
						default:
							b := uint16(math.Float32bits(v) >> 16)
							*(*uint16)(unsafe.Pointer(&w[2*i])) = b
							wv[i] = float64(math.Float32frombits(uint32(b) << 16))
						}
					}
					x := make([]float32, nt*k)
					for i := range x {
						x[i] = float32(rng.NormFloat64())
					}
					stride := rows + guard
					out := make([]float32, nt*stride)
					for i := range out {
						out[i] = -7
					}
					c.Call(&cpu.Args{Out: &out[0], W: &w[0], A: (*int8)(unsafe.Pointer(&x[0])),
						Rows: rows, K: int64(k), RowStr: int64(k * es), OutStr: int64(4 * stride)})
					for j := 0; j < nt; j++ {
						for r := 0; r < rows; r++ {
							var want float64
							for i := 0; i < k; i++ {
								want += wv[r*k+i] * float64(x[j*k+i])
							}
							got := float64(out[j*stride+r])
							if d := math.Abs(got - want); !(d <= 1e-4*(1+math.Abs(want))*math.Sqrt(float64(k))) {
								bad++
								if !short {
									t.Errorf("%v %s nt=%d k=%d token %d row %d: %v, want %v", em.Tier, typ, nt, k, j, r, got, want)
								}
							}
						}
						for r := rows; r < stride; r++ {
							if out[j*stride+r] != -7 {
								bad++
								if !short {
									t.Errorf("%v %s nt=%d k=%d token %d: wrote row %d past Rows", em.Tier, typ, nt, k, j, r)
								}
							}
						}
					}
				}
				c.Close()
			}
		}
	}
	return bad
}
