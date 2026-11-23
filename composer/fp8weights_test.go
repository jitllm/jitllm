package composer

import (
	bin "encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
)

// fp8Bank is sheets sheets of rows rows of k random weights of NVFP4
// (jlm.TypeNV4) or MXFP8 (jlm.TypeMX8) as the container lays them out, its
// words, and the oracle's dequantized values. Scales sit near one and each
// sheet's global differs, so a kernel that reads one sheet's global for
// another's rows parts from the oracle.
func fp8Bank(t *testing.T, ty jlm.Type, sheets, rows, k int, rng *rand.Rand) ([]uint32, []float32) {
	t.Helper()
	want := make([]float32, sheets*rows*k)
	var raw []byte
	var err error
	if ty == jlm.TypeNV4 {
		codes := make([]byte, sheets*rows*k/2)
		for i := range codes {
			codes[i] = byte(rng.Uint32())
		}
		scales := make([]byte, sheets*rows*k/16)
		for i := range scales {
			scales[i] = byte(0x28 + rng.Intn(0x28))
		}
		global := make([]float32, sheets)
		for s := range global {
			global[s] = float32(0.1+rng.Float64()) / 2688 * 448
			oracle.DequantNVFP4(want[s*rows*k:], codes[s*rows*k/2:], scales[s*rows*k/16:], global[s], rows, k)
		}
		raw, err = jlm.NV4Rows(codes, scales, global, sheets, rows, k)
	} else {
		vals := make([]byte, sheets*rows*k)
		for i := range vals {
			vals[i] = byte(rng.Uint32())
			if vals[i]&0x7F == 0x7F {
				vals[i] = 0x38
			}
		}
		scales := make([]byte, sheets*rows*k/32)
		for i := range scales {
			scales[i] = byte(121 + rng.Intn(6))
		}
		oracle.DequantMXFP8(want, vals, scales, sheets*rows, k)
		raw, err = jlm.MX8Rows(vals, scales, sheets*rows, k)
	}
	if err != nil {
		t.Fatal(err)
	}
	words := make([]uint32, (len(raw)+3)/4)
	for i := range words {
		var b [4]byte
		copy(b[:], raw[4*i:])
		words[i] = bin.LittleEndian.Uint32(b[:])
	}
	return words, want
}

// fp8Ref is each of rows output rows' dot of the dequantized weights w
// (rows by k) with x, summed in f64.
func fp8Ref(w, x []float32, rows, k int) []float32 {
	out := make([]float32, rows)
	for r := range rows {
		var s float64
		for c := range k {
			s += float64(w[r*k+c]) * float64(x[c])
		}
		out[r] = float32(s)
	}
	return out
}

// TestComposedCPUFP8Weights runs a matvec and an expert matvec over NVFP4
// and MXFP8 weights on the CPU device, the format's basic decoded kernel,
// at ragged widths (one NVFP4 block, rows not a multiple of any vector) and
// a checkpoint's 1152, on one worker and fourteen, held to the oracle. The
// violation: one sheet's scales changed must move the picked rows.
func TestComposedCPUFP8Weights(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	ran := 0
	for _, ty := range []jlm.Type{jlm.TypeNV4, jlm.TypeMX8} {
		for _, sh := range [][2]int{{16, 5}, {48, 7}, {96, 33}, {1152, 9}} {
			K, rows := sh[0], sh[1]
			if ty == jlm.TypeMX8 && K%32 != 0 {
				continue
			}
			for _, slots := range []int{1, 14} {
				name := fmt.Sprintf("%v/%dx%d/%d slots", ty, rows, K, slots)
				dev := cpuDevice(hostArm())
				dev.Slots = slots

				words, w := fp8Bank(t, ty, 1, rows, K, rng)
				b := graph.New()
				x := b.In("x", K)
				y := b.MatVec(x, jlm.RoleAttnQ, rows)
				b.Out(y)
				_, codes, err := Default().WithoutInt8(DeviceCPU).Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: ty},
					Step: Step{Rows: 1, Page: 32, Keys: 64, ActWindow: 32, Seqs: 1}, Device: dev})
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				xs := make([]float32, K)
				for i := range xs {
					xs[i] = float32(rng.NormFloat64())
				}
				vals := map[graph.ValueID][]uint32{x: unsafe.Slice((*uint32)(unsafe.Pointer(&xs[0])), K)}
				if err := runCPUValues(hostArm(), codes, b.G, vals, [3][]uint32{words, {0}, {0}}); err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				got := unsafe.Slice((*float32)(unsafe.Pointer(&vals[y][0])), rows)
				if e := cpuNMSE(got, fp8Ref(w, xs, rows, K)); math.IsNaN(e) || !(e < 1e-9) {
					t.Errorf("%s: matvec %.3e from the oracle", name, e)
				}

				const E, P = 3, 2
				bank, bw := fp8Bank(t, ty, E, rows, K, rng)
				b = graph.New()
				x = b.In("x", P*K)
				sel := b.In("sel", P)
				b.G.Values[sel].Elem = graph.ElemU32
				y = b.Emit(graph.OpExpertMatVec, []graph.ValueID{x, sel}, []graph.Result{{Name: "y", Width: P * rows}},
					[]graph.Binding{{Role: jlm.RoleExpGateBank, K: K, Rows: rows, Sheets: E}}, graph.Attrs{}, nil, nil)[0]
				b.Out(y)
				_, codes, err = Default().WithoutInt8(DeviceCPU).Form(Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleExpGateBank: ty},
					Step: Step{Rows: 1, Page: 32, Keys: 64, ActWindow: 32, Seqs: 1}, Device: dev})
				if err != nil {
					t.Fatalf("%s: experts: %v", name, err)
				}
				xs = make([]float32, P*K)
				for i := range xs {
					xs[i] = float32(rng.NormFloat64())
				}
				picks := []uint32{2, 1}
				run := func(bank []uint32) []float32 {
					vals := map[graph.ValueID][]uint32{x: unsafe.Slice((*uint32)(unsafe.Pointer(&xs[0])), P*K), sel: picks}
					if err := runCPUValues(hostArm(), codes, b.G, vals, [3][]uint32{bank, {0}, {0}}); err != nil {
						t.Fatalf("%s: experts: %v", name, err)
					}
					return append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(&vals[y][0])), P*rows)...)
				}
				want := make([]float32, 0, P*rows)
				for j, e := range picks {
					want = append(want, fp8Ref(bw[int(e)*rows*K:], xs[j*K:], rows, K)...)
				}
				if e := cpuNMSE(run(bank), want); math.IsNaN(e) || !(e < 1e-9) {
					t.Errorf("%s: experts %.3e from the oracle", name, e)
				}
				mut := append([]uint32(nil), bank...)
				per := len(bank) / E
				for i := 2 * per; i < 3*per; i++ {
					mut[i] ^= 0x01010101
				}
				if e := cpuNMSE(run(mut), want); !(e > 1e-6) {
					t.Errorf("%s: sheet 2's bytes changed reads %.3e: the gate cannot see the sheet", name, e)
				}
				ran++
			}
		}
	}
	if ran != 12 {
		t.Fatalf("%d arms ran, want 12", ran)
	}
}

// FP8Bank and FP8Ref are fp8Bank and fp8Ref for the device gates.
func FP8Bank(t *testing.T, ty jlm.Type, sheets, rows, k int, rng *rand.Rand) ([]uint32, []float32) {
	return fp8Bank(t, ty, sheets, rows, k, rng)
}

func FP8Ref(w, x []float32, rows, k int) []float32 { return fp8Ref(w, x, rows, k) }
