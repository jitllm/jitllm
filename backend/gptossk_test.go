package backend_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

func readF32(t *testing.T, b backend.Buf, n int) []float64 {
	t.Helper()
	raw := make([]byte, n*4)
	if err := b.Read(raw); err != nil {
		t.Fatal(err)
	}
	out := make([]float64, n)
	for i := range out {
		out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])))
	}
	return out
}

// TestSoftmaxSinkMatchesTheHost runs the sink softmax -- gpt-oss's learned
// logit per head, in the maximum and the denominator and not in the output --
// against float64, on every backend, for the thread-per-head kernel and (where
// the device guarantees 32 lanes) the warp one, one query row and three.
//
// The sinks sit inside the scores' range, so they move both the maximum and
// the denominator, and one head's sink is far above every score, where the row
// must sum to nearly zero. A kernel that dropped the sink sums every row to 1;
// one that forgot it in the maximum overflows on the big sink.
func TestSoftmaxSinkMatchesTheHost(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const nh, maxSeq = 3, 64
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := newGPU(t, d)
			defer g.free()
			ran := 0
			for _, lanes := range []int{1, 32} {
				if lanes == 32 {
					if ok, _ := backend.GuaranteedLanes(d, ir.SubgroupLanes); !ok {
						continue
					}
				}
				for _, rows := range []int{1, 3} {
					if rows > 1 && lanes != 32 {
						continue // the scalar batched kernel takes one width, pN[0]
					}
					heads := nh * rows
					scores := make([]float32, heads*maxSeq)
					for i := range scores {
						scores[i] = float32(4 * math.Sin(0.37*float64(i)))
					}
					sinks := []float32{1.5, -2, 40} // the last swamps its row
					counts := []uint32{37}
					for r := 0; r < rows; r++ {
						counts = append(counts, uint32(20+7*r))
					}
					bA, bN := g.up(f32bytes(scores)), g.up(u32bytes(counts))
					bOut, bS := g.up(make([]byte, len(scores)*4)), g.up(f32bytes(sinks))
					kk, err := kernels.SoftmaxRowsSink(nh, maxSeq, lanes, rows, 1, true)
					if err != nil {
						t.Fatal(err)
					}
					k, err := d.Compile(kk)
					if err != nil {
						t.Fatalf("lanes=%d rows=%d: %v", lanes, rows, err)
					}
					groups, width := heads, 32
					if lanes == 1 {
						groups, width = (heads+63)/64, 64
					}
					if err := k.Launch(groups, width, bA, bN, bOut, bS); err != nil {
						t.Fatal(err)
					}
					k.Close()
					got := readF32(t, bOut, len(scores))
					for h := 0; h < heads; h++ {
						n := int(counts[0])
						if rows > 1 {
							n = int(counts[1+h/nh])
						}
						s := float64(sinks[h%nh])
						mx := s
						for i := 0; i < n; i++ {
							mx = math.Max(mx, float64(scores[h*maxSeq+i]))
						}
						sum := math.Exp(s - mx)
						for i := 0; i < n; i++ {
							sum += math.Exp(float64(scores[h*maxSeq+i]) - mx)
						}
						for i := 0; i < n; i++ {
							want := math.Exp(float64(scores[h*maxSeq+i])-mx) / sum
							if d := math.Abs(got[h*maxSeq+i] - want); d > 1e-5 {
								t.Fatalf("lanes=%d rows=%d head %d pos %d: %v, want %v",
									lanes, rows, h, i, got[h*maxSeq+i], want)
							}
						}
					}
					ran++
				}
			}
			if ran == 0 {
				t.Fatal("no softmax shape ran")
			}
		})
	}
}

// TestIndexedBiasAddMatchesTheHost adds each slot's expert bias, indexed by the
// true expert id, on every backend.
func TestIndexedBiasAddMatchesTheHost(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const k, rows, nExpert = 3, 70, 8
	in := make([]float32, k*rows)
	bias := make([]float32, nExpert*rows)
	for i := range in {
		in[i] = float32(i%17) - 8
	}
	for i := range bias {
		bias[i] = float32(1000 + i)
	}
	sel := []uint32{5, 0, 6} // not slot order: slot j reads expert sel[j]
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := newGPU(t, d)
			defer g.free()
			bIn, bSel, bB := g.up(f32bytes(in)), g.up(u32bytes(sel)), g.up(f32bytes(bias))
			bOut := g.up(make([]byte, len(in)*4))
			kk, err := kernels.IndexedBiasAdd(k, rows)
			if err != nil {
				t.Fatal(err)
			}
			kern, err := d.Compile(kk)
			if err != nil {
				t.Fatal(err)
			}
			defer kern.Close()
			if err := kern.Launch((k*rows+127)/128, 128, bIn, bSel, bB, bOut); err != nil {
				t.Fatal(err)
			}
			got := readF32(t, bOut, len(in))
			for j := 0; j < k; j++ {
				for r := 0; r < rows; r++ {
					want := float64(in[j*rows+r] + bias[int(sel[j])*rows+r])
					if got[j*rows+r] != want {
						t.Fatalf("slot %d row %d: %v, want %v", j, r, got[j*rows+r], want)
					}
				}
			}
		})
	}
}
