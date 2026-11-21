package backend_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
)

// TestMMAVoltaProbeMatchesTheHost runs sm_70's m8n8k4 through the IR on every
// backend that lowers it (PTX; the others refuse fragment MMA by name) and
// holds each quad-pair's 8x8 product to a CPU one. Small integers, so the f16
// operands and f32 sums are exact and the compare is equality.
func TestMMAVoltaProbeMatchesTheHost(t *testing.T) {
	gpuLock(t)
	devs := backend.OpenForTest()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(3))
	A, B := make([]float32, 32*4), make([]float32, 4*32)
	for i := range A {
		A[i] = float32(rng.Intn(33) - 16)
	}
	for i := range B {
		B[i] = float32(rng.Intn(33) - 16)
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		pk, perr := MMAVoltaProbe()
		if perr != nil {
			t.Fatal(perr)
		}
		kern, err := d.Compile(pk)
		if err != nil {
			t.Logf("%s: %v", d.API(), err)
			continue
		}
		ran++
		g := newGPU(t, d)
		bA, bB, out := g.up(f32bytes(A)), g.up(f32bytes(B)), g.up(f32bytes(nanFill(32*8)))
		if err := kern.Launch(1, 32, bA, bB, out); err != nil {
			t.Fatal(err)
		}
		raw := make([]byte, 32*8*4)
		if err := out.Read(raw); err != nil {
			t.Fatal(err)
		}
		for q := 0; q < 4; q++ {
			for r := 0; r < 8; r++ {
				for c := 0; c < 8; c++ {
					var want float32
					for k := 0; k < 4; k++ {
						want += A[(8*q+r)*4+k] * B[k*32+8*q+c]
					}
					got := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*((8*q+r)*8+c):]))
					if got != want {
						t.Fatalf("%s: quad %d row %d col %d: %v, want %v", d.API(), q, r, c, got, want)
					}
				}
			}
		}
		kern.Close()
		g.free()
	}
	if ran == 0 {
		t.Skip("no backend lowers the m8n8k4 shape here")
	}
}
