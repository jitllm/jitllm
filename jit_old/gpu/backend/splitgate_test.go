package backend_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestSplitHeadGateMatchesTheHostDeinterleave runs the q/gate split on every
// device and compares it with the host's own loop.
//
// The bug it catches is a permutation, so the bound is equality: splitQGate
// takes the first headDim of each head as q and the second as its gate, and
// reading the row as two halves instead pairs every head with the wrong gate.
func TestSplitHeadGateMatchesTheHostDeinterleave(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	// Deferred first, so it runs last: the buffers deferred below are freed
	// while their device is still open ("vkFreeMemory: Invalid device"
	// otherwise).
	for _, d := range devs {
		defer d.Close()
	}
	// qwen3next-80B's shape, and a ragged one whose grid is not a whole number
	// of workgroups so the clamp on the last work item is exercised.
	for _, c := range []struct{ nHead, headDim, rows int }{
		{16, 256, 1},
		{4, 8, 3},
	} {
		qdim := c.nHead * c.headDim
		src := make([]float32, c.rows*2*qdim)
		for i := range src {
			src[i] = float32(i%97) - 48
		}
		wantQ := make([]float32, c.rows*qdim)
		wantG := make([]float32, c.rows*qdim)
		for r := 0; r < c.rows; r++ {
			in := src[r*2*qdim:]
			for hh := 0; hh < c.nHead; hh++ {
				copy(wantQ[r*qdim+hh*c.headDim:][:c.headDim], in[hh*2*c.headDim:][:c.headDim])
				copy(wantG[r*qdim+hh*c.headDim:][:c.headDim], in[hh*2*c.headDim+c.headDim:][:c.headDim])
			}
		}
		k, err := mainkernels.SplitHeadGate(c.nHead, c.headDim, c.rows)
		if err != nil {
			t.Fatalf("SplitHeadGate: %v", err)
		}
		if err := k.Validate(); err != nil {
			t.Fatalf("validate: %v", err)
		}
		for _, d := range devs {
			kern, err := d.Compile(k)
			if err != nil {
				t.Fatalf("%s: compile: %v", d.API(), err)
			}
			bufs := make([]backend.Buf, 0, 3)
			for _, n := range []int{len(src) * 4, len(wantQ) * 4, len(wantG) * 4} {
				b, err := d.Alloc(n)
				if err != nil {
					t.Fatalf("alloc: %v", err)
				}
				defer b.Free()
				bufs = append(bufs, b)
			}
			if err := bufs[0].Write(f32le(src)); err != nil {
				t.Fatalf("write: %v", err)
			}
			if err := kern.Launch((c.rows*qdim+127)/128, 128, bufs[0], bufs[1], bufs[2]); err != nil {
				t.Fatalf("launch: %v", err)
			}
			kern.Close()
			for bi, want := range [][]float32{wantQ, wantG} {
				raw := make([]byte, len(want)*4)
				if err := bufs[bi+1].Read(raw); err != nil {
					t.Fatalf("read: %v", err)
				}
				for i, w := range want {
					got := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
					if got != w {
						name := []string{"q", "gate"}[bi]
						t.Fatalf("%s %dx%dx%d %s[%d] = %v, want %v",
							d.API(), c.nHead, c.headDim, c.rows, name, i, got, w)
					}
				}
			}
		}
	}
}

func f32le(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	return b
}
