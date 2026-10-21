package backend_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestSplitDeltaGatesMatchesTheHostDeinterleave runs the beta/alpha split on
// every device and compares it with engine/model/delta.go's own loop.
//
// Byte equality, because the bug is a permutation: `ssm_ba` reshapes to
// {2*rep, n_head_k}, beta then alpha within each key head's group, and reading
// it as two contiguous halves pairs every value head with the wrong decay.
//
// The second shape is ragged on purpose: 6 value heads is not a whole number of
// 128-thread workgroups, so the surplus threads clamp onto the last work item
// and must write what it would have written.
func TestSplitDeltaGatesMatchesTheHostDeinterleave(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
	}
	// Qwen3-Next-80B is 16 key heads of 2, and a ragged one beside it.
	for _, c := range []struct{ kHeads, rep int }{{16, 2}, {3, 2}} {
		vHeads := c.kHeads * c.rep
		ba := make([]float32, 2*vHeads)
		for i := range ba {
			ba[i] = float32(i%53) - 26
		}
		// engine/model/delta.go's loop, transcribed rather than re-derived.
		wantB := make([]float32, vHeads)
		wantA := make([]float32, vHeads)
		for kh := 0; kh < c.kHeads; kh++ {
			base := kh * 2 * c.rep
			for r := 0; r < c.rep; r++ {
				vh := kh*c.rep + r
				wantB[vh] = ba[base+r]
				wantA[vh] = ba[base+c.rep+r]
			}
		}
		k, err := mainkernels.SplitDeltaGatesRows(c.kHeads, c.rep, 1)
		if err != nil {
			t.Fatalf("SplitDeltaGates: %v", err)
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
			for _, n := range []int{len(ba) * 4, vHeads * 4, vHeads * 4} {
				b, err := d.Alloc(n)
				if err != nil {
					t.Fatalf("alloc: %v", err)
				}
				defer b.Free()
				bufs = append(bufs, b)
			}
			if err := bufs[0].Write(f32le(ba)); err != nil {
				t.Fatalf("write: %v", err)
			}
			if err := kern.Launch((vHeads+127)/128, 128, bufs[0], bufs[1], bufs[2]); err != nil {
				t.Fatalf("launch: %v", err)
			}
			kern.Close()
			for bi, want := range [][]float32{wantB, wantA} {
				raw := make([]byte, len(want)*4)
				if err := bufs[bi+1].Read(raw); err != nil {
					t.Fatalf("read: %v", err)
				}
				for i, w := range want {
					got := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
					if got != w {
						name := []string{"beta", "alpha"}[bi]
						t.Fatalf("%s kHeads=%d rep=%d %s[%d] = %v, want %v -- the "+
							"beta/alpha deinterleave does not match the host's",
							d.API(), c.kHeads, c.rep, name, i, got, w)
					}
				}
			}
		}
	}
}
