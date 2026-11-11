//go:build linux

package composer

import (
	"math"
	"syscall"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUFramesReadNothingBeforeTheClip: OpFrames over a clip that
// starts a page whose page before is unreadable reads no sample before the
// clip's first -- the pre-emphasis of the first sample is zero -- and frames it
// as TestComposedCPUFrames does. A read of the word before the clip faults
// the test binary.
func TestComposedCPUFramesReadNothingBeforeTheClip(t *testing.T) {
	const n, N, hop, pre = 3000, 512, 160, 0.97
	const R = n/hop + 1
	pg := syscall.Getpagesize()
	mem, err := syscall.Mmap(-1, 0, pg+(4*n+pg-1)/pg*pg, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(mem)
	if err := syscall.Mprotect(mem[:pg], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}
	clip := unsafe.Slice((*float32)(unsafe.Pointer(&mem[pg])), n)
	for i := range clip {
		clip[i] = float32(math.Sin(0.01*float64(i)) + 0.5)
	}
	b := graph.NewBlock(-1)
	b.Out(b.Emit(graph.OpFrames, nil, []graph.Result{{Name: "frames", Width: N}}, nil,
		graph.Attrs{Width: N, Stride: hop, Const: pre}, nil, nil)[0])
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	_, codes, err := Default().WithoutInt8(DeviceCPU).Form(Request{Graph: b.G, Step: Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32},
		Device: dev, Apart: true})
	if err != nil {
		t.Fatal(err)
	}
	req := make([]uint32, RequestPicW+1)
	req[RequestPicW] = n
	pos := make([]uint32, R+1)
	pos[0] = R
	got := make([]float32, R*N)
	_, err = runCPUBound(hostArm(), codes, nil, R*N, [3][]uint32{}, nil, func(o Operand) []uint32 {
		if o.Kind == OperandValue && o.Value == 0 {
			return f32Words(got)
		}
		switch o.Name {
		case "request":
			return req
		case "rope.pos":
			return pos
		}
		if o.Name == TablePicture.op().Name {
			return f32Words(clip)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for tt := range R {
		for i := range N {
			k := tt*hop + i - N/2
			want := 0.0
			if k >= 0 && k < n {
				want = float64(clip[k])
				if k > 0 {
					want -= pre * float64(clip[k-1])
				}
			}
			if d := math.Abs(float64(got[tt*N+i]) - want); math.IsNaN(d) || d > 1e-6 {
				t.Fatalf("frame %d sample %d: %g, want %g", tt, i, got[tt*N+i], want)
			}
		}
	}
	t.Logf("%d frames of %d from a clip that starts a page after an unreadable one", R, N)
}
