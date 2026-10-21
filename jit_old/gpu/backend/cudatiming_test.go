package backend_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
)

// TestCUDAGraphTimingReportsEveryReplay: with SetCUDATiming on, every replay
// after the first reports the previous one's device time (positive) and,
// from the third on, the device idle before it (non-negative) -- and the
// replays still compute the right thing, because the instrument sits on the
// stream the graph runs on. Off, it reports nothing, which is the selection
// check: the entries below come from the timed path.
func TestCUDAGraphTimingReportsEveryReplay(t *testing.T) {
	gpuLock(t)
	const n, replays = 1 << 16, 6
	for _, d := range backend.Open() {
		defer d.Close()
		if d.API() != "ptx" { // CUDA
			continue
		}
		b := ir.New("k", [3]int{128, 1, 1})
		pIn, pOut := b.Param("pIn", ir.F32), b.Param("pOut", ir.F32)
		i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()), b.Const(ir.U32, n-1))
		x := b.Load(ir.F32, pIn, i, 0)
		b.Store(pOut, i, b.Add(ir.F32, x, x), 0)
		kern, err := d.Compile(b.Done())
		if err != nil {
			t.Fatal(err)
		}
		defer kern.Close()
		bIn, _ := d.Alloc(n * 4)
		defer bIn.Free()
		bOut, _ := d.Alloc(n * 4)
		defer bOut.Free()
		in := make([]byte, n*4)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(in[i*4:], math.Float32bits(float32(i)))
		}
		if err := bIn.Write(in); err != nil {
			t.Fatal(err)
		}
		run := func(timed bool) (busy, gap []float32) {
			backend.SetCUDATiming(timed)
			defer backend.SetCUDATiming(false)
			backend.CUDAGraphTimes()
			var rec backend.Recording
			out := make([]byte, n*4)
			for it := 0; it < replays; it++ {
				var e error
				d.Session(func(s backend.Session) {
					r := s.(backend.Recorder)
					if rec == nil {
						if rec, e = backend.Record(r, func() {
							for j := 0; j < 20; j++ {
								e = s.Launch(kern, n/128, 128, bIn, bOut)
							}
						}); e != nil {
							return
						}
					}
					if e = r.Replay(rec); e == nil {
						e = s.Read(bOut, out)
					}
				})
				if e != nil {
					t.Fatal(e)
				}
				if v := math.Float32frombits(binary.LittleEndian.Uint32(out[(n-1)*4:])); v != 2*(n-1) {
					t.Fatalf("replay %d: got %v, want %v", it, v, float32(2*(n-1)))
				}
			}
			rec.Free()
			return backend.CUDAGraphTimes()
		}
		if busy, _ := run(false); len(busy) != 0 {
			t.Fatalf("timing off reported %d replays", len(busy))
		}
		busy, gap := run(true)
		if len(busy) != replays-1 {
			t.Fatalf("timing on reported %d replays, want %d", len(busy), replays-1)
		}
		for k := range busy {
			// Twenty launches cannot take under 10 us: an after-event recorded
			// on the wrong side of the launch reads ~2 us and fails here.
			if !(busy[k] > 10) || (k > 0 && !(gap[k] >= 0)) {
				t.Fatalf("replay %d: device %v us, gap %v us", k, busy[k], gap[k])
			}
		}
		t.Logf("device %v us, gaps %v us", busy, gap)
		return
	}
	t.Skip("no CUDA device")
}
