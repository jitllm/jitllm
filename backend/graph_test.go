package backend_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestRecordReplayMatchesDirect holds a replayed graph to the same bar as every
// other generated thing in jitllm: it must produce what the path it replaces
// produces, bit for bit.
//
// It tests the ordering as much as the arithmetic: the capture stream is a
// blocking stream, so the legacy-stream copies around a replay stay ordered
// with it. If not, an iteration reads the previous one's answer, so it runs
// several iterations with different inputs.
func TestRecordReplayMatchesDirect(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const n = 1024
	ran := 0
	for _, d := range devs {
		defer d.Close()
		// Asked once, before the subtest exists: only CUDA implements
		// backend.Recorder, by design, so other backends get no subtest.
		records := false
		d.Session(func(s backend.Session) { _, records = s.(backend.Recorder) })
		if !records {
			continue
		}
		ran++
		t.Run(d.API(), func(t *testing.T) {
			// scale multiplies by a factor read from a buffer, so the answer
			// depends on a value the host writes between replays -- which is
			// exactly the shape of jitllm's position and KV offset, and the only
			// way a graph can be told anything new.
			b := ir.New("k", [3]int{128, 1, 1})
			pIn := b.Param("pIn", ir.F32)
			pF := b.Param("pF", ir.F32)
			pOut := b.Param("pOut", ir.F32)
			i := b.Min(ir.U32, b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID()),
				b.Const(ir.U32, n-1))
			f := b.Load(ir.F32, pF, b.Const(ir.U32, 0), 0)
			b.Store(pOut, i, b.Mul(ir.F32, b.Load(ir.F32, pIn, i, 0), f), 0)
			kern, err := d.Compile(mustBuild(t, b))
			if err != nil {
				t.Skipf("compile: %v", err)
			}
			defer kern.Close()

			bIn, err := d.Alloc(n * 4)
			if err != nil {
				t.Fatal(err)
			}
			defer bIn.Free()
			bF, err := d.Alloc(4)
			if err != nil {
				t.Fatal(err)
			}
			defer bF.Free()
			bOut, err := d.Alloc(n * 4)
			if err != nil {
				t.Fatal(err)
			}
			defer bOut.Free()

			in := make([]byte, n*4)
			for i := 0; i < n; i++ {
				binary.LittleEndian.PutUint32(in[i*4:], math.Float32bits(float32(i)+0.5))
			}
			if err := bIn.Write(in); err != nil {
				t.Fatal(err)
			}
			out := make([]byte, n*4)
			got := func(i int) float32 {
				return math.Float32frombits(binary.LittleEndian.Uint32(out[i*4:]))
			}

			var rec backend.Recording
			defer func() {
				if rec != nil {
					rec.Free()
				}
			}()
			// Iteration 0 runs the launch directly, 1 records and replays, and
			// 2 and 3 replay what 1 recorded -- with a different factor each
			// time, so a replay that raced its readback reads the wrong one.
			for it := 0; it < 4; it++ {
				factor := float32(it) + 1
				fb := make([]byte, 4)
				binary.LittleEndian.PutUint32(fb, math.Float32bits(factor))
				var e error
				d.Session(func(s backend.Session) {
					if e = s.Write(bF, fb); e != nil {
						return
					}
					r, ok := s.(backend.Recorder)
					if !ok || it == 0 {
						if e = s.Launch(kern, n/128, 128, bIn, bF, bOut); e != nil {
							return
						}
					} else {
						if rec == nil {
							if rec, e = backend.Record(r, func() {
								e = s.Launch(kern, n/128, 128, bIn, bF, bOut)
							}); e != nil {
								return
							}
						}
						if e = r.Replay(rec); e != nil {
							return
						}
					}
					e = s.Read(bOut, out)
				})
				if e != nil {
					t.Fatalf("iteration %d: %v", it, e)
				}
				if it > 0 && rec == nil {
					t.Fatal("this device answered backend.Recorder and then recorded " +
						"nothing; the capability check above and Record disagree")
				}
				for i := 0; i < n; i++ {
					want := (float32(i) + 0.5) * factor
					if got(i) != want {
						t.Fatalf("iteration %d, element %d: got %v, want %v", it, i, got(i), want)
					}
				}
			}
		})
	}
	if ran == 0 {
		t.Skip("no backend on this host records launches (only CUDA does)")
	}
}
