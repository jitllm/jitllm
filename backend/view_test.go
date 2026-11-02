package backend_test

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend"
)

// TestViewBindsAtItsOffset binds three streams of one stream-major buffer,
// each a view at its offset, as a block of several residual streams binds
// its inputs and outputs: stream 0 becomes stream 1 plus alpha times stream
// 2, through a kernel launch and through a session's, and the streams read
// and the guard after them keep their bytes. A view bound at the buffer's
// start would read stream 0's poison, so the comparison sees it. On Vulkan
// an offset that is not a multiple of the device's
// minStorageBufferOffsetAlignment is refused by name.
func TestViewBindsAtItsOffset(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const n, alpha = 1 << 10, float32(0.5)
	host := make([]float32, 4*n)
	for i := range host {
		switch i / n {
		case 0, 3:
			host[i] = float32(math.NaN())
		case 1:
			host[i] = float32(i%97) - 40
		case 2:
			host[i] = float32(i%31) * 0.25
		}
	}
	k, err := AddScaled(n)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range devs {
		defer d.Close()
		for _, arm := range []string{"kernel", "session"} {
			t.Run(d.API()+"/"+arm, func(t *testing.T) {
				kk, err := d.Compile(k)
				if err != nil {
					t.Fatal(err)
				}
				defer kk.Close()
				g := newGPU(t, d)
				defer g.free()
				buf := g.up(f32bytes(host))
				views := make([]backend.Buf, 3)
				for s := range views {
					if views[s], err = backend.View(buf, s*4*n, 4*n); err != nil {
						t.Fatal(err)
					}
				}
				al := g.up(f32bytes([]float32{alpha}))
				launch := func() error { return kk.Launch(n/128, 128, views[1], views[2], al, views[0]) }
				if arm == "session" {
					launch = func() error {
						var err error
						d.Session(func(s backend.Session) {
							if err = s.Launch(kk, n/128, 128, views[1], views[2], al, views[0]); err == nil {
								err = s.Sync()
							}
						})
						return err
					}
				}
				if err := launch(); err != nil {
					t.Fatal(err)
				}
				for _, v := range views {
					v.Free()
				}
				raw := make([]byte, 16*n)
				if err := buf.Read(raw); err != nil {
					t.Fatal(err)
				}
				for i := range 4 * n {
					got := math.Float32frombits(uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24)
					want := host[i]
					if i < n {
						want = float32(math.FMA(float64(alpha), float64(host[2*n+i]), float64(host[n+i])))
					}
					if math.Float32bits(got) != math.Float32bits(want) {
						t.Fatalf("element %d (stream %d) is %g, want %g", i, i/n, got, want)
					}
				}
				if d.API() != "spirv" {
					return
				}
				if _, err := backend.View(buf, 4, 4*n); err == nil || !strings.Contains(err.Error(), "minStorageBufferOffsetAlignment") {
					t.Fatalf("a view at byte 4: %v, want it refused by its alignment", err)
				}
			})
		}
	}
}
