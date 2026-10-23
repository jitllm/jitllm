package backend_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestHeadNorm holds kernels.HeadNorm to a float64 reference on real hardware.
//
// The shapes are the point: (16, 128) is qwen3's per-head q/k norm and
// (1, 2048) is olmoe's whole-projection norm over the same buffer. They
// normalise by different denominators, so a kernel that ignored nHeads would
// pass one and fail the other.
//
// On a device with a subgroup range (Intel: 8..32) the 32-lane kernel is right
// only when its width is pinned (VK_EXT_subgroup_size_control); on a device
// that will not promise it, the 32-lane kernel must not be created at all,
// which this test asserts directly.
func TestHeadNorm(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	cases := []struct{ nHeads, headDim int }{
		{16, 128}, // qwen3: one norm per head
		{1, 2048}, // olmoe: one norm over the whole projection
		{8, 64},
		{1, 32}, // the narrowest legal width: one lane-group, one pass
		// gemma3's shape: head_dim 256 is eight elements per lane at 32
		// lanes where qwen3 is four, a different number of reduction passes.
		{8, 256},
		{4, 256},
	}
	const eps = 1e-6
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			wide, why := backend.GuaranteedLanes(d, 32)
			t.Logf("%s: 32-lane subgroup guaranteed=%v %s", d.Name(), wide, why)

			// The refusal is asserted: a device that will not promise 32
			// lanes must fail to compile the 32-lane kernel, so the scalar
			// twin is chosen by construction.
			k32, err := kernels.HeadNorm(16, 128, eps, 32)
			if err != nil {
				t.Fatal(err)
			}
			switch c, err := d.Compile(k32); {
			case wide && err != nil:
				t.Fatalf("%s guarantees 32 lanes and refused the 32-lane kernel: %v", d.Name(), err)
			case !wide && err == nil:
				c.Close()
				t.Fatalf("%s promises nothing about subgroup width (%s) and COMPILED the "+
					"32-lane kernel; selection would then depend on a probe again", d.Name(), why)
			case err == nil:
				c.Close()
			default:
				t.Logf("%s refused the 32-lane kernel, as it must: %v", d.Name(), err)
			}

			for _, c := range cases {
				// The width the engine would pick on this device, and the
				// scalar twin, which every device must run.
				lanes := []int{1}
				if wide {
					lanes = append([]int{32}, lanes...)
				}
				for _, l := range lanes {
					t.Run(name(c.nHeads, c.headDim)+"/lanes"+itoa32(l), func(t *testing.T) {
						headNormCase(t, d, c.nHeads, c.headDim, eps, l)
					})
				}
			}
		})
	}
}

func name(nHeads, headDim int) string {
	return itoa32(nHeads) + "x" + itoa32(headDim)
}

func itoa32(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

func headNormCase(t *testing.T, d backend.Device, nHeads, headDim int, eps float64, lanes int) {
	n := nHeads * headDim
	rng := rand.New(rand.NewSource(int64(n) * 7919))
	in := make([]float32, n)
	w := make([]float32, headDim)
	for i := range in {
		in[i] = float32(rng.NormFloat64())
	}
	for i := range w {
		// Not all ones: a weight of 1 hides a kernel that drops it entirely.
		w[i] = float32(0.5 + rng.Float64())
	}

	g := newGPU(t, d)
	defer g.free()
	bIn, bW, bOut := g.up(f32bytes(in)), g.up(f32bytes(w)), g.up(make([]byte, n*4))

	kk, err := kernels.HeadNorm(nHeads, headDim, eps, lanes)
	if err != nil {
		t.Fatal(err)
	}
	// lanes=1 must declare nothing (it has no cross-lane op, so it may run
	// anywhere); lanes=32 must declare 32, or it would be offered to a device
	// that promised nothing.
	wantDecl := 0
	if lanes == 32 {
		wantDecl = 32
	}
	if kk.Lanes != wantDecl {
		t.Fatalf("HeadNorm(lanes=%d) declares Lanes=%d, want %d", lanes, kk.Lanes, wantDecl)
	}
	kern, err := d.Compile(kk)
	if err != nil {
		t.Fatal(err)
	}
	defer kern.Close()
	// One group per head at 32 lanes; one thread per head at 1, so the same
	// heads need ceil(n/64) groups of 64.
	groups, width := nHeads, 32
	if lanes == 1 {
		groups, width = (nHeads+63)/64, 64
	}
	if err := kern.Launch(groups, width, bIn, bW, bOut); err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, n*4)
	if err := bOut.Read(raw); err != nil {
		t.Fatal(err)
	}

	// Reference: RMSNorm over each group of headDim, in float64.
	for h := 0; h < nHeads; h++ {
		var ss float64
		for i := 0; i < headDim; i++ {
			v := float64(in[h*headDim+i])
			ss += v * v
		}
		inv := 1 / math.Sqrt(ss/float64(headDim)+eps)
		for i := 0; i < headDim; i++ {
			want := float64(in[h*headDim+i]) * inv * float64(w[i])
			got := float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[(h*headDim+i)*4:])))
			if math.Abs(got-want) > 1e-5*(1+math.Abs(want)) {
				t.Fatalf("head %d element %d: got %v want %v", h, i, got, want)
			}
		}
	}
}
