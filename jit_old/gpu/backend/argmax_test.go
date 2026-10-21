package backend_test

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestArgmaxTakesTheLowestIndexOfTheMax: the device argmax against a scan in
// the test, over ragged lengths around the group and a vocabulary's, with the
// maximum planted twice (the lower index must win), in the last element, and
// an array where every value ties (index 0).
func TestArgmaxTakesTheLowestIndexOfTheMax(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(7))
	for _, d := range devs {
		defer d.Close()
		for _, n := range []int{1, 7, 1023, 1024, 1025, 32000, 128256} {
			for _, shape := range []string{"random", "tie", "last", "flat"} {
				x := make([]float32, n)
				for i := range x {
					x[i] = float32(rng.NormFloat64())
				}
				switch shape {
				case "tie":
					a, b := rng.Intn(n), rng.Intn(n)
					x[a], x[b] = 99, 99
				case "last":
					x[n-1] = 99
				case "flat":
					for i := range x {
						x[i] = 1.5
					}
				}
				want := 0
				for i, v := range x {
					if v > x[want] {
						want = i
					}
				}
				name := fmt.Sprintf("%s/%d/%s", d.API(), n, shape)
				kk, err := mainkernels.Argmax(n)
				if err != nil {
					t.Fatal(err)
				}
				if err := kk.Validate(); err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				kern, err := d.Compile(kk)
				if err != nil {
					t.Fatalf("%s: compile: %v", name, err)
				}
				g := newGPU(t, d)
				in := g.up(f32bytes(x))
				out := g.up([]byte{0xEF, 0xBE, 0xAD, 0xDE, 0xED, 0x5E, 0xED, 0x5E}) // poison, guard
				if err := kern.Launch(1, mainkernels.ArgmaxGroup, in, out); err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				got := make([]byte, 8)
				if err := out.Read(got); err != nil {
					t.Fatal(err)
				}
				if i := binary.LittleEndian.Uint32(got); int(i) != want {
					t.Errorf("%s: index %d, want %d", name, i, want)
				}
				if binary.LittleEndian.Uint32(got[4:]) != 0x5EED5EED {
					t.Errorf("%s: wrote past the index", name)
				}
				kern.Close()
				g.free()
			}
		}
	}
}

// TestArgmaxRowsTakesEachRowsOwnMax: ArgmaxRows over rows of a vocabulary's
// width, each row with its maximum planted twice at its own positions -- the
// lower of the two must win, and no row may report another row's index.
func TestArgmaxRowsTakesEachRowsOwnMax(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const n, rows = 128256, 5
	rng := rand.New(rand.NewSource(9))
	x := make([]float32, n*rows)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	want := make([]int, rows)
	for r := 0; r < rows; r++ {
		a, b := rng.Intn(n), rng.Intn(n)
		x[r*n+a], x[r*n+b] = 50+float32(r), 50+float32(r)
		want[r] = min(a, b)
	}
	for _, d := range devs {
		defer d.Close()
		kk, err := mainkernels.ArgmaxRows(n, rows)
		if err != nil {
			t.Fatal(err)
		}
		if err := kk.Validate(); err != nil {
			t.Fatal(err)
		}
		kern, err := d.Compile(kk)
		if err != nil {
			t.Fatalf("%s: %v", d.API(), err)
		}
		g := newGPU(t, d)
		in, out := g.up(f32bytes(x)), g.up(make([]byte, 4*rows))
		if err := kern.Launch(rows, mainkernels.ArgmaxGroup, in, out); err != nil {
			t.Fatal(err)
		}
		got := make([]byte, 4*rows)
		if err := out.Read(got); err != nil {
			t.Fatal(err)
		}
		for r := 0; r < rows; r++ {
			if i := int(binary.LittleEndian.Uint32(got[4*r:])); i != want[r] {
				t.Errorf("%s: row %d index %d, want %d", d.API(), r, i, want[r])
			}
		}
		kern.Close()
		g.free()
	}
}
