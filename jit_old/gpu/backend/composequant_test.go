package backend_test

import (
	"bytes"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/compose"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestComposedQuantizeMatchesMain holds the int8 operand the composer forms
// from nodes (Row.Quantize8: window maxima, correctly rounded scales, round
// half away, packing, sums per 16) to main's Quantize on the same rows, bit
// for bit: codes, scales and sums. Windows of 32 and of a whole row, a zero
// block among them.
func TestComposedQuantizeMatchesMain(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(3))
	for _, d := range devs {
		defer d.Close()
		for _, c := range []struct{ k, window int }{{256, 32}, {1024, 32}, {256, 256}, {96, 32}} {
			x := make([]float32, c.k)
			for i := range x {
				x[i] = float32(rng.NormFloat64() * math.Exp(rng.Float64()*6-3))
			}
			for i := 32; i < 64 && i < c.k; i++ {
				x[i] = 0 // an all-zero block
			}
			up := func(b []byte) backend.Buf {
				buf, err := d.Alloc(max(len(b), 4))
				if err != nil {
					t.Fatal(err)
				}
				if err := buf.Write(b); err != nil {
					t.Fatal(err)
				}
				return buf
			}
			run := func(launch func(a, ax backend.Buf)) ([]byte, []byte) {
				a, ax := up(make([]byte, c.k)), up(make([]byte, 4*(c.k/32+c.k/16)))
				defer a.Free()
				defer ax.Free()
				launch(a, ax)
				ga, gax := make([]byte, c.k), make([]byte, 4*(c.k/32+c.k/16))
				if err := a.Read(ga); err != nil {
					t.Fatal(err)
				}
				if err := ax.Read(gax); err != nil {
					t.Fatal(err)
				}
				return ga, gax
			}
			in := up(f32le(x))
			ma, max_ := run(func(a, ax backend.Buf) {
				mk, err := mainkernels.Quantize(c.k, c.window)
				if err != nil {
					t.Fatal(err)
				}
				k, err := d.Compile(mk)
				if err != nil {
					t.Fatal(err)
				}
				defer k.Close()
				th := mainkernels.QuantizeThreads(c.k / 32)
				if err := k.Launch((th+127)/128, 128, in, a, ax); err != nil {
					t.Fatal(err)
				}
			})
			ca, cax := run(func(a, ax backend.Buf) {
				g := &compose.Row{Width: c.k / 4, Rows: 1}
				g.Quantize8(g.Buf(compose.BufRow), g.Buf(compose.BufOutU32), g.Buf(compose.BufOut), c.window)
				ps, err := g.Plan(compose.Facts{})
				if err != nil {
					t.Fatal(err)
				}
				ck, err := ps[0].G.Emit(ps[0].S)
				if err != nil {
					t.Fatal(err)
				}
				k, err := d.Compile(ck)
				if err != nil {
					t.Fatal(err)
				}
				defer k.Close()
				if err := k.Launch((ps[0].Threads+ps[0].S.Group-1)/ps[0].S.Group, ps[0].S.Group, in, a, ax); err != nil {
					t.Fatal(err)
				}
			})
			in.Free()
			if !bytes.Equal(ma, ca) {
				t.Errorf("%s %+v: the codes part from main's", d.API(), c)
			}
			if !bytes.Equal(max_, cax) {
				t.Errorf("%s %+v: the scales or sums part from main's", d.API(), c)
			}
		}
	}
}
