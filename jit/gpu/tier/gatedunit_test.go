package tier

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/oldkernels"
)

// TestGatedUnitMatchesTheKernels holds the gated unit composed from a
// container's bounds (rowBuild.gated over jlm.Config.GLU) to the kernels'
// own gated forms bit for bit, for each form a container states: gpt-oss's
// swiglu-oai, the clamped SwiGLU, situ, and plain SwiGLU. A bound moved is
// its violation.
func TestGatedUnitMatchesTheKernels(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const n = 1024
	vals := func(s, scale float64) []byte {
		b := make([]byte, 4*n)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(float32(math.Sin(float64(i)*s)*scale)))
		}
		return b
	}
	gate, up := vals(0.37, 14), vals(0.11, 13) // past every bound
	cases := []struct {
		name string
		cfg  jlm.Config
		k    kernels.ActKind
	}{
		{"swiglu-oai", jlm.Config{Flags: jlm.FlagSwiGLUOAI}, kernels.ActSwiGLUOAI},
		{"swiglu-clamp", jlm.Config{Flags2: jlm.Flag2SwiGLUClamp}, kernels.ActSwiGLUClamp},
		{"situ", jlm.Config{SituBeta: kernels.SituBeta, SituLinearBeta: kernels.SituLinearBeta}, kernels.ActSitu},
		{"swiglu", jlm.Config{}, kernels.ActSiLU},
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			run := func(ker interface {
				Launch(int, int, ...backend.Buf) error
			}, groups, width int) []byte {
				var bufs []backend.Buf
				for _, src := range [][]byte{gate, up, make([]byte, 4*n)} {
					b, err := d.Alloc(len(src))
					if err != nil {
						t.Fatal(err)
					}
					defer b.Free()
					if err := b.Write(src); err != nil {
						t.Fatal(err)
					}
					bufs = append(bufs, b)
				}
				if err := ker.Launch(groups, width, bufs...); err != nil {
					t.Fatal(err)
				}
				out := make([]byte, 4*n)
				if err := bufs[2].Read(out); err != nil {
					t.Fatal(err)
				}
				return out
			}
			unit := func(u jlm.GLU) []byte {
				rb := &rowBuild{r: compose.Row{Width: n, Rows: 1}}
				o := &devOp{Kind: dkActMul, Act: graph.ActSiLU, A: graph.Attrs{GLU: u}}
				y, ok := rb.gated(o, rb.load(compose.BufRow, gaIn0), rb.load(compose.BufRow, gaIn1))
				if !ok {
					t.Fatalf("%+v: not composed", u)
				}
				rb.store(y, gaOut0)
				ker, err := rb.r.Emit(compose.Schedule{Group: 128})
				if err != nil {
					t.Fatal(err)
				}
				c, err := d.Compile(ker)
				if err != nil {
					t.Fatal(err)
				}
				defer c.Close()
				return run(c, 1, 128)
			}
			for _, tc := range cases {
				tk, err := oldkernels.ActMul(n, tc.k)
				if err != nil {
					t.Fatal(err)
				}
				c, err := d.Compile(tk)
				if err != nil {
					t.Fatal(err)
				}
				want := run(c, (n+127)/128, 128)
				c.Close()
				u := tc.cfg.GLU()
				if got := unit(u); !bytes.Equal(got, want) {
					t.Errorf("%s: the unit composed from %+v parts from the kernels' gated form", tc.name, u)
				}
				if !u.Bounded() {
					continue
				}
				bad := u
				bad.UpMax, bad.UpTanh, bad.GateTanh = bad.UpMax+1, bad.UpTanh+1, bad.GateTanh+1
				if got := unit(bad); bytes.Equal(got, want) {
					t.Errorf("%s: a moved bound matched: the gate cannot see the bounds", tc.name)
				}
			}
		})
	}
}
