package composer_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/kernels"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// TestComposedGPUSiblingWidths runs q, k and v of one row, 4096 and twice 1024
// rows of 4096 floats, whose own splits take different workgroups, on every
// GPU device present: they must run as one launch at one width, each row held
// to its dot in float64.
func TestComposedGPUSiblingWidths(t *testing.T) {
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	const K = 4096
	rows := map[jlm.Role]int{jlm.RoleAttnQ: 4096, jlm.RoleAttnK: 1024, jlm.RoleAttnV: 1024}
	rng := rand.New(rand.NewSource(9))
	w := map[jlm.Role][]byte{}
	for r, n := range rows {
		w[r] = tiledWeights(quant.F32, kernels.Float32, n, K, rng)
	}
	x := make([]float32, K)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			b := graph.New()
			in := b.In("x", K)
			outs := map[jlm.Role]graph.ValueID{}
			formats := map[jlm.Role]jlm.Type{}
			for _, r := range []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV} {
				outs[r] = b.MatVec(in, r, rows[r])
				formats[r] = jlm.TypeF32
			}
			b.Out(outs[jlm.RoleAttnQ])
			b.G.Out = append(b.G.Out, outs[jlm.RoleAttnK], outs[jlm.RoleAttnV])
			_, codes, err := composer.Default().WithoutInt8(composer.DeviceGPU).Form(composer.Request{Graph: b.G, Formats: formats,
				Step: composer.Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32}, Device: gpuDevice(d)})
			if err != nil {
				t.Fatal(err)
			}
			if len(codes) != 1 {
				t.Fatalf("q, k and v formed as %d launches, want one", len(codes))
			}
			op, err := compileLone(d, codes)
			if err != nil {
				t.Fatal(err)
			}
			defer op.close()
			var bufs []backend.Buf
			defer func() {
				for _, bf := range bufs {
					bf.Free()
				}
			}()
			up := func(p []byte) backend.Buf {
				bf, err := d.Alloc(max(len(p), 4))
				if err != nil {
					t.Fatal(err)
				}
				if err := bf.Write(p); err != nil {
					t.Fatal(err)
				}
				bufs = append(bufs, bf)
				return bf
			}
			words := func(v []uint32) []byte {
				p := make([]byte, 4*len(v))
				for i, u := range v {
					binary.LittleEndian.PutUint32(p[4*i:], u)
				}
				return p
			}
			xs := make([]uint32, K)
			for i, v := range x {
				xs[i] = math.Float32bits(v)
			}
			outBuf := map[graph.ValueID]backend.Buf{}
			var args []backend.Buf
			for _, bd := range codes[0].Binds {
				o := bd.Operand
				var bf backend.Buf
				switch {
				case o.Kind == composer.OperandValue && o.Value == in:
					bf = up(words(xs))
				case o.Kind == composer.OperandValue:
					bf = up(words(poisoned(int(max(bd.Bytes/4, 1)))))
					outBuf[o.Value] = bf
				case o.Kind == composer.OperandWeight:
					qs, dw, sc, err := kernels.PackWeights(kernels.Float32, w[o.Role], rows[o.Role], K)
					if err != nil {
						t.Fatal(err)
					}
					bf = up(words([][]uint32{qs, dw, sc}[o.Region]))
				default:
					ws := bd.Words
					if ws == nil {
						ws = make([]uint32, int(max(bd.Bytes, 4)/4))
					}
					bf = up(words(ws))
				}
				args = append(args, bf)
			}
			if err := op.kerns[0].Launch(codes[0].LiveGroups(1), codes[0].GPU.Group[0], args...); err != nil {
				t.Fatal(err)
			}
			for r, v := range outs {
				bf := outBuf[v]
				if bf == nil {
					t.Fatalf("%v: no output bound", r)
				}
				p := make([]byte, 4*rows[r])
				if err := bf.Read(p); err != nil {
					t.Fatal(err)
				}
				var num, den float64
				for i := range rows[r] {
					want := 0.0
					for j := range K {
						want += float64(math.Float32frombits(binary.LittleEndian.Uint32(w[r][4*(i*K+j):]))) * float64(x[j])
					}
					got := float64(math.Float32frombits(binary.LittleEndian.Uint32(p[4*i:])))
					num, den = num+(got-want)*(got-want), den+want*want
				}
				if e := num / den; !(e < 1e-10) {
					t.Fatalf("%v: NMSE %.3g against float64", r, e)
				}
			}
			t.Logf("%s: q, k and v one launch of %d workgroups of %d", d.API(), codes[0].LiveGroups(1), codes[0].GPU.Group[0])
		})
	}
}
