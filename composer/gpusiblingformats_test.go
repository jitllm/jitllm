package composer_test

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/kernels"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUSiblingsOfTwoFormats runs q, k and v of one row with their
// biases, q and k Q4_K and v Q6_K, whose own splits cut their sums in
// different segments, on every GPU device present: one launch at one
// workgroup width, each projection's rows its launch-apart form's within
// the order of its sum. The violation: v's bias scaled must part.
func TestComposedGPUSiblingsOfTwoFormats(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			for _, sh := range [][3]int{{1536, 1536, 256}, {2048, 2048, 512}, {3584, 3584, 512}} {
				K, Q, KV := sh[0], sh[1], sh[2]
				rng := rand.New(rand.NewSource(int64(K)))
				roles := []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV}
				biases := []jlm.Role{jlm.RoleAttnQBias, jlm.RoleAttnKBias, jlm.RoleAttnVBias}
				rows := []int{Q, KV, KV}
				weights, vecs := map[jlm.Role][3][]uint32{}, map[jlm.Role][]float32{}
				formats := map[jlm.Role]jlm.Type{}
				b := graph.New()
				in := b.In("x", K)
				var outs []graph.ValueID
				for i, r := range roles {
					qt, kt, jt := quant.Q4_K, kernels.Q4_K, jlm.TypeQ4S
					if r == jlm.RoleAttnV {
						qt, kt, jt = quant.Q6_K, kernels.Q6_K, jlm.TypeQ6S
					}
					weights[r], formats[r], formats[biases[i]] = packed(t, qt, kt, rows[i], K, rng), jt, jlm.TypeF32
					bias := make([]float32, rows[i])
					for j := range bias {
						bias[j] = float32(1e4 * rng.NormFloat64())
					}
					vecs[biases[i]] = bias
					outs = append(outs, b.MatVec(in, r, rows[i], biases[i]))
				}
				b.Out(outs...)
				x := make([]float32, K)
				for i := range x {
					x[i] = float32(rng.NormFloat64())
				}
				codes := formWith(t, composer.Default(), b.G, formats, dev)
				apart := formWith(t, without("siblings"), b.G, formats, dev)
				var widths []int
				for _, c := range codes {
					if slices.ContainsFunc(c.Binds, func(bd composer.Binding) bool { return bd.Operand.Kind == composer.OperandWeight }) {
						widths = append(widths, c.GPU.Group[0])
					}
				}
				if len(widths) != 1 {
					t.Fatalf("%d/%d/%d: q, k and v formed as %d launches (widths %v), want one", Q, KV, KV, len(widths), widths)
				}
				name := fmt.Sprintf("%d/%d/%d at %d", Q, KV, KV, widths[0])
				vb := vecs[jlm.RoleAttnVBias]
				for i, o := range outs {
					ref, err := runRoles(d, apart, map[graph.ValueID][]float32{in: x}, weights, vecs, o, rows[i])
					if err != nil {
						t.Fatalf("%s: %v", name, err)
					}
					got, err := runRoles(d, codes, map[graph.ValueID][]float32{in: x}, weights, vecs, o, rows[i])
					if err != nil {
						t.Fatalf("%s: %v", name, err)
					}
					if e := nmseOf(got, ref); !(e < 1e-9) {
						t.Errorf("%s: %v parts from its launch apart: NMSE %.3g", name, roles[i], e)
					}
				}
				scaled := slices.Clone(vb)
				for j := range scaled {
					scaled[j] *= 1.25
				}
				vecs[jlm.RoleAttnVBias] = scaled
				got, err := runRoles(d, codes, map[graph.ValueID][]float32{in: x}, weights, vecs, outs[2], KV)
				vecs[jlm.RoleAttnVBias] = vb
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				ref, err := runRoles(d, apart, map[graph.ValueID][]float32{in: x}, weights, vecs, outs[2], KV)
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				if e := nmseOf(got, ref); !(e > 1e-4) {
					t.Errorf("%s: v's bias scaled reads NMSE %.3g: the gate cannot see v", name, e)
				}
				t.Logf("%s: q, k and v one launch at %d, apart %d launches", d.API(), widths[0], len(apart))
			}
		})
	}
}

// TestComposedGPUHeadNormsInOneLaunch runs q's and k's head norms, 16 and 8
// heads of 128, on every GPU device present: two launches each leaving the
// device idle run as one where it states its wave, each head its
// launch-apart form's. The violation:
// k's norm weight scaled must part.
func TestComposedGPUHeadNormsInOneLaunch(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	const hd = 128
	rng := rand.New(rand.NewSource(17))
	vec := func(n int, mean float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(mean + rng.NormFloat64())
		}
		return v
	}
	q, k := vec(16*hd, 0), vec(8*hd, 0)
	vecs := map[jlm.Role][]float32{jlm.RoleAttnQNorm: vec(hd, 1), jlm.RoleAttnKNorm: vec(hd, 1)}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			b := graph.New()
			b.G.Block = 0
			qi, ki := b.In("q", len(q)), b.In("k", len(k))
			qo, ko := b.HeadNorm(qi, jlm.RoleAttnQNorm, hd, 1e-6), b.HeadNorm(ki, jlm.RoleAttnKNorm, hd, 1e-6)
			b.Out(qo, ko)
			formats := map[jlm.Role]jlm.Type{jlm.RoleAttnQNorm: jlm.TypeF32, jlm.RoleAttnKNorm: jlm.TypeF32}
			codes := formWith(t, composer.Default(), b.G, formats, dev)
			apart := formWith(t, without("siblings"), b.G, formats, dev)
			// A device stating no units or resident threads has no wave to fill.
			want := 1
			if dev.Units == 0 || dev.Slots == 0 {
				want = 2
			}
			if len(codes) != want || len(apart) != 2 {
				t.Fatalf("the head norms formed as %d launches, %d apart: want %d and two", len(codes), len(apart), want)
			}
			ins := map[graph.ValueID][]float32{qi: q, ki: k}
			run := func(codes []composer.Code, out graph.ValueID, n int) []float64 {
				res, err := runRoles(d, codes, ins, nil, vecs, out, n)
				if err != nil {
					t.Fatal(err)
				}
				return res
			}
			for _, c := range []struct {
				out graph.ValueID
				n   int
			}{{qo, len(q)}, {ko, len(k)}} {
				if e := nmseOf(run(codes, c.out, c.n), run(apart, c.out, c.n)); !(e < 1e-12) {
					t.Errorf("value %d parts from its launch apart: NMSE %.3g", c.out, e)
				}
			}
			kw := vecs[jlm.RoleAttnKNorm]
			scaled := slices.Clone(kw)
			for i := range scaled {
				scaled[i] *= 1.25
			}
			vecs[jlm.RoleAttnKNorm] = scaled
			got := run(codes, ko, len(k))
			vecs[jlm.RoleAttnKNorm] = kw
			if e := nmseOf(got, run(apart, ko, len(k))); !(e > 1e-3) {
				t.Errorf("k's norm weight scaled reads NMSE %.3g: the gate cannot see k's share", e)
			}
			t.Logf("%s: q's and k's head norms %d launch(es), %d units, %d slots", d.API(), len(codes), dev.Units, dev.Slots)
		})
	}
}

func without(pass string) *composer.Registry {
	r := composer.Default()
	p := r.Profiles[composer.DeviceGPU]
	p.Off = map[string]bool{pass: true}
	r.Profiles[composer.DeviceGPU] = p
	return r
}

func formWith(t *testing.T, r *composer.Registry, g *graph.Graph, formats map[jlm.Role]jlm.Type, dev composer.Device) []composer.Code {
	t.Helper()
	_, codes, err := r.Form(composer.Request{Graph: g, Formats: formats, Step: composer.Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
	if err != nil {
		t.Fatal(err)
	}
	return codes
}

func packed(t *testing.T, qt quant.Type, kt kernels.Quant, rows, k int, rng *rand.Rand) [3][]uint32 {
	t.Helper()
	qs, dw, sc, err := kernels.PackWeights(kt, tiledWeights(qt, kt, rows, k, rng), rows, k)
	if err != nil {
		t.Fatal(err)
	}
	return [3][]uint32{qs, dw, sc}
}

func nmseOf(a, b []float64) float64 {
	var num, den float64
	for i := range a {
		num, den = num+(a[i]-b[i])*(a[i]-b[i]), den+b[i]*b[i]
	}
	return num / den
}

// runRoles runs codes over the inputs ins with each weight role's planes and
// each vector role's floats, and reads out's n values.
func runRoles(d backend.Device, codes []composer.Code, ins map[graph.ValueID][]float32, weights map[jlm.Role][3][]uint32,
	vecs map[jlm.Role][]float32, out graph.ValueID, n int) ([]float64, error) {
	op, err := compileLone(d, codes)
	if err != nil {
		return nil, err
	}
	defer op.close()
	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	alloc := func(w []uint32) (backend.Buf, error) {
		p := make([]byte, max(4*len(w), 4))
		for i, v := range w {
			binary.LittleEndian.PutUint32(p[4*i:], v)
		}
		b, err := d.Alloc(len(p))
		if err != nil {
			return nil, err
		}
		bufs = append(bufs, b)
		return b, b.Write(p)
	}
	f32 := func(v []float32) []uint32 {
		w := make([]uint32, len(v))
		for i, f := range v {
			w[i] = math.Float32bits(f)
		}
		return w
	}
	own := map[composer.Operand]backend.Buf{}
	var outBuf backend.Buf
	for ci, c := range codes {
		args := make([]backend.Buf, len(c.Binds))
		for i, bd := range c.Binds {
			o := bd.Operand
			if b, ok := own[o]; ok {
				args[i] = b
				continue
			}
			var w []uint32
			switch {
			case o.Kind == composer.OperandValue && ins[o.Value] != nil:
				w = f32(ins[o.Value])
			case o.Kind == composer.OperandVector || o.Kind == composer.OperandWeight && o.Format == jlm.TypeF32:
				v, ok := vecs[o.Role]
				if !ok {
					return nil, fmt.Errorf("%s binds %v, which the gate does not hold", c.Name, o.Role)
				}
				w = f32(v)
			case o.Kind == composer.OperandWeight:
				p, ok := weights[o.Role]
				if !ok {
					return nil, fmt.Errorf("%s binds %v, which the gate does not hold", c.Name, o.Role)
				}
				w = p[o.Region]
			case bd.Words != nil:
				w = bd.Words
			default:
				w = poisoned(int(max(bd.Bytes, 4)/4) + 64)
			}
			b, err := alloc(w)
			if err != nil {
				return nil, err
			}
			own[o], args[i] = b, b
			if o.Kind == composer.OperandValue && o.Value == out {
				outBuf = b
			}
		}
		if err := op.kerns[ci].Launch(c.Groups, c.GPU.Group[0], args...); err != nil {
			return nil, err
		}
	}
	if outBuf == nil {
		return nil, errors.New("no code binds the output")
	}
	p := make([]byte, 4*n)
	if err := outBuf.Read(p); err != nil {
		return nil, err
	}
	res := make([]float64, n)
	for i := range res {
		f := math.Float32frombits(binary.LittleEndian.Uint32(p[4*i:]))
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			return nil, errors.New("a non-finite output")
		}
		res[i] = float64(f)
	}
	return res, nil
}
