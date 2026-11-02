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
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUPooledProjection forms a picture projector as Gemma 3's
// states it -- each merged row's S*S slices summed, scaled by 1/(S*S), its
// RMSNorm, then a Q8_0 projection -- at a picture's merged rows on every
// GPU present, with the device's matrix unit stated where its backend
// lowers a tile and without, and holds every row to the definition in
// float64 at calls of all its rows and of fewer. The violation: the norm's
// weight scaled must fail.
func TestComposedGPUPooledProjection(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	const E, S, N, R = 1152, 4, 2560, 256
	b := graph.NewBlock(-1)
	in := b.In("merged", E*S*S)
	slice := func(s int) graph.ValueID {
		return b.Emit(graph.OpSlice, []graph.ValueID{in}, []graph.Result{{Name: "slot", Width: E}}, nil,
			graph.Attrs{Offset: s * E, Width: E}, nil, nil)[0]
	}
	acc := slice(0)
	for s := 1; s < S*S; s++ {
		acc = b.Add(acc, slice(s))
	}
	acc = b.Emit(graph.OpMap, []graph.ValueID{acc}, []graph.Result{{Name: "pooled", Width: E}}, nil,
		graph.Attrs{Fn: graph.MapMul, Const: 1 / float64(S*S)}, nil, nil)[0]
	const eps = 1e-6
	out := b.MatVec(b.Norm(acc, jlm.RoleVProjNorm, eps), jlm.RoleVProj, N)
	b.Out(out)
	q8, ok := jlm.TypeOf(quant.Q8_0)
	if !ok {
		t.Fatal("Q8_0 has no container type")
	}
	formats := map[jlm.Role]jlm.Type{jlm.RoleVProj: q8, jlm.RoleVProjNorm: jlm.TypeF32}

	rng := rand.New(rand.NewSource(37))
	wq := tiledWeights(quant.Q8_0, layout.Q8_0, N, E, rng)
	qs, dw, sc, err := layout.PackWeights(layout.Q8_0, wq, N, E)
	if err != nil {
		t.Fatal(err)
	}
	weights := map[jlm.Role][3][]uint32{jlm.RoleVProj: {qs, dw, sc}}
	nw := make([]float32, E)
	for i := range nw {
		nw[i] = float32(1 + 0.3*rng.NormFloat64())
	}
	x := make([]float32, R*E*S*S)
	for i := range x {
		x[i] = float32(3 * rng.NormFloat64())
	}
	define := func(nw []float32, rows int) []float64 {
		want := make([]float64, rows*N)
		normed, o := make([]float32, E), make([]float32, N)
		for r := range rows {
			pooled := make([]float64, E)
			var ss float64
			for e := range E {
				for s := range S * S {
					pooled[e] += float64(x[r*E*S*S+s*E+e])
				}
				pooled[e] /= S * S
				ss += pooled[e] * pooled[e]
			}
			inv := 1 / math.Sqrt(ss/E+eps)
			for e := range E {
				normed[e] = float32(pooled[e] * inv * float64(nw[e]))
			}
			if err := oracle.MatVec(o, quant.Q8_0, wq, normed, N, E); err != nil {
				t.Fatal(err)
			}
			for i, v := range o {
				want[r*N+i] = float64(v)
			}
		}
		return want
	}
	want := define(nw, R)
	for _, d := range devs {
		defer d.Close()
		for _, matrix := range []bool{false, true} {
			dev := gpuDevice(d)
			if matrix {
				if ok, _ := backend.LowersTile(d, matrixTile); !ok {
					continue
				}
				dev.Matrix = [3]int{matrixTile.M, matrixTile.N, matrixTile.K}
			}
			t.Run(fmt.Sprintf("%s/matrix%v", d.API(), matrix), func(t *testing.T) {
				_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Formats: formats,
					Step: composer.Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
				if err != nil {
					t.Fatal(err)
				}
				onUnit := slices.ContainsFunc(codes, func(c composer.Code) bool { return slices.Contains(c.Passes, "matrix") })
				if matrix != onUnit {
					t.Fatalf("the projection on the matrix unit %v where the device states it %v", onUnit, matrix)
				}
				for _, c := range codes {
					t.Logf("%s: %d workgroups, passes %v", c.Name, c.Groups, c.Passes)
				}
				// The float form takes the activations as int8, a 32-block's step
				// amax/127 (about 3e-5 here); the matrix unit's as binary16.
				band := 1e-4
				if matrix {
					band = 2e-6
				}
				for _, rows := range []int{R, 100} {
					got, err := runPooled(d, codes, in, x[:rows*E*S*S], R*E*S*S, weights, map[jlm.Role][]float32{jlm.RoleVProjNorm: nw}, out, rows, N)
					if err != nil {
						t.Fatal(err)
					}
					e := nmseOf(got, want[:rows*N])
					t.Logf("a call of %d rows: NMSE %.3e against the definition (band %.0e)", rows, e, band)
					if math.IsNaN(e) || !(e < band) {
						t.Errorf("a call of %d rows reads NMSE %.3e against the definition, past %.0e", rows, e, band)
					}
				}
				scaled := slices.Clone(nw)
				for i := range scaled {
					scaled[i] *= 1 + 0.5*float32(i%3)
				}
				got, err := runPooled(d, codes, in, x, R*E*S*S, weights, map[jlm.Role][]float32{jlm.RoleVProjNorm: scaled}, out, R, N)
				if err != nil {
					t.Fatal(err)
				}
				if e := nmseOf(got, want); !(e > 1e-3) {
					t.Errorf("the norm's weight scaled reads NMSE %.3e: the gate cannot see the norm", e)
				}
			})
		}
	}
}

// runPooled launches codes over the first rows of the input x (its buffer
// capacity words, poisoned past them), the role's weight planes and
// vectors, the call's rows in the "rows.n" table, and reads back rows rows
// of n of out.
func runPooled(d backend.Device, codes []composer.Code, in graph.ValueID, x []float32, capacity int, weights map[jlm.Role][3][]uint32,
	vecs map[jlm.Role][]float32, out graph.ValueID, rows, n int) ([]float64, error) {
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
	f32 := func(v []float32, n int) []uint32 {
		w := poisoned(n)
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
			case o.Kind == composer.OperandValue && o.Value == in:
				w = f32(x, capacity)
			case o.Kind == composer.OperandTable && o.Name == "rows.n":
				w = []uint32{uint32(rows)}
			case o.Kind == composer.OperandVector || o.Kind == composer.OperandWeight && o.Format == jlm.TypeF32:
				v, ok := vecs[o.Role]
				if !ok {
					return nil, fmt.Errorf("%s binds %v, which the gate does not hold", c.Name, o.Role)
				}
				w = f32(v, len(v))
			case o.Kind == composer.OperandWeight:
				p, ok := weights[o.Role]
				if !ok {
					return nil, fmt.Errorf("%s binds %v, which the gate does not hold", c.Name, o.Role)
				}
				w = p[o.Region]
			case o.Kind == composer.OperandTable:
				return nil, fmt.Errorf("%s binds the table %q, which the gate does not hold", c.Name, o.Name)
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
	p := make([]byte, 4*rows*n)
	if err := outBuf.Read(p); err != nil {
		return nil, err
	}
	res := make([]float64, rows*n)
	for i := range res {
		res[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(p[4*i:])))
	}
	return res, nil
}
