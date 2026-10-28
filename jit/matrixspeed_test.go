package jit_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// BenchmarkComposedMatrix times the composer's projection on the matrix unit
// against main's staged GEMM (kernels.GemmVolta at the tile and k split the
// tier ships for the shape, its partials' sum not timed) on Llama-3.1-8B
// Q4_K_M's projection shapes at a 512-row chunk, on every CUDA
// device, interleaved: each round launches both arms in turn, and the figure
// is each arm's median microseconds and the median of the per-round ratios.
// Main's arm reads binary16 activations a kernel before it wrote, which it
// does not time; the composer's reads the float rows. It runs its own rounds:
//
//	go test ./jit -run '^$' -bench ComposedMatrix -benchtime 1x
func BenchmarkComposedMatrix(t *testing.B) {
	testlock.GPU(t)
	const R, rounds, reps = 512, 16, 5
	for _, d := range backend.Open() {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		ok, why := backend.LowersTile(d, matrixTile)
		if !ok {
			t.Fatalf("%s: no binary16 tile: %s", d.Name(), why)
		}
		dev := gpuDevice(d)
		dev.Matrix = [3]int{matrixTile.M, matrixTile.N, matrixTile.K}
		for _, sh := range []struct {
			what  string
			typ   quant.Type
			n, k  int
			split int
		}{
			{"q/o", quant.Q4_K, 4096, 4096, 1},
			{"k", quant.Q4_K, 1024, 4096, 2},
			{"gate/up", quant.Q4_K, 14336, 4096, 1},
			{"down", quant.Q6_K, 4096, 14336, 1},
		} {
			jt, _ := jlm.TypeOf(sh.typ)
			q, _ := kernels.QuantOf(sh.typ)
			rng := rand.New(rand.NewSource(5))
			w := tiledWeights(sh.typ, q, sh.n, sh.k, rng)
			qs, dw, sc, err := kernels.PackWeights(q, w, sh.n, sh.k)
			if err != nil {
				t.Fatal(err)
			}
			if len(sc) == 0 {
				sc = []uint32{0}
			}
			for _, mx := range matrixConfigs(t) {
				reg := jit.Default()
				if mx.Rows != 0 {
					p := reg.Profiles[jit.DeviceGPU]
					p.Passes = slices.Clone(p.Passes)
					for i, ps := range p.Passes {
						if _, ok := ps.(jit.Matrix); ok {
							p.Passes[i] = mx
						}
					}
					reg.Profiles[jit.DeviceGPU] = p
				}
				b := graph.New()
				in := b.In("x", sh.k)
				out := b.MatVec(in, jlm.RoleFFNUp, sh.n)
				b.Out(out)
				_, codes, err := reg.Form(jit.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleFFNUp: jt},
					Step: jit.Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
				if err != nil {
					t.Fatal(err)
				}
				op, err := compileLone(d, codes)
				if err != nil {
					t.Errorf("%v %s: %v", mx, sh.what, err)
					continue
				}
				var bufs []backend.Buf
				up := func(words []uint32) backend.Buf {
					bf, err := d.Alloc(max(4*len(words), 4))
					if err != nil {
						t.Fatal(err)
					}
					p := make([]byte, 4*len(words))
					for i, v := range words {
						binary.LittleEndian.PutUint32(p[4*i:], v)
					}
					if len(p) > 0 {
						if err := bf.Write(p); err != nil {
							t.Fatal(err)
						}
					}
					bufs = append(bufs, bf)
					return bf
				}
				x := make([]uint32, R*sh.k)
				for i := range x {
					x[i] = math.Float32bits(float32(rng.NormFloat64()))
				}
				planes := [3]backend.Buf{up(qs), up(dw), up(sc)}
				xBuf, outBuf := up(x), up(make([]uint32, R*sh.n))
				own := map[jit.Operand]backend.Buf{}
				args := make([][]backend.Buf, len(codes))
				for ci, c := range codes {
					for _, bd := range c.Binds {
						o := bd.Operand
						var bf backend.Buf
						switch {
						case o.Kind == jit.OperandValue && o.Value == in:
							bf = xBuf
						case o.Kind == jit.OperandValue && o.Value == out:
							bf = outBuf
						case o.Kind == jit.OperandWeight:
							bf = planes[o.Region]
						case o.Kind == jit.OperandTable && o.Name == "rows.n":
							bf = up([]uint32{R})
						default:
							if bf = own[o]; bf == nil {
								w := bd.Words
								if w == nil {
									w = make([]uint32, max(bd.Bytes, 4)/4)
								}
								bf = up(w)
								own[o] = bf
							}
						}
						args[ci] = append(args[ci], bf)
					}
				}
				composed := func() {
					for ci, c := range codes {
						if err := op.kerns[ci].Launch(c.LiveGroups(R), c.GPU.Group[0], args[ci]...); err != nil {
							t.Fatal(err)
						}
					}
				}
				// Main's shipping tile: the first of the tier's blockings
				// (voltaTiles) that compiles, k split where its grid is thin.
				sub, _, _, _ := kernels.Layout(q)
				s := kernels.MatVecShape{T: q, K: sh.k, Rows: sh.n, NTok: R, Split: sh.split}
				var tl kernels.VoltaTile
				var mainKern backend.Kernel
				for _, c := range []kernels.VoltaTile{{MT: 4, NT: 4, WM: 1, WN: 4}, {MT: 4, NT: 2, WM: 1, WN: 4}, {MT: 4, NT: 1, WM: 1, WN: 4}} {
					c.KB = max(32/sub, 1)
					mk, err := kernels.GemmVolta(s, c)
					if err != nil {
						continue
					}
					if mainKern, err = d.Compile(mk); err == nil {
						tl = c
						break
					}
				}
				if mainKern == nil {
					t.Fatalf("%s: no shipping tile compiles", sh.what)
				}
				act16 := up(make([]uint32, R*sh.k/2))
				mainOut := up(make([]uint32, R*sh.n*max(sh.split, 1)))
				mainArm := func() {
					if err := mainKern.Launch(kernels.GemmVoltaGroups(s, tl), tl.Threads(), planes[0], planes[1], planes[2], act16, mainOut); err != nil {
						t.Fatal(err)
					}
				}
				sync := func() { outBuf.Read(make([]byte, 4)) }
				time1 := func(f func()) float64 {
					sync()
					t0 := time.Now()
					for range reps {
						f()
					}
					sync()
					return float64(time.Since(t0).Microseconds()) / reps
				}
				composed()
				mainArm()
				first := func() {
					if err := op.kerns[0].Launch(codes[0].LiveGroups(R), codes[0].GPU.Group[0], args[0]...); err != nil {
						t.Fatal(err)
					}
				}
				var cu, mu, fu, ratio []float64
				for range rounds {
					c, m, f := time1(composed), time1(mainArm), time1(first)
					cu, mu, fu, ratio = append(cu, c), append(mu, m), append(fu, f), append(ratio, c/m)
				}
				med := func(v []float64) float64 {
					s := append([]float64(nil), v...)
					sort.Float64s(s)
					return s[len(s)/2]
				}
				flop := 2 * float64(R) * float64(sh.n) * float64(sh.k)
				if r, ok := backend.KernelResourcesOf(op.kerns[len(codes)-1]); ok {
					t.Logf("%s %s: composed %d workgroups of %d, regs %d, spill %d, shared %d, resident %d", d.Name(), sh.what,
						codes[len(codes)-1].LiveGroups(R), codes[len(codes)-1].GPU.Group[0], r.Regs, r.SpillBytes, r.SharedBytes, r.Resident)
				}
				if r, ok := backend.KernelResourcesOf(mainKern); ok {
					t.Logf("%s %s: main %+v, %d workgroups of %d, regs %d, spill %d, shared %d, resident %d", d.Name(), sh.what,
						tl, kernels.GemmVoltaGroups(s, tl), tl.Threads(), r.Regs, r.SpillBytes, r.SharedBytes, r.Resident)
				}
				t.Logf("%s %v %-8s %v %5dx%-5d rows %d: composed %8.1f us (%5.1f TFLOP/s, %d kernels, the first %.1f us), main %8.1f us (%5.1f TFLOP/s), composed/main %.2f",
					d.Name(), mx, sh.what, sh.typ, sh.n, sh.k, R, med(cu), flop/med(cu)/1e6, len(codes), med(fu), med(mu), flop/med(mu)/1e6, med(ratio))
				mainKern.Close()
				op.close()
				for _, bf := range bufs {
					bf.Free()
				}
			}
		}
	}
}

// matrixConfigs is the matrix forms the probe times: JITLLM_MATRIX_SHAPES,
// a comma list of ROWSxCOLSxDEPTH:WRxWN:RESIDENT, or the profile's alone.
func matrixConfigs(t testing.TB) []jit.Matrix {
	env := os.Getenv("JITLLM_MATRIX_SHAPES")
	if env == "" {
		return []jit.Matrix{{}}
	}
	var out []jit.Matrix
	for _, f := range strings.Split(env, ",") {
		var v []int
		for _, x := range strings.FieldsFunc(f, func(r rune) bool { return r == 'x' || r == ':' }) {
			n, err := strconv.Atoi(x)
			if err != nil {
				t.Fatalf("JITLLM_MATRIX_SHAPES %q: %v", f, err)
			}
			v = append(v, n)
		}
		if len(v) != 6 {
			t.Fatalf("JITLLM_MATRIX_SHAPES %q: want ROWSxCOLSxDEPTH:WRxWN:RESIDENT", f)
		}
		out = append(out, jit.Matrix{Rows: v[0], Cols: v[1], Depth: v[2], Warps: [2]int{v[3], v[4]}, Resident: v[5]})
	}
	return out
}
