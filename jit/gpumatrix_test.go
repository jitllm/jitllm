package jit_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// matrixTile is the binary16 tile the gate asks a device to multiply.
var matrixTile = backend.TileShape{M: 16, N: 16, K: 16, Elem: ir.TileF16, Acc: ir.TileF32}

// matrixDevices is every device present whose matrix unit takes binary16
// tiles staged in workgroup memory, as the composer states it, and a CUDA
// card lowered for sm_70 (the quad-pair layout) where one runs it.
func matrixDevices(t *testing.T) (map[string]backend.Device, func()) {
	out := map[string]backend.Device{}
	var all []backend.Device
	for _, d := range backend.Open() {
		all = append(all, d)
		if ok, why := backend.LowersTile(d, matrixTile); ok {
			out[d.API()] = d
		} else {
			t.Logf("%s: no binary16 tile: %s", d.API(), why)
		}
	}
	if d, err := backend.OpenCUDAWith(0, backend.Opts{CUDATarget: "sm_70"}); err == nil {
		all = append(all, d)
		if ok, _ := backend.LowersTile(d, matrixTile); ok {
			out["ptx-sm_70"] = d
		}
	}
	return out, func() {
		for _, d := range all {
			d.Close()
		}
	}
}

// matrixWeights is rows of k of t, random, with finite scales near one; a
// binary16 matrix's elements normal.
func matrixWeights(t quant.Type, q kernels.Quant, rows, k int, rng *rand.Rand) []byte {
	if t != quant.F16 {
		return tiledWeights(t, q, rows, k, rng)
	}
	b := make([]byte, 2*rows*k)
	for i := range rows * k {
		binary.LittleEndian.PutUint16(b[2*i:], quant.EncodeHalf(float32(rng.NormFloat64())))
	}
	return b
}

// TestComposedGPUMatrix forms a projection at a prompt chunk's rows on every
// device whose matrix unit takes binary16 tiles staged in workgroup memory,
// and on a CUDA card lowered for sm_70, over binary16 and every packed format
// of the served models, at outputs and a capacity that do not fill the
// workgroups' blocks, and runs the one form at calls of 1 to the capacity's
// rows: the projection must be formed on the matrix unit (its kernel the
// matrix pass's, its tiles multiplied), each row held to the oracle within
// binary16's rounding of both factors, and every word past the call's rows
// and the guard after them untouched. Two violations must fail: the
// weight's words moved by one, and the reference over the next row's input.
func TestComposedGPUMatrix(t *testing.T) {
	testlock.GPU(t)
	devs, closeAll := matrixDevices(t)
	defer closeAll()
	if len(devs) == 0 {
		t.Skip("no device here multiplies binary16 tiles")
	}
	const guard, capacity = 64, 100
	calls := []int{1, 15, 33, 64, 100}
	formats := []quant.Type{quant.F16, quant.Q8_0, quant.Q3_K, quant.Q4_K, quant.Q5_K, quant.Q6_K}
	names := slices.Sorted(func(yield func(string) bool) {
		for n := range devs {
			if !yield(n) {
				return
			}
		}
	})
	for _, name := range names {
		d := devs[name]
		t.Run(name, func(t *testing.T) {
			dev := gpuDevice(d)
			dev.Matrix = [3]int{matrixTile.M, matrixTile.N, matrixTile.K}
			rng := rand.New(rand.NewSource(53))
			ran := 0
			for _, typ := range formats {
				for _, sh := range []struct{ n, k int }{{100, 256}, {1056, 512}, {160, 2048}} {
					for _, tile := range gpuMatrixTiles() {
						n, k := sh.n, sh.k
						td := dev
						td.Tile = tile
						jt, ok := jlm.TypeOf(typ)
						q, qok := kernels.QuantOf(typ)
						if typ == quant.F16 {
							jt, q, ok, qok = jlm.TypeF16, kernels.Float16, true, true
						}
						if !ok || !qok {
							t.Fatalf("%v has no container type or kernel format", typ)
						}
						what := fmt.Sprintf("%v/%d outputs of %d at %v", typ, n, k, tile)
						w := matrixWeights(typ, q, n, k, rng)
						qs, dw, sc, err := kernels.PackWeights(q, w, n, k)
						if err != nil {
							t.Fatal(err)
						}
						b := graph.New()
						in := b.In("x", k)
						out := b.MatVec(in, jlm.RoleAttnQ, n)
						b.Out(out)
						_, formed, err := poisonedRegistry().Form(jit.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: jt},
							Step: jit.Step{Rows: capacity, Page: 64, Keys: 1, ActWindow: 32}, Device: td})
						if err != nil {
							t.Fatalf("%s: forming: %v", what, err)
						}
						tiles, atTile := 0, 0
						for _, c := range formed {
							if slices.Contains(c.Passes, "matrix") {
								if c.Matrix != [2]int{} {
									if !formedAtTile(c, tile, capacity, n) {
										t.Fatalf("%s: the matrix kernel's %d workgroups of %d threads are not the tile's", what, c.Groups, c.GPU.Group[0])
									}
									atTile++
								}
								for _, o := range c.GPU.Ops {
									if o.Kind == ir.OpTileMMA {
										tiles++
									}
								}
							}
						}
						if tiles == 0 || atTile == 0 {
							t.Fatalf("%s: no kernel was formed on the matrix unit (%d tiles, %d kernels stating their shape)", what, tiles, atTile)
						}
						op, err := compileLone(d, formed)
						if err != nil {
							t.Fatalf("%s: %v", what, err)
						}
						x := make([]float32, capacity*k)
						for i := range x {
							x[i] = float32(rng.NormFloat64())
						}
						want := make([]float32, capacity*n)
						for r := range capacity {
							if err := oracle.MatVec(want[r*n:(r+1)*n], typ, w, x[r*k:(r+1)*k], n, k); err != nil {
								t.Fatal(err)
							}
						}
						run := func(L int, planes [3][]uint32) []float32 {
							xs := make([]float32, capacity*k)
							copy(xs, x[:L*k])
							for i := L * k; i < len(xs); i++ {
								xs[i] = math.Float32frombits(guardWord)
							}
							res, tail, err := runGraphRows(d, formed, op.kerns, in, out, xs, capacity*n, guard, planes, L)
							if err != nil {
								t.Fatalf("%s: %v", what, err)
							}
							for i, v := range res[L*n:] {
								if math.Float32bits(v) != guardWord {
									t.Fatalf("%s: a call of %d rows wrote row %d's element %d", what, L, L+i/n, i%n)
								}
							}
							for i, v := range tail {
								if v != guardWord {
									t.Fatalf("%s: the guard word %d past the result was written", what, i)
								}
							}
							return res[:L*n]
						}
						planes := [3][]uint32{qs, dw, sc}
						worst := 0.0
						for _, L := range calls {
							e := gpuNMSE(run(L, planes), want[:L*n])
							if math.IsNaN(e) || !(e < 2e-6) {
								t.Fatalf("%s: a call of %d rows reads NMSE %.3e against the oracle", what, L, e)
							}
							worst = max(worst, e)
						}
						if ran == 0 {
							for _, c := range formed {
								t.Logf("%s: %s, %d workgroups of %d, passes %v", what, c.Name, c.Groups, c.GPU.Group[0], c.Passes)
							}
						}
						t.Logf("%s: worst NMSE %.3e", what, worst)
						if e := gpuNMSE(run(capacity, planes)[:(capacity-1)*n], want[n:]); !(e > 1e-2) {
							t.Fatalf("%s: against the next row's reference the NMSE is %.3e: the gate cannot see the rows", what, e)
						}
						moved := [3][]uint32{append(qs[1:len(qs):len(qs)], qs[0]), dw, sc}
						if e := gpuNMSE(run(capacity, moved), want); !(e > 1e-4) {
							t.Fatalf("%s: with the weight's words moved by one the NMSE is %.3e", what, e)
						}
						op.close()
						ran++
					}
				}
			}
			t.Logf("%s: %d projections formed on the matrix unit, held to the oracle at calls of %v rows", name, ran, calls)
		})
	}
}

// gpuMatrixTiles is the GPU profile's matrix tile, then its alternatives.
func gpuMatrixTiles() []jit.MatrixTile {
	for _, p := range jit.Default().Profiles[jit.DeviceGPU].Passes {
		if m, ok := p.(jit.Matrix); ok {
			return append([]jit.MatrixTile{m.Tile()}, m.Tiles...)
		}
	}
	return nil
}

// formedAtTile is whether c, a matrix kernel over capacity rows by n
// outputs, holds tile's workgroups: its rows and warps along them halved
// where the device's workgroup memory takes fewer.
func formedAtTile(c jit.Code, t jit.MatrixTile, capacity, n int) bool {
	for rows, wr := t.Rows, t.Warps[0]; wr >= 1; rows, wr = rows/2, wr/2 {
		if c.GPU.Group[0] == 32*wr*t.Warps[1] && c.Groups == (capacity+rows-1)/rows*((n+t.Cols-1)/t.Cols) {
			return true
		}
	}
	return false
}

// poisonedRegistry is the default registry with its forms' workgroup
// arrays poisoned: a kernel reading one it never wrote moves its result.
func poisonedRegistry() *jit.Registry {
	r := jit.Default()
	r.PoisonWorkgroupMemory()
	return r
}
