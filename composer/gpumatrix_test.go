package composer_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testlock"
)

// matrixTile() is the binary16 tile the gate asks a device to multiply.
func matrixTile() backend.TileShape {
	return backend.TileShape{M: 16, N: 16, K: 16, Elem: ir.TileF16, Acc: ir.TileF32}
}

// matrixDevices is every device present whose matrix unit takes binary16
// tiles staged in workgroup memory, as the composer states it, and a CUDA
// card lowered for sm_70 (the quad-pair layout) where one runs it.
func matrixDevices(t *testing.T) (map[string]backend.Device, func()) {
	out := map[string]backend.Device{}
	var all []backend.Device
	for _, d := range openDevices() {
		all = append(all, d)
		if ok, why := backend.LowersTile(d, matrixTile()); ok {
			out[d.API()] = d
		} else {
			t.Logf("%s: no binary16 tile: %s", d.API(), why)
		}
	}
	if d, ok := sm70(t); ok {
		all = append(all, d)
		if ok, _ := backend.LowersTile(d, matrixTile()); ok {
			out["ptx-sm_70"] = d
		}
	}
	return out, func() {
		for _, d := range all {
			d.Close()
		}
	}
}

// sm70 is cuda:0 opened at an sm_70 target where the selection holds it: a
// card named outside it would end the binary (testdevices.Admit).
func sm70(t *testing.T) (backend.Device, bool) {
	if !testdevices.Has(t, "cuda", 0) {
		return nil, false
	}
	d, err := backend.OpenCUDAWith(0, backend.Opts{Admit: testdevices.Admit(), CUDATarget: "sm_70"})
	return d, err == nil
}

// matrixWeights is rows of k of t, random, with finite scales near one; a
// binary16 matrix's elements normal.
func matrixWeights(t quant.Type, q layout.Quant, rows, k int, rng *rand.Rand) []byte {
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
// matrix pass's, its tiles multiplied), and on a card stating its units
// beside a vector form the calls of few rows run (alternatives), each form
// launched at some call and k cut by the call's rows at one tile at least;
// each row held to the oracle within
// binary16's rounding of both factors, rows whose magnitudes reach past
// binary16's range among them (every fifth times 2^20), and every word past
// the call's rows and the guard after them untouched. Two violations must fail: the
// weight's words moved by one, and the reference over the next row's input.
func TestComposedGPUMatrix(t *testing.T) {
	testlock.GPU(t)
	devs, closeAll := matrixDevices(t)
	defer closeAll()
	if len(devs) == 0 {
		t.Skip("no device here multiplies binary16 tiles")
	}
	const guard = 64
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
			dev.Matrix = [3]int{matrixTile().M, matrixTile().N, matrixTile().K}
			rng := rand.New(rand.NewSource(53))
			ran, cuts, banded, vecRan, matRan := 0, 0, 0, 0, 0
			calls100 := []int{1, 15, 33, 64, 100}
			for _, typ := range formats {
				// The last shape's row tiles and outputs fill the units at
				// its capacity and its k is long, so the model leaves k whole
				// there and cuts it at launch by the call's rows (Cut): run
				// at the profile's tile and its model's split alone, over
				// binary16 and the served models' commonest format.
				for _, sh := range []struct {
					n, k, capacity int
					calls          []int
				}{{100, 256, 100, calls100}, {1056, 512, 100, calls100}, {160, 2048, 100, calls100}, {2048, 4096, 512, []int{1, 15, 33, 100, 300, 512}}} {
					// One weight, one call and their oracles a format and shape:
					// every tile runs the same projection.
					var w []byte
					var qs, dw, sc []uint32
					var x, want, big, wantBig, want8 []float32
					for _, tile := range splitTiles(gpuMatrixTiles()) {
						n, k, capacity, calls := sh.n, sh.k, sh.capacity, sh.calls
						own := gpuMatrixTiles()[0]
						own.Split, own.Convert = 0, -1
						if capacity != 100 && (tile != own || typ != quant.F16 && typ != quant.Q4_K) {
							continue
						}
						td := dev
						td.Tile = tile
						jt, ok := jlm.TypeOf(typ)
						q, qok := layout.QuantOf(typ)
						if typ == quant.F16 {
							jt, q, ok, qok = jlm.TypeF16, layout.Float16, true, true
						}
						if !ok || !qok {
							t.Fatalf("%v has no container type or kernel format", typ)
						}
						what := fmt.Sprintf("%v/%d outputs of %d at %v", typ, n, k, tile)
						if w == nil {
							w = matrixWeights(typ, q, n, k, rng)
							var err error
							if qs, dw, sc, err = layout.PackWeights(q, w, n, k); err != nil {
								t.Fatal(err)
							}
						}
						b := graph.New()
						in := b.In("x", k)
						out := b.MatVec(in, jlm.RoleAttnQ, n)
						b.Out(out)
						_, formed, err := poisonedRegistry().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: jt},
							Step: composer.Step{Rows: capacity, Page: 64, Keys: 1, ActWindow: 32}, Device: td})
						if err != nil {
							t.Fatalf("%s: forming: %v", what, err)
						}
						tiles, atTile := 0, 0
						for _, c := range formed {
							if c.Cut.Most > 1 && !c.Cut.Sums {
								cuts++
							}
							if c.Band.Upto > 0 {
								banded++
							}
							for _, L := range calls {
								switch {
								case c.Band.Upto > 0 && c.Runs(L, 0):
									vecRan++
								case c.Band.Past > 0 && c.Runs(L, 0):
									matRan++
								}
							}
							if slices.Contains(c.Passes, "matrix") {
								if c.Matrix != [2]int{} {
									if !formedAtTile(c, tile, capacity, n) {
										t.Fatalf("%s: the matrix kernel's %d workgroups of %d threads are not the tile's", what, c.Groups, c.GPU.Group[0])
									}
									atTile++
								}
								for _, o := range c.GPU.Ops {
									if o.Kind == ir.OpTileMMA || o.Kind == ir.OpMMA {
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
						if x == nil {
							x = make([]float32, capacity*k)
							for i := range x {
								x[i] = float32(rng.NormFloat64())
							}
							want = make([]float32, capacity*n)
							if err := oracle.MatMul(want, typ, w, x, capacity, n, k); err != nil {
								t.Fatal(err)
							}
							// Rows past binary16's range, every fifth times
							// 2^20: scaled into it and their sums back, as
							// exactly as the rest.
							big = slices.Clone(x)
							for r := 2; r < capacity; r += 5 {
								for i := r * k; i < (r+1)*k; i++ {
									big[i] *= 1 << 20
								}
							}
							wantBig = make([]float32, capacity*n)
							if err := oracle.MatMul(wantBig, typ, w, big, capacity, n, k); err != nil {
								t.Fatal(err)
							}
							want8 = int8Projection(t, typ, w, x, n, k, 32, false)
						}
						runX := func(x []float32, L int, planes [3][]uint32) []float32 {
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
						run := func(L int, planes [3][]uint32) []float32 { return runX(x, L, planes) }
						planes := [3][]uint32{qs, dw, sc}
						if e := gpuNMSE(runX(big, capacity, planes), wantBig); math.IsNaN(e) || !(e < 2e-6) {
							t.Fatalf("%s: rows past binary16's range read NMSE %.3e against the oracle", what, e)
						}
						// A call of few rows runs the vector form beside the
						// matrix form (alternatives): the int8 matvec over the
						// step's activation window where the format takes one,
						// else the decoded weight's.
						vectorAt := func(L int) bool {
							for _, c := range formed {
								if c.Band.Upto > 0 && c.Runs(L, 0) {
									return true
								}
							}
							return false
						}
						worst := 0.0
						for _, L := range calls {
							got := run(L, planes)
							e := gpuNMSE(got, want[:L*n])
							if vectorAt(L) {
								e = min(e, gpuNMSE(got, want8[:L*n]))
							}
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
			if dev.Units > 0 && (banded == 0 || vecRan == 0 || matRan == 0) {
				t.Fatalf("%s: the vector form beside the matrix form: %d codes banded, %d launches of the vector form's and %d of the matrix form's", name, banded, vecRan, matRan)
			}
			if dev.Units > 0 && cuts == 0 {
				t.Fatalf("%s: no projection's k was cut at launch: the calls of few rows never ran a cut", name)
			}
			t.Logf("%s: %d projections formed on the matrix unit, %d cut at launch, held to the oracle at calls of 1 to their capacity's rows", name, ran, cuts)
		})
	}
}

// gpuMatrixTiles is the GPU profile's matrix tile, then its alternatives.
func gpuMatrixTiles() []composer.MatrixTile {
	for _, p := range composer.Default().Profiles[composer.DeviceGPU].Passes {
		if m, ok := p.(composer.Matrix); ok {
			return append([]composer.MatrixTile{m.Tile()}, m.Tiles...)
		}
	}
	return nil
}

// splitTiles is each tile with k whole and cut in four, the row's factor
// written as halves first, k whole with the factor converted in the stage,
// and k cut as the model chooses, at launch by the call's rows (Cut).
func splitTiles(tiles []composer.MatrixTile) []composer.MatrixTile {
	var out []composer.MatrixTile
	for _, t := range tiles {
		for _, c := range [][2]int{{1, -1}, {4, -1}, {1, 1}, {0, -1}} {
			t.Split, t.Convert = c[0], c[1]
			out = append(out, t)
		}
	}
	return out
}

// formedAtTile is whether c, a matrix kernel over capacity rows by n
// outputs, holds tile's workgroups, k cut in its segments (any the model
// chooses where the tile states none): its rows and
// warps along them halved where the device's workgroup memory takes fewer.
func formedAtTile(c composer.Code, t composer.MatrixTile, capacity, n int) bool {
	for rows, wr := t.Rows, t.Warps[0]; wr >= 1; rows, wr = rows/2, wr/2 {
		base := (capacity + rows - 1) / rows * ((n + t.Cols - 1) / t.Cols)
		for s := max(t.Split, 1); s <= max(t.Split, 32); s *= 2 {
			if c.GPU.Group[0] == 32*wr*t.Warps[1] && c.Groups == base*s {
				return true
			}
		}
	}
	return false
}

// poisonedRegistry is the default registry with its forms' workgroup
// arrays poisoned: a kernel reading one it never wrote moves its result.
func poisonedRegistry() *composer.Registry {
	r := composer.Default()
	r.PoisonWorkgroupMemory()
	return r
}
