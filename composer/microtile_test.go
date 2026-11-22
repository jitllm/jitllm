package composer_test

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// v100 is a V100's statement as the engine makes it: its units, threads,
// registers a thread at full occupancy, workgroup memory and matrix unit.
func v100() composer.Device {
	return composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, Regs: 32, Shared: 96 << 10, Matrix: [3]int{16, 16, 16}}
}

func projection(n, k int, typ jlm.Type, d composer.Device) composer.Request {
	b := graph.New()
	in := b.In("x", k)
	b.Out(b.MatVec(in, jlm.RoleAttnQ, n))
	return composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: typ},
		Step: composer.Step{Rows: 512, Page: 64, Keys: 1, ActWindow: 32}, Device: d}
}

// pickAt is the band a call of rows rows runs in, false past the last.
func pickAt(bands []composer.VectorBand, rows int) (composer.VectorBand, bool) {
	for _, b := range bands {
		if rows <= b.Upto {
			return b, true
		}
	}
	return composer.VectorBand{}, false
}

func describe(b composer.VectorBand, ok bool) string {
	switch {
	case !ok:
		return "capacity form"
	case b.Rows > 0:
		return fmt.Sprintf("vector %dx%d", b.Rows, b.Outs)
	}
	return fmt.Sprintf("matrix %dx%d", b.Tile.Rows, b.Tile.Cols)
}

// TestVectorMicrotilesFollowTheModel forms Llama-3.1-8B's projections --
// 4096x4096, 14336x4096 and 4096x14336 over Q4S and a 128256x4096 head over
// Q6S -- at a 512-row capacity on a V100's statement and logs the form the
// device's model picks at calls of 1, 2, 3, 4, 8 and 16 rows. At 2 and 8
// rows of the 4096x4096 projection the pick is the vector form with two or
// more outputs a thread, and its emitted code loads each activation word
// as many times fewer per output as it holds outputs, counted over a launch
// at the capacity against the one-output tile of the same rows.
func TestVectorMicrotilesFollowTheModel(t *testing.T) {
	d := v100()
	shapes := []struct {
		n, k int
		typ  jlm.Type
	}{{4096, 4096, jlm.TypeQ4S}, {14336, 4096, jlm.TypeQ4S}, {4096, 14336, jlm.TypeQ4S}, {128256, 4096, jlm.TypeQ6S}}
	for i, sh := range shapes {
		bands, err := composer.Bands(composer.Default(), projection(sh.n, sh.k, sh.typ, d))
		if err != nil {
			t.Fatal(err)
		}
		line := fmt.Sprintf("%dx%d %v:", sh.n, sh.k, sh.typ)
		for _, rows := range []int{1, 2, 3, 4, 8, 16} {
			b, ok := pickAt(bands, rows)
			line += fmt.Sprintf(" %d rows %s;", rows, describe(b, ok))
		}
		t.Log(line)
		if i > 0 {
			continue
		}
		for _, rows := range []int{2, 8} {
			b, ok := pickAt(bands, rows)
			if !ok || b.Rows == 0 || b.Outs < 2 {
				t.Fatalf("%s at %d rows: the model picks %s, want the vector form at two or more outputs a thread", line, rows, describe(b, ok))
			}
			loads := func(u int) (act, weight int) {
				f := d
				f.Matrix, f.RowTile, f.OutTile = [3]int{}, b.Rows, u
				req := projection(sh.n, sh.k, sh.typ, f)
				a, err := composer.ActivationLoads(composer.Default(), req)
				if err != nil {
					t.Fatal(err)
				}
				w, err := composer.WeightLoads(composer.Default(), req)
				if err != nil {
					t.Fatal(err)
				}
				return a, w
			}
			a1, w1 := loads(1)
			a2, w2 := loads(b.Outs)
			outputs := 512 * sh.n
			t.Logf("%d rows: the %dx1 tile loads %d activation words and %d weight words an output, the %dx%d tile %d and %d",
				rows, b.Rows, a1/outputs, w1/outputs, b.Rows, b.Outs, a2/outputs, w2/outputs)
			if a1 == 0 || a2*b.Outs != a1 {
				t.Fatalf("%d rows: the %dx%d tile loads %d activation words at the capacity, the %dx1 tile %d: want %d times fewer",
					rows, b.Rows, b.Outs, a2, b.Rows, a1, b.Outs)
			}
		}
	}
}

// TestComposedGPUMicrotiles forms a projection of 4136 outputs -- no
// multiple of a tile's outputs -- over 2048 at a 512-row capacity, at every
// tile of rows by outputs the model weighs and the ones it picks on a
// V100's statement, over Q4_K and Q6_K with int8 activations, on every
// device present, and runs each at calls of 1 to 9 rows, the rows past
// them poisoned: every row held to the oracle, bit for bit to its value in
// a call of 9 rows, and nothing written past the call's rows. Every tile of
// two or more outputs formed with each output copy reading the first
// output's weight (RowTiles.Alias) must fail the oracle.
func TestComposedGPUMicrotiles(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		testmodels.Missing(t, "no GPU backend on this host: the gate runs on one")
	}
	const n, k, capacity, guard, calls = 4136, 2048, 512, 64, 9
	tiles := [][2]int{{2, 1}, {2, 2}, {2, 4}, {2, 8}, {4, 2}, {4, 4}, {8, 1}, {8, 2}, {16, 1}}
	bands, err := composer.Bands(composer.Default(), projection(4096, 4096, jlm.TypeQ4S, v100()))
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range bands {
		if b.Rows > 0 && !slices.Contains(tiles, [2]int{b.Rows, b.Outs}) {
			tiles = append(tiles, [2]int{b.Rows, b.Outs})
		}
	}
	aliased := composer.Default()
	p := aliased.Profiles[composer.DeviceGPU]
	p.Passes = slices.Clone(p.Passes)
	for i, x := range p.Passes {
		if rt, ok := x.(composer.RowTiles); ok {
			rt.Alias = true
			p.Passes[i] = rt
		}
	}
	aliased.Profiles[composer.DeviceGPU] = p
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(47))
			ran, seen := 0, 0
			for _, typ := range []quant.Type{quant.Q4_K, quant.Q6_K} {
				jt, ok := jlm.TypeOf(typ)
				q, qok := layout.QuantOf(typ)
				if !ok || !qok {
					t.Fatalf("%v has no container type or kernel format", typ)
				}
				w := tiledWeights(typ, q, n, k, rng)
				qs, dw, sc, err := layout.PackWeights(q, w, n, k)
				if err != nil {
					t.Fatal(err)
				}
				planes := [3][]uint32{qs, dw, sc}
				x := make([]float32, calls*k)
				for i := range x {
					x[i] = float32(rng.NormFloat64() * math.Exp2(float64(rng.Intn(9)-4)))
				}
				want := int8Projection(t, typ, w, x, n, k, 32, false)
				for _, tile := range tiles {
					for _, alias := range []bool{false, true} {
						if alias && tile[1] < 2 {
							continue
						}
						r := composer.Default()
						if alias {
							r = aliased
						}
						f := dev
						f.RowTile, f.OutTile = tile[0], tile[1]
						name := fmt.Sprintf("%v %dx%d tile %dx%d alias %v", typ, n, k, tile[0], tile[1], alias)
						b := graph.New()
						in := b.In("x", k)
						out := b.MatVec(in, jlm.RoleAttnQ, n)
						b.Out(out)
						_, formed, err := r.Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: jt},
							Step: composer.Step{Rows: capacity, Page: 64, Keys: 1, ActWindow: 32}, Device: f})
						if err != nil {
							t.Fatalf("%s: forming: %v", name, err)
						}
						tiled := false
						for _, c := range formed {
							tiled = tiled || slices.Contains(c.Passes, "rowtiles")
						}
						if !tiled {
							t.Fatalf("%s: no code is tiled", name)
						}
						op, err := compileLone(d, formed)
						if err != nil {
							t.Fatalf("%s: %v", name, err)
						}
						run := func(L int) ([]float32, bool) {
							xs := make([]float32, capacity*k)
							copy(xs, x[:L*k])
							for i := L * k; i < len(xs); i++ {
								xs[i] = math.Float32frombits(guardWord)
							}
							res, tail, err := runGraphRows(d, formed, op.kerns, in, out, xs, capacity*n, guard, planes, L)
							if err != nil {
								t.Fatalf("%s: %v", name, err)
							}
							clean := true
							for _, v := range res[L*n:] {
								clean = clean && math.Float32bits(v) == guardWord
							}
							for _, v := range tail {
								clean = clean && v == guardWord
							}
							return res[:L*n], clean
						}
						if alias {
							L := min(tile[0], calls)
							got, _ := run(L)
							e := gpuNMSE(got, want[:L*n])
							if !(e > 1e-6) {
								t.Fatalf("%s: every output reading the first's weight reads NMSE %.3e against the oracle: the gate cannot see it", name, e)
							}
							seen++
							op.close()
							continue
						}
						full, _ := run(calls)
						for L := 1; L <= calls; L++ {
							got, clean := run(L)
							if !clean {
								t.Fatalf("%s: a call of %d rows wrote past them", name, L)
							}
							for i, v := range got {
								if math.Float32bits(v) != math.Float32bits(full[i]) {
									t.Fatalf("%s: a call of %d rows writes row %d's output %d as %08x, a call of %d rows as %08x",
										name, L, i/n, i%n, math.Float32bits(v), calls, math.Float32bits(full[i]))
								}
							}
							for row := range L {
								if e := gpuNMSE(got[row*n:(row+1)*n], want[row*n:(row+1)*n]); math.IsNaN(e) || !(e < 1e-9) {
									t.Fatalf("%s: a call of %d rows: row %d reads NMSE %.3e against the oracle", name, L, row, e)
								}
							}
							ran++
						}
						op.close()
					}
				}
			}
			t.Logf("%s: %d calls of %d tiles right against the oracle and bit for bit against a call of %d rows, %d aliased tiles seen",
				d.API(), ran, len(tiles), calls, seen)
		})
	}
}
