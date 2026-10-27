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
)

// tiledFormats are the weight formats the row-tile gates form a projection
// over: a float matrix and the packed types of the served models.
var tiledFormats = []quant.Type{quant.F32, quant.Q8_0, quant.Q3_K, quant.Q4_K, quant.Q5_K, quant.Q6_K}

// ffnBlock is a Llama-shaped block's MLP at the model's widths: a norm, gate
// and up, SiLU, down and the residual add.
func ffnBlock(embd, ffn int) *graph.Graph {
	b := graph.New()
	b.G.Block = 0
	x := b.In("x", embd)
	n := b.Norm(x, jlm.RoleFFNNorm, 1e-5)
	gate := b.MatVec(n, jlm.RoleFFNGate, ffn)
	up := b.MatVec(n, jlm.RoleFFNUp, ffn)
	b.Out(b.Add(x, b.MatVec(siluMul(b, gate, up, ffn), jlm.RoleFFNDown, embd)))
	return b.G
}

// TestRowTilesHoldTheOps forms a Llama-3.2-1B-shaped MLP over Q4_K, Q6_K and
// Q8_0 weights at a prompt chunk's rows -- ragged counts among them, whose
// last tile is guarded -- with int8 and with float activations, on a device
// of CUDA's statement and one of Vulkan's: from 128 rows on the largest
// kernel's op count stays within 1% (an address constant's form, a tail's
// guard), from two rows on every projection is tiled, and no kernel passes
// 1,000 ops. At one row nothing is tiled. On the device whose arithmetic is
// exact each tiled projection has its form for few rows beside it, and on
// the other none. With the pass off every row loads the weight itself, at
// eight times the weight loads a row or more: the gate sees it.
func TestRowTilesHoldTheOps(t *testing.T) {
	cuda := jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, ExactDiv: true, ExactArith: true}
	formats := func(typ jlm.Type) map[jlm.Role]jlm.Type {
		return map[jlm.Role]jlm.Type{jlm.RoleFFNGate: typ, jlm.RoleFFNUp: typ, jlm.RoleFFNDown: typ, jlm.RoleFFNNorm: jlm.TypeF32}
	}
	for _, typ := range []jlm.Type{jlm.TypeQ4S, jlm.TypeQ6S, jlm.TypeQ8} {
		for _, d := range []jit.Device{cuda, {Kind: jit.DeviceGPU, Lanes: 32}} {
			for _, float := range []bool{false, true} {
				reg := jit.Default()
				if float {
					reg = reg.WithoutInt8(jit.DeviceGPU)
				}
				name := fmt.Sprintf("%v/units %d/float %v", typ, d.Units, float)
				var held []int
				for _, R := range []int{1, 3, 7, 31, 33, 100, 128, 256, 511, 512} {
					_, codes, err := reg.Form(jit.Request{Graph: ffnBlock(2048, 8192), Formats: formats(typ),
						Step: jit.Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32}, Device: d})
					if err != nil {
						t.Fatalf("%s at %d rows: %v", name, R, err)
					}
					largest, tiled, few := 0, 0, 0
					for _, c := range codes {
						largest = max(largest, len(c.GPU.Ops))
						switch {
						case c.Few > 0:
							few++
						case slices.Contains(c.Passes, "rowtiles"):
							tiled++
						}
					}
					fewWant := 0
					if R > 1 && d.ExactArith {
						fewWant = 3
					}
					switch {
					case R == 1 && tiled != 0:
						t.Fatalf("%s: %d kernels tiled at one row", name, tiled)
					case R > 1 && tiled != 3:
						t.Fatalf("%s at %d rows: %d of 3 projections tiled", name, R, tiled)
					case few != fewWant:
						t.Fatalf("%s at %d rows: %d forms for few rows, want %d", name, R, few, fewWant)
					case largest > 1000:
						t.Fatalf("%s at %d rows: a kernel of %d ops", name, R, largest)
					}
					if R >= 128 {
						held = append(held, largest)
					}
					t.Logf("%s at %d rows: %d kernels, the largest %d ops, %d tiled", name, R, len(codes), largest, tiled)
				}
				if 100*slices.Max(held) > 101*slices.Min(held) {
					t.Fatalf("%s: the largest kernel runs from %d to %d ops past 128 rows", name, slices.Min(held), slices.Max(held))
				}
			}
		}
	}
	// What the pass saves is the weight's loads: a tile's rows read each
	// word once. With it off every row loads the weight itself, and the
	// projections' loads a row grow by the tile's rows: eight times or more.
	for _, d := range []jit.Device{cuda, {Kind: jit.DeviceGPU, Lanes: 32}} {
		for _, float := range []bool{false, true} {
			loads := func(off bool) int {
				reg := jit.Default()
				if float {
					reg = reg.WithoutInt8(jit.DeviceGPU)
				}
				if off {
					p := reg.Profiles[jit.DeviceGPU]
					p.Off = map[string]bool{"rowtiles": true}
					reg.Profiles[jit.DeviceGPU] = p
				}
				n, err := jit.WeightLoads(reg, jit.Request{Graph: ffnBlock(2048, 8192), Formats: formats(jlm.TypeQ4S),
					Step: jit.Step{Rows: 32, Page: 64, Keys: 1, ActWindow: 32}, Device: d})
				if err != nil {
					t.Fatal(err)
				}
				return n / 32
			}
			on, off := loads(false), loads(true)
			if on == 0 || off < 8*on {
				t.Fatalf("units %d/float %v: the weight loads a row at 32 rows are %d with the pass and %d without: the gate cannot see it",
					d.Units, float, on, off)
			}
			t.Logf("units %d/float %v: the weight loads a row at 32 rows, %d with the pass and %d without", d.Units, float, on, off)
		}
	}
	// The device's registers size the tile: a V100's 32 a thread hold the
	// sixteen rows a device stating none takes, and four hold two.
	sized := map[int]int{}
	for _, regs := range []int{0, 32, 4} {
		d := cuda
		d.Regs = regs
		_, codes, err := jit.Default().Form(jit.Request{Graph: ffnBlock(2048, 8192), Formats: formats(jlm.TypeQ4S),
			Step: jit.Step{Rows: 512, Page: 64, Keys: 1, ActWindow: 32}, Device: d})
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range codes {
			if slices.Contains(c.Passes, "rowtiles") {
				sized[regs] = max(sized[regs], len(c.GPU.Ops))
			}
		}
	}
	if sized[0] == 0 || sized[32] != sized[0] || sized[4] >= sized[32] {
		t.Fatalf("the largest tiled kernel at 512 rows: %d ops with no registers stated, %d at 32 a thread, %d at 4",
			sized[0], sized[32], sized[4])
	}
	t.Logf("the largest tiled kernel at 512 rows: %d ops with no registers stated or 32 a thread, %d at 4", sized[32], sized[4])
}

// tiledWeights is rows of k of t, random, with finite scales near one.
func tiledWeights(t quant.Type, q kernels.Quant, rows, k int, rng *rand.Rand) []byte {
	if t == quant.F32 {
		b := make([]byte, 4*rows*k)
		for i := range rows * k {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(float32(rng.NormFloat64())))
		}
		return b
	}
	f16At := map[kernels.Quant][]int{kernels.Q8_0: {0}, kernels.Q4_K: {0, 2}, kernels.Q5_K: {0, 2}, kernels.Q3_K: {108}, kernels.Q6_K: {208}}
	nblk, bb := k/q.Elems(), q.BlockBytes()
	raw := make([]byte, rows*nblk*bb)
	rng.Read(raw)
	for i := range rows * nblk {
		for _, off := range f16At[q] {
			binary.LittleEndian.PutUint16(raw[i*bb+off:], uint16(0x2000|rng.Intn(0x0C00)))
		}
	}
	return raw
}

// TestComposedGPURowTiles forms a projection of 2048 rows of 256 once at a
// capacity of 512 rows and once at decode's one row, on every device
// present, over a float matrix and every packed format of the served
// models, with int8 and with float activations, and runs the one form at
// every count of rows a call may state -- 1, 3, 7, 31, 33, 100, 511 and 512,
// the rows past them poisoned -- each row held to the oracle and bit for bit
// to its value in a call of the capacity's rows (a call of few rows runs the
// form for few rows where the device's arithmetic is exact, and one must),
// and every word past the call's rows, to the capacity and the guard after
// it, untouched.
// The decode form runs its one row the same way. Three violations must
// fail: the weight's words moved by one, the input's rows moved by one, and
// a call stating one row more than its input holds (the threads past the
// call's rows not held).
func TestComposedGPURowTiles(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	const guard, n, k, capacity = 64, 2048, 256, 512
	calls := []int{1, 3, 7, 31, 33, 100, 511, 512}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(43))
			ran, tiledRan, fewRan := 0, 0, 0
			for _, typ := range tiledFormats {
				jt, ok := jlm.TypeOf(typ)
				q, qok := kernels.QuantOf(typ)
				if typ == quant.F32 {
					jt, q, ok, qok = jlm.TypeF32, kernels.Float32, true, true
				}
				if !ok || !qok {
					t.Fatalf("%v has no container type or kernel format", typ)
				}
				w := tiledWeights(typ, q, n, k, rng)
				qs, dw, sc, err := kernels.PackWeights(q, w, n, k)
				if err != nil {
					t.Fatal(err)
				}
				planes := [3][]uint32{qs, dw, sc}
				moved := [3][]uint32{append(qs[1:len(qs):len(qs)], qs[0]), dw, sc}
				x := make([]float32, capacity*k)
				for i := range x {
					x[i] = float32(rng.NormFloat64() * math.Exp2(float64(rng.Intn(9)-4)))
				}
				for _, float := range []bool{false, true} {
					r := jit.Default()
					if float {
						r = r.WithoutInt8(jit.DeviceGPU)
					}
					want := make([]float32, capacity*n)
					if float || typ == quant.F32 {
						for row := range capacity {
							if err := oracle.MatVec(want[row*n:(row+1)*n], typ, w, x[row*k:(row+1)*k], n, k); err != nil {
								t.Fatal(err)
							}
						}
					} else {
						want = int8Projection(t, typ, w, x, n, k, 32, false)
					}
					for _, R := range []int{1, capacity} {
						name := fmt.Sprintf("%v/float %v/%d rows", typ, float, R)
						b := graph.New()
						in := b.In("x", k)
						out := b.MatVec(in, jlm.RoleAttnQ, n)
						b.Out(out)
						_, formed, err := r.Form(jit.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: jt},
							Step: jit.Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
						if err != nil {
							t.Fatalf("%s: forming: %v", name, err)
						}
						tiled := false
						for _, c := range formed {
							tiled = tiled || slices.Contains(c.Passes, "rowtiles")
						}
						if tiled != (R > 1) {
							t.Fatalf("%s: tiled %v", name, tiled)
						}
						op, err := compileLone(d, formed)
						if err != nil {
							t.Fatalf("%s: %v", name, err)
						}
						// The call's rows, the rest of the capacity poisoned.
						input := func(L int, x []float32) []float32 {
							in := make([]float32, R*k)
							copy(in, x[:L*k])
							for i := L * k; i < len(in); i++ {
								in[i] = math.Float32frombits(guardWord)
							}
							return in
						}
						run := func(L, stated int, x []float32, pl [3][]uint32) ([]float32, bool) {
							rows := stated
							if R == 1 {
								rows = 0
							}
							res, tail, err := runGraphRows(d, formed, op.kerns, in, out, input(L, x), R*n, guard, pl, rows)
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
						few := 0
						for _, c := range formed {
							few = max(few, c.Few)
						}
						if few == 0 && dev.ExactArith && R > 1 {
							t.Fatalf("%s: no form for few rows on a device whose arithmetic is exact", name)
						}
						full, _ := run(R, R, x, planes)
						for _, L := range calls {
							if L > R {
								continue
							}
							got, clean := run(L, L, x, planes)
							for i, v := range got {
								if math.Float32bits(v) != math.Float32bits(full[i]) {
									t.Fatalf("%s: a call of %d rows writes row %d's element %d as %08x, a call of %d rows as %08x",
										name, L, i/n, i%n, math.Float32bits(v), R, math.Float32bits(full[i]))
								}
							}
							if R > 1 && L <= few {
								fewRan++
							}
							if !clean {
								t.Fatalf("%s: a call of %d rows wrote past them", name, L)
							}
							for row := range L {
								if e := gpuNMSE(got[row*n:(row+1)*n], want[row*n:(row+1)*n]); math.IsNaN(e) || !(e < 1e-9) {
									t.Fatalf("%s: a call of %d rows: row %d reads NMSE %.3e against the oracle", name, L, row, e)
								}
							}
							ran++
							if tiled {
								tiledRan++
							}
						}
						L := min(R, 33)
						if got, _ := run(L, L, x, moved); !(gpuNMSE(got, want[:L*n]) > 1e-6) {
							t.Fatalf("%s: with the weight's words moved by one the NMSE is %.3e", name, gpuNMSE(got, want[:L*n]))
						}
						if R > 1 {
							rot := append(slices.Clone(x[k:L*k]), x[:k]...)
							if got, _ := run(L, L, rot, planes); !(gpuNMSE(got, want[:L*n]) > 1e-6) {
								t.Fatalf("%s: with the input's rows moved by one the NMSE is %.3e", name, gpuNMSE(got, want[:L*n]))
							}
							if _, clean := run(L, L+1, x, planes); clean {
								t.Fatalf("%s: a call stating %d rows over %d wrote nothing past them: the gate cannot see a thread not held",
									name, L+1, L)
							}
						}
						op.close()
					}
				}
			}
			if dev.ExactArith && fewRan == 0 {
				t.Fatalf("%s: no call ran a form for few rows", d.API())
			}
			t.Logf("%s: %d calls right against the oracle and bit for bit against the capacity's call, %d of them tiled, %d on the form for few rows, every violation seen", d.API(), ran, tiledRan, fewRan)
		})
	}
}
