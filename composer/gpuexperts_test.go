package composer_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
)

// TestComposedGPUExperts forms an expert matvec at a prompt chunk's rows on
// every device whose matrix unit takes binary16 tiles, and on a CUDA card
// lowered for sm_70: the chunk's pairs grouped by the expert their
// selection names and the grouped rows projected on the matrix unit, over
// the packed formats of the served mixtures, a selection that leaves
// experts without a pair and gives one most of them, at calls of 1 to the
// capacity's rows, and a bank of 64 experts at 512 rows, whose grouping no
// call's rows may cut. Each pair must match the oracle's matvec of its row
// against its expert's sheet, a call must write no pair of a row past its
// own and nothing past the result, and a kernel must multiply tiles. The
// same 64 experts at 64 rows of two pairs would pad each pair to 33
// grouped rows, past groupSlack: the per-pair form runs there, multiplying
// no tile, held to the oracle the same way. Two violations must fail:
// every pair's expert moved by one, and a pair's result against the next
// pair's reference.
func TestComposedGPUExperts(t *testing.T) {
	testlock.GPU(t)
	devs, closeAll := matrixDevices(t)
	defer closeAll()
	if len(devs) == 0 {
		t.Skip("no device here multiplies binary16 tiles")
	}
	const guard, P, rows = 64, 2, 96
	names := slices.Sorted(func(yield func(string) bool) {
		for n := range devs {
			if !yield(n) {
				return
			}
		}
	})
	for _, name := range names {
		d := devs[name]
		for _, bank := range []struct {
			capacity, E, K int
			tiled, grouped bool
		}{{100, 8, 256, false, true}, {512, 64, 256, false, true}, {64, 64, 256, false, false}, {100, 8, 2048, false, true},
			{100, 8, 256, true, true}} {
			capacity, E, K, tiled := bank.capacity, bank.E, bank.K, bank.tiled
			calls := []int{1, 17, 64}
			if capacity != 64 {
				calls = append(calls, capacity)
			}
			t.Run(fmt.Sprintf("%s/%dx%dx%d/tiled=%v", name, capacity, E, K, tiled), func(t *testing.T) {
				dev := gpuDevice(d)
				dev.Matrix = [3]int{matrixTile().M, matrixTile().N, matrixTile().K}
				rng := rand.New(rand.NewSource(71))
				ran := 0
				for _, typ := range []quant.Type{quant.Q8_0, quant.Q4_K, quant.Q6_K, quant.MXFP4} {
					jt, ok := jlm.TypeOf(typ)
					q, qok := layout.QuantOf(typ)
					if !ok || !qok {
						t.Fatalf("%v has no container type or kernel format", typ)
					}
					sheets := make([][]byte, E)
					var planes [3][]uint32
					for e := range sheets {
						sheets[e] = expertWeights(typ, q, rows, K, rng)
						qs, dw, sc, err := layout.PackWeights(q, sheets[e], rows, K)
						if err != nil {
							t.Fatal(err)
						}
						planes[0], planes[1], planes[2] = append(planes[0], qs...), append(planes[1], dw...), append(planes[2], sc...)
					}
					for i := range planes {
						if len(planes[i]) == 0 {
							planes[i] = []uint32{0}
						}
					}
					b := graph.New()
					var x, in graph.ValueID
					if tiled {
						x = b.In("x", K)
						in = b.Emit(graph.OpTile, []graph.ValueID{x}, []graph.Result{{Name: "x.tiled", Width: P * K}}, nil, graph.Attrs{Span: K}, nil, nil)[0]
					} else {
						x = b.In("x", P*K)
						in = x
					}
					sel := b.In("sel", P)
					b.G.Values[sel].Elem = graph.ElemU32
					y := b.Emit(graph.OpExpertMatVec, []graph.ValueID{in, sel}, []graph.Result{{Name: "y", Width: P * rows}},
						[]graph.Binding{{Role: jlm.RoleExpGateBank, K: K, Rows: rows, Sheets: E}}, graph.Attrs{}, nil, nil)
					b.Out(y[0])
					for _, tile := range gpuMatrixTiles() {
						td := dev
						td.Tile = tile
						if tiled {
							// The halves kernel's, which reads the tile's rows.
							td.Tile.Convert = -1
						}
						what := fmt.Sprintf("%v at %v", typ, tile)
						_, formed, err := poisonedRegistry().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleExpGateBank: jt},
							Step: composer.Step{Rows: capacity, Page: 64, Keys: 1, ActWindow: 32}, Device: td})
						if err != nil {
							t.Fatalf("%s: forming: %v", what, err)
						}
						tiles, cuts, byRow := 0, 0, 0
						for _, c := range formed {
							if strings.HasSuffix(c.Name, ".combine") {
								cuts++
							}
							if strings.HasSuffix(c.Name, ".halves") && c.Groups*c.GPU.Group[0] == capacity*K/2 {
								byRow++
							}
							if c.Matrix != [2]int{} && !threadsAtTile(c, tile) {
								t.Fatalf("%s: the matrix kernel's workgroup of %d threads is not the tile's", what, c.GPU.Group[0])
							}
							for _, o := range c.GPU.Ops {
								if o.Kind == ir.OpTileMMA || o.Kind == ir.OpMMA {
									tiles++
								}
							}
						}
						if bank.grouped && tiles == 0 {
							t.Fatalf("%s: no kernel was formed on the matrix unit", what)
						}
						if !bank.grouped && tiles > 0 {
							t.Fatalf("%s: %d experts at %d rows grouped onto the matrix unit past groupSlack", what, E, capacity)
						}
						if tiled && byRow == 0 {
							t.Fatalf("%s: the tiled input's halves are not its %d rows'", what, capacity)
						}
						if K > 256 && cuts == 0 {
							t.Fatalf("%s: k of %d over %d experts' blocks: the model cut no projection", what, K, E)
						}
						op, err := compileLone(d, formed)
						if err != nil {
							t.Fatalf("%s: %v", what, err)
						}
						xs := make([]float32, capacity*P*K)
						for i := range xs {
							xs[i] = float32(rng.NormFloat64())
						}
						xin := xs
						if tiled {
							xin = make([]float32, capacity*K)
							for r := range capacity {
								copy(xin[r*K:(r+1)*K], xs[r*P*K:r*P*K+K])
								for j := 1; j < P; j++ {
									copy(xs[(r*P+j)*K:(r*P+j+1)*K], xin[r*K:(r+1)*K])
								}
							}
						}
						// Experts 5 and 6 take no pair; expert 2 takes most.
						pick := []uint32{0, 1, 2, 2, 2, 2, 3, 4, 7, 2}
						if E > 8 {
							pick = append(pick, uint32(E/2), uint32(E-1))
						}
						sels := make([]uint32, capacity*P)
						for i := range sels {
							sels[i] = pick[rng.Intn(len(pick))]
						}
						want := func(shift uint32) []float32 {
							out := make([]float32, capacity*P*rows)
							for s := range capacity * P {
								e := (sels[s] + shift) % uint32(E)
								if err := oracle.MatVec(out[s*rows:(s+1)*rows], typ, sheets[e], xs[s*K:(s+1)*K], rows, K); err != nil {
									t.Fatal(err)
								}
							}
							return out
						}
						ref := want(0)
						run := func(L int) []float32 {
							n := capacity * P * rows
							res, tail, err := runExperts(d, formed, op.kerns, x, sel, y[0], xin, sels, n, guard, planes, L)
							if err != nil {
								t.Fatalf("%s: %v", what, err)
							}
							for i, v := range res[L*P*rows:] {
								if math.Float32bits(v) != guardWord {
									t.Fatalf("%s: a call of %d rows wrote row %d's element %d", what, L, L+i/(P*rows), i%(P*rows))
								}
							}
							for i, v := range tail {
								if v != guardWord {
									t.Fatalf("%s: the guard word %d past the result was written", what, i)
								}
							}
							return res[:L*P*rows]
						}
						// The per-pair form is the int8 matvec a pair at a time
						// (expertInt8) over the step's activation window where the
						// format takes one, else the decoded sheets': held to
						// whichever of the two references it computes.
						ref8 := ref
						if !bank.grouped {
							ref8 = make([]float32, len(ref))
							for s := range capacity * P {
								copy(ref8[s*rows:], int8Projection(t, typ, sheets[sels[s]], xs[s*K:(s+1)*K], rows, K, 32, false))
							}
						}
						worst := 0.0
						for _, L := range calls {
							got := run(L)
							e := min(gpuNMSE(got, ref[:L*P*rows]), gpuNMSE(got, ref8[:L*P*rows]))
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
						got := run(capacity)
						if e := gpuNMSE(got, want(1)); !(e > 1e-2) {
							t.Fatalf("%s: against every pair's next expert the NMSE is %.3e: the gate cannot see the experts", what, e)
						}
						if e := gpuNMSE(got[:len(got)-rows], ref[rows:]); !(e > 1e-2) {
							t.Fatalf("%s: against the next pair's reference the NMSE is %.3e: the gate cannot see the pairs", what, e)
						}
						op.close()
						ran++
					}
				}
				t.Logf("%s: %d expert matvecs (grouped onto the matrix unit: %v) held to the oracle at calls of %v rows", name, ran,
					bank.grouped, calls)
			})
		}
	}
}

// runExperts runs codes over input x, selection sel and the bank's planes
// at a call of rows rows, and reads n words of the result and guard words
// past it.
func runExperts(d backend.Device, codes []composer.Code, kerns []backend.Kernel, in, sel, out graph.ValueID, x []float32, sels []uint32, n, guard int,
	planes [3][]uint32, rows int) (res []float32, tail []uint32, err error) {
	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	alloc := func(w []uint32) (backend.Buf, error) {
		b, err := d.Alloc(max(4*len(w), 4))
		if err != nil {
			return nil, err
		}
		bufs = append(bufs, b)
		p := make([]byte, 4*len(w))
		for i, v := range w {
			binary.LittleEndian.PutUint32(p[4*i:], v)
		}
		if len(p) > 0 {
			err = b.Write(p)
		}
		return b, err
	}
	xw := make([]uint32, 0, len(x)+guard)
	for _, v := range x {
		xw = append(xw, math.Float32bits(v))
	}
	inBuf, err := alloc(append(xw, poisoned(guard)...))
	if err != nil {
		return nil, nil, err
	}
	selBuf, err := alloc(append(slices.Clone(sels), poisoned(guard)...))
	if err != nil {
		return nil, nil, err
	}
	outBuf, err := alloc(poisoned(n + guard))
	if err != nil {
		return nil, nil, err
	}
	reach := map[composer.Operand]int64{}
	for _, c := range codes {
		for _, bd := range c.Binds {
			reach[bd.Operand] = max(reach[bd.Operand], bd.Bytes)
		}
	}
	own := map[composer.Operand]backend.Buf{}
	for ci, c := range codes {
		if !c.RunsAt(rows) {
			continue
		}
		args := make([]backend.Buf, len(c.Binds))
		for i, bd := range c.Binds {
			o := bd.Operand
			switch {
			case o.Kind == composer.OperandValue && o.Value == in:
				args[i] = inBuf
			case o.Kind == composer.OperandValue && o.Value == sel:
				args[i] = selBuf
			case o.Kind == composer.OperandValue && o.Value == out:
				args[i] = outBuf
			case o.Kind == composer.OperandWeight && o.Region >= 0 && o.Region < 3:
				b, ok := own[o]
				if !ok {
					if b, err = alloc(planes[o.Region]); err != nil {
						return nil, nil, err
					}
					own[o] = b
				}
				args[i] = b
			case o.Kind == composer.OperandWeight || o.Kind == composer.OperandVector:
				return nil, nil, fmt.Errorf("%s binds %+v, which the gate does not hold", c.Name, o)
			case o.Kind == composer.OperandTable && o.Name == "rows.n":
				b, ok := own[o]
				if !ok {
					if b, err = alloc(append([]uint32{uint32(rows)}, poisoned(guard)...)); err != nil {
						return nil, nil, err
					}
					own[o] = b
				}
				args[i] = b
			default:
				b, ok := own[o]
				if !ok {
					w := bd.Words
					if w == nil {
						w = poisoned(int(max(reach[o], 4)/4) + guard)
					}
					if b, err = alloc(w); err != nil {
						return nil, nil, err
					}
					own[o] = b
				}
				args[i] = b
			}
		}
		if err := kerns[ci].Launch(c.LiveGroups(rows), c.GPU.Group[0], args...); err != nil {
			return nil, nil, fmt.Errorf("launching %s: %w", c.Name, err)
		}
	}
	raw := make([]byte, 4*(n+guard))
	if err := outBuf.Read(raw); err != nil {
		return nil, nil, err
	}
	words := make([]uint32, n+guard)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(raw[4*i:])
	}
	res = make([]float32, n)
	for i := range res {
		res[i] = math.Float32frombits(words[i])
	}
	return res, words[n:], nil
}

// expertWeights is matrixWeights, an MXFP4 block's scale held near one: a
// random E8M0 exponent overflows binary16.
func expertWeights(t quant.Type, q layout.Quant, rows, k int, rng *rand.Rand) []byte {
	if t != quant.MXFP4 {
		return matrixWeights(t, q, rows, k, rng)
	}
	b := make([]byte, rows*k/q.Elems()*q.BlockBytes())
	rng.Read(b)
	for i := 0; i < len(b); i += q.BlockBytes() {
		b[i] = byte(126 + rng.Intn(3))
	}
	return b
}

// threadsAtTile is whether c's workgroup is tile's warps, those along its
// rows halved where its rows or the device's workgroup memory take fewer.
func threadsAtTile(c composer.Code, t composer.MatrixTile) bool {
	for wr := t.Warps[0]; wr >= 1; wr /= 2 {
		if c.GPU.Group[0] == 32*wr*t.Warps[1] {
			return true
		}
	}
	return false
}
