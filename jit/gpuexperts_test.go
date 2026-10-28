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

// TestComposedGPUExperts forms an expert matvec at a prompt chunk's rows on
// every device whose matrix unit takes binary16 tiles, and on a CUDA card
// lowered for sm_70: the chunk's pairs grouped by the expert their
// selection names and the grouped rows projected on the matrix unit, over
// the packed formats of the served mixtures, a selection that leaves
// experts without a pair and gives one most of them, at calls of 1 to the
// capacity's rows. Each pair must match the oracle's matvec of its row
// against its expert's sheet, a call must write no pair of a row past its
// own and nothing past the result, and a kernel must multiply tiles. Two
// violations must fail: every pair's expert moved by one, and a pair's
// result against the next pair's reference.
func TestComposedGPUExperts(t *testing.T) {
	testlock.GPU(t)
	devs, closeAll := matrixDevices(t)
	defer closeAll()
	if len(devs) == 0 {
		t.Skip("no device here multiplies binary16 tiles")
	}
	const guard, capacity, E, P, K, rows = 64, 100, 8, 2, 256, 96
	calls := []int{1, 17, 64, capacity}
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
			rng := rand.New(rand.NewSource(71))
			ran := 0
			for _, typ := range []quant.Type{quant.Q8_0, quant.Q4_K, quant.Q6_K, quant.MXFP4} {
				what := fmt.Sprintf("%v", typ)
				jt, ok := jlm.TypeOf(typ)
				q, qok := kernels.QuantOf(typ)
				if !ok || !qok {
					t.Fatalf("%v has no container type or kernel format", typ)
				}
				sheets := make([][]byte, E)
				var planes [3][]uint32
				for e := range sheets {
					sheets[e] = expertWeights(typ, q, rows, K, rng)
					qs, dw, sc, err := kernels.PackWeights(q, sheets[e], rows, K)
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
				x := b.In("x", P*K)
				sel := b.In("sel", P)
				b.G.Values[sel].Elem = graph.ElemU32
				y := b.Emit(graph.OpExpertMatVec, []graph.ValueID{x, sel}, []graph.Result{{Name: "y", Width: P * rows}},
					[]graph.Binding{{Role: jlm.RoleExpGateBank, K: K, Rows: rows, Sheets: E}}, graph.Attrs{}, nil, nil)
				b.Out(y[0])
				_, formed, err := jit.Default().Form(jit.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleExpGateBank: jt},
					Step: jit.Step{Rows: capacity, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
				if err != nil {
					t.Fatalf("%s: forming: %v", what, err)
				}
				tiles := 0
				for _, c := range formed {
					for _, o := range c.GPU.Ops {
						if o.Kind == ir.OpTileMMA {
							tiles++
						}
					}
				}
				if tiles == 0 {
					t.Fatalf("%s: no kernel was formed on the matrix unit", what)
				}
				op, err := compileLone(d, formed)
				if err != nil {
					t.Fatalf("%s: %v", what, err)
				}
				xs := make([]float32, capacity*P*K)
				for i := range xs {
					xs[i] = float32(rng.NormFloat64())
				}
				// Experts 5 and 6 take no pair; expert 2 takes most.
				pick := []uint32{0, 1, 2, 2, 2, 2, 3, 4, 7, 2}
				sels := make([]uint32, capacity*P)
				for i := range sels {
					sels[i] = pick[rng.Intn(len(pick))]
				}
				want := func(shift uint32) []float32 {
					out := make([]float32, capacity*P*rows)
					for s := range capacity * P {
						e := (sels[s] + shift) % E
						if err := oracle.MatVec(out[s*rows:(s+1)*rows], typ, sheets[e], xs[s*K:(s+1)*K], rows, K); err != nil {
							t.Fatal(err)
						}
					}
					return out
				}
				ref := want(0)
				run := func(L int) []float32 {
					n := capacity * P * rows
					res, tail, err := runExperts(d, formed, op.kerns, x, sel, y[0], xs, sels, n, guard, planes, L)
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
				worst := 0.0
				for _, L := range calls {
					e := gpuNMSE(run(L), ref[:L*P*rows])
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
			t.Logf("%s: %d expert matvecs grouped onto the matrix unit, held to the oracle at calls of %v rows", name, ran, calls)
		})
	}
}

// runExperts runs codes over input x, selection sel and the bank's planes
// at a call of rows rows, and reads n words of the result and guard words
// past it.
func runExperts(d backend.Device, codes []jit.Code, kerns []backend.Kernel, in, sel, out graph.ValueID, x []float32, sels []uint32, n, guard int,
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
	reach := map[jit.Operand]int64{}
	for _, c := range codes {
		for _, bd := range c.Binds {
			reach[bd.Operand] = max(reach[bd.Operand], bd.Bytes)
		}
	}
	own := map[jit.Operand]backend.Buf{}
	for ci, c := range codes {
		if !c.RunsAt(rows) {
			continue
		}
		args := make([]backend.Buf, len(c.Binds))
		for i, bd := range c.Binds {
			o := bd.Operand
			switch {
			case o.Kind == jit.OperandValue && o.Value == in:
				args[i] = inBuf
			case o.Kind == jit.OperandValue && o.Value == sel:
				args[i] = selBuf
			case o.Kind == jit.OperandValue && o.Value == out:
				args[i] = outBuf
			case o.Kind == jit.OperandWeight && o.Region >= 0 && o.Region < 3:
				b, ok := own[o]
				if !ok {
					if b, err = alloc(planes[o.Region]); err != nil {
						return nil, nil, err
					}
					own[o] = b
				}
				args[i] = b
			case o.Kind == jit.OperandWeight || o.Kind == jit.OperandVector:
				return nil, nil, fmt.Errorf("%s binds %+v, which the gate does not hold", c.Name, o)
			case o.Kind == jit.OperandTable && o.Name == "rows.n":
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
func expertWeights(t quant.Type, q kernels.Quant, rows, k int, rng *rand.Rand) []byte {
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
