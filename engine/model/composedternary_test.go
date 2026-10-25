package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/cpu"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/x86"
)

// The statement of a projection over int8 inputs per token (graph's
// Int8Codes, the product over the codes, the row's scale after), executed as
// the composer forms it for the host's CPU over ternary weights and held to
// the oracle: the oracle's quantizer over the whole row, its matvec over the
// codes, times the row's scale.

// ternaryWeights is rows rows of k random ternary blocks of t, each block's
// scale near one.
func ternaryWeights(t quant.Type, rows, k int, rng *rand.Rand) []byte {
	bb := int(t.BlockBytes())
	raw := make([]byte, rows*k/256*bb)
	rng.Read(raw)
	for at := 0; at < len(raw); at += bb {
		if t == quant.TQ2_F32 {
			binary.LittleEndian.PutUint32(raw[at+64:], math.Float32bits(0.5+rng.Float32()))
			continue
		}
		binary.LittleEndian.PutUint16(raw[at+64:], uint16(0x3800|rng.Intn(0x0400)))
	}
	return raw
}

// perRowWant is the oracle's projection of each row of k of x over w: the
// row's int8 form over windows of window (the oracle's quantizer), its matvec
// over the form, a window as wide as the row the codes themselves, times the
// row's scale after; with even, the codes rounded half to even.
func perRowWant(t *testing.T, typ quant.Type, w []byte, x []float32, rows, k, window int, even bool) []float32 {
	t.Helper()
	tokens := len(x) / k
	want := make([]float32, tokens*rows)
	codes := make([]int8, k)
	pairs := make([]float32, 2*k/32)
	row := make([]float32, k)
	for tk := range tokens {
		oracle.QuantizeQ8Window(codes, pairs, x[tk*k:(tk+1)*k], 0, 0, k/32, window)
		for i, c := range codes {
			row[i] = float32(c)
			if d := pairs[2*(i/32)]; even && d != 0 {
				row[i] = float32(math.RoundToEven(float64(x[tk*k+i] * (1 / d))))
			}
			if window != k {
				row[i] *= pairs[2*(i/32)]
			}
		}
		out := want[tk*rows : (tk+1)*rows]
		if err := oracle.MatVec(out, typ, w, row, rows, k); err != nil {
			t.Fatal(err)
		}
		if window == k {
			for i := range out {
				out[i] *= pairs[0]
			}
		}
	}
	return want
}

// TestComposedTernaryMatVecPerToken forms the statement for the CPU over
// TQ2_0 and TQ2_F32 weights at ragged row counts, two widths, one token and
// three, through the decoded form the CPU's profile ships and through the
// int8 form (the codes' integer dot with the weight's ternary fields), each
// held to the oracle with guard words after the output. The int8 arm must
// take the integer path (its codes packed into words) and the decoded arm
// must not, and three violations must fail: the weight's words moved by one,
// the reference over 32-wide windows rather than the row, and the reference
// rounding the codes half to even. A shape the CPU's lowering does not emit
// is its work list, named; what forms must be right. The device's form
// (TestComposedGPUTernaryPerToken, jit) forms every shape.
func TestComposedTernaryMatVecPerToken(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	int8Reg := jit.Default()
	p := int8Reg.Profiles[jit.DeviceCPU]
	p.Int8 = &jit.Int8Form{Window: 32, Word: 4, Sums: 16, UnsignedDot: uint64(x86.ISAVNNI)}
	int8Reg.Profiles[jit.DeviceCPU] = p
	arms := []struct {
		name string
		reg  *jit.Registry
		int8 bool
	}{
		{"decoded", jit.Default().WithoutInt8(jit.DeviceCPU), false},
		{"int8", int8Reg, true},
	}
	right, evens, unformed := map[string]int{}, map[string]int{}, map[string]string{}
	for _, typ := range []quant.Type{quant.TQ2_0, quant.TQ2_F32} {
		jt, ok := jlm.TypeOf(typ)
		q, qok := kernels.QuantOf(typ)
		if !ok || !qok {
			t.Fatalf("%v has no container type or kernel format", typ)
		}
		for _, k := range []int{256, 768} {
			for _, rows := range []int{1, 2, 3, 5, 7, 8, 9, 17, 33} {
				w := ternaryWeights(typ, rows, k, rng)
				qs, d, sc, err := kernels.PackWeights(q, w, rows, k)
				if err != nil {
					t.Fatal(err)
				}
				planes := map[jlm.Role][3][]uint32{jlm.RoleAttnQ: {qs, d, sc}}
				moved := map[jlm.Role][3][]uint32{jlm.RoleAttnQ: {append(qs[1:len(qs):len(qs)], qs[0]), d, sc}}
				for _, tokens := range []int{1, 3} {
					x := make([]float32, tokens*k)
					for i := range x {
						x[i] = float32(rng.NormFloat64() * math.Exp2(float64(rng.Intn(9)-4)))
					}
					// The row's scale one, and codes exactly between two
					// integers that half to even rounds the other way.
					x[0] = 127
					for j := 1; j <= 16; j++ {
						x[j] = float32(2*j) + 0.5
						if j%2 == 0 {
							x[j] = -x[j]
						}
					}
					want := perRowWant(t, typ, w, x, rows, k, k, false)
					b := graph.New()
					in := b.In("x", k)
					codes, scale := b.Int8Codes(in)
					y := b.MatVec(codes, jlm.RoleAttnQ, rows)
					out := b.Emit(graph.OpMap, []graph.ValueID{y, scale}, []graph.Result{{Name: "y", Width: rows}}, nil,
						graph.Attrs{Fn: graph.MapMul}, nil, nil)[0]
					b.Out(out)
					for _, arm := range arms {
						name := fmt.Sprintf("%v/k %d/%d rows/%d tokens/%s", typ, k, rows, tokens, arm.name)
						reg := arm.reg
						pr := reg.Profiles[jit.DeviceCPU]
						pr.Off = map[string]bool{"inplace": true}
						reg.Profiles[jit.DeviceCPU] = pr
						_, formed, err := reg.Form(jit.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: jt},
							Step:   jit.Step{Rows: tokens, Page: 64, Keys: 1, ActWindow: 32},
							Device: jit.Device{Kind: jit.DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}})
						if err != nil {
							unformed[name] = err.Error()
							continue
						}
						packs := false
						for _, c := range formed {
							for _, bd := range c.Binds {
								packs = packs || strings.HasSuffix(bd.Operand.Name, ".q8a")
							}
						}
						if packs != arm.int8 {
							t.Fatalf("%s: the codes packed into words: %v, want %v", name, packs, arm.int8)
						}
						run := func(pl map[jlm.Role][3][]uint32) []float32 {
							row, back := guarded(tokens * rows)
							cpuExec(t, formed, nil, pl, map[graph.ValueID][]float32{in: x, out: row})
							checkGuard(t, name, back, tokens*rows)
							return row
						}
						got := run(planes)
						if e := rowsNMSE(got, want); math.IsNaN(e) || !(e < 1e-9) {
							t.Fatalf("%s: NMSE %.3e against the oracle", name, e)
						}
						if e := rowsNMSE(got, perRowWant(t, typ, w, x, rows, k, 32, false)); !(e > 1e-6) {
							t.Fatalf("%s: the reference over 32-wide windows reads %.3e: the gate cannot see the window", name, e)
						}
						// The ties' products with a row's ternary weights can
						// cancel: where the two references agree there is
						// nothing to see, and the gate counts where they do not.
						if even := perRowWant(t, typ, w, x, rows, k, k, true); rowsNMSE(even, want) > 1e-9 {
							if e := rowsNMSE(got, even); !(e > 1e-9) {
								t.Fatalf("%s: the reference rounding half to even reads %.3e: the gate cannot see the rounding", name, e)
							}
							evens[arm.name]++
						}
						if e := rowsNMSE(run(moved), want); !(e > 1e-6) {
							t.Fatalf("%s: with the weight's words moved by one the NMSE is %.3e", name, e)
						}
						right[arm.name]++
					}
				}
			}
		}
	}
	for name, why := range unformed {
		t.Logf("work list: %s: %s", name, why)
	}
	t.Logf("formed, right and seeing the violations: %v, the rounding at %v; %d not formed", right, evens, len(unformed))
	for _, arm := range arms {
		if right[arm.name] == 0 {
			t.Fatalf("the %s arm formed nothing right: this gate proved nothing of it", arm.name)
		}
		if evens[arm.name] < right[arm.name]/2 {
			t.Fatalf("the %s arm: the rounding's violation was visible at %d of %d shapes", arm.name, evens[arm.name], right[arm.name])
		}
	}
}

// quantMaps are the maps the int8 statement adds -- |x|, x rounded half away
// from zero, 1/x, x/127 -- each with Go's own result and a violation of it
// the values must tell apart.
var quantMaps = []struct {
	fn    graph.MapFn
	k     float64
	want  func(float32) float32
	wrong func(float32) float32
}{
	{graph.MapAbs, 0, func(v float32) float32 { return float32(math.Abs(float64(v))) }, func(v float32) float32 { return v }},
	{graph.MapRound, 0, func(v float32) float32 { return float32(math.Round(float64(v))) },
		func(v float32) float32 { return float32(math.RoundToEven(float64(v))) }},
	{graph.MapRecip, 0, func(v float32) float32 { return 1 / v }, func(v float32) float32 { return 1 / (v * 1.0000001) }},
	{graph.MapDiv, 127, func(v float32) float32 { return v / 127 }, func(v float32) float32 { return v * (1.0 / 127) }},
}

// TestComposedQuantMapsMatchTheReference runs quantMaps on the CPU as the
// composer forms them, over sweep values, the rounding's halves and its 2^23
// edge, each held to Go's own result bit for bit. The CPU's division is
// exact; the integer forms a device without one takes are held on the GPU
// (TestComposedGPUQuantMaps, jit).
func TestComposedQuantMapsMatchTheReference(t *testing.T) {
	xs := sweep(23, 1e30)
	xs = append(xs, 0.5, 1.5, 2.5, -0.5, -2.5, 0.49999997, -0.49999997, 126.5, 127.49999, 8388607.5, 8388608, 16777217, 1e-40)
	for _, cs := range quantMaps {
		name := cs.fn.String()
		b := graph.New()
		x := b.In("x", len(xs))
		y := mapOver(b, "y", graph.Attrs{Fn: cs.fn, Const: cs.k}, nil, x)
		b.Out(y)
		reg := jit.Default().WithoutInt8(jit.DeviceCPU)
		p := reg.Profiles[jit.DeviceCPU]
		p.Off = map[string]bool{"inplace": true}
		reg.Profiles[jit.DeviceCPU] = p
		_, codes, err := reg.Form(jit.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{},
			Step:   jit.Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32},
			Device: jit.Device{Kind: jit.DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.CPU().ISA()), ExactDiv: true}})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		row, back := guarded(len(xs))
		cpuExec(t, codes, nil, nil, map[graph.ValueID][]float32{x: xs, y: row})
		checkGuard(t, name, back, len(xs))
		bad, seen := 0, false
		for i, v := range xs {
			w := cs.want(v)
			if math.Float32bits(row[i]) != math.Float32bits(w) {
				bad++
				t.Errorf("%s(%g) = %g (%#x), want %g (%#x)", name, v, row[i], math.Float32bits(row[i]), w, math.Float32bits(w))
			}
			seen = seen || math.Float32bits(cs.wrong(v)) != math.Float32bits(w)
		}
		if !seen {
			t.Errorf("%s: no value tells the reference from its violation: the sweep cannot see one", name)
		}
		if bad > 0 {
			t.Fatalf("%s: %d of %d values off", name, bad, len(xs))
		}
	}
}
