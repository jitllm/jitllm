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
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
)

// ternaryRows is rows rows of k random ternary blocks of t, each block's
// scale near one.
func ternaryRows(t quant.Type, rows, k int, rng *rand.Rand) []byte {
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

// int8Projection is the oracle's projection of each row of k of x over w:
// the row's int8 form over windows of window (the oracle's quantizer), its
// matvec over the form, a window as wide as the row the codes themselves,
// times the row's scale after; with even, the row's codes rounded half to
// even.
func int8Projection(t *testing.T, typ quant.Type, w []byte, x []float32, rows, k, window int, even bool) []float32 {
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

// runGraph launches codes in order: the graph's input x, its output of n
// floats followed by guard words, the weight's planes by region, and every
// other buffer -- the graph's values between, the composer's scratch --
// poisoned, constant blocks written. It returns the result and the words
// after it.
func runGraph(d backend.Device, codes []composer.Code, kerns []backend.Kernel, in, out graph.ValueID, x []float32, n, guard int,
	planes [3][]uint32) (res []float32, tail []uint32, err error) {
	return runGraphRows(d, codes, kerns, in, out, x, n, guard, planes, 0)
}

// runGraphRows is runGraph at a call of rows rows, the call's table's word,
// each kernel launched at its groups for them
// (Code.LiveGroups); 0 leaves the table poisoned and launches the form's
// capacity.
func runGraphRows(d backend.Device, codes []composer.Code, kerns []backend.Kernel, in, out graph.ValueID, x []float32, n, guard int,
	planes [3][]uint32, rows int) (res []float32, tail []uint32, err error) {
	return runGraphRowsWith(d, codes, kerns, in, out, x, n, guard, planes, rows, nil)
}

// runGraphRowsWith is runGraphRows with the graph's other inputs' words,
// each followed by guard poisoned words.
func runGraphRowsWith(d backend.Device, codes []composer.Code, kerns []backend.Kernel, in, out graph.ValueID, x []float32, n, guard int,
	planes [3][]uint32, rows int, inputs map[graph.ValueID][]uint32) (res []float32, tail []uint32, err error) {
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
			case o.Kind == composer.OperandValue && o.Value == out:
				args[i] = outBuf
			case o.Kind == composer.OperandValue && inputs[o.Value] != nil:
				b, ok := own[o]
				if !ok {
					if b, err = alloc(append(slices.Clone(inputs[o.Value]), poisoned(guard)...)); err != nil {
						return nil, nil, err
					}
					own[o] = b
				}
				args[i] = b
			case o.Kind == composer.OperandWeight && o.Region >= 0 && o.Region < 3 && len(planes[o.Region]) > 0:
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
			case rows > 0 && o.Kind == composer.OperandTable && o.Name == "rows.n":
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
		groups := c.Groups
		if rows > 0 {
			groups = c.LiveGroups(rows)
		}
		if err := kerns[ci].Launch(groups, c.GPU.Group[0], args...); err != nil {
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

// TestComposedGPUTernaryPerToken forms a projection over int8 inputs per
// token (graph's Int8Codes, the product over the codes, times the row's
// scale) on every device present, over TQ2_0 and TQ2_F32 weights at ragged
// row counts, two widths, one token and three, and holds it to the oracle
// with poisoned guard words after the output. The product must be the
// integer one (the codes packed into words against the weight's ternary
// fields, no weight decoded to a float), and three violations must fail: the
// weight's words moved by one, the reference over 32-wide windows rather
// than the row, and the reference rounding the row's codes half to even.
func TestComposedGPUTernaryPerToken(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	const guard = 64
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			r := composer.Default()
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(31))
			ran, evens := 0, 0
			for _, typ := range []quant.Type{quant.TQ2_0, quant.TQ2_F32} {
				jt, ok := jlm.TypeOf(typ)
				q, qok := layout.QuantOf(typ)
				if !ok || !qok {
					t.Fatalf("%v has no container type or kernel format", typ)
				}
				for _, k := range []int{256, 768} {
					for _, rows := range []int{1, 2, 3, 5, 7, 8, 9, 17, 33, 129} {
						w := ternaryRows(typ, rows, k, rng)
						qs, dw, sc, err := layout.PackWeights(q, w, rows, k)
						if err != nil {
							t.Fatal(err)
						}
						for _, tokens := range []int{1, 3} {
							name := fmt.Sprintf("%v/k %d/%d rows/%d tokens", typ, k, rows, tokens)
							b := graph.New()
							in := b.In("x", k)
							codes, scale := b.Int8Codes(in)
							y := b.MatVec(codes, jlm.RoleAttnQ, rows)
							out := b.Emit(graph.OpMap, []graph.ValueID{y, scale}, []graph.Result{{Name: "y", Width: rows}}, nil,
								graph.Attrs{Fn: graph.MapMul}, nil, nil)[0]
							b.Out(out)
							_, formed, err := r.Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: jt},
								Step: composer.Step{Rows: tokens, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
							if err != nil {
								t.Fatalf("%s: forming: %v", name, err)
							}
							packs := false
							for _, c := range formed {
								for _, bd := range c.Binds {
									packs = packs || strings.HasSuffix(bd.Operand.Name, ".q8a")
								}
							}
							if !packs {
								t.Fatalf("%s: the codes are not packed into words: the product is not the integer one", name)
							}
							op, err := compileLone(d, formed)
							if err != nil {
								t.Fatalf("%s: %v", name, err)
							}
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
							run := func(pl [3][]uint32) []float32 {
								res, tail, err := runGraph(d, formed, op.kerns, in, out, x, tokens*rows, guard, pl)
								if err != nil {
									t.Fatalf("%s: %v", name, err)
								}
								for i, v := range tail {
									if v != guardWord {
										t.Fatalf("%s: the guard word %d past the result was written", name, i)
									}
								}
								return res
							}
							want := int8Projection(t, typ, w, x, rows, k, k, false)
							got := run([3][]uint32{qs, dw, sc})
							if e := gpuNMSE(got, want); math.IsNaN(e) || !(e < 1e-9) {
								t.Fatalf("%s: NMSE %.3e against the oracle", name, e)
							}
							if e := gpuNMSE(got, int8Projection(t, typ, w, x, rows, k, 32, false)); !(e > 1e-6) {
								t.Fatalf("%s: the reference over 32-wide windows reads %.3e: the gate cannot see the window", name, e)
							}
							// The ties' products with a row's ternary weights can
							// cancel: where the two references agree there is
							// nothing to see, and the gate counts where they do not.
							if even := int8Projection(t, typ, w, x, rows, k, k, true); gpuNMSE(even, want) > 1e-9 {
								if e := gpuNMSE(got, even); !(e > 1e-9) {
									t.Fatalf("%s: the reference rounding half to even reads %.3e: the gate cannot see the rounding", name, e)
								}
								evens++
							}
							if e := gpuNMSE(run([3][]uint32{append(qs[1:len(qs):len(qs)], qs[0]), dw, sc}), want); !(e > 1e-6) {
								t.Fatalf("%s: with the weight's words moved by one the NMSE is %.3e", name, e)
							}
							op.close()
							ran++
						}
					}
				}
			}
			t.Logf("%s: %d formed, right and seeing the violations, the rounding at %d", d.API(), ran, evens)
			if evens < ran/2 {
				t.Fatalf("%s: the rounding's violation was visible at %d of %d shapes: the ties do not reach the result", d.API(), evens, ran)
			}
		})
	}
}

// TestComposedGPUQuantMaps runs the maps the int8 statement adds -- |x|, x
// rounded half away from zero, 1/x, x/127 -- on every device present, with
// the device's own division and with the integer forms a device without a
// correctly rounded one takes, over values from 1e-40 to 1e30, the
// rounding's halves and its 2^23 edge, each held to Go's own result bit for
// bit with guard words after it -- through the integer forms, every value
// and result in the normal range.
func TestComposedGPUQuantMaps(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	xs := []float32{0, 0.5, 1.5, 2.5, -0.5, -2.5, 0.49999997, -0.49999997, 126.5, 127.49999, 8388607.5, 8388608, 16777217, 1e-40, 1e30, -3e-7}
	rng := rand.New(rand.NewSource(41))
	for len(xs) < 1000 {
		xs = append(xs, float32(rng.NormFloat64()*math.Exp2(rng.Float64()*60-30)))
	}
	maps := []struct {
		fn   graph.MapFn
		k    float64
		want func(float32) float32
	}{
		{graph.MapAbs, 0, func(v float32) float32 { return float32(math.Abs(float64(v))) }},
		{graph.MapRound, 0, func(v float32) float32 { return float32(math.Round(float64(v))) }},
		{graph.MapRecip, 0, func(v float32) float32 { return 1 / v }},
		{graph.MapDiv, 127, func(v float32) float32 { return v / 127 }},
	}
	const guard = 64
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			for _, exact := range []bool{dev.ExactDiv, false} {
				dv := dev
				dv.ExactDiv = exact
				for _, m := range maps {
					name := fmt.Sprintf("%v/exact division %v", m.fn, exact)
					b := graph.New()
					x := b.In("x", len(xs))
					b.Out(b.Emit(graph.OpMap, []graph.ValueID{x}, []graph.Result{{Name: "y", Width: len(xs)}}, nil,
						graph.Attrs{Fn: m.fn, Const: m.k, Exact: true}, nil, nil)[0])
					_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{},
						Step: composer.Step{Rows: 1, Page: 64, Keys: 1}, Device: dv})
					if err != nil {
						t.Fatalf("%s: %v", name, err)
					}
					op, err := compileLone(d, codes)
					if err != nil {
						t.Fatalf("%s: %v", name, err)
					}
					res, tail, err := op.run(xs, len(xs), guard, nil)
					op.close()
					if err != nil {
						t.Fatalf("%s: %v", name, err)
					}
					for i, v := range tail {
						if v != guardWord {
							t.Fatalf("%s: the guard word %d past the result was written", name, i)
						}
					}
					bad := 0
					for i, v := range xs {
						w := m.want(v)
						if !exact && (subnormal(v) || subnormal(w)) {
							// The integer forms leave a subnormal to the
							// device's division, as main's quantizer does
							// (kernels.div127).
							continue
						}
						if math.Float32bits(res[i]) != math.Float32bits(w) {
							bad++
							if bad < 5 {
								t.Errorf("%s(%g) = %g (%#x), want %g (%#x)", name, v, res[i], math.Float32bits(res[i]), w, math.Float32bits(w))
							}
						}
					}
					if bad > 0 {
						t.Fatalf("%s: %d of %d values off", name, bad, len(xs))
					}
				}
			}
		})
	}
}

// subnormal reports whether v is a nonzero float below the normal range.
func subnormal(v float32) bool { return v != 0 && math.Abs(float64(v)) < 0x1p-126 }

// TestComposedGPUDivision holds a map's division by a vector to the IEEE
// quotient bit for bit on every device present, with the device's own
// division where it is exact and with the integer form a device without
// one takes (exprs.div): signed operands over sixty binades, the repeat
// penalty's divisors among them, every value and quotient in the normal
// range, guard words after the result. Its violation is the device's own
// quotient on a device that is not exact (TestComposedGPUDivisionViolation).
func TestComposedGPUDivision(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	xs, ds := divisionOperands()
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			for _, exact := range []bool{dev.ExactDiv, false} {
				dv := dev
				dv.ExactDiv = exact
				bad, ran := divisionMisses(t, d, dv, xs, ds)
				if bad > 0 {
					t.Fatalf("exact %v: %d of %d quotients off", exact, bad, ran)
				}
				t.Logf("exact %v: %d quotients the IEEE ones", exact, ran)
			}
		})
	}
}

// TestComposedGPUDivisionViolation is the gate's violation: a device that
// is not exact told it is, so the map takes the device's own division. On
// Vulkan that is allowed 2.5 ulp and some quotient must be off; where the
// device's division is exact there is nothing to violate.
func TestComposedGPUDivisionViolation(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate runs on one")
	}
	xs, ds := divisionOperands()
	ran := 0
	for _, d := range devs {
		defer d.Close()
		dev := gpuDevice(d)
		if dev.ExactDiv {
			continue
		}
		ran++
		dev.ExactDiv = true
		if bad, n := divisionMisses(t, d, dev, xs, ds); bad == 0 {
			t.Errorf("%s: the device's own division matched the IEEE quotient at all %d: the violation is invisible", d.API(), n)
		} else {
			t.Logf("%s: the device's own division is off at %d of %d", d.API(), bad, n)
		}
	}
	if ran == 0 {
		t.Fatal("no device without an exact division here: the violation ran on none")
	}
}

// divisionOperands are TestComposedGPUDivision's numerators and divisors.
func divisionOperands() (xs, ds []float32) {
	const n = 4096
	rng := rand.New(rand.NewSource(43))
	xs, ds = make([]float32, n), make([]float32, n)
	for i := range xs {
		xs[i] = float32(rng.NormFloat64() * math.Exp2(rng.Float64()*60-30))
		switch i % 4 {
		case 0:
			ds[i] = float32(1 + rng.Float64())
		case 1:
			ds[i] = 1.1
		case 2:
			ds[i] = 127
		default:
			ds[i] = float32(rng.NormFloat64() * math.Exp2(rng.Float64()*40-20))
		}
	}
	xs[0], ds[0] = 0, 3
	return xs, ds
}

// divisionMisses forms x / ds on d under dv and counts the quotients that
// are not Go's, of those in the normal range.
func divisionMisses(t *testing.T, d backend.Device, dv composer.Device, xs, ds []float32) (bad, ran int) {
	t.Helper()
	const guard = 64
	n := len(xs)
	b := graph.New()
	x := b.In("x", n)
	b.Out(b.Emit(graph.OpMap, []graph.ValueID{x}, []graph.Result{{Name: "y", Width: n}},
		[]graph.Binding{{Role: jlm.RoleOutputNorm, K: n, Rows: 1}}, graph.Attrs{Fn: graph.MapDiv, Exact: true}, nil, nil)[0])
	_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{},
		Step: composer.Step{Rows: 1, Page: 64, Keys: 1}, Device: dv})
	if err != nil {
		t.Fatalf("exact %v: %v", dv.ExactDiv, err)
	}
	op, err := compileLone(d, codes)
	if err != nil {
		t.Fatalf("exact %v: %v", dv.ExactDiv, err)
	}
	res, tail, err := op.run(xs, n, guard, map[jlm.Role][]float32{jlm.RoleOutputNorm: ds})
	op.close()
	if err != nil {
		t.Fatalf("exact %v: %v", dv.ExactDiv, err)
	}
	for i, v := range tail {
		if v != guardWord {
			t.Fatalf("exact %v: the guard word %d past the result was written", dv.ExactDiv, i)
		}
	}
	for i, v := range xs {
		w := v / ds[i]
		if subnormal(v) || subnormal(w) || math.IsInf(float64(w), 0) {
			continue
		}
		ran++
		if math.Float32bits(res[i]) != math.Float32bits(w) {
			bad++
			if bad < 5 && !dv.ExactDiv {
				t.Errorf("%g/%g = %g (%#x), want %g (%#x)", v, ds[i], res[i], math.Float32bits(res[i]), w, math.Float32bits(w))
			}
		}
	}
	return bad, ran
}
