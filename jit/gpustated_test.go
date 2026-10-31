package jit_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/jit"
)

// TestComposedGPUStatedReduce holds a stated expression whose reduction
// reads a broadcast operand to its reference on every device backend:
// sparsemixer's routed weight (Phi-3.5-MoE) over the logits with an earlier
// pick removed (-inf) and the pick's own logit, as graph.Stated lowers it --
// the terms before the sum an OpExpr, the sum an OpReduce, the division an
// OpExpr over the two. Every weight must be finite and match, at ragged
// widths and several rows, with the words past the result untouched.
func TestComposedGPUStatedReduce(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate proved nothing")
	}
	const guard = 64
	e := convert.SparseMixerWeight(jlm.PhiMoEJitter)
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			r := jit.Default()
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(7))
			right := 0
			for _, rows := range []int{1, 3} {
				for _, E := range []int{2, 4, 8, 16, 33, 60, 128} {
					name := fmt.Sprintf("%d rows/%d experts", rows, E)
					b := graph.New()
					x := b.In("x", E+1)
					rest := b.Emit(graph.OpSlice, []graph.ValueID{x}, []graph.Result{{Name: "rest", Width: E}}, nil,
						graph.Attrs{Offset: 0, Width: E}, nil, nil)[0]
					pick := b.Emit(graph.OpSlice, []graph.ValueID{x}, []graph.Result{{Name: "pick", Width: 1}}, nil,
						graph.Attrs{Offset: E, Width: 1}, nil, nil)[0]
					w, err := b.Stated("w", &e, []graph.ValueID{rest, pick}, E, graph.Facts{})
					if err != nil {
						t.Fatalf("%s: %v", name, err)
					}
					b.Out(w)
					_, codes, err := r.Form(jit.Request{Graph: b.G, Step: jit.Step{Rows: rows, Page: 64, Keys: 1}, Device: dev})
					if err != nil {
						t.Errorf("%s: %v", name, err)
						continue
					}
					op, err := compileLone(d, codes)
					if err != nil {
						t.Errorf("%s: %v", name, err)
						continue
					}
					in := make([]float32, rows*(E+1))
					want := make([]float32, rows*E)
					for row := range rows {
						s := in[row*(E+1) : row*(E+1)+E]
						for i := range s {
							s[i] = float32(2 * rng.NormFloat64())
						}
						p := rng.Intn(E)
						if E > 1 {
							s[(p+1)%E] = float32(math.Inf(-1))
						}
						in[row*(E+1)+E] = s[p]
						statedMixer(want[row*E:(row+1)*E], s, s[p])
					}
					got, tail, err := runBound(op, d, in, 0, w, len(want), guard)
					op.close()
					if err != nil {
						t.Errorf("%s: %v", name, err)
						continue
					}
					for i, v := range tail {
						if v != guardWord {
							t.Errorf("%s: the guard word %d past the result was written", name, i)
							break
						}
					}
					bad := 0
					for i := range want {
						if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-want[i])) > 1e-6*max(1, math.Abs(float64(want[i]))) {
							bad++
						}
					}
					if bad != 0 {
						t.Errorf("%s: %d of %d weights differ from the reference (got %v, want %v)", name, bad, len(want), got[:min(8, len(got))], want[:min(8, len(want))])
						continue
					}
					right++
				}
			}
			if right == 0 {
				t.Fatal("no shape held to the reference: this gate proved nothing")
			}
		})
	}
}

// statedMixer is sparsemixer's routed weight at PhiMoEJitter, the reference
// a test holds the composed form to: over s, each element kept where
// (m - s)/max(|s|, m) is not past twice the jitter, the softmax of the kept
// ones relative to m.
func statedMixer(dst, s []float32, m float32) {
	eps2 := float32(2 * float64(jlm.PhiMoEJitter))
	var sum float32
	for i, v := range s {
		ratio := (m - v) / max(float32(math.Abs(float64(v))), m)
		if eps2 < ratio {
			dst[i] = 0
		} else {
			dst[i] = float32(math.Exp(float64(v - m)))
		}
		sum += dst[i]
	}
	for i := range dst {
		dst[i] /= sum
	}
}

// runBound launches op's codes with graph value inID bound to x and value
// outID to a result of n floats followed by guard words; scratch is poisoned.
func runBound(l *loneOp, d backend.Device, x []float32, inID, outID graph.ValueID, n, guard int) ([]float32, []uint32, error) {
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
			if err := b.Write(p); err != nil {
				return nil, err
			}
		}
		return b, nil
	}
	xw := make([]uint32, 0, len(x)+guard)
	for _, v := range x {
		xw = append(xw, math.Float32bits(v))
	}
	in, err := alloc(append(xw, poisoned(guard)...))
	if err != nil {
		return nil, nil, err
	}
	out, err := alloc(poisoned(n + guard))
	if err != nil {
		return nil, nil, err
	}
	own := map[jit.Operand]backend.Buf{}
	for ci, c := range l.codes {
		args := make([]backend.Buf, len(c.Binds))
		for i, bd := range c.Binds {
			o := bd.Operand
			switch {
			case o.Kind == jit.OperandValue && o.Value == inID:
				args[i] = in
			case o.Kind == jit.OperandValue && o.Value == outID:
				args[i] = out
			case bd.Words != nil || o.Kind == jit.OperandScratch || o.Kind == jit.OperandValue || o.Kind == jit.OperandTable && o.Name == "rows.n":
				b, ok := own[o]
				if !ok {
					w := bd.Words
					if w == nil {
						w = poisoned(int(max(bd.Bytes, 4) / 4))
					}
					if b, err = alloc(w); err != nil {
						return nil, nil, err
					}
					own[o] = b
				}
				args[i] = b
			default:
				return nil, nil, fmt.Errorf("%s binds %+v, which this test does not provide", c.Name, o)
			}
		}
		if err := l.kerns[ci].Launch(c.Groups, c.GPU.Group[0], args...); err != nil {
			return nil, nil, fmt.Errorf("launching %s: %w", c.Name, err)
		}
	}
	raw := make([]byte, 4*(n+guard))
	if err := out.Read(raw); err != nil {
		return nil, nil, err
	}
	words := make([]uint32, n+guard)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(raw[4*i:])
	}
	res := make([]float32, n)
	for i := range res {
		res[i] = math.Float32frombits(words[i])
	}
	return res, words[n:], nil
}
