package jit_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// guardWord is a quiet NaN: what the words past a result and the composer's
// scratch hold before a launch, so a read of what nothing wrote shows.
const guardWord = 0x7FC0DEAD

// gpuDevice is the composer's description of d, as the tier states it.
func gpuDevice(d backend.Device) jit.Device {
	lanes := 1
	if ok, _ := backend.GuaranteedLanes(d, ir.SubgroupLanes); ok {
		lanes = ir.SubgroupLanes
	}
	units := 0
	if m, ok := d.(backend.MultiprocessorDevice); ok {
		units = m.Multiprocessors()
	}
	return jit.Device{Kind: jit.DeviceGPU, Slots: d.Slots(), Lanes: lanes, Units: units, ExactDiv: kernels.DivExact(d.API())}
}

// writesInput reports whether the formed kernels write the graph's input
// (value 0) rather than its output: the in-place pass stores a value over an
// input of its shape that nothing reads after, and the caller then holds the
// result where it held the input.
func writesInput(codes []jit.Code) bool {
	for _, c := range codes {
		for _, b := range c.Binds {
			if b.Operand.Kind == jit.OperandValue && b.Operand.Value == 0 && b.Writes {
				return true
			}
		}
	}
	return false
}

// loneOp is a lone op's formed codes compiled on a device: run it on an input
// and the vector weights as many times as a gate wants.
type loneOp struct {
	d     backend.Device
	codes []jit.Code
	kerns []backend.Kernel
}

func compileLone(d backend.Device, codes []jit.Code) (*loneOp, error) {
	l := &loneOp{d: d, codes: codes}
	for _, c := range codes {
		if c.GPU == nil {
			l.close()
			return nil, fmt.Errorf("%s has no GPU code", c.Name)
		}
		k, err := d.Compile(c.GPU)
		if err != nil {
			l.close()
			return nil, fmt.Errorf("compiling %s: %w", c.Name, err)
		}
		l.kerns = append(l.kerns, k)
	}
	return l, nil
}

func (l *loneOp) close() {
	for _, k := range l.kerns {
		k.Close()
	}
	l.kerns = nil
}

func poisoned(n int) []uint32 {
	w := make([]uint32, n)
	for i := range w {
		w[i] = guardWord
	}
	return w
}

// run launches the op in order: the input x (graph value 0) and the output
// (value 1) of n floats each followed by guard words, the vector weights by
// role, and the kernels' own buffers, scratch poisoned and constant blocks
// written. It returns the result, where the form left it, and the words after
// it.
func (l *loneOp) run(x []float32, n, guard int, vecs map[jlm.Role][]float32) (res []float32, tail []uint32, err error) {
	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	alloc := func(w []uint32) (backend.Buf, error) {
		b, err := l.d.Alloc(max(4*len(w), 4))
		if err != nil {
			return nil, err
		}
		bufs = append(bufs, b)
		if len(w) > 0 {
			p := make([]byte, 4*len(w))
			for i, v := range w {
				binary.LittleEndian.PutUint32(p[4*i:], v)
			}
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
	vec := map[jlm.Role]backend.Buf{}
	own := map[jit.Operand]backend.Buf{}
	for ci, c := range l.codes {
		args := make([]backend.Buf, len(c.Binds))
		for i, bd := range c.Binds {
			o := bd.Operand
			switch {
			case o.Kind == jit.OperandValue && o.Value == 0:
				args[i] = in
			case o.Kind == jit.OperandValue && o.Value == 1:
				args[i] = out
			case o.Kind == jit.OperandVector:
				b, ok := vec[o.Role]
				if !ok {
					v := vecs[o.Role]
					w := make([]uint32, len(v))
					for j, f := range v {
						w[j] = math.Float32bits(f)
					}
					if b, err = alloc(w); err != nil {
						return nil, nil, err
					}
					vec[o.Role] = b
				}
				args[i] = b
			case bd.Words != nil || o.Kind == jit.OperandScratch:
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
				return nil, nil, fmt.Errorf("%s binds %+v, which a lone op does not provide", c.Name, o)
			}
		}
		if err := l.kerns[ci].Launch(c.Groups, c.GPU.Group[0], args...); err != nil {
			return nil, nil, fmt.Errorf("launching %s: %w", c.Name, err)
		}
	}
	held := out
	if writesInput(l.codes) {
		held = in
	}
	raw := make([]byte, 4*(n+guard))
	if err := held.Read(raw); err != nil {
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

// gpuNMSE is the squared error of got from want over want's energy, and the
// squared error itself where want is all zeros (a width-one LayerNorm without
// a bias is exactly zero, which no ratio can hold a kernel to).
func gpuNMSE(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num, den = num+d*d, den+float64(want[i])*float64(want[i])
	}
	if den == 0 {
		return num
	}
	return num / den
}

// TestComposedGPULayerNorm executes the composer's LayerNorm on every device
// present, with and without a bias, at widths from one element through
// several workgroups and at several rows, each held to the oracle's f32
// LayerNorm with a guard of poisoned words after the result. The rows' mean is
// far from zero, so the RMS norm of the same shape lands far from the oracle,
// and with the scale's or the bias's words moved by one the kernel is wrong
// too: the gate sees both.
func TestComposedGPULayerNorm(t *testing.T) {
	testlock.GPU(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const eps, guard = 1e-5, 64
	// The bias adds one elementwise term after the scale, so the widths that
	// matter to it are few; the scale-only form takes the ragged ones whole.
	shapes := func(bias bool) (rows, widths []int) {
		if bias {
			return []int{1, 3, 9}, []int{1, 2, 3, 5, 7, 8, 9, 17, 33, 129, 257, 1000, 4544}
		}
		return []int{1, 3}, []int{1, 3, 9, 129, 1000}
	}
	rotated := func(v []float32) []float32 { return append(append([]float32(nil), v[1:]...), v[0]) }
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			r := jit.Default()
			dev := gpuDevice(d)
			form := func(g *graph.Graph, rows int) (*loneOp, error) {
				_, codes, err := r.Form(jit.Request{
					Graph:   g,
					Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnNorm: jlm.TypeF32, jlm.RoleAttnNormBias: jlm.TypeF32},
					Step:    jit.Step{Rows: rows, Page: 64, Keys: 1},
					Device:  dev,
				})
				if err != nil {
					return nil, err
				}
				return compileLone(d, codes)
			}
			rng := rand.New(rand.NewSource(17))
			ran := 0
			for _, bias := range []bool{false, true} {
				rowCounts, widths := shapes(bias)
				for _, rows := range rowCounts {
					for _, w := range widths {
						name := fmt.Sprintf("bias %v/%d rows/width %d", bias, rows, w)
						var roles []jlm.Role
						if bias {
							roles = append(roles, jlm.RoleAttnNormBias)
						}
						b := graph.New()
						b.Out(b.LayerNorm(b.In("x", w), jlm.RoleAttnNorm, eps, roles...))
						ln, err := form(b.G, rows)
						if err != nil {
							t.Errorf("%s: %v", name, err)
							continue
						}
						x := make([]float32, rows*w)
						for i := range x {
							x[i] = float32(2 + rng.NormFloat64())
						}
						scale, shift := make([]float32, w), make([]float32, w)
						for i := range scale {
							scale[i], shift[i] = float32(0.5+rng.Float64()), float32(rng.NormFloat64())
						}
						if !bias {
							shift = nil
						}
						want := make([]float32, rows*w)
						for tk := range rows {
							oracle.LayerNorm32(want[tk*w:(tk+1)*w], x[tk*w:(tk+1)*w], scale, shift, eps)
						}
						run := func(op *loneOp, scale, shift []float32) ([]float32, error) {
							res, tail, err := op.run(x, rows*w, guard, map[jlm.Role][]float32{jlm.RoleAttnNorm: scale, jlm.RoleAttnNormBias: shift})
							if err != nil {
								return nil, err
							}
							for i, v := range tail {
								if v != guardWord {
									return nil, fmt.Errorf("the guard word %d past the result was written", i)
								}
							}
							return res, nil
						}
						check := func() {
							defer ln.close()
							got, err := run(ln, scale, shift)
							if err != nil {
								t.Errorf("%s: %v", name, err)
								return
							}
							if e := gpuNMSE(got, want); math.IsNaN(e) || !(e < 1e-9) {
								t.Errorf("%s: NMSE %.3e against the oracle", name, e)
								return
							}
							ran++
							if w == 1 {
								return
							}
							bad, err := run(ln, rotated(scale), shift)
							if err != nil {
								t.Errorf("%s: the violation did not run: %v", name, err)
							} else if e := gpuNMSE(bad, want); !(e > 1e-6) {
								t.Errorf("%s: with the scale's words moved by one the NMSE is %.3e: the gate cannot see a wrong kernel", name, e)
							}
							if bias {
								bad, err := run(ln, scale, rotated(shift))
								if err != nil {
									t.Errorf("%s: the violation did not run: %v", name, err)
								} else if e := gpuNMSE(bad, want); !(e > 1e-6) {
									t.Errorf("%s: with the bias's words moved by one the NMSE is %.3e: the gate cannot see a wrong kernel", name, e)
								}
							}
							rb := graph.New()
							rb.Out(rb.Norm(rb.In("x", w), jlm.RoleAttnNorm, eps))
							rms, err := form(rb.G, rows)
							if err != nil {
								t.Errorf("%s: the RMS norm does not form: %v", name, err)
								return
							}
							defer rms.close()
							if bad, err := run(rms, scale, shift); err != nil {
								t.Errorf("%s: the violation did not run: %v", name, err)
							} else if e := gpuNMSE(bad, want); !(e > 1e-6) {
								t.Errorf("%s: an RMS norm lands %.3e from the oracle's LayerNorm: the rows' mean does not reach the gate", name, e)
							}
						}
						check()
					}
				}
			}
			if ran == 0 {
				t.Fatal("no LayerNorm formed and ran on this device: this gate proved nothing")
			}
			t.Logf("%d LayerNorm shapes formed and right on %s (%s)", ran, d.Name(), d.API())
		})
	}
}
