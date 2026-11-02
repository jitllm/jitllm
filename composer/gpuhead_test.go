package composer_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/kernels"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
)

// headRun launches the formed head on rows rows of x: its logits and its
// normed rows read back, each followed by guard words.
func headRun(d backend.Device, codes []composer.Code, kerns []backend.Kernel, g *graph.Graph, x []float32, n [2]int, guard int,
	planes [3][]uint32, vecs map[jlm.Role][]float32, rows int) (res [2][]float32, tail [2][]uint32, err error) {
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
		return
	}
	var outBuf [2]backend.Buf
	for i := range outBuf {
		if outBuf[i], err = alloc(poisoned(n[i] + guard)); err != nil {
			return
		}
	}
	reach := map[composer.Operand]int64{}
	for _, c := range codes {
		for _, bd := range c.Binds {
			reach[bd.Operand] = max(reach[bd.Operand], bd.Bytes)
		}
	}
	own := map[composer.Operand]backend.Buf{}
	vec := map[jlm.Role]backend.Buf{}
	for ci, c := range codes {
		if !c.RunsAt(rows) {
			continue
		}
		args := make([]backend.Buf, len(c.Binds))
		for i, bd := range c.Binds {
			o := bd.Operand
			switch {
			case o.Kind == composer.OperandValue && o.Value == g.In[0]:
				args[i] = inBuf
			case o.Kind == composer.OperandValue && o.Value == g.Out[0]:
				args[i] = outBuf[0]
			case len(g.Out) > 1 && o.Kind == composer.OperandValue && o.Value == g.Out[1]:
				args[i] = outBuf[1]
			case o.Kind == composer.OperandWeight && o.Region >= 0 && o.Region < 3 && len(planes[o.Region]) > 0:
				b, ok := own[o]
				if !ok {
					if b, err = alloc(planes[o.Region]); err != nil {
						return
					}
					own[o] = b
				}
				args[i] = b
			case o.Kind == composer.OperandVector:
				b, ok := vec[o.Role]
				if !ok {
					v := vecs[o.Role]
					w := make([]uint32, len(v))
					for j, f := range v {
						w[j] = math.Float32bits(f)
					}
					if b, err = alloc(w); err != nil {
						return
					}
					vec[o.Role] = b
				}
				args[i] = b
			case o.Kind == composer.OperandWeight:
				err = fmt.Errorf("%s binds %+v, which the gate does not hold", c.Name, o)
				return
			case rows > 0 && o.Kind == composer.OperandTable && o.Name == "rows.n":
				b, ok := own[o]
				if !ok {
					if b, err = alloc(append([]uint32{uint32(rows)}, poisoned(guard)...)); err != nil {
						return
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
						return
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
		if err = kerns[ci].Launch(groups, c.GPU.Group[0], args...); err != nil {
			err = fmt.Errorf("launching %s: %w", c.Name, err)
			return
		}
	}
	for i := range outBuf {
		raw := make([]byte, 4*(n[i]+guard))
		if err = outBuf[i].Read(raw); err != nil {
			return
		}
		res[i] = make([]float32, n[i])
		tail[i] = make([]uint32, guard)
		for j := range n[i] + guard {
			w := binary.LittleEndian.Uint32(raw[4*j:])
			if j < n[i] {
				res[i][j] = math.Float32frombits(w)
			} else {
				tail[i][j-n[i]] = w
			}
		}
	}
	return
}

// TestComposedGPUHead executes the composer's head -- a LayerNorm with its
// bias, the vocabulary's float projection with its bias, the normed rows
// written beside the logits -- on every device present and on a CUDA card
// lowered for sm_70, at the widths and capacities a batch is formed for, at
// calls of fewer rows than the capacity, each held to the oracle with
// poisoned guard words after both outputs.
func TestComposedGPUHead(t *testing.T) {
	testlock.GPU(t)
	devs := map[string]backend.Device{}
	for _, d := range backend.Open() {
		devs[d.API()] = d
		defer d.Close()
	}
	if d, err := backend.OpenCUDAWith(0, backend.Opts{CUDATarget: "sm_70"}); err == nil {
		devs["ptx-sm_70"] = d
		defer d.Close()
	}
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const eps, guard = 1e-5, 64
	for name, d := range devs {
		t.Run(name, func(t *testing.T) {
			for _, matrix := range []bool{false, true} {
				dev := gpuDevice(d)
				if matrix {
					ok, _ := backend.LowersTile(d, matrixTile)
					if !ok {
						continue
					}
					dev.Matrix = [3]int{matrixTile.M, matrixTile.N, matrixTile.K}
				}
				rng := rand.New(rand.NewSource(71))
				for _, sh := range []struct{ k, v int }{{64, 1000}, {64, 32064}, {96, 777}} {
					for _, capacity := range []int{1, 3, 8, 16, 32} {
						what := fmt.Sprintf("matrix %v/%d embd/%d vocab/capacity %d", matrix, sh.k, sh.v, capacity)
						h := graph.Head{NEmbd: sh.k, NVocab: sh.v, Eps: eps, LayerNorm: true, NormBias: true,
							Output: jlm.RoleOutput, Bias: true, Hidden: true}
						g := h.Build()
						_, codes, err := composer.Default().Form(composer.Request{Graph: g,
							Formats: map[jlm.Role]jlm.Type{jlm.RoleOutput: jlm.TypeF32, jlm.RoleOutputBias: jlm.TypeF32},
							Step:    composer.Step{Rows: capacity, Page: 64, Keys: 1}, Device: dev})
						if err != nil {
							t.Errorf("%s: forming: %v", what, err)
							continue
						}
						op, err := compileLone(d, codes)
						if err != nil {
							t.Errorf("%s: %v", what, err)
							continue
						}
						w := make([]float32, sh.v*sh.k)
						raw := make([]byte, 4*len(w))
						for i := range w {
							w[i] = float32(rng.NormFloat64() / 8)
							binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(w[i]))
						}
						qs, dw, sc, err := kernels.PackWeights(kernels.Float32, raw, sh.v, sh.k)
						if err != nil {
							t.Fatal(err)
						}
						vecs := map[jlm.Role][]float32{jlm.RoleOutputNorm: nil, jlm.RoleOutputNormBias: nil, jlm.RoleOutputBias: nil}
						for r, n := range map[jlm.Role]int{jlm.RoleOutputNorm: sh.k, jlm.RoleOutputNormBias: sh.k, jlm.RoleOutputBias: sh.v} {
							v := make([]float32, n)
							for i := range v {
								v[i] = float32(0.5 + rng.Float64())
								if r != jlm.RoleOutputNorm {
									v[i] = float32(rng.NormFloat64())
								}
							}
							vecs[r] = v
						}
						x := make([]float32, capacity*sh.k)
						for i := range x {
							x[i] = float32(1 + rng.NormFloat64())
						}
						wantN := make([]float32, capacity*sh.k)
						wantL := make([]float32, capacity*sh.v)
						for r := range capacity {
							oracle.LayerNorm32(wantN[r*sh.k:(r+1)*sh.k], x[r*sh.k:(r+1)*sh.k], vecs[jlm.RoleOutputNorm], vecs[jlm.RoleOutputNormBias], eps)
							if err := oracle.MatVec(wantL[r*sh.v:(r+1)*sh.v], quant.F32, raw, wantN[r*sh.k:(r+1)*sh.k], sh.v, sh.k); err != nil {
								t.Fatal(err)
							}
							for i := range sh.v {
								wantL[r*sh.v+i] += vecs[jlm.RoleOutputBias][i]
							}
						}
						for _, L := range []int{1, 3, capacity} {
							if L > capacity {
								continue
							}
							res, tail, err := headRun(d, codes, op.kerns, g, x, [2]int{capacity * sh.v, capacity * sh.k}, guard,
								[3][]uint32{qs, dw, sc}, vecs, L)
							if err != nil {
								t.Errorf("%s: %v", what, err)
								continue
							}
							for o := range tail {
								for i, v := range tail[o] {
									if v != guardWord {
										t.Errorf("%s: the guard word %d past output %d was written", what, i, o)
										break
									}
								}
							}
							if e := gpuNMSE(res[1][:L*sh.k], wantN[:L*sh.k]); math.IsNaN(e) || !(e < 1e-9) {
								t.Errorf("%s: a call of %d rows reads the normed rows at NMSE %.3e against the oracle", what, L, e)
							}
							if e := gpuNMSE(res[0][:L*sh.v], wantL[:L*sh.v]); math.IsNaN(e) || !(e < 1e-9) {
								rowsBad := ""
								for r := range L {
									nan := 0
									for _, v := range res[0][r*sh.v : (r+1)*sh.v] {
										if v != v {
											nan++
										}
									}
									rowsBad += fmt.Sprintf(" row %d: nmse %.2e nan %d;", r, gpuNMSE(res[0][r*sh.v:(r+1)*sh.v], wantL[r*sh.v:(r+1)*sh.v]), nan)
								}
								t.Errorf("%s: a call of %d rows reads the logits at NMSE %.3e against the oracle:%s", what, L, e, rowsBad)
							}
						}
						op.close()
					}
				}
			}
		})
	}
}
