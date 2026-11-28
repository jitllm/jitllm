package composer_test

import (
	"encoding/binary"
	"fmt"
	"maps"
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
	"github.com/jitllm/jitllm/internal/testlock"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// projectionRoles is the roles of a Llama block's projections.
func projectionRoles() []jlm.Role {
	return []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV, jlm.RoleAttnOut, jlm.RoleFFNGate, jlm.RoleFFNUp, jlm.RoleFFNDown}
}

// bindsRole reports whether c binds the weight of role r.
func bindsRole(c composer.Code, r jlm.Role) bool {
	return slices.ContainsFunc(c.Binds, func(b composer.Binding) bool {
		return b.Operand.Kind == composer.OperandWeight && b.Operand.Role == r
	})
}

// projectionLaunches is the codes binding a projection's weight a call of
// rows rows launches, and whether one of them binds every role of each of
// together.
func projectionLaunches(codes []composer.Code, rows int, together ...[]jlm.Role) (int, bool) {
	n, joined := 0, true
	for _, c := range codes {
		if c.Runs(rows, 0) && slices.ContainsFunc(projectionRoles(), func(r jlm.Role) bool { return bindsRole(c, r) }) {
			n++
		}
	}
	for _, roles := range together {
		one := false
		for _, c := range codes {
			all := c.Runs(rows, 0)
			for _, r := range roles {
				all = all && bindsRole(c, r)
			}
			one = one || all
		}
		joined = joined && one
	}
	return n, joined
}

// TestCapacitySiblingsLaunchAsOneRow forms a Llama-3.1-8B-shaped block at a
// joint step's capacity of 512 rows on a device of a V100's CUDA statement,
// and counts the projections' launches a call of 1, 2, 4, 8 and 16 rows
// makes: the vector form at the capacity runs q, k and v as one launch and
// gate and up as one (Siblings, row tile by row tile), so a call of its few
// rows launches the projections a form of one row does, and the matrix
// form's calls launch each projection apart, as they do without Siblings.
func TestCapacitySiblingsLaunchAsOneRow(t *testing.T) {
	d := composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, ExactDiv: true, Regs: 32,
		Matrix: [3]int{16, 16, 16}, Fragment: ir.MMAVolta(), HalfPairs: true, VectorLoads: true, Shared: 96 << 10, Buffers: 32}
	formats := map[jlm.Role]jlm.Type{}
	for _, r := range projectionRoles() {
		formats[r] = jlm.TypeQ4S
	}
	formats[jlm.RoleFFNDown] = jlm.TypeQ6S
	formats[jlm.RoleAttnNorm], formats[jlm.RoleFFNNorm] = jlm.TypeF32, jlm.TypeF32
	form := func(reg *composer.Registry, rows, seqs int) []composer.Code {
		_, codes, err := reg.Form(composer.Request{Graph: blockOf(4096, 32, 8, 128, 14336), Formats: formats,
			Step: composer.Step{Rows: rows, Page: 64, Keys: 1024, ActWindow: 32, Seqs: seqs}, Device: d,
			History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: true}}})
		if err != nil {
			t.Fatal(err)
		}
		return codes
	}
	qkv, kv, gu := []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV}, []jlm.Role{jlm.RoleAttnK, jlm.RoleAttnV}, []jlm.Role{jlm.RoleFFNGate, jlm.RoleFFNUp}
	one, joined := projectionLaunches(form(composer.Default(), 1, 1), 1, qkv, gu)
	if !joined || one != 4 {
		t.Fatalf("a form of one row launches %d projections, q, k and v and gate and up together %v: want 4, together", one, joined)
	}
	codes, apart := form(composer.Default(), 512, 2), form(without("siblings"), 512, 2)
	for _, rows := range []int{1, 2, 4, 8, 12, 16} {
		n, together := projectionLaunches(codes, rows, kv, gu)
		a, _ := projectionLaunches(apart, rows)
		all, matrix := 0, false
		for _, c := range codes {
			if c.Runs(rows, 0) {
				all++
				matrix = matrix || c.Matrix != [2]int{}
			}
		}
		t.Logf("%d rows: %d projection launches (%d apart), %d launches in all, the matrix form %v", rows, n, a, all, matrix)
		switch {
		case !matrix && !together:
			t.Errorf("a call of %d rows runs k and v, or gate and up, as launches of their own", rows)
		case !matrix && n >= a:
			t.Errorf("a call of %d rows launches %d projections, the siblings apart %d", rows, n, a)
		case matrix && n != a:
			t.Errorf("a call of %d rows, the matrix form's, launches %d projections, the siblings apart %d", rows, n, a)
		}
	}
	for _, rows := range []int{2, 4} {
		if n, together := projectionLaunches(codes, rows, qkv, gu); n != one || !together {
			t.Errorf("a call of %d rows launches %d projections, q, k and v and gate and up together %v: want the one-row form's %d", rows, n, together, one)
		}
	}
}

// capacityProjections is q, k and v of one normed input and gate and up of
// another, as a Llama block forms them.
type capacityProjections struct {
	embd, q, kv, ffn int
}

// TestComposedGPUCapacitySiblings forms q, k and v over one normed input and
// gate and up over another at a capacity of 512 rows on every GPU device
// present, a Llama-3.1-8B shape and Qwen2-1.5B's ragged one (outputs of
// 1536, 256 and 8960), and runs them at calls of 1, 2, 3, 4, 5, 8 and 9
// rows -- and Qwen2's at a capacity of 64 rows at every call from 1 to 64,
// across each band of the vector form's row tiles and the matrix unit's --
// the input's rows past the call's poisoned and guard words after
// every output, on the device's statement with no matrix unit, where every
// call runs the vector form, and with the one it lowers: each output's rows
// the oracle's (the norm, then the int8 matvec over 32-wide windows the
// vector form runs or the float one the matrix unit does), nothing past the
// call's rows written, and k and v and
// gate and up one launch at every call the vector form runs (Siblings over
// a form of a capacity) and launches of their own where the matrix unit
// does. Three violations must fail: v's weight words moved by one, up's
// likewise, and a call stating a row more than its input holds, whose
// launch must write that row.
func TestComposedGPUCapacitySiblings(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		testmodels.Missing(t, "no GPU backend on this host: the gate runs on one")
	}
	const guard = 64
	few, every := []int{1, 2, 3, 4, 5, 8, 9}, make([]int, 64)
	for i := range every {
		every[i] = i + 1
	}
	qwen2 := capacityProjections{1536, 1536, 256, 8960}
	sweeps := []struct {
		capacity int
		calls    []int
		sh       capacityProjections
	}{{512, few, capacityProjections{4096, 4096, 1024, 14336}}, {512, few, qwen2}, {64, every, qwen2}}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, unit := range []bool{false, true} {
				dev := gpuDevice(d)
				if unit {
					if ok, _ := backend.LowersTile(d, matrixTile()); !ok {
						continue
					}
					dev.Matrix = [3]int{matrixTile().M, matrixTile().N, matrixTile().K}
				}
				for _, sw := range sweeps {
					capacity, calls, most, sh := sw.capacity, sw.calls, slices.Max(sw.calls), sw.sh
					name := fmt.Sprintf("capacity %d, %d: %d/%d/%d, %d/%d, matrix unit %v", capacity, sh.embd, sh.q, sh.kv, sh.kv, sh.ffn, sh.ffn, unit)
					rng := rand.New(rand.NewSource(int64(sh.embd + sh.ffn)))
					b := graph.New()
					b.G.Block = 0
					x1, x2 := b.In("x", sh.embd), b.In("x2", sh.embd)
					n1, n2 := b.Norm(x1, jlm.RoleAttnNorm, 1e-5), b.Norm(x2, jlm.RoleFFNNorm, 1e-5)
					roles := []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV, jlm.RoleFFNGate, jlm.RoleFFNUp}
					widths := []int{sh.q, sh.kv, sh.kv, sh.ffn, sh.ffn}
					var outs []graph.ValueID
					for i, r := range roles {
						in := n1
						if i >= 3 {
							in = n2
						}
						outs = append(outs, b.MatVec(in, r, widths[i]))
					}
					b.Out(outs...)
					formats := map[jlm.Role]jlm.Type{jlm.RoleAttnNorm: jlm.TypeF32, jlm.RoleFFNNorm: jlm.TypeF32}
					raw, weights := map[jlm.Role][]byte{}, map[jlm.Role][3][]uint32{}
					for i, r := range roles {
						formats[r] = jlm.TypeQ4S
						raw[r] = tiledWeights(quant.Q4_K, layout.Q4_K, widths[i], sh.embd, rng)
						qs, dw, sc, err := layout.PackWeights(layout.Q4_K, raw[r], widths[i], sh.embd)
						if err != nil {
							t.Fatal(err)
						}
						weights[r] = [3][]uint32{qs, dw, sc}
					}
					vec := func(n int, mean, sd float64) []float32 {
						v := make([]float32, n)
						for i := range v {
							v[i] = float32(mean + sd*rng.NormFloat64())
						}
						return v
					}
					vecs := map[jlm.Role][]float32{jlm.RoleAttnNorm: vec(sh.embd, 1, 0.1), jlm.RoleFFNNorm: vec(sh.embd, 1, 0.1)}
					xs := map[graph.ValueID][]float32{x1: vec(capacity*sh.embd, 0, 1), x2: vec(capacity*sh.embd, 0, 1)}
					// The oracle's rows: each norm's, then each projection's float
					// and int8 matvecs over them.
					normed := map[graph.ValueID][]float32{}
					for _, c := range []struct {
						x graph.ValueID
						r jlm.Role
					}{{x1, jlm.RoleAttnNorm}, {x2, jlm.RoleFFNNorm}} {
						y := make([]float32, most*sh.embd)
						for row := range most {
							oracle.RMSNorm32(y[row*sh.embd:(row+1)*sh.embd], xs[c.x][row*sh.embd:(row+1)*sh.embd], vecs[c.r], 1e-5)
						}
						normed[c.x] = y
					}
					want, want8 := map[graph.ValueID][]float32{}, map[graph.ValueID][]float32{}
					for i, o := range outs {
						y := normed[x1]
						if i >= 3 {
							y = normed[x2]
						}
						w := make([]float32, most*widths[i])
						if err := oracle.MatMul(w, quant.Q4_K, raw[roles[i]], y, most, widths[i], sh.embd); err != nil {
							t.Fatal(err)
						}
						want[o], want8[o] = w, int8Projection(t, quant.Q4_K, raw[roles[i]], y, widths[i], sh.embd, 32, false)
					}
					_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Formats: formats,
						Step: composer.Step{Rows: capacity, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
					if err != nil {
						t.Fatalf("%s: forming: %v", name, err)
					}
					op, err := compileLone(d, codes)
					if err != nil {
						t.Fatalf("%s: %v", name, err)
					}
					// run is a call of L rows of the input stating stated rows: each
					// output's L rows, and whether a word past them was written.
					run := func(L, stated int, weights map[jlm.Role][3][]uint32) (map[graph.ValueID][]float32, bool) {
						ins := map[graph.ValueID][]float32{}
						for x, v := range xs {
							in := slices.Clone(v)
							for i := L * sh.embd; i < len(in); i++ {
								in[i] = math.Float32frombits(guardWord)
							}
							ins[x] = in
						}
						sizes := map[graph.ValueID]int{}
						for i, o := range outs {
							sizes[o] = capacity * widths[i]
						}
						words, err := runRolesAt(d, op, ins, weights, vecs, sizes, stated, guard)
						if err != nil {
							t.Fatalf("%s: %v", name, err)
						}
						res, past := map[graph.ValueID][]float32{}, false
						for i, o := range outs {
							w, n := words[o], L*widths[i]
							if w == nil {
								t.Fatalf("%s: no code writes %v", name, roles[i])
							}
							past = past || slices.ContainsFunc(w[n:], func(v uint32) bool { return v != guardWord })
							res[o] = make([]float32, n)
							for j := range n {
								res[o][j] = math.Float32frombits(w[j])
							}
						}
						return res, past
					}
					merged := 0
					for _, L := range calls {
						got, past := run(L, L, weights)
						if past {
							t.Fatalf("%s: a call of %d rows wrote past them", name, L)
						}
						for i, o := range outs {
							n := L * widths[i]
							e := min(gpuNMSE(got[o], want[o][:n]), gpuNMSE(got[o], want8[o][:n]))
							if math.IsNaN(e) || !(e < 2e-6) {
								t.Fatalf("%s: a call of %d rows: %v reads NMSE %.3e against the oracle", name, L, roles[i], e)
							}
						}
						n, together := projectionLaunches(codes, L, []jlm.Role{jlm.RoleAttnK, jlm.RoleAttnV}, []jlm.Role{jlm.RoleFFNGate, jlm.RoleFFNUp})
						matrix := slices.ContainsFunc(codes, func(c composer.Code) bool { return c.Runs(L, 0) && c.Matrix != [2]int{} })
						switch {
						case together == matrix:
							t.Errorf("%s: a call of %d rows, the matrix unit's %v, runs k and v and gate and up together %v", name, L, matrix, together)
						case !matrix:
							merged++
						}
						t.Logf("%s: %s: a call of %d rows: %d projection launches, k and v and gate and up together %v", d.API(), name, L, n, together)
					}
					if !unit && merged != len(calls) {
						t.Errorf("%s: %d of %d calls ran the vector form, with no matrix unit to run", name, merged, len(calls))
					}
					for _, r := range []jlm.Role{jlm.RoleAttnV, jlm.RoleFFNUp} {
						i := slices.Index(roles, r)
						moved := maps.Clone(weights)
						qs := weights[r][0]
						moved[r] = [3][]uint32{append(qs[1:len(qs):len(qs)], qs[0]), weights[r][1], weights[r][2]}
						for _, L := range []int{3, most} {
							got, _ := run(L, L, moved)
							n := L * widths[i]
							if e := gpuNMSE(got[outs[i]], want[outs[i]][:n]); !(e > 1e-4) {
								t.Fatalf("%s: a call of %d rows with %v's weight words moved by one reads NMSE %.3e: the gate cannot see its share", name, L, r, e)
							}
						}
					}
					for _, L := range []int{3, 5, most - 1} {
						if _, past := run(L, L+1, weights); !past {
							t.Fatalf("%s: a call stating %d rows over %d wrote nothing past them: the gate cannot see a launch not held", name, L+1, L)
						}
					}
					op.close()
				}
			}
		})
	}
}

// runRolesAt runs op's codes at a call of rows rows (0: the form's
// capacity) over the inputs ins, the weights' planes by role and region and
// the vectors by role: each code the call runs (Code.Runs) launched at its
// workgroups for them (Code.LiveGroups), the call's table its rows, every
// other buffer poisoned. Each output of sizes is that many words followed by
// guard poisoned ones, and so is each input; it returns the outputs' words.
func runRolesAt(d backend.Device, op *loneOp, ins map[graph.ValueID][]float32, weights map[jlm.Role][3][]uint32,
	vecs map[jlm.Role][]float32, sizes map[graph.ValueID]int, rows, guard int) (map[graph.ValueID][]uint32, error) {
	var bufs []backend.Buf
	defer func() {
		for _, b := range bufs {
			b.Free()
		}
	}()
	alloc := func(w []uint32) (backend.Buf, error) {
		p := make([]byte, max(4*len(w), 4))
		for i, v := range w {
			binary.LittleEndian.PutUint32(p[4*i:], v)
		}
		b, err := d.Alloc(len(p))
		if err != nil {
			return nil, err
		}
		bufs = append(bufs, b)
		return b, b.Write(p)
	}
	f32 := func(v []float32) []uint32 {
		w := make([]uint32, len(v))
		for i, f := range v {
			w[i] = math.Float32bits(f)
		}
		return w
	}
	own := map[composer.Operand]backend.Buf{}
	outBufs := map[graph.ValueID]backend.Buf{}
	for ci, c := range op.codes {
		args := make([]backend.Buf, len(c.Binds))
		for i, bd := range c.Binds {
			o := bd.Operand
			if b, ok := own[o]; ok {
				args[i] = b
				continue
			}
			var w []uint32
			switch {
			case o.Kind == composer.OperandValue && ins[o.Value] != nil:
				w = append(f32(ins[o.Value]), poisoned(guard)...)
			case o.Kind == composer.OperandValue && sizes[o.Value] > 0:
				w = poisoned(sizes[o.Value] + guard)
			case o.Kind == composer.OperandVector || o.Kind == composer.OperandWeight && o.Format == jlm.TypeF32:
				v, ok := vecs[o.Role]
				if !ok {
					return nil, fmt.Errorf("%s binds %v, which the gate does not hold", c.Name, o.Role)
				}
				w = f32(v)
			case o.Kind == composer.OperandWeight:
				p, ok := weights[o.Role]
				if !ok {
					return nil, fmt.Errorf("%s binds %v, which the gate does not hold", c.Name, o.Role)
				}
				w = p[o.Region]
			case rows > 0 && o.Kind == composer.OperandTable && o.Name == "rows.n":
				w = append([]uint32{uint32(rows)}, poisoned(guard)...)
			case bd.Words != nil:
				w = bd.Words
			default:
				w = poisoned(int(max(bd.Bytes, 4)/4) + guard)
			}
			b, err := alloc(w)
			if err != nil {
				return nil, err
			}
			own[o], args[i] = b, b
			if o.Kind == composer.OperandValue && sizes[o.Value] > 0 {
				outBufs[o.Value] = b
			}
		}
		if !c.Runs(rows, 0) {
			continue
		}
		groups := c.Groups
		if rows > 0 {
			groups = c.LiveGroups(rows)
		}
		if err := op.kerns[ci].Launch(groups, c.GPU.Group[0], args...); err != nil {
			return nil, fmt.Errorf("launching %s: %w", c.Name, err)
		}
	}
	out := map[graph.ValueID][]uint32{}
	for v, b := range outBufs {
		p := make([]byte, 4*(sizes[v]+guard))
		if err := b.Read(p); err != nil {
			return nil, err
		}
		w := make([]uint32, sizes[v]+guard)
		for i := range w {
			w[i] = binary.LittleEndian.Uint32(p[4*i:])
		}
		out[v] = w
	}
	return out, nil
}
