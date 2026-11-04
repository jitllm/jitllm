package composer_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testlock"
)

// runGPUGraph forms nothing: it launches codes, formed from g, on d over x
// (g's input), every value at rowsOf rows of its width followed by guard
// words the widest value's most rows deep, F32 weights and vectors bound by
// role, tables by name and read back into them; it returns g's output, or
// an error where any value's guard was written.
func runGPUGraph(d backend.Device, codes []composer.Code, g *graph.Graph, rowsOf func(graph.ValueID) int, x []float32, roles map[jlm.Role][]float32,
	tables map[string][]uint32) ([]float32, error) {
	var kerns []backend.Kernel
	var bufs []backend.Buf
	defer func() {
		for _, k := range kerns {
			k.Close()
		}
		for _, b := range bufs {
			b.Free()
		}
	}()
	w, r := 0, 0
	for v := range g.Values {
		w, r = max(w, g.Values[v].Width), max(r, rowsOf(graph.ValueID(v)))
	}
	guard := 16 + w*r
	alloc := func(words []uint32) (backend.Buf, error) {
		b, err := d.Alloc(max(4*len(words), 4))
		if err != nil {
			return nil, err
		}
		bufs = append(bufs, b)
		p := make([]byte, 4*len(words))
		for i, v := range words {
			binary.LittleEndian.PutUint32(p[4*i:], v)
		}
		return b, b.Write(p)
	}
	extent := func(v graph.ValueID) int { return g.Values[v].Width * rowsOf(v) }
	vals := map[graph.ValueID]backend.Buf{}
	value := func(v graph.ValueID) (backend.Buf, error) {
		if b, ok := vals[v]; ok {
			return b, nil
		}
		words := poisoned(extent(v) + guard)
		if v == g.In[0] {
			for i, f := range x {
				words[i] = math.Float32bits(f)
			}
		}
		b, err := alloc(words)
		vals[v] = b
		return b, err
	}
	if _, err := value(g.In[0]); err != nil {
		return nil, err
	}
	vec := map[jlm.Role]backend.Buf{}
	own := map[composer.Operand]backend.Buf{}
	tab := map[string]backend.Buf{}
	for _, c := range codes {
		if c.GPU == nil {
			return nil, fmt.Errorf("%s has no GPU code", c.Name)
		}
		k, err := d.Compile(c.GPU)
		if err != nil {
			return nil, fmt.Errorf("compiling %s: %w", c.Name, err)
		}
		kerns = append(kerns, k)
		args := make([]backend.Buf, len(c.Binds))
		for i, bd := range c.Binds {
			o := bd.Operand
			switch {
			case o.Kind == composer.OperandValue:
				if args[i], err = value(o.Value); err != nil {
					return nil, err
				}
			case o.Kind == composer.OperandVector || o.Kind == composer.OperandWeight && o.Format == jlm.TypeF32:
				b, ok := vec[o.Role]
				if !ok {
					v := roles[o.Role]
					words := make([]uint32, len(v))
					for j, f := range v {
						words[j] = math.Float32bits(f)
					}
					if b, err = alloc(words); err != nil {
						return nil, err
					}
					vec[o.Role] = b
				}
				args[i] = b
			case o.Kind == composer.OperandTable && tables[o.Name] != nil:
				b, ok := tab[o.Name]
				if !ok {
					if b, err = alloc(tables[o.Name]); err != nil {
						return nil, err
					}
					tab[o.Name] = b
				}
				args[i] = b
			case bd.Words != nil || o.Kind == composer.OperandScratch || o.Kind == composer.OperandTable && o.Name == "rows.n":
				b, ok := own[o]
				if !ok {
					words := bd.Words
					if words == nil {
						words = poisoned(int(max(bd.Bytes, 4) / 4))
					}
					if b, err = alloc(words); err != nil {
						return nil, err
					}
					own[o] = b
				}
				args[i] = b
			default:
				return nil, fmt.Errorf("%s binds %+v, which the grid gate does not provide", c.Name, o)
			}
		}
		if err := k.Launch(c.Groups, c.GPU.Group[0], args...); err != nil {
			return nil, fmt.Errorf("launching %s: %w", c.Name, err)
		}
	}
	for name, b := range tab {
		raw := make([]byte, 4*len(tables[name]))
		if err := b.Read(raw); err != nil {
			return nil, err
		}
		for i := range tables[name] {
			tables[name][i] = binary.LittleEndian.Uint32(raw[4*i:])
		}
	}
	read := func(v graph.ValueID) ([]uint32, error) {
		n := extent(v) + guard
		raw := make([]byte, 4*n)
		if err := vals[v].Read(raw); err != nil {
			return nil, err
		}
		words := make([]uint32, n)
		for i := range words {
			words[i] = binary.LittleEndian.Uint32(raw[4*i:])
		}
		return words, nil
	}
	for v := range vals {
		if v == g.In[0] {
			continue
		}
		words, err := read(v)
		if err != nil {
			return nil, err
		}
		for i, wd := range words[extent(v):] {
			if wd != guardWord {
				return nil, fmt.Errorf("value %s: word %#x written %d past its end", g.Values[v].Name, wd, i)
			}
		}
	}
	out := g.Out[0]
	if _, ok := vals[out]; !ok {
		// The in-place pass stored the output over the last value written.
		for i := len(codes) - 1; i >= 0 && vals[out] == nil; i-- {
			for _, b := range codes[i].Binds {
				if b.Writes && b.Operand.Kind == composer.OperandValue && vals[b.Operand.Value] != nil {
					vals[out] = vals[b.Operand.Value]
				}
			}
		}
		if vals[out] == nil {
			return nil, fmt.Errorf("no kernel wrote the output")
		}
	}
	words, err := read(out)
	if err != nil {
		return nil, err
	}
	res := make([]float32, extent(out))
	for i := range res {
		res[i] = math.Float32frombits(words[i])
	}
	return res, nil
}

// gridRowsOf is graph g's value rows at a step of step rows.
func gridRowsOf(g *graph.Graph, step int) func(graph.ValueID) int {
	return func(v graph.ValueID) int {
		if r := g.Values[v].Rows; r > 0 {
			return r
		}
		return step
	}
}

// TestComposedGPUGridRows is TestComposedCPUGridRows on every device
// backend: the row-wise ops over a stride-2 depthwise convolution's grid
// held to the oracle, guards past every value, and each op read at the
// input's rows seen -- in the chain or alone.
func TestComposedGPUGridRows(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate proved nothing")
	}
	kinds := []graph.OpKind{graph.OpMatVec, graph.OpNorm, graph.OpExpr, graph.OpAdd}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(41))
			right := 0
			seen := map[graph.OpKind]int{}
			for _, g := range composer.GridRowsCases() {
				gr, err := composer.GridRowsGraph(g)
				if err != nil {
					t.Fatalf("%v: %v", g, err)
				}
				x := composer.RandRow(rng, g.H*g.W*g.C, 1)
				filt := composer.RandRow(rng, g.K*g.K*g.C, 0.5)
				w1, n1 := composer.RandRow(rng, g.E*g.C, 1/math.Sqrt(float64(g.C))), composer.RandRow(rng, g.E, 1)
				w2 := composer.RandRow(rng, g.C*g.E, 1/math.Sqrt(float64(g.E)))
				want := composer.GridRowsWant(g, x, filt, w1, n1, w2)
				roles := map[jlm.Role][]float32{jlm.RoleVMNDwMid: filt, jlm.RoleVMNPwExp: w1, jlm.RoleVMNPwExpNorm: n1, jlm.RoleVMNPwProj: w2}
				formats := map[jlm.Role]jlm.Type{jlm.RoleVMNDwMid: jlm.TypeF32, jlm.RoleVMNPwExp: jlm.TypeF32,
					jlm.RoleVMNPwExpNorm: jlm.TypeF32, jlm.RoleVMNPwProj: jlm.TypeF32}
				step := composer.Step{Rows: g.H * g.W, Page: 64, Keys: 1}
				run := func(r *composer.Registry, gr *graph.Graph) ([]float32, error) {
					_, codes, err := r.Form(composer.Request{Graph: gr, Formats: formats, Step: step, Device: dev})
					if err != nil {
						return nil, err
					}
					return runGPUGraph(d, codes, gr, gridRowsOf(gr, step.Rows), x, roles, nil)
				}
				got, err := run(composer.Default(), gr)
				if err != nil {
					t.Errorf("%v: %v", g, err)
					continue
				}
				if e := gpuNMSE(got, want); math.IsNaN(e) || !(e < 1e-10) {
					t.Errorf("%v: NMSE %.3e against the oracle", g, e)
					continue
				}
				right++
				if g.S == 1 {
					continue
				}
				for _, k := range kinds {
					lone, lwant := composer.GridRowsLone(g, k, x, filt, n1)
					lgot, err := run(composer.Default(), lone)
					if err != nil {
						t.Errorf("%v: %s alone: %v", g, k, err)
						continue
					}
					if e := gpuNMSE(lgot, lwant); math.IsNaN(e) || !(e < 1e-10) {
						t.Errorf("%v: %s alone over the grid reads NMSE %.3e", g, k, e)
						continue
					}
					for _, arm := range []*graph.Graph{gr, lone} {
						w := want
						if arm == lone {
							w = lwant
						}
						bad, err := run(composer.Misrowed(composer.Default(), k, step.Rows), arm)
						if err != nil {
							seen[k]++
							continue
						}
						if e := gpuNMSE(bad, w); e > 1e-6 || math.IsNaN(e) {
							seen[k]++
						}
					}
				}
			}
			t.Logf("%d of %d grids' row-wise ops held to the oracle; the violations seen by op %v", right, len(composer.GridRowsCases()), seen)
			if right != len(composer.GridRowsCases()) {
				t.Fatalf("%d of %d grids' row-wise ops held to the oracle", right, len(composer.GridRowsCases()))
			}
			for _, k := range kinds {
				if seen[k] == 0 {
					t.Errorf("%s read at the input's rows over a stride-2 grid is seen at no grid: the gate cannot see it", k)
				}
			}
		})
	}
}

// TestComposedGPUUnfold is TestComposedCPUUnfold on every device backend:
// a full convolution as OpUnfold and a matrix product, held to the oracle's
// convolution, the filter transposed seen.
func TestComposedGPUUnfold(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate proved nothing")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(43))
			right := 0
			for _, g := range composer.GridRowsCases() {
				kp := (g.K*g.K*g.C + 31) / 32 * 32
				b := graph.New()
				xv := b.In("x", g.C)
				res := graph.Result{Name: "unfold", Width: kp}
				if g.S > 1 {
					res.Rows = g.Out()
				}
				u := b.Emit(graph.OpUnfold, []graph.ValueID{xv}, []graph.Result{res}, nil, graph.Attrs{Taps: g.K, GridW: g.W, Stride: g.S}, nil, nil)[0]
				b.Out(b.MatVec(u, jlm.RoleVMNConvExp, g.E))
				x := composer.RandRow(rng, g.H*g.W*g.C, 1)
				wt := composer.RandRow(rng, g.E*kp, 1/math.Sqrt(float64(g.K*g.K*g.C)))
				for o := range g.E {
					for j := g.K * g.K * g.C; j < kp; j++ {
						wt[o*kp+j] = 0
					}
				}
				want := make([]float32, g.Out()*g.E)
				oracle.Conv32(want, x, g.H, g.W, g.C, g.K, g.S, wt, g.E, kp)
				step := composer.Step{Rows: g.H * g.W, Page: 64, Keys: 1}
				_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleVMNConvExp: jlm.TypeF32}, Step: step, Device: dev})
				if err != nil {
					t.Errorf("%v: %v", g, err)
					continue
				}
				run := func(wt []float32) ([]float32, error) {
					return runGPUGraph(d, codes, b.G, gridRowsOf(b.G, step.Rows), x, map[jlm.Role][]float32{jlm.RoleVMNConvExp: wt}, nil)
				}
				got, err := run(wt)
				if err != nil {
					t.Errorf("%v: %v", g, err)
					continue
				}
				if e := gpuNMSE(got, want); math.IsNaN(e) || !(e < 1e-10) {
					t.Errorf("%v: NMSE %.3e against the oracle's convolution", g, e)
					continue
				}
				right++
				if g.H*g.W < 4 {
					continue
				}
				tr := make([]float32, len(wt))
				for o := range g.E {
					for ky := range g.K {
						for kx := range g.K {
							copy(tr[o*kp+(kx*g.K+ky)*g.C:o*kp+(kx*g.K+ky+1)*g.C], wt[o*kp+(ky*g.K+kx)*g.C:o*kp+(ky*g.K+kx+1)*g.C])
						}
					}
				}
				if bad, err := run(tr); err != nil {
					t.Errorf("%v: the violation did not run: %v", g, err)
				} else if e := gpuNMSE(bad, want); !(e > 1e-6) {
					t.Errorf("%v: the filter transposed reads %.3e: the gate cannot see it", g, e)
				}
			}
			if right != len(composer.GridRowsCases()) {
				t.Fatalf("%d of %d full convolutions held to the oracle", right, len(composer.GridRowsCases()))
			}
			t.Logf("%d full convolutions held to the oracle", right)
		})
	}
}

// TestComposedGPUGridAttend is TestComposedCPUGridAttend on every device
// backend: MobileNet-V5's attention over a stride-2 grid's keys held to the
// oracle, the pools poisoned past the key grid, and the append at the query
// grid's rows seen writing past it.
func TestComposedGPUGridAttend(t *testing.T) {
	testlock.GPU(t)
	devs := openDevices()
	if len(devs) == 0 {
		t.Fatal("no GPU backend on this host: the gate proved nothing")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			dev := gpuDevice(d)
			rng := rand.New(rand.NewSource(43))
			right, seen := 0, 0
			for _, g := range composer.GridAttendCases() {
				gr, err := composer.GridAttendGraph(g)
				if err != nil {
					t.Fatalf("%v: %v", g, err)
				}
				roles, x, want := composer.GridAttendData(g, rng)
				n, m := g.H*g.W, g.Out()
				if g.S == 1 {
					m = n
				}
				step := composer.Step{Rows: n, Page: m, Keys: m}
				run := func(r *composer.Registry) ([]float32, map[string][]uint32, error) {
					_, codes, err := r.Form(composer.Request{Graph: gr, Formats: composer.GridAttendFormats(), Step: step, Device: dev})
					if err != nil {
						return nil, nil, err
					}
					tabs := composer.GridAttendTables(g, m, n+m)
					got, err := runGPUGraph(d, codes, gr, gridRowsOf(gr, n), x, roles, tabs)
					return got, tabs, err
				}
				got, tabs, err := run(composer.Default())
				if err != nil {
					t.Errorf("%v: %v", g, err)
					continue
				}
				if e := gpuNMSE(got, want); math.IsNaN(e) || !(e < 1e-7) {
					t.Errorf("%v: NMSE %.3e against the oracle", g, e)
					continue
				}
				if w := composer.PoolWrittenPast(tabs, m*g.KD); w >= 0 {
					t.Errorf("%v: the key pool written at word %d, past the key grid's %d rows", g, w, m)
					continue
				}
				right++
				if g.S == 1 {
					continue
				}
				if _, tabs, err = run(composer.Misrowed(composer.Default(), graph.OpKVAppend, n)); err != nil || composer.PoolWrittenPast(tabs, m*g.KD) >= 0 {
					seen++
				}
			}
			t.Logf("%d of %d grids' attention held to the oracle; the append at the query grid's rows seen at %d", right, len(composer.GridAttendCases()), seen)
			if right != len(composer.GridAttendCases()) {
				t.Fatalf("%d of %d grids' attention held to the oracle", right, len(composer.GridAttendCases()))
			}
			if seen == 0 {
				t.Fatal("the append at the query grid's rows is seen at no grid: the gate cannot see it")
			}
		})
	}
}
