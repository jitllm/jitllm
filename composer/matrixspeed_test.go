package composer_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/composer/kernels"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testlock"
)

// BenchmarkComposedMatrix times the composer's projection on the matrix unit
// at each form JITLLM_MATRIX_SHAPES lists (the profile's alone without it)
// against main as it ships (kernels.GemmVolta at the tier's blocking and k
// split, timed with its activations' conversion and its partials' sum), on the projection
// shapes of the models JITLLM_MATRIX_MODELS names (matrixModels; Llama-3.1-8B
// without it) at JITLLM_MATRIX_ROWS rows (512 without it), on every CUDA
// device. Each round launches main, every form, then main again: a form's
// figure is the median of its per-round ratios to the mean of main's two,
// and main's second over its first is the A/A. It runs its own rounds:
//
//	go test ./jit -run '^$' -bench ComposedMatrix -benchtime 1x
func BenchmarkComposedMatrix(t *testing.B) {
	testlock.GPU(t)
	const rounds, reps = 16, 5
	R := 512
	if v := os.Getenv("JITLLM_MATRIX_ROWS"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			t.Fatal(err)
		}
		R = n
	}
	models := []string{"Llama-3.1-8B"}
	if v := os.Getenv("JITLLM_MATRIX_MODELS"); v != "" {
		models = strings.Split(v, ",")
	}
	for _, d := range backend.Open() {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		ok, why := backend.LowersTile(d, matrixTile)
		if !ok {
			t.Fatalf("%s: no binary16 tile: %s", d.Name(), why)
		}
		dev := gpuDevice(d)
		dev.Matrix = [3]int{matrixTile.M, matrixTile.N, matrixTile.K}
		for _, model := range models {
			shapes, ok := matrixModels[model]
			if !ok {
				t.Fatalf("no shapes for %q", model)
			}
			for _, sh := range shapes {
				timeMatrixShape(t, d, dev, model, sh, R, rounds, reps)
			}
		}
	}
}

type matrixShape struct {
	what string
	typ  quant.Type
	n, k int
}

// matrixModels is each model's projection shapes, outputs by k, in the type
// its Q4_K_M file holds most of them in (Q8_0 where k is not a multiple of
// 256).
var matrixModels = map[string][]matrixShape{
	"Llama-3.2-1B":  {{"q/o", quant.Q4_K, 2048, 2048}, {"k/v", quant.Q4_K, 512, 2048}, {"gate/up", quant.Q4_K, 8192, 2048}, {"down", quant.Q6_K, 2048, 8192}},
	"Llama-3.1-8B":  {{"q/o", quant.Q4_K, 4096, 4096}, {"k/v", quant.Q4_K, 1024, 4096}, {"gate/up", quant.Q4_K, 14336, 4096}, {"down", quant.Q6_K, 4096, 14336}},
	"Qwen3-0.6B":    {{"q", quant.Q4_K, 2048, 1024}, {"o", quant.Q4_K, 1024, 2048}, {"k/v", quant.Q4_K, 1024, 1024}, {"gate/up", quant.Q4_K, 3072, 1024}, {"down", quant.Q6_K, 1024, 3072}},
	"Qwen3-1.7B":    {{"q/o", quant.Q4_K, 2048, 2048}, {"k/v", quant.Q4_K, 1024, 2048}, {"gate/up", quant.Q4_K, 6144, 2048}, {"down", quant.Q6_K, 2048, 6144}},
	"Qwen3-30B-A3B": {{"q", quant.Q4_K, 4096, 2048}, {"o", quant.Q4_K, 2048, 4096}, {"k/v", quant.Q4_K, 512, 2048}, {"router", quant.Q4_K, 128, 2048}, {"expert gate/up", quant.Q4_K, 768, 2048}, {"expert down", quant.Q6_K, 2048, 768}},
	"gpt-oss-20b":   {{"q", quant.Q8_0, 4096, 2880}, {"o", quant.Q8_0, 2880, 4096}, {"k/v", quant.Q8_0, 512, 2880}, {"router", quant.Q8_0, 32, 2880}, {"expert", quant.Q8_0, 2880, 2880}},
}

// matrixArm is one composed form of a shape, compiled and bound.
type matrixArm struct {
	mx    composer.Matrix
	codes []composer.Code
	op    *loneOp
	args  [][]backend.Buf
}

func timeMatrixShape(t *testing.B, d backend.Device, dev composer.Device, model string, sh matrixShape, R, rounds, reps int) {
	jt, _ := jlm.TypeOf(sh.typ)
	q, _ := kernels.QuantOf(sh.typ)
	rng := rand.New(rand.NewSource(5))
	w := tiledWeights(sh.typ, q, sh.n, sh.k, rng)
	qs, dw, sc, err := kernels.PackWeights(q, w, sh.n, sh.k)
	if err != nil {
		t.Fatal(err)
	}
	if len(sc) == 0 {
		sc = []uint32{0}
	}
	var bufs []backend.Buf
	defer func() {
		for _, bf := range bufs {
			bf.Free()
		}
	}()
	up := func(words []uint32) backend.Buf {
		bf, err := d.Alloc(max(4*len(words), 4))
		if err != nil {
			t.Fatal(err)
		}
		p := make([]byte, 4*len(words))
		for i, v := range words {
			binary.LittleEndian.PutUint32(p[4*i:], v)
		}
		if len(p) > 0 {
			if err := bf.Write(p); err != nil {
				t.Fatal(err)
			}
		}
		bufs = append(bufs, bf)
		return bf
	}
	x := make([]uint32, R*sh.k)
	for i := range x {
		x[i] = math.Float32bits(float32(rng.NormFloat64()))
	}
	planes := [3]backend.Buf{up(qs), up(dw), up(sc)}
	xBuf, outBuf := up(x), up(make([]uint32, R*sh.n))
	var arms []*matrixArm
	defer func() {
		for _, a := range arms {
			a.op.close()
		}
	}()
	for _, mx := range matrixConfigs(t) {
		ad := dev
		if mx.Rows != 0 {
			ad.Tile = mx.Tile()
		}
		b := graph.New()
		in := b.In("x", sh.k)
		out := b.MatVec(in, jlm.RoleFFNUp, sh.n)
		b.Out(out)
		_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleFFNUp: jt},
			Step: composer.Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32}, Device: ad})
		if err != nil {
			t.Fatal(err)
		}
		op, err := compileLone(d, codes)
		if err != nil {
			t.Logf("%s %s %v: %v", model, sh.what, mx.Tile(), err)
			continue
		}
		a := &matrixArm{mx: mx, codes: codes, op: op, args: make([][]backend.Buf, len(codes))}
		arms = append(arms, a)
		own := map[composer.Operand]backend.Buf{}
		for ci, c := range codes {
			for _, bd := range c.Binds {
				o := bd.Operand
				var bf backend.Buf
				switch {
				case o.Kind == composer.OperandValue && o.Value == in:
					bf = xBuf
				case o.Kind == composer.OperandValue && o.Value == out:
					bf = outBuf
				case o.Kind == composer.OperandWeight:
					bf = planes[o.Region]
				case o.Kind == composer.OperandTable && o.Name == "rows.n":
					bf = up([]uint32{uint32(R)})
				default:
					if bf = own[o]; bf == nil {
						ws := bd.Words
						if ws == nil {
							ws = make([]uint32, max(bd.Bytes, 4)/4)
						}
						bf = up(ws)
						own[o] = bf
					}
				}
				a.args[ci] = append(a.args[ci], bf)
			}
		}
	}
	run := func(a *matrixArm) func() {
		return func() {
			for ci, c := range a.codes {
				if err := a.op.kerns[ci].Launch(c.LiveGroups(R), c.GPU.Group[0], a.args[ci]...); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	// Main as it ships: the tier's first blocking that divides the shape and
	// compiles unspilled, k split by its rule, its activations' conversion
	// before and its partials' sum after.
	sub, _, _, _ := kernels.Layout(q)
	kbN := max(32/sub, 1)
	var s kernels.MatVecShape
	var tl kernels.VoltaTile
	var mainKern, actKern, redKern backend.Kernel
	for _, c := range kernels.VoltaTiles() {
		c.KB = kbN
		if sh.n%c.Rows() != 0 || R%c.Toks() != 0 {
			continue
		}
		cs := kernels.MatVecShape{T: q, K: sh.k, Rows: sh.n, NTok: R, Split: kernels.VoltaSplit(sh.k/sub/kbN, sh.n/c.Rows()*(R/c.Toks()))}
		mk, err := kernels.GemmVolta(cs, c)
		if err != nil {
			continue
		}
		k, err := d.Compile(mk)
		if err != nil {
			continue
		}
		if r, ok := backend.KernelResourcesOf(k); ok && r.SpillBytes > 0 {
			k.Close()
			continue
		}
		mainKern, tl, s = k, c, cs
		break
	}
	if len(arms) == 0 {
		t.Fatalf("%s %s: no form compiles", model, sh.what)
	}
	compile := func(k *ir.Kernel, err error) backend.Kernel {
		if err != nil {
			t.Fatal(err)
		}
		c, err := d.Compile(k)
		if err != nil {
			t.Fatal(err)
		}
		return c
	}
	split := max(s.Split, 1)
	act16 := up(make([]uint32, R*sh.k/2))
	mainOut := up(make([]uint32, R*sh.n*split))
	if mainKern != nil {
		actKern = compile(kernels.ActF16T(q, R, sh.k))
		defer actKern.Close()
		if split > 1 {
			redKern = compile(kernels.Reduce(sh.n*R, split))
			defer redKern.Close()
		}
	}
	mainArm := func() {
		if err := actKern.Launch((R*sh.k/sub+127)/128, 128, xBuf, act16); err != nil {
			t.Fatal(err)
		}
		if err := mainKern.Launch(kernels.GemmVoltaGroups(s, tl), tl.Threads(), planes[0], planes[1], planes[2], act16, mainOut); err != nil {
			t.Fatal(err)
		}
		if redKern != nil {
			if err := redKern.Launch((sh.n*R+127)/128, 128, mainOut, outBuf); err != nil {
				t.Fatal(err)
			}
		}
	}
	if mainKern == nil {
		t.Logf("%s %s: no shipping tile compiles; the first form is the baseline", model, sh.what)
		mainArm = run(arms[0])
	} else {
		defer mainKern.Close()
	}
	sync := func() { outBuf.Read(make([]byte, 4)) }
	time1 := func(f func()) float64 {
		sync()
		t0 := time.Now()
		for range reps {
			f()
		}
		sync()
		return float64(time.Since(t0).Nanoseconds()) / 1e3 / float64(reps)
	}
	mainArm()
	for _, a := range arms {
		run(a)()
	}
	var mu, aa []float64
	ratio, us := make([][]float64, len(arms)), make([][]float64, len(arms))
	for range rounds {
		m1 := time1(mainArm)
		cs := make([]float64, len(arms))
		for i, a := range arms {
			cs[i] = time1(run(a))
		}
		m2 := time1(mainArm)
		for i, c := range cs {
			us[i] = append(us[i], c)
			ratio[i] = append(ratio[i], c/((m1+m2)/2))
		}
		mu, aa = append(mu, (m1+m2)/2), append(aa, m2/m1)
	}
	mid := func(v []float64) (float64, float64) {
		s := append([]float64(nil), v...)
		sort.Float64s(s)
		return s[len(s)/2], (s[3*len(s)/4] - s[len(s)/4]) / s[len(s)/2]
	}
	mm, _ := mid(mu)
	am, ai := mid(aa)
	t.Logf("%s %s %s %v %dx%d rows %d: main %+v split %d %.1f us, A/A %.3f (IQR/median %.3f)", d.Name(), model, sh.what, sh.typ, sh.n, sh.k, R, tl, split, mm, am, ai)
	for i, a := range arms {
		c, _ := mid(us[i])
		r, ri := mid(ratio[i])
		var mat [2]int
		var regs, spill, shared, groups int
		for ci, cd := range a.codes {
			if cd.Matrix != [2]int{} {
				mat, groups = cd.Matrix, cd.Groups
				if rs, ok := backend.KernelResourcesOf(a.op.kerns[ci]); ok {
					regs, spill, shared = rs.Regs, rs.SpillBytes, rs.SharedBytes
				}
			}
		}
		var each []string
		for ci, cd := range a.codes {
			var ts []float64
			for range rounds / 2 {
				ts = append(ts, time1(func() {
					if err := a.op.kerns[ci].Launch(cd.LiveGroups(R), cd.GPU.Group[0], a.args[ci]...); err != nil {
						t.Fatal(err)
					}
				}))
			}
			m, _ := mid(ts)
			each = append(each, fmt.Sprintf("%s %.1f", cd.Name, m))
		}
		t.Logf("  %v shape %v %d workgroups, regs %d spill %d shared %d: %.1f us, composed/main %.3f (IQR/median %.3f); us each: %s", a.mx.Tile(), mat, groups, regs, spill, shared, c, r, ri, strings.Join(each, ", "))
	}
}

// matrixConfigs is the matrix forms the probe times: JITLLM_MATRIX_SHAPES,
// a comma list of ROWSxCOLSxDEPTH:WRxWN:RESIDENT[:SPLIT], or the profile's alone.
func matrixConfigs(t testing.TB) []composer.Matrix {
	env := os.Getenv("JITLLM_MATRIX_SHAPES")
	if env == "" {
		return []composer.Matrix{{}}
	}
	var out []composer.Matrix
	for _, f := range strings.Split(env, ",") {
		var v []int
		for _, x := range strings.FieldsFunc(f, func(r rune) bool { return r == 'x' || r == ':' }) {
			n, err := strconv.Atoi(x)
			if err != nil {
				t.Fatalf("JITLLM_MATRIX_SHAPES %q: %v", f, err)
			}
			v = append(v, n)
		}
		if len(v) == 6 {
			v = append(v, 0)
		}
		if len(v) != 7 {
			t.Fatalf("JITLLM_MATRIX_SHAPES %q: want ROWSxCOLSxDEPTH:WRxWN:RESIDENT[:SPLIT]", f)
		}
		out = append(out, composer.Matrix{Rows: v[0], Cols: v[1], Depth: v[2], Warps: [2]int{v[3], v[4]}, Resident: v[5], Split: v[6]})
	}
	return out
}
