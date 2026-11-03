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
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testlock"
)

// BenchmarkComposedDecode times the composer's decode projections behind
// their producer on every CUDA device: the profile's form against each form
// JITLLM_DECODE_SPLITS lists (g16: 16 segments in a workgroup, g16w512 at 512
// threads, p32: 32 partial planes, x: no prologue, R4: four workgroups a
// unit), on JITLLM_MATRIX_MODELS' shapes. A measurement harness, not a tuner:
//
//	go test ./jit -run '^$' -bench ComposedDecode -benchtime 1x
func BenchmarkComposedDecode(t *testing.B) {
	testlock.GPU(t)
	const rounds, reps = 16, 20
	models := []string{"Llama-3.1-8B"}
	if v := os.Getenv("JITLLM_MATRIX_MODELS"); v != "" {
		models = strings.Split(v, ",")
	}
	var splits []string
	if v := os.Getenv("JITLLM_DECODE_SPLITS"); v != "" {
		splits = strings.Split(v, ",")
	}
	for _, d := range backend.Open() {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		dev := gpuDevice(d)
		if r, ok := d.(backend.RegisterDevice); ok {
			dev.Regs = r.ThreadRegisters()
		}
		for _, model := range models {
			shapes := matrixModels()[model]
			if model == "Llama-3.1-8B" {
				shapes = append(shapes, matrixShape{"down q4", quant.Q4_K, 4096, 14336}, matrixShape{"qkv", quant.Q4_K, 4096, 4096}, matrixShape{"gu", quant.Q4_K, 14336, 4096})
			}
			for _, sh := range shapes {
				timeDecodeShape(t, d, dev, model, sh, splits, rounds, reps)
			}
		}
	}
}

func decodeRegistry(t testing.TB, spec string) *composer.Registry {
	reg := composer.Default()
	if spec == "" || spec[0] == 'R' {
		return reg
	}
	if spec == "x" {
		p := reg.Profiles[composer.DeviceGPU]
		p.Off = map[string]bool{"prologue": true}
		reg.Profiles[composer.DeviceGPU] = p
		return reg
	}
	num, width, _ := strings.Cut(spec[1:], "w")
	S, err := strconv.Atoi(num)
	if err != nil {
		t.Fatalf("split %q: %v", spec, err)
	}
	W := 0
	if width != "" {
		if W, err = strconv.Atoi(width); err != nil {
			t.Fatalf("split %q: %v", spec, err)
		}
	}
	p := reg.Profiles[composer.DeviceGPU]
	ps := append([]composer.Pass(nil), p.Passes...)
	for i, q := range ps {
		if s, ok := q.(composer.Split); ok && s.Matrix && s.Parts == 0 && s.Block == 1 {
			s.Width = W
			if S > 0 {
				s.Slots, s.Max, s.Group = 1<<30, S, spec[0] == 'g'
			}
			ps[i] = s
		}
	}
	p.Passes = ps
	reg.Profiles[composer.DeviceGPU] = p
	return reg
}

type decodeArm struct {
	name  string
	codes []composer.Code
	op    *loneOp
	args  [][]backend.Buf
}

func timeDecodeShape(t *testing.B, d backend.Device, dev composer.Device, model string, sh matrixShape, splits []string, rounds, reps int) {
	jt, _ := jlm.TypeOf(sh.typ)
	q, _ := layout.QuantOf(sh.typ)
	rng := rand.New(rand.NewSource(5))
	w := tiledWeights(sh.typ, q, sh.n, sh.k, rng)
	qs, dw, sc, err := layout.PackWeights(q, w, sh.n, sh.k)
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
		if err := bf.Write(p); err != nil {
			t.Fatal(err)
		}
		bufs = append(bufs, bf)
		return bf
	}
	x := make([]uint32, sh.k)
	for i := range x {
		x[i] = math.Float32bits(float32(rng.NormFloat64()))
	}
	planes := [3]backend.Buf{up(qs), up(dw), up(sc)}
	xBuf, outBuf := up(x), up(make([]uint32, sh.n))
	ones := make([]uint32, sh.k)
	for i := range ones {
		ones[i] = math.Float32bits(1)
	}
	normBuf := up(ones)
	var arms []*decodeArm
	defer func() {
		for _, a := range arms {
			a.op.close()
		}
	}()
	for _, spec := range append([]string{""}, splits...) {
		b := graph.New()
		in := b.In("x", sh.k)
		act := b.Norm(in, jlm.RoleFFNNorm, 1e-5)
		if strings.HasPrefix(sh.what, "down") {
			act = siluMul(b, in, b.In("u", sh.k), sh.k)
		}
		formats := map[jlm.Role]jlm.Type{jlm.RoleFFNUp: jt, jlm.RoleFFNNorm: jlm.TypeF32}
		out := b.MatVec(act, jlm.RoleFFNUp, sh.n)
		b.Out(out)
		if sh.what == "gu" {
			formats[jlm.RoleFFNGate] = jt
			b.G.Out = append(b.G.Out, b.MatVec(act, jlm.RoleFFNGate, sh.n))
		}
		if sh.what == "qkv" {
			formats[jlm.RoleAttnK], formats[jlm.RoleAttnV] = jt, jt
			b.G.Out = append(b.G.Out, b.MatVec(act, jlm.RoleAttnK, sh.n/4), b.MatVec(act, jlm.RoleAttnV, sh.n/4))
		}
		_, codes, err := decodeRegistry(t, spec).Form(composer.Request{Graph: b.G, Formats: formats,
			Step: composer.Step{Rows: 1, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
		if err != nil {
			t.Fatal(err)
		}
		if spec != "" && spec[0] == 'R' {
			r, err := strconv.Atoi(spec[1:])
			if err != nil {
				t.Fatal(err)
			}
			for _, c := range codes {
				if strings.HasSuffix(c.Name, ".group") {
					c.GPU.Resident = r
				}
			}
		}
		op, err := compileLone(d, codes)
		if err != nil {
			t.Logf("%s %s %q: %v", model, sh.what, spec, err)
			continue
		}
		name := spec
		if name == "" {
			name = "profile"
		}
		a := &decodeArm{name: name, codes: codes, op: op, args: make([][]backend.Buf, len(codes))}
		arms = append(arms, a)
		own := map[composer.Operand]backend.Buf{}
		for ci, c := range codes {
			for _, bd := range c.Binds {
				o := bd.Operand
				var bf backend.Buf
				switch {
				case o.Kind == composer.OperandValue && o.Value < out && (o.Value == in || b.G.Values[o.Value].Name == "u"):
					bf = xBuf
				case o.Kind == composer.OperandValue && o.Value == out:
					bf = outBuf
				case o.Kind == composer.OperandWeight && o.Role == jlm.RoleFFNNorm:
					bf = normBuf
				case o.Kind == composer.OperandWeight:
					bf = planes[o.Region]
				case o.Kind == composer.OperandTable && o.Name == "rows.n":
					bf = up([]uint32{1})
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
	if len(arms) == 0 {
		t.Fatalf("%s %s: no form compiles", model, sh.what)
	}
	run := func(a *decodeArm) func() {
		return func() {
			for ci, c := range a.codes {
				if err := a.op.kerns[ci].Launch(c.LiveGroups(1), c.GPU.Group[0], a.args[ci]...); err != nil {
					t.Fatal(err)
				}
			}
		}
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
	for _, a := range arms {
		run(a)()
	}
	var mu, aa []float64
	ratio, us := make([][]float64, len(arms)), make([][]float64, len(arms))
	for range rounds {
		m1 := time1(run(arms[0]))
		cs := make([]float64, len(arms))
		for i, a := range arms {
			cs[i] = time1(run(a))
		}
		m2 := time1(run(arms[0]))
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
	t.Logf("%s %s %s %v %dx%d: profile %.1f us, A/A %.3f (IQR/median %.3f)", d.Name(), model, sh.what, sh.typ, sh.n, sh.k, mm, am, ai)
	for i, a := range arms {
		c, _ := mid(us[i])
		r, ri := mid(ratio[i])
		var desc []string
		for ci, cd := range a.codes {
			s := fmt.Sprintf("%s %dx%d", cd.Name, cd.LiveGroups(1), cd.GPU.Group[0])
			if rs, ok := backend.KernelResourcesOf(a.op.kerns[ci]); ok {
				s += fmt.Sprintf(" r%d", rs.Regs)
				if rs.SpillBytes > 0 {
					s += fmt.Sprintf(" spill%d", rs.SpillBytes)
				}
			}
			desc = append(desc, s)
		}
		t.Logf("  %-8s %.1f us, /profile %.3f (IQR/median %.3f) [%s]", a.name, c, r, ri, strings.Join(desc, "; "))
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
func matrixModels() map[string][]matrixShape {
	return map[string][]matrixShape{
		"Llama-3.2-1B":  {{"q/o", quant.Q4_K, 2048, 2048}, {"k/v", quant.Q4_K, 512, 2048}, {"gate/up", quant.Q4_K, 8192, 2048}, {"down", quant.Q6_K, 2048, 8192}},
		"Llama-3.1-8B":  {{"q/o", quant.Q4_K, 4096, 4096}, {"k/v", quant.Q4_K, 1024, 4096}, {"gate/up", quant.Q4_K, 14336, 4096}, {"down", quant.Q6_K, 4096, 14336}},
		"Qwen3-0.6B":    {{"q", quant.Q4_K, 2048, 1024}, {"o", quant.Q4_K, 1024, 2048}, {"k/v", quant.Q4_K, 1024, 1024}, {"gate/up", quant.Q4_K, 3072, 1024}, {"down", quant.Q6_K, 1024, 3072}},
		"Qwen3-1.7B":    {{"q/o", quant.Q4_K, 2048, 2048}, {"k/v", quant.Q4_K, 1024, 2048}, {"gate/up", quant.Q4_K, 6144, 2048}, {"down", quant.Q6_K, 2048, 6144}},
		"Qwen3-30B-A3B": {{"q", quant.Q4_K, 4096, 2048}, {"o", quant.Q4_K, 2048, 4096}, {"k/v", quant.Q4_K, 512, 2048}, {"router", quant.Q4_K, 128, 2048}, {"expert gate/up", quant.Q4_K, 768, 2048}, {"expert down", quant.Q6_K, 2048, 768}},
		"gpt-oss-20b":   {{"q", quant.Q8_0, 4096, 2880}, {"o", quant.Q8_0, 2880, 4096}, {"k/v", quant.Q8_0, 512, 2880}, {"router", quant.Q8_0, 32, 2880}, {"expert", quant.Q8_0, 2880, 2880}},
	}
}
