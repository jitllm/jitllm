package composer_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testlock"
)

// BenchmarkComposedFlash times the composer's prompt attention on the matrix
// unit (the flash pass) at Llama's geometries, a 512-row chunk at the end of a
// 2048-position prompt in the served models' pool layout, its values binary16
// and float32 (the card's default history), on every CUDA
// device: each code's median microseconds over its own rounds, and the flash
// kernel's registers, spills and workgroup memory. JITLLM_FLASH_SHAPES, a
// comma list of ROWS:KEYS:SHARED, times those flash forms in place of the
// profile's. It runs its own rounds:
//
//	go test ./composer -run '^$' -bench ComposedFlash -benchtime 1x
func BenchmarkComposedFlash(t *testing.B) {
	testlock.GPU(t)
	const page, R, p0, keys, rounds, reps = 64, 512, 1536, 4608, 16, 5
	for _, d := range openDevices() {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		for _, c := range []attendRowsCase{{32, 8, 64, 0, false, false}, {32, 8, 128, 0, false, false}, {32, 8, 64, 0, false, true}, {32, 8, 128, 0, false, true}} {
			for _, fl := range flashConfigs(t) {
				reg := composer.Default()
				if fl.Rows != 0 {
					p := reg.Profiles[composer.DeviceGPU]
					p.Passes = slices.Clone(p.Passes)
					for i, ps := range p.Passes {
						if _, ok := ps.(composer.Flash); ok {
							p.Passes[i] = fl
						}
					}
					reg.Profiles[composer.DeviceGPU] = p
				}
				b := graph.New()
				n := b.KVAppend(b.In("k", c.kvh*c.dim), b.In("v", c.kvh*c.dim), 0)
				scale := 1 / math.Sqrt(float64(c.dim))
				b.Out(b.Attend(b.In("q", c.heads*c.dim), n, graph.Attrs{NHead: c.heads, NKVHead: c.kvh, HeadDim: c.dim, Scale: scale}))
				g := b.G
				dev := gpuDevice(d)
				dev.Matrix = [3]int{16, 16, 16}
				_, codes, err := reg.Form(composer.Request{Graph: g, Step: composer.Step{Rows: R, Page: page, Keys: keys, Seqs: 1},
					Device: dev, History: composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: !c.f32v}}})
				if err != nil {
					t.Fatal(err)
				}
				rng := rand.New(rand.NewSource(3))
				rnd := func(n int) []uint32 {
					x := make([]uint32, n)
					for i := range x {
						x[i] = math.Float32bits(float32(rng.NormFloat64()))
					}
					return x
				}
				w := c.kvh * c.dim
				rows := make([]uint32, R*4)
				for r := range R {
					p := uint32(p0 + r)
					copy(rows[r*4:], []uint32{0, 0, p + 1, p})
				}
				pages := make([]uint32, keys/page)
				for i := range pages {
					pages[i] = uint32(i)
				}
				vals := map[graph.ValueID][]uint32{g.In[0]: rnd(R * w), g.In[1]: rnd(R * w), g.In[2]: rnd(R * c.heads * c.dim), g.Out[0]: make([]uint32, R*c.heads*c.dim)}
				vwords := keys * w / 2
				if c.f32v {
					vwords = keys * w
				}
				tables := map[string][]uint32{"rows": rows, "rows.n": {R}, "pages": pages,
					"kv.k": rnd(keys * w), "kv.v": make([]uint32, vwords)}
				var bufs []backend.Buf
				held := map[string]backend.Buf{}
				get := func(key string, words []uint32) backend.Buf {
					if bf, ok := held[key]; ok {
						return bf
					}
					bf, err := d.Alloc(max(4*len(words), 4))
					if err != nil {
						t.Fatal(err)
					}
					p := make([]byte, 4*len(words))
					for i, x := range words {
						binary.LittleEndian.PutUint32(p[4*i:], x)
					}
					if len(p) > 0 {
						if err := bf.Write(p); err != nil {
							t.Fatal(err)
						}
					}
					held[key], bufs = bf, append(bufs, bf)
					return bf
				}
				op, err := compileLone(d, codes)
				if err != nil {
					t.Fatal(err)
				}
				args := make([][]backend.Buf, len(codes))
				for ci, cd := range codes {
					for i, bd := range cd.Binds {
						o := bd.Operand
						var key string
						var words []uint32
						switch {
						case o.Kind == composer.OperandValue && vals[o.Value] != nil:
							key, words = fmt.Sprintf("v%d", o.Value), vals[o.Value]
						case o.Kind == composer.OperandTable && tables[o.Name] != nil:
							key, words = "t"+o.Name, tables[o.Name]
						case bd.Words != nil:
							key, words = fmt.Sprintf("c%d.%d", ci, i), bd.Words
						case o.Kind == composer.OperandScratch || o.Kind == composer.OperandValue:
							key, words = fmt.Sprintf("s%d.%s.%d", o.Kind, o.Name, o.Value), make([]uint32, max(bd.Bytes, 4)/4)
						default:
							t.Fatalf("%s binds %+v, which the probe does not hold", cd.Name, o)
						}
						args[ci] = append(args[ci], get(key, words))
					}
				}
				out := held[fmt.Sprintf("v%d", g.Out[0])]
				sync := func() { out.Read(make([]byte, 4)) }
				one := func(ci int) {
					if err := op.kerns[ci].Launch(codes[ci].LiveGroups(R), codes[ci].GPU.Group[0], args[ci]...); err != nil {
						t.Fatal(err)
					}
				}
				time1 := func(f func()) float64 {
					sync()
					t0 := time.Now()
					for range reps {
						f()
					}
					sync()
					return float64(time.Since(t0).Microseconds()) / reps
				}
				med := func(v []float64) float64 {
					s := append([]float64(nil), v...)
					sort.Float64s(s)
					return s[len(s)/2]
				}
				for ci, cd := range codes {
					one(ci)
					var us []float64
					for range rounds {
						us = append(us, time1(func() { one(ci) }))
					}
					res := ""
					if r, ok := backend.KernelResourcesOf(op.kerns[ci]); ok {
						res = fmt.Sprintf("regs %d, spill %d, shared %d, resident %d", r.Regs, r.SpillBytes, r.SharedBytes, r.Resident)
					}
					t.Logf("%s %v %dq%dkv%dd f32 values %v rows %d at %d: %-24s %v %4d workgroups of %4d: %8.1f us  %s", d.Name(), fl, c.heads, c.kvh, c.dim,
						c.f32v, R, p0, cd.Name, cd.Passes[len(cd.Passes)-1], cd.LiveGroups(R), cd.GPU.Group[0], med(us), res)
				}
				op.close()
				for _, bf := range bufs {
					bf.Free()
				}
			}
		}
	}
}

// flashConfigs is the flash forms the probe times: JITLLM_FLASH_SHAPES, a
// comma list of ROWS:KEYS:SHARED, or the profile's alone.
func flashConfigs(t testing.TB) []composer.Flash {
	env := os.Getenv("JITLLM_FLASH_SHAPES")
	if env == "" {
		return []composer.Flash{{}}
	}
	var out []composer.Flash
	for _, f := range strings.Split(env, ",") {
		var v []int
		for _, x := range strings.Split(f, ":") {
			n, err := strconv.Atoi(x)
			if err != nil {
				t.Fatalf("JITLLM_FLASH_SHAPES %q: %v", f, err)
			}
			v = append(v, n)
		}
		if len(v) != 3 {
			t.Fatalf("JITLLM_FLASH_SHAPES %q: want ROWS:KEYS:SHARED", f)
		}
		out = append(out, composer.Flash{Rows: v[0], Keys: v[1], Shared: v[2], Half: true})
	}
	return out
}
