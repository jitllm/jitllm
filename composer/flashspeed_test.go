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
// profile's. JITLLM_FLASH_AT puts the chunk at that position in place of
// 1536. It runs its own rounds:
//
//	go test ./composer -run '^$' -bench ComposedFlash -benchtime 1x
func BenchmarkComposedFlash(t *testing.B) {
	testlock.GPU(t)
	p0 := 1536
	if v, err := strconv.Atoi(os.Getenv("JITLLM_FLASH_AT")); err == nil && v >= 0 {
		p0 = v
	}
	for _, d := range openDevices() {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		for _, c := range []attendRowsCase{{32, 8, 64, 0, false, false}, {32, 8, 128, 0, false, false}, {32, 8, 64, 0, false, true}, {32, 8, 128, 0, false, true}, {16, 8, 256, 0, false, false}} {
			for _, fl := range flashConfigs(t) {
				dev := gpuDevice(d)
				dev.Matrix = [3]int{16, 16, 16}
				arms := []bool{dev.ScaleAdd}
				if dev.ScaleAdd && !dev.ScaleRows {
					arms = []bool{false, true}
				}
				for _, add := range arms {
					dev.ScaleAdd = add
					flashRun(t, d, flashRegistry(fl), dev, c, p0, 16)
				}
			}
		}
	}
}

// BenchmarkComposedFlashLatent is BenchmarkComposedFlash at a latent
// attention's geometry, DeepSeek-V2-Lite's: 16 query heads over one kv head
// whose binary16 row of 576 is its key and its value.
//
//	go test ./composer -run '^$' -bench ComposedFlashLatent -benchtime 1x
func BenchmarkComposedFlashLatent(t *testing.B) {
	testlock.GPU(t)
	p0 := 1536
	if v, err := strconv.Atoi(os.Getenv("JITLLM_FLASH_AT")); err == nil && v >= 0 {
		p0 = v
	}
	for _, d := range openDevices() {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		dev := gpuDevice(d)
		dev.Matrix = [3]int{16, 16, 16}
		flashRunWith(t, d, composer.Default(), dev, attendRowsCase{heads: 16, kvh: 1, dim: 576}, p0, 8, true, nil)
	}
}

// TestFlashScaleAddIsTheSharedSum holds the flash form whose weighted sums
// stay in tiles, each block's products added to them in one fused
// multiply-add (Device.ScaleAdd), to the form that carries them through
// workgroup memory: the same arithmetic, so every output word is equal, on
// every CUDA device that lowers the row scale and does not carry sums in its
// accumulator, at Llama's head width and Gemma 3's (sliced, flashSlices),
// with float32 and binary16 values and a chunk 13k positions in. The violation is the binary16 arm's values moved
// by one ulp at one position the chunk attends.
func TestFlashScaleAddIsTheSharedSum(t *testing.T) {
	testlock.GPU(t)
	ran := 0
	for _, d := range openDevices() {
		defer d.Close()
		dev := gpuDevice(d)
		if d.API() != "ptx" || !dev.ScaleAdd || dev.ScaleRows {
			continue
		}
		dev.Matrix = [3]int{16, 16, 16}
		for _, c := range []attendRowsCase{{32, 8, 128, 0, false, true}, {32, 8, 128, 0, false, false}, {32, 8, 64, 0, false, false}, {16, 8, 256, 0, false, false}} {
			for _, p0 := range []int{0, 12800} {
				t.Run(fmt.Sprintf("%s/%dq%dkv%dd/f32values%v/at%d", d.Name(), c.heads, c.kvh, c.dim, c.f32v, p0), func(t *testing.T) {
					shared := dev
					shared.ScaleAdd = false
					want := flashRun(t, d, composer.Default(), shared, c, p0, 0)
					got := flashRun(t, d, composer.Default(), dev, c, p0, 0)
					if n := differ(got, want); n > 0 {
						t.Fatalf("%d of %d output words differ from the form that sums through workgroup memory", n, len(want))
					}
					if !c.f32v && p0 > 0 {
						moved := flashRunWith(t, d, composer.Default(), dev, c, p0, 0, false, func(v []uint32) { v[len(v)/3] ^= 1 })
						if differ(moved, want) == 0 {
							t.Fatal("a value moved by one ulp changed no output word: the comparison sees nothing")
						}
					}
					ran++
				})
			}
		}
	}
	if ran == 0 {
		t.Skip("no CUDA device lowers the row scale without carrying sums in its accumulator")
	}
}

func differ(a, b []uint32) int {
	n := 0
	for i := range a {
		if a[i] != b[i] {
			n++
		}
	}
	return n + abs(len(a)-len(b))
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// flashRegistry is the default registry, its flash pass fl where fl states
// a shape.
func flashRegistry(fl composer.Flash) *composer.Registry {
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
	return reg
}

// flashRun forms case c's attention for a 512-row chunk at position p0 in
// the served models' pool layout on d, runs it on seeded inputs and returns
// its output's words; with rounds past zero it logs each code's median
// microseconds over that many rounds, and the kernels' resources.
func flashRun(t testing.TB, d backend.Device, reg *composer.Registry, dev composer.Device, c attendRowsCase, p0, rounds int) []uint32 {
	return flashRunWith(t, d, reg, dev, c, p0, rounds, false, nil)
}

// flashRunWith is flashRun with the value history's words passed through
// edit before they are written, where it is set; latent states a history
// whose values are its keys' rows, one kv head of binary16 rows (MLA's).
func flashRunWith(t testing.TB, d backend.Device, reg *composer.Registry, dev composer.Device, c attendRowsCase, p0, rounds int, latent bool, edit func([]uint32)) []uint32 {
	const page, R, reps = 64, 512, 5
	keys := (p0+R+page-1)/page*page + 2*page
	b := graph.New()
	hist := composer.History{Keys: composer.Pool{ByDim: true}, Values: composer.Pool{Half: !c.f32v}}
	var n int
	if latent {
		k := b.In("k", c.kvh*c.dim)
		n = b.KVAppend(k, k, 0)
		hist = composer.History{Keys: composer.Pool{Half: true}, Values: composer.Pool{Half: true}, ValuesInKeys: true}
	} else {
		n = b.KVAppend(b.In("k", c.kvh*c.dim), b.In("v", c.kvh*c.dim), 0)
	}
	scale := 1 / math.Sqrt(float64(c.dim))
	b.Out(b.Attend(b.In("q", c.heads*c.dim), n, graph.Attrs{NHead: c.heads, NKVHead: c.kvh, HeadDim: c.dim, Scale: scale}))
	g := b.G
	_, codes, err := reg.Form(composer.Request{Graph: g, Step: composer.Step{Rows: R, Page: page, Keys: keys, Seqs: 1},
		Device: dev, History: hist})
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
	vals := map[graph.ValueID][]uint32{}
	for _, in := range g.In {
		vals[in] = rnd(R * g.Values[in].Width)
	}
	vals[g.Out[0]] = make([]uint32, R*c.heads*c.dim)
	vpool := rnd(keys * w)
	if !c.f32v || latent {
		vpool = vpool[:keys*w/2]
		for i, x := range vpool {
			// Two binary16 values a word, each of a magnitude below one.
			vpool[i] = x&0xBBFFBBFF | 0x30003000
		}
	}
	if edit != nil {
		edit(vpool)
	}
	tables := map[string][]uint32{"rows": rows, "rows.n": {R}, "pages": pages, "kv.k": rnd(keys * w), "kv.v": vpool}
	if latent {
		tables["kv.k"] = vpool
	}
	var bufs []backend.Buf
	defer func() {
		for _, bf := range bufs {
			bf.Free()
		}
	}()
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
	defer op.close()
	args := make([][]backend.Buf, len(codes))
	scratch := scratchBytes(codes)
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
				key = scratchKey(o)
				words = make([]uint32, scratch[key]/4)
			default:
				t.Fatalf("%s binds %+v, which the probe does not hold", cd.Name, o)
			}
			args[ci] = append(args[ci], get(key, words))
		}
	}
	out := held[fmt.Sprintf("v%d", g.Out[0])]
	sync := func() {
		if err := out.Read(make([]byte, 4)); err != nil {
			t.Fatal(err)
		}
	}
	one := func(ci int) {
		if err := op.kerns[ci].Launch(codes[ci].LiveGroups(R), codes[ci].GPU.Group[0], args[ci]...); err != nil {
			t.Fatal(err)
		}
	}
	for ci := range codes {
		one(ci)
	}
	res := make([]byte, 4*R*c.heads*c.dim)
	if err := out.Read(res); err != nil {
		t.Fatal(err)
	}
	words := make([]uint32, len(res)/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(res[4*i:])
	}
	if rounds == 0 {
		return words
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
		var us []float64
		for range rounds {
			us = append(us, time1(func() { one(ci) }))
		}
		res := ""
		if r, ok := backend.KernelResourcesOf(op.kerns[ci]); ok {
			res = fmt.Sprintf("regs %d, spill %d, shared %d, resident %d", r.Regs, r.SpillBytes, r.SharedBytes, r.Resident)
		}
		t.Logf("%s %dq%dkv%dd f32 values %v scaleadd %v rows %d at %d: %-24s %v %4d workgroups of %4d: %8.1f us  %s", d.Name(), c.heads, c.kvh, c.dim,
			c.f32v, dev.ScaleAdd && !dev.ScaleRows, R, p0, cd.Name, cd.Passes[len(cd.Passes)-1], cd.LiveGroups(R), cd.GPU.Group[0], med(us), res)
	}
	return words
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
