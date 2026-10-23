//go:build jitllmbench

package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/dev/bench"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// benchModel is a decode attention shape the paged benches measure.
type benchModel struct {
	name                     string
	heads, kv, dim, attnLays int
	f16                      bool
}

var benchModels = []benchModel{
	{"Llama-3.2-1B", 32, 8, 64, 16, false},
	{"Qwen3.5-0.8B", 8, 2, 256, 6, false},
}

// benchDepths is 512/4096/16384 keys, or JITLLM_PAGED_DEPTH (a list).
func benchDepths() []int {
	dd := os.Getenv("JITLLM_PAGED_DEPTH")
	if dd == "" {
		return []int{512, 4096, 16384}
	}
	var out []int
	for _, f := range strings.Split(dd, ",") {
		var x int
		fmt.Sscanf(f, "%d", &x)
		out = append(out, x)
	}
	return out
}

// benchDevices is every paged-bench device JITLLM_PAGED_API and _DEV select,
// each with its read wall measured in this process.
func benchDevices(t *testing.T) (devs []backend.Device, walls []float64) {
	for _, d := range pagedDevices(t) {
		t.Cleanup(d.Close)
		if api := os.Getenv("JITLLM_PAGED_API"); api != "" && api != d.API() {
			continue
		}
		if dn := os.Getenv("JITLLM_PAGED_DEV"); dn != "" && !strings.Contains(d.Name(), dn) {
			continue
		}
		wall := deviceWall(t, d)
		t.Logf("%s/%s: read wall %.1f GB/s, %d slots", d.API(), d.Name(), wall/1e9, d.Slots())
		devs, walls = append(devs, d), append(walls, wall)
	}
	return devs, walls
}

// TestPagedAttentionAB is the paged decode kernels' performance bar
// (docs/design/device-kv-paging.md): the paged kernel against the contiguous
// one at equal history lengths, same pass, A/A control first, two ABBA passes,
// with the device's read wall measured in the same process.
//
// Each arm owns `layers` independent KV histories and a launch reads the next
// one, as a token walks its layers, so the KV is not served from L2 between
// launches. The contiguous arm's K is transposed at stride n+1, as the tier's
// is; the paged arm's pages are shuffled in its pool, P=256.
//
//	JITLLM_PAGED_API=ptx|spirv|msl   one backend
//	JITLLM_PAGED_DEV=<substring>     one device by name
//	JITLLM_PAGED_KERN=kv|flash       one kernel family
//	JITLLM_PAGED_DEPTH=4096          history lengths, comma separated
//	JITLLM_PAGED_MODEL=Llama         one model's shape
//	JITLLM_PAGED_PAGE=256            the page size
//	JITLLM_PAGED_KSTRIDE=n           the contiguous arm's K stride (n+1)
//	JITLLM_PAGED_FASPLITS=1          FlashAttention's key partitions (the
//	                                 tier's default is 1)
func TestPagedAttentionAB(t *testing.T) {
	gpuLock(t)
	devs, walls := benchDevices(t)
	if len(devs) == 0 {
		t.Fatal("no GPU")
	}
	for i, d := range devs {
		for _, kv := range []bool{true, false} {
			if k := os.Getenv("JITLLM_PAGED_KERN"); (k == "kv" && !kv) || (k == "flash" && kv) {
				continue
			}
			kind := map[bool]string{true: "kv", false: "flash"}[kv]
			for _, m := range benchModels {
				if mn := os.Getenv("JITLLM_PAGED_MODEL"); mn != "" && !strings.Contains(m.name, mn) {
					continue
				}
				for _, n := range benchDepths() {
					t.Run(fmt.Sprintf("%s/%s/kv%v/%s/n%d", d.API(), d.Name(), kv, m.name, n), func(t *testing.T) {
						e := newDecodeEnv(t, d, m, n, walls[i])
						defer e.g.free()
						e.ab(t, flashArm{name: "contiguous", kind: kind}, flashArm{name: "paged", kind: kind, paged: true})
					})
				}
			}
		}
	}
}

// TestFlashChangeAB is one decode-attention change against the form before
// it: JITLLM_FLASH_EXP names the experiment (flashExperiments), and each
// model and depth runs A/A first, then two ABBA passes of B against A.
func TestFlashChangeAB(t *testing.T) {
	gpuLock(t)
	exp := os.Getenv("JITLLM_FLASH_EXP")
	mk, ok := flashExperiments[strings.SplitN(exp, "=", 2)[0]]
	if !ok {
		t.Skipf("JITLLM_FLASH_EXP=%q names no experiment", exp)
	}
	arg := 0
	if i := strings.IndexByte(exp, '='); i > 0 {
		fmt.Sscanf(exp[i+1:], "%d", &arg)
	}
	devs, walls := benchDevices(t)
	if len(devs) == 0 {
		t.Fatal("no GPU")
	}
	for i, d := range devs {
		for _, m := range benchModels {
			if mn := os.Getenv("JITLLM_PAGED_MODEL"); mn != "" && !strings.Contains(m.name, mn) {
				continue
			}
			for _, n := range benchDepths() {
				t.Run(fmt.Sprintf("%s/%s/%s/%s/n%d", exp, d.API(), d.Name(), m.name, n), func(t *testing.T) {
					e := newDecodeEnv(t, d, m, n, walls[i])
					defer e.g.free()
					a, b := mk(e, arg)
					e.ab(t, a, b)
				})
			}
		}
	}
}

// flashExperiments are TestFlashChangeAB's comparisons, (before, after).
var flashExperiments = map[string]func(e *decodeEnv, arg int) (flashArm, flashArm){
	// The contiguous FlashAttention's V loads grouped ahead of their FMAs.
	"vgroup": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		return flashArm{name: "interleaved", kind: "flash", knobs: map[string]int{"flash-interleaved": 1}},
			flashArm{name: "grouped", kind: "flash", knobs: map[string]int{"flash-vgroup": arg}}
	},
	// The paged FlashAttention's V group size (keys a group): arg against 64/dims.
	"pvgroup": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		return flashArm{name: "paged", kind: "flash", paged: true},
			flashArm{name: fmt.Sprintf("paged-g%d", arg), kind: "flash", paged: true, knobs: map[string]int{"flash-vgroup": arg}}
	},
	// Paged FlashDecodeKV at arg splits against the tier's split.
	"kvsplit": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		return flashArm{name: "kv-tier", kind: "kv", paged: true},
			flashArm{name: fmt.Sprintf("kv-s%d", arg), kind: "kv", paged: true, splits: arg}
	},
	// FlashDecodeKV's V as vector loads, paged (arg 0) or contiguous (arg 1).
	"vecv": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		vec := func(s *kernels.FlashShape) { s.VecV = true }
		return flashArm{name: "kv-scalar", kind: "kv", paged: arg == 0},
			flashArm{name: "kv-vecv", kind: "kv", paged: arg == 0, tweak: vec}
	},
	// binary16 K against f32 K: paged FlashDecodeKV (arg 0) or the paged
	// staged sequence (arg 1).
	"f16k": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		kind := map[int]string{0: "kv", 1: "staged"}[arg]
		f16k := func(s *kernels.FlashShape) { s.F16K = true }
		return flashArm{name: kind + "-f32k", kind: kind, paged: true},
			flashArm{name: kind + "-f16k", kind: kind, paged: true, tweak: f16k}
	},
	// Paged FlashDecodeKV's partials merged by FlashAttentionMergeWide, at arg
	// splits (0: the tier's), against the tier's split and merge.
	"widemerge": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		return flashArm{name: "kv-tier", kind: "kv", paged: true},
			flashArm{name: fmt.Sprintf("kv-wide-s%d", arg), kind: "kv", paged: true, splits: arg, wideMerge: true}
	},
	// Paged FlashDecodeKV serving arg query heads a workgroup (Group) against
	// flashKVGroup's choice, each at the tier's split for its chunk.
	"kvgroup": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		return flashArm{name: "kv-tier", kind: "kv", paged: true},
			flashArm{name: fmt.Sprintf("kv-g%d", arg), kind: "kv", paged: true, tweak: func(s *kernels.FlashShape) { s.Group = arg }}
	},
	// Paged FlashDecodeKV at arg warps a workgroup against the default.
	"kvwarps": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		return flashArm{name: "kv-tier", kind: "kv", paged: true},
			flashArm{name: fmt.Sprintf("kv-w%d", arg), kind: "kv", paged: true, tweak: func(s *kernels.FlashShape) { s.Warps = arg }}
	},
	// Paged FlashDecodeKV configured by JITLLM_FLASH_B (warps=, splits=,
	// group=, wide=1, vecv=1, f16k=1, comma separated) against the tier's.
	"kvcfg": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		b := flashArm{name: "kv-" + os.Getenv("JITLLM_FLASH_B"), kind: "kv", paged: true}
		var warps, group int
		var vecv, f16k bool
		for _, kv := range strings.Split(os.Getenv("JITLLM_FLASH_B"), ",") {
			var k string
			var v int
			if i := strings.IndexByte(kv, '='); i > 0 {
				k = kv[:i]
				fmt.Sscanf(kv[i+1:], "%d", &v)
			}
			switch k {
			case "warps":
				warps = v
			case "splits":
				b.splits = v
			case "group":
				group = v
			case "wide":
				b.wideMerge = v == 1
			case "vecv":
				vecv = v == 1
			case "f16k":
				f16k = v == 1
			}
		}
		b.tweak = func(s *kernels.FlashShape) { s.Warps, s.Group, s.VecV, s.F16K = warps, group, vecv, f16k }
		return flashArm{name: "kv-tier", kind: "kv", paged: true}, b
	},
	// kernels.FlashKVPlan's shape and the wide merge against the tier's rule.
	// arg 1: both contiguous, as the tier runs FlashDecodeKV.
	"kvplan": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		return flashArm{name: "kv-tier", kind: "kv", paged: arg == 0}, flashArm{name: "kv-plan", kind: "kv", paged: arg == 0, plan: true}
	},
	// kernels.PagedStagedPlan's splits against sqrt(n/8).
	"stplan": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		return flashArm{name: "staged-sqrt", kind: "staged", paged: true}, flashArm{name: "staged-plan", kind: "staged", paged: true, plan: true}
	},
	// The two paths, each at its plan: >1 is staged faster.
	"pathplan": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		return flashArm{name: "kv-plan", kind: "kv", paged: true, plan: true}, flashArm{name: "staged-plan", kind: "staged", paged: true, plan: true}
	},
	// kernels.ChooseDecodePlan against what the tier runs: FlashDecodeKV
	// at its rule on CUDA and Metal, the staged path at sqrt(n/8) splits on
	// Vulkan.
	"choose": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		now := flashArm{name: "tier-now", kind: "kv", paged: true}
		if e.d.API() == "spirv" {
			now.kind = "staged"
		}
		return now, flashArm{name: "chosen", choose: true}
	},
	// Paged FlashDecodeKV with its descriptor baked (the row's table at 0,
	// keys [0, n)): the ceiling of hoisting the first dependent load.
	"constdesc": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		return flashArm{name: "kv-desc", kind: "kv", paged: true},
			flashArm{name: "kv-baked", kind: "kv", paged: true, knobs: map[string]int{"kv-constdesc": e.n}}
	},
	// Paged staged at arg splits against sqrt(n/8).
	"stsplit": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		return flashArm{name: "staged-sqrt", kind: "staged", paged: true},
			flashArm{name: fmt.Sprintf("staged-s%d", arg), kind: "staged", paged: true, splits: arg}
	},
	// Paged staged (splits arg, 0 the sqrt rule) against paged FlashDecodeKV at
	// the tier's split: which decode path a depth should take.
	"path": func(e *decodeEnv, arg int) (flashArm, flashArm) {
		return flashArm{name: "kv-tier", kind: "kv", paged: true},
			flashArm{name: "staged", kind: "staged", paged: true, splits: arg}
	},
}

// flashArm is one decode attention configuration: the kernel family (kv,
// flash, or the staged paged sequence), paged or contiguous, a split count
// (0: the tier's choice) and the bench knobs its kernels are generated under.
type flashArm struct {
	name, kind string
	paged      bool
	splits     int
	wideMerge  bool // FlashAttentionMergeWide in place of FlashAttentionMerge
	// plan takes the shape from kernels.FlashKVPlan or PagedStagedPlan (and
	// the wide merge) instead of the tier's rule.
	plan bool
	// choose takes the path and shape from kernels.ChooseDecodePlan.
	choose bool
	knobs  map[string]int
	tweak  func(*kernels.FlashShape)
}

// decodeEnv is one shape's buffers, shared by every arm: `layers` independent
// KV histories in both layouts, a launch reading the next one as a token walks
// its layers, so the KV is not served from L2 between launches.
type decodeEnv struct {
	d                      backend.Device
	g                      *gpu
	m                      benchModel
	n, page, layers, iters int
	wall                   float64
	bytes                  map[string]float64 // an arm's cache bytes a launch
	q, nb, out, part, desc backend.Buf
	ck, cv, pk, pv, tabs   []backend.Buf
	// pk16 is each layer's K as binary16 pairs (FlashShape.F16K), built
	// when an arm first asks.
	pk16                    []backend.Buf
	scores, probs           backend.Buf
	partFloats, planeFloats int
}

func newDecodeEnv(t *testing.T, d backend.Device, m benchModel, n int, wall float64) *decodeEnv {
	page := 256
	fmt.Sscanf(os.Getenv("JITLLM_PAGED_PAGE"), "%d", &page)
	kvRow := m.kv * m.dim
	vb := 4
	if m.f16 {
		vb = 2
	}
	perLayer := n * kvRow * (4 + vb)
	e := &decodeEnv{d: d, g: newGPU(t, d), m: m, n: n, page: page, wall: wall, bytes: map[string]float64{}}
	e.layers = max(1, min(m.attnLays, (256<<20)/perLayer))
	e.iters = max(20, min(400, (400<<20)/perLayer))
	rng := rand.New(rand.NewSource(int64(n)))
	rnd := func(c int) []float32 {
		a := make([]float32, c)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		return a
	}
	g := e.g
	e.q = g.up(f32bytes(rnd(m.heads * m.dim)))
	e.nb = g.up(u32bytes([]uint32{uint32(n)}))
	e.out = g.up(make([]byte, m.heads*m.dim*4))
	// Partials for up to 512 splits of either kernel.
	e.partFloats = m.heads * 512 * ((m.dim+31)/32*32 + 64)
	e.part = g.up(make([]byte, e.partFloats*4))
	npages := (n + page - 1) / page
	vbytes := func(c int) []byte {
		if m.f16 {
			return make([]byte, c*2)
		}
		return f32bytes(rnd(c))
	}
	for l := 0; l < e.layers; l++ {
		e.ck = append(e.ck, g.up(f32bytes(rnd(kvRow*(n+1)))))
		e.cv = append(e.cv, g.up(vbytes(n*kvRow)))
		e.pk = append(e.pk, g.up(f32bytes(rnd(npages*page*kvRow))))
		e.pv = append(e.pv, g.up(vbytes(npages*page*kvRow)))
		perm := rng.Perm(npages)
		tab := make([]uint32, npages)
		for i := range tab {
			tab[i] = uint32(perm[i])
		}
		e.tabs = append(e.tabs, g.up(u32bytes(tab)))
	}
	e.desc = g.up(u32bytes([]uint32{0, 0, uint32(n), uint32(n - 1)}))
	return e
}

// kPool is layer l's paged K for s: f32, or binary16 pairs, which are
// generated the first time an arm asks.
func (e *decodeEnv) kPool(s kernels.FlashShape, l int) backend.Buf {
	if !s.F16K {
		return e.pk[l]
	}
	if e.pk16 == nil {
		rng := rand.New(rand.NewSource(int64(e.n) + 1))
		words := (e.n + e.page - 1) / e.page * e.page * e.m.kv * e.m.dim / 2
		for range e.pk {
			b := make([]byte, words*4)
			for i := 0; i < 2*words; i++ {
				binary.LittleEndian.PutUint16(b[2*i:], quant.EncodeHalf(float32(rng.NormFloat64())))
			}
			e.pk16 = append(e.pk16, e.g.up(b))
		}
	}
	return e.pk16[l]
}

// armBytes is what one launch of an arm over s reads from the cache.
func (e *decodeEnv) armBytes(s kernels.FlashShape) float64 {
	kb, vb := 4, 4
	if s.F16K {
		kb = 2
	}
	if s.F16 {
		vb = 2
	}
	return float64(e.n * e.m.kv * e.m.dim * (kb + vb))
}

// baseShape is the model's decode shape, one row.
func (e *decodeEnv) baseShape() kernels.FlashShape {
	return kernels.FlashShape{Heads: e.m.heads, KVHeads: e.m.kv, Dim: e.m.dim, Rows: 1,
		Scale: float32(1 / math.Sqrt(float64(e.m.dim))), F16: e.m.f16}
}

// compile builds k under the arm's knobs and compiles it.
func (e *decodeEnv) compile(t *testing.T, a flashArm, build func() (*ir.Kernel, error)) backend.Kernel {
	for n, v := range a.knobs {
		kernels.SetBenchKnob(n, v)
	}
	k, err := build()
	for n := range a.knobs {
		kernels.SetBenchKnob(n, 0)
	}
	if err != nil {
		t.Fatal(err)
	}
	if ok, why := backend.GuaranteedLanes(e.d, k.Lanes); !ok {
		t.Skip(why)
	}
	c, err := e.d.Compile(k)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.Close)
	return c
}

// arm builds a's kernels and returns its timed case.
func (e *decodeEnv) arm(t *testing.T, a flashArm) bench.Case {
	s := e.baseShape()
	if a.tweak != nil {
		a.tweak(&s)
	}
	var chosen kernels.FlashShape
	if a.choose {
		p := kernels.ChooseDecodePlan(e.d.API(), s, e.n, e.d.Slots())
		t.Logf("%s: ChooseDecodePlan takes %v", a.name, p.Path)
		a.plan, a.paged, a.kind, chosen = true, true, "kv", p.Shape
		if p.Path == kernels.PathStaged {
			a.kind = "staged"
		}
	}
	if a.kind == "staged" {
		return e.stagedArm(t, a, s)
	}
	kv := a.kind == "kv"
	s.Splits = a.splits
	if a.choose {
		s = chosen
		a.wideMerge = true
	} else if a.plan && kv {
		s = kernels.FlashKVPlan(s, e.n, e.d.Slots())
		a.wideMerge = true
	} else if s.Splits == 0 {
		s.Splits = splitsFor(e.d, s, kv, e.n)
	}
	if kv {
		// The kernel's floor: Splits*FlashKVChunk covers the keys.
		sh := s
		sh.Splits = 1
		s.Splits = max(s.Splits, (e.n+kernels.FlashKVChunk(sh)-1)/kernels.FlashKVChunk(sh))
	}
	if a.paged {
		s.Page = e.page
		// The tier's per-API choice (kernels.FlashShape.KImm).
		s.KImm = e.d.API() == "msl"
	} else {
		s.KStride = e.n + 1
		fmt.Sscanf(os.Getenv("JITLLM_PAGED_KSTRIDE"), "%d", &s.KStride)
	}
	if kernels.FlashPartialFloats(s) > e.partFloats {
		t.Fatalf("%s: %d splits overrun the partials", a.name, s.Splits)
	}
	build, groups, width := kernels.FlashAttention, s.Heads*max(1, s.Splits), 32
	if kv {
		build, groups, width = kernels.FlashDecodeKV, kernels.FlashKVGroups(s), kernels.FlashKVWidth(s)
	}
	kc := e.compile(t, a, func() (*ir.Kernel, error) { return build(s) })
	var merge backend.Kernel
	if s.Splits > 1 {
		mk := kernels.FlashAttentionMerge
		if a.wideMerge {
			mk = kernels.FlashAttentionMergeWide
		}
		merge = e.compile(t, a, func() (*ir.Kernel, error) { return mk(s) })
	}
	dst := e.out
	if merge != nil {
		dst = e.part
	}
	t.Logf("%s: splits %d, %d groups of %d", a.name, s.Splits, groups, width)
	d := e.d
	kpool := make([]backend.Buf, e.layers)
	for l := range kpool {
		if a.paged {
			kpool[l] = e.kPool(s, l)
		}
	}
	e.bytes[a.name] = e.armBytes(s)
	return bench.Case{Name: a.name, Fn: func(iters int) {
		d.Session(func(se backend.Session) {
			for i := 0; i < iters; i++ {
				l := i % e.layers
				var err error
				if a.paged {
					err = se.Launch(kc, groups, width, e.q, kpool[l], e.pv[l], e.nb, dst, e.tabs[l], e.desc)
				} else {
					err = se.Launch(kc, groups, width, e.q, e.ck[l], e.cv[l], e.nb, dst)
				}
				if err == nil && merge != nil {
					if a.wideMerge {
						err = se.Launch(merge, (s.Heads*s.Dim+127)/128, 128, e.part, e.out)
					} else {
						err = se.Launch(merge, s.Heads, 32, e.part, e.out)
					}
				}
				if err != nil {
					panic(err)
				}
			}
			if err := se.Sync(); err != nil {
				panic(err)
			}
		})
	}}
}

// stagedArm is the paged staged sequence: scores, the per-split softmax, the
// accumulate and the wide merge, at a.splits splits (0: sqrt(n/8), as
// TestPagedStagedAB takes it).
func (e *decodeEnv) stagedArm(t *testing.T, a flashArm, s kernels.FlashShape) bench.Case {
	S := a.splits
	if S == 0 {
		S = 1 << int(math.Log2(math.Sqrt(max(1, float64(e.n)/8))))
	}
	s.Page, s.Splits, s.Chunk = e.page, S, ((e.n+S-1)/S+31)/32*32
	if a.plan {
		s = kernels.PagedStagedPlan(s, e.n, e.d.Slots())
		S = s.Splits
	}
	lanes := 1
	if ok, _ := backend.GuaranteedLanes(e.d, 32); ok {
		lanes = 32
	}
	ms := s
	ms.Page, ms.Chunk = 0, 0
	if kernels.FlashPartialFloats(ms) > e.partFloats {
		t.Fatalf("%s: %d splits overrun the partials", a.name, S)
	}
	if plane := kernels.PagedStagedPlane(s); plane > e.planeFloats {
		e.scores, e.probs = e.g.up(make([]byte, plane*4)), e.g.up(make([]byte, plane*4))
		e.planeFloats = plane
	}
	pScore := e.compile(t, a, func() (*ir.Kernel, error) { return kernels.PagedStagedScores(s, 1) })
	pSoft := e.compile(t, a, func() (*ir.Kernel, error) { return kernels.PagedStagedSoftmax(s, lanes) })
	pAcc := e.compile(t, a, func() (*ir.Kernel, error) { return kernels.PagedStagedAcc(s) })
	pMerge := e.compile(t, a, func() (*ir.Kernel, error) { return kernels.FlashAttentionMergeWide(ms) })
	smG := (s.Heads*S + 1) / 2
	if lanes == 1 {
		smG = (s.Heads*S + 63) / 64
	}
	t.Logf("%s: staged S=%d C=%d, softmax lanes %d", a.name, S, s.Chunk, lanes)
	kpool := make([]backend.Buf, e.layers)
	for l := range kpool {
		kpool[l] = e.kPool(s, l)
	}
	e.bytes[a.name] = e.armBytes(s)
	d, scores, probs := e.d, e.scores, e.probs
	return bench.Case{Name: a.name, Fn: func(iters int) {
		d.Session(func(se backend.Session) {
			var err error
			launch := func(c backend.Kernel, groups, width int, args ...backend.Buf) {
				if err == nil {
					err = se.Launch(c, groups, width, args...)
				}
			}
			for i := 0; i < iters; i++ {
				l := i % e.layers
				launch(pScore, (kernels.PagedStagedScoreItems(s)+127)/128, 128, e.q, kpool[l], e.nb, scores, e.tabs[l], e.desc)
				launch(pSoft, smG, 64, scores, e.nb, probs, e.part, e.tabs[l], e.desc)
				launch(pAcc, (kernels.PagedStagedAccThreads(s)+127)/128, 128, probs, e.pv[l], e.nb, e.part, e.tabs[l], e.desc)
				launch(pMerge, (s.Heads*s.Dim+127)/128, 128, e.part, e.out)
			}
			if err == nil {
				err = se.Sync()
			}
			if err != nil {
				panic(err)
			}
		})
	}}
}

// ab measures B against A: warm-up, the A/A control, two passes.
func (e *decodeEnv) ab(t *testing.T, aa, ba flashArm) {
	a, b := e.arm(t, aa), e.arm(t, ba)
	for until := time.Now().Add(2 * time.Second); time.Now().Before(until); {
		a.Fn(e.iters)
		b.Fn(e.iters)
	}
	gbs := func(name string, dt time.Duration) string {
		r := e.bytes[name] * float64(e.iters) / dt.Seconds()
		return fmt.Sprintf("%.1f GB/s = %.0f%% of wall", r/1e9, 100*r/e.wall)
	}
	ctl := bench.AB(a, a, 20, e.iters)
	t.Logf("%d layers, %d iters; A/A control %s", e.layers, e.iters, ctl)
	if !ctl.Stable() || math.Abs(ctl.Median-1) > .03 {
		t.Logf("A/A control outside 3%%: the passes below are not quotable")
	}
	for pass := 0; pass < 2; pass++ {
		r := bench.AB(a, b, 20, e.iters)
		t.Logf("pass%d %s (>1: %s faster); %s %s, %s %s", pass, r, ba.name, aa.name, gbs(aa.name, r.AMedian), ba.name, gbs(ba.name, r.BMedian))
	}
}

// deviceWall is the device's streaming read rate over 256 MiB, bytes/s: the
// best of five timed passes of kernels.StreamWide.
func deviceWall(t *testing.T, d backend.Device) float64 {
	const words, per, w = 64 << 20, 64, 4
	k, err := kernels.StreamWide(per, w)
	if err != nil {
		t.Fatal(err)
	}
	c, err := d.Compile(k)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	g := newGPU(t, d)
	defer g.free()
	threads := words / per
	in, nb, out := g.up(make([]byte, words*4)), g.up(u32bytes([]uint32{uint32(threads)})), g.up(make([]byte, threads*4))
	best := time.Duration(math.MaxInt64)
	for pass := 0; pass < 6; pass++ {
		t0 := time.Now()
		d.Session(func(s backend.Session) {
			for i := 0; i < 10; i++ {
				if e := s.Launch(c, threads/256, 256, in, nb, out); e != nil {
					panic(e)
				}
			}
			if e := s.Sync(); e != nil {
				panic(e)
			}
		})
		if el := time.Since(t0); pass > 0 && el < best {
			best = el
		}
	}
	return 10 * words * 4 / best.Seconds()
}

// splitsFor is the tier's split choice for n keys (tier/flash.go).
func splitsFor(d backend.Device, s kernels.FlashShape, kv bool, n int) int {
	if !kv {
		sp := 1
		fmt.Sscanf(os.Getenv("JITLLM_PAGED_FASPLITS"), "%d", &sp)
		return sp
	}
	s.Splits = 1
	slots := d.Slots()
	if slots <= 0 {
		slots = 16384
	}
	cp := max(1, slots/1280/max(1, kernels.FlashKVGroups(s)))
	sp := 1 << int(math.Round(math.Log2(math.Sqrt(max(1, float64(n)/8)))))
	sp = max(1, min(sp, cp))
	c := kernels.FlashKVChunk(s)
	return max(sp, (n+c-1)/c)
}
