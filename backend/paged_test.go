package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// The gates for the paged KV kernels (docs/design/device-kv-paging.md). Every
// pool is built with its pages SHUFFLED, every word nobody wrote is NaN --
// page tails, pages below a row's window, the gaps between tables, the dummy
// page on the read side -- and a row's keys cross page edges and start
// mid-page. Each gate also runs the same pool through a WRONG table and
// demands that it fails, which is what says the table was read.

var nan32 = float32(math.NaN())

// pagedRow is one query row: its keys [ks, ke) and, for the writers, the
// position it writes. A padded row has ks == ke.
type pagedRow struct{ ks, ke, wp int }

// pagedPool is a host image of one layer's K and V pools and the tables that
// address them.
type pagedPool struct {
	page, kvRow, pages int
	f16, mla, f16k     bool
	k, v               []float32 // v holds the f16-rounded values when f16
	tab, desc          []uint32
	// kh/vh are each row's logical history, [t][kvRow].
	kh, vh [][]float32
	dummy  int // the zero page padded rows point at
	poison int // a NaN page no row owns
	// wrong is the table with every entry moved to the next page id: a
	// kernel that does not read the table cannot tell it from tab.
	wrong []uint32
}

func (p *pagedPool) kAt(pid, t, e int) int {
	if p.mla {
		return p.vAt(pid, t, e)
	}
	return pid*p.page*p.kvRow + e*p.page + t%p.page
}
func (p *pagedPool) vAt(pid, t, e int) int { return (pid*p.page+t%p.page)*p.kvRow + e }

// newPagedPool lays rows' histories into shuffled pages. Pages a row's window
// starts past are mapped to the poison page, and every table is followed by a
// poison entry, so reading outside one's own table or below one's window
// reads NaN.
func newPagedPool(rng *rand.Rand, page, kvRow int, f16 bool, rows []pagedRow) *pagedPool {
	return newPagedPoolOrder(rng, page, kvRow, f16, rows, false, false)
}

// newPagedPoolOrder is newPagedPool with the pages in order when ordered:
// the identity table, which proves nothing alone and must pass too. mla lays
// K out row-major, the MLA region, whose value is its row's prefix: vh is kh.
func newPagedPoolOrder(rng *rand.Rand, page, kvRow int, f16 bool, rows []pagedRow, ordered, mla bool) *pagedPool {
	p := &pagedPool{page: page, kvRow: kvRow, f16: f16, mla: mla}
	need := 0
	for _, r := range rows {
		need += (max(r.ke, r.wp+1) + page - 1) / page
	}
	p.pages = need + 3
	perm := rng.Perm(p.pages)
	if ordered {
		for i := range perm {
			perm[i] = (i + p.pages - 2) % p.pages // rows from page 0, dummy and poison last
		}
	}
	p.dummy, p.poison = perm[0], perm[1]
	next := 2
	n := p.pages * page * kvRow
	p.k, p.v = make([]float32, n), make([]float32, n)
	for i := range p.k {
		p.k[i], p.v[i] = nan32, nan32
	}
	// The dummy page is zero-filled: padded rows may write it, nobody reads it.
	for t := 0; t < page; t++ {
		for e := 0; e < kvRow; e++ {
			p.k[p.kAt(p.dummy, t, e)], p.v[p.vAt(p.dummy, t, e)] = 0, 0
		}
	}
	for _, r := range rows {
		off := len(p.tab)
		kh, vh := make([]float32, r.ke*kvRow), make([]float32, r.ke*kvRow)
		if r.ke == r.ks {
			p.tab = append(p.tab, uint32(p.dummy))
		} else {
			for j := 0; j*page < max(r.ke, r.wp+1); j++ {
				if (j+1)*page <= r.ks {
					p.tab = append(p.tab, uint32(p.poison))
					continue
				}
				p.tab = append(p.tab, uint32(perm[next]))
				next++
			}
		}
		for t := r.ks; t < r.ke; t++ {
			pid := int(p.tab[off+t/page])
			for e := 0; e < kvRow; e++ {
				kv, vv := float32(rng.NormFloat64()), float32(rng.NormFloat64())
				if f16 {
					vv = float32(quant.DecodeHalf(quant.EncodeHalf(vv)))
				}
				if mla {
					vv = kv
				}
				kh[t*kvRow+e], vh[t*kvRow+e] = kv, vv
				p.k[p.kAt(pid, t, e)], p.v[p.vAt(pid, t, e)] = kv, vv
			}
		}
		p.kh, p.vh = append(p.kh, kh), append(p.vh, vh)
		p.desc = append(p.desc, uint32(off), uint32(r.ks), uint32(r.ke), uint32(r.wp))
		p.tab = append(p.tab, uint32(p.poison))
	}
	p.wrong = make([]uint32, len(p.tab))
	for i, x := range p.tab {
		p.wrong[i] = (x + 1) % uint32(p.pages)
	}
	return p
}

// withF16K makes p's K binary16 (FlashShape.F16K): every value rounded, and
// kBytes packs the pairs. NaN stays NaN.
func (p *pagedPool) withF16K() *pagedPool {
	p.f16k = true
	round := func(x float32) float32 { return float32(quant.DecodeHalf(quant.EncodeHalf(x))) }
	for i, x := range p.k {
		p.k[i] = round(x)
	}
	for _, kh := range p.kh {
		for i, x := range kh {
			kh[i] = round(x)
		}
	}
	return p
}

// kBytes is K as the device holds it: f32 [kvRow][P] a page, or binary16 with
// dimensions e and e+1 of a position one word, [kvRow/2][P] words a page.
func (p *pagedPool) kBytes() []byte {
	if !p.f16k {
		return f32bytes(p.k)
	}
	out := make([]byte, len(p.k)*2)
	for pid := 0; pid < p.pages; pid++ {
		for t := 0; t < p.page; t++ {
			for e := 0; e < p.kvRow; e++ {
				w := pid*p.page*p.kvRow/2 + e/2*p.page + t
				binary.LittleEndian.PutUint16(out[4*w+2*(e%2):], quant.EncodeHalf(p.k[p.kAt(pid, t, e)]))
			}
		}
	}
	return out
}

// vBytes is V as the device holds it: f32, or binary16 two to a word.
func (p *pagedPool) vBytes() []byte {
	if !p.f16 {
		return f32bytes(p.v)
	}
	out := make([]byte, len(p.v)*2)
	for i, x := range p.v {
		binary.LittleEndian.PutUint16(out[2*i:], quant.EncodeHalf(x))
	}
	return out
}

// pagedWant is the attention every row should produce, in float64: keys
// [ks, ke), sinks and softcap as FlashAttention defines them. An empty row
// is 0.
func pagedWant(s kernels.FlashShape, p *pagedPool, rows []pagedRow, q, sinks []float32) []float64 {
	vd := s.Dim
	if s.MLA > 0 {
		vd = s.MLA
	}
	want := make([]float64, len(rows)*s.Heads*vd)
	gqa := s.Heads / s.KVHeads
	for r, row := range rows {
		for h := 0; h < s.Heads; h++ {
			kvh := h / gqa
			mx := math.Inf(-1)
			if s.Sink {
				mx = float64(sinks[h])
			}
			sc := make([]float64, row.ke-row.ks)
			for t := row.ks; t < row.ke; t++ {
				var d float64
				for x := 0; x < s.Dim; x++ {
					d += float64(q[(r*s.Heads+h)*s.Dim+x]) * float64(p.kh[r][t*p.kvRow+kvh*s.Dim+x])
				}
				d *= float64(s.Scale)
				if s.Softcap > 0 {
					d = float64(s.Softcap) * math.Tanh(d/float64(s.Softcap))
				}
				sc[t-row.ks] = d
				mx = math.Max(mx, d)
			}
			var sum float64
			for i := range sc {
				sc[i] = math.Exp(sc[i] - mx)
				sum += sc[i]
			}
			if s.Sink {
				sum += math.Exp(float64(sinks[h]) - mx)
			}
			if sum == 0 {
				continue
			}
			for x := 0; x < vd; x++ {
				var acc float64
				for t := row.ks; t < row.ke; t++ {
					acc += sc[t-row.ks] * float64(p.vh[r][t*p.kvRow+kvh*s.Dim+x])
				}
				want[(r*s.Heads+h)*vd+x] = acc / sum
			}
		}
	}
	return want
}

// pagedAttn runs one paged decode kernel (FlashDecodeKV when kv, else
// FlashAttention) plus the merge over rows, reading through tab, and returns
// the NMSE against the oracle, or an error for a non-finite or overrun output.
func pagedAttn(t *testing.T, d backend.Device, s kernels.FlashShape, kv bool, p *pagedPool, tab []uint32, rows []pagedRow) (float64, error) {
	t.Helper()
	s.Rows = len(rows)
	build, groups, width := kernels.FlashAttention, s.Rows*s.Heads*max(1, s.Splits), 32
	if kv {
		build = kernels.FlashDecodeKV
		groups, width = kernels.FlashKVGroups(s), kernels.FlashKVWidth(s)
	}
	ker, err := build(s)
	if err != nil {
		t.Fatal(err)
	}
	// The kernel under test is the paged one, not a contiguous kernel given
	// the same buffers.
	if ker.Name != map[bool]string{true: "flashkvpaged", false: "flashpaged"}[kv] || len(ker.Params) < 7 || ker.Params[5].Name != "pTab" {
		t.Fatalf("not the paged kernel: %s %v", ker.Name, ker.Params)
	}
	c, err := d.Compile(ker)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	g := newGPU(t, d)
	defer g.free()
	rng := rand.New(rand.NewSource(97))
	q := make([]float32, s.Rows*s.Heads*s.Dim)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	sinks := make([]float32, s.Heads)
	for h := range sinks {
		sinks[h] = float32(h%4) - 1.5
	}
	guard := 32
	outInit := make([]float32, len(q)+guard)
	for i := range outInit {
		outInit[i] = nan32
	}
	qo, ko, vo, oo := g.up(f32bytes(q)), g.up(p.kBytes()), g.up(p.vBytes()), g.up(f32bytes(outInit))
	// pN is not read by a paged kernel: a NaN-count buffer says so.
	no := g.up(u32bytes([]uint32{0xFFFFFFFF, 0xFFFFFFFF}))
	to, ro := g.up(u32bytes(tab)), g.up(u32bytes(p.desc))
	dst := oo
	if s.Splits > 1 {
		parts := make([]float32, kernels.FlashPartialFloats(s))
		for i := range parts {
			parts[i] = nan32
		}
		dst = g.up(f32bytes(parts))
	}
	args := []backend.Buf{qo, ko, vo, no, dst, to, ro}
	var so backend.Buf
	if s.Sink {
		so = g.up(f32bytes(sinks))
		args = append(args, so)
	}
	if err := c.Launch(groups, width, args...); err != nil {
		t.Fatal(err)
	}
	if s.Splits > 1 {
		mk, e := kernels.FlashAttentionMerge(s)
		if e != nil {
			t.Fatal(e)
		}
		mc, e := d.Compile(mk)
		if e != nil {
			t.Fatal(e)
		}
		defer mc.Close()
		ma := []backend.Buf{dst, oo}
		if s.Sink {
			ma = append(ma, so)
		}
		if e = mc.Launch(s.Rows*s.Heads, 32, ma...); e != nil {
			t.Fatal(e)
		}
	}
	got := readF32(t, oo, len(outInit))
	for i := len(q); i < len(got); i++ {
		if !math.IsNaN(got[i]) {
			return 0, fmt.Errorf("output guard %d overwritten", i)
		}
	}
	want := pagedWant(s, p, rows, q, sinks)
	var se, ss float64
	for i, w := range want {
		a := got[i]
		if math.IsNaN(a) || math.IsInf(a, 0) {
			return math.Inf(1), fmt.Errorf("non-finite output %d (row %d)", i, i/(s.Heads*s.Dim))
		}
		se += (a - w) * (a - w)
		ss += w * w
	}
	return se / max(ss, 1e-30), nil
}

// pagedDevices is every GPU this host has, Vulkan's integrated one included
// and a software rasteriser (llvmpipe) not: the paged gates sweep thousands of
// shapes, which on a CPU device is hours of a correct answer the GPUs already
// gave. JITLLM_TEST_DEVICES narrows the sweep to its devices, and a software
// rasteriser it names is swept.
func pagedDevices(t *testing.T) []backend.Device {
	var out []backend.Device
	for _, d := range backend.Open() {
		if d.API() == "spirv" {
			d.Close()
			continue
		}
		out = append(out, d)
	}
	infos, err := backend.VulkanDevices()
	if err != nil {
		t.Logf("no Vulkan enumeration: %v", err)
		return out
	}
	for _, in := range infos {
		if !testdevices.Has(t, "vulkan", in.Index) {
			continue
		}
		if in.Software && !testdevices.Selected() {
			t.Logf("vulkan device %d (%s) is a software rasteriser: not swept (JITLLM_TEST_DEVICES=vulkan:%d runs it)",
				in.Index, in.Name, in.Index)
			continue
		}
		d, err := backend.OpenVulkan(strconv.Itoa(in.Index))
		if err != nil {
			t.Logf("vulkan device %d: %v", in.Index, err)
			continue
		}
		out = append(out, d)
	}
	return out
}

// laneOK reports whether d can run a kernel needing k.Lanes, logging why not.
func laneOK(t *testing.T, d backend.Device, k *ir.Kernel) bool {
	ok, why := backend.GuaranteedLanes(d, k.Lanes)
	if !ok {
		t.Logf("%s/%s: skips the %d-lane kernel: %s", d.API(), d.Name(), k.Lanes, why)
	}
	return ok
}

// pagedCases are the row sets every read gate runs: page edges crossed,
// windows starting mid-page and mid-tile, rows of different lengths in one
// launch, a padded row, and one row alone.
func pagedCases(page int) map[string][]pagedRow {
	return map[string][]pagedRow{
		"one-long":  {{0, 5*page + 17, 0}},
		"one-short": {{0, 1, 0}},
		"rows": {
			{0, 3*page + 5, 0},
			{page/2 + 7, 2*page + 1, 0},  // a window from mid-page and mid-tile
			{0, 0, 0},                    // padded
			{2*page - 1, 2*page + 40, 0}, // a window from a page's last slot
			{0, page, 0},                 // exactly one page
			{33, 34, 0},                  // one key, below it a masked tile
		},
	}
}

func TestPagedFlashDecode(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API()+"/"+d.Name(), func(t *testing.T) {
			for _, kv := range []bool{true, false} {
				for _, page := range []int{64, 256} {
					for _, sh := range []kernels.FlashShape{
						{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125},
						{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, Splits: 3, Sink: true, Softcap: 3},
						{Heads: 6, KVHeads: 6, Dim: 48, Scale: .14, F16: true, Sink: true},
						{Heads: 8, KVHeads: 1, Dim: 128, Scale: .088, Splits: 4, F16: true},
						{Heads: 4, KVHeads: 2, Dim: 7, Scale: .3, Splits: 2},
						{Heads: 16, KVHeads: 8, Dim: 256, Scale: .06, Splits: 2, Sink: true},
						// binary16 K: whole pairs a slice, an odd slice (34
						// over two slices of 17), f16 V beside it, and the
						// 256-wide head.
						{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, F16K: true, Softcap: 3},
						{Heads: 4, KVHeads: 2, Dim: 34, Scale: .17, Splits: 2, F16K: true},
						{Heads: 6, KVHeads: 6, Dim: 48, Scale: .14, F16: true, F16K: true, Sink: true},
						{Heads: 8, KVHeads: 2, Dim: 256, Scale: .06, Splits: 2, F16K: true, Sink: true},
					} {
						if sh.F16 && sh.Dim%2 != 0 {
							continue
						}
						sh.Page = page
						for name, rows := range pagedCases(page) {
							s := sh
							// Both forms of the K addresses, alternating by case.
							s.KImm = (len(name)+page/64)%2 == 0
							// Vector V loads where the backend lowers them, on
							// the other parity.
							s.VecV = d.API() != "spirv" && !s.KImm
							s.Rows = len(rows)
							if kv {
								// Splits*chunk covers the longest row's aligned span.
								c, longest := kernels.FlashKVChunk(s), 0
								for _, r := range rows {
									longest = max(longest, r.ke)
								}
								s.Splits = max(s.Splits, (longest+c-1)/c)
							}
							k, err := map[bool]func(kernels.FlashShape) (*ir.Kernel, error){true: kernels.FlashDecodeKV, false: kernels.FlashAttention}[kv](s)
							if err != nil {
								t.Fatal(err)
							}
							if !laneOK(t, d, k) {
								return
							}
							t.Run(fmt.Sprintf("kv%v/P%d/%s/h%d-%d/d%d/split%d/f16%v/f16k%v/sink%v/kimm%v/vec%v", kv, page, name, s.Heads, s.KVHeads, s.Dim, s.Splits, s.F16, s.F16K, s.Sink, s.KImm, s.VecV), func(t *testing.T) {
								rng := rand.New(rand.NewSource(int64(page + s.Dim)))
								p := newPagedPool(rng, page, s.KVHeads*s.Dim, s.F16, rows)
								if s.F16K {
									p.withF16K()
								}
								nmse, err := pagedAttn(t, d, s, kv, p, p.tab, rows)
								if err != nil || nmse > 1e-10 {
									t.Fatalf("shuffled pages: NMSE %.3g, %v", nmse, err)
								}
								ordered := newPagedPoolOrder(rand.New(rand.NewSource(int64(page+s.Dim))), page, s.KVHeads*s.Dim, s.F16, rows, true, false)
								if s.F16K {
									ordered.withF16K()
								}
								if n2, err := pagedAttn(t, d, s, kv, ordered, ordered.tab, rows); err != nil || n2 > 1e-10 {
									t.Fatalf("pages in order: NMSE %.3g, %v", n2, err)
								}
								// The same pool through a wrong table must fail:
								// otherwise the table was not what the kernel read.
								if bad, err := pagedAttn(t, d, s, kv, p, p.wrong, rows); err == nil && bad < 1e-6 {
									t.Fatalf("a wrong table over shuffled pages passed (NMSE %.3g): the table is not read", bad)
								} else {
									t.Logf("NMSE %.3g; wrong table: %.3g %v", nmse, bad, err)
								}
							})
							ran++
						}
					}
				}
			}
		})
	}
	if ran == 0 {
		t.Fatal("no paged decode case ran on any device")
	}
}
