package backend_test

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// flashForm is which flash prefill kernel a gate builds.
type flashForm int

const (
	flash70Form   flashForm = iota // FlashPrefill70, m8n8k4
	flashTileForm                  // FlashPrefillTile, Metal's simdgroup matrices
	flash80Form                    // FlashPrefill80, m16n8k16
	flash80K8Form                  // FlashPrefill80K8, m16n8k8 (sm_75)
)

// formOf is the form a tile flag names: the tile, else sm_70's.
func formOf(tile bool) flashForm {
	if tile {
		return flashTileForm
	}
	return flash70Form
}

// flashPrefillBuild builds one flash prefill kernel for a shape in form, with
// its launch.
func flashPrefillBuild(t *testing.T, fs kernels.FlashPrefill70Shape, form flashForm) (*ir.Kernel, int, int) {
	t.Helper()
	if form == flash80Form || form == flash80K8Form {
		build := kernels.FlashPrefill80
		if form == flash80K8Form {
			build = kernels.FlashPrefill80K8
		}
		k, err := build(fs)
		if err != nil {
			t.Fatal(err)
		}
		return k, kernels.FlashPrefill80Groups(fs), kernels.FlashPrefill80Threads
	}
	if form == flash70Form {
		k, err := kernels.FlashPrefill70(fs)
		if err != nil {
			t.Fatal(err)
		}
		return k, kernels.FlashPrefill70Groups(fs), kernels.FlashPrefill70Threads
	}
	tl, ok := kernels.FlashTileFor(fs.Dim)
	if !ok {
		t.Fatalf("no tile blocking at head width %d", fs.Dim)
	}
	k, err := kernels.FlashPrefillTile(fs, tl)
	if err != nil {
		t.Fatal(err)
	}
	return k, kernels.FlashPrefillTileGroups(fs, tl), tl.Threads()
}

// pagedFlashPrefill runs a paged flash prefill kernel (and, at Splits > 0, the
// merge, with the sink when sink) over the pool through tab and desc, and
// returns the output and its NMSE against the oracle.
func pagedFlashPrefill(t *testing.T, d backend.Device, fs kernels.FlashPrefill70Shape, sink bool, form flashForm, p *pagedPool, tab, desc []uint32, rows []pagedRow) ([]float64, float64, error) {
	t.Helper()
	fs.Rows = len(rows)
	k, groups, threads := flashPrefillBuild(t, fs, form)
	// The kernel under test is the paged one.
	if !strings.Contains(k.Name, "paged") || k.Params[len(k.Params)-1].Name != "pRow" || k.Params[len(k.Params)-2].Name != "pTab" {
		t.Fatalf("not the paged kernel: %s %v", k.Name, k.Params)
	}
	ms := kernels.FlashShape{Heads: fs.Heads, KVHeads: fs.KVHeads, Dim: fs.Dim, Rows: fs.Rows, Splits: max(1, fs.Splits),
		Scale: fs.Scale, Softcap: fs.Softcap, Sink: sink}
	cs := []*ir.Kernel{k}
	if fs.Splits > 0 {
		mk, err := kernels.FlashAttentionMergeWide(ms)
		if err != nil {
			t.Fatal(err)
		}
		cs = append(cs, mk)
	}
	var cc []backend.Kernel
	for _, k := range cs {
		c, err := d.Compile(k)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		cc = append(cc, c)
	}
	g := newGPU(t, d)
	defer g.free()
	q, sinks := prefillInputs(ms)
	nOut := fs.Rows * fs.Heads * fs.Dim
	qo, ko, vo := g.up(f32bytes(q)), g.up(f32bytes(p.k)), g.up(p.vBytes())
	no := g.up(u32bytes([]uint32{0xFFFFFFFF, 0xFFFFFFFF}))
	to, ro := g.up(u32bytes(tab)), g.up(u32bytes(desc))
	oo := g.up(f32bytes(nanFill(nOut + 32)))
	dst := oo
	if fs.Splits > 0 {
		dst = g.up(f32bytes(nanFill(kernels.FlashPartialFloats(ms))))
	}
	so := g.up(f32bytes(sinks))
	var err error
	d.Session(func(se backend.Session) {
		err = se.Launch(cc[0], groups, threads, qo, ko, vo, no, dst, to, ro)
		if err == nil && fs.Splits > 0 {
			ma := []backend.Buf{dst, oo}
			if sink {
				ma = append(ma, so)
			}
			err = se.Launch(cc[1], (nOut+127)/128, 128, ma...)
		}
		if err == nil {
			err = se.Sync()
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	got := readF32(t, oo, nOut+32)
	for i := nOut; i < len(got); i++ {
		if !math.IsNaN(got[i]) {
			return nil, 0, fmt.Errorf("output guard %d overwritten", i)
		}
	}
	nmse, err := prefillNMSE(ms, p, rows, q, sinks, got[:nOut])
	return got[:nOut], nmse, err
}

// contigFlash runs the contiguous FlashPrefill70 (or FlashPrefillTile) on the
// same history and queries with the case's sliding window baked in.
func contigFlash(t *testing.T, d backend.Device, fs kernels.FlashPrefill70Shape, form flashForm, p *pagedPool, rows []pagedRow, window int) []float64 {
	t.Helper()
	kc, vc, L, kStride := contigCache(p, rows)
	fs.Rows, fs.Page, fs.Splits, fs.F16, fs.KStride, fs.Window = len(rows), 0, 0, false, kStride, window
	k, groups, threads := flashPrefillBuild(t, fs, form)
	c, err := d.Compile(k)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	g := newGPU(t, d)
	defer g.free()
	q, _ := prefillInputs(kernels.FlashShape{Heads: fs.Heads, Dim: fs.Dim, Rows: fs.Rows})
	nOut := fs.Rows * fs.Heads * fs.Dim
	out := g.up(f32bytes(nanFill(nOut)))
	if err := c.Launch(groups, threads, g.up(f32bytes(q)), g.up(f32bytes(kc)), g.up(f32bytes(vc)),
		g.up(u32bytes(contigCounts(rows, L))), out); err != nil {
		t.Fatal(err)
	}
	return readF32(t, out, nOut)
}

// flashPrefillShapes are the flash prefill gates' geometries: GQA at both head
// widths, packed f16 V, a softcap, a sink (applied by the merge), one KV head.
var flashPrefillShapes = []struct {
	fs   kernels.FlashPrefill70Shape
	sink bool
}{
	{kernels.FlashPrefill70Shape{Heads: 16, KVHeads: 8, Dim: 128, Scale: .088}, false},
	{kernels.FlashPrefill70Shape{Heads: 32, KVHeads: 8, Dim: 64, Scale: .125, F16: true}, false},
	{kernels.FlashPrefill70Shape{Heads: 8, KVHeads: 4, Dim: 128, Scale: .088, Softcap: 50}, false},
	{kernels.FlashPrefill70Shape{Heads: 8, KVHeads: 1, Dim: 128, Scale: .088}, true},
	{kernels.FlashPrefill70Shape{Heads: 8, KVHeads: 2, Dim: 64, Scale: .125, F16: true}, true},
}

// TestPagedFlashPrefill holds the paged FlashPrefill70 (m8n8k4: sm_70, and
// sm_86 runs it too) and FlashPrefillTile (the collective tiles: Metal's simdgroup matrices, PTX's mma.sync) to the
// oracle over every prefill case, at no split, one (the merge normalising and
// adding the sink) and more than the keys need (empty runs); to their
// contiguous kernels where those can express the case; and demands failure
// through a wrong table and keyEnds one short.
func TestPagedFlashPrefill(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	for _, d := range devs {
		defer d.Close()
	}
	ran := 0
	for _, tile := range []bool{false, true} {
		var tdevs []backend.Device
		if tile {
			for _, d := range devs {
				if ok, _ := backend.LowersTile(d, tile8f32); ok {
					tdevs = append(tdevs, d)
				}
			}
		} else {
			tdevs = mmaDevices(t, devs, ir.MMAVolta())
		}
		for _, d := range tdevs {
			t.Run(fmt.Sprintf("tile%v/%s/%s", tile, d.API(), d.Name()), func(t *testing.T) {
				for _, page := range []int{64, 256} {
					for _, sh := range flashPrefillShapes {
						for _, c := range prefillCases(page, 64) {
							rows := prefillRows(c)
							for _, splits := range []int{0, 1, 3, 40} {
								if sh.sink && splits == 0 {
									continue
								}
								fs := sh.fs
								fs.Page, fs.Splits = page, splits
								t.Run(fmt.Sprintf("P%d/%s/h%d-%d/d%d/f16%v/cap%g/sink%v/S%d", page, c.name, fs.Heads, fs.KVHeads, fs.Dim, fs.F16, fs.Softcap, sh.sink, splits), func(t *testing.T) {
									p := newPrefillPool(rand.New(rand.NewSource(int64(page+fs.Dim+c.pos0))), page, fs.KVHeads*fs.Dim, fs.F16, false, false, rows)
									got, nmse, err := pagedFlashPrefill(t, d, fs, sh.sink, formOf(tile), p, p.tab, p.desc, rows)
									if err != nil || nmse > 1e-5 {
										t.Fatalf("shuffled pages: NMSE %.3g, %v", nmse, err)
									}
									if _, bad, err := pagedFlashPrefill(t, d, fs, sh.sink, formOf(tile), p, p.wrong, p.desc, rows); err == nil && bad < 1e-4 {
										t.Fatalf("a wrong table passed (NMSE %.3g): the table is not read", bad)
									}
									if _, bad, err := pagedFlashPrefill(t, d, fs, sh.sink, formOf(tile), p, p.tab, shortDesc(p.desc), rows); err == nil && bad < 1e-4 {
										t.Fatalf("keyEnds one short passed (NMSE %.3g)", bad)
									}
									msg := fmt.Sprintf("NMSE %.3g", nmse)
									// The contiguous kernel has no sink, no f16 V and no chunked window.
									if !sh.sink && !fs.F16 && c.window >= 0 {
										ref := contigFlash(t, d, fs, formOf(tile), p, rows, c.window)
										n := realRowsNMSE(got, ref, rows)
										if math.IsNaN(n) || n > 1e-5 {
											t.Fatalf("paged against contiguous: NMSE %.3g", n)
										}
										msg += fmt.Sprintf(", against contiguous %.3g", n)
									}
									t.Log(msg)
								})
								ran++
							}
						}
					}
				}
			})
		}
	}
	if ran == 0 {
		t.Skip("no device lowers m8n8k4 or the tile ops")
	}
}
