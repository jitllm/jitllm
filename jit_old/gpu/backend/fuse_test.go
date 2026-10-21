package backend_test

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestRoPEQKVPagedMatchesItsParts holds the fused decode position writer
// (mainkernels.RoPEQKVPaged) to the three kernels it replaces -- RoPERows on q,
// PagedRoPERowsT on k, PagedCopyRows on v -- bit for bit: the rotated q, and
// every word of both pools, which start NaN so a write either arm missed or
// made in the wrong place shows. Heads of different counts for q and k (GQA,
// so the three parts have different lengths and the clamping is exercised),
// both pair layouts, f32 and packed-f16 V, one row (decode) and a batch whose
// rows land on either side of a page edge through a shuffled table.
func TestRoPEQKVPagedMatchesItsParts(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Skip("no GPU")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		for _, c := range []struct {
			nh, nkv, hd, nRot int
			neox, f16         bool
			pos               []int
		}{
			{32, 8, 64, 64, false, false, []int{77}},
			{32, 8, 64, 64, true, true, []int{63, 64, 200, 5}},
			{24, 8, 128, 96, true, false, []int{1}},
			{8, 8, 128, 128, false, true, []int{255, 256, 257}},
			{16, 2, 256, 256, true, true, []int{12}},
		} {
			name := fmt.Sprintf("%s/h%d-%d/d%d/rot%d/neox%v/f16%v/r%d", d.API(), c.nh, c.nkv, c.hd, c.nRot, c.neox, c.f16, len(c.pos))
			t.Run(name, func(t *testing.T) {
				fusedCase(t, d, c.nh, c.nkv, c.hd, c.nRot, c.neox, c.f16, c.pos)
			})
			ran++
		}
	}
	if ran == 0 {
		t.Skip("no device")
	}
}

func fusedCase(t *testing.T, d backend.Device, nh, nkv, hd, nRot int, neox, f16 bool, pos []int) {
	const page = 64
	R := len(pos)
	kvRow, qw := nkv*hd, nh*hd
	rng := rand.New(rand.NewSource(int64(nh*hd + nRot + R)))
	rnd := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		return x
	}
	q, k, v := rnd(R*qw), rnd(R*kvRow), rnd(R*kvRow)
	cs := make([]float32, R*nRot)
	for i := 0; i < len(cs); i += 2 {
		a := rng.Float64() * 6.28
		cs[i], cs[i+1] = float32(math.Cos(a)), float32(math.Sin(a))
	}
	// Every row its own shuffled pages up to its position.
	pages := 0
	for _, p := range pos {
		pages += p/page + 1
	}
	perm := rng.Perm(pages)
	var tab, desc []uint32
	for _, p := range pos {
		off := len(tab)
		for j := 0; j <= p/page; j++ {
			tab = append(tab, uint32(perm[len(tab)]))
		}
		desc = append(desc, uint32(off), 0, uint32(p+1), uint32(p))
	}
	poolWords := pages * page * kvRow
	nan := make([]float32, max(poolWords, R*qw))
	for i := range nan {
		nan[i] = nan32
	}
	qOff := make([]uint32, R)
	for r := range qOff {
		qOff[r] = uint32(r * qw)
	}
	type out struct{ q, k, v []byte }
	arm := func(fused bool) out {
		g := newGPU(t, d)
		defer g.free()
		qb, kb, vb := g.up(f32bytes(q)), g.up(f32bytes(k)), g.up(f32bytes(v))
		csb, offb := g.up(f32bytes(cs)), g.up(u32bytes(qOff))
		qd := g.up(f32bytes(nan[:R*qw]))
		kp, vp := g.up(f32bytes(nan[:poolWords])), g.up(f32bytes(nan[:poolWords]))
		tb, rb := g.up(u32bytes(tab)), g.up(u32bytes(desc))
		run := func(kk *ir.Kernel, err error, threads int, bufs ...backend.Buf) {
			if err != nil {
				t.Fatal(err)
			}
			c, err := d.Compile(kk)
			if err != nil {
				t.Fatalf("%s: %v", kk.Name, err)
			}
			defer c.Close()
			if err := c.Launch((threads+127)/128, 128, bufs...); err != nil {
				t.Fatal(err)
			}
		}
		pairs := nRot / 2
		if fused {
			kk, err := mainkernels.RoPEQKVPaged(nh, nkv, hd, nRot, neox, R, page, f16)
			run(kk, err, mainkernels.RoPEQKVPagedThreads(nh, nkv, hd, nRot, R, f16),
				qb, csb, offb, qd, kb, kp, vb, vp, tb, rb)
		} else {
			kk, err := mainkernels.RoPERows(nh, hd, nRot, neox, R)
			run(kk, err, R*nh*pairs, qb, csb, offb, qd)
			kk, err = mainkernels.PagedRoPERowsT(nkv, hd, nRot, neox, R, page)
			run(kk, err, R*nkv*pairs, kb, csb, offb, kp, tb, rb)
			kk, err = mainkernels.PagedCopyRows(kvRow, R, page, kvRow, 0, f16)
			nv := R * kvRow
			if f16 {
				nv /= 2
			}
			run(kk, err, nv, vb, vp, offb, tb, rb)
		}
		rd := func(b backend.Buf, n int) []byte {
			p := make([]byte, 4*n)
			if err := b.Read(p); err != nil {
				t.Fatal(err)
			}
			return p
		}
		return out{rd(qd, R*qw), rd(kp, poolWords), rd(vp, poolWords)}
	}
	want, got := arm(false), arm(true)
	for _, c := range []struct {
		name string
		w, g []byte
	}{{"q", want.q, got.q}, {"k pool", want.k, got.k}, {"v pool", want.v, got.v}} {
		if !bytes.Equal(c.w, c.g) {
			diff := 0
			for i := 0; i+4 <= len(c.w); i += 4 {
				if !bytes.Equal(c.w[i:i+4], c.g[i:i+4]) {
					diff++
				}
			}
			t.Fatalf("%s: %d of %d words differ from the three kernels", c.name, diff, len(c.w)/4)
		}
		// The selection check: the per-op arm must have written something,
		// or two arms that both wrote nothing agree.
		if bytes.Equal(c.w, f32bytes(nan[:len(c.w)/4])) {
			t.Fatalf("%s: nothing was written", c.name)
		}
	}
}
