package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/ir"
)

// depScale is out[i] = in[i]*2 + 1: elementwise, so a consumer of it waits
// tile by tile.
func depScale(t testing.TB, width int) *ir.Kernel {
	b := ir.New("depscale", [3]int{width, 1, 1})
	in := b.Param("in", ir.F32)
	out := b.Param("out", ir.F32)
	i := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	b.Store(out, i, b.Fma(b.Load(ir.F32, in, i, 0), b.ConstF32(2), b.ConstF32(1)), 0)
	return mustBuild(t, b)
}

// depSlow is depScale with group 0 first spinning through 1<<16 loads of its
// input: it finishes long after every other, and a consumer reading from
// element 0 up that does not wait for it reads its outputs before they are
// written.
func depSlow(t testing.TB, width int) *ir.Kernel {
	b := ir.New("depslow", [3]int{width, 1, 1})
	in := b.Param("in", ir.F32)
	out := b.Param("out", ir.F32)
	i := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	z := b.ConstF32(0)
	one := b.Const(ir.U32, 1)
	b.LoopN(b.Mul(ir.U32, b.Sub(ir.U32, one, b.Min(ir.U32, b.CTAID(), one)), b.Const(ir.U32, 1<<16)))
	acc := b.Phi(ir.F32, z)
	b.SetPhi(acc, b.Fma(b.Load(ir.F32, in, i, 0), b.ConstF32(1), acc))
	b.EndLoop()
	v := b.Fma(b.Load(ir.F32, in, i, 0), b.ConstF32(2), b.ConstF32(1))
	b.Store(out, i, b.Fma(acc, b.ConstF32(0), v), 0)
	return mustBuild(t, b)
}

// depSum is out[i] = in[i] + the sum of in[0..n): every workgroup reads every
// element, so a consumer of its input waits for the whole producer.
func depSum(t testing.TB, width, n int) *ir.Kernel {
	b := ir.New("depsum", [3]int{width, 1, 1})
	in := b.Param("in", ir.F32)
	out := b.Param("out", ir.F32)
	i := b.Add(ir.U32, b.Mul(ir.U32, b.CTAID(), b.NTID()), b.TID())
	acc0 := b.Load(ir.F32, in, i, 0)
	j0 := b.Const(ir.U32, 0)
	b.Loop(int64(n))
	j := b.Phi(ir.U32, j0)
	acc := b.Phi(ir.F32, acc0)
	b.SetPhi(acc, b.Fma(b.Load(ir.F32, in, j, 0), b.ConstF32(1), acc))
	b.SetPhi(j, b.Add(ir.U32, j, b.Const(ir.U32, 1)))
	b.EndLoop()
	b.Store(out, i, acc, 0)
	return mustBuild(t, b)
}

// TestDepCountersOrderTheChain composes a chain of tile and whole edges with
// a write after read (ir.PersistDeps) and holds every buffer to the host,
// poisoned before each of five launches (the epoch must carry the counters
// from one to the next), at two workgroups and at the co-residency limit,
// two kernels sharing one dep buffer at different regions, and inside a
// recorded graph. Its violation is TestDepCountersSeeABrokenWait.
func TestDepCountersOrderTheChain(t *testing.T) {
	ran := 0
	eachGridDevice(t, func(d backend.Device, width, limit int) {
		if d.API() != "ptx" {
			return
		}
		for _, grid := range dedupe(2, limit) {
			t.Run(fmt.Sprintf("%s/w%d/g%d", d.API(), width, grid), func(t *testing.T) {
				if bad := depChain(t, d, width, grid); bad != 0 {
					t.Fatalf("%d words differ from the host", bad)
				}
			})
			ran++
		}
	})
	if ran == 0 {
		t.Skip("no device lowers the dependency counters")
	}
}

// depChain runs the chain and returns how many words differ from the host.
func depChain(t *testing.T, d backend.Device, width, grid int) int {
	groups := 3 * grid
	n := groups * width
	sc, sl, sm := depScale(t, width), depSlow(t, width), depSum(t, width, n)
	// x0 -> scale -> x1 -> slow scale -> x2 -> sum -> x3 -> scale -> x4,
	// then scale x4 -> x1 (overwriting what the second part read).
	parts := []ir.Part{
		{K: sc, Groups: groups, Bind: []int{0, 1}},
		{K: sl, Groups: groups, Bind: []int{1, 2}},
		{K: sm, Groups: groups, Bind: []int{2, 3}},
		{K: sc, Groups: groups, Bind: []int{3, 4}},
		{K: sc, Groups: groups, Bind: []int{4, 1}},
	}
	alloc := func(words int) backend.Buf {
		b, err := d.Alloc(4 * words)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(b.Free)
		return b
	}
	pa, err := ir.PersistDeps("depchain", parts, width, grid, 0)
	if err != nil {
		t.Fatal(err)
	}
	whole, tile := 0, 0
	for _, o := range pa.K.Ops {
		if o.Kind == ir.OpDepWait {
			if o.Imm&ir.DepTile != 0 {
				tile++
			} else {
				whole++
			}
		}
	}
	// Both kinds of edge are what the chain is for: selected, not assumed.
	if whole != 1 || tile != 5 {
		t.Fatalf("%d whole and %d tile waits, want 1 and 5", whole, tile)
	}
	pb, err := ir.PersistDeps("depchain2", parts[:3], width, grid, pa.Words)
	if err != nil {
		t.Fatal(err)
	}
	ka, err := d.Compile(pa.K)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(ka.Close)
	kb, err := d.Compile(pb.K)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(kb.Close)
	if lim, err := backend.GridLimit(d, ka); err != nil || lim < grid {
		t.Skipf("the composed kernel holds %d workgroups resident (%v), not %d", lim, err, grid)
	}
	dep := alloc(pa.Words + pb.Words)
	if err := dep.Write(make([]byte, 4*(pa.Words+pb.Words))); err != nil {
		t.Fatal(err)
	}
	xs := make([]backend.Buf, 5)
	for i := range xs {
		xs[i] = alloc(n)
	}
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(i%7) - 3
	}
	want := depChainHost(in, n)
	args := func(p *ir.Persisted) []backend.Buf {
		var a []backend.Buf
		for _, x := range p.Bufs {
			a = append(a, xs[x])
		}
		return append(a, dep)
	}
	reset := func() {
		raw := make([]byte, 4*n)
		for i, v := range in {
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
		}
		if err := xs[0].Write(raw); err != nil {
			t.Fatal(err)
		}
		poison := wordBytes(poisonWords(n, 0x7FC0DEAD))
		for _, x := range xs[1:] {
			if err := x.Write(poison); err != nil {
				t.Fatal(err)
			}
		}
	}
	check := func() int {
		bad := 0
		raw := make([]byte, 4*n)
		for b := 1; b < 5; b++ {
			if err := xs[b].Read(raw); err != nil {
				t.Fatal(err)
			}
			for i := 0; i < n; i++ {
				if binary.LittleEndian.Uint32(raw[4*i:]) != math.Float32bits(want[b][i]) {
					bad++
				}
			}
		}
		return bad
	}
	bad := 0
	for rep := 0; rep < 5; rep++ {
		// The second kernel's region sits beside the first's in one buffer;
		// it runs first, and the outputs are poisoned again before the
		// first, so nothing it left can stand in for a word not yet written.
		reset()
		if err := kb.Launch(grid, width, args(pb)...); err != nil {
			t.Fatal(err)
		}
		reset()
		if err := ka.Launch(grid, width, args(pa)...); err != nil {
			t.Fatal(err)
		}
		bad += check()
	}
	reset()
	var serr error
	d.Session(func(s backend.Session) {
		rec, ok := s.(backend.Recorder)
		if !ok {
			if serr = s.Launch(ka, grid, width, args(pa)...); serr == nil {
				serr = s.Sync()
			}
			return
		}
		g, err := backend.Record(rec, func() { serr = s.Launch(ka, grid, width, args(pa)...) })
		if err != nil {
			serr = err
			return
		}
		defer g.Free()
		serr = rec.Replay(g)
		if serr == nil {
			serr = s.Sync()
		}
	})
	if serr != nil {
		t.Fatal(serr)
	}
	return bad + check()
}

// depChainHost is the chain's answer; want[1] is the last part's x1.
func depChainHost(in []float32, n int) [5][]float32 {
	var w [5][]float32
	for i := range w {
		w[i] = make([]float32, n)
	}
	copy(w[0], in)
	scale := func(dst, src []float32) {
		for i := range src {
			dst[i] = src[i]*2 + 1
		}
	}
	scale(w[1], w[0])
	scale(w[2], w[1])
	// Every sum adds the same elements in the same order after its own; the
	// values are small integers, so each partial sum is exact.
	var total float32
	for j := 0; j < n; j++ {
		total += w[2][j]
	}
	for i := 0; i < n; i++ {
		w[3][i] = w[2][i] + total
	}
	scale(w[4], w[3])
	scale(w[1], w[4])
	return w
}
