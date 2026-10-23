package backend_test

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/ir"
)

// TestActInIsTheTwoLaunches: a matvec with the operand prologue
// (kernels.ActIn) writes, bit for bit, what the launch it replaces followed by
// the plain matvec writes -- RMSNormQuantRows (the warp reduction on CUDA, the
// tree everywhere) or Quantize, then MatVec. Every split and in-group split,
// the residual (bias) and gate epilogues, the q/k/v segments, both amax
// windows, k down to three blocks (ragged against every group width), every
// output NaN-poisoned with a guard word after it.
func TestActInIsTheTwoLaunches(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		warps := []bool{false}
		if d.API() == "ptx" {
			warps = append(warps, true)
		}
		for _, c := range []struct {
			q    kernels.Quant
			k    int
			rows int
		}{
			{kernels.Q4_K, 2048, 512}, {kernels.Q6_K, 2048, 256}, {kernels.Q4_K, 8192, 128},
			{kernels.Q8_0, 96, 64}, {kernels.Q8_0, 2080, 128}, {kernels.Q4_K, 1536, 256},
		} {
			for _, sp := range []struct {
				split int
				grp   bool
			}{{1, false}, {2, false}, {4, true}, {16, true}, {32, true}} {
				for _, win := range []int{32, 256} {
					if c.k%win != 0 {
						continue
					}
					for _, in := range []kernels.ActIn{
						{On: true, Window: win},
						{On: true, Norm: true, Eps: 1e-5, Window: win},
						{On: true, Norm: true, Eps: 1e-6, AddOne: true, Window: win},
						{On: true, Gated: true, Act: kernels.ActSiLU, Window: win},
						{On: true, Gated: true, Act: kernels.ActGELU, Window: win},
					} {
						for _, warp := range warps {
							if warp && !in.Norm {
								continue
							}
							in.Warp = warp
							for _, epi := range []string{"plain", "res", "gate"} {
								s := kernels.MatVecShape{T: c.q, K: c.k, Rows: c.rows, Split: sp.split,
									GroupSplit: sp.grp, Bias: epi == "res", Gate: epi == "gate", GateAct: kernels.ActSiLU}
								name := fmt.Sprintf("%s/%v-%dx%d/s%d-g%v/w%d/norm%v-one%v-warp%v-gated%v-%v/%s", d.API(), c.q,
									c.rows, c.k, sp.split, sp.grp, win, in.Norm, in.AddOne, warp, in.Gated, in.Act, epi)
								if actInCase(t, d, name, []kernels.MatVecShape{s}, in) {
									ran++
								}
							}
							if sp.split == 1 || sp.grp {
								segs := []kernels.MatVecShape{{T: c.q, K: c.k, Rows: c.rows},
									{T: c.q, K: c.k, Rows: 64, Bias: true}, {T: kernels.Q8_0, K: c.k, Rows: 64}}
								name := fmt.Sprintf("%s/seg-%v-%d/s%d-g%v/w%d/norm%v-warp%v-gated%v", d.API(), c.q, c.k,
									sp.split, sp.grp, win, in.Norm, warp, in.Gated)
								s0 := segs[0]
								s0.Split, s0.GroupSplit = sp.split, sp.grp
								if actInCase(t, d, name, append([]kernels.MatVecShape{s0}, segs...), in) {
									ran++
								}
							}
						}
					}
				}
			}
		}
	}
	if ran == 0 {
		t.Fatal("no case ran")
	}
	t.Logf("%d cases bit-identical", ran)
}

// actInCase runs one case. shapes is one matvec, or the segment kernel's
// split carrier followed by its segments. It reports whether the case was a
// shape the kernels build; a refusal of the reference is not a failure.
func actInCase(t *testing.T, d backend.Device, name string, shapes []kernels.MatVecShape, in kernels.ActIn) bool {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(len(name)) * 7919))
	seg := len(shapes) > 1
	split, grp := shapes[0].Split, shapes[0].GroupSplit
	if seg {
		shapes = shapes[1:]
	}
	k := shapes[0].K
	g := newGPU(t, d)
	defer g.free()
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64()) * float32(1+i%5)
	}
	nw := make([]float32, k)
	for i := range nw {
		nw[i] = float32(rng.NormFloat64())
	}
	bX, bNW := g.up(f32bytes(x)), g.up(f32bytes(nw))
	// nw doubles as the up row of a gated operand.
	bU := bNW
	const guard = 0x5EED5EED
	poisoned := func(n int) backend.Buf {
		p := make([]float32, n+1)
		for i := range p {
			p[i] = float32(math.NaN())
		}
		b := f32bytes(p)
		binary.LittleEndian.PutUint32(b[n*4:], guard)
		return g.up(b)
	}
	// The reference operand: the launch the prologue replaces.
	var ref *ir.Kernel
	var err error
	switch {
	case in.Norm && in.Warp:
		ref, err = kernels.RMSNormQuantRowsWarp(k, 1, in.Eps, in.AddOne, false, in.Window)
	case in.Norm:
		ref, err = kernels.RMSNormQuantRows(k, 1, in.Eps, in.AddOne, false, in.Window)
	case in.Gated:
		ref, err = kernels.QuantizeAct(k, in.Window, in.Act)
	default:
		ref, err = kernels.Quantize(k, in.Window)
	}
	if err != nil {
		t.Logf("%s: no reference (%v)", name, err)
		return false
	}
	rk, err := d.Compile(ref)
	if err != nil {
		t.Fatalf("%s: compile reference: %v", name, err)
	}
	defer rk.Close()
	bA, bAX := poisoned(k/4), poisoned(k/32+k/16)
	if in.Norm {
		err = rk.Launch(1, kernels.RMSNormGroup, bX, bNW, poisoned(k), bA, bAX)
	} else if in.Gated {
		err = rk.Launch((kernels.QuantizeThreads(k/32)+127)/128, 128, bX, bU, poisoned(k), bA, bAX)
	} else {
		err = rk.Launch((kernels.QuantizeThreads(k/32)+127)/128, 128, bX, bA, bAX)
	}
	if err != nil {
		t.Fatalf("%s: reference: %v", name, err)
	}
	type w struct{ qs, dw, sc, extra, sep, got backend.Buf }
	ws := make([]w, len(shapes))
	width := 128
	if grp {
		width = kernels.GroupSplitWidth(split)
	}
	// run reports false for a kernel the device refuses at its width
	// (backend.ErrGroupTooWide): the tier then keeps the two launches.
	run := func(kk *ir.Kernel, groups int, bufs ...backend.Buf) bool {
		t.Helper()
		if err := kk.Validate(); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		kern, err := d.Compile(kk)
		if errors.Is(err, backend.ErrGroupTooWide) {
			t.Logf("%s: not a shape on this device (%v)", name, err)
			return false
		}
		if err != nil {
			t.Fatalf("%s: compile: %v", name, err)
		}
		defer kern.Close()
		if err := kern.Launch(groups, width, bufs...); err != nil {
			t.Fatalf("%s: launch: %v", name, err)
		}
		return true
	}
	for i, s := range shapes {
		qs, dw, scw, err := kernels.PackWeights(s.T, rawWeights(s.T, s.Rows, k, rng), s.Rows, k)
		if err != nil {
			t.Fatal(err)
		}
		ws[i] = w{qs: g.up(u32bytes(qs)), dw: g.up(u32bytes(dw)), sc: g.up(u32bytes(scw)),
			sep: poisoned(s.Rows), got: poisoned(s.Rows)}
		if s.Bias || s.Gate {
			e := make([]float32, s.Rows)
			for j := range e {
				e[j] = float32(rng.NormFloat64())
			}
			ws[i].extra = g.up(f32bytes(e))
		}
		one := s
		one.Split, one.GroupSplit = split, grp
		kk, err := kernels.MatVec(one)
		if err != nil {
			t.Logf("%s: not a shape (%v)", name, err)
			return false
		}
		bufs := []backend.Buf{ws[i].qs, ws[i].dw, ws[i].sc, bA, bAX, ws[i].sep}
		if ws[i].extra != nil {
			bufs = append(bufs, ws[i].extra)
		}
		if split > 1 && !grp {
			// A partial-writing split: the fused and reference arms write the
			// same partial plane, compared as the output.
			ws[i].sep, ws[i].got = poisoned(s.Rows*split), poisoned(s.Rows*split)
			bufs[5] = ws[i].sep
		}
		if !seg {
			if !run(kk, (s.Rows*split+width-1)/width, bufs...) {
				return false
			}
			one.In = in
			fk, err := kernels.MatVec(one)
			if err != nil {
				t.Fatalf("%s: the prologue refuses a shape its matvec takes: %v", name, err)
			}
			fb := []backend.Buf{ws[i].qs, ws[i].dw, ws[i].sc, bX}
			if in.Norm {
				fb = append(fb, bNW)
			}
			if in.Gated {
				fb = append(fb, bU)
			}
			fb = append(fb, ws[i].got)
			if ws[i].extra != nil {
				fb = append(fb, ws[i].extra)
			}
			if !run(fk, (s.Rows*split+width-1)/width, fb...) {
				return false
			}
		} else if !run(kk, (s.Rows*split+width-1)/width, bufs...) {
			return false
		}
	}
	if seg {
		kk, err := kernels.MatVecSegmentsIn(shapes, split, grp, in)
		if err != nil {
			t.Logf("%s: not a segment shape (%v)", name, err)
			return false
		}
		blocks, _, err := kernels.SegmentsGrid(shapes, split, grp)
		if err != nil {
			t.Fatal(err)
		}
		total := 0
		for _, n := range blocks {
			total += n
		}
		bufs := []backend.Buf{bX}
		if in.Norm {
			bufs = append(bufs, bNW)
		}
		if in.Gated {
			bufs = append(bufs, bU)
		}
		for i := range shapes {
			bufs = append(bufs, ws[i].qs, ws[i].dw, ws[i].sc, ws[i].got)
			if ws[i].extra != nil {
				bufs = append(bufs, ws[i].extra)
			}
		}
		if !run(kk, total, bufs...) {
			return false
		}
	}
	for i, s := range shapes {
		n := s.Rows
		if split > 1 && !grp {
			n *= split
		}
		want, got := make([]byte, (n+1)*4), make([]byte, (n+1)*4)
		if ws[i].sep.Read(want) != nil || ws[i].got.Read(got) != nil {
			t.Fatalf("%s: read back", name)
		}
		bad := 0
		var num, den float64
		for j := 0; j <= n; j++ {
			a, b := binary.LittleEndian.Uint32(want[4*j:]), binary.LittleEndian.Uint32(got[4*j:])
			fa, fb := float64(math.Float32frombits(a)), float64(math.Float32frombits(b))
			if j < n && math.IsNaN(fa) {
				t.Fatalf("%s: output %d row %d of the reference is NaN -- a degenerate oracle", name, i, j)
			}
			if j < n && !bitExact(d) {
				// SPIR-V carries no NoContraction and MSL compiles with fast
				// math, so a driver may contract the prologue's arithmetic
				// differently from the separate kernel's: held to a band
				// there, bit for bit on PTX.
				if math.IsNaN(fb) {
					t.Fatalf("%s: output %d row %d is NaN", name, i, j)
				}
				num += (fa - fb) * (fa - fb)
				den += fa * fa
				continue
			}
			if a != b {
				if bad++; bad <= 2 {
					t.Errorf("%s: output %d word %d = %#x, want %#x", name, i, j, b, a)
				}
			}
		}
		if bad > 0 {
			t.Fatalf("%s: output %d: %d of %d words differ", name, i, bad, n+1)
		}
		if den > 0 && num/den > 1e-6 {
			t.Fatalf("%s: output %d: NMSE %.3g against the two launches", name, i, num/den)
		}
	}
	return true
}

// bitExact: the backends whose lowering pins the float contraction, where a
// fused kernel must reproduce its parts' bits.
func bitExact(d backend.Device) bool { return d.API() == "ptx" }
