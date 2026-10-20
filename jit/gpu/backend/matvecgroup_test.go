package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestGroupEpiloguesAreTheSeparateLaunches: a few-sequence step's matvec with
// every token in one thread (tier.groupKern) writing the residual sum
// (BiasRows) or act(gate)*up (Gate) is, bit for bit, the plain matvec
// followed by the Add or the ActMul a step would otherwise launch -- per
// token, so a token reading another's residual or gate fails. Live rows lead a
// scratch of 8, the outputs are poisoned with NaN and carry a guard word.
func TestGroupEpiloguesAreTheSeparateLaunches(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		for _, c := range []struct {
			q       kernels.Quant
			rows, k int
		}{{kernels.Q4_K, 2048, 2048}, {kernels.Q6_K, 512, 2048}, {kernels.Q4_K, 1024, 4096}} {
			for _, sp := range []struct {
				split int
				grp   bool
			}{{1, false}, {4, true}, {16, true}} {
				for _, ntok := range []int{2, 3, 4} {
					for _, epi := range []string{"res", "silu", "gelu"} {
						name := fmt.Sprintf("%s/%v-%dx%d/s%d/t%d/%s", d.API(), c.q, c.rows, c.k, sp.split, ntok, epi)
						groupEpiCase(t, d, name, c.q, c.rows, c.k, ntok, sp.split, sp.grp, epi)
					}
				}
			}
		}
	}
}

func groupEpiCase(t *testing.T, d backend.Device, name string, q kernels.Quant, rows, k, ntok, split int, grp bool, epi string) {
	t.Helper()
	const actRows = 8
	rng := rand.New(rand.NewSource(int64(len(name) + rows + ntok*7)))
	g := newGPU(t, d)
	defer g.free()
	x := make([]float32, actRows*k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	av, as, asum, err := kernels.PackActivations(x)
	if err != nil {
		t.Fatal(err)
	}
	bA, bAX := g.up(u32bytes(av)), g.up(f32bytes(append(append([]float32{}, as...), asum...)))
	qs, dw, scw, err := kernels.PackWeights(q, rawWeights(q, rows, k, rng), rows, k)
	if err != nil {
		t.Fatal(err)
	}
	bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw))
	n := rows * ntok
	const guard = 0x5EED5EED
	poisoned := func() backend.Buf {
		p := make([]float32, n+1)
		for i := range p {
			p[i] = float32(math.NaN())
		}
		b := f32bytes(p)
		binary.LittleEndian.PutUint32(b[n*4:], guard)
		return g.up(b)
	}
	// The second operand, one vector per token, distinct per token.
	side := make([]float32, n)
	for i := range side {
		side[i] = float32(rng.NormFloat64())
	}
	bSide := g.up(f32bytes(side))
	base := mainkernels.MatVecShape{T: q, K: k, Rows: rows, NTok: ntok, Tok: ntok, ActRows: actRows,
		Split: split, GroupSplit: grp}
	fused := base
	act := kernels.ActSiLU
	if epi == "res" {
		fused.Bias, fused.BiasRows = true, true
	} else {
		if epi == "gelu" {
			act = kernels.ActGELU
		}
		fused.Gate, fused.GateAct = true, act
	}
	width := 128
	if grp {
		width = mainkernels.GroupSplitWidth(split)
	}
	run := func(s mainkernels.MatVecShape, bufs ...backend.Buf) {
		t.Helper()
		kk, err := mainkernels.MatVec(s)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		kern, err := d.Compile(kk)
		if err != nil {
			t.Fatalf("%s: compile: %v", name, err)
		}
		defer kern.Close()
		if err := kern.Launch((rows*split+width-1)/width, width, bufs...); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
	plain, want, got := poisoned(), poisoned(), poisoned()
	run(base, bQS, bD, bSC, bA, bAX, plain)
	run(fused, bQS, bD, bSC, bA, bAX, got, bSide)
	// The separate launch, as the tier issues it: Add(resid, out) or
	// ActMul(gate, up).
	elem := func() backend.Kernel {
		if epi == "res" {
			kk, err := mainkernels.Add(n)
			if err != nil {
				t.Fatal(err)
			}
			c, err := d.Compile(kk)
			if err != nil {
				t.Fatal(err)
			}
			return c
		}
		kk, err := mainkernels.ActMul(n, act)
		if err != nil {
			t.Fatal(err)
		}
		c, err := d.Compile(kk)
		if err != nil {
			t.Fatal(err)
		}
		return c
	}()
	defer elem.Close()
	if err := elem.Launch((n+127)/128, 128, bSide, plain, want); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	wb, gb := make([]byte, (n+1)*4), make([]byte, (n+1)*4)
	if want.Read(wb) != nil || got.Read(gb) != nil {
		t.Fatalf("%s: read back", name)
	}
	bad := 0
	for j := 0; j <= n; j++ {
		a, b := binary.LittleEndian.Uint32(wb[4*j:]), binary.LittleEndian.Uint32(gb[4*j:])
		if j < n && math.IsNaN(float64(math.Float32frombits(a))) {
			t.Fatalf("%s: word %d of the separate launches is NaN -- a degenerate oracle", name, j)
		}
		if a != b {
			if bad++; bad <= 2 {
				t.Errorf("%s: word %d = %#x, want %#x", name, j, b, a)
			}
		}
	}
	if bad > 0 {
		t.Fatalf("%s: %d of %d words differ", name, bad, n+1)
	}
}
