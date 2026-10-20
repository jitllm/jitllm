package paritytest

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestGemm16IsMains holds the composer's binary16 m16n8 GEMM over a q8_0
// weight to main's GemmVolta at its m16n8 forms: the blockings the tier
// takes, a k split, two trips, the k8 instruction and the one-buffer stage.
func TestGemm16IsMains(t *testing.T) {
	type gc struct {
		rows, k, ntok, mt, nt, wm, wn, kb, split int
		k8, single                               bool
	}
	cases := []gc{
		{rows: 128, k: 64, ntok: 64, mt: 4, nt: 4, wm: 2, wn: 2, kb: 1, split: 2},
		{rows: 192, k: 64, ntok: 64, mt: 3, nt: 4, wm: 2, wn: 2, kb: 1, split: 2},
		{rows: 32, k: 64, ntok: 64, mt: 1, nt: 2, wm: 2, wn: 2, kb: 1, split: 2},
		{rows: 64, k: 128, ntok: 64, mt: 2, nt: 4, wm: 2, wn: 2, kb: 1, split: 4},
		{rows: 64, k: 96, ntok: 64, mt: 2, nt: 4, wm: 2, wn: 2, kb: 1, split: 1},
		{rows: 128, k: 256, ntok: 64, mt: 4, nt: 4, wm: 2, wn: 2, kb: 2, split: 1},
		{rows: 64, k: 64, ntok: 64, mt: 2, nt: 4, wm: 2, wn: 2, kb: 1, split: 1, k8: true},
		{rows: 128, k: 256, ntok: 64, mt: 4, nt: 4, wm: 2, wn: 2, kb: 4, split: 1, single: true},
	}
	for _, c := range cases {
		name := fmt.Sprintf("%dx%d_t%d_m%dn%dw%dx%dk%d_s%d_k8%v_1b%v", c.rows, c.k, c.ntok, c.mt, c.nt, c.wm, c.wn, c.kb, c.split, c.k8, c.single)
		t.Run(name, func(t *testing.T) {
			f16k := 16
			if c.k8 {
				f16k = 8
			}
			mk, err := mainkernels.GemmVolta(mainkernels.MatVecShape{T: mainkernels.Q8_0, K: c.k, Rows: c.rows, NTok: c.ntok, Split: c.split},
				mainkernels.VoltaTile{MT: c.mt, NT: c.nt, WM: c.wm, WN: c.wn, KB: c.kb, F16K: f16k, Single: c.single})
			if err != nil {
				t.Fatal(err)
			}
			s := compose.Gemm16{T: kernels.Q8_0, Rows: c.rows, K: c.k, NTok: c.ntok, MT: c.mt, NT: c.nt, WM: c.wm, WN: c.wn, KB: c.kb,
				K8: c.k8, Single: c.single, Split: c.split}
			g := &compose.Row{Width: s.Threads(), Rows: s.Groups()}
			qs, d, sc, b, out := g.Buf(compose.BufU32), g.Buf(compose.BufU32), g.Buf(compose.BufU32), g.Buf(compose.BufU32), g.Buf(compose.BufOut)
			if err := g.Gemm16(s, qs, d, sc, b, out); err != nil {
				t.Fatal(err)
			}
			same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{Target: sm80}))
		})
	}
}

// TestActF16TGatherIsMains holds the gathered operand to main's.
func TestActF16TGatherIsMains(t *testing.T) {
	for _, c := range []struct {
		q             kernels.Quant
		mq            mainkernels.Quant
		ntok, k, nsrc int
	}{{kernels.Q8_0, mainkernels.Q8_0, 64, 64, 20}, {kernels.Q4_K, mainkernels.Q4_K, 64, 512, 33}, {kernels.MXFP4, mainkernels.MXFP4, 32, 128, 7}} {
		name := fmt.Sprintf("%v_%dx%d_%d", c.q, c.ntok, c.k, c.nsrc)
		t.Run(name, func(t *testing.T) {
			mk, err := mainkernels.ActF16TGather(c.mq, c.ntok, c.k, c.nsrc)
			if err != nil {
				t.Fatal(err)
			}
			g := &compose.Row{Width: c.ntok * c.k / c.q.Info().Sub, Rows: 1}
			x, perm, out := g.Buf(compose.BufRow), g.Buf(compose.BufU32), g.Buf(compose.BufOutU32)
			if err := g.ActF16TGather(c.q, c.k, x, perm, c.nsrc, out); err != nil {
				t.Fatal(err)
			}
			same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{Target: sm80}))
		})
	}
}

// TestActF16TIsMains holds the composer's token-major binary16 operand to
// main's ActF16T over byte-ordered and interleaved formats.
func TestActF16TIsMains(t *testing.T) {
	for _, c := range []struct {
		q       kernels.Quant
		mq      mainkernels.Quant
		ntok, k int
	}{{kernels.Q8_0, mainkernels.Q8_0, 64, 64}, {kernels.Q8_0, mainkernels.Q8_0, 32, 256}, {kernels.Q4_0, mainkernels.Q4_0, 64, 128}} {
		name := fmt.Sprintf("%v_%dx%d", c.q, c.ntok, c.k)
		t.Run(name, func(t *testing.T) {
			mk, err := mainkernels.ActF16T(c.mq, c.ntok, c.k)
			if err != nil {
				t.Fatal(err)
			}
			g := &compose.Row{Width: c.ntok * c.k / c.q.Info().Sub, Rows: 1}
			x, out := g.Buf(compose.BufRow), g.Buf(compose.BufOutU32)
			if err := g.ActF16T(c.q, c.k, x, out); err != nil {
				t.Fatal(err)
			}
			same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{Target: sm80}))
		})
	}
}
