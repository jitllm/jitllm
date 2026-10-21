//go:build linux && (amd64 || arm64)

package nn

import (
	"math/rand"
	"syscall"
	"testing"
	"unsafe"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit_old/cpu"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// onNode copies b into fresh anonymous memory bound to node, so the page the
// kernels read is where a test says it is.
func onNode(t *testing.T, b []byte, node int) []byte {
	t.Helper()
	m, mapped, err := kernels.MapOffHeap(len(b))
	if err != nil || !mapped {
		t.Skipf("no anonymous mapping here: %v", err)
	}
	mask := make([]uint64, 1+node/64)
	mask[node/64] = 1 << uint(node%64)
	const mpolBind, mpolMFMove = 2, 1 << 1
	if _, _, e := syscall.Syscall6(syscall.SYS_MBIND, uintptr(unsafe.Pointer(&m[0])), uintptr(len(m)),
		mpolBind, uintptr(unsafe.Pointer(&mask[0])), uintptr(len(mask)*64+1), mpolMFMove); e != 0 {
		t.Skipf("mbind refused: %v", e)
	}
	copy(m, b)
	return m[:len(b):len(b)]
}

// TestMixtureRegionsRunWhereTheExpertsLive: on a host of two or more NUMA
// nodes with a node-affine pool, the mixture's gate/up and down regions over
// experts whose pages alternate the nodes run node-affine (MixtureAffine
// counts them) and give the plain regions' bits exactly, gather with the
// gated activation included. One node is a visible skip.
func TestMixtureRegionsRunWhereTheExpertsLive(t *testing.T) {
	nodes := sched.NUMANodes()
	if len(nodes) < 2 {
		t.Skip("one NUMA node: nothing to place -- this gate proved nothing here")
	}
	const m, rows, k = 5, 256, 512
	gt := quant.Q4_K
	if !cpu.PackedFusedSupported(gt) {
		t.Skip("no fused Q4_K kernel")
	}
	ps := make([]*Packed, m)
	for i := range ps {
		p := packSheet(t, gt, rows, k, i+1)
		node := nodes[(i*3)%len(nodes)] // not in index order: the order is rebuilt
		p.QS, p.D, p.SC = onNode(t, p.QS, node), onNode(t, p.D, node), onNode(t, p.SC, node)
		ps[i] = p
	}
	stride := k + ElemLanes
	rng := rand.New(rand.NewSource(3))
	x := make([]float32, (m-1)*stride+k)
	up := make([]float32, len(x))
	for i := range x {
		x[i], up[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
	}
	run := func(numa bool) ([][]float32, [][]float32, int64) {
		j := NewJIT(k, rows, []quant.Type{gt}, WithTune(TuneOff), WithMixtureBatch(1), WithGEMMExact(true),
			WithSched(sched.WithNUMA(numa)))
		if j == nil {
			t.Fatal("no JIT")
		}
		defer j.Close()
		multi, gather := make([][]float32, m), make([][]float32, m)
		for i := range multi {
			multi[i], gather[i] = poison(rows), poison(rows)
		}
		if !j.MatVecPackedMulti(multi, gt, ps, x[:k], rows, k) {
			t.Fatal("MatVecPackedMulti declined")
		}
		xa := append([]float32(nil), x...)
		if !j.MatVecPackedGatherAct(gather, gt, ps, xa, up, ActSiLU, stride, rows, k) {
			t.Fatal("MatVecPackedGatherAct declined")
		}
		return multi, gather, j.MixtureAffine()
	}
	wm, wg, a0 := run(false)
	gm, gg, a1 := run(true)
	if a0 != 0 || a1 != 2 {
		t.Fatalf("node-affine mixture regions: %d without NUMA, %d with (want 0 and 2)", a0, a1)
	}
	for i := 0; i < m; i++ {
		for r := 0; r < rows; r++ {
			if gm[i][r] != wm[i][r] || gg[i][r] != wg[i][r] {
				t.Fatalf("matrix %d row %d: node-affine multi %v gather %v, plain %v %v",
					i, r, gm[i][r], gg[i][r], wm[i][r], wg[i][r])
			}
		}
	}
}
