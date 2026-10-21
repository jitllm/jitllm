//go:build jitllmbench

package backend_test

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/dev/bench"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestPagedStagedAB is the staged decode's bar: the tier's contiguous staged
// sequence (scores, the softmax over a context-wide plane, the split
// accumulate and its reduce) against the paged, bounded one (scores, the
// per-split softmax, the accumulate, the merge), same pass, A/A first.
// JITLLM_PAGED_API, _DEV and _DEPTH select as for TestPagedAttentionAB.
func TestPagedStagedAB(t *testing.T) {
	gpuLock(t)
	devs := pagedDevices(t)
	if len(devs) == 0 {
		t.Fatal("no GPU")
	}
	depths := []int{512, 4096, 16384}
	if dd := os.Getenv("JITLLM_PAGED_DEPTH"); dd != "" {
		var x int
		fmt.Sscanf(dd, "%d", &x)
		depths = []int{x}
	}
	for _, d := range devs {
		defer d.Close()
		if api := os.Getenv("JITLLM_PAGED_API"); api != "" && api != d.API() {
			continue
		}
		if dn := os.Getenv("JITLLM_PAGED_DEV"); dn != "" && !strings.Contains(d.Name(), dn) {
			continue
		}
		wall := deviceWall(t, d)
		t.Logf("%s/%s: read wall %.1f GB/s", d.API(), d.Name(), wall/1e9)
		for _, m := range []struct {
			name                     string
			heads, kv, dim, attnLays int
		}{{"Llama-3.2-1B", 32, 8, 64, 16}, {"Qwen3.5-0.8B", 8, 2, 256, 6}} {
			if mn := os.Getenv("JITLLM_PAGED_MODEL"); mn != "" && !strings.Contains(m.name, mn) {
				continue
			}
			for _, n := range depths {
				t.Run(fmt.Sprintf("%s/%s/%s/n%d", d.API(), d.Name(), m.name, n), func(t *testing.T) {
					page := 256
					fmt.Sscanf(os.Getenv("JITLLM_PAGED_PAGE"), "%d", &page)
					stagedAB(t, d, m.heads, m.kv, m.dim, m.attnLays, n, page, wall)
				})
			}
		}
	}
}

func stagedAB(t *testing.T, d backend.Device, heads, nkv, dim, modelLayers, n, page int, wall float64) {
	kvRow, gqa := nkv*dim, heads/nkv
	perLayer := n * kvRow * 8
	layers := max(1, min(modelLayers, (256<<20)/perLayer))
	lanes := 1
	if ok, _ := backend.GuaranteedLanes(d, 32); ok {
		lanes = 32
	}
	compile := func(k *ir.Kernel, e error) backend.Kernel {
		if e != nil {
			t.Fatal(e)
		}
		c, e := d.Compile(k)
		if e != nil {
			t.Fatal(e)
		}
		t.Cleanup(c.Close)
		return c
	}
	scale := float32(1 / math.Sqrt(float64(dim)))
	// Contiguous: the tier's decode schedule.
	slots := d.Slots()
	if slots <= 0 {
		slots = 16384
	}
	accSplit := 1
	for accSplit < 32 && heads*dim*accSplit*2 <= slots {
		accSplit *= 2
	}
	cScore := compile(mainkernels.AttnScoresTiled(heads, dim, kvRow, gqa, n, scale, 1, 1, 1, n+1))
	cSoft := compile(mainkernels.SoftmaxRows(heads, n, lanes, 1, 1))
	cAcc := compile(mainkernels.AttnAccSplit(heads, dim, kvRow, gqa, n, accSplit))
	cRed := compile(mainkernels.Reduce(heads*dim, accSplit))
	// Paged: splits near sqrt(n/8) (JITLLM_PAGED_SPLITS pins), chunks covering n.
	S := 1 << int(math.Log2(math.Sqrt(max(1, float64(n)/8))))
	fmt.Sscanf(os.Getenv("JITLLM_PAGED_SPLITS"), "%d", &S)
	C := ((n+S-1)/S + 31) / 32 * 32
	s := mainkernels.FlashShape{Heads: heads, KVHeads: nkv, Dim: dim, Rows: 1, Scale: scale, Page: page, Splits: S, Chunk: C}
	fmt.Sscanf(os.Getenv("JITLLM_PAGED_GROUP"), "%d", &s.Group)
	ms := s
	ms.Page, ms.Chunk = 0, 0
	pScore := compile(mainkernels.PagedStagedScores(s, 1))
	pSoft := compile(mainkernels.PagedStagedSoftmax(s, lanes))
	pAcc := compile(mainkernels.PagedStagedAcc(s))
	pMerge := compile(mainkernels.FlashAttentionMergeWide(ms))
	g := newGPU(t, d)
	defer g.free()
	rng := rand.New(rand.NewSource(int64(n)))
	rnd := func(m int) []float32 {
		a := make([]float32, m)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		return a
	}
	q := g.up(f32bytes(rnd(heads * dim)))
	nb := g.up(u32bytes([]uint32{uint32(n)}))
	out := g.up(make([]byte, heads*dim*4))
	scores, probs := g.up(make([]byte, heads*n*4)), g.up(make([]byte, heads*n*4))
	accPart := g.up(make([]byte, heads*dim*accSplit*4))
	plane := mainkernels.PagedStagedPlane(s)
	pscores, pprobs := g.up(make([]byte, plane*4)), g.up(make([]byte, plane*4))
	part := g.up(make([]byte, mainkernels.FlashPartialFloats(ms)*4))
	var ck, cv, pk, pv, tabs []backend.Buf
	npages := (n + page - 1) / page
	for l := 0; l < layers; l++ {
		ck = append(ck, g.up(f32bytes(rnd(kvRow*(n+1)))))
		cv = append(cv, g.up(f32bytes(rnd(n*kvRow))))
		pk = append(pk, g.up(f32bytes(rnd(npages*page*kvRow))))
		pv = append(pv, g.up(f32bytes(rnd(npages*page*kvRow))))
		perm := rng.Perm(npages)
		tab := make([]uint32, npages)
		for i := range tab {
			tab[i] = uint32(perm[i])
		}
		tabs = append(tabs, g.up(u32bytes(tab)))
	}
	desc := g.up(u32bytes([]uint32{0, 0, uint32(n), uint32(n - 1)}))
	smG, smW := heads, 32
	if lanes == 1 {
		smG, smW = (heads+63)/64, 64
	}
	pSmG := (heads*S + 1) / 2
	if lanes == 1 {
		pSmG = (heads*S + 63) / 64
	}
	arm := func(isPaged bool) func(int) {
		return func(iters int) {
			d.Session(func(se backend.Session) {
				var e error
				launch := func(c backend.Kernel, groups, width int, args ...backend.Buf) {
					if e == nil {
						e = se.Launch(c, groups, width, args...)
					}
				}
				for i := 0; i < iters; i++ {
					l := i % layers
					if isPaged {
						skip := os.Getenv("XSKIP")
						if !strings.Contains(skip, "S") {
							launch(pScore, (mainkernels.PagedStagedScoreItems(s)+127)/128, 128, q, pk[l], nb, pscores, tabs[l], desc)
						}
						if !strings.Contains(skip, "M") {
							launch(pSoft, pSmG, 64, pscores, nb, pprobs, part, tabs[l], desc)
						}
						if !strings.Contains(skip, "A") {
							launch(pAcc, (mainkernels.PagedStagedAccThreads(s)+127)/128, 128, pprobs, pv[l], nb, part, tabs[l], desc)
						}
						if !strings.Contains(skip, "G") {
							launch(pMerge, (heads*dim+127)/128, 128, part, out)
						}
					} else {
						launch(cScore, (heads*n+127)/128, 128, q, ck[l], nb, scores)
						launch(cSoft, smG, smW, scores, nb, probs)
						launch(cAcc, (heads*dim*accSplit+127)/128, 128, probs, cv[l], nb, accPart)
						launch(cRed, (heads*dim+127)/128, 128, accPart, out)
					}
				}
				if e == nil {
					e = se.Sync()
				}
				if e != nil {
					panic(e)
				}
			})
		}
	}
	a := bench.Case{Name: "contiguous", Fn: arm(false)}
	b := bench.Case{Name: "paged", Fn: arm(true)}
	iters := max(20, min(400, (400<<20)/perLayer))
	for until := time.Now().Add(2 * time.Second); time.Now().Before(until); {
		a.Fn(iters)
		b.Fn(iters)
	}
	bytes := float64(n*kvRow*8) * float64(iters)
	gbs := func(dt time.Duration) string {
		r := bytes / dt.Seconds()
		return fmt.Sprintf("%.1f GB/s = %.0f%% of wall", r/1e9, 100*r/wall)
	}
	aa := bench.AB(a, a, 20, iters)
	t.Logf("paged S=%d C=%d, contiguous acc split %d, softmax lanes %d, %d layers; A/A control %s", S, C, accSplit, lanes, layers, aa)
	for pass := 0; pass < 2; pass++ {
		r := bench.AB(a, b, 20, iters)
		t.Logf("pass%d %s (>1: paged faster); contiguous %s, paged %s", pass, r, gbs(r.AMedian), gbs(r.BMedian))
	}
}
