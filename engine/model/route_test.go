//go:build amd64 || arm64

package model

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestRoutingLocality prices speculative expert prefetch from a routing trace:
// the previous-token and last-W-tokens predictors, a static hot set, coverage
// by resident-set size, and an LRU/Belady replay of per-expert pages. See
// docs/engineering-history/placement.md for the conclusions.
//
//	JITLLM_ROUTE_MODEL=models/Qwen3-MOE-4x0.6B-Q4_K_M.gguf \
//	  ./scripts/cap 8G -- go test ./engine/model -run TestRoutingLocality -v
func TestRoutingLocality(t *testing.T) {
	path := testmodels.Resolve(os.Getenv("JITLLM_ROUTE_MODEL"))
	if path == "" {
		t.Skip("set JITLLM_ROUTE_MODEL")
	}
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()
	if !m.Cfg.MoE() {
		t.Fatalf("%s is not a mixture of experts", path)
	}
	nTok := envInt("JITLLM_ROUTE_N", 200)

	// prev[layer] is the previous token's selection; hit/tot count how much of
	// it the next token reused.
	prev := make([][]int, m.Cfg.NLayer)
	hit := make([]int, m.Cfg.NLayer)
	tot := make([]int, m.Cfg.NLayer)
	exact := make([]int, m.Cfg.NLayer)
	steps := make([]int, m.Cfg.NLayer)
	freq := make([]map[int]int, m.Cfg.NLayer)
	for i := range freq {
		freq[i] = map[int]int{}
	}
	// The window predictor fetches the union of the last W selections, which
	// raises both the hit rate and the bytes fetched; both are reported.
	//
	// hist[layer] is the last W selections, newest last.
	windows := []int{1, 2, 4, 8, 16}
	maxW := windows[len(windows)-1]
	hist := make([][][]int, m.Cfg.NLayer)
	wHit := make([]int, len(windows))
	wTot := make([]int, len(windows))
	wSize := make([]int, len(windows)) // experts the predictor would fetch
	wSteps := make([]int, len(windows))
	// seq is every selection in order, for the cache replay at the end; decode
	// marks where the prompt stops, so the replay warms on the prompt and is
	// charged for the generated tokens only.
	type access struct{ layer, expert int }
	var seq []access
	decode := -1
	m.Trace(func(layer int, name string, v []float32) {
		if name != "moe_topk" {
			return
		}
		for _, x := range v {
			seq = append(seq, access{layer, int(x)})
		}
		sel := make([]int, len(v))
		for i, x := range v {
			sel[i] = int(x)
			freq[layer][sel[i]]++
		}
		h := hist[layer]
		for wi, w := range windows {
			if len(h) < w {
				continue
			}
			set := map[int]bool{}
			for _, s := range h[len(h)-w:] {
				for _, e := range s {
					set[e] = true
				}
			}
			n := 0
			for _, e := range sel {
				if set[e] {
					n++
				}
			}
			wHit[wi] += n
			wTot[wi] += len(sel)
			wSize[wi] += len(set)
			wSteps[wi]++
		}
		h = append(h, sel)
		if len(h) > maxW {
			h = h[len(h)-maxW:]
		}
		hist[layer] = h
		if p := prev[layer]; p != nil {
			n := 0
			for _, e := range sel {
				for _, q := range p {
					if e == q {
						n++
						break
					}
				}
			}
			hit[layer] += n
			tot[layer] += len(sel)
			if n == len(sel) {
				exact[layer]++
			}
			steps[layer]++
		}
		prev[layer] = sel
	})
	defer m.Trace(nil)

	prompt := os.Getenv("JITLLM_ROUTE_PROMPT")
	if prompt == "" {
		prompt = "The capital of France is Paris and the history of Europe is long. " +
			"In 1789 the revolution began. Mathematics describes motion through differential " +
			"equations, while proteins fold into shapes that determine what they do."
	}
	ids := m.Vocab.Encode(prompt, true)
	s := m.NewInstance(len(ids) + nTok + 1)
	defer s.Close()
	var lg []float32
	for _, id := range ids {
		if lg, err = s.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	decode = len(seq)
	for i := 0; i < nTok; i++ {
		if lg, err = s.Forward(Greedy(lg)); err != nil {
			t.Fatal(err)
		}
	}

	var sumHit, sumTot, sumExact, sumSteps int
	for li := range hit {
		sumHit += hit[li]
		sumTot += tot[li]
		sumExact += exact[li]
		sumSteps += steps[li]
	}
	if sumTot == 0 {
		t.Fatal("no routing was traced; the moe_topk trace is not firing")
	}
	// The baseline a predictor must beat: picking NExpertUsed of NExpert at
	// random reuses NExpertUsed/NExpert of them by chance.
	chance := float64(m.Cfg.NExpertUsed) / float64(m.Cfg.NExpert)
	fmt.Printf("\n  %s\n  %d layers x %d experts, %d used, %d tokens\n",
		path, m.Cfg.NLayer, m.Cfg.NExpert, m.Cfg.NExpertUsed, nTok)
	fmt.Printf("  PREVIOUS-TOKEN PREDICTOR: %.1f%% of experts reused  (chance %.1f%%)\n",
		100*float64(sumHit)/float64(sumTot), 100*chance)
	fmt.Printf("  whole set unchanged:      %.1f%% of layer-steps\n",
		100*float64(sumExact)/float64(sumSteps))

	// And what a static hot-set predictor would get: the NExpertUsed globally
	// hottest experts of each layer, scored against every step of that layer.
	var hotHit, hotTot int
	for li := range freq {
		type ec struct{ e, n int }
		cs := make([]ec, 0, len(freq[li]))
		for e, n := range freq[li] {
			cs = append(cs, ec{e, n})
		}
		sort.Slice(cs, func(i, j int) bool { return cs[i].n > cs[j].n })
		top := map[int]bool{}
		for i := 0; i < m.Cfg.NExpertUsed && i < len(cs); i++ {
			top[cs[i].e] = true
		}
		for e, n := range freq[li] {
			hotTot += n
			if top[e] {
				hotHit += n
			}
		}
	}
	fmt.Printf("  STATIC HOT-SET PREDICTOR: %.1f%% of selections covered\n",
		100*float64(hotHit)/float64(hotTot))

	// Hit rate and experts fetched per expert used must be read together.
	fmt.Printf("\n  WINDOW PREDICTOR (union of the last W tokens, same layer)\n")
	fmt.Printf("  %-4s %-10s %-12s %s\n", "W", "hit", "fetched", "fetched/used")
	for wi, w := range windows {
		if wSteps[wi] == 0 {
			continue
		}
		size := float64(wSize[wi]) / float64(wSteps[wi])
		fmt.Printf("  %-4d %-10.1f %-12.2f %.2fx\n", w,
			100*float64(wHit[wi])/float64(wTot[wi]), size,
			size/float64(m.Cfg.NExpertUsed))
	}

	// Coverage by resident-set size: for the r hottest experts of each layer,
	// the share of all selections already resident. A placement policy needs
	// what every budget buys, not one point.
	fmt.Printf("\n  COVERAGE vs RESIDENT SET (per layer, %d experts, %d used)\n",
		m.Cfg.NExpert, m.Cfg.NExpertUsed)
	fmt.Printf("  %-8s %-10s %-12s %s\n", "resident", "coverage", "chance", "bank share")
	sorted := make([][]int, len(freq)) // expert ids by descending use
	for li := range freq {
		type ec struct{ e, n int }
		cs := make([]ec, 0, len(freq[li]))
		for e, n := range freq[li] {
			cs = append(cs, ec{e, n})
		}
		sort.Slice(cs, func(i, j int) bool {
			if cs[i].n != cs[j].n {
				return cs[i].n > cs[j].n
			}
			return cs[i].e < cs[j].e
		})
		ids := make([]int, len(cs))
		for i, c := range cs {
			ids[i] = c.e
		}
		sorted[li] = ids
	}
	// JITLLM_ROUTE_DUMP writes the hot set so two prompts can be compared: a
	// hotness policy only works if hotness belongs to the model, not the text.
	if out := os.Getenv("JITLLM_ROUTE_DUMP"); out != "" {
		f, err := os.Create(out)
		if err != nil {
			t.Fatal(err)
		}
		for li := range sorted {
			fmt.Fprintf(f, "%d", li)
			for _, e := range sorted[li] {
				fmt.Fprintf(f, " %d:%d", e, freq[li][e])
			}
			fmt.Fprintln(f)
		}
		f.Close()
	}
	for _, r := range []int{1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128} {
		if r > m.Cfg.NExpert {
			break
		}
		var cov, tot int
		for li := range freq {
			res := map[int]bool{}
			for i := 0; i < r && i < len(sorted[li]); i++ {
				res[sorted[li][i]] = true
			}
			for e, n := range freq[li] {
				tot += n
				if res[e] {
					cov += n
				}
			}
		}
		fmt.Printf("  %-8d %-10.1f %-12.1f %.1f%%\n", r,
			100*float64(cov)/float64(tot),
			100*float64(r)/float64(m.Cfg.NExpert),
			100*float64(r)/float64(m.Cfg.NExpert))
	}

	// Replay this run's routing through a per-expert cache of each budget's
	// size, with every block's base resident: the bytes a token reads under
	// LRU, and under Belady (evict the one needed farthest ahead), the ceiling
	// no policy can beat.
	expBytes, baseBytes, err := expertAndBase(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	gen := len(seq) - decode
	if decode < 0 || gen <= 0 {
		t.Fatal("no decode routing was traced")
	}
	fmt.Printf("\n  SPLIT-PAGE REPLAY  one expert %.2f MiB, base of every block %.2f GiB resident\n",
		float64(expBytes)/(1<<20), float64(baseBytes)/(1<<30))
	fmt.Printf("  %-10s %-9s %-14s %-14s %s\n", "budget GiB", "experts", "LRU hit", "LRU GB/token", "Belady GB/token")
	budgets := []float64{8, 14.68, 20, 28}
	if b := os.Getenv("JITLLM_ROUTE_BUDGETS"); b != "" {
		budgets = budgets[:0]
		for _, f := range strings.Split(b, ",") {
			v, err := strconv.ParseFloat(f, 64)
			if err != nil {
				t.Fatal(err)
			}
			budgets = append(budgets, v)
		}
	}
	key := func(a access) int { return a.layer<<20 | a.expert }
	// next[i] is the index of the next access to seq[i]'s expert, for Belady.
	next := make([]int, len(seq))
	last := map[int]int{}
	for i := len(seq) - 1; i >= 0; i-- {
		k := key(seq[i])
		if j, ok := last[k]; ok {
			next[i] = j
		} else {
			next[i] = len(seq)
		}
		last[k] = i
	}
	for _, gib := range budgets {
		room := int64(gib*(1<<30)) - baseBytes
		capN := 0
		if room > 0 {
			capN = int(room / expBytes)
		}
		lruMiss := replayLRU(seq, decode, capN, key)
		belMiss := replayBelady(seq, decode, capN, key, next)
		perTok := func(miss int) float64 { return float64(miss) * float64(expBytes) / float64(nTok) / 1e9 }
		fmt.Printf("  %-10.2f %-9d %-14.1f %-14.3f %.3f\n", gib, capN,
			100*(1-float64(lruMiss)/float64(gen)), perTok(lruMiss), perTok(belMiss))
	}
}

// expertAndBase sizes one expert (its sheet of every bank in a block) and the
// base weights of every block -- a page minus its banks -- off the container.
func expertAndBase(path string) (expert, base int64, err error) {
	f, err := jlm.Open(path)
	if err != nil {
		return 0, 0, err
	}
	defer f.Close()
	perBlockExp := map[int32]int64{}
	perBlockBase := map[int32]int64{}
	for i := range f.Entries() {
		e := &f.Entries()[i]
		if e.Block < 0 || e.Role.Vision() {
			continue
		}
		if r := e.Role; r == jlm.RoleExpGateBank || r == jlm.RoleExpUpBank || r == jlm.RoleExpDownBank {
			qs, d, sc, _, err := f.SheetSpan(e)
			if err != nil {
				return 0, 0, err
			}
			perBlockExp[e.Block] += int64(qs + d + sc)
			continue
		}
		qs, d, sc := f.SpanLen(e)
		perBlockBase[e.Block] += int64(qs + d + sc)
	}
	for _, n := range perBlockExp {
		expert = max(expert, n)
	}
	for _, n := range perBlockBase {
		base += n
	}
	if expert == 0 {
		return 0, 0, fmt.Errorf("%s has no expert bank", path)
	}
	return expert, base, nil
}

// replayLRU counts misses from index from on, with the cache warmed by the
// accesses before it.
func replayLRU[A any](seq []A, from, capN int, key func(A) int) (miss int) {
	if capN == 0 {
		return len(seq) - from
	}
	stamp := map[int]int{}
	for i, a := range seq {
		k := key(a)
		if _, ok := stamp[k]; !ok {
			if i >= from {
				miss++
			}
			if len(stamp) >= capN {
				oldK, oldT := 0, int(^uint(0)>>1)
				for kk, t := range stamp { // O(capacity) eviction; a replay, not a pager
					if t < oldT {
						oldK, oldT = kk, t
					}
				}
				delete(stamp, oldK)
			}
		}
		stamp[k] = i
	}
	return miss
}

// replayBelady is replayLRU evicting the entry whose next use is farthest.
func replayBelady[A any](seq []A, from, capN int, key func(A) int, next []int) (miss int) {
	if capN == 0 {
		return len(seq) - from
	}
	nextUse := map[int]int{}
	for i, a := range seq {
		k := key(a)
		if _, ok := nextUse[k]; !ok {
			if i >= from {
				miss++
			}
			if len(nextUse) >= capN {
				farK, farT := 0, -1
				for kk, t := range nextUse {
					if t > farT {
						farK, farT = kk, t
					}
				}
				delete(nextUse, farK)
			}
		}
		nextUse[k] = next[i]
	}
	return miss
}
