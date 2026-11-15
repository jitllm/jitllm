package oracle

import "math"

// Dry is llama.cpp's DRY sampler over v (llama_sampler_dry_apply), in
// place. hist is the history, of which the last lastN tokens are read (ctx
// where lastN is -1); breakers maps a restart sequence's head to its tails.
// The nearest restart sequence limits how far back a repeat reaches; each
// earlier place the history's tail repeats is found by matching back from
// it token by token -- what llama.cpp's Z-algorithm computes; the token
// after a repeat at least allowed long would extend it, and falls by mult
// times base to the longest such repeat's length past allowed, as llama.cpp
// computes it (the power in double, the product rounded to binary32), the
// length clamped where base to it passes binary32's largest. A token that
// is a breaker alone is not penalized. Off at mult 0, base below 1 or lastN
// 0.
func Dry(v []float32, hist []int32, mult, base float32, allowed, lastN, ctx int, breakers map[int32][][]int32) {
	if mult == 0 || base < 1 || lastN == 0 {
		return
	}
	window := max(lastN, 0)
	if lastN == -1 {
		window = ctx
	}
	nr := min(len(hist), window, ctx)
	if nr <= allowed {
		return
	}
	last := hist[len(hist)-nr:]
	back := func(i int) int32 { return last[nr-1-i] }
	limit := nr
	for i := range nr {
		tails, ok := breakers[back(i)]
		if !ok {
			continue
		}
		longest := -1
		for _, tail := range tails {
			if l := len(tail); l > longest && l <= i {
				match := true
				for off := range l {
					if tail[off] != back(i-off-1) {
						match = false
						break
					}
				}
				if match {
					longest = l
				}
			}
		}
		if longest >= 0 {
			limit = i - longest
			break
		}
	}
	if limit < allowed {
		return
	}
	longest := map[int32]int{}
	for j := range nr - 1 {
		n := 0
		for n <= j && last[j-n] == last[nr-1-n] {
			n++
		}
		if n = min(n, limit); n >= allowed {
			longest[last[j+1]] = max(longest[last[j+1]], n)
		}
	}
	maxExp := 0
	if base > 1.000001 {
		maxExp = int(float32(88.7228391) / float32(math.Log(float64(base))))
	}
	for t, n := range longest {
		alone := false
		for _, tail := range breakers[t] {
			alone = alone || len(tail) == 0
		}
		if alone || int(t) >= len(v) {
			continue
		}
		e := n - allowed
		if maxExp > 0 && e > maxExp {
			e = maxExp
		}
		v[t] -= float32(float64(mult) * math.Pow(float64(base), float64(e)))
	}
}
