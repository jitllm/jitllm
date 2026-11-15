package engine

import (
	"math"
	"slices"
	"strings"

	"github.com/jitllm/jitllm/tok"
)

// dryState is DRY's working rows on an instance: row, each token's exponent
// plus one where the next draw penalizes it (0 elsewhere), the tokens row
// marks, the repeat counts the scan writes, and the sequence breakers read
// off the vocabulary for the strings key names.
type dryState struct {
	row     []float32
	marked  []int32
	counts  []int
	key     string
	breaker map[int32][][]int32
}

// dryBreakers are llama.cpp's default sequence breakers (common_params_sampling).
func dryBreakers() []string { return []string{"\n", ":", "\"", "*"} }

// breakersOf is strs read off v as llama.cpp's DRY sampler reads its
// breakers at init (llama_sampler_init_dry, get_overlapping_token_sequences):
// each cut to 40 bytes, an empty one skipped; every token whose text holds
// one heads it alone (no tail), and every token whose text ends in one's
// beginning heads the rest of it, tokenized without specials and cut to 20
// tokens, once each.
func breakersOf(v *tok.Vocab, strs []string) map[int32][][]int32 {
	out := map[int32][][]int32{}
	if v == nil {
		return out
	}
	for _, s := range strs {
		if s == "" {
			continue
		}
		if len(s) > 40 {
			s = s[:40]
		}
		for id := range int32(v.Size()) {
			w := v.Literal(id)
			if strings.Contains(w, s) {
				out[id] = append(out[id], nil)
				continue
			}
			for pos := strings.IndexByte(w, s[0]); pos >= 0; {
				i, match := 1, true
				for ; i < len(s) && pos+i < len(w); i++ {
					if w[pos+i] != s[i] {
						match = false
						break
					}
				}
				if match {
					tail := v.Encode(s[i:], false)
					if len(tail) > 20 {
						tail = tail[:20]
					}
					if !slices.ContainsFunc(out[id], func(t []int32) bool { return slices.Equal(t, tail) }) {
						out[id] = append(out[id], tail)
					}
				}
				next := strings.IndexByte(w[pos+1:], s[0])
				if next < 0 {
					break
				}
				pos += 1 + next
			}
		}
	}
	return out
}

// dryObserve records t in DRY's history, which holds its window's last
// tokens (llama.cpp's ring buffer of dry_penalty_last_n; -1, the context,
// holds them all).
func (s *Sampler) dryObserve(t int32) {
	if n := s.DryPenaltyLastN; n > 0 && len(s.dhist) >= n {
		copy(s.dhist, s.dhist[len(s.dhist)-n+1:])
		s.dhist = s.dhist[:n-1]
	}
	s.dhist = append(s.dhist, t)
}

// dryRow writes s.dry.row for smp's next draw as llama.cpp's DRY sampler
// finds what it penalizes (llama_sampler_dry_apply, steps 1 to 3, and step
// 4's exponent): over the last tokens of its history, the nearest restart
// sequence limits how far back a repeat reaches; the Z-algorithm, reversed,
// finds each earlier place the history's tail repeats and how long; the
// token after each repeat at least DryAllowedLength long would extend it,
// and is marked with the longest, less the allowed length, clamped where
// the base to it would overflow binary32 -- a token that is a breaker alone
// excepted. It reports whether it marked any.
func (s *Instance) dryRow(smp *Sampler) bool {
	d, n := &s.dry, s.c.NVocab
	if len(d.row) != n {
		d.row = make([]float32, n)
	}
	for _, t := range d.marked {
		d.row[t] = 0
	}
	d.marked = d.marked[:0]
	strs := smp.DryBreakers
	if strs == nil {
		strs = dryBreakers()
	}
	if key := strings.Join(strs, "\x00"); d.breaker == nil || key != d.key {
		d.key, d.breaker = key, breakersOf(s.m.Vocab, strs)
	}
	last := smp.dhist
	window := smp.DryPenaltyLastN
	if window < 0 {
		window = s.maxSeq
	}
	nr := min(len(last), window, s.maxSeq)
	allowed := smp.DryAllowedLength
	if nr <= allowed {
		return false
	}
	rat := func(i int) int32 { return last[len(last)-1-i] }
	// Step 1: the nearest restart sequence, its head i tokens from the end.
	limit := nr
	for i := range nr {
		tails, ok := d.breaker[rat(i)]
		if !ok {
			continue
		}
		longest := -1
		for _, tail := range tails {
			if l := len(tail); l > longest && l <= i {
				match := true
				for off := range l {
					if tail[off] != rat(i-off-1) {
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
		return false
	}
	// Step 2: the repeat counts, by the Z-algorithm over the reversed tokens.
	if cap(d.counts) < nr {
		d.counts = make([]int, nr)
	}
	cnt := d.counts[:nr]
	clear(cnt)
	end := nr - 1
	rt, lt := 0, 0
	for k := 1; k < nr; k++ {
		if k > rt {
			m := 0
			for m+k < nr && rat(m) == rat(m+k) {
				m++
			}
			cnt[end-k] = min(m, limit)
			if m > 0 {
				lt, rt = k, k+m-1
			}
			continue
		}
		p, right := k-lt, rt-k+1
		if cnt[end-p] < right {
			cnt[end-k] = min(cnt[end-p], limit)
			continue
		}
		i := rt + 1
		for i < nr && rat(i) == rat(i-k) {
			i++
		}
		cnt[end-k] = min(i-k, limit)
		lt, rt = k, i-1
	}
	// Step 3: the token after each repeat, the longest repeat it would extend.
	for i := range nr - 1 {
		if l := cnt[i]; l >= allowed {
			t := rat(nr - 2 - i)
			if d.row[t] == 0 {
				d.marked = append(d.marked, t)
			}
			d.row[t] = max(d.row[t], float32(l+1))
		}
	}
	// Step 4's exponent: the length past the allowed one, clamped where
	// the base to it would pass binary32's largest.
	maxExp := 0
	if smp.DryBase > 1.000001 {
		maxExp = int(float32(88.7228391) / float32(math.Log(float64(float32(smp.DryBase)))))
	}
	marked := false
	for _, t := range d.marked {
		if slices.ContainsFunc(d.breaker[t], func(tail []int32) bool { return len(tail) == 0 }) {
			d.row[t] = 0
			continue
		}
		e := int(d.row[t]) - 1 - allowed
		if maxExp > 0 && e > maxExp {
			e = maxExp
		}
		d.row[t], marked = float32(e+1), true
	}
	return marked
}
