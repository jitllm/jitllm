package grammar

import (
	"fmt"
	"slices"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"
)

// Vocabulary is what the mask reads of a tokenizer: each token's bytes, and
// which tokens end a generation.
type Vocabulary interface {
	Size() int
	// Piece is the bytes the token adds to the text, false for one that adds
	// none (a control token).
	Piece(id int32) (string, bool)
	IsEOG(id int32) bool
}

// State is where a constrained output stands: the automaton's set, and a
// code point begun by an earlier token's last bytes (need more bytes to come,
// cp the bits so far).
type State struct {
	set     int32
	need, n uint8
	cp      rune
}

// Tokens is a tokenizer's pieces sorted by their bytes, built once per
// tokenizer and shared by every grammar over it: the trie the mask walks is
// ranges of this list, a node being the pieces that share a prefix. The
// caller keeps it beside the tokenizer it was built from (IndexTokens).
type Tokens struct {
	pieces []string
	ids    []int32
	eog    []int32
	// at is each id's place in pieces, -1 for none.
	at []int32
}

// IndexTokens indexes v's tokens.
func IndexTokens(v Vocabulary) *Tokens {
	x := &Tokens{}
	for id := range int32(v.Size()) {
		if v.IsEOG(id) {
			x.eog = append(x.eog, id)
			continue
		}
		if p, ok := v.Piece(id); ok && p != "" {
			x.pieces = append(x.pieces, p)
			x.ids = append(x.ids, id)
		}
	}
	order := make([]int, len(x.pieces))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return x.pieces[order[i]] < x.pieces[order[j]] })
	ps, is := make([]string, len(order)), make([]int32, len(order))
	for i, o := range order {
		ps[i], is[i] = x.pieces[o], x.ids[o]
	}
	x.pieces, x.ids = ps, is
	x.at = make([]int32, v.Size())
	for i := range x.at {
		x.at[i] = -1
	}
	for i, id := range is {
		x.at[id] = int32(i)
	}
	return x
}

// allowed is the tokens a state allows, and whether it may end.
type allowed struct {
	toks []int32
	end  bool
}

// Matcher is a grammar over one tokenizer: the automaton's states, and the
// allowed tokens of each state reached, kept for the life of the Matcher.
// It is safe for concurrent use; the per-output position is a State.
type Matcher struct {
	mu    sync.Mutex
	a     *automaton
	v     *Tokens
	first int32
	masks map[State]*allowed
}

// NewMatcher is g over a tokenizer's indexed tokens.
func NewMatcher(g *Grammar, v *Tokens) *Matcher {
	m := &Matcher{a: newAutomaton(g), v: v, masks: map[State]*allowed{}}
	m.first = m.a.start()
	return m
}

// Start is the state an output begins in.
func (m *Matcher) Start() State { return State{set: m.first} }

// feed is st after one byte of output; ok false where the grammar refuses
// it.
func (m *Matcher) feed(st State, b byte) (State, bool) {
	a := m.a
	if st.need > 0 {
		if b&0xC0 != 0x80 {
			return st, false
		}
		st.cp, st.need = st.cp<<6|rune(b&0x3F), st.need-1
		if st.need == 0 {
			c := st.cp
			if c < utf8Min(st.n) || c > utf8.MaxRune || c >= 0xD800 && c <= 0xDFFF {
				return st, false // overlong, past the last code point, or a surrogate
			}
			st.set, st.cp, st.n = a.step(st.set, c), 0, 0
			return st, st.set != 0
		}
		return st, m.plausible(st)
	}
	switch {
	case b < 0x80:
		st.set = a.step(st.set, rune(b))
		return st, st.set != 0
	case b >= 0xC2 && b <= 0xDF:
		st.need, st.n, st.cp = 1, 2, rune(b&0x1F)
	case b >= 0xE0 && b <= 0xEF:
		st.need, st.n, st.cp = 2, 3, rune(b&0x0F)
	case b >= 0xF0 && b <= 0xF4:
		st.need, st.n, st.cp = 3, 4, rune(b&0x07)
	default:
		return st, false
	}
	return st, m.plausible(st)
}

// utf8Min is the least code point a sequence of n bytes may encode.
func utf8Min(n uint8) rune {
	switch n {
	case 2:
		return 0x80
	case 3:
		return 0x800
	case 4:
		return 0x10000
	}
	return 0
}

// plausible is whether a code point begun in st can still finish as one some
// stack takes: the range its remaining bytes can reach, cut to what a valid
// sequence of its length encodes.
func (m *Matcher) plausible(st State) bool {
	lo := st.cp << (6 * st.need)
	hi := lo | (1<<(6*st.need) - 1)
	lo, hi = max(lo, utf8Min(st.n)), min(hi, utf8.MaxRune)
	// The surrogates are no code point's encoding.
	return lo <= min(hi, 0xD7FF) && m.a.meets(st.set, lo, min(hi, 0xD7FF)) ||
		max(lo, 0xE000) <= hi && m.a.meets(st.set, max(lo, 0xE000), hi)
}

// allowedAt is st's allowed tokens, computed on its first visit by walking
// the sorted pieces as a trie: a prefix the grammar refuses prunes every
// piece that shares it.
func (m *Matcher) allowedAt(st State) *allowed {
	if al, ok := m.masks[st]; ok {
		return al
	}
	al := &allowed{end: st.need == 0 && m.a.done(st.set)}
	m.walk(st, 0, len(m.v.pieces), 0, &al.toks)
	slices.Sort(al.toks)
	m.masks[st] = al
	return al
}

func (m *Matcher) walk(st State, lo, hi, d int, out *[]int32) {
	ps := m.v.pieces
	i := lo
	for ; i < hi && len(ps[i]) == d; i++ {
		*out = append(*out, m.v.ids[i])
	}
	for i < hi {
		b := ps[i][d]
		j := i + sort.Search(hi-i, func(k int) bool { return ps[i+k][d] > b })
		if next, ok := m.feed(st, b); ok {
			m.walk(next, i, j, d+1, out)
		}
		i = j
	}
}

// Allowed is the tokens st allows (sorted, shared: not to be written) and
// whether the output may end there, where an end-of-generation token is
// allowed too.
func (m *Matcher) Allowed(st State) ([]int32, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	al := m.allowedAt(st)
	return al.toks, al.end
}

// Allow writes v into bias, a row over the vocabulary, at every token st
// allows, and reports how many it wrote. A state that allows nothing and may
// not end -- a vocabulary that cannot spell what the grammar needs next --
// allows the end-of-generation tokens, so the output stops rather than
// leaves the grammar. A constrained draw opens its row of -inf with 0 before
// the draw (its logits plus the row, the caller's) and closes it with -inf
// after.
func (m *Matcher) Allow(st State, bias []float32, v float32) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	al := m.allowedAt(st)
	n := 0
	for _, t := range al.toks {
		if int(t) < len(bias) {
			bias[t] = v
			n++
		}
	}
	if al.end || len(al.toks) == 0 {
		for _, t := range m.v.eog {
			if int(t) < len(bias) {
				bias[t] = v
				n++
			}
		}
	}
	return n
}

// Accept is st after token id: false where the grammar does not allow it.
// An end-of-generation token is allowed where the output may end, and
// leaves the state as it was.
func (m *Matcher) Accept(st State, id int32) (State, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if slices.Contains(m.v.eog, id) {
		return st, st.need == 0 && m.a.done(st.set)
	}
	if id < 0 || int(id) >= len(m.v.at) || m.v.at[id] < 0 {
		return st, false
	}
	i := m.v.at[id]
	return m.acceptBytes(st, m.v.pieces[i])
}

// AcceptText is st after the bytes of s.
func (m *Matcher) AcceptText(st State, s string) (State, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.acceptBytes(st, s)
}

func (m *Matcher) acceptBytes(st State, s string) (State, bool) {
	for i := 0; i < len(s); i++ {
		var ok bool
		if st, ok = m.feed(st, s[i]); !ok {
			return st, false
		}
	}
	return st, true
}

// Done is whether the output may end at st.
func (m *Matcher) Done(st State) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return st.need == 0 && m.a.done(st.set)
}

// Final is whether st allows nothing more but the end.
func (m *Matcher) Final(st State) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	al := m.allowedAt(st)
	return al.end && len(al.toks) == 0
}

// String names a state, for errors and logs.
func (st State) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "set %d", st.set)
	if st.need > 0 {
		fmt.Fprintf(&b, " (%d byte(s) into a code point)", st.need)
	}
	return b.String()
}
