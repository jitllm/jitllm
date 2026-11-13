package tok

import "slices"

// Infill is what llama.cpp's infill sampler reads of a vocabulary, as tables a
// composed draw gathers from. Each token's piece is the bytes it adds to
// decoded text (Piece: none for a control or unused token).
//
// Class[t] is the lowest id whose piece is t's, t itself for a token with no
// piece. Parent[c], for a class c, is the class of the longest proper prefix
// of c's piece that is a token's piece, c itself where there is none; Depth is
// the longest chain of parents from any class to its root. Token a's piece is
// a prefix of token b's (a != b, a's piece not empty) exactly when Class[a] is
// Class[b] or one of the Depth parents up from Class[b].
//
// EOG is the end-of-generation ids (IsEOG), and EOT the token the sampler
// ends on when nothing but end-of-generation tokens survives: the first of
// llama.cpp's end-of-turn names the vocabulary holds, else EOS.
//
// The tables are built from the pieces when a model opens. They could be
// stated at conversion instead, with the next container version.
type Infill struct {
	Class, Parent []uint32
	Depth         int
	EOG           []int32
	EOT           int32
}

// Infill is the vocabulary's infill tables.
func (v *Vocab) Infill() Infill {
	pieces := make([]string, len(v.text))
	for t := range pieces {
		pieces[t], _ = v.Piece(int32(t))
	}
	in := Infill{EOG: slices.Clone(v.eog), EOT: v.EOS}
	in.Class, in.Parent, in.Depth = InfillOf(pieces)
	for _, name := range []string{"<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>",
		"<|end_of_text|>", "<EOT>", "_<EOT>", "[EOT]", "<｜end▁of▁sentence｜>", "<end_of_utterance>"} {
		if id, ok := v.ids[name]; ok {
			in.EOT = id
			break
		}
	}
	return in
}

// InfillOf is Infill's class and parent tables, and their depth, over each
// token's piece ("" for a token that adds none).
func InfillOf(pieces []string) (class, parent []uint32, depth int) {
	n := len(pieces)
	class, parent = make([]uint32, n), make([]uint32, n)
	first := make(map[string]uint32, n)
	for t, p := range pieces {
		class[t], parent[t] = uint32(t), uint32(t)
		if p == "" {
			continue
		}
		if c, seen := first[p]; seen {
			class[t] = c
		} else {
			first[p] = uint32(t)
		}
	}
	for t, p := range pieces {
		if p == "" || class[t] != uint32(t) {
			continue
		}
		for l := len(p) - 1; l > 0; l-- {
			if c, ok := first[p[:l]]; ok {
				parent[t] = c
				break
			}
		}
	}
	// A parent's piece is shorter than its child's, so classes in order of
	// their pieces' lengths see their parent's depth already set.
	order := make([]int, 0, len(first))
	for _, c := range first {
		order = append(order, int(c))
	}
	slices.SortFunc(order, func(a, b int) int { return len(pieces[a]) - len(pieces[b]) })
	at := make([]int, n)
	for _, c := range order {
		if p := int(parent[c]); p != c {
			at[c] = at[p] + 1
			depth = max(depth, at[c])
		}
	}
	return class, parent, depth
}
