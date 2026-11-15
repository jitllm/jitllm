package graph

import (
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// TailOp lowers a stated op over the tail a recurrent state keeps past its
// rule's (jlm.TOpNGramRows, jlm.TOpTailConv), the op's tail at off floats
// into a state of span floats a sequence. It returns the op's value and how
// many floats of the tail the op keeps, which a later tail op steps past.
func TailOp(b *Builder, ins []ValueID, op *jlm.TowerOp, off, span int) (ValueID, int, error) {
	at := b.G.StateVersion(StateRecState)
	in, out := []StateRef{{State: StateRecState, Version: at}}, []StateRef{{State: StateRecState, Version: at + 1}}
	switch op.Code {
	case jlm.TOpTailConv:
		w, taps, dil := int(op.Width), int(op.Span), int(op.Offset)
		return b.Emit(OpStateConv, ins, []Result{{Name: "tail.conv", Width: w}},
			[]Binding{{Role: op.Bind[0], K: taps, Rows: w}},
			Attrs{Taps: taps, Dil: dil, Offset: off, Span: span}, in, out)[0], (taps - 1) * dil * w, nil
	case jlm.TOpNGramRows:
		hs := op.Hash
		n := int(op.Span)
		heads := (n - 1) * int(hs.HeadsPer)
		if heads <= 0 || int(op.Width)%heads != 0 || len(hs.Off) < heads || len(hs.Voc) < heads {
			return 0, 0, fmt.Errorf("%d-grams %d wide over %d heads' ranges", n, op.Width, len(hs.Off))
		}
		rowW := int(op.Width) / heads
		tok := b.Emit(OpToken, nil, []Result{{Name: "ng.tok", Width: 1, Elem: ElemU32}}, nil, Attrs{}, nil, nil)[0]
		win := b.Emit(OpTokenWindow, []ValueID{tok}, []Result{{Name: "ng.win", Width: n, Elem: ElemU32}}, nil,
			Attrs{Taps: n, Offset: off, Span: span, Words: []uint64{uint64(hs.Image)}}, in, out)[0]
		hashed := ngramHashRows(b, win, n, hs)
		rows := make([]ValueID, heads)
		for h := range heads {
			col := b.Emit(OpSlice, []ValueID{hashed}, []Result{{Name: "ng.row", Width: 1, Elem: ElemU32}}, nil,
				Attrs{Offset: h, Width: 1}, nil, nil)[0]
			rows[h] = b.Emit(OpEmbed, []ValueID{col}, []Result{{Name: "ng.emb", Width: rowW}},
				[]Binding{{Role: op.Bind[0], K: rowW, Rows: int(hs.Voc[h]), From: int(hs.Off[h])}},
				Attrs{}, nil, nil)[0]
		}
		return b.Emit(OpConcat, rows, []Result{{Name: "ngram", Width: int(op.Width)}}, nil, Attrs{}, nil, nil)[0], n - 1, nil
	}
	return 0, 0, fmt.Errorf("op code %d keeps no tail", op.Code)
}

// ngramHashRows is each row's table row a head of hs over its window win (n
// words: the row's token, then its predecessors nearest first, each its id
// plus one, 0 for none), in words: the window's ids, a predecessor that is
// none or past an end of sequence read as the end of sequence; each g-gram's
// product, the (g-1)-gram's xored with its last id times that id's
// multiplier, modulo 2^64 (the 2-gram's the first two ids'); and each head's
// row, its gram's product modulo the head's vocabulary plus its offset.
func ngramHashRows(b *Builder, win ValueID, n int, hs *jlm.NGramHash) ValueID {
	at := func(s int) ValueID {
		return b.Emit(OpSlice, []ValueID{win}, []Result{{Name: fmt.Sprintf("ng.w%d", s), Width: 1, Elem: ElemU32}}, nil,
			Attrs{Offset: s, Width: 1}, nil, nil)[0]
	}
	eos := Word(hs.EOS)
	ids := []ValueID{at(0)}
	for s := 1; s < n; s++ {
		var e WordTerms
		none := e.Fn(MapEqual, WordIn(0), Word(0))
		in := []ValueID{at(s)}
		if s > 1 {
			none = e.Fn(MapAdd, none, e.Fn(MapEqual, WordIn(1), eos))
			in = append(in, ids[s-1])
		}
		e.Fn(MapWhere, none, eos, e.Fn(MapSub, WordIn(0), Word(1)))
		ids = append(ids, b.Words(fmt.Sprintf("ng.id%d", s), 1, &e, in...))
	}
	gram := b.MulXor("ng.g1", nil, ids[0], hs.Mul[0])
	var rows []ValueID
	for g := 2; g <= n; g++ {
		gram = b.MulXor(fmt.Sprintf("ng.g%d", g), &gram, ids[g-1], hs.Mul[g-1])
		for k := range int(hs.HeadsPer) {
			h := (g-2)*int(hs.HeadsPer) + k
			rows = append(rows, b.ModAdd(fmt.Sprintf("ng.h%d", h), gram, hs.Voc[h], hs.Off[h]))
		}
	}
	return b.Emit(OpConcat, rows, []Result{{Name: "ng.rows", Width: len(rows), Elem: ElemU32}}, nil, Attrs{}, nil, nil)[0]
}
