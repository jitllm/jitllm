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
		hashed, err := b.NGramRows(win, n, hs)
		if err != nil {
			return 0, 0, err
		}
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
