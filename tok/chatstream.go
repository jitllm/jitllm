package tok

import (
	"strings"
	"unicode/utf8"
)

// ChatStream is DecodeChat one id at a time: the strings Next returns,
// concatenated, are DecodeChat of the ids it was given, byte for byte, at every
// prefix. A server streaming a generation decodes each new token once rather
// than the whole completion again on every token.
//
// It holds no bytes back: a token that ends inside a multi-byte rune returns
// that rune's leading bytes, as DecodeChat of the prefix does, and the caller
// decides what to show.
type ChatStream struct {
	v *Vocab
	// started is whether the current text run has produced any byte: a
	// WordPiece vocabulary drops the one space a run opens with
	// (decodeWPM's TrimPrefix), and a harmony vocabulary starts a run at
	// every flush.
	started bool

	// The harmony state machine, as DecodeChat runs it.
	header bool
	think  bool
	name   []int32
}

// NewChatStream starts an incremental DecodeChat.
func (v *Vocab) NewChatStream() *ChatStream { return &ChatStream{v: v} }

// Next returns the text id adds to DecodeChat of the ids so far.
func (c *ChatStream) Next(id int32) string {
	h := c.v.harmony
	if !h.ok {
		return c.piece(id)
	}
	switch id {
	case h.start, h.channel:
		c.started = false
		c.header, c.name = true, c.name[:0]
	case h.message:
		c.header = false
		if strings.HasPrefix(strings.TrimSpace(c.v.Decode(c.name)), "analysis") {
			c.think = true
			return thinkOpen
		}
	case h.end, h.ret, h.call:
		c.started = false
		c.header, c.name = true, c.name[:0]
		if c.think {
			c.think = false
			return thinkClose
		}
	default:
		if !c.header {
			return c.piece(id)
		}
		if id != h.constrain {
			c.name = append(c.name, id)
		}
	}
	return ""
}

// piece is one id's share of Decode: every vocabulary's Decode is the
// concatenation of its ids' pieces, except that WordPiece drops the one space
// its text opens with.
func (c *ChatStream) piece(id int32) string {
	one := [1]int32{id}
	if c.v.wpm == nil {
		return c.v.Decode(one[:])
	}
	if id < 0 || int(id) >= c.v.text.Len() || c.v.kind[id] == typeControl {
		return ""
	}
	s := strings.ReplaceAll(c.v.text.At(int(id)), spaceMark, " ")
	if !c.started && s != "" {
		c.started = true
		s = strings.TrimPrefix(s, " ")
	}
	return s
}

// Complete is how much of s can be shown now: all of it but a trailing
// multi-byte rune whose remaining bytes have not arrived. Only the last three
// bytes can be such a rune. An invalid byte anywhere else is not waiting for
// anything and is shown as it is; holding it back would hold back every byte
// after it.
func Complete(s string) int {
	for i := len(s) - 1; i >= 0 && i >= len(s)-utf8.UTFMax+1; i-- {
		if utf8.RuneStart(s[i]) {
			if utf8.FullRuneInString(s[i:]) {
				return len(s)
			}
			return i
		}
	}
	return len(s)
}
