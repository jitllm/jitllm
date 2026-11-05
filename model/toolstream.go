package model

import (
	"strings"

	"github.com/jitllm/jitllm/tok"
)

// ToolStream reads tool calls out of a reply as it streams, in one syntax.
//
// It is fed the reply token by token (Push) or, from a source with no ids, as
// text (PushText). Each push returns the text that is now safe to show -- it
// is not, and cannot become, part of a call -- and the calls whose markup has
// closed, so a client sees each call as soon as the model has finished
// writing it. Finish returns what is left.
//
// Calls are read from the RAW text, with control tokens spelled out
// (tok.Vocab.Literal), while the text shown is the chat text a person reads
// (tok.ChatStream: control tokens dropped, harmony's headers removed and its
// analysis channel between <think> tags). A token is shown or not as a whole
// unless its chat text IS its raw text, which is every ordinary token, and is
// then cut at the byte.
type ToolStream struct {
	syn   ToolSyntax
	tools ToolSet
	v     *tok.Vocab
	cs    *tok.ChatStream

	raw    strings.Builder
	pieces []toolPiece
	// shown is how far the raw text has been resolved and its text shown;
	// next the first piece not wholly shown.
	shown, next int
	// sent is how many calls have been returned.
	sent int
}

// toolPiece is one token's (or one text push's) share of the reply: its raw
// bytes [start, end), and its chat text. plain is chat == raw.
type toolPiece struct {
	start, end int
	chat       string
	plain      bool
}

// NewToolStream reads a reply of this model's for calls in its syntax to
// the declared tools.
func (m *Model) NewToolStream(tools ToolSet) *ToolStream {
	return &ToolStream{syn: m.ToolSyntax(), tools: tools, v: m.Vocab, cs: m.Vocab.NewChatStream()}
}

// NewTextToolStream reads text pushes, with no vocabulary to spell control
// tokens: a source that has only text.
func NewTextToolStream(syn ToolSyntax, tools ToolSet) *ToolStream {
	return &ToolStream{syn: syn, tools: tools}
}

// Syntax is the syntax the stream reads.
func (t *ToolStream) Syntax() ToolSyntax { return t.syn }

// Push takes the next token of the reply.
func (t *ToolStream) Push(id int32) (string, []ToolCall) {
	raw := t.v.Literal(id)
	chat := t.cs.Next(id)
	return t.add(raw, chat)
}

// PushText takes the next text of the reply.
func (t *ToolStream) PushText(s string) (string, []ToolCall) { return t.add(s, s) }

func (t *ToolStream) add(raw, chat string) (string, []ToolCall) {
	st := t.raw.Len()
	t.raw.WriteString(raw)
	t.pieces = append(t.pieces, toolPiece{start: st, end: t.raw.Len(), chat: chat, plain: raw == chat})
	text := t.raw.String()
	spans := t.syn.find(text, t.tools)
	f := t.syn.hold(text, spans)
	if f < 0 {
		f = len(text)
	}
	return t.emit(spans, f)
}

// Finish ends the reply: the text still owed, the calls not yet returned,
// and the whole reply's text with every call cut out.
func (t *ToolStream) Finish() (tail string, calls []ToolCall, content string) {
	text := t.raw.String()
	spans := t.syn.find(text, t.tools)
	// At the end every call is whatever it is: one whose closing marker never
	// came (the model stopped at its end-of-generation token) still counts.
	for i := range spans {
		spans[i].closed = true
	}
	tail, calls = t.emit(spans, len(text))
	var b strings.Builder
	for _, p := range t.pieces {
		b.WriteString(t.outside(p, spans, p.start, p.end))
	}
	return tail, calls, strings.TrimSpace(b.String())
}

// emit shows the pieces resolved before raw offset f and returns the calls of
// the closed spans not yet returned.
func (t *ToolStream) emit(spans []toolSpan, f int) (string, []ToolCall) {
	var out strings.Builder
	for t.next < len(t.pieces) {
		p := t.pieces[t.next]
		if p.end <= f {
			out.WriteString(t.outside(p, spans, max(p.start, t.shown), p.end))
			t.shown = p.end
			t.next++
			continue
		}
		if p.plain && f > t.shown {
			out.WriteString(t.outside(p, spans, max(p.start, t.shown), f))
			t.shown = f
		}
		break
	}
	var calls []ToolCall
	n := 0
	for _, sp := range spans {
		if !sp.closed {
			break
		}
		if !sp.ok {
			continue
		}
		for _, c := range sp.calls {
			if n >= t.sent {
				calls = append(calls, c)
			}
			n++
		}
	}
	t.sent = max(t.sent, n)
	return out.String(), calls
}

// outside is piece p's text over raw [from, to) less every call's markup: a
// plain piece cut at the byte, any other shown whole when it begins outside
// every call.
func (t *ToolStream) outside(p toolPiece, spans []toolSpan, from, to int) string {
	if !p.plain {
		if from != p.start {
			return ""
		}
		for _, sp := range spans {
			if sp.ok && p.start >= sp.start && p.start < sp.end {
				return ""
			}
		}
		return p.chat
	}
	text := t.raw.String()
	var b strings.Builder
	at := from
	for _, sp := range spans {
		if !sp.ok || sp.end <= at || sp.start >= to {
			continue
		}
		if sp.start > at {
			b.WriteString(text[at:sp.start])
		}
		at = max(at, min(sp.end, to))
	}
	if at < to {
		b.WriteString(text[at:to])
	}
	return b.String()
}
