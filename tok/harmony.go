package tok

import "strings"

// harmony is gpt-oss's message framing: every message is
//
//	<|start|>ROLE<|channel|>CHANNEL<|message|>TEXT<|end|>
//
// and the assistant's turn is several of them -- its reasoning on the
// "analysis" channel, then the answer on "final", ending in <|return|> (or
// <|call|> for a tool call). The ids are looked up once; ok is false for every
// other vocabulary.
type harmony struct {
	ok                                bool
	start, channel, message, end, ret int32
	call, constrain                   int32
}

func harmonyOf(v *Vocab) harmony {
	var h harmony
	for _, x := range []struct {
		name string
		dst  *int32
	}{{"<|start|>", &h.start}, {"<|channel|>", &h.channel}, {"<|message|>", &h.message},
		{"<|end|>", &h.end}, {"<|return|>", &h.ret}, {"<|call|>", &h.call}} {
		id, ok := v.ids.find(x.name)
		if !ok {
			return harmony{}
		}
		*x.dst = id
	}
	h.constrain = -1
	if id, ok := v.ids.find("<|constrain|>"); ok {
		h.constrain = id
	}
	h.ok = true
	return h
}

// DecodeChat is Decode for text a person reads. On a harmony vocabulary it
// renders the channels in the form every reasoning model here uses -- the
// analysis channel between <think> and </think>, the final and commentary
// channels as plain text -- and drops the role and channel headers, which
// Decode would otherwise leave glued to the text ("analysisThe user asks").
// On any other vocabulary it is Decode.
func (v *Vocab) DecodeChat(ids []int32) string {
	h := v.harmony
	if !h.ok {
		return v.Decode(ids)
	}
	const (
		body   = iota // message text
		header        // a role, or a channel name, until <|message|>
	)
	var b strings.Builder
	var text, name []int32
	state, think := body, false
	flush := func() {
		b.WriteString(v.Decode(text))
		text = text[:0]
	}
	for _, id := range ids {
		switch id {
		case h.start:
			flush()
			state, name = header, name[:0]
		case h.channel:
			flush()
			state, name = header, name[:0]
		case h.message:
			if strings.HasPrefix(strings.TrimSpace(v.Decode(name)), "analysis") {
				b.WriteString(thinkOpen)
				think = true
			}
			state = body
		case h.end, h.ret, h.call:
			flush()
			if think {
				b.WriteString(thinkClose)
				think = false
			}
			state = header
			name = name[:0]
		default:
			if state == body {
				text = append(text, id)
			} else if id != h.constrain {
				name = append(name, id)
			}
		}
	}
	flush()
	return b.String()
}

// The reasoning delimiters DecodeChat renders; ui/engine splits on the same.
const (
	thinkOpen  = "<think>"
	thinkClose = "</think>"
)
