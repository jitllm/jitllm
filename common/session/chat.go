package session

import "strings"

// ChatRequest is one generation, as the screen hands it over.
type ChatRequest struct {
	// Chat selects the template path. False is a raw completion, the default,
	// matching the engine's benchmarks and golden comparisons.
	Chat bool

	// Messages is the whole conversation, rendered for the template. It is set
	// when Chat, and also on a completion that carries a picture, since a VLM
	// has nowhere to put one outside its turn structure. The engine must pass it
	// to Model.ChatIDs, never ChatPrompt followed by Encode: a rendered template
	// carries its own specials and must be encoded with addSpecial=false.
	Messages []ChatMessage

	// Prompt is the raw text the user typed. On the completion path it is the
	// prompt; on the chat path it is carried only so the engine can report it.
	Prompt string

	// Reply is the transcript index of the assistant turn the engine fills. The
	// screen has already appended it, empty, so the bubble shows before the
	// first token.
	Reply int

	// MaxTokens caps the decode.
	MaxTokens int

	// Sampling is how this reply is drawn from the logits. It is per request,
	// so a knob moved between turns applies to the next one and the worker never
	// reads a UI signal.
	Sampling Sampling
}

// ChatMessage is one turn as a chat template reads it. Role is the template's
// vocabulary, not this package's: "system", "user" and "assistant" are
// conventional and a template is free to read others.
type ChatMessage struct {
	Role    string
	Content string
	// Images are the pictures this turn carries, as paths, placed before
	// Content. No tower runs them yet: the engine refuses a turn with one.
	Images []string
}

// DefaultMaxTokens is the decode cap when a front end does not set one. A
// reply that hits it says so; the session's window bounds it in any case.
const DefaultMaxTokens = 4096

// ChatMessages renders the transcript into the message list a chat template
// reads. It drops turns that never happened as far as the model is concerned:
//
//   - an error turn is the engine talking to the user, not the assistant.
//   - an empty assistant turn is a placeholder or a reply stopped before any
//     output -- and the question it was answering goes with it, since nothing
//     answered it. Kept, it would put two user messages in a row, which a
//     template like Mistral's refuses to render at all ("Conversation roles
//     must alternate"), so one failed reply would break every turn after it.
//   - a system turn in the transcript is display; the system prompt is its
//     own field, and emitting both would send it twice.
//
// system is emitted first when non-empty, and prompt last as the user's turn,
// carrying images.
func ChatMessages(system string, turns []Turn, prompt string, images ...string) []ChatMessage {
	out := make([]ChatMessage, 0, len(turns)+2)
	if s := strings.TrimSpace(system); s != "" {
		out = append(out, ChatMessage{Role: "system", Content: s})
	}
	// asked is the question waiting for its answer; it is emitted only once
	// one arrives.
	var asked *ChatMessage
	for _, t := range turns {
		switch t.Role {
		case RoleUser:
			if t.Text != "" || len(t.Images) > 0 {
				asked = &ChatMessage{Role: "user", Content: t.Text, Images: t.Images}
			}
		case RoleAssistant:
			if t.Text == "" {
				continue
			}
			if asked != nil {
				out = append(out, *asked)
				asked = nil
			}
			out = append(out, ChatMessage{Role: "assistant", Content: t.Text})
		}
	}
	if p := strings.TrimSpace(prompt); p != "" || len(images) > 0 {
		out = append(out, ChatMessage{Role: "user", Content: p, Images: images})
	}
	return out
}
