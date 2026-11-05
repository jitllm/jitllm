package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/tok"
)

// `jitllm run model.jlm` with no prompt and a terminal on stdin is a chat:
// the conversation is re-rendered through the model's own template every
// turn and prefilled with PrefillCached onto a reset session, which restores
// the history's KV pages from an in-memory store rather than recomputing
// them -- the same shape as server.Engine.Generate's sessions.

const replHelp = `Commands:
  /help     this list
  /clear    forget the conversation
  /bye      exit (Ctrl+D also exits)
Ctrl+C stops the reply being written.
End a line with \ to continue it on the next, or wrap several lines in """.
`

// stdinIsTerminal is whether a person is typing: a pipe or a file is a prompt
// to read whole. The null device is a character device too, and is what a
// process started with no stdin gets; it is no one typing.
func stdinIsTerminal() bool {
	fi, err := os.Stdin.Stat()
	if err != nil || fi.Mode()&os.ModeCharDevice == 0 {
		return false
	}
	null, err := os.Stat(os.DevNull)
	return err != nil || !os.SameFile(fi, null)
}

// readTurn reads one message: a line, lines joined by a trailing backslash,
// or a block between two """ markers. It returns io.EOF at end of input.
func readTurn(r *bufio.Reader, w io.Writer) (string, error) {
	var b strings.Builder
	block := false
	for {
		line, err := r.ReadString('\n')
		if err != nil && line == "" {
			if b.Len() > 0 && !block {
				return b.String(), nil
			}
			return "", io.EOF
		}
		line = strings.TrimRight(line, "\r\n")
		switch {
		case block:
			if i := strings.Index(line, `"""`); i >= 0 {
				b.WriteString(line[:i])
				return b.String(), nil
			}
			b.WriteString(line + "\n")
		case b.Len() == 0 && strings.HasPrefix(line, `"""`):
			rest := line[3:]
			if i := strings.Index(rest, `"""`); i >= 0 {
				return rest[:i], nil
			}
			block = true
			if rest != "" {
				b.WriteString(rest + "\n")
			}
		case strings.HasSuffix(line, `\`):
			b.WriteString(strings.TrimSuffix(line, `\`) + "\n")
		default:
			b.WriteString(line)
			return b.String(), nil
		}
		fmt.Fprint(w, "... ")
	}
}

// repl runs the conversation on st until /bye or end of input. n caps each
// reply; the session's context caps the conversation.
func repl(m *model.Model, st *engine.Instance, sm *engine.Sampler, system string, n int, cached bool, in io.Reader, out io.Writer) error {
	if !m.HasChatTemplate() {
		return fmt.Errorf("the model carries no chat template, so there is no conversation to hold; " +
			"give a prompt and it completes it instead")
	}
	// The history's pages are restored from here at every turn. A cache key
	// of its own, so nothing outside this process names these pages.
	if !cached {
		st.SetKVStore(engine.NewMemStore())
		if err := st.SetCacheKey("repl"); err != nil {
			return err
		}
	}
	var stop atomic.Bool
	var generating atomic.Bool
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt)
	defer signal.Stop(sig)
	go func() {
		for range sig {
			if generating.Load() {
				stop.Store(true)
			} else {
				fmt.Fprint(out, "\nUse Ctrl+D or /bye to exit.\n>>> ")
			}
		}
	}()

	var msgs []model.ChatMessage
	if system != "" {
		msgs = append(msgs, model.ChatMessage{Role: "system", Content: system})
	}
	base := len(msgs)
	r := bufio.NewReader(in)
	fmt.Fprintln(out, `Send a message (/? for help)`)
	for {
		fmt.Fprint(out, ">>> ")
		text, err := readTurn(r, out)
		if errors.Is(err, io.EOF) {
			fmt.Fprintln(out)
			return nil
		}
		switch strings.TrimSpace(text) {
		case "":
			continue
		case "/bye", "/exit":
			return nil
		case "/help", "/?":
			fmt.Fprint(out, replHelp)
			continue
		case "/clear":
			msgs = msgs[:base]
			fmt.Fprintln(out, "Cleared the conversation.")
			continue
		}
		if strings.HasPrefix(text, "/") && !strings.ContainsAny(strings.TrimSpace(text), " \n") {
			fmt.Fprintf(out, "Unknown command %q; /help lists them.\n", text)
			continue
		}
		turn := append(msgs, model.ChatMessage{Role: "user", Content: text})
		ids, err := m.ChatIDs(turn, true)
		if err != nil {
			return err
		}
		if len(ids) >= st.MaxSeq() {
			fmt.Fprintf(out, "The conversation is %d positions and fills the context of %d; /clear starts again.\n",
				len(ids), st.MaxSeq())
			continue
		}
		stop.Store(false)
		generating.Store(true)
		reply, err := replyTo(m, st, sm, ids, n, &stop, out)
		generating.Store(false)
		if err != nil {
			return err
		}
		if stop.Load() {
			fmt.Fprintln(out, "\n[stopped]")
		} else {
			fmt.Fprintln(out)
		}
		// A reply stopped before any output answered nothing, and a template
		// may refuse two user turns in a row: the question goes with it.
		if strings.TrimSpace(reply) != "" {
			msgs = append(turn, model.ChatMessage{Role: "assistant", Content: reply})
		}
	}
}

// replyTo prefills ids onto a reset session and streams the reply to out until
// the model ends it, n tokens, the context, or stop.
func replyTo(m *model.Model, st *engine.Instance, sm *engine.Sampler, ids []int32, n int, stop *atomic.Bool, out io.Writer) (string, error) {
	st.Reset()
	logits, err := st.PrefillCached(ids)
	if err != nil {
		return "", err
	}
	room := st.MaxSeq() - len(ids) - 1
	if n <= 0 || n > room {
		n = room
	}
	var reply strings.Builder
	cs, held := m.Vocab.NewChatStream(), ""
	greedy := sm.Temp <= 0
	pending := int32(-1)
	for i := 0; i < n && !stop.Load(); i++ {
		next := pending
		if next < 0 {
			var err error
			if next, err = st.Sample(sm, logits); err != nil {
				return "", err
			}
		}
		sm.Observe(next)
		if m.Vocab.IsEOG(next) {
			break
		}
		held += cs.Next(next)
		if k := tok.Complete(held); k > 0 {
			fmt.Fprint(out, held[:k])
			reply.WriteString(held[:k])
			held = held[k:]
		}
		if greedy {
			pending, err = st.ForwardGreedy(next)
		} else {
			pending, err = st.ForwardSample(next, sm)
		}
		if err != nil {
			return reply.String(), err
		}
	}
	return reply.String(), nil
}
