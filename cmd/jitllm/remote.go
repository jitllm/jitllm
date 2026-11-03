package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"

	"github.com/jitllm/jitllm/internal/localserver"
)

// `jitllm run NAME` uses the local service when one answers: the service
// resolves the name (an id, a file name, an alias) and loads it on demand,
// so a second engine is not started beside the one already holding the
// model. With no service it finds NAME in the models folder and runs it in
// this process, as a path always does. Which one it is, it says in one line
// on stderr.
//
// The service is reached through its OpenAI-compatible HTTP API, which is
// plain JSON: the root module keeps its one dependency, and the generated
// ConnectRPC stubs live in the server module.

// remoteFlags are the run flags the service carries. Any other flag set on
// the command line is about this process's engine (devices, placement, the
// KV cache, a picture, a profile), so the run is embedded.
func remoteFlags() map[string]bool {
	return map[string]bool{
		"n": true, "chat": true, "system": true, "temp": true, "top-k": true, "top-p": true,
		"min-p": true, "repeat-penalty": true, "seed": true, "addr": true, "api-key": true, "embedded": true,
	}
}

// keyOf is the API key to send: the -api-key flag, else $JITLLM_API_KEY. The
// libraries read no environment; this command does, and hands the key in.
func keyOf(flagValue string) string {
	if flagValue != "" {
		return flagValue
	}
	return os.Getenv("JITLLM_API_KEY")
}

// looksLikePath is an argument that names a file rather than a model: one
// that exists, or that has a directory or a model file's extension in it.
func looksLikePath(arg string) bool {
	if _, err := os.Stat(arg); err == nil {
		return true
	}
	ext := strings.ToLower(filepath.Ext(arg))
	return strings.ContainsAny(arg, `/\`) || ext == ".jlm" || ext == ".gguf"
}

// embedRemoteFlags are embed's flags the service carries.
func embedRemoteFlags() map[string]bool {
	return map[string]bool{"lines": true, "addr": true, "api-key": true, "embedded": true}
}

// engineFlags is the flags set that only an embedded run honours: those
// not in remote.
func engineFlags(fs *flag.FlagSet, remote map[string]bool) []string {
	var out []string
	fs.Visit(func(f *flag.Flag) {
		if !remote[f.Name] {
			out = append(out, "-"+f.Name)
		}
	})
	return out
}

// runTarget decides where `jitllm run arg` runs. It returns the service to
// use, or the container to open in this process. addrOpts lets a gate stand in
// other default and old-default addresses (localserver.WithDefaults), so it
// does not depend on what runs on this box.
func runTarget(ctx context.Context, arg, addr, key string, embedded bool, engineOnly []string, note io.Writer, addrOpts ...localserver.Option) (svc *localserver.Info, path string, err error) {
	if looksLikePath(arg) {
		fmt.Fprintf(note, "jitllm: running %s in this process\n", arg)
		return nil, arg, nil
	}
	if !embedded && len(engineOnly) == 0 {
		in, err := localserver.Locate(ctx, addr, append([]localserver.Option{localserver.WithAPIKey(key)}, addrOpts...)...)
		if err == nil {
			fmt.Fprintf(note, "jitllm: using the service at %s (jitllmd %s) for %s\n",
				localserver.BaseURL(in.Addr), in.Version, arg)
			if in.Legacy {
				fmt.Fprintf(note, "jitllm: %s\n", localserver.LegacyNotice)
			}
			return &in, "", nil
		}
		if errors.Is(err, localserver.ErrIncompatible) || errors.Is(err, localserver.ErrUnauthorized) {
			return nil, "", fmt.Errorf("jitllm: %v; or run it in this process with -embedded", err)
		}
	}
	dirs := localserver.SearchDirs(defaultModelDir())
	p, _, err := localserver.Find(dirs, arg)
	if err != nil {
		return nil, "", err
	}
	if p == "" {
		return nil, "", fmt.Errorf("jitllm: no model %q in %s (by file name or alias); "+
			"`jitllm convert %s` fetches a catalog model there (`jitllm library` lists them)",
			arg, strings.Join(dirs, ", "), arg)
	}
	why := "no service answers at " + addr
	switch {
	case embedded:
		why = "-embedded"
	case len(engineOnly) > 0:
		why = strings.Join(engineOnly, " ") + " configures this process's engine"
	}
	fmt.Fprintf(note, "jitllm: running %s in this process (%s)\n", p, why)
	return nil, p, nil
}

// remoteRun is what a run asks the service for.
type remoteRun struct {
	base, key, model string
	n                int
	system           string
	sampling         map[string]any
}

func newRemoteRun(addr, key, model string, n int, system string, fs *flag.FlagSet, sm samplingFlags) remoteRun {
	s := map[string]any{"temperature": sm.temp}
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "top-k":
			s["top_k"] = sm.topK
		case "top-p":
			s["top_p"] = sm.topP
		case "min-p":
			s["min_p"] = sm.minP
		case "repeat-penalty":
			s["repetition_penalty"] = sm.repPen
		case "seed":
			s["seed"] = sm.seed
		}
	})
	return remoteRun{base: localserver.BaseURL(addr), key: key, model: model, n: n, system: system, sampling: s}
}

// samplingFlags are run's sampling flags' values.
type samplingFlags struct {
	temp, topP, minP, repPen float64
	topK                     int
	seed                     int64
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// complete streams one request to out and returns the text. chat sends
// messages to /v1/chat/completions; otherwise prompt goes to /v1/completions.
func (r remoteRun) complete(ctx context.Context, chat bool, msgs []chatMessage, prompt string, out io.Writer) (string, error) {
	body := map[string]any{"model": r.model, "stream": true}
	for k, v := range r.sampling {
		body[k] = v
	}
	if r.n > 0 {
		body["max_tokens"] = r.n
	}
	route := "/v1/completions"
	if chat {
		route = "/v1/chat/completions"
		body["messages"] = msgs
	} else {
		body["prompt"] = prompt
	}
	b, err := json.Marshal(body)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.base+route, bytes.NewReader(b))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	localserver.Authorize(req, r.key)
	resp, err := localserver.Client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		var e struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		raw, _ := io.ReadAll(resp.Body)
		if json.Unmarshal(raw, &e) == nil && e.Error.Message != "" {
			return "", fmt.Errorf("the service: %s", e.Error.Message)
		}
		return "", fmt.Errorf("the service answered %s: %s", resp.Status, bytes.TrimSpace(raw))
	}
	var text strings.Builder
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64<<10), 16<<20)
	for sc.Scan() {
		line, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok {
			continue
		}
		if line == "[DONE]" {
			break
		}
		var ev struct {
			Choices []struct {
				Text  string `json:"text"`
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			return text.String(), fmt.Errorf("the service's stream: %w", err)
		}
		if ev.Error != nil {
			return text.String(), fmt.Errorf("the service: %s", ev.Error.Message)
		}
		for _, c := range ev.Choices {
			piece := c.Text + c.Delta.Content
			text.WriteString(piece)
			fmt.Fprint(out, piece)
		}
	}
	if err := sc.Err(); err != nil && ctx.Err() == nil {
		return text.String(), err
	}
	return text.String(), nil
}

// embed asks the service's /v1/embeddings for each text's vector and writes
// one JSON array per text, as `jitllm embed` does in this process.
func embedRemote(addr, key, model string, texts []string, out io.Writer) error {
	for _, t := range texts {
		b, err := json.Marshal(map[string]any{"model": model, "input": t})
		if err != nil {
			return err
		}
		req, err := http.NewRequest(http.MethodPost, localserver.BaseURL(addr)+"/v1/embeddings", bytes.NewReader(b))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		localserver.Authorize(req, key)
		resp, err := localserver.Client.Do(req)
		if err != nil {
			return err
		}
		var r struct {
			Data []struct {
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		err = json.NewDecoder(resp.Body).Decode(&r)
		resp.Body.Close()
		switch {
		case err != nil:
			return fmt.Errorf("the service's answer: %w", err)
		case r.Error != nil:
			return fmt.Errorf("the service: %s", r.Error.Message)
		case len(r.Data) != 1:
			return fmt.Errorf("the service returned %d vectors for one text", len(r.Data))
		}
		v, err := json.Marshal(r.Data[0].Embedding)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(out, "%s\n", v); err != nil {
			return err
		}
	}
	return nil
}

// once is a one-shot run against the service.
func (r remoteRun) once(chat bool, prompt string, out io.Writer) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	var msgs []chatMessage
	if chat {
		if r.system != "" {
			msgs = append(msgs, chatMessage{"system", r.system})
		}
		msgs = append(msgs, chatMessage{"user", prompt})
	}
	_, err := r.complete(ctx, chat, msgs, prompt, out)
	fmt.Fprintln(out)
	if ctx.Err() != nil {
		return nil
	}
	return err
}

// repl is the interactive chat (repl.go's) against the service: the
// conversation is kept here and sent whole each turn; the service renders it
// through the model's template.
func (r remoteRun) repl(in io.Reader, out io.Writer) error {
	var msgs []chatMessage
	if r.system != "" {
		msgs = append(msgs, chatMessage{"system", r.system})
	}
	base := len(msgs)
	// ^C stops the reply being written, as in this process's chat.
	var cancel context.CancelFunc
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt)
	defer signal.Stop(sig)
	busy := make(chan context.CancelFunc, 1)
	go func() {
		for range sig {
			select {
			case c := <-busy:
				c()
			default:
				fmt.Fprint(out, "\nUse Ctrl+D or /bye to exit.\n>>> ")
			}
		}
	}()
	br := bufio.NewReader(in)
	fmt.Fprintln(out, `Send a message (/? for help)`)
	for {
		fmt.Fprint(out, ">>> ")
		text, err := readTurn(br, out)
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
		turn := append(msgs, chatMessage{"user", text})
		var ctx context.Context
		ctx, cancel = context.WithCancel(context.Background())
		busy <- cancel
		reply, err := r.complete(ctx, true, turn, "", out)
		select {
		case <-busy:
		default:
		}
		stopped := ctx.Err() != nil
		cancel()
		if err != nil && !stopped {
			return err
		}
		if stopped {
			fmt.Fprintln(out, "\n[stopped]")
		} else {
			fmt.Fprintln(out)
		}
		if strings.TrimSpace(reply) != "" {
			msgs = append(turn, chatMessage{"assistant", reply})
		}
	}
}
