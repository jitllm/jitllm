package hb

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Sample is one request, as measured by the client. Every time is from the
// moment the request was handed to the transport.
type Sample struct {
	Arm   string `json:"arm"`
	Model string `json:"model"`
	Level int    `json:"concurrency"`
	Round int    `json:"round"`

	// TTFB is the response headers; TTFT the first chunk carrying text.
	TTFB time.Duration `json:"ttfb_ns"`
	TTFT time.Duration `json:"ttft_ns"`
	// ITL is each gap between consecutive chunks that carried text. An engine
	// that holds text back (a stop string's prefix, a partial UTF-8 rune)
	// sends one chunk for several tokens, so the gaps are per chunk, and
	// Chunks beside CompletionTokens says how far apart the two are.
	ITL   []time.Duration `json:"itl_ns"`
	Total time.Duration   `json:"total_ns"`

	Chunks           int    `json:"chunks"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
	FinishReason     string `json:"finish_reason"`
	// UsageSeen is whether the stream carried a usage object. Without one
	// the token count is the chunk count, which undercounts.
	UsageSeen bool `json:"usage_seen"`
	// Proto is the protocol the response came over, so an HTTP/2 arm is
	// shown to have spoken it.
	Proto string `json:"proto"`
	Err   string `json:"error,omitempty"`
}

// DecodeRate is the request's own tokens per second after its first token,
// (tokens - 1) over the time from first to last text chunk. NaN when it has
// fewer than two tokens or no measurable decode time.
func (s *Sample) DecodeRate() float64 {
	if s.CompletionTokens < 2 || len(s.ITL) == 0 {
		return nan()
	}
	var d time.Duration
	for _, g := range s.ITL {
		d += g
	}
	if d <= 0 {
		return nan()
	}
	return float64(s.CompletionTokens-1) / d.Seconds()
}

// TPOT is the mean gap between text chunks: vLLM's time per output token.
func (s *Sample) TPOT() time.Duration {
	if len(s.ITL) == 0 {
		return 0
	}
	var d time.Duration
	for _, g := range s.ITL {
		d += g
	}
	return d / time.Duration(len(s.ITL))
}

// Request is what one sample sends.
type Request struct {
	URL    string // the endpoint's base, without /v1
	APIKey string
	API    string // "completions" or "chat"
	Model  string // the name the endpoint serves the model under
	Prompt string
	Max    int
	Temp   float64
	// Extra is merged into the body as is: ignore_eos, an engine's own fields.
	Extra map[string]any
	// OnText, when set, is called with each text chunk's text as it is read,
	// for a client that shows the stream live. It runs on the reading
	// goroutine, so a slow one delays the next chunk's timestamp.
	OnText func(text string)
}

func (r Request) body() ([]byte, error) {
	b := map[string]any{
		"model":       r.Model,
		"max_tokens":  r.Max,
		"temperature": r.Temp,
		"stream":      true,
		// vLLM and SGLang send usage on a stream only when asked; jitllm and
		// llama-server send it either way.
		"stream_options": map[string]any{"include_usage": true},
	}
	if r.API == "chat" {
		b["messages"] = []map[string]string{{"role": "user", "content": r.Prompt}}
	} else {
		b["prompt"] = r.Prompt
	}
	for k, v := range r.Extra {
		b[k] = v
	}
	return json.Marshal(b)
}

// Do sends r and reads its stream to the end, timing every text chunk.
func Do(ctx context.Context, c *http.Client, r Request) Sample {
	var s Sample
	s.Model = r.Model
	body, err := r.body()
	if err != nil {
		s.Err = err.Error()
		return s
	}
	path := "/v1/completions"
	if r.API == "chat" {
		path = "/v1/chat/completions"
	}
	hr, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(r.URL, "/")+path, bytes.NewReader(body))
	if err != nil {
		s.Err = err.Error()
		return s
	}
	hr.Header.Set("Content-Type", "application/json")
	hr.Header.Set("Accept", "text/event-stream")
	if r.APIKey != "" {
		hr.Header.Set("Authorization", "Bearer "+r.APIKey)
	}
	t0 := time.Now()
	resp, err := c.Do(hr)
	if err != nil {
		s.Err = err.Error()
		s.Total = time.Since(t0)
		return s
	}
	defer resp.Body.Close()
	s.TTFB = time.Since(t0)
	s.Proto = resp.Proto
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		s.Err = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
		s.Total = time.Since(t0)
		return s
	}
	err = readStream(resp.Body, t0, &s, r.OnText)
	s.Total = time.Since(t0)
	if err != nil {
		s.Err = err.Error()
	}
	return s
}

// chunk is the part of a streamed OpenAI chunk, chat or legacy, the harness
// reads.
type chunk struct {
	Choices []struct {
		Text  string `json:"text"`
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	Error *json.RawMessage `json:"error"`
}

// readStream parses Server-Sent Events: `data: <json>` frames, an `event:`
// line naming the next frame (jitllm names its in-stream error `error`), and
// `data: [DONE]` ending the stream. The time of each frame with text is
// taken when its line is read, which is when the bytes arrived to within the
// reader's buffering.
func readStream(r io.Reader, t0 time.Time, s *Sample, onText func(string)) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 16<<20)
	var last time.Time
	event := ""
	done := false
	for sc.Scan() {
		line := sc.Text()
		now := time.Now()
		if line == "" {
			event = ""
			continue
		}
		if v, ok := strings.CutPrefix(line, "event:"); ok {
			event = strings.TrimSpace(v)
			continue
		}
		data, ok := strings.CutPrefix(line, "data:")
		if !ok {
			continue
		}
		data = strings.TrimSpace(data)
		if data == "[DONE]" {
			done = true
			break
		}
		if event == "error" {
			return fmt.Errorf("stream error: %s", data)
		}
		var c chunk
		if err := json.Unmarshal([]byte(data), &c); err != nil {
			return fmt.Errorf("unparseable chunk %q: %v", data, err)
		}
		if c.Error != nil {
			return fmt.Errorf("stream error: %s", *c.Error)
		}
		if c.Usage != nil {
			s.UsageSeen = true
			s.PromptTokens, s.CompletionTokens = c.Usage.PromptTokens, c.Usage.CompletionTokens
		}
		text := false
		for _, ch := range c.Choices {
			if ch.Text != "" || ch.Delta.Content != "" || ch.Delta.ReasoningContent != "" {
				text = true
				if onText != nil {
					onText(ch.Text + ch.Delta.ReasoningContent + ch.Delta.Content)
				}
			}
			if ch.FinishReason != nil && *ch.FinishReason != "" {
				s.FinishReason = *ch.FinishReason
			}
		}
		if !text {
			continue
		}
		s.Chunks++
		if last.IsZero() {
			s.TTFT = now.Sub(t0)
		} else {
			s.ITL = append(s.ITL, now.Sub(last))
		}
		last = now
	}
	if err := sc.Err(); err != nil {
		return err
	}
	if !done {
		return fmt.Errorf("the stream ended without data: [DONE]")
	}
	if !s.UsageSeen {
		s.CompletionTokens = s.Chunks
	}
	return nil
}
