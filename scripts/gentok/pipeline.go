package main

// Fetching a model's tokenizer.json. The parsing lives in package pretok, so
// the generator and the loader share one parser: pretok.ParsePreTokenizer
// produces the []pretok.Op the engine runs, and pretok.Op.GoSource prints it.
//
// Only the first few hundred KB is fetched: pre_tokenizer sits near the top of
// tokenizer.json and the megabytes below it are vocabulary the GGUF already
// carries.

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/jitllm/jitllm/tok/pretok"
)

// fetchPipeline reads a model's pre_tokenizer and returns its ordered ops, the
// raw JSON object they were parsed from (for -dump), and the repo that served
// it -- a mirror or stand-in when the defining repo could not.
func fetchPipeline(repo string) ([]pretok.Op, []byte, string, error) {
	if m, ok := mirrors()[repo]; ok {
		if ops, raw, err := fetchPipelineFrom(m); err == nil {
			return ops, raw, m, nil
		}
	}
	ops, raw, err := fetchPipelineFrom(repo)
	if si, ok := standIns()[repo]; ok && err != nil {
		ops, raw, err = fetchPipelineFrom(si)
		return ops, raw, si, err
	}
	return ops, raw, repo, err
}

// standIns maps a defining repo that publishes no tokenizer.json at all to
// another artefact of the same tokenizer, chosen by the user rather than
// proven byte-identical (there is nothing to compare against). Consulted only
// when the defining repo fails.
//
// THUDM/glm-4-9b-chat (zai-org) ships a tiktoken tokenizer.model and
// Python; zai-org/glm-4-9b-chat-hf is the same organisation's
// HuggingFace-format conversion of it and does ship a tokenizer.json.
func standIns() map[string]string {
	return map[string]string{
		"THUDM/glm-4-9b-chat": "zai-org/glm-4-9b-chat-hf",
	}
}

func fetchPipelineFrom(repo string) ([]pretok.Op, []byte, error) {
	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/tokenizer.json", repo)
	cl := &http.Client{Timeout: 90 * time.Second}
	var lastErr error
	// Growing ranges: a truncated object is an error from the parser, so a
	// pre_tokenizer that sits below the first window is re-fetched rather than
	// half-read.
	for _, n := range []int{512 << 10, 4 << 20, 24 << 20} {
		req, err := http.NewRequest("GET", url, nil)
		if err != nil {
			return nil, nil, err
		}
		req.Header.Set("Range", fmt.Sprintf("bytes=0-%d", n-1))
		if t := os.Getenv("HF_TOKEN"); t != "" {
			req.Header.Set("Authorization", "Bearer "+t)
		}
		resp, err := doWithBackoff(cl, req)
		if err != nil {
			return nil, nil, err
		}
		body, rerr := io.ReadAll(resp.Body)
		resp.Body.Close()
		switch resp.StatusCode {
		case 401, 403:
			if bytes.Contains(body, []byte("restricted")) {
				return nil, nil, ErrGated
			}
			return nil, nil, ErrMissing
		case 200, 206:
		default:
			return nil, nil, fmt.Errorf("http %d", resp.StatusCode)
		}
		if rerr != nil {
			return nil, nil, rerr
		}
		ops, err := pretok.ParsePreTokenizer(bytes.NewReader(body))
		if err != nil {
			lastErr = err
			if len(body) < n {
				break // the whole file fits in this window; a bigger one adds nothing
			}
			continue
		}
		raw, err := pretok.PreTokenizerObject(body)
		if err != nil {
			return nil, nil, err
		}
		return ops, raw, nil
	}
	return nil, nil, lastErr
}

// mirrors maps a gated defining repo to an ungated copy of the same tokenizer.
//
// A mirror is only usable if it is provably the same file. HuggingFace exposes
// Git blob IDs, so identity is checkable rather than assumed:
//
//	SHA1("blob " + len + NUL + bytes)
//
// NousResearch/Meta-Llama-3-8B and meta-llama/Meta-Llama-3-8B list the same
// tokenizer.json blob, so the mirror is the artefact and not a lookalike.
//
// Anything not verified this way stays unresolved.
//
// An LFS file is proved the same way, one level up: the blob a tree listing
// reports is the LFS pointer's, and the pointer carries the content's sha256
// and size. Llama 4's tokenizer.json is stored this way; a gated repo redacts
// the sha256 itself but not the pointer blob, so equal pointer blobs mean one
// sha256 and one file.
//
// command-r's defining repo is gated and ungated ones carry its file:
// CohereForAI/aya-expanse-8b (gated too) and mlx-community/aya-23-8B-8bit
// (not) list the same blob as CohereForAI/c4ai-command-r-v01.
// Xenova/c4ai-command-r-v01-tokenizer, the obvious "mirror", is not it: its
// blob differs and is 27 bytes shorter.
func mirrors() map[string]string {
	return map[string]string{
		"meta-llama/Meta-Llama-3-8B":                "NousResearch/Meta-Llama-3-8B",
		"meta-llama/Llama-4-Scout-17B-16E-Instruct": "unsloth/Llama-4-Scout-17B-16E-Instruct",
		"CohereForAI/c4ai-command-r-v01":            "mlx-community/aya-23-8B-8bit",
	}
}

// doWithBackoff retries a rate-limited or transiently failed fetch.
//
// HuggingFace answers 429 freely when many repos are read back to back;
// retrying separates "we could not ask" from "the answer is no", so a transport
// hiccup does not become an unresolved (and so refused) name.
//
// The request body is nil on every caller here, so the request is replayable as
// is. Retry-After is honoured when the server sends a sane one; the ceiling
// stops one throttled repo from eating the whole run.
func doWithBackoff(cl *http.Client, req *http.Request) (*http.Response, error) {
	backoff := []time.Duration{2 * time.Second, 5 * time.Second, 12 * time.Second}
	var resp *http.Response
	var err error
	for attempt := 0; ; attempt++ {
		resp, err = cl.Do(req)
		retryable := err != nil
		if err == nil {
			switch resp.StatusCode {
			case 429, 500, 502, 503, 504:
				retryable = true
			}
		}
		if !retryable || attempt >= len(backoff) {
			return resp, err
		}
		wait := backoff[attempt]
		if err == nil {
			if d, ok := retryAfter(resp.Header.Get("Retry-After")); ok && d > wait && d <= 30*time.Second {
				wait = d
			}
			// The body must be drained and closed or the connection is not reused.
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			fmt.Fprintf(os.Stderr, "  http %d on %s, retrying in %v\n", resp.StatusCode, req.URL.Path, wait)
		} else {
			fmt.Fprintf(os.Stderr, "  %v, retrying in %v\n", err, wait)
		}
		time.Sleep(wait)
	}
}

// retryAfter reads the delta-seconds form of Retry-After. The HTTP-date form is
// deliberately not read: a clock-skewed date is a silent multi-hour sleep, and
// the caller already has a bounded default.
func retryAfter(h string) (time.Duration, bool) {
	if h == "" {
		return 0, false
	}
	n, err := strconv.Atoi(strings.TrimSpace(h))
	if err != nil || n < 0 {
		return 0, false
	}
	return time.Duration(n) * time.Second, true
}
