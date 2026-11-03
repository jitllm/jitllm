package main

// The directory: which HF repo defines each tokenizer.ggml.pre name. The
// tokenizer.json fetch itself is pipeline.go and the parse is package pretok.
//
// This is all llama.cpp is used for: convert_hf_to_gguf_update.py's list of
// which repo defines each name, a checkable pointer rather than a claim. The
// pipeline itself comes from the model (AGENTS.md RULE 7m).

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"slices"
	"strings"
	"time"
)

func repoLine() *regexp.Regexp {
	return regexp.MustCompile(`\{"name":\s*"([^"]+)",\s*"tokt":\s*TOKENIZER_TYPE\.(\w+),\s*"repo":\s*"https://huggingface\.co/([^"]+)"`)
}

// Latest, not pinned: the directory's value is currency, since new models mint
// new names. The generated file records the resolved SHA and is checked in, so
// a regeneration is a reviewable diff.

// fetchDirectory pulls the name -> HF repo map from llama.cpp at lcppRev, so the
// generator needs no local checkout of anything.
func fetchDirectory() (map[string][]string, string, error) {
	rev, err := resolveHead()
	if err != nil {
		return nil, "", err
	}
	url := "https://raw.githubusercontent.com/ggml-org/llama.cpp/" + rev +
		"/convert_hf_to_gguf_update.py"
	cl := &http.Client{Timeout: 45 * time.Second}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, "", err
	}
	resp, err := doWithBackoff(cl, req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, "", fmt.Errorf("http %d fetching the directory", resp.StatusCode)
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", err
	}
	m := hfRepos(string(b))
	if len(m) == 0 {
		return nil, "", fmt.Errorf("the directory parsed to zero entries; the script's shape changed")
	}
	return m, rev, nil
}

// resolveHead asks GitHub what master is, so the generated file can
// name the exact commit it was produced from.
func resolveHead() (string, error) {
	cl := &http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequest("GET", "https://api.github.com/repos/ggml-org/llama.cpp/commits/master", nil)
	if err != nil {
		return "", err
	}
	resp, err := doWithBackoff(cl, req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("http %d resolving llama.cpp master", resp.StatusCode)
	}
	var out struct {
		SHA string `json:"sha"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if len(out.SHA) < 40 {
		return "", fmt.Errorf("bad sha %q", out.SHA)
	}
	return out.SHA, nil
}

// hfRepos maps pre name -> HF repos, from llama.cpp's update script. A name
// can appear more than once: the first entry is the model that defines it, and
// later ones are chkhsh aliases for models llama.cpp found to tokenize the same
// way. Only BPE has a regex pre-tokenizer.
func hfRepos(updateScript string) map[string][]string {
	out := map[string][]string{}
	for _, m := range repoLine().FindAllStringSubmatch(updateScript, -1) {
		r := strings.TrimSuffix(m[3], "/")
		if m[2] == "BPE" && !slices.Contains(out[m[1]], r) {
			out[m[1]] = append(out[m[1]], r)
		}
	}
	return out
}

// ErrGated is a repo HuggingFace will not serve without credentials, and
// ErrMissing one it does not have at all. Both answer 401 anonymously; the body
// tells them apart.
const (
	ErrGated   fetchError = "gated repo: needs HF_TOKEN"
	ErrMissing fetchError = "repo not found on HuggingFace"
)

type fetchError string

func (e fetchError) Error() string { return string(e) }
