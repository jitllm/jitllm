// Package discover is the model library a front end offers: the models
// jitllm fetches by name (package library), ranked for this machine, and the
// download that takes one from the Hub to a container in the model folder.
// Every library row is pinned to a commit that scripts/library-gate.sh
// downloaded, converted and ran.
package discover

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/jitllm/jitllm/common/convertjob"
	"github.com/jitllm/jitllm/common/session"
	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/hf"
	"github.com/jitllm/jitllm/convert/library"
)

// DefaultBalance is the preference a first look starts at: Balanced.
const DefaultBalance = 2

// Dir is where downloads land: the first configured model folder, else def.
func Dir(modelDirs []string, def string) string {
	if len(modelDirs) > 0 {
		return modelDirs[0]
	}
	return def
}

// ContainerFor is where a library model's container is written in dir.
func ContainerFor(dir string, m library.Model) string {
	return filepath.Join(dir, strings.TrimSuffix(m.File, ".gguf")+".jlm")
}

// OnDisk is, for each library model, whether a current container for it is
// in dir.
func OnDisk(dir string) []bool {
	have := make([]bool, len(library.Models()))
	for i, m := range library.Models() {
		have[i] = convertjob.LoadableAt(ContainerFor(dir, m))
	}
	return have
}

// Need is the disk a download of m takes at its peak when it is not streamed
// (see [Get]): the GGUF and the container coexist for a moment, with a tenth
// to spare.
func Need(m library.Model) int64 { return m.Bytes*2 + m.Bytes/10 }

// FreeFunc is how much disk is free on a folder's filesystem, false when
// that cannot be told; [FreeBytes] is the real one.
type FreeFunc func(dir string) (int64, bool)

// SpaceNote is why m cannot download into dir, or "" when it can. The disk is
// checked before a byte moves.
func SpaceNote(m library.Model, dir string, free FreeFunc) string {
	if free == nil {
		free = FreeBytes
	}
	need := Need(m)
	if !hasSource(m, dir) && gguf.Streamable() {
		need = Streamed(m) // see Get: no GGUF is stored
	}
	if n, ok := free(dir); ok && n < need {
		return fmt.Sprintf("Not enough disk space in %s: this needs about %s and %s is free.",
			dir, session.Bytes(uint64(need)), session.Bytes(uint64(n)))
	}
	return ""
}

// FetchFunc fetches one file of a repository into dir and returns its path;
// [HubFetch] is the real one.
type FetchFunc func(cl *hf.Client, r hf.Ref, dir string) (string, error)

// HubFetch fetches from the Hugging Face Hub.
func HubFetch(cl *hf.Client, r hf.Ref, dir string) (string, error) { return cl.Fetch(r, dir) }

// Fetched is what a download put in the folder.
type Fetched struct {
	// Paths is the model's GGUF, then its vision tower when it has one.
	Paths []string
	// Had says which of Paths were there before the download: those are the
	// user's, kept whatever happens.
	Had []bool
}

// Request is the conversion that turns what was fetched into dst.
func (f Fetched) Request(dst string) convertjob.Request {
	req := convertjob.Request{Src: f.Paths[0], Dst: dst}
	if len(f.Paths) > 1 {
		req.MMProj = f.Paths[1]
	}
	return req
}

// Cleanup removes the files the download brought, once the container is
// written: the container is self-sufficient. Files that were already there
// stay.
func (f Fetched) Cleanup() {
	for i, p := range f.Paths {
		if !f.Had[i] {
			os.Remove(p) // ours, and the container is self-sufficient
		}
	}
}

// Download fetches m and its vision tower into dir. progress, which may be
// nil, hears the share of m's bytes fetched so far, below 1, from the
// downloading goroutine. fetch nil is [HubFetch]; token is the Hub token
// (hf.FindToken), "" for none. An interrupted download resumes on the next
// call: hf keeps the partial file and checks it before appending.
func Download(m library.Model, dir, token string, fetch FetchFunc, progress func(float64)) (Fetched, error) {
	if fetch == nil {
		fetch = HubFetch
	}
	refs := []hf.Ref{m.Ref()}
	if tr, ok := m.TowerRef(); ok {
		refs = append(refs, tr)
	}
	var f Fetched
	for _, r := range refs {
		_, err := os.Stat(filepath.Join(dir, filepath.Base(r.File)))
		f.Had = append(f.Had, err == nil)
	}
	var base int64
	for _, r := range refs {
		cl := &hf.Client{
			Token: token,
			OnProgress: func(done, total int64) {
				if m.Bytes > 0 && progress != nil {
					progress(min(0.99, float64(base+done)/float64(m.Bytes)))
				}
			},
		}
		p, err := fetch(cl, r, dir)
		if err != nil {
			return f, err
		}
		if fi, err := os.Stat(p); err == nil {
			base += fi.Size()
		}
		f.Paths = append(f.Paths, p)
	}
	return f, nil
}

// FailMessage says why m did not download, in words a person can act on.
func FailMessage(m library.Model, err error) string {
	if errors.Is(err, hf.ErrAccess) {
		return m.Title + " is gated or private on Hugging Face. Accept its licence at huggingface.co/" + m.Repo +
			", then log in with huggingface-cli login (or set HF_TOKEN) and press Download again."
	}
	return m.Title + " was not downloaded: " + err.Error() + ". Press Download to try again; it resumes where it stopped."
}
