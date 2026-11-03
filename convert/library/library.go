// Package library is the list of models jitllm can fetch by name: for each, a
// Hugging Face repository, the file in it, and the vision tower beside it when
// there is one.
//
// An entry is a claim that the file converts and runs: it gets in by being
// downloaded, converted and made to generate on this engine
// (scripts/library-gate.sh records it in Verified), and it is pinned to the
// repository commit that was checked.
//
// It is a Go slice, not a data file: the CLI and the desktop app both import
// it and a change to it is a reviewed change to code. Sizes are what the Hub
// reports, so a disk check and a will-it-fit estimate need no request.
package library

import (
	"strings"

	"github.com/jitllm/jitllm/convert/hf"
)

// Model is one fetchable model.
type Model struct {
	// Name is what a person types: `jitllm convert qwen3-8b`.
	Name string
	// Title is what a list shows.
	Title string
	// Arch is the converter's architecture name (convert's list).
	Arch string
	// Params is the size people quote, "8B" or "30B-A3B".
	Params string
	// Quant is the file's weight format, as its publisher names it.
	Quant string

	Repo string // "owner/name"
	Rev  string // the commit the entry was verified at
	File string // the model's GGUF

	// MMProj is the vision tower's GGUF, "" for a text model. MMProjRepo and
	// MMProjRev name its repository when it is not Repo.
	MMProj, MMProjRepo, MMProjRev string

	// Bytes is File plus MMProj as the Hub reports them.
	Bytes int64

	// Note is one line on what it is for, or what to know before fetching.
	Note string

	// Verified is when and where the entry was last downloaded, converted and
	// made to generate: "<date> <os/arch> jlm v<container version>".
	Verified string
}

// Vision reports whether the model sees images.
func (m Model) Vision() bool { return m.MMProj != "" }

// Ref is the model file's Hugging Face reference.
func (m Model) Ref() hf.Ref { return hf.Ref{Repo: m.Repo, Rev: m.rev(), File: m.File} }

// TowerRef is the vision tower's reference; ok is false for a text model.
func (m Model) TowerRef() (r hf.Ref, ok bool) {
	if m.MMProj == "" {
		return hf.Ref{}, false
	}
	repo := m.MMProjRepo
	if repo == "" {
		repo = m.Repo
	}
	rev := m.rev()
	if m.MMProjRepo != "" {
		rev = m.MMProjRev
		if rev == "" {
			rev = "main"
		}
	}
	return hf.Ref{Repo: repo, Rev: rev, File: m.MMProj}, true
}

func (m Model) rev() string {
	if m.Rev == "" {
		return "main"
	}
	return m.Rev
}

// Find returns the model with that name, ignoring case.
func Find(name string) (Model, bool) {
	for _, m := range Models() {
		if strings.EqualFold(m.Name, name) {
			return m, true
		}
	}
	return Model{}, false
}
