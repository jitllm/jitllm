package library

import (
	"bytes"
	"embed"
	"encoding/json"
)

//go:embed models.json
var catalogueFS embed.FS

// entry is one line of models.json: a Model, with its tower's bytes apart
// from the model file's, and a comment on how the entry was checked.
type entry struct {
	Name        string `json:"name"`
	Title       string `json:"title"`
	Arch        string `json:"arch"`
	Params      string `json:"params"`
	Quant       string `json:"quant"`
	Repo        string `json:"repo"`
	Rev         string `json:"rev"`
	File        string `json:"file"`
	MMProj      string `json:"mmproj"`
	MMProjRepo  string `json:"mmproj_repo"`
	MMProjRev   string `json:"mmproj_rev"`
	Bytes       int64  `json:"bytes"`
	MMProjBytes int64  `json:"mmproj_bytes"`
	Note        string `json:"note"`
	Verified    string `json:"verified"`
	Comment     string `json:"comment"`
}

// Models is the list in models.json, smallest first within each family.
// Sizes are the Hub's.
//
// Sized for desktop-class machines: 31 GiB of RAM with a 4 GB card, and a
// 16 GB arm64 host. The last few do not fit either in memory and run
// paged off the disk; their Note says so. A file that does not read is an
// empty list, which TestTheCatalogueReads fails on.
func Models() []Model {
	ms, err := catalogue()
	if err != nil {
		return nil
	}
	return ms
}

func catalogue() ([]Model, error) {
	b, err := catalogueFS.ReadFile("models.json")
	if err != nil {
		return nil, err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	var es []entry
	if err := d.Decode(&es); err != nil {
		return nil, err
	}
	out := make([]Model, len(es))
	for i, e := range es {
		out[i] = Model{Name: e.Name, Title: e.Title, Arch: e.Arch, Params: e.Params, Quant: e.Quant,
			Repo: e.Repo, Rev: e.Rev, File: e.File, MMProj: e.MMProj, MMProjRepo: e.MMProjRepo, MMProjRev: e.MMProjRev,
			Bytes: e.Bytes + e.MMProjBytes, Note: e.Note, Verified: e.Verified}
	}
	return out, nil
}
