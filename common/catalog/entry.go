// Package catalog enumerates the model files on disk and reads what a
// container says about itself.
//
// It is a leaf of the UI: it imports jitllm's jlm reader and no UI package,
// so a probe is testable without a window.
package catalog

import (
	"time"

	"github.com/jitllm/jitllm/format/jlm"
)

// Kind classifies a file the catalog found.
type Kind uint8

// The kinds a catalog entry can have.
const (
	KindUnknown Kind = iota
	// KindContainer is a .jlm container, the one weight format this engine
	// runs.
	KindContainer
	// KindGGUF is the converter's input, not a runnable model.
	KindGGUF
	// KindSafetensors is the other converter input.
	KindSafetensors
)

func (k Kind) String() string {
	switch k {
	case KindContainer:
		return "container"
	case KindGGUF:
		return "gguf"
	case KindSafetensors:
		return "safetensors"
	}
	return "unknown"
}

// Entry is one file in a configured model directory.
//
// The probed fields are filled only for a container, by [Probe]. A GGUF row
// carries its path, its size and what [Scan] learns about the container it
// converts to, and nothing about the model inside it: this app reads
// containers, and a GGUF is something to convert.
type Entry struct {
	Path  string
	Name  string
	Kind  Kind
	Size  int64
	MTime time.Time

	// For a source (GGUF or safetensors): Target is the container it converts
	// to, and TargetVersion that container's version word -- zero when it does
	// not exist or cannot be read. Tower is a vision tower by its name
	// ("mmproj"), which goes in the converter's second field, not its first.
	Target        string
	TargetVersion uint32
	Tower         bool

	// Probed is true once [Probe] has run on this entry, whether or not it
	// succeeded. A zero Arch with Probed false means "not looked at yet"; with
	// Probed true and ProbeErr set it means "looked at and refused".
	Probed   bool
	ProbeErr string

	// Version is the container version in the file, read from its first twelve
	// bytes rather than parsed out of an error string. Stale reports whether it
	// differs from the jlm.Version this build writes.
	Version uint32
	Stale   bool

	// Quant is the weights' quantization: the file name's own label when it
	// has one, otherwise the container's dominant tensor type.
	Quant string

	Arch        string
	NLayer      uint32
	NEmbd       uint32
	NCtx        uint32
	NVocab      uint32
	NExpert     uint32
	NExpertUsed uint32

	HasVision    bool
	VisionBlocks uint32
	HasChat      bool

	PageSize     uint64
	VisPageSize  uint64
	NBlocks      uint32
	DenseBytes   uint64
	ChunkBytes   uint64
	StreamGroups uint32
	Writer       string
}

// CurrentVersion is the container version this build writes. An entry whose
// Version this build does not read (Readable) has to be reconverted.
const CurrentVersion = jlm.Version

// Readable reports whether this build opens a container of version v
// (jlm.ReadsVersion): the current one, or the previous one without the one
// section that changed, which jlm.Open checks.
func Readable(v uint32) bool { return jlm.ReadsVersion(v) }
