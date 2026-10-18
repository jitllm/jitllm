package catalog

import (
	"encoding/binary"
	"fmt"
	"os"

	"github.com/jitllm/jitllm/format/jlm"
)

// ReadVersion reads a container's format version from its first twelve bytes.
//
// Staleness is read structurally, not matched out of jlm.Open's error
// message: the magic and version are the file's first twelve bytes.
func ReadVersion(path string) (uint32, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	var b [12]byte
	if _, err := f.ReadAt(b[:], 0); err != nil {
		return 0, fmt.Errorf("catalog: %s is too short to be a container: %w", path, err)
	}
	if string(b[0:8]) != jlm.Magic {
		return 0, fmt.Errorf("catalog: %s is not a jlm container (magic %q)", path, b[0:8])
	}
	return binary.LittleEndian.Uint32(b[8:12]), nil
}

// towerSince is the first container version whose header carries the vision
// fields where this build reads them: VisLen at 128 and NVisBlocks at 152, since
// the tower moved into the container (v12) and unchanged through v21.
const towerSince = 12

// HasTower reports whether the container at path carries a vision tower,
// reading two header fields rather than opening it -- so it answers for a
// stale container too, which jlm.Open refuses. known is false for a file from
// before towerSince. One-click reconvert uses it so a vision model does not
// come back text-only.
func HasTower(path string) (tower, known bool, err error) {
	v, err := ReadVersion(path)
	if err != nil || v < towerSince {
		return false, false, err
	}
	f, err := os.Open(path)
	if err != nil {
		return false, false, err
	}
	defer f.Close()
	var b [156]byte
	if _, err := f.ReadAt(b[:], 0); err != nil {
		return false, false, err
	}
	visLen := binary.LittleEndian.Uint64(b[128:136])
	nVis := binary.LittleEndian.Uint32(b[152:156])
	return visLen != 0 || nVis != 0, true, nil
}

// Probe fills a container entry's description.
//
// It is cheap: jlm.Open reads the header, config, vocabulary and tensor table
// and no pages, so a whole directory can be probed. A failure is recorded in
// the entry rather than returned, since a stale container is exactly what the
// row has to show.
func Probe(e *Entry) {
	e.Probed = true
	e.ProbeErr = ""

	if e.Kind != KindContainer {
		return
	}

	v, err := ReadVersion(e.Path)
	if err != nil {
		e.ProbeErr = err.Error()
		return
	}
	e.Version = v
	e.Stale = v != CurrentVersion
	if e.Stale {
		e.ProbeErr = fmt.Sprintf("container is v%d, this build reads v%d -- reconvert", e.Version, CurrentVersion)
		return
	}

	f, err := jlm.Open(e.Path)
	if err != nil {
		e.ProbeErr = err.Error()
		return
	}
	defer f.Close()

	h := f.H
	e.PageSize, e.NBlocks = h.PageSize, h.NBlocks
	e.VisPageSize, e.VisionBlocks = h.VisPageSize, h.NVisBlocks
	e.DenseBytes = h.DenseLen
	e.ChunkBytes = f.ChunkBytes()
	e.StreamGroups = uint32(f.StreamGroups())
	e.Writer = f.FP.Writer

	if c := f.Config(); c != nil {
		e.Arch = c.Arch.String()
		e.NLayer, e.NEmbd, e.NCtx, e.NVocab = c.NLayer, c.NEmbd, c.NCtx, c.NVocab
		e.NExpert, e.NExpertUsed = c.NExpert, c.NExpertUsed
	}
	e.HasVision = f.Vision() != nil
	if e.Quant == "" {
		e.Quant = QuantFromTensors(f.Entries())
	}
	if v := f.Vocab(); v != nil {
		e.HasChat = len(v.Templates) > 0
	}
}

// MoE reports whether the entry is a mixture of experts.
func (e Entry) MoE() bool { return e.NExpert > 1 }
