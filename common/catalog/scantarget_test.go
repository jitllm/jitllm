package catalog

import (
	"encoding/binary"
	"github.com/jitllm/jitllm/format/jlm"
	"os"
	"path/filepath"
	"testing"
)

// A container that exists is the source's target wherever it sits, so a model
// converted in another directory is not offered for conversion again.
func TestScanFindsAContainerInAnotherModelDir(t *testing.T) {
	d1, d2 := t.TempDir(), t.TempDir()
	src := filepath.Join(d2, "tiny.gguf")
	if err := os.WriteFile(src, []byte("GGUF\x03\x00\x00\x00"), 0o600); err != nil {
		t.Fatal(err)
	}
	// A container beside it, with this build's version word.
	hdr := make([]byte, 16)
	copy(hdr, jlm.Magic)
	binary.LittleEndian.PutUint32(hdr[8:], CurrentVersion)
	if err := os.WriteFile(filepath.Join(d2, "tiny.jlm"), hdr, 0o600); err != nil {
		t.Fatal(err)
	}
	for _, e := range Scan([]string{d1, d2}) {
		if e.Kind != KindGGUF {
			continue
		}
		if e.TargetVersion != CurrentVersion {
			t.Fatalf("%s: TargetVersion %d, want %d -- the container beside it was not found "+
				"(target %q)", e.Name, e.TargetVersion, CurrentVersion, e.Target)
		}
		if e.Target != filepath.Join(d2, "tiny.jlm") {
			t.Errorf("target %q, want the existing container in the second dir", e.Target)
		}
	}
}
