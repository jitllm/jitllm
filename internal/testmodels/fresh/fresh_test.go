package fresh

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestAnOldContainerIsReconverted: a container stamped with another container
// version, as an earlier build leaves one, opens after Container when its GGUF
// is beside it, and is left as it is when none is.
func TestAnOldContainerIsReconverted(t *testing.T) {
	src := testmodels.Path("stories260K.gguf")
	if _, err := os.Stat(src); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v -- this gate proved nothing", err)
	}
	old := func(dir string) string {
		t.Helper()
		dst := filepath.Join(dir, "stories260K"+jlm.Ext)
		if _, err := convert.FromGGUFs(src, "", dst, jlm.Fingerprint{Host: "test"}); err != nil {
			t.Fatal(err)
		}
		f, err := os.OpenFile(dst, os.O_RDWR, 0)
		if err != nil {
			t.Fatal(err)
		}
		var v [4]byte
		binary.LittleEndian.PutUint32(v[:], jlm.Version-1)
		if _, err := f.WriteAt(v[:], 8); err != nil {
			t.Fatal(err)
		}
		f.Close()
		if c, err := jlm.Open(dst); err == nil {
			c.Close()
			t.Fatal("a container stamped with the previous version opened: the gate cannot see a reconversion")
		}
		return dst
	}

	with := t.TempDir()
	dst := old(with)
	if err := os.Symlink(src, filepath.Join(with, "stories260K.gguf")); err != nil {
		t.Fatal(err)
	}
	if err := Container(dst); err != nil {
		t.Fatal(err)
	}
	c, err := jlm.Open(dst)
	if err != nil {
		t.Fatalf("after Container the container still does not open: %v", err)
	}
	c.Close()

	alone := old(t.TempDir())
	if err := Container(alone); err != nil {
		t.Fatalf("a container with no source beside it: %v, want it left for the load to report", err)
	}
	if c, err := jlm.Open(alone); err == nil {
		c.Close()
		t.Fatal("a container with no source beside it changed")
	}
}
