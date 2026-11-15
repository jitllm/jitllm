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

// TestACausalPictureBesideBidirectionalLayersIsReconverted: a merged vision
// container whose text model states the layers a picture's rows see each
// other on and whose tower states its pictures causal, as the converter wrote
// Gemma 4's before it stated the span, is reconverted from its sources.
func TestACausalPictureBesideBidirectionalLayersIsReconverted(t *testing.T) {
	dir := t.TempDir()
	srcs := []string{testmodels.Path("synth-gemma4v.gguf"), testmodels.Path("mmproj-synth-gemma4v.gguf")}
	for _, p := range srcs {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v -- this gate proved nothing", err)
		}
		if err := os.Symlink(p, filepath.Join(dir, filepath.Base(p))); err != nil {
			t.Fatal(err)
		}
	}
	dst := filepath.Join(dir, "synth-gemma4v-vlm"+jlm.Ext)
	if _, err := convert.FromGGUFs(srcs[0], srcs[1], dst, jlm.Fingerprint{Host: "test"}); err != nil {
		t.Fatal(err)
	}
	if changed, err := jlm.Restate(dst, func(f *jlm.File, c *jlm.Config) (bool, error) {
		v := f.Vision()
		if len(c.BidirLayers) == 0 || v == nil || v.Tower == nil || !v.Tower.Span.Bidir {
			t.Fatalf("the fixture states %d bidirectional layers and no bidirectional span: it tests nothing", len(c.BidirLayers))
		}
		v.Tower.Span.Bidir = false
		return true, nil
	}); err != nil || !changed {
		t.Fatalf("stating the pictures causal: changed %v, %v", changed, err)
	}
	bidir := func() bool {
		t.Helper()
		c, err := jlm.Open(dst)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		return c.Vision().Tower.Span.Bidir
	}
	if bidir() {
		t.Fatal("the restated container states its pictures bidirectional: the gate cannot see a reconversion")
	}
	if err := Container(dst); err != nil {
		t.Fatal(err)
	}
	if !bidir() {
		t.Fatal("a container stating its pictures causal beside bidirectional layers was left as it is")
	}
}

// TestUnstatedHeadBlocksAreReconverted: a marker head's container whose
// statement leaves its head blocks out, as the converter wrote one before it
// stated them, is reconverted from its source.
func TestUnstatedHeadBlocksAreReconverted(t *testing.T) {
	dir := t.TempDir()
	src := testmodels.Path("laya/fixture/laya-fixture.gguf")
	if _, err := os.Stat(src); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v -- this gate proved nothing", err)
	}
	if err := os.Symlink(src, filepath.Join(dir, filepath.Base(src))); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(dir, "laya-fixture"+jlm.Ext)
	if _, err := convert.FromGGUFs(src, "", dst, jlm.Fingerprint{Host: "test"}); err != nil {
		t.Fatal(err)
	}
	stated := func() int {
		t.Helper()
		c, err := jlm.Open(dst)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		return len(c.Config().Blocks.Of)
	}
	whole := stated()
	if changed, err := jlm.Restate(dst, func(_ *jlm.File, c *jlm.Config) (bool, error) {
		if c.HeadBlocks() == 0 || whole != int(c.NLayer+c.HeadBlocks()) {
			t.Fatalf("the fixture states %d graphs for %d blocks and %d head blocks: it tests nothing", whole, c.NLayer, c.HeadBlocks())
		}
		var short jlm.BlockGraphs
		at := map[uint16]uint16{}
		for _, i := range c.Blocks.Of[:c.NLayer] {
			j, ok := at[i]
			if !ok {
				j = uint16(len(short.Graphs))
				at[i] = j
				short.Graphs = append(short.Graphs, c.Blocks.Graphs[i])
			}
			short.Of = append(short.Of, j)
		}
		c.Blocks = short
		return true, nil
	}); err != nil || !changed {
		t.Fatalf("leaving the head blocks out: changed %v, %v", changed, err)
	}
	if stated() == whole {
		t.Fatal("the restated container states its head blocks: the gate cannot see a reconversion")
	}
	if err := Container(dst); err != nil {
		t.Fatal(err)
	}
	if n := stated(); n != whole {
		t.Fatalf("a container leaving its head blocks out was left as it is: %d graphs, want %d", n, whole)
	}
}
