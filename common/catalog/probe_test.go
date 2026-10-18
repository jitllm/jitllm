package catalog

import (
	"encoding/binary"
	"github.com/jitllm/jitllm/format/jlm"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

func TestReadVersionRefusesATruncatedFile(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "short.jlm")
	if err := os.WriteFile(p, []byte("JITLLM"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadVersion(p); err == nil {
		t.Fatal("a six-byte file must not read as a container")
	}
}

func TestReadVersionRefusesAWrongMagic(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "gguf-in-disguise.jlm")
	if err := os.WriteFile(p, []byte("GGUF\x00\x00\x00\x00\x15\x00\x00\x00"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadVersion(p); err == nil {
		t.Fatal("a wrong magic must be refused, not read as version 21")
	}
}

func TestReadVersionReadsTheStoredNumber(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "v19.jlm")
	// Magic, then a little-endian 19: the shape of a stale container.
	b := append([]byte{'J', 'I', 'T', 'L', 'L', 'M', 0, 0}, 19, 0, 0, 0)
	if err := os.WriteFile(p, b, 0o600); err != nil {
		t.Fatal(err)
	}
	v, err := ReadVersion(p)
	if err != nil {
		t.Fatal(err)
	}
	if v != 19 {
		t.Fatalf("ReadVersion = %d, want 19", v)
	}
}

func TestProbeMarksAStaleContainerWithoutOpeningIt(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "v19.jlm")
	b := append([]byte{'J', 'I', 'T', 'L', 'L', 'M', 0, 0}, 19, 0, 0, 0)
	if err := os.WriteFile(p, b, 0o600); err != nil {
		t.Fatal(err)
	}
	e := Entry{Path: p, Name: "v19.jlm", Kind: KindContainer}
	Probe(&e)
	if !e.Probed {
		t.Fatal("Probe must record that it ran")
	}
	if !e.Stale {
		t.Fatalf("v19 against v%d must be stale", CurrentVersion)
	}
	if e.ProbeErr == "" {
		t.Fatal("a stale container needs a reason a person can read")
	}
}

func TestKindOf(t *testing.T) {
	cases := map[string]Kind{
		"model.jlm":         KindContainer,
		"MODEL.JLM":         KindContainer,
		"model.gguf":        KindGGUF,
		"model.safetensors": KindSafetensors,
		"notes.md":          KindUnknown,
		"model.jlm.part":    KindUnknown,
	}
	for name, want := range cases {
		if got := KindOf(name); got != want {
			t.Errorf("KindOf(%q) = %v, want %v", name, got, want)
		}
	}
}

func TestScanSkipsAMissingDirectory(t *testing.T) {
	got := Scan([]string{filepath.Join(t.TempDir(), "nope")})
	if len(got) != 0 {
		t.Fatalf("a missing model directory must be skipped, got %d entries", len(got))
	}
}

func TestCacheKeyIsDerivedFromTheFileIdentity(t *testing.T) {
	a := Entry{Path: "/models/llama.jlm", Size: 1234}
	b := Entry{Path: "/other/llama.jlm", Size: 1234}
	if CacheKey(a) != CacheKey(b) {
		t.Error("the same file copied elsewhere shares a cache namespace")
	}
	c := Entry{Path: "/models/llama.jlm", Size: 999}
	if CacheKey(a) == CacheKey(c) {
		t.Error("two different files must never share a cache namespace")
	}
}

// A stale container's tower must be readable without opening it.
//
// Without it, one-click Reconvert rebuilt a vision model text-only.
func TestHasTowerReadsAStaleContainer(t *testing.T) {
	cases := []struct {
		path  string
		tower bool
	}{
		{testmodels.Path("qwen2vl-vlm.jlm"), true},
		{testmodels.Path("gemma-2b-v13.jlm"), false},
		{testmodels.Path("SmolVLM-256M-Instruct-Q8_0-vlm.jlm"), true},
		{testmodels.Path("stories260K.jlm"), false},
	}
	for _, c := range cases {
		if _, err := os.Stat(c.path); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proves nothing without it", err)
		}
		tower, known, err := HasTower(c.path)
		if err != nil || !known {
			t.Fatalf("%s: known=%v err=%v", c.path, known, err)
		}
		if tower != c.tower {
			t.Errorf("%s: tower=%v, want %v", c.path, tower, c.tower)
		}
	}
}

// A source row says which container it converts to and whether that container
// is there and current, so the Convert tab can tell "converted" from "not yet"
// from "outdated" without anyone opening the Models tab.
func TestScanTiesASourceToItsContainer(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []string{"a.gguf", "b.gguf", "c.gguf", "mmproj-c.gguf"} {
		if err := os.WriteFile(filepath.Join(dir, n), []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeVersion := func(name string, v uint32) {
		b := make([]byte, 12)
		copy(b, jlm.Magic)
		binary.LittleEndian.PutUint32(b[8:], v)
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeVersion("b.jlm", CurrentVersion)
	writeVersion("c.jlm", CurrentVersion-1)

	got := map[string]Entry{}
	for _, e := range Scan([]string{dir}) {
		got[e.Name] = e
	}
	for name, want := range map[string]uint32{"a.gguf": 0, "b.gguf": CurrentVersion, "c.gguf": CurrentVersion - 1} {
		e := got[name]
		if e.Target != filepath.Join(dir, strings.TrimSuffix(name, ".gguf")+".jlm") || e.TargetVersion != want {
			t.Errorf("%s: target %q v%d, want v%d", name, e.Target, e.TargetVersion, want)
		}
		if e.Tower {
			t.Errorf("%s is marked a vision tower", name)
		}
	}
	if !got["mmproj-c.gguf"].Tower {
		t.Error("mmproj-c.gguf is not marked a vision tower")
	}
	if got["b.jlm"].Target != "" {
		t.Error("a container row carries a conversion target")
	}
}

// A symlinked model is listed at its target's size, not the link's.
func TestScanFollowsASymlinkForTheSize(t *testing.T) {
	dir := t.TempDir()
	real := filepath.Join(t.TempDir(), "real.gguf")
	if err := os.WriteFile(real, make([]byte, 4096), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(real, filepath.Join(dir, "linked.gguf")); err != nil {
		t.Skipf("no symlinks here: %v", err)
	}
	es := Scan([]string{dir})
	if len(es) != 1 || es[0].Size != 4096 {
		t.Fatalf("scan listed %+v, want linked.gguf at 4096 bytes", es)
	}
}
