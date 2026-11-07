package engine

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
)

// jlmOf converts a GGUF to a container beside it and returns that path; the
// engine reads only containers, so the suite converts to run what ships. A
// failed conversion fails rather than skips. A path that is already a container
// passes straight through (see jlmOfPair).
func jlmOf(t testing.TB, path string) string { return jlmOfPair(t, path, "") }

// q8Suffix names a safetensors fixture converted with its matrices quantized
// to Q8_0 (convert.WithQ8): "synth-kimilinear@q8" is synth-kimilinear's
// transformers weights, every parameter and buffer random (scripts/hfgold.py),
// in the container's Q8_0, so a gate holding int8 activations to a float
// reference measures a distance on it.
const q8Suffix = "@q8"

// fixtureSource is the file or directory a model name reads from.
func fixtureSource(path string) string { return strings.TrimSuffix(path, q8Suffix) }

// BuildIDForTest identifies this test binary by its bytes; it is what
// jlm.Fingerprint.Writer carries for a test-converted file. The mtime is not an
// identity (`go test` relinks every run), the content is. An unreadable
// executable yields "", which forces a reconversion, the safe direction. It is
// exported only so model_test's jlmOf shares one identity with jlmOfPair.
// Hashing a test binary takes long enough to matter once per call, and the
// package holds no state to remember it in, so the hash is kept in a file named
// for the binary's path, size and mtime.
func BuildIDForTest() string {
	p, err := os.Executable()
	if err != nil {
		return ""
	}
	fi, err := os.Stat(p)
	if err != nil {
		return ""
	}
	key := sha256.Sum256([]byte(fmt.Sprintf("%s %d %d", p, fi.Size(), fi.ModTime().UnixNano())))
	memo := filepath.Join(os.TempDir(), "jitllm-buildid-"+hex.EncodeToString(key[:8]))
	if b, err := os.ReadFile(memo); err == nil && len(b) > 0 {
		return string(b)
	}
	f, err := os.Open(p)
	if err != nil {
		return ""
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return ""
	}
	id := "test " + hex.EncodeToString(h.Sum(nil))[:16]
	if tmp, err := os.CreateTemp(os.TempDir(), "jitllm-buildid-*"); err == nil {
		_, werr := tmp.WriteString(id)
		tmp.Close()
		if werr != nil || os.Rename(tmp.Name(), memo) != nil {
			os.Remove(tmp.Name())
		}
	}
	return id
}

// sourcePresent reports whether every input a conversion needs is on disk.
func sourcePresent(paths ...string) bool {
	for _, p := range paths {
		if p == "" {
			continue
		}
		if _, err := os.Stat(p); err != nil {
			return false
		}
	}
	return true
}

// containerAlone serves a container whose source is gone.
//
// A .jlm carries everything, so once converted the source may be deleted (a host
// may hold no GGUF). A missing source costs only the freshness check; a
// container this build cannot open is still refused.
func containerAlone(dst, src string) (string, error) {
	c, err := jlm.Open(dst)
	if err != nil {
		return "", &missingModel{src: src, dst: dst, err: err}
	}
	c.Close()
	return dst, nil
}

// missingModel is a source that is gone with no readable container beside it.
type missingModel struct {
	src, dst string
	err      error
}

func (e *missingModel) Error() string {
	return fmt.Sprintf("MODEL MISSING: %s, and its container %s does not open (%v) (set JITLLM_MODELS to the model directory)",
		e.src, e.dst, e.err)
}

// sweepJLM is jlmOf for a sweep over files: each converts in a subtest of its
// own, so a file that is part of a model (a head meant for a trunk) or one
// refused by scope skips there and the sweep goes on to the next. false for
// a file that did not convert.
func sweepJLM(t *testing.T, p string) (string, bool) {
	var dst string
	t.Run("convert/"+filepath.Base(p), func(t *testing.T) { dst = jlmOf(t, p) })
	return dst, dst != ""
}

// jlmOfPair converts a text model and, when given one, its vision tower into one
// container. There is no second file on the weight path: see
// convert.FromGGUFs.
func jlmOfPair(t testing.TB, path, mmproj string) string {
	t.Helper()
	dst, err := convertPair(path, mmproj)
	var miss *missingModel
	switch {
	case errors.As(err, &miss):
		t.Skipf("%v -- this gate proved nothing", err)
	case errors.Is(err, convert.ErrPart):
		t.Skipf("PART OF A MODEL: %s -- %v", filepath.Base(path), err)
	case errors.Is(err, convert.ErrNotImplemented):
		// A scope refusal reaches the caller as a skip naming it; a sweep
		// converts in a subtest of its own so the skip stays that model's.
		t.Skipf("NOT IMPLEMENTED: %s -- %v (RULE 7 scope)", filepath.Base(path), err)
	case err != nil:
		t.Fatalf("convert %s: %v", path, err)
	}
	return dst
}

// jlmOfErr is jlmOf for a caller that decides what a failure means: a sweep
// that must not let one model's refusal skip the whole gate.
func jlmOfErr(path string) (string, error) { return convertPair(path, "") }

// convertPair is jlmOfPair without a test: the container's path, or why there
// is none.
func convertPair(path, mmproj string) (string, error) {
	if strings.HasSuffix(path, jlm.Ext) {
		// A container an earlier build left is reconverted from the GGUF
		// beside it; one that opens passes straight through.
		return path, fresh.Container(path)
	}
	// A HuggingFace directory converts through the safetensors path to a
	// container beside its real files, named for the directory; named with
	// q8Suffix, its matrices quantized to Q8_0 (convert.WithQ8).
	dir, q8 := strings.CutSuffix(path, q8Suffix)
	if fi, err := os.Stat(dir); err == nil && fi.IsDir() && mmproj == "" {
		real, err := filepath.EvalSymlinks(dir)
		if err != nil {
			return "", err
		}
		return convertDir(real, q8)
	}
	dst := strings.TrimSuffix(path, filepath.Ext(path)) + jlm.Ext
	if mmproj != "" {
		// A distinct destination: the merged container is not the text-only one
		// and must not be served from its cache.
		dst = strings.TrimSuffix(path, filepath.Ext(path)) + "-vlm" + jlm.Ext
	}
	if !sourcePresent(path, mmproj) {
		return containerAlone(dst, path)
	}
	si, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	// Cached: conversion is deterministic, so a container newer than its
	// source is the one this would produce, provided this build can still
	// read it.
	if mi, err := os.Stat(mmproj); mmproj != "" && err == nil && mi.ModTime().After(si.ModTime()) {
		si = mi // the newer of the two inputs decides staleness
	}
	if writtenByThisBuild(dst, si.ModTime()) {
		return dst, nil
	}
	tmp, err := servedName(dst)
	if err != nil {
		return "", err
	}
	sweepServedCopies(dst)
	if _, err := convert.FromGGUFs(path, mmproj, tmp, jlm.Fingerprint{Host: "test", Writer: BuildIDForTest()}); err != nil {
		os.Remove(tmp)
		return "", err
	}
	return place(tmp, dst)
}

// writtenByThisBuild reports whether dst is a container newer than newest and
// stamped with this binary's identity. The binary is an input too: most of a
// container is decisions the converter makes, so one written by an earlier
// build is stale however new its mtime. jlm.WriterID() cannot tell builds apart
// here (`go test -c` stamps no VCS info), so BuildIDForTest hashes the binary
// and a container whose Writer disagrees is reconverted.
func writtenByThisBuild(dst string, newest time.Time) bool {
	di, err := os.Stat(dst)
	if err != nil || !di.ModTime().After(newest) {
		return false
	}
	c, err := jlm.Open(dst)
	if err != nil {
		return false
	}
	defer c.Close()
	return c.FP.Writer == BuildIDForTest()
}

// place renames a finished conversion onto dst and returns what to read. On
// Windows a file another process has open cannot be replaced, and `go test
// ./...` runs packages in parallel over one model directory, each a different
// binary and so each reconverting; there the fresh container, complete and
// written by this build, is served under its own name instead. A conversion
// goes to a private name and is renamed, so a reader never opens a
// half-written container, and the loser of a race writes a byte-identical
// result.
func place(tmp, dst string) (string, error) {
	err := os.Rename(tmp, dst)
	if err == nil {
		return dst, nil
	}
	if _, statErr := os.Stat(dst); runtime.GOOS == "windows" && statErr == nil {
		return tmp, nil
	}
	os.Remove(tmp)
	return "", err
}

// servedName is a conversion's private name: dst with a unique tag before its
// extension, which stays a container's so the file can be served as it is.
func servedName(dst string) (string, error) {
	f, err := os.CreateTemp(filepath.Dir(dst), strings.TrimSuffix(filepath.Base(dst), jlm.Ext)+".tmp-*"+jlm.Ext)
	if err != nil {
		return "", err
	}
	name := f.Name()
	f.Close()
	if err := os.Remove(name); err != nil {
		return "", err
	}
	return name, nil
}

// sweepServedCopies removes the copies place kept in earlier runs. Windows
// refuses to remove a file that is open, so a copy a running test binary still
// reads survives this; elsewhere place never keeps one.
func sweepServedCopies(dst string) {
	if runtime.GOOS != "windows" {
		return
	}
	old, _ := filepath.Glob(strings.TrimSuffix(dst, jlm.Ext) + ".tmp-*" + jlm.Ext)
	for _, f := range old {
		os.Remove(f)
	}
}

// convertDir is convertPair for a HuggingFace directory: the container is
// <dir>.jlm, reconverted when any file in the directory is newer or another
// build wrote it.
func convertDir(dir string, q8 bool) (string, error) {
	dst := filepath.Clean(dir) + jlm.Ext
	var opts []convert.Option
	if q8 {
		dst = filepath.Clean(dir) + "-q8" + jlm.Ext
		opts = append(opts, convert.WithQ8())
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	var newest time.Time
	for _, e := range ents {
		if fi, err := e.Info(); err == nil && fi.ModTime().After(newest) {
			newest = fi.ModTime()
		}
	}
	if writtenByThisBuild(dst, newest) {
		return dst, nil
	}
	tmp, err := servedName(dst)
	if err != nil {
		return "", err
	}
	sweepServedCopies(dst)
	if _, err := convert.FromSafetensors(dir, tmp, jlm.Fingerprint{Host: "test", Writer: BuildIDForTest()}, opts...); err != nil {
		os.Remove(tmp)
		return "", err
	}
	return place(tmp, dst)
}

// The gates' own options: the package reads no environment, so a knob set with
// t.Setenv would reach nothing. They apply at Open, the only path that reaches a
// instance's JIT.

// benchModel is the model a gate runs against when the caller names none.
//
// JITLLM_BENCH_MODEL is an override, not a switch: unset, correctness gates such
// as TestPrefillMatchesForward run against the smallest quantized language model
// in the model directory rather than skipping. Measurements still need their
// own opt-in.
func benchModel(t testing.TB) string {
	t.Helper()
	if p := os.Getenv("JITLLM_BENCH_MODEL"); p != "" {
		return testmodels.Resolve(p)
	}
	paths := modelFiles()
	best, bestSize := "", int64(0)
	for _, p := range paths {
		base := filepath.Base(p)
		// A projector carries no language model and a float model exercises no
		// packed kernel, so neither answers the question these gates ask.
		if strings.HasPrefix(base, "mmproj") || strings.Contains(base, "f32") ||
			strings.Contains(base, "stories") {
			continue
		}
		fi, err := os.Stat(p)
		if err != nil {
			continue
		}
		if best == "" || fi.Size() < bestSize {
			best, bestSize = p, fi.Size()
		}
	}
	if best == "" {
		testmodels.Missing(t, "%s", "no quantized model under "+testmodels.Dir()+" (set JITLLM_MODELS to the model directory) -- RULE 11: fetch one, do not "+
			"record the absence (set JITLLM_BENCH_MODEL to override the choice)")
	}
	return best
}

// languageModels is the glob every whole-box sweep wants: the GGUFs on this box
// that are language models, with the vision PROJECTORS left out.
//
// A projector is the wrong input, not missing work, so it is filtered here
// rather than reported as a skip. The filter reads general.architecture, as
// Open does, not the file name.
func languageModels(t testing.TB) []string {
	t.Helper()
	paths := testmodels.Glob("*.gguf")
	if len(paths) == 0 {
		// No GGUF is not "no models": a host may hold only containers.
		// The fallback likewise reads the architecture, not the name.
		return containerModels(t)
	}
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		f, err := gguf.Open(p)
		if err != nil {
			continue
		}
		arch, _ := f.KV["general.architecture"].String()
		f.Close()
		if arch == "clip" {
			continue
		}
		out = append(out, p)
	}
	if len(out) == 0 {
		t.Fatal("every GGUF on this box is a projector; this sweep would prove nothing")
	}
	return out
}

// containerModels is languageModels' arm for a box that holds .jlm and no GGUF.
// The paths it returns are containers, which jlmOf passes straight through.
func containerModels(t testing.TB) []string {
	t.Helper()
	paths := testmodels.Glob("*.jlm")
	if len(paths) == 0 {
		testmodels.Missing(t, "%s", "no models under "+testmodels.Dir()+" (set JITLLM_MODELS to the model directory) -- RULE 11: fetch some, do not record the absence")
	}
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		f, err := jlm.Open(p)
		if err != nil {
			// A container this build cannot read is stale; say so rather
			// than drop it silently.
			t.Logf("skipping %s: %v", filepath.Base(p), err)
			continue
		}
		c := f.Config()
		lm := c != nil && c.Arch != jlm.ArchNone && c.NLayer > 0
		f.Close()
		if lm {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		t.Fatal("every container on this box is a projector or unreadable; this sweep would prove nothing")
	}
	return out
}

// existingModel resolves a test's model path to something that is actually on
// this box: the GGUF it names, the container beside it, or for a name with
// q8Suffix the source directory convertPair quantizes.
//
// A hardcoded ".gguf" is a claim about the disk that is false on a host
// holding only containers, where the gate would otherwise skip.
func existingModel(path string) (string, bool) {
	if _, err := os.Stat(path); err == nil {
		return path, true
	}
	if src, ok := strings.CutSuffix(path, q8Suffix); ok {
		if _, err := os.Stat(src); err == nil {
			return path, true
		}
	}
	c := strings.TrimSuffix(path, filepath.Ext(path)) + jlm.Ext
	if _, err := os.Stat(c); err == nil {
		return c, true
	}
	// A HuggingFace directory's container is named for the whole directory
	// (convertDir).
	if _, err := os.Stat(path + jlm.Ext); err == nil {
		return path + jlm.Ext, true
	}
	return "", false
}

// modelFiles is every model this box holds, as a path jlmOf accepts.
//
// GGUF where there is GGUF, containers where there is not; jlmOfPair passes a
// .jlm straight through, so the two differ only in what the glob finds.
func modelFiles() []string {
	if p := testmodels.Glob("*.gguf"); len(p) > 0 {
		return p
	}
	p := testmodels.Glob("*.jlm")
	return p
}

// TestAServedCopyOpens: the copy place serves on Windows, when another test
// binary holds the container and the rename is refused, is a container Open
// reads. Named <dst>.tmp-<tag> it was refused for its extension, and every
// gate that drew it failed with "is not a .jlm container".
func TestAServedCopyOpens(t *testing.T) {
	src := testmodels.Path("stories260K.gguf")
	dst := filepath.Join(t.TempDir(), "stories260K"+jlm.Ext)
	tmp, err := servedName(dst)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := convert.FromGGUFs(src, "", tmp, jlm.Fingerprint{Host: "test", Writer: BuildIDForTest()}); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v", err)
	}
	m, err := model.Open(tmp)
	if err != nil {
		t.Fatalf("the served copy %s does not open: %v", filepath.Base(tmp), err)
	}
	m.Close()
}
