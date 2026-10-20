package model

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/backend/recorder"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

var (
	irOut      = flag.String("ir-out", "", "TestDumpEmittedKernels writes <dir>/<profile>/<model>.ir here")
	irProfiles = flag.String("ir-profiles", "cuda-sm70,cuda-sm75,cuda-sm80,cuda-sm86,cuda-sm89,cuda-sm90,hip-gfx942,hip-gfx1030,hip-gfx1100,hip-gfx1201,vulkan,vulkan-irisxe,vulkan-lavapipe,metal-m4", "the recorder profiles TestDumpEmittedKernels stands in for")
)

// TestDumpEmittedKernels opens every fixture's blocks on a device that runs
// nothing and records what is compiled and launched on it (recorder), for
// each device profile, prefills a prompt chunk and steps once, and writes
// each model's instructions: <profile>/<model>.code, every launched kernel's
// code as lowered for that device, in the order first launched. Run on main
// it records the pre-made kernels' instructions; the composer's are diffed
// against them instruction by instruction.
func TestDumpEmittedKernels(t *testing.T) {
	if *irOut == "" {
		t.Skip("-ir-out names the directory to write to")
	}
	paths, _ := filepath.Glob(filepath.Join(testmodels.Dir(), "synth-*.jlm"))
	for _, pn := range strings.Split(*irProfiles, ",") {
		prof, ok := recorder.Profiles[pn]
		if !ok {
			t.Fatalf("no recorder profile %q", pn)
		}
		dir := filepath.Join(*irOut, pn)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		for _, p := range paths {
			name := strings.TrimSuffix(filepath.Base(p), ".jlm")
			if strings.HasSuffix(name, "-vlm") || strings.Contains(name, "mmproj") {
				continue
			}
			code, err := emittedCode(t, prof, p)
			if err != nil {
				t.Logf("%s on %s: %v", name, pn, err)
				code = "// error: " + strings.ReplaceAll(err.Error(), "\n", " ") + "\n" + code
			}
			if code == "" {
				continue
			}
			if err := os.WriteFile(filepath.Join(dir, name+".code"), []byte(code), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
}

// emittedCode is every kernel launched for the model at path on a recording
// device standing in for prof, each one's code as lowered for it, in the
// order first launched.
func emittedCode(t *testing.T, prof recorder.Profile, path string) (string, error) {
	m, err := Open(path, noTune)
	if err != nil {
		return "", err
	}
	defer m.Close()
	if m.IsEncoder() {
		return "", nil
	}
	rec := recorder.New(prof, 1<<40)
	g, err := tier.New([]tier.Slot{{Dev: rec, Bytes: 1 << 40}}, append(testTierOpts(t),
		tier.WithConfig(func(c *tier.Config) { c.NoGraph = true }))...)
	if err != nil {
		return "", err
	}
	defer g.Close()
	s := m.NewState(32)
	defer s.Close()
	if err := s.SetDevice(g); err != nil {
		return "", err
	}
	if s.GPULayers() == 0 {
		return "", fmt.Errorf("no block on the device")
	}
	lg, err := s.Prefill(m.Vocab.Encode("Hello there", true))
	if err == nil {
		_, err = s.Forward(Greedy(lg))
	}
	var b strings.Builder
	for i, e := range rec.Launched() {
		// A kernel the composer formed from ops is marked; one launched from
		// anywhere else (a pre-made kernel) is not, and is not the composer's.
		mark := ""
		if strings.HasPrefix(e.K.Comment, "composed") {
			mark = " [composed]"
		}
		fmt.Fprintf(&b, "// kernel %d: %s%s\n%s\n", i, e.K.Name, mark, e.Code)
	}
	return b.String(), err
}
