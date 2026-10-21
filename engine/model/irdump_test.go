package model

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/backend/recorder"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
	"github.com/jitllm/jitllm/paritytest/maintier"
)

var (
	irOut      = flag.String("ir-out", "", "TestDumpEmittedKernels writes <dir>/<profile>/<model>.ir here")
	irMain     = flag.Bool("ir-main", false, "TestDumpEmittedKernels runs main's tier (paritytest/maintier), not the composer's")
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
			code, irText, _ := strings.Cut(code, "\x00")
			if err := os.WriteFile(filepath.Join(dir, name+".code"), []byte(code), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, name+".ir"), []byte(irText), 0o644); err != nil {
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
	// The composer's tier, or main's (its kernels the parity golden), on
	// the same fake device.
	var g nn.Device
	if *irMain {
		mg, err := maintier.New([]maintier.Slot{{Dev: rec, Bytes: 1 << 40}},
			maintier.WithConfig(func(c *maintier.Config) { c.NoGraph = true }))
		if err != nil {
			return "", err
		}
		defer mg.Close()
		g = mg
	} else {
		cg, err := tier.New([]tier.Slot{{Dev: rec, Bytes: 1 << 40}}, append(testTierOpts(t),
			tier.WithConfig(func(c *tier.Config) { c.NoGraph = true }))...)
		if err != nil {
			return "", err
		}
		defer cg.Close()
		g = cg
	}
	s := m.NewState(32)
	defer s.Close()
	if err := s.SetDevice(g); err != nil {
		return "", err
	}
	if s.GPULayers() == 0 {
		return "", fmt.Errorf("no block on the device")
	}
	// A prompt chunk, then one decode step, each asked whatever the other
	// did: the prologue forms every block's kernels for the step's rows
	// before the first launch, so a step that stops at an op the composer
	// cannot form has still asked for the rest.
	lg, err := s.Prefill(m.Vocab.Encode("Hello there", true))
	tok := int32(1)
	if err == nil {
		tok = Greedy(lg)
	}
	_, ferr := s.Forward(tok)
	if err == nil {
		err = ferr
	}
	var b, irb strings.Builder
	// What was launched, in order (main's pre-made kernels, on main); then
	// what the composer formed and the step never reached.
	seen := map[*ir.Kernel]bool{}
	i := 0
	emit := func(e recorder.Emitted) {
		if seen[e.K] {
			return
		}
		seen[e.K] = true
		mark := ""
		if strings.HasPrefix(e.K.Comment, "composed") {
			mark = " [composed]"
		}
		fmt.Fprintf(&b, "// kernel %d: %s%s\n%s\n", i, e.K.Name, mark, e.Code)
		fmt.Fprintf(&irb, "// kernel %d: %s%s\n%s\n", i, e.K.Name, mark, recorder.Text(e.K))
		i++
	}
	for _, e := range rec.Launched() {
		emit(e)
	}
	for _, e := range rec.Compiled(func(k *ir.Kernel) bool { return strings.HasPrefix(k.Comment, "composed") }) {
		emit(e)
	}
	return b.String() + "\x00" + irb.String(), err
}
