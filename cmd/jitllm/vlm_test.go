package main

import (
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine"
)

// TestImageRunsThroughTheTower runs `jitllm run -image` on SmolVLM-256M, on
// the CPU device, completed by its chat template, and with a cached prefix:
// the coloured quadrants and yellow disc are described by the disc's colour,
// and a blank grey picture is not -- a tower wired to nothing still answers
// fluently, so the blank is the violation.
func TestImageRunsThroughTheTower(t *testing.T) {
	path := modelPath(t, "SmolVLM-256M-Instruct-Q8_0-vlm.jlm")
	pic := filepath.Join("..", "..", "engine", "testdata", "quad-and-disc.png")
	if _, err := os.Stat(pic); err != nil {
		t.Fatalf("IMAGE MISSING: %v", err)
	}
	blank := filepath.Join(t.TempDir(), "blank.png")
	g := image.NewGray(image.Rect(0, 0, 384, 384))
	for i := range g.Pix {
		g.Pix[i] = 128
	}
	f, err := os.Create(blank)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(f, g); err != nil {
		t.Fatal(err)
	}
	f.Close()
	ask := func(img, cache string) string {
		t.Helper()
		out, stderr, err := capture(t, func() error {
			return run(path, "Describe this image in one sentence.", 32, 0, "cpu", -1, 0, 1<<30, false, false,
				false, false, &engine.Sampler{}, img, nil, nil, cache, 0, 0, nil, nil)
		})
		if err != nil {
			t.Fatalf("run -image %s: %v\n%s", img, err, stderr)
		}
		return strings.ToLower(out)
	}
	got := ask(pic, "")
	none := ask(blank, "")
	t.Logf("picture %q", got)
	t.Logf("blank   %q", none)
	if !strings.Contains(got, "yellow") {
		t.Errorf("the description does not name the yellow disc: %q", got)
	}
	if strings.Contains(none, "yellow") {
		t.Errorf("a blank picture is yellow too (%q): the picture is not what answered", none)
	}
	// The same through a prompt cache, written by one run and read by the
	// next: the run restored past the picture answers as the uncached run did.
	dir := t.TempDir()
	if first := ask(pic, dir); first != got {
		t.Errorf("a cached run's first answer %q differs from the uncached %q", first, got)
	}
	if again := ask(pic, dir); again != got {
		t.Errorf("a run restored past the picture answers %q, not %q", again, got)
	}
}
