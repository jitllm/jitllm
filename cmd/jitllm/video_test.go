package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/engine"
)

// writeFrames writes frames into a new directory, frame i under names[i].
func writeFrames(t *testing.T, frames []image.Image, names []string) string {
	t.Helper()
	dir := t.TempDir()
	for i, img := range frames {
		f, err := os.Create(filepath.Join(dir, names[i]))
		if err != nil {
			t.Fatal(err)
		}
		if err := png.Encode(f, img); err != nil {
			t.Fatal(err)
		}
		f.Close()
	}
	return dir
}

// TestVideoRunsThroughTheTower runs `jitllm run -video` on the Qwen2-VL
// fixture, completed by its chat template: a directory of four frames of a
// disc moving across a gradient, read in name order. The same frames under
// names that sort them backwards answer as the frames reversed do and not
// as the frames in order, so the video reaches the model and its frames in
// the directory's name order; a directory holding no frame is refused.
func TestVideoRunsThroughTheTower(t *testing.T) {
	path := modelPath(t, "synth-qwen2vl-vlm.jlm")
	var frames, rev []image.Image
	for f := range 4 {
		img := image.NewNRGBA(image.Rect(0, 0, 112, 84))
		for y := range 84 {
			for x := range 112 {
				c := color.NRGBA{R: uint8(255 * x / 112), G: uint8(255 * y / 84), B: 64, A: 255}
				if dx, dy := x-12-10*f, y-42; dx*dx+dy*dy < 100 {
					c = color.NRGBA{R: 250, G: 250, B: 250, A: 255}
				}
				img.SetNRGBA(x, y, c)
			}
		}
		frames = append(frames, img)
		rev = append([]image.Image{img}, rev...)
	}
	inOrder := []string{"a.png", "b.png", "c.png", "d.png"}
	backwards := []string{"d.png", "c.png", "b.png", "a.png"}
	ask := func(dir string) (string, error) {
		t.Helper()
		out, stderr, err := capture(t, func() error {
			return run(path, "What happens in the video?", 12, 0, "cpu", -1, 0, 1<<30, false, false,
				false, false, &engine.Sampler{}, mediaSpec{video: dir}, nil, nil, "", 0, 0, nil, nil)
		})
		if err != nil {
			return "", fmt.Errorf("%w\n%s", err, stderr)
		}
		return out, nil
	}
	got, err := ask(writeFrames(t, frames, inOrder))
	if err != nil {
		t.Fatalf("run -video: %v", err)
	}
	named, err := ask(writeFrames(t, frames, backwards))
	if err != nil {
		t.Fatalf("run -video, the names backwards: %v", err)
	}
	reversed, err := ask(writeFrames(t, rev, inOrder))
	if err != nil {
		t.Fatalf("run -video, the frames reversed: %v", err)
	}
	t.Logf("in order %q; named backwards %q; reversed %q", got, named, reversed)
	if named != reversed {
		t.Errorf("frames named backwards answer %q, the frames reversed %q: the directory is not read in name order", named, reversed)
	}
	if got == reversed {
		t.Errorf("VIOLATION: the frames reversed answer as the frames in order (%q): the video does not reach the model", got)
	}
	if _, err := ask(t.TempDir()); err == nil {
		t.Error("a directory holding no frame was accepted")
	}
}
