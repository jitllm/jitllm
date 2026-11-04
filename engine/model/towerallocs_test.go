package model

import (
	"image"
	"testing"
)

// TestWarmImageEncodeDoesNotAllocate encodes a picture twice and then a
// third time under an allocation profile: the warm encode makes no engine
// heap allocation (RULE 2f) -- on a tower of one square (SmolVLM's) and on
// one of pictures of their own size, its entry formed for classes and its
// buffers grown to the largest picture yet (synth-qwen2vl). The counter
// must see an allocation where there is one: the first encode, which grows
// the tower's buffers and its forms' own.
func TestWarmImageEncodeDoesNotAllocate(t *testing.T) {
	arms := []struct {
		name string
		open func(*testing.T) (*Model, image.Image)
	}{
		{"smolvlm", openSmolVLM},
		{"qwen2vl", func(t *testing.T) (*Model, image.Image) {
			m, g := openQVLTower(t, "synth-qwen2vl")
			return m, rgbPicture(g.Image.RGB, g.Image.W, g.Image.H)
		}},
	}
	for _, arm := range arms {
		t.Run(arm.name, func(t *testing.T) {
			m, img := arm.open(t)
			defer m.Close()
			s := m.NewInstance(64)
			defer s.Close()
			encode := func() {
				if _, err := s.EncodePicture(img); err != nil {
					t.Fatal(err)
				}
			}
			cold := countAllocs(encode)
			if cold.engine == 0 {
				t.Fatal("VIOLATION: the first encode, which grows the tower's buffers, counted no allocation")
			}
			encode()
			warm := countAllocs(encode)
			if warm.engine != 0 {
				t.Fatalf("a warm encode made %d engine allocations:\n%s", warm.engine, warm.where)
			}
			t.Logf("cold %d engine allocations, warm %d (%d in the runtime's caches)", cold.engine, warm.engine, warm.caches)
		})
	}
}
