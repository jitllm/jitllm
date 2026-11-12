package engine

import (
	"image"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestWarmImageEncodeDoesNotAllocate encodes a picture twice and then a
// third time under an allocation profile: the warm encode makes no engine
// heap allocation (RULE 2f) -- on a tower of one square (SmolVLM's) and on
// one of pictures of their own size, its entry formed for classes and its
// buffers grown to the largest picture yet (synth-qwen2vl), and on every
// stated tower's fixture over a picture of no tower's square. The counter
// must see an allocation where there is one: the first encode, which grows
// the tower's buffers and its forms' own.
func TestWarmImageEncodeDoesNotAllocate(t *testing.T) {
	type encodeArm struct {
		name string
		open func(*testing.T) (*model.Model, image.Image, []Option)
	}
	arms := []encodeArm{
		{"smolvlm", func(t *testing.T) (*model.Model, image.Image, []Option) { m, img := openSmolVLM(t); return m, img, nil }},
		{"qwen2vl", func(t *testing.T) (*model.Model, image.Image, []Option) {
			m, g, openOpts := openQVLTower(t, "synth-qwen2vl")
			return m, rgbPicture(g.Image.RGB, g.Image.W, g.Image.H), openOpts
		}},
	}
	for _, src := range towerFixtures(t) {
		arms = append(arms, encodeArm{src.name, func(t *testing.T) (*model.Model, image.Image, []Option) {
			m, err := model.Open(pairContainer(t, src))
			if err != nil {
				t.Fatal(err)
			}
			return m, principlePicture(), nil
		}})
	}
	for _, arm := range arms {
		t.Run(arm.name, func(t *testing.T) {
			m, img, io := arm.open(t)
			defer m.Close()
			s := testInstance(t, m, 64, io...)
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
