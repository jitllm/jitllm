//go:build linux

package model

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
)

// TestParsePlacementExperts: %host and %card name where a block's experts run,
// anything else is refused, and the host cannot take them beside itself.
func TestParsePlacementExperts(t *testing.T) {
	p, err := ParsePlacement("1-3=cuda:0%host,4=cuda:0~%card", false)
	if err != nil {
		t.Fatal(err)
	}
	if p.Blocks[2].Experts != "host" || p.Blocks[4].Experts != "card" || !p.Blocks[4].Stream || p.Blocks[2].On != "cuda:0" {
		t.Fatalf("parsed %+v", p.Blocks)
	}
	for _, bad := range []string{"1=cuda:0%gpu", "1=host%host"} {
		if _, err := ParsePlacement(bad, false); err == nil {
			t.Errorf("%q parsed", bad)
		}
	}
}

// runExperts runs ids through m, its blocks on tier g (the host alone when g
// is nil), and returns the logits.
func runExperts(t *testing.T, m *Model, g *gpu.GPU, ids []int32) [][]float32 {
	t.Helper()
	s := m.NewInstance(32)
	defer s.Close()
	if g != nil {
		s.SetDeviceLayers(g, m.Cfg.NLayer)
	}
	var out [][]float32
	for _, id := range ids {
		l, err := s.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), l...))
	}
	return out
}

func argmaxOf(v []float32) int {
	b := 0
	for i := range v {
		if v[i] > v[b] {
			b = i
		}
	}
	return b
}
