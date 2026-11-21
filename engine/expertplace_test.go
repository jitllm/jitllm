//go:build linux

package engine

import (
	"testing"
)

// TestParsePlacementExperts: %cpu and %card name where a block's experts run,
// anything else is refused, and the host cannot take them beside itself.
func TestParsePlacementExperts(t *testing.T) {
	p, err := ParsePlacement("1-3=cuda:0%cpu,4=cuda:0~%card", false)
	if err != nil {
		t.Fatal(err)
	}
	if p.Blocks[2].Experts != "cpu" || p.Blocks[4].Experts != "card" || !p.Blocks[4].Stream || p.Blocks[2].On != "cuda:0" {
		t.Fatalf("parsed %+v", p.Blocks)
	}
	for _, bad := range []string{"1=cuda:0%gpu", "1=cpu%cpu"} {
		if _, err := ParsePlacement(bad, false); err == nil {
			t.Errorf("%q parsed", bad)
		}
	}
}
