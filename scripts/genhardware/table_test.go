package main

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/cost"
)

// TestMergeIsTheTablesSyntax holds the merge to the renderer: the table
// merged with itself is itself, and a file that replaces one hardware model's
// entry and adds another merges into exactly what the renderer writes for
// the profiles edited the same way.
func TestMergeIsTheTablesSyntax(t *testing.T) {
	table, err := os.ReadFile(filepath.Join("..", "..", cost.HardwareTable))
	if err != nil {
		t.Fatal(err)
	}
	same, err := merge(table, table)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(same, table) {
		t.Fatal("the table merged with itself changed")
	}
	ps := cost.Hardware()
	if len(ps) == 0 {
		t.Fatal("an empty table")
	}
	changed := ps[0]
	changed.Device.Bandwidth *= 2
	added := ps[0]
	added.Name = "a model the table lacks"
	from, err := cost.RenderHardware([]cost.Profile{changed, added})
	if err != nil {
		t.Fatal(err)
	}
	got, err := merge(table, from)
	if err != nil {
		t.Fatal(err)
	}
	want, err := cost.RenderHardware(append(slices.Concat([]cost.Profile{changed}, ps[1:]), added))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("merged:\n%s\nrendered:\n%s", got, want)
	}
}
