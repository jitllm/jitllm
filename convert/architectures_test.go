package convert

import (
	"slices"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestArchitecturesAreTheLibrarys holds the list the library's files name
// to every name the converter reads a graph for: a name a file adds that no
// reader reads, or a reader's name no file names, fails.
func TestArchitecturesAreTheLibrarys(t *testing.T) {
	got := Architectures()
	var want []string
	for n := range archOf() {
		want = append(want, n)
	}
	sort.Strings(want)
	if len(got) < 40 || !slices.Equal(got, want) {
		t.Fatalf("the library names %v\nthe readers %v", got, want)
	}
}

// TestEncodersAreTheLibrarys holds the encoders the library states to the
// GGUF architectures that convert as one.
func TestEncodersAreTheLibrarys(t *testing.T) {
	if got, want := Encoders(), []string{"bert", "modern-bert", "nomic-bert"}; !slices.Equal(got, want) {
		t.Fatalf("encoders %v, want %v", got, want)
	}
}

// statedBy is jc with what the library states for a GGUF of architecture
// arch that states no keys.
func statedBy(t *testing.T, arch string, jc *jlm.Config) *Config {
	t.Helper()
	c := &Config{Config: jc}
	f := meta.New(arch+".gguf", map[string]meta.Value{"general.architecture": meta.MakeString(arch)}, nil)
	if err := stateContainer(f, c); err != nil {
		t.Fatalf("%s: %v", arch, err)
	}
	return c
}
