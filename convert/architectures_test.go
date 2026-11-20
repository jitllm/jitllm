package convert

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestUnstatedConfigsAreRefused holds every stage that reads the library's
// statement to refusing a Config the library has not stated, rather than
// reading it as stating nothing: ModernBERT's erf GELU and PhiMoE's
// sparsemixer would otherwise fall to the flags' defaults.
func TestUnstatedConfigsAreRefused(t *testing.T) {
	jc := func() *jlm.Config { return &jlm.Config{NLayer: 1, NExpert: 4} }
	if _, err := Activation(&Config{Config: jc()}, 0, true, false); err == nil {
		t.Error("an unstated config's activation was read")
	}
	if err := StateRevision(&Config{Config: jc()}); err == nil {
		t.Error("an unstated config's revision was stated")
	}
	if _, err := Activation(Unstated(jc()), 0, true, false); err != nil {
		t.Errorf("a config stated as stating nothing: %v", err)
	}
	if err := StateRevision(Unstated(jc())); err != nil {
		t.Errorf("a config stated as stating nothing: %v", err)
	}
}

// TestEncodersAreTheLibrarys holds the encoders the library states to the
// GGUF architectures that convert as one.
func TestEncodersAreTheLibrarys(t *testing.T) {
	got, err := Encoders()
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"bert", "gemma-embedding", "modern-bert", "nomic-bert"}; !slices.Equal(got, want) {
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
