package convert

import (
	"maps"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestTheProjectorLibraryReads: projectors.json parses, every expression
// compiles, and each entry's code is the container's projector of its name
// (an "as" entry's, the one it names).
func TestTheProjectorLibraryReads(t *testing.T) {
	lib, err := projectorLibrary()
	if err != nil {
		t.Fatal(err)
	}
	if len(lib) == 0 || !slices.Equal(Projectors(), slices.Sorted(maps.Keys(lib))) {
		t.Fatalf("Projectors() %v is not the library's", Projectors())
	}
	if len(AudioProjectors()) == 0 {
		t.Fatal("the library lists no audio tower")
	}
	for name, p := range lib {
		got := jlm.Projector(p.Code).String()
		if got != name && lib[got] != p {
			t.Errorf("%s states code %d, which is the container's %s", name, p.Code, got)
		}
	}
}

// TestProjectorStatementsAreChecked: an entry that states what no reader
// reads is refused.
func TestProjectorStatementsAreChecked(t *testing.T) {
	out := func(dim int) projOutput { return projOutput{"mm.2.weight", dim} }
	for _, c := range []struct {
		what string
		p    projStates
	}{
		{"no code", projStates{Output: out(1)}},
		{"no output", projStates{Code: 2}},
		{"an output dim past a matrix", projStates{Code: 2, Output: out(2)}},
		{"an unknown flag", projStates{Code: 2, Output: out(1), Flags: []string{"rope2d"}}},
		{"an unknown block count", projStates{Code: 2, Output: out(1), Blocks: "triples"}},
		{"an unknown activation", projStates{Code: 2, Output: out(1), Activation: "silu"}},
		{"pair-named blocks", projStates{Code: 2, Output: out(1), Blocks: "pairs"}},
		{"a text row of no role", projStates{Code: 2, Output: out(1),
			TextRows: []projTextRow{{Role: "v.token_embd.no_such_row", Token: "[IMG_BREAK]"}}}},
		{"a text row of a text role", projStates{Code: 2, Output: out(1),
			TextRows: []projTextRow{{Role: "token_embd", Token: "[IMG_BREAK]"}}}},
		{"a text row of no token", projStates{Code: 2, Output: out(1),
			TextRows: []projTextRow{{Role: "v.token_embd.img_break"}}}},
		{"a condition with no statement", projStates{Code: 2, Output: out(1), When: "true"}},
		{"an expression over a name", projStates{Code: 2, Output: out(1), Scale: "config['merge']"}},
		{"an expression that does not parse", projStates{Code: 2, Output: out(1), Scale: "src[["}},
	} {
		if err := c.p.check(); err == nil {
			t.Errorf("%s: accepted", c.what)
		}
	}
	ok := projStates{Code: 2, Output: out(1), Scale: "src['vision.projector.scale_factor'] | default(1)"}
	if err := ok.check(); err != nil {
		t.Errorf("a plain entry: %v", err)
	}
}
