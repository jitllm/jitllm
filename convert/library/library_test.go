package library

import (
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/convert/hf"
)

// Every entry must be fetchable as written: a unique name, an architecture
// the converter implements, and references that parse back to themselves.
func TestEveryEntryIsWellFormed(t *testing.T) {
	if len(Models()) == 0 {
		t.Fatal("the library is empty")
	}
	seen := map[string]bool{}
	for _, m := range Models() {
		if m.Name == "" || seen[m.Name] {
			t.Errorf("name %q is empty or repeated", m.Name)
		}
		seen[m.Name] = true
		if !convert.Supported(m.Arch) {
			t.Errorf("%s: architecture %q is not one the converter implements", m.Name, m.Arch)
		}
		if m.Bytes <= 0 || m.Title == "" {
			t.Errorf("%s: missing size or title", m.Name)
		}
		refs := []hf.Ref{m.Ref()}
		if tr, ok := m.TowerRef(); ok {
			refs = append(refs, tr)
		}
		for _, r := range refs {
			back, remote, err := hf.Parse(r.String())
			if err != nil || !remote || back != r {
				t.Errorf("%s: %s does not parse back to itself (%+v, %v)", m.Name, r, back, err)
			}
		}
		if got, ok := Find(m.Name); !ok || got.Name != m.Name {
			t.Errorf("Find(%q) does not find it", m.Name)
		}
	}
}
