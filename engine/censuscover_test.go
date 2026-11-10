package engine

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/convert/gguf"
)

// TestTheCensusCoversEveryGraph holds the census's models to the converter's
// lists, which docs/models.md is generated from: every graph a GGUF
// architecture name or a Hugging Face class converts to, and every vision
// projector, is read by at least one model the census runs, so a green census
// is every entry of the page opening and running. A graph or projector no
// model reads fails by name; with no models every entry is a gap.
func TestTheCensusCoversEveryGraph(t *testing.T) {
	graphs, projectors := censusGaps(t, auditModels(t))
	for _, g := range graphs {
		t.Errorf("no census model reads graph %s", g)
	}
	for _, p := range projectors {
		t.Errorf("no census model carries projector %s", p)
	}
	want, wantProj := censusWants()
	t.Logf("%d graphs and %d projectors in the converter's lists; %d and %d without a census model",
		len(want), len(wantProj), len(graphs), len(projectors))
	g0, p0 := censusGaps(t, nil)
	if len(g0) != len(want) || len(p0) != len(wantProj) {
		t.Fatalf("VIOLATION: no models left %d graph and %d projector gaps, want %d and %d",
			len(g0), len(p0), len(want), len(wantProj))
	}
}

// censusWants is the graphs the converter's two lists convert to, each with
// the first name that reaches it, and the projectors it reads.
func censusWants() (map[convert.Arch]string, map[string]bool) {
	want := map[convert.Arch]string{}
	for _, a := range convert.Architectures() {
		// A projector's own container: its towers are the projectors below.
		if a == "clip" {
			continue
		}
		if g, ok := convert.ArchitectureGraph(a); ok {
			if _, in := want[g]; !in {
				want[g] = a
			}
		}
	}
	for _, c := range convert.HFClasses() {
		if g, ok := convert.HFClassGraph(c); ok {
			if _, in := want[g]; !in {
				want[g] = c
			}
		}
	}
	proj := map[string]bool{}
	for _, p := range convert.Projectors() {
		proj[p] = true
	}
	return want, proj
}

// censusGaps is the graphs and projectors of censusWants that none of ms
// reads: a GGUF's general.architecture, a safetensors directory's class, a
// catalog model's projector.
func censusGaps(t *testing.T, ms []auditModel) (graphs, projectors []string) {
	t.Helper()
	want, proj := censusWants()
	have := map[convert.Arch]bool{}
	carried := map[string]bool{}
	for _, m := range ms {
		if m.hf {
			b, err := os.ReadFile(filepath.Join(m.path, "config.json"))
			if err != nil {
				t.Errorf("%s: %v", m.name, err)
				continue
			}
			var cfg struct {
				Architectures []string `json:"architectures"`
			}
			if err := json.Unmarshal(b, &cfg); err != nil {
				t.Errorf("%s: config.json: %v", m.name, err)
				continue
			}
			for _, c := range cfg.Architectures {
				if g, ok := convert.HFClassGraph(c); ok {
					have[g] = true
				}
			}
			continue
		}
		if g, ok := convert.ArchitectureGraph(ggufKey(t, m.path, "general.architecture")); ok {
			have[g] = true
		}
		if m.mmproj != "" {
			pt := ggufProjector(t, m.mmproj)
			if !carried[pt] {
				t.Logf("projector %s: %s (%s)", pt, m.name, m.path)
			}
			carried[pt] = true
		}
	}
	for g, name := range want {
		if !have[g] {
			graphs = append(graphs, g.String()+" ("+name+")")
		}
	}
	for p := range proj {
		if !carried[p] {
			projectors = append(projectors, p)
		}
	}
	sort.Strings(graphs)
	sort.Strings(projectors)
	return graphs, projectors
}

// ggufKey is a GGUF's string key k, "" where the file or the key is absent.
func ggufKey(t *testing.T, p, k string) string {
	t.Helper()
	f, err := gguf.Open(p)
	if err != nil {
		t.Errorf("%s: %v", p, err)
		return ""
	}
	defer f.Close()
	v, _ := f.KV[k].String()
	return v
}

// ggufProjector is a projector GGUF's projector type, under either key the
// converter reads.
func ggufProjector(t *testing.T, p string) string {
	if v := ggufKey(t, p, "clip.projector_type"); v != "" {
		return v
	}
	return ggufKey(t, p, "clip.vision.projector_type")
}
