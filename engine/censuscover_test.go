package engine

import (
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestTheCensusCoversEveryFeature holds the census's models to what the
// library states: every feature (convert.Features -- each graph's op by what
// it computes, a graph's residual streams, each field a container statement
// sets) is taken by a model the census runs (convert.FeaturesOf), and every
// vision projector is carried by one, so a green census is every statement
// the converter can write opening and running. A feature or projector no
// model takes fails by name; with no models every one is a gap.
func TestTheCensusCoversEveryFeature(t *testing.T) {
	features, projectors := censusGaps(t, auditModels(t))
	if testmodels.Free() && len(features)+len(projectors) > 0 {
		testmodels.Missing(t, "%d features and %d projectors take no model here: the census's models are not in the model-free run",
			len(features), len(projectors))
	}
	for _, f := range features {
		by, err := convert.StatedBy(f)
		if err != nil {
			t.Fatal(err)
		}
		t.Errorf("no census model takes %s (stated by %v)", f, by)
	}
	for _, p := range projectors {
		t.Errorf("no census model carries projector %s", p)
	}
	want, err := convert.Features()
	if err != nil {
		t.Fatal(err)
	}
	wantProj := convert.Projectors()
	t.Logf("%d features and %d projectors in the library; %d and %d without a census model",
		len(want), len(wantProj), len(features), len(projectors))
	f0, p0 := censusGaps(t, nil)
	if len(f0) != len(want) || len(p0) != len(wantProj) {
		t.Fatalf("VIOLATION: no models left %d feature and %d projector gaps, want %d and %d",
			len(f0), len(p0), len(want), len(wantProj))
	}
}

// censusGaps is the library's features and projectors none of ms takes:
// what converting each would state (convert.FeaturesOf), and a catalog
// model's projector.
func censusGaps(t *testing.T, ms []auditModel) (features, projectors []string) {
	t.Helper()
	want, err := convert.Features()
	if err != nil {
		t.Fatal(err)
	}
	taken := map[string]bool{}
	carried := map[string]bool{}
	for _, m := range ms {
		var opts []convert.Option
		path := m.path
		if src, sf, ok := statedSource(m.path); ok {
			// A restated fixture takes what its restatement states.
			path = src
			opts = append(opts, convert.WithStatedSource(statedSuffixes()[sf]))
		}
		if c := ModelConfigForTest(path); c != "" {
			opts = append(opts, convert.WithModelConfig(c))
		}
		if tpl, ok := censusTemplates()[modelRel(m.path)]; ok {
			opts = append(opts, convert.WithChatTemplate(testmodels.Path(tpl)))
		}
		fs, err := convert.FeaturesOf(path, m.mmproj, opts...)
		if errors.Is(err, convert.ErrPart) {
			// A part takes its features joined to the trunk it was exported
			// beside, as `jitllm convert -head` converts it.
			trunk := trunkOf(m.path)
			if trunk == "" {
				t.Errorf("%s is part of a model and no trunk is paired with it (trunkPairs): %v", m.name, err)
				continue
			}
			fs, err = convert.FeaturesOf(trunk, m.mmproj, append(opts, convert.WithHead(m.path))...)
			if err != nil {
				t.Errorf("%s joined to %s: %v", m.name, filepath.Base(trunk), err)
				continue
			}
			t.Logf("%s: a part, counted joined to %s", m.name, filepath.Base(trunk))
		}
		if err != nil {
			// A file the converter refuses takes nothing; the gates that run
			// every model hold it to its refusal.
			t.Logf("%s takes no feature: %v", m.name, err)
			continue
		}
		for _, f := range fs {
			taken[f] = true
		}
		if m.mmproj != "" {
			pt := ggufProjector(t, m.mmproj)
			if !carried[pt] {
				t.Logf("projector %s: %s (%s)", pt, m.name, m.path)
			}
			carried[pt] = true
		}
	}
	for _, f := range want {
		if !taken[f] {
			features = append(features, f)
		}
	}
	for _, p := range convert.Projectors() {
		if !carried[p] {
			projectors = append(projectors, p)
		}
	}
	sort.Strings(projectors)
	return features, projectors
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

// trunkOf is the trunk GGUF a prediction head published alone (convert.ErrPart)
// joins, by trunkPairs; "" where none is paired or the box does not hold it.
func trunkOf(p string) string {
	t, ok := trunkPairs()[modelRel(p)]
	if !ok {
		return ""
	}
	q := testmodels.Path(t)
	if _, err := os.Stat(q); err != nil {
		return ""
	}
	return q
}

// modelRel is p relative to the model directory, with forward slashes.
func modelRel(p string) string {
	return filepath.ToSlash(strings.TrimPrefix(p, testmodels.Dir()+string(filepath.Separator)))
}

// censusTemplates are the models, relative to the model directory, the census
// converts with their publisher's chat template (convert.WithChatTemplate, as
// `jitllm convert -chat-template` does), and the template: a tool-call form
// no model here carries the template of is taken from its family's own.
func censusTemplates() map[string]string {
	return map[string]string{
		"Mistral-7B-Instruct-v0.3-Q4_K_M.gguf": "Mistral-7B-Instruct-v0.3/tokenizer_config.json",
		"synth-kimilinear":                     "Kimi-Linear-48B-A3B-Instruct/chat_template.jinja",
		"synth-minimaxm2.gguf":                 "MiniMax-M2/chat_template.jinja",
	}
}

// trunkPairs are the heads published alone, relative to the model directory,
// and the trunk each was exported beside.
func trunkPairs() map[string]string {
	return map[string]string{
		"synth-qwen4exp-mtp-head.gguf": "synth-qwen4exp-mtp-trunk.gguf",
	}
}
