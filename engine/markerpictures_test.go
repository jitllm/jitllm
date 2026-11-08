package engine

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestMarkerPromptWithPicturesMatchesItsCode holds a marker head's prompt
// over pictures (jlm.PromptPictures) to the model's own code
// (scripts/d1omnigold.py): the pictures' prefix rows, and each question's
// prompt ids and option markers token for token.
func TestMarkerPromptWithPicturesMatchesItsCode(t *testing.T) {
	src, ok := existingModel(testmodels.Path("d1-omni/d1-omni-600M-F16.gguf"))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: %s", testmodels.Path("d1-omni/d1-omni-600M-F16.gguf"))
	}
	proj := towerOf(src)
	if proj == "" {
		t.Fatalf("%s has no projector beside it", src)
	}
	m, err := model.Open(jlmOfPair(t, src, proj))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	b, err := os.ReadFile(filepath.Join("testdata", "decision", "d1-omni.pictures.reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var gold struct {
		Questions []struct {
			ID      string  `json:"id"`
			IDs     []int32 `json:"ids"`
			Markers []int   `json:"markers"`
			Prefix  int     `json:"prefix"`
		} `json:"questions"`
	}
	if err := json.Unmarshal(b, &gold); err != nil {
		t.Fatal(err)
	}
	in, err := NewInstance(m, 4096, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	d, err := NewDecider(in)
	if err != nil {
		t.Fatal(err)
	}
	state, qs := decisionRequest(t, "pictures")
	pics := decisionPictures(t, m, "pictures")
	prefix := 0
	for i := range pics {
		prefix += pics[i].N(m.Cfg.NEmbd)
	}
	if len(gold.Questions) != len(qs) {
		t.Fatalf("%d questions, the reference's %d", len(qs), len(gold.Questions))
	}
	for i, g := range gold.Questions {
		if prefix != g.Prefix {
			t.Fatalf("%s: the pictures take %d rows, the reference's %d", g.ID, prefix, g.Prefix)
		}
		ids, marks, err := markedPrompt(d.mp, d.encodePlain, state, &qs[i], in.MaxSeq()-prefix, len(pics))
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(ids, g.IDs) || !slices.Equal(marks, g.Markers) {
			t.Fatalf("%s: ids %v markers %v, the reference's %v %v", g.ID, ids, marks, g.IDs, g.Markers)
		}
	}
	t.Logf("%d questions after %d picture rows match the reference token for token", len(qs), prefix)
}
