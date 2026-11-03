package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// hfGoldDeclined names goldens whose architecture converts and does not yet
// run, with the substring of the engine's refusal that says so. It is a list,
// not a `continue`: a listed model that opens fails its subtest, and matching on
// the refusal's text keeps any other failure fatal. The golden stays, since it
// is what the graph will be gated against when it lands.
func hfGoldDeclined() map[string]string { return map[string]string{} }

// declinedRefusal reports whether err is Open declining an ARCHITECTURE this
// engine converts and does not yet run, as opposed to any other failure.
//
// A declined container is still a valid one, so the stale-format filter the
// every-model sweeps have cannot see it. Matching the refusal's text from the
// same list keeps it narrow.
func declinedRefusal(err error) bool {
	if err == nil {
		return false
	}
	for _, why := range hfGoldDeclined() {
		if strings.Contains(err.Error(), why) {
			return true
		}
	}
	return false
}

// TestSafetensorsMatchTransformers gates the safetensors path against
// transformers itself: scripts/hfgold.py runs the reference class over fixed ids
// in float32 and records the logits, so the container's graph is compared with
// no tokenizer or quantisation in between. Random weights make it sharp: a
// wiring error is NMSE ~1, not a near-miss. It cannot see a norm on fixtures
// whose norm weights are all one; qwen3 is the exception.
func TestSafetensorsMatchTransformers(t *testing.T) {
	golds, _ := filepath.Glob(filepath.Join("..", "..", "testdata", "golden", "hf", "*.json"))
	if len(golds) == 0 {
		t.Fatal("no goldens -- run scripts/hfgold.py; this gate proved nothing")
	}
	// The KV cache is f32, since the bound is the whole gate: the binary16
	// cache's own rounding is near the weakest violation. WithKVF16 configures
	// only the model this gate opens.
	for _, g := range golds {
		name := strings.TrimSuffix(filepath.Base(g), ".json")
		t.Run(name, func(t *testing.T) {
			b, err := os.ReadFile(g)
			if err != nil {
				t.Fatal(err)
			}
			var gold struct {
				IDs []int32 `json:"ids"`
				Tok []struct {
					Text string  `json:"text"`
					IDs  []int32 `json:"ids"`
					Dec  string  `json:"dec"`
					// sentencepiece's, where the directory ships tokenizer.model
					SPIDs []int32 `json:"sp_ids"`
					SPDec string  `json:"sp_dec"`
				} `json:"tok"`
				Pos []struct {
					Argmax int       `json:"argmax"`
					Max    float64   `json:"max"`
					Head   []float64 `json:"head"`
				} `json:"pos"`
			}
			if err := json.Unmarshal(b, &gold); err != nil {
				t.Fatal(err)
			}
			m, err := Open(hfContainer(t, name), noTune(), WithKVF16(false))
			if err != nil {
				if why, listed := hfGoldDeclined()[name]; listed && strings.Contains(err.Error(), why) {
					t.Skipf("DECLINED BY NAME, and the golden is kept because it is what the "+
						"graph will be gated against when it lands (RULE 11: a missing "+
						"artefact is a task): %v", err)
				}
				t.Fatal(err)
			}
			defer m.Close()
			if _, listed := hfGoldDeclined()[name]; listed {
				t.Fatalf("%s is in hfGoldDeclined and it OPENED -- delete its entry so this "+
					"gate actually runs, which is the whole point of the list being explicit", name)
			}
			// The tokenizer half: HuggingFace's fast tokenizer over text with a
			// leading space, a double space, a newline and byte fallback. The
			// graph half uses fixed ids, so neither hides behind the other.
			if m.Vocab == nil {
				t.Fatalf("no tokenizer: %v", m.TokErr)
			}
			for _, tk := range gold.Tok {
				// Where sentencepiece (what the weights were trained against)
				// exists, it wins; tokenizer.json's departures are logged.
				if tk.SPIDs != nil {
					if !slices.Equal(tk.IDs, tk.SPIDs) {
						t.Logf("tokenizer.json departs from sentencepiece on %q: %v against %v -- "+
							"matching sentencepiece", tk.Text, tk.IDs, tk.SPIDs)
					}
					tk.IDs, tk.Dec = tk.SPIDs, tk.SPDec
				}
				if got := m.Vocab.Encode(tk.Text, true); !slices.Equal(got, tk.IDs) {
					t.Errorf("Encode(%q)\n  ours         %v\n  reference    %v", tk.Text, got, tk.IDs)
				}
				// Decode keeps the dummy-prefix space and the reference strips
				// it, deliberately (a streamed " Paris" keeps its space).
				want := tk.Dec
				if m.Vocab.AddSpacePrefix {
					want = " " + want
				}
				if got := m.Vocab.Decode(tk.IDs); got != want {
					t.Errorf("Decode(%v)\n  ours         %q\n  reference    %q", tk.IDs, got, want)
				}
			}
			if len(gold.Tok) == 0 {
				t.Error("golden has no tokenizer half -- regenerate with scripts/hfgold.py")
			}
			st := m.NewInstance(len(gold.IDs) + 1)
			defer st.Close()
			worst := 0.0
			for p, id := range gold.IDs {
				lg, err := st.Forward(id)
				if err != nil {
					t.Fatal(err)
				}
				want := gold.Pos[p]
				var num, den float64
				for i, w := range want.Head {
					d := float64(lg[i]) - w
					num, den = num+d*d, den+w*w
				}
				nmse := num / den
				worst = max(worst, nmse)
				best := 0
				for i, v := range lg {
					if v > lg[best] {
						best = i
					}
				}
				// Non-finite first: NaN > bound is false.
				if math.IsNaN(nmse) || math.IsInf(nmse, 0) || nmse > hfNMSE {
					t.Fatalf("pos %d: NMSE %.3e against transformers (bound %.0e)", p, nmse, hfNMSE)
				}
				if best != want.Argmax {
					t.Errorf("pos %d: argmax %d, transformers %d (max %.4f, ours %.4f)",
						p, best, want.Argmax, want.Max, lg[best])
				}
			}
			t.Logf("%d positions, worst NMSE %.2e", len(gold.IDs), worst)
		})
	}
}

// hfContainer converts a fixture fresh, every run: jlmOfSafetensors' cache is
// keyed on config.json's mtime, which a converter change does not touch. These
// fixtures convert in well under a second. Where the directory is absent (a
// host holding containers only) it takes models/<name>.jlm and says so.
func hfContainer(t *testing.T, name string) string {
	t.Helper()
	dir := testmodels.Path(name)
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err == nil {
		dst := filepath.Join(t.TempDir(), name+jlm.Ext)
		if _, err := convert.FromSafetensors(dir, dst, jlm.Fingerprint{Host: "test"}); err != nil {
			t.Fatalf("convert %s: %v", dir, err)
		}
		return dst
	}
	if p := dir + jlm.Ext; fileExists(p) {
		t.Logf("no %s here; using %s, converted elsewhere", dir, p)
		return p
	}
	t.Skipf("MODEL MISSING: %s and %s%s (set JITLLM_MODELS to the model directory) -- this gate proved nothing", dir, dir, jlm.Ext)
	return ""
}

func fileExists(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && !fi.IsDir()
}

// hfNMSE is measured: correct arms with an f32 cache read 1.5e-13 to 1.1e-11,
// and the weakest violation found (gate and up swapped in every OLMoE expert)
// reads 1.7e-06.
const hfNMSE = 1e-9
