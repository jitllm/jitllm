package convert

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// phi3 slides every layer, and a converted phi3 must say so (llama.cpp applies
// phi3.attention.sliding_window to every layer). Phi-3.5's 262,144 window over
// a 131,072 context cannot bite, but it is the file's statement and is carried.
func TestPhi3SlidesEveryLayer(t *testing.T) {
	p := testmodels.Path("Phi-3.5-mini-instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	c := ggufSource(t, p).Config
	if c.SWAWindow != 262144 {
		t.Errorf("SWAWindow %d, the file says 262144", c.SWAWindow)
	}
	// Config.SWA's own rule, il%P < P-1, must hold for every layer.
	for il := uint32(0); il < c.NLayer; il++ {
		if c.SWAPeriod == 0 || il%c.SWAPeriod >= c.SWAPeriod-1 {
			t.Fatalf("layer %d is not local under period %d", il, c.SWAPeriod)
		}
	}
}

// phi-4 has no window at all. A file that states its key converts as it
// says; one written before the key existed is refused, an absent key being
// no statement, and converts with the publisher's config.json
// (WithModelConfig) as no window.
func TestPhi4AttendsFully(t *testing.T) {
	p := testmodels.Path("phi-4-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	f, err := gguf.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	c, err := configOf(f)
	if err != nil {
		t.Fatal(err)
	}
	if c.SWAWindow != 0 || c.SWAPeriod != 0 {
		t.Errorf("SWAWindow %d period %d, phi-4 attends every position", c.SWAWindow, c.SWAPeriod)
	}
	// A file written before the key existed.
	delete(f.KV, "phi3.attention.sliding_window")
	if _, err := configOf(f); err == nil || !strings.Contains(err.Error(), "-config") {
		t.Fatalf("phi-4 with no attention.sliding_window: %v, want it refused naming -config", err)
	}
	dir := t.TempDir()
	cfg := `{"architectures": ["Phi3ForCausalLM"], "hidden_size": 5120, "intermediate_size": 17920,
		"max_position_embeddings": 16384, "num_attention_heads": 40, "num_hidden_layers": 40,
		"num_key_value_heads": 10, "rms_norm_eps": 1e-05, "rope_theta": 250000, "sliding_window": null,
		"vocab_size": 100352, "tie_word_embeddings": false}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	o := options{modelConfig: dir}
	if err := o.stateAbsentKeys(f); err != nil {
		t.Fatal(err)
	}
	if c, err = configOf(f); err != nil {
		t.Fatal(err)
	}
	if c.SWAWindow != 0 || c.SWAPeriod != 0 {
		t.Errorf("with its config.json: SWAWindow %d period %d, phi-4 attends every position", c.SWAWindow, c.SWAPeriod)
	}
}
