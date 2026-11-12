package engine

import (
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestRealTowersSeeThePicture asks each real checkpoint the reference gates
// hold to its processor and transformers (TestRealTowersMatchTransformers,
// TestQwenVLRealMatchesTransformers, TestLFM2VLRealMatchesTransformers,
// TestKimiVLRealMatchesMoonshot, TestPhi4RealMatchesTransformers) to
// describe the quadrant picture and a blank grey one (seesThePicture): the
// picture's description names its colours, the blank's fewer. Those gates
// compare a tower's rows; a text model that reads no picture still answers
// fluently, and only an answer about the picture shows it read one.
func TestRealTowersSeeThePicture(t *testing.T) {
	f, err := os.Open(filepath.Join("testdata", "quad-and-disc.png"))
	if err != nil {
		t.Fatal(err)
	}
	img, err := png.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ name, text, proj, config string }{
		{"gemma-4-E2B-it", "vlmb/gemma-4-E2B-it-Q4_K_M.gguf", "vlmb/mmproj-gemma-4-E2B-it-BF16.gguf", ""},
		{"gemma-3n-E2B-it", "gemma3n/gemma-3n-E2B-it-Q4_K_M.gguf", "gemma3n/mmproj-gemma-3n-E2B-it-f16.gguf", ""},
		{"Ministral-3-3B", "newarch/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", "vlmb/mmproj-Ministral-3-3B-BF16.gguf", ""},
		{"Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf", ""},
		{"Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Instruct-f16.gguf", "mmproj-Qwen3-VL-2B-Instruct-f16.gguf", ""},
		{"Qwen3.5-0.8B", "Qwen3.5-0.8B-f16.gguf", "mmproj-Qwen3.5-0.8B-f16.gguf", ""},
		{"LFM2-VL-450M", "vlmb/LFM2-VL-450M-F16.gguf", "vlmb/mmproj-LFM2-VL-450M-F16.gguf", "vlmb/hf/LFM2-VL-450M"},
		{"Kimi-VL-A3B", "Kimi-VL-A3B-Thinking-2506-Q4_K_M.gguf", "mmproj-Kimi-VL-A3B-Instruct-f16.gguf", ""},
		{"Phi-4-reasoning-vision-15B", "minicpmv/Phi-4-reasoning-vision-15B-Q4_K_M.gguf", "minicpmv/mmproj-Phi-4-reasoning-vision-15B-f16.gguf", ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			text, proj := testmodels.Path(c.text), testmodels.Path(c.proj)
			if !sourcePresent(text, proj) {
				testmodels.Missing(t, "MODEL MISSING: %s and %s (RULE 11)", text, proj)
			}
			path := ""
			if c.config != "" {
				path = jlmWithConfig(t, text, proj, testmodels.Path(c.config))
			} else {
				path = jlmOfPair(t, text, proj)
			}
			m, err := model.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			seesThePicture(t, m, img)
		})
	}
}
