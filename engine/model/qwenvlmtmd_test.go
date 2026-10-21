package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestQwenVLMatchesLlamaCppTeacherForced runs a real Qwen-VL checkpoint on a
// real picture through llama-mtmd-cli's own answer (scripts/qwenvlmtmd.py):
// the same GGUFs converted into one container, the prompt rendered by the
// model's own template, and llama.cpp's greedy reply teacher-forced, every
// position's argmax held to llama.cpp's token or to a tie within tieMargin --
// on the host and with the tower and the text model on each device.
//
// Teacher forcing is what makes the agreement countable: free-running, one
// tie parts the two transcripts for good. The preprocessing differs by
// design -- llama.cpp resamples with its own bicubic, this engine with
// transformers' (PIL's, TestQwenPreprocessMatchesTheProcessor) -- which is a
// difference in the pixels, not in the model. Where transformers settles a
// disagreement it is named (realFirst): Qwen2-VL-2B's first token, "The" to
// transformers and this engine and "A" to llama.cpp, is such a case.
//
// Qwen2.5-VL-3B is not here: llama.cpp's answer leaves transformers' at the
// first token ("A" by 3.6 logits, where transformers and this engine say
// "The") and every later position is then conditioned on a word the reference
// never wrote, so the 3B is held to transformers instead
// (TestQwenVLRealMatchesTransformers).
func TestQwenVLMatchesLlamaCppTeacherForced(t *testing.T) {
	for _, c := range []struct {
		name, text, proj string
		// realFirst is transformers' own first token where llama.cpp's
		// differs from it (scripts/qwenvlreal.py); the engine is held to it.
		realFirst string
	}{
		{"Qwen2-VL-2B-Instruct", "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", "The"},
		{"Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf", ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			// One golden per picture: quad-and-disc.png as each engine
			// preprocesses it, and quad-and-disc-392.png, the same picture
			// already at transformers' own resize (PIL's BICUBIC to the
			// 392x392 smart_resize picks), which neither engine resizes again
			// -- so on it the two see the same pixels and only the models differ.
			goldens, err := filepath.Glob(filepath.Join("..", "..", "testdata", "golden", "qwenvl", "mtmd-"+c.name+"*.json"))
			if err != nil || len(goldens) == 0 {
				t.Skipf("GOLDEN MISSING: %v -- run scripts/qwenvlmtmd.py (RULE 11); this gate proved nothing", err)
			}
			text, proj := testmodels.Path(c.text), testmodels.Path(c.proj)
			if !vlmAvailable(text, proj) {
				t.Skipf("MODEL MISSING: %s and its mmproj (set JITLLM_MODELS) -- this gate proved nothing", text)
			}
			m, err := Open(jlmOfPair(t, text, proj), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			for _, gp := range goldens {
				raw, err := os.ReadFile(gp)
				if err != nil {
					t.Fatal(err)
				}
				var g struct {
					Prompt, Image, Reply string
				}
				if err := json.Unmarshal(raw, &g); err != nil {
					t.Fatal(err)
				}
				t.Run(g.Image, func(t *testing.T) { qvlTeacherForce(t, m, g.Prompt, g.Image, g.Reply, c.realFirst) })
			}
		})
	}
}

// qvlTeacherForce is one golden of TestQwenVLMatchesLlamaCppTeacherForced:
// the host arm, then each device's.
func qvlTeacherForce(t *testing.T, m *Model, prompt, image, replyText, realFirst string) {
	reply := m.Vocab.Encode(replyText, false)
	arm := func(t *testing.T, gpu *tier.GPU) {
		tw := m.Tower()
		ts := tw.testState()
		defer ts.Close()
		if gpu != nil {
			ts.SetDevice(gpu)
			if ts.GPUBlocks() != tw.Cfg.NLayer {
				t.Fatalf("the device took %d of %d tower blocks: %v", ts.GPUBlocks(), tw.Cfg.NLayer, gpu.Err())
			}
		}
		f, err := os.Open(filepath.Join("testdata", image))
		if err != nil {
			t.Fatal(err)
		}
		px, err := ts.Preprocess(f)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		emb, err := ts.Encode(px)
		if err != nil {
			t.Fatal(err)
		}
		spans, err := m.ChatSpansImages([]ChatMessage{{Role: "user", Content: prompt, Images: 1}},
			[]Image{{Embd: emb, Grid: ts.Grid()}}, true)
		if err != nil {
			t.Fatal(err)
		}
		st := m.NewState(SpanPositions(spans, m.Cfg.NEmbd) + len(reply) + 2)
		defer st.Close()
		if gpu != nil {
			st.SetDevice(gpu)
			if st.GPULayers() == 0 {
				t.Fatalf("the device took no text block: %v", gpu.Err())
			}
		}
		lg, err := st.PrefillMixed(spans...)
		if err != nil {
			t.Fatal(err)
		}
		agree, ties := 0, 0
		var ours []int32
		for i, want := range reply {
			got := Greedy(lg)
			ours = append(ours, got)
			switch {
			case got == want:
				agree++
			case lg[got]-lg[want] <= tieMargin:
				ties++
				t.Logf("position %d: %q against llama.cpp's %q, a tie of %.3f", i,
					m.Vocab.Decode([]int32{got}), m.Vocab.Decode([]int32{want}), lg[got]-lg[want])
			case i == 0 && realFirst != "" && m.Vocab.Decode([]int32{got}) == realFirst:
				t.Logf("position 0: %q against llama.cpp's %q by %.3f -- transformers' own first "+
					"token is %q", m.Vocab.Decode([]int32{got}), m.Vocab.Decode([]int32{want}),
					lg[got]-lg[want], realFirst)
			default:
				t.Errorf("position %d: %q against llama.cpp's %q by %.3f, past the %.1f tie margin", i,
					m.Vocab.Decode([]int32{got}), m.Vocab.Decode([]int32{want}), lg[got]-lg[want], tieMargin)
			}
			if lg, err = st.Forward(want); err != nil {
				t.Fatal(err)
			}
		}
		t.Logf("%dx%d merged grid, %d of %d tower and %d of %d text blocks placed; %d of %d "+
			"teacher-forced positions agree with llama-mtmd-cli, %d ties\n  llama.cpp %q\n  jitllm    %q",
			ts.Grid().W, ts.Grid().H, ts.GPUBlocks(), tw.Cfg.NLayer, st.GPULayers(), m.Cfg.NLayer,
			agree, len(reply), ties, replyText, m.Vocab.Decode(ours))
	}
	t.Run("host", func(t *testing.T) { arm(t, nil) })
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			gpu, err := openTier(t, tier.WithDevices(spec))
			if err != nil || gpu == nil {
				t.Skipf("%s: not present (%v)", spec, err)
			}
			defer gpu.Close()
			arm(t, gpu)
		})
	}
}
