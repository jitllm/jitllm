package tier_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestDecodeRowTileAgreesWithPlainRows drives a REAL container through the
// device twice -- one row per thread, then a tile of four ADJACENT rows -- and
// compares the token ids.
//
// The kernel gates hold the tiled matvec to an oracle but never open a model,
// so they cannot catch the tier handing the tile a row count it does not
// divide, a grid still sized for one row per thread, or a mis-sized partial
// buffer. All three are silent, since surplus threads clamp to the last item.
// It also asserts the configuration was selected: the tile once reached zero
// matvecs because tuneSplit chose a split first, so it demands
// RowtSplitTiles > 0 and a control arm with no tiles.
func TestDecodeRowTileAgreesWithPlainRows(t *testing.T) {
	var src string
	for _, name := range []string{
		"Llama-3.2-1B-Instruct-Q4_K_M.gguf",
		"tinyllama-1.1b-q3_K_M.gguf",
		"Qwen3-MOE-4x0.6B-Q4_K_M.gguf",
		"gemma-2b.gguf",
	} {
		p := testmodels.Path(name)
		if _, err := os.Stat(p); err == nil {
			src = p
			break
		}
	}
	if src == "" {
		t.Skip("MODEL MISSING: none of the candidates are in " + testmodels.Dir() + " (set JITLLM_MODELS to the model directory) -- this gate proved nothing")
	}
	t.Logf("model %s", filepath.Base(src))
	dst := filepath.Join(t.TempDir(), "m"+jlm.Ext)
	if _, err := convert.FromGGUF(src, dst, jlm.Fingerprint{Host: "test"}); err != nil {
		t.Fatalf("convert: %v", err)
	}

	const prompt = "The capital of France is"
	const n = 16

	run := func(rowt int) ([]int32, tier.Stats, int) {
		m, err := model.Open(dst)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		// The split tuner is pinned off: it measures at load, so the split
		// and the f32 reduction order would vary per process. Pinned, the
		// arms differ only in which thread owns which row.
		g, err := tier.OpenWith(
			tier.WithDevices("gpu:0"),
			tier.WithDeviceTune(tier.TuneOff),
			tier.WithConfig(func(c *tier.Config) { c.DecodeRowt = rowt }),
		)
		if err != nil {
			t.Skipf("no accelerator: %v -- this gate proved nothing", err)
		}
		defer g.Close()
		ids := m.Vocab.Encode(prompt, true)
		st := m.NewState(len(ids) + n + 1)
		defer st.Close()
		st.SetDeviceLayers(g, -1)
		placed := st.GPULayers()
		if placed == 0 {
			t.Skip("the device took no block -- this gate proved nothing")
		}
		logits, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]int32, 0, n)
		for i := 0; i < n; i++ {
			best, bi := float32(-1e30), int32(0)
			for j, v := range logits {
				if v > best {
					best, bi = v, int32(j)
				}
			}
			out = append(out, bi)
			if logits, err = st.Forward(bi); err != nil {
				t.Fatal(err)
			}
		}
		return out, g.Stats(), placed
	}

	plain, ps, placed := run(1)
	tiled, ts, _ := run(4)

	if ps.RowtTiles != 0 {
		t.Fatalf("the CONTROL arm built %d tile(s) at DecodeRowt=1 -- "+
			"the two arms are not different configurations", ps.RowtTiles)
	}
	if ts.RowtTiles == 0 {
		t.Fatalf("DecodeRowt=4 reached NO matvec (%d plain) -- "+
			"the gate would have passed on a knob that never ran", ts.RowtPlain)
	}
	if ts.RowtSplitTiles == 0 {
		t.Fatalf("DecodeRowt=4 built %d tile(s) and NONE of them with a k-split -- "+
			"the combination this gate exists for was never selected", ts.RowtTiles)
	}
	t.Logf("%d block(s) placed; tiles %d of %d decode matvecs, %d of them split",
		placed, ts.RowtTiles, ts.RowtTiles+ts.RowtPlain, ts.RowtSplitTiles)

	// Token ids, exactly: with the tuner pinned both arms perform the same
	// reduction in the same order, so a near-tie allowance would only hide a
	// real disagreement.
	for i := range plain {
		if plain[i] != tiled[i] {
			t.Fatalf("token %d: plain %d, tiled %d\n  plain %v\n  tiled %v",
				i, plain[i], tiled[i], plain, tiled)
		}
	}
	t.Logf("%d ids identical", len(plain))
}
