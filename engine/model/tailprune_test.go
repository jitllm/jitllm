package model

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// tailRun is one prompt through a State: the prefill logits, a decode run
// after it, the host KV pages and the tail counters.
type tailRun struct {
	pre, dec []float32
	kv       [][]float32
	tap      []float32
	would    int
	ran      int
}

// tailPrompt is a prompt of n tokens inside any fixture's vocabulary.
func tailPrompt(m *Model, n int) []int32 {
	p := make([]int32, n)
	for i := range p {
		p[i] = int32((i*37 + 5) % min(m.Cfg.NVocab, 200))
	}
	return p
}

// tailExec prefills a prompt in chunks of chunk rows (host) and decodes 6
// tokens. dev is the number of blocks placed on the device named by spec ("" is
// the host alone); tap installs a row tap that records every row's residual.
func tailExec(t *testing.T, path string, prune bool, chunk, rows, dev int, spec string, tap bool) tailRun {
	t.Helper()
	m, err := Open(jlmOf(t, path), noTune, WithKVF16(false), WithTailPrune(prune),
		WithJITOptions(nn.WithPrefillChunk(chunk), nn.WithTune(nn.TuneOff)))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	st := m.NewState(rows + 16)
	defer st.Close()
	if spec != "" {
		g, err := openTier(t, tier.WithDevices(spec))
		if err != nil || g == nil {
			t.Fatalf("%s: %v", spec, err)
		}
		defer g.Close()
		if err := st.SetDeviceLayers(g, dev); err != nil || st.GPULayers() == 0 {
			t.Fatalf("%d blocks on %s: %v (%d placed): %s", dev, spec, err, st.GPULayers(), g.Err())
		}
	}
	var r tailRun
	if tap {
		st.rowTap = func(_ int, x []float32) { r.tap = append(r.tap, x...) }
	}
	lg, err := st.Prefill(tailPrompt(m, rows))
	if err != nil {
		t.Fatal(err)
	}
	r.pre = slices.Clone(lg)
	st.rowTap = nil
	for i := 0; i < 6; i++ {
		lg, err = st.Forward(int32(11 + i))
		if err != nil {
			t.Fatal(err)
		}
		r.dec = append(r.dec, lg...)
	}
	if spec == "" {
		for li := range st.kv.layers {
			var all []float32
			for _, p := range st.kv.layers[li].k {
				all = append(all, p...)
			}
			for _, p := range st.kv.layers[li].v {
				all = append(all, p...)
			}
			r.kv = append(r.kv, all)
		}
	}
	r.would, r.ran = st.TailPruneRowBlocks()
	return r
}

func tailModelPath(t *testing.T, name string) string {
	p, ok := existingModel(testmodels.Path(name + ".gguf"))
	if !ok {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path(name+".gguf"))
	}
	return p
}

func tailEqual(t *testing.T, what string, a, b tailRun) {
	t.Helper()
	if !slices.Equal(a.pre, b.pre) {
		t.Errorf("%s: prefill logits differ", what)
	}
	if !slices.Equal(a.dec, b.dec) {
		t.Errorf("%s: decode logits after the prompt differ", what)
	}
	if len(a.kv) != len(b.kv) {
		t.Fatalf("%s: %d against %d KV layers", what, len(a.kv), len(b.kv))
	}
	for li := range a.kv {
		if !slices.Equal(a.kv[li], b.kv[li]) {
			t.Errorf("%s: block %d's live KV differs", what, li)
		}
	}
}

// TestTailPruneKeepsTheAnswer holds a pruned prompt to the unpruned one: the
// logits, the decode after it and every block's live KV, across chunk
// boundaries, with the counter proving the row-blocks fell P*T to T.
func TestTailPruneKeepsTheAnswer(t *testing.T) {
	const rows = 40
	for _, name := range []string{"synth-gemma4-e", "synth-gemma4-e-ragged"} {
		path := tailModelPath(t, name)
		for _, chunk := range []int{7, 16, 64} {
			t.Run(name, func(t *testing.T) {
				a := tailExec(t, path, false, chunk, rows, 0, "", false)
				b := tailExec(t, path, true, chunk, rows, 0, "", false)
				tailEqual(t, name, a, b)
				m, err := Open(jlmOf(t, path), noTune)
				if err != nil {
					t.Fatal(err)
				}
				T := int(m.Cfg.NKVShared)
				m.Close()
				if T == 0 {
					t.Fatal("the fixture shares no KV: the gate covers nothing")
				}
				if a.would != 0 || a.ran != 0 {
					t.Errorf("the control pruned: %d, %d", a.would, a.ran)
				}
				if b.would != rows*T || b.ran != T {
					t.Errorf("row-blocks %d -> %d, want %d -> %d", b.would, b.ran, rows*T, T)
				}
			})
		}
	}
}

// TestTailPruneKeepsWhatIsConsumed: a row tap wants every row's output, so
// nothing is pruned and the rows are the unpruned ones; the violation, a
// prune that ignores the tap, changes them.
func TestTailPruneKeepsWhatIsConsumed(t *testing.T) {
	path := tailModelPath(t, "synth-gemma4-e")
	a := tailExec(t, path, false, 7, 40, 0, "", true)
	b := tailExec(t, path, true, 7, 40, 0, "", true)
	if b.would != 0 || b.ran != 0 {
		t.Errorf("a tap's rows were pruned: %d, %d", b.would, b.ran)
	}
	if len(a.tap) == 0 || !slices.Equal(a.tap, b.tap) {
		t.Errorf("the tapped rows differ (%d, %d values)", len(a.tap), len(b.tap))
	}
	tailPruneIgnoresDemand = true
	v := tailExec(t, path, true, 7, 40, 0, "", true)
	tailPruneIgnoresDemand = false
	if v.ran == 0 || slices.Equal(a.tap, v.tap) {
		t.Errorf("the violation went unseen (ran %d): a pruned row that is needed is not detected", v.ran)
	}
}

// TestTailPruneDeclinesAltUp: AltUp's streams are laid out per chunk, so
// gemma3n is declined by name and runs every row.
func TestTailPruneDeclinesAltUp(t *testing.T) {
	path := tailModelPath(t, "synth-gemma3n")
	b := tailExec(t, path, true, 7, 40, 0, "", false)
	if b.would != 0 || b.ran != 0 {
		t.Errorf("AltUp pruned: %d, %d", b.would, b.ran)
	}
}
