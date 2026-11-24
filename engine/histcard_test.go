package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestBinary16HistoryCrossesTheCard writes a binary16 history on the CPU,
// moves every block to a card and home again, and holds every position
// to what the CPU wrote, bit for bit: binary16 crosses unrounded both ways.
// The step arm runs one position on the card between the moves, and the
// positions before it must still be the CPU's: the card's write of the
// next position leaves the moved ones alone, and its logits are the CPU's
// own step's, within the band two devices' reduction orders differ by. A
// model with an ordinary attention history (not latent, not recurrent) is
// the case.
func TestBinary16HistoryCrossesTheCard(t *testing.T) {
	for _, name := range []string{"tinyllama-1.1b-q3_K_M.gguf", "synth-falconh1.gguf", "synth-glm4.gguf"} {
		for _, step := range []bool{false, true} {
			arm := name + "/move"
			if step {
				arm = name + "/step"
			}
			t.Run(arm, func(t *testing.T) { binary16HistoryCrossesTheCard(t, jlmOf(t, testmodels.Path(name)), step) })
		}
	}
}

func binary16HistoryCrossesTheCard(t *testing.T, path string, step bool) {
	g, err := openGPU(t, gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	in := m.Vocab.Encode("The capital of France is", true)
	st := testInstance(t, m, len(in)+8, WithKVF16(true))
	defer st.Close()
	if _, err := st.Prefill(in); err != nil {
		t.Fatal(err)
	}
	if st.kvl.fmt != nn.KVF16 {
		t.Fatalf("the instance's history is %v, not binary16", st.kvl.fmt)
	}
	want, _ := histories(t, st)
	pos := st.Pos()
	if err := st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if st.GPULayers() == 0 {
		t.Fatal("the card took no block: nothing crossed")
	}
	if step {
		ref := testInstance(t, m, len(in)+8, WithKVF16(true))
		defer ref.Close()
		if _, err := ref.Prefill(in); err != nil {
			t.Fatal(err)
		}
		wantL, err := ref.Forward(in[len(in)-1])
		if err != nil {
			t.Fatal(err)
		}
		wantL = append([]float32(nil), wantL...)
		gotL, err := st.Forward(in[len(in)-1])
		if err != nil {
			t.Fatal(err)
		}
		if e, worst := nmseOf(gotL, wantL); !(e < 1e-3) {
			t.Errorf("the card's step over the moved history is NMSE %.3g (worst %.3g) from the CPU's: it reads the history wrongly", e, worst)
		} else {
			t.Logf("the card's step over the moved history: NMSE %.3g from the CPU's", e)
		}
	}
	if err := st.SetDevice(nil); err != nil {
		t.Fatal(err)
	}
	got, _ := histories(t, st)
	if len(got) != len(want) {
		t.Fatalf("%d histories came home, %d went", len(got), len(want))
	}
	for i := range want {
		per := len(want[i]) / pos
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("block %d's %s at position %d element %d came home %v, the CPU wrote %v (%d blocks on the card, step %v)",
					i/2, [2]string{"keys", "values"}[i%2], j/per, j%per, got[i][j], want[i][j], st.GPULayers(), step)
			}
		}
	}
}

// TestCardWrittenHistoryComesHome prefills on a card, which writes the
// binary16 history itself, and brings every block home: each block's keys
// and values are the CPU's own prefill's, within the band the two devices'
// projections differ by. The card's attention reads what its own writer
// wrote, so only a move can see a writer and a migrator that disagree on
// the pool's layout.
func TestCardWrittenHistoryComesHome(t *testing.T) {
	for _, name := range []string{"tinyllama-1.1b-q3_K_M.gguf", "synth-falconh1.gguf", "synth-glm4.gguf"} {
		t.Run(name, func(t *testing.T) { cardWrittenHistoryComesHome(t, jlmOf(t, testmodels.Path(name))) })
	}
}

func cardWrittenHistoryComesHome(t *testing.T, path string) {
	cardWrittenHistoryComesHomeAfter(t, path, false)
}

// TestSharedCardHistoryComesHome is TestCardWrittenHistoryComesHome for a
// second instance of the same prompt on the card, after a first closed:
// its prefix comes from the card's named pages (kvshare.go), and going home
// it brings their rows.
func TestSharedCardHistoryComesHome(t *testing.T) {
	for _, name := range []string{"synth-falconh1.gguf", "synth-glm4.gguf"} {
		t.Run(name, func(t *testing.T) { cardWrittenHistoryComesHomeAfter(t, jlmOf(t, testmodels.Path(name)), true) })
	}
}

func cardWrittenHistoryComesHomeAfter(t *testing.T, path string, after bool) {
	g, err := openGPU(t, gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	in := m.Vocab.Encode("The capital of France is", true)
	ref := testInstance(t, m, len(in)+8, WithKVF16(true))
	defer ref.Close()
	if _, err := ref.Prefill(in); err != nil {
		t.Fatal(err)
	}
	want, _ := histories(t, ref)
	if after {
		first := testInstance(t, m, len(in)+8, WithKVF16(true))
		if err := first.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if _, err := first.Prefill(in); err != nil {
			t.Fatal(err)
		}
		first.Close()
	}
	st := testInstance(t, m, len(in)+8, WithKVF16(true))
	defer st.Close()
	if err := st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	n := st.GPULayers()
	if n == 0 {
		t.Fatal("the card took no block: it wrote nothing")
	}
	mapped := g.Stats().KVPagesMapped
	if _, err := st.Prefill(in); err != nil {
		t.Fatal(err)
	}
	t.Logf("the prefill ran %d of %d rows, mapping %d named page(s)", st.PrefillRows(), len(in), g.Stats().KVPagesMapped-mapped)
	if err := st.SetDevice(nil); err != nil {
		t.Fatal(err)
	}
	got, _ := histories(t, st)
	if len(got) != len(want) {
		t.Fatalf("%d histories came home, %d were written", len(got), len(want))
	}
	pos := st.Pos()
	for i := range want {
		per := len(want[i]) / pos
		for p := range pos {
			// A real checkpoint's blocks deep in the stack carry the band
			// grown through the blocks below (RULE 11c); a layout fault is
			// orders past it.
			e, worst := nmseOf(got[i][p*per:(p+1)*per], want[i][p*per:(p+1)*per])
			if !(e < 1e-2) {
				t.Errorf("block %d's %s at position %d came home NMSE %.3g (worst %.3g) from the CPU's (%d blocks on the card): %v against %v",
					i/2, [2]string{"keys", "values"}[i%2], p, e, worst, n, got[i][p*per:p*per+4], want[i][p*per:p*per+4])
			}
		}
	}
	t.Logf("%d blocks' histories of %d positions came home within the band", n, pos)
}

// TestHistoryCrossesTiers is TestLayerRelocatesToAnotherDevice's shape held
// to the CPU: a prefill on tier a, the sequence moved to tier b, one step
// there, its logits the CPU's within the band. The same arm opens b on a's
// card, the other on a second card, and the stay arm does not move: the
// step runs on the tier that wrote the history. The after arm first runs
// and closes another instance of the same prompt on tier a, as the
// relocation gate's static run does, then hops; afterstay does not hop.
func TestHistoryCrossesTiers(t *testing.T) {
	for _, name := range []string{"synth-falconh1.gguf", "synth-glm4.gguf"} {
		for _, arm := range []string{"same", "other", "stay", "after", "afterstay"} {
			t.Run(name+"/"+arm, func(t *testing.T) { historyCrossesTiers(t, jlmOf(t, testmodels.Path(name)), arm, WithKVF16(true)) })
			// The instance's own default format, as the relocation gates
			// make it.
			t.Run(name+"/"+arm+"/default", func(t *testing.T) { historyCrossesTiers(t, jlmOf(t, testmodels.Path(name)), arm) })
		}
	}
}

func historyCrossesTiers(t *testing.T, path, arm string, kv ...Option) {
	spec := testdevices.One("cuda:0")
	a, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || a == nil {
		noDevice(t, spec, err)
	}
	defer a.Close()
	if arm == "other" {
		// A second card the selection lists: one past a's.
		var cards []string
		for _, c := range testdevices.Within([]string{"cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"}) {
			if c != spec {
				cards = append(cards, c)
			}
		}
		if len(cards) == 0 {
			noDevice(t, "a second CUDA card", nil)
		}
		spec = cards[0]
	}
	b, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || b == nil {
		noDevice(t, spec, err)
	}
	defer b.Close()
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	in := m.Vocab.Encode("The capital of France is", true)
	ref := testInstance(t, m, len(in)+8, kv...)
	defer ref.Close()
	if _, err := ref.Prefill(in); err != nil {
		t.Fatal(err)
	}
	wantL, err := ref.Forward(in[len(in)-1])
	if err != nil {
		t.Fatal(err)
	}
	wantL = append([]float32(nil), wantL...)
	if arm == "after" || arm == "afterstay" {
		first := testInstance(t, m, len(in)+8, kv...)
		if err := first.SetDevice(a); err != nil {
			t.Fatal(err)
		}
		if _, err := first.Prefill(in); err != nil {
			t.Fatal(err)
		}
		if _, err := first.Forward(in[len(in)-1]); err != nil {
			t.Fatal(err)
		}
		first.Close()
	}
	st := testInstance(t, m, len(in)+8, kv...)
	defer st.Close()
	if err := st.SetDevice(a); err != nil {
		t.Fatal(err)
	}
	if _, err := st.Prefill(in); err != nil {
		t.Fatal(err)
	}
	if arm != "stay" && arm != "afterstay" {
		if err := st.SetDevice(b); err != nil {
			t.Fatal(err)
		}
	}
	if st.GPULayers() == 0 {
		t.Fatal("the card took no block: nothing ran there")
	}
	gotL, err := st.Forward(in[len(in)-1])
	if err != nil {
		t.Fatal(err)
	}
	if e, worst := nmseOf(gotL, wantL); !(e < 1e-3) {
		t.Errorf("%s, history %v: the step is NMSE %.3g (worst %.3g) from the CPU's", arm, st.kvl.fmt, e, worst)
	} else {
		t.Logf("%s, history %v: the step is NMSE %.3g from the CPU's", arm, st.kvl.fmt, e)
	}
}
