package engine

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestACardOfTwoRunsTakesItsPiecesInTurn places a model's first and last
// thirds on one card and its middle on another, as a block spilled past a
// full card leaves them, and prefills a prompt of three device chunks. The
// card holding two runs is one tier for the session, which the pipeline
// would drive from two goroutines at once (a submission's staging bytes and
// its running function are the session's, one a card); its runs take the
// pieces one after another instead (GPU.pieces, Stats.SplitPieces), and no
// piece is pipelined. The prompt's logits and four decode steps after it are
// held to the host's. Two physical cards where JITLLM_TEST_DEVICES names
// them.
func TestACardOfTwoRunsTakesItsPiecesInTurn(t *testing.T) {
	d := testdevices.Take(t, 2, "cuda:0,vulkan:1")
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode(strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 60), true)[:1300]
	run := func(st *Instance) [][]float32 {
		lg, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		out := [][]float32{append([]float32(nil), lg...)}
		for range 4 {
			if lg, err = st.Forward(Greedy(lg)); err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	instOpts := []Option{WithKVF16(false)}
	host := testInstance(t, m, len(ids)+8, instOpts...)
	want := run(host)
	host.Close()

	g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(d[0] + "," + d[1])}, testTierOpts(t)...)...)
	if err != nil || g == nil {
		noDevice(t, d[0]+","+d[1], err)
	}
	defer g.Close()
	nb := m.Cfg.NLayer
	blocks := map[int]Place{}
	for li := range nb {
		on := d[0]
		if li >= nb/3 && li < 2*nb/3 {
			on = d[1]
		}
		blocks[li] = Place{On: on}
	}
	st := testInstance(t, m, len(ids)+8, append(instOpts, WithStreamTrial(false),
		WithPlacement(Placement{Blocks: blocks, Head: &Place{On: d[0]}, Strict: true}))...)
	defer st.Close()
	if err := st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if st.GPULayers() != nb {
		t.Fatalf("%d of %d blocks on the cards (%v): the gate needs them all", st.GPULayers(), nb, g.Placed())
	}
	got := run(st)
	var piped, inTurn int
	for _, s := range g.DevStats() {
		piped += s.PipelinePieces
		inTurn += s.SplitPieces
	}
	if piped != 0 || inTurn == 0 {
		t.Fatalf("%v blocks: %d pieces pipelined, %d taken in turn; a card holding two runs must take them in turn",
			g.Placed(), piped, inTurn)
	}
	worst, worstD := worstOf(want, got)
	t.Logf("%v blocks; %d pieces taken in turn; worst logit NMSE %.3e, |dlogit| %.4f", g.Placed(), inTurn, worst, worstD)
	if !(worst < mlaDeviceNMSE) {
		t.Fatalf("a prompt over a card of two runs disagrees with the host: NMSE %.3e, |dlogit| %.3f", worst, worstD)
	}
}
