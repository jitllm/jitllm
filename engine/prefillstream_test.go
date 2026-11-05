package engine

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
)

// prefillStreamNMSE is the bound a device arm's logits are held to against
// the host's: the paged gates' order-only band on this model is ~2e-3, and a
// block whose history did not come home reads O(1).
const prefillStreamNMSE = 1e-2

// TestPrefillStreamsHostBlocks runs a three-and-a-half-chunk prompt on a card that holds
// half of the model: forced on, the prompt's host blocks stream through the
// card and come home after it, and the prompt's logits and eight decoded tokens
// -- which read the history the streamed blocks wrote on the card and brought
// home -- stay with the host's; forced off nothing streams; on auto exactly one
// decision is taken.
func TestPrefillStreamsHostBlocks(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	path := jlmOf(t, p)
	probe, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	probeLayers := probe.Cfg.NLayer
	// Three device chunks and part of a fourth: auto measures the first two.
	n := 3*nn.MaxDevicePrefillChunk + nn.MaxDevicePrefillChunk/2
	prompt := probe.Vocab.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog, and then it runs. ", n/10), true)[:n]
	const gen = 8
	// Half the model and a few blocks' room: the head and a few blocks resident,
	// the rest home, and the slots a streamed block pages through. The budget
	// is the card's whole charge, scratch and KV pool with the weights, so how
	// many blocks stay is the build's and the card's: a 3050 Ti's plain build
	// charged 363 MB of 401 MB at half the model and left no slot for a 39 MB
	// block. The gate takes the least room, in blocks, at which the prompt
	// streams.
	specFor := func(room uint64) string {
		return testdevices.Like(t, "cuda:0") + "=" +
			strconv.FormatUint(probe.WeightBytes()/2+room*probe.WeightBytes()/uint64(probeLayers), 10)
	}
	spec := specFor(0)
	host := testInstance(t, probe, len(prompt)+gen+1)
	want := []([]float32){}
	lg, err := host.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	want = append(want, append([]float32(nil), lg...))
	ids := make([]int32, 0, gen)
	for i := 0; i < gen; i++ {
		id := Greedy(lg)
		ids = append(ids, id)
		if lg, err = host.Forward(id); err != nil {
			t.Fatal(err)
		}
		want = append(want, append([]float32(nil), lg...))
	}
	host.Close()
	probe.Close()

	// run returns why, not an error, where the budget did not leave the
	// configuration the gate tests: no blocks left on the host or none on
	// the card. done frees the arm.
	type arm struct {
		st       *Instance
		g        *gpu.GPU
		resident int
		got      [][]float32
		done     func()
	}
	run := func(t *testing.T, mode PrefillStream) (a arm, why string) {
		g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
		if err != nil || g == nil {
			noDevice(t, spec, err)
		}
		instOpts := []Option{WithPrefillStream(mode)}
		m, err := model.Open(path)
		if err != nil {
			g.Close()
			t.Fatal(err)
		}
		st := testInstance(t, m, len(prompt)+gen+1, instOpts...)
		var once sync.Once
		done := func() { once.Do(func() { st.Close(); g.Close(); m.Close() }) }
		t.Cleanup(done)
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		resident := st.devCount()
		if resident == 0 || resident == m.Cfg.NLayer {
			return arm{done: done}, fmt.Sprintf("%d of %d blocks resident: the card does not leave blocks on the host, so nothing "+
				"could stream", resident, m.Cfg.NLayer)
		}
		lg, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		got := [][]float32{append([]float32(nil), lg...)}
		// No streamed block stays on the device. Fewer than before is the
		// history outgrowing the card, which relocation answers as it always
		// has.
		if st.devCount() > resident {
			t.Fatalf("after the prompt %d blocks are on the device, %d before it: the streamed blocks "+
				"did not come home", st.devCount(), resident)
		}
		for _, id := range ids {
			if lg, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
			got = append(got, append([]float32(nil), lg...))
		}
		return arm{st, g, resident, got, done}, ""
	}
	check := func(t *testing.T, got [][]float32) {
		t.Helper()
		for i := range want {
			if e := logitNMSE(got[i], want[i]); !(e < prefillStreamNMSE) {
				t.Fatalf("step %d: logit NMSE %.3e against the host, bound %.0e", i, e, prefillStreamNMSE)
			}
		}
	}

	t.Run("on", func(t *testing.T) {
		const rooms = 4
		var a arm
		for room := uint64(0); ; room++ {
			spec = specFor(room)
			var why string
			if a, why = run(t, PrefillStreamOn); why == "" {
				if s, k := a.st.PrefillStreams(); s != 1 || k != 0 {
					why = fmt.Sprintf("%d prompt(s) streamed, %d kept on the host: the forced stream did not run", s, k)
				} else if ps := a.g.Stats(); ps.PageIns == 0 {
					why = fmt.Sprintf("no page-in: nothing streamed through the card (%d of %d blocks resident, %d after; "+
						"budget %d of %d used; page-outs %d, relocations %d, %q)", a.resident, probeLayers, a.st.devCount(),
						a.g.Bytes(), a.g.Budgets()[0].Limit, ps.PageOuts, a.st.Relocations(), a.g.Err())
				}
			}
			if why == "" {
				t.Logf("a card of %s: half the model and %d block(s) of room", spec, room)
				break
			}
			a.done()
			t.Logf("%s: %s", spec, why)
			if room == rooms-1 {
				t.Fatalf("no budget from half the model to %d blocks more streamed the prompt", rooms-1)
			}
		}
		check(t, a.got)
		t.Logf("%d blocks resident, the rest streamed for a %d-token prompt: %d page-ins, back to %d after",
			a.resident, len(prompt), a.g.Stats().PageIns, a.st.devCount())
	})
	t.Run("off", func(t *testing.T) {
		a, why := run(t, PrefillStreamOff)
		if why != "" {
			t.Fatal(why)
		}
		if s, k := a.st.PrefillStreams(); s != 0 || k != 0 {
			t.Fatalf("forced off, %d prompt(s) streamed and %d measured", s, k)
		}
		check(t, a.got)
	})
	t.Run("auto", func(t *testing.T) {
		a, why := run(t, PrefillStreamAuto)
		if why != "" {
			t.Fatal(why)
		}
		s, k := a.st.PrefillStreams()
		if s+k != 1 {
			t.Fatalf("auto took %d decision(s) on one prompt (%d streamed, %d kept)", s+k, s, k)
		}
		check(t, a.got)
		t.Logf("auto: streamed %d, kept on the host %d", s, k)
	})
}
