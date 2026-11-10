package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
)

// TestBlocksMoveToTheCPUAndBack: the CPU is a device like the cards, and one
// MoveBlocks reaches it. Mid-sequence every block goes from one card to the
// CPU, from the CPU to the other card by name, half of them to the CPU again,
// and all of them to the first card -- the CPU half placed there, the card
// half moved card to card -- each with its history, teacher-forced on a
// CPU control's tokens. Every block is where it was sent (WhereIs and
// Locations name it), and every token stays in the band of the control.
//
// VIOLATION SIGNATURE. Drop the history from moveOffCards (no migrateKV) and the
// tokens after the first move leave the band.
func TestBlocksMoveToTheCPUAndBack(t *testing.T) {
	names := append([]string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf"}, relocModels()...)
	for _, name := range names {
		t.Run(name, func(t *testing.T) { blocksMoveToTheCPUAndBack(t, jlmOf(t, testmodels.Path(name)), false) })
	}
	// The CPU's blocks on what the composer forms for it: each block composed
	// for the CPU at its placement and again at each move back, and every
	// block step the CPU runs between the moves served.
	for _, name := range cpuServedModels() {
		t.Run(name+"/composed-cpu", func(t *testing.T) { blocksMoveToTheCPUAndBack(t, jlmOf(t, testmodels.Path(name)), true) })
	}
}

// composedMoves holds st's composed CPU census to the moves: on the control
// the prompt's and every decode step of every block served; on the moved
// run each block
// composed at the instance's placement, again at the move to the CPU and once
// more for the upper half, and the block steps between the moves served.
func composedMoves(t *testing.T, st *Instance, n, half int, devices bool, ntok, toCPU, toB, halfCPU, toA int) {
	t.Helper()
	served, declined := st.Account(nn.CPU).Steps.Of(nn.ComposedBlock)
	for i, k := range declined {
		if k > 0 {
			t.Errorf("%d block steps declined on the CPU: %s (%v)", k, nn.DeclineName(i), st.Account(nn.CPU).FormErr)
		}
	}
	if !devices {
		// The prompt's chunks, every block of each, and the decode's steps.
		if served < (ntok+1)*n || served%n != 0 {
			t.Errorf("the composed CPU served %d block steps, where the CPU ran the prompt and %d", served, ntok*n)
		}
		return
	}
	if want := (toB-toCPU)*n + (toA-halfCPU)*(n-half); served != want {
		t.Errorf("the composed CPU served %d block steps, where the CPU ran %d", served, want)
	}
	for li, k := range st.Account(nn.CPU).Forms {
		w := 2
		if li >= half {
			w = 3
		}
		if k != w {
			t.Errorf("block %d was composed for the CPU %d times, where it was placed there %d times", li, k, w)
		}
	}
}

func blocksMoveToTheCPUAndBack(t *testing.T, path string, composed bool) {
	a, b := moveDevices(t)
	g, err := openGPU(t, gpu.WithDevices(a+","+b), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, a+","+b, err)
	}
	defer g.Close()
	distinctCards(t, g)
	opt := WithKVF16(false)
	if composed {
		// The composed CPU keeps its history binary16 (composer.Halves).
		opt = WithKVF16(true)
	}
	instOpts := []Option{opt}
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	n := m.Cfg.NLayer
	const ntok, toCPU, toB, halfCPU, toA = 12, 2, 5, 8, 10
	prompt := m.Vocab.Encode("The capital of France is a city that", true)

	// where fails unless every block of [lo, hi) is on dev, by both reads.
	where := func(st *Instance, lo, hi int, dev string) {
		t.Helper()
		locs := st.Locations()
		for li := lo; li < hi; li++ {
			if got := st.WhereIs(li); got != nn.DeviceName(dev) || locs[li] != got {
				t.Fatalf("block %d is on %q (Locations %q), want %q", li, got, locs[li], dev)
			}
		}
	}
	move := func(st *Instance, lo, hi int, to string) BlockMove {
		t.Helper()
		mv, err := st.MoveBlocks(lo, hi, to)
		if err != nil {
			t.Fatalf("blocks %d..%d to %s: %v (%s)", lo, hi-1, to, err, g.Err())
		}
		if len(mv.Declined) > 0 {
			t.Fatalf("%s declined blocks %v: %s", to, mv.Declined, g.Err())
		}
		where(st, lo, hi, to)
		return mv
	}
	half := n / 2
	run := func(ids []int32, devices bool) [][]float32 {
		st := testInstance(t, m, len(prompt)+ntok+1, instOpts...)
		defer st.Close()
		if composed {
			defer func() { composedMoves(t, st, n, half, devices, ntok, toCPU, toB, halfCPU, toA) }()
		}
		if devices {
			if err := st.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			move(st, 0, n, a)
		} else {
			where(st, 0, n, nn.CPU)
		}
		logits, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		var lg [][]float32
		for i := 0; i < ntok; i++ {
			lg = append(lg, append([]float32(nil), logits...))
			next := Greedy(logits)
			if ids != nil {
				next = ids[i]
			}
			if devices {
				switch i {
				case toCPU:
					if mv := move(st, 0, n, nn.CPU); len(mv.Moved) != n || st.devCount() != 0 {
						t.Fatalf("%d blocks moved to the CPU, %d left on a card", len(mv.Moved), st.devCount())
					}
				case toB:
					if mv := move(st, 0, n, b); len(mv.Moved) != n {
						t.Fatalf("%d of %d blocks moved from the CPU to %s", len(mv.Moved), n, b)
					}
				case halfCPU:
					// A KV-sharing group the cut splits comes to the CPU
					// whole (fixKVGroups), so the cut is where none crosses.
					for li := 0; li < n; li++ {
						if glo, ghi := st.kvSpan(li); glo < half && ghi > half {
							half, li = glo, -1
						}
					}
					move(st, half, n, nn.CPU)
					where(st, 0, half, b)
				case toA:
					if mv := move(st, 0, n, a); len(mv.Moved) != n {
						t.Fatalf("%d of %d blocks moved from the CPU and %s to %s", len(mv.Moved), n, b, a)
					}
				}
			}
			if logits, err = st.Forward(next); err != nil {
				t.Fatal(err)
			}
		}
		return lg
	}
	ctl := run(nil, false)
	var ids []int32
	for _, l := range ctl {
		ids = append(ids, Greedy(l))
	}
	moved := run(ids, true)
	// The card parts from the CPU in the reduction-order band, which an
	// absolute logit difference cannot state: it grows with the logits'
	// scale (mamba-130m's reach 43, and a coherent prefill reads 3.18 apart).
	// The band is the card census's; a history a move dropped or misplaced
	// reads in tenths at every token after it. A mixture's routed bank is
	// computed on the card in arithmetic of the card's own, within its band,
	// and that can carry the next block's router across a near-tie: one
	// token takes another expert and parts in tenths, the next agrees again
	// (synth-qwen35moe-hybrid-mtp-q8_0's tokens 2 and 4, 0.117 and 0.089,
	// the rest 1e-3). A mixture is held at the median of its tokens, and
	// every token below 1.
	const moveBand, mixtureWorst = 5e-2, 1.0
	mixture := m.Cfg.NExpert > 1
	var worstAll, nmseAll float64
	var es []float64
	for i := range ctl {
		var worst float64
		for j, v := range ctl[i] {
			worst = max(worst, math.Abs(float64(v-moved[i][j])))
		}
		e, _ := nmseOf(moved[i], ctl[i])
		es = append(es, e)
		switch {
		case math.IsNaN(worst) || math.IsNaN(e):
			t.Fatalf("token %d: a non-finite logit", i)
		case !mixture && !(e < moveBand):
			t.Fatalf("token %d: %.3e NMSE from the CPU control (worst logit %.4g), outside the band %.0e", i, e, worst, moveBand)
		case mixture && !(e < mixtureWorst):
			t.Fatalf("token %d: the mixture's %.3e NMSE from the CPU control (worst logit %.4g), past %.0g at any token", i, e, worst, mixtureWorst)
		}
		worstAll, nmseAll = max(worstAll, worst), max(nmseAll, e)
	}
	if med := median(es); mixture && !(med < moveBand) {
		t.Fatalf("the mixture's tokens are %.3e NMSE from the CPU control at the median, outside the band %.0e", med, moveBand)
	}
	t.Logf("%d tokens, %d blocks moved %s -> cpu -> %s -> half cpu -> %s; worst %.3e NMSE (logit %.4g), median %.3e, from the CPU control",
		ntok, n, a, b, a, nmseAll, worstAll, median(es))
}
