package model

import (
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestSchedulerOnSplitPlacementsMatchesEachSequenceAlone runs the Scheduler
// with the model split: half its blocks on a device and half on the host, and
// spread over two devices. ForwardBatch runs the host's blocks itself and hands
// each run of device blocks over as rows with their own slots; PrefillSeq does
// the same with the prompt as rows of one sequence. Each sequence must give
// exactly the tokens it gives alone, as a batch of one on the same placement.
func TestSchedulerOnSplitPlacementsMatchesEachSequenceAlone(t *testing.T) {
	schedulerOnSplit(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
}

// TestSchedulerOnSplitPlacementsHybrid is the same on a hybrid, whose prompt
// chunks step a device linear block's state as runs and a host one's a row at
// a time in position order.
func TestSchedulerOnSplitPlacementsHybrid(t *testing.T) {
	schedulerOnSplit(t, testmodels.Path("qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"))
}

func schedulerOnSplit(t *testing.T, p string) {
	t.Helper()
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	// The host's blocks run three rows wide here and one wide in the
	// reference; without noGEMM the first can take the weight-stationary GEMM,
	// which is not bit-identical to the matvec the second runs.
	m, err := Open(jlmOf(t, p), noGEMM)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompts := []string{
		"The capital of France is",
		"Once upon a time, in a small village by the sea, there lived",
		"1, 2, 3,",
		"The quick brown fox jumps over the lazy dog. The quick brown fox",
		"Water boils at",
	}
	gens := []int{12, 20, 8, 16, 10}
	// The longest prompt and generation here need under 40 positions: the rows
	// are sized to that, not to a context nothing uses, so a card's room goes
	// to blocks.
	const seq, rows = 64, 3

	ids := make([][]int32, len(prompts))
	for i, pr := range prompts {
		ids[i] = m.Vocab.Encode(pr, true)
	}
	open := func(t *testing.T, spec string, paging bool) *tier.GPU {
		g, err := openTier(t, append([]tier.Option{tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff)},
			testTierOpts(t)...)...)
		if err != nil || g == nil {
			noDevice(t, spec, err)
		}
		if paging {
			for li := 0; li < m.Cfg.NLayer; li++ {
				g.Stream(li, true)
			}
		}
		return g
	}

	half := m.Cfg.NLayer / 2
	// A card's budget holds more than weights -- the prompt chunk's scratch
	// reserved at placement, the driver's page rounding, the head, the history
	// and the recurrent states (tier/scratch.go) -- so a share of the model is
	// that share of the weights plus what the whole model costs a card beside
	// its weights.
	var sc uint64
	if g, err := openTier(t, tier.WithDevices(testdevices.Like(t, "cuda:0")), tier.WithDeviceTune(tier.TuneOff)); err == nil && g != nil {
		sz := NewScheduler(m, rows, seq)
		if err := sz.st.SetDeviceLayers(g, -1); err == nil && g.Stats().BudgetUsed > m.WeightBytes() {
			sc = g.Stats().BudgetUsed - m.WeightBytes()
		}
		sz.Close()
		g.Close()
	}
	share := func(n uint64) string { return strconv.FormatUint(sc+m.WeightBytes()/n, 10) }
	// Enough of cuda:0 for about a third of the model, so the rest spills.
	budget := share(3)
	// Four cards, each given a quarter of the blocks by name (and the host the
	// last two in the second arm). A budget per card cannot say this: the
	// share above charges every card the whole model's overhead, the head's
	// included, which only the tail card carries -- so on four cards
	// the first three took every block ([5 5 6 0]) and the gate refused a
	// fourth card that was never needed. The placer filling the fastest card
	// first is its design (TestLayerRelocatesToAnotherDevice); what this gate
	// asks is that the step runs across four devices, so it names them.
	four, fourWhy := fourDevices(t)
	quarters := func(onHost int) map[int]Place {
		pl := map[int]Place{}
		n := m.Cfg.NLayer - onHost
		for li := 0; li < m.Cfg.NLayer; li++ {
			if li >= n || len(four) < 4 {
				pl[li] = Place{On: "host"}
				continue
			}
			pl[li] = Place{On: four[li*4/n]}
		}
		return pl
	}
	fourSpec := strings.Join(four, ",")
	for _, arm := range []struct {
		name string
		spec string
		max  int // blocks to place; -1 is all
		devs int // devices that must hold blocks
		host bool
		// place, when set, is an explicit strict placement (WithPlacement),
		// checked block by block after the Scheduler's State takes it.
		place map[int]Place
		// paging turns device paging on: the budget holds a third of the
		// model and blocks swap through its slots every step.
		paging bool
	}{
		{"device-then-host", "gpu:0", half, 1, true, nil, false},
		// cuda:0 capped so it fills first and the rest spill to the second.
		{"two-devices", "cuda:0=" + budget + ",vulkan:1", -1, 2, false, nil, false},
		// Three tiers: the discrete card, the integrated GPU and the host.
		{"two-devices-and-host", "cuda:0=" + budget + ",vulkan:1", m.Cfg.NLayer - 2, 2, true, nil, false},
		// Four cards (fourDevices: CUDA's, then Vulkan's other cards), a
		// quarter of the blocks each, and then the same with the last two
		// blocks left on the host.
		{"four-devices", fourSpec, -1, 4, false, quarters(0), false},
		{"four-devices-and-host", fourSpec, -1, 4, true, quarters(2), false},
		// host, card, card, host, card: block 0 and the middle block on the
		// host, a small cuda:0 holding the next few and the second device the
		// rest -- an integrated GPU or another card.
		{"host-cuda-vulkan-host-vulkan", "cuda:0,vulkan:1", -1, 2, true, sandwich("vulkan:1"), false},
		{"host-cuda-cuda-host-cuda", "cuda:0,cuda:1", -1, 2, true, sandwich("cuda:1"), false},
		{"paged", "cuda:0=" + budget, -1, 1, false, nil, true},
	} {
		t.Run(arm.name, func(t *testing.T) {
			if arm.devs == 4 && len(four) < 4 {
				t.Skipf("%s -- this arm proved nothing here", fourWhy)
			}
			testdevices.Require(t, arm.spec)
			m := m
			if arm.place != nil {
				km, err := Open(jlmOf(t, p), noGEMM, WithPlacement(Placement{Blocks: arm.place, Strict: true}))
				if err != nil {
					t.Fatal(err)
				}
				defer km.Close()
				m = km
			}
			// Each sequence alone through the same batched path, one row on the
			// same placement. A solo decode runs other kernels, and through a
			// recurrent summary their last-bit differences grow past any tie
			// band within a few tokens. The batched path is not one kernel
			// either: a device picks a step's matvec by its row count (decode's
			// own at two to four rows, tier.groupMV, a tiled twin padded to its
			// grain otherwise), and a prompt rides in a step whose width the
			// other sequences set. So the tokens match exactly up to a first
			// difference, which must be a tie on the host within the device's
			// own band (tieMargin, as TestSchedulerOnDeviceMatchesEachSequenceAlone
			// bounds it): a row reading another row's history, or a chunk at the
			// wrong positions, parts by whole logits.
			want := make([][]int32, len(prompts))
			for i := range prompts {
				g := open(t, arm.spec, arm.paging)
				one := NewScheduler(m, 1, seq)
				if err := one.st.SetDeviceLayers(g, arm.max); err != nil {
					t.Fatal(err)
				}
				sq := &Seq{Prompt: ids[i], MaxTokens: gens[i]}
				one.Submit(sq)
				for steps := 0; !sq.Done; steps++ {
					if _, err := one.Step(); err != nil || steps > 200 {
						t.Fatalf("alone, row %d: %v after %d steps", i, err, steps)
					}
				}
				if sq.Err != nil {
					t.Fatalf("alone, row %d: %v", i, sq.Err)
				}
				want[i] = sq.Out
				one.Close()
				g.Close()
			}
			g := open(t, arm.spec, arm.paging)
			defer g.Close()
			sc := NewScheduler(m, rows, seq)
			defer sc.Close()
			if err := sc.st.SetDeviceLayers(g, arm.max); err != nil {
				t.Fatal(err)
			}
			placed := g.Placed()
			// The configuration under test must be the one that ran: a split,
			// not everything on one device or everything on the host.
			// The configuration under test must be the one that ran: every
			// named device holding blocks, and the host some when it should.
			if n := spread(placed); n < arm.devs {
				t.Fatalf("placed %v (head on device %d): %d devices hold blocks, want %d (%q)",
					placed, g.HeadDevice(), n, arm.devs, g.Err())
			}
			dev, _ := sc.st.ld.(nn.NamedDevice)
			for li, pl := range arm.place {
				got := "host"
				if n, ok := dev.DeviceOf(li); ok {
					got = n
				}
				if got != pl.On {
					t.Fatalf("block %d is on %s; the placement put it on %s", li, got, pl.On)
				}
			}
			if got := sc.st.devCount(); arm.host != (got < m.Cfg.NLayer) || got == 0 {
				t.Fatalf("placed %d of %d blocks on devices; host blocks wanted: %v (%q)",
					got, m.Cfg.NLayer, arm.host, g.Err())
			}
			if arm.paging {
				defer func() {
					if st := g.Stats(); st.PageIns == 0 {
						t.Errorf("paging arm: no page-ins, so every block fit and nothing paged")
					} else {
						t.Logf("paged: %d page-in(s), %d page-out(s)", st.PageIns, st.PageOuts)
					}
				}()
			}
			t.Logf("placed %v, %d of %d blocks in runs %v, head on a device: %v",
				placed, sc.st.devCount(), m.Cfg.NLayer, sc.st.devRuns(), sc.st.HeadOnDevice())
			seqs := make([]*Seq, len(prompts))
			for i := range prompts {
				seqs[i] = &Seq{Prompt: ids[i], MaxTokens: gens[i]}
				sc.Submit(seqs[i])
			}
			for retired, steps := 0, 0; retired < len(seqs); steps++ {
				if steps > 200 {
					for i, s := range seqs {
						t.Logf("row %d: done %v, %d tokens, err %v", i, s.Done, len(s.Out), s.Err)
					}
					t.Fatalf("no progress: %d of %d retired", retired, len(seqs))
				}
				done, err := sc.Step()
				if err != nil {
					t.Fatal(err)
				}
				retired += len(done)
			}
			for i, s := range seqs {
				if s.Err != nil {
					t.Fatalf("row %d: %v", i, s.Err)
				}
				for j := range s.Out {
					if s.Out[j] == want[i][j] {
						continue
					}
					if gap := hostGap(t, m, ids[i], want[i][:j], want[i][j], s.Out[j]); gap > tieMargin {
						t.Fatalf("row %d token %d: scheduled %d, alone %d, %.4f apart on the host\ngot  %v\nwant %v",
							i, j, s.Out[j], want[i][j], gap, s.Out, want[i])
					} else {
						t.Logf("row %d token %d: a tie (%.4f apart on the host), not compared past it", i, j, gap)
					}
					break
				}
			}
		})
	}
}

// fourDevices is four distinct GPUs to split a model over: the first four of
// JITLLM_TEST_DEVICES when it is set, otherwise every CUDA device and then each Vulkan GPU
// that is not one of them -- so a host whose CUDA_VISIBLE_DEVICES shows one
// card still runs the arm on the cards Vulkan sees, the same physical card
// never counted twice (backend.Identity) and a software rasteriser never
// counted at all. Fewer than four is the reason, for the arm's skip.
func fourDevices(t *testing.T) ([]string, string) {
	t.Helper()
	if testdevices.Selected() {
		specs := testdevices.Selection()
		if len(specs) < 4 {
			return specs, fmt.Sprintf("%d device(s) selected (%s), and the arm needs four",
				len(specs), strings.Join(specs, ","))
		}
		return specs[:4], ""
	}
	var specs []string
	var ids []backend.Identity
	n, err := backend.CUDACount()
	if err != nil {
		n = 0
	}
	for i := 0; i < n && len(specs) < 4; i++ {
		d, err := backend.OpenCUDA(i)
		if err != nil {
			continue
		}
		ids = append(ids, backend.IdentityOf(d))
		d.Close()
		specs = append(specs, "cuda:"+strconv.Itoa(i))
	}
	infos, err := backend.VulkanDevices()
	if err != nil {
		infos = nil
	}
	for _, in := range infos {
		if len(specs) == 4 {
			break
		}
		if in.Software || !in.Compute {
			continue
		}
		id := backend.UUIDIdentity(in.UUID[:], in.UUIDOK, "VkPhysicalDeviceIDProperties.deviceUUID")
		if slices.ContainsFunc(ids, id.Same) {
			continue
		}
		ids = append(ids, id)
		specs = append(specs, "vulkan:"+strconv.Itoa(in.Index))
	}
	if len(specs) < 4 {
		return specs, fmt.Sprintf("%d distinct GPU(s) here (%s), and the arm needs four (JITLLM_TEST_DEVICES names them)",
			len(specs), strings.Join(specs, ","))
	}
	return specs, ""
}

// spread is how many devices hold at least one block.
func spread(placed []int) int {
	n := 0
	for _, p := range placed {
		if p > 0 {
			n++
		}
	}
	return n
}

// sandwich places host, cuda:0, second, host, second over half the model each
// side: block 0 and the middle block on the host, blocks 1-3 on cuda:0, and the
// rest on the second device.
func sandwich(second string) map[int]Place {
	return map[int]Place{0: {On: "host"}, 1: {On: "cuda:0"}, 2: {On: "cuda:0"}, 3: {On: "cuda:0"},
		4: {On: second}, 5: {On: second}, 6: {On: second}, 7: {On: second}, 8: {On: "host"},
		9: {On: second}, 10: {On: second}, 11: {On: second}, 12: {On: second},
		13: {On: second}, 14: {On: second}, 15: {On: second}}
}
