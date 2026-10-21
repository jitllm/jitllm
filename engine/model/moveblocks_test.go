package model

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// moveDevices is the two devices of one tier the device-to-device gates move
// blocks between: the first two of JITLLM_TEST_DEVICES, or the laptop's one
// card under two backends.
func moveDevices(t *testing.T) (string, string) {
	t.Helper()
	d := testdevices.Take(t, 2, "cuda:0,vulkan:1")
	return d[0], d[1]
}

// TestBlocksMoveBetweenDevices moves every block of a sequence from one device
// of a tier to another mid-generation and back (MoveBlocks), teacher-forced
// on an unmoved control's tokens: no prompt row is computed again, every block
// lands where it was sent, and the logits after each move are the control's --
// bit for bit between two devices running the same kernels, and inside the
// reduction-order band between two backends (RULE 11c).
//
// The default arm prepares each move while the sequence keeps generating
// (PrepareMove) and commits it at the first token the target is ready, the
// history copied device to device: it asserts the direct path ran and the
// blocks were staged. The host arm (WithDirectMoves(false)) is the control
// the direct path is measured against: every history through the State's
// cache. The upper arm moves only the upper half, so the output projection
// stays behind on a device that no longer holds the last block: the
// projection's own submission, which no placement reached before a block
// could be moved.
func TestBlocksMoveBetweenDevices(t *testing.T) {
	names := append([]string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf"}, relocModels()...)
	for _, name := range names {
		t.Run(name, func(t *testing.T) { blocksMoveBetweenDevices(t, jlmOf(t, testmodels.Path(name)), moveWhole) })
		t.Run(name+"/host", func(t *testing.T) {
			blocksMoveBetweenDevices(t, jlmOf(t, testmodels.Path(name)), moveHost, WithDirectMoves(false))
		})
		t.Run(name+"/upper", func(t *testing.T) { blocksMoveBetweenDevices(t, jlmOf(t, testmodels.Path(name)), moveUpper) })
	}
}

// The arms of TestBlocksMoveBetweenDevices.
const (
	moveWhole = iota
	moveHost
	moveUpper
)

func blocksMoveBetweenDevices(t *testing.T, path string, arm int, opts ...Option) {
	a, b := moveDevices(t)
	g, err := openTier(t, tier.WithDevices(a+","+b), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, a+","+b, err)
	}
	defer g.Close()
	distinctCards(t, g)
	m, err := Open(path, append(opts, WithPlacement(Placement{Rest: &Place{On: a}, Head: &Place{On: a}}))...)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const ntok, moveAt, backAt = 12, 4, 8
	prompt := m.Vocab.Encode("The capital of France is a city that", true)
	maxSeq := len(prompt) + ntok + 1

	// run feeds ids (nil: its own greedy picks) and moves the blocks at
	// moveAt and back at backAt when move is set.
	run := func(ids []int32, move bool) ([]int32, [][]float32, []BlockMove) {
		st := m.NewState(maxSeq)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		placed := st.placedOn(a)
		if len(placed) == 0 {
			t.Skipf("%s took no block of this model: %s", a, g.Err())
		}
		if arm == moveUpper {
			if len(placed) < 2 {
				t.Skipf("%s took %d block(s); the upper half of one is all of it", a, len(placed))
			}
			placed = placed[len(placed)/2:]
		}
		logits, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		// The range the moves name: the placed blocks and every KV-sharing
		// group they reach, whole.
		lo, hi := placed[0], placed[len(placed)-1]+1
		for _, li := range placed {
			glo, ghi := st.kvSpan(li)
			lo, hi = min(lo, glo), max(hi, ghi)
		}
		var out []int32
		var lg [][]float32
		var moves []BlockMove
		var pm *PendingMove
		to, from := b, a
		for i := 0; i < ntok; i++ {
			lg = append(lg, append([]float32(nil), logits...))
			next := Greedy(logits)
			if ids != nil {
				next = ids[i]
			}
			out = append(out, next)
			if move && (i == moveAt || i == backAt) {
				to, from = b, a
				if i == backAt {
					to, from = a, b
				}
				if pm, err = st.PrepareMove(lo, hi, to); err != nil {
					t.Fatalf("%s -> %s, blocks %d..%d: %v", from, to, lo, hi-1, err)
				}
			}
			// The host arm commits at once, as MoveBlocks does; the others at
			// the first token the target is ready, and the last before the
			// next move at the latest.
			if pm != nil && (arm == moveHost || pm.Ready() || i == backAt-1 || i == ntok-1) {
				rows := st.PrefillRows()
				mv, err := pm.Commit()
				if err != nil {
					t.Fatalf("%s -> %s, blocks %d..%d: %v", from, to, lo, hi-1, err)
				}
				pm = nil
				if got := st.PrefillRows() - rows; got != 0 {
					t.Fatalf("the move %s -> %s computed %d prompt rows again", from, to, got)
				}
				for _, li := range placed {
					if at, _ := st.DeviceOf(li); at != to {
						t.Fatalf("block %d is on %q after a move to %s (declined %v)", li, at, to, mv.Declined)
					}
				}
				moves = append(moves, mv)
			}
			if logits, err = st.Forward(next); err != nil {
				t.Fatal(err)
			}
		}
		return out, lg, moves
	}

	ctl, ctlLogits, _ := run(nil, false)
	_, movedLogits, moves := run(ctl, true)
	if len(moves) != 2 || len(moves[0].Moved) == 0 || moves[0].KVBytes+moves[0].RecBytes == 0 {
		t.Fatalf("the moves did not run: %+v", moves)
	}
	for i, mv := range moves {
		t.Logf("move %d: %d block(s), %d KV and %d recurrent bytes, path %q, %d copied, head moved %v, "+
			"%d staged in %v, pause %v", i, len(mv.Moved), mv.KVBytes, mv.RecBytes, mv.Path, mv.Copied,
			mv.Head, mv.Staged, mv.Prepare, mv.Pause)
	}
	_, mover := any(g.Attach()).(nn.BlockMover)
	switch {
	case arm == moveHost && moves[0].Path != "host":
		t.Fatalf("the host arm's history went by %q", moves[0].Path)
	case arm != moveHost && mover && moves[0].Path == "host":
		t.Fatalf("the direct arm's history came home: the direct path did not run")
	case arm == moveWhole && mover && !m.Cfg.MoE() && moves[0].Staged != len(moves[0].Moved):
		t.Fatalf("%d of %d blocks were prepared ahead of the move", moves[0].Staged, len(moves[0].Moved))
	}
	same := strings.SplitN(a, ":", 2)[0] == strings.SplitN(b, ":", 2)[0]
	var band float64
	defer func() { t.Logf("largest logit difference after a move: %g", band) }()
	for i := range ctlLogits {
		var worst float64
		for j, v := range ctlLogits[i] {
			worst = max(worst, math.Abs(float64(v-movedLogits[i][j])))
		}
		if math.IsNaN(worst) {
			t.Fatalf("token %d: a non-finite logit", i)
		}
		band = max(band, worst)
		switch {
		case i <= moveAt && worst != 0:
			t.Fatalf("token %d is before any move and its logits differ by %g", i, worst)
		case same && worst != 0:
			t.Fatalf("token %d: two devices running the same kernels differ by %g after a move", i, worst)
		case worst > 2.5:
			t.Fatalf("token %d: the moved logits are %g from the control's, outside the band", i, worst)
		}
	}
}

// placedOn lists the blocks the named device holds for this State.
func (s *State) placedOn(name string) []int {
	var out []int
	for li := s.lo; li < s.hi; li++ {
		if at, ok := s.DeviceOf(li); ok && at == nn.DeviceName(name) {
			out = append(out, li)
		}
	}
	return out
}

// moveAll moves the blocks to the named device, a KV-sharing group whole,
// failing the test on an error.
func (s *State) moveAll(t *testing.T, from, to string, blocks []int) BlockMove {
	t.Helper()
	var all BlockMove
	done := map[int]bool{}
	for _, li := range blocks {
		lo, hi := s.kvSpan(li)
		if done[lo] {
			continue
		}
		done[lo] = true
		mv, err := s.MoveBlocks(lo, hi, to)
		if err != nil {
			t.Fatalf("%s -> %s, blocks %d..%d: %v", from, to, lo, hi-1, err)
		}
		all.Moved = append(all.Moved, mv.Moved...)
		all.Declined = append(all.Declined, mv.Declined...)
		all.KVBytes += mv.KVBytes
		all.RecBytes += mv.RecBytes
		all.Path, all.Head = mv.Path, all.Head || mv.Head
	}
	return all
}

// TestBlocksMoveToADeviceAddedLater: a tier opened on one card takes a second
// mid-sequence (tier.GPU.AddDevices), the blocks move onto it and back with
// their history, and the logits stay in the band of a run that never moved --
// the plasticity a running model needs to reach a card it was not opened with,
// without a second tier holding its weights twice.
func TestBlocksMoveToADeviceAddedLater(t *testing.T) {
	a, b := moveDevices(t)
	g, err := openTier(t, tier.WithDevices(a), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, a, err)
	}
	defer g.Close()
	m, err := Open(jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")),
		WithPlacement(Placement{Rest: &Place{On: a}, Head: &Place{On: a}}))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const ntok, moveAt, backAt = 10, 3, 7
	prompt := m.Vocab.Encode("The capital of France is a city that", true)
	run := func(ids []int32, move bool) [][]float32 {
		st := m.NewState(len(prompt) + ntok + 1)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		placed := st.placedOn(a)
		if len(placed) == 0 {
			t.Skipf("%s took no block: %s", a, g.Err())
		}
		lo, hi := placed[0], placed[len(placed)-1]+1
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
			if move && i == moveAt {
				added, err := g.AddDevices(b)
				if err != nil || len(added) != 1 {
					t.Fatalf("adding %s: %v %v", b, added, err)
				}
				if _, err := g.AddDevices(b); err == nil {
					t.Fatalf("%s was added twice", b)
				}
			}
			if move && (i == moveAt || i == backAt) {
				to := b
				if i == backAt {
					to = a
				}
				mv, err := st.MoveBlocks(lo, hi, to)
				if err != nil || len(mv.Moved) == 0 {
					t.Fatalf("move to %s: %+v %v", to, mv, err)
				}
				if at, _ := st.DeviceOf(placed[0]); at != nn.DeviceName(to) {
					t.Fatalf("block %d is on %q after a move to %s", placed[0], at, to)
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
	for i := range ctl {
		var worst float64
		for j, v := range ctl[i] {
			worst = max(worst, math.Abs(float64(v-moved[i][j])))
		}
		switch {
		case math.IsNaN(worst):
			t.Fatalf("token %d: a non-finite logit", i)
		case i <= moveAt && worst != 0:
			t.Fatalf("token %d is before any move and differs by %g", i, worst)
		case worst > 2.5:
			t.Fatalf("token %d: %g from the control, outside the band", i, worst)
		}
	}
}

// TestABinary16HistoryMovesToTheDevice: a session that built its history on
// the host in the f16 cache moves onto a device and back mid-sequence -- its
// rows widened on the way up and narrowed home (migrateKVAs) -- and the logits
// stay in the band of a host run that never moved. A host-loaded model need not
// give up the f16 cache to keep its plasticity.
func TestABinary16HistoryMovesToTheDevice(t *testing.T) {
	a, _ := moveDevices(t)
	g, err := openTier(t, tier.WithDevices(a), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, a, err)
	}
	defer g.Close()
	m, err := Open(jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")), WithKVF16(true))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const ntok, upAt, homeAt = 10, 3, 7
	prompt := m.Vocab.Encode("The capital of France is a city that", true)
	run := func(ids []int32, move bool) [][]float32 {
		st := m.NewState(len(prompt) + ntok + 1)
		defer st.Close()
		if st.kvFmt != cpu.KVF16 {
			t.Skipf("the host cache is %v, not binary16: this gate proved nothing", st.kvFmt)
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
			if move && i == upAt {
				if err := st.SetDevice(g); err != nil {
					t.Fatalf("a binary16 history with %d positions could not attach: %v", st.pos, err)
				}
				if st.SetGPULayers(m.Cfg.NLayer); st.GPULayers() == 0 {
					t.Skipf("%s took no block: %s", a, g.Err())
				}
			}
			if move && i == homeAt {
				if st.SetGPULayers(0); st.GPULayers() != 0 {
					t.Fatalf("%d blocks stayed on the device", st.GPULayers())
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
	for i := range ctl {
		var worst float64
		for j, v := range ctl[i] {
			worst = max(worst, math.Abs(float64(v-moved[i][j])))
		}
		switch {
		case math.IsNaN(worst):
			t.Fatalf("token %d: a non-finite logit", i)
		case i <= upAt && worst != 0:
			t.Fatalf("token %d is before the move and differs by %g", i, worst)
		case worst > 2.5:
			t.Fatalf("token %d: %g from the host control, outside the band", i, worst)
		}
	}
}

// TestAnyRangeMovesWithItsHistory: placement is a set. A middle range of
// blocks goes to the device mid-sequence (PlaceBlocks), part of it comes home
// leaving a hole between two device runs (ReleaseBlocks), the rest after it,
// and every token stays in the band of a host run that never moved.
func TestAnyRangeMovesWithItsHistory(t *testing.T) {
	a, _ := moveDevices(t)
	g, err := openTier(t, tier.WithDevices(a), tier.WithDeviceTune(tier.TuneOff))
	if err != nil || g == nil {
		noDevice(t, a, err)
	}
	defer g.Close()
	m, err := Open(jlmOf(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")), WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	n := m.Cfg.NLayer
	const ntok, placeAt, holeAt, homeAt = 12, 3, 6, 9
	prompt := m.Vocab.Encode("The capital of France is a city that", true)
	run := func(ids []int32, move bool) [][]float32 {
		st := m.NewState(len(prompt) + ntok + 1)
		defer st.Close()
		if move {
			if err := st.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			st.SetGPULayers(0)
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
			switch {
			case move && i == placeAt:
				got, err := st.PlaceBlocks(n/4, 3*n/4)
				if err != nil || got != n/2 {
					t.Fatalf("placed %d of blocks %d..%d: %v (%s)", got, n/4, 3*n/4-1, err, g.Err())
				}
				if st.devAt(0) || st.devAt(n-1) || !st.devAt(n/4) {
					t.Fatalf("the placement is not the range asked")
				}
			case move && i == holeAt:
				got, err := st.ReleaseBlocks(n/4+1, n/4+3)
				if err != nil || got != 2 {
					t.Fatalf("released %d: %v", got, err)
				}
				if !st.devAt(n/4) || st.devAt(n/4+1) || !st.devAt(n/4+3) {
					t.Fatalf("no hole between two device runs")
				}
			case move && i == homeAt:
				if _, err := st.ReleaseBlocks(0, n); err != nil {
					t.Fatal(err)
				}
				if st.devCount() != 0 {
					t.Fatalf("%d blocks stayed on the device", st.devCount())
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
	for i := range ctl {
		var worst float64
		for j, v := range ctl[i] {
			worst = max(worst, math.Abs(float64(v-moved[i][j])))
		}
		switch {
		case math.IsNaN(worst):
			t.Fatalf("token %d: a non-finite logit", i)
		case i <= placeAt && worst != 0:
			t.Fatalf("token %d is before the move and differs by %g", i, worst)
		case worst > 2.5:
			t.Fatalf("token %d: %g from the host control, outside the band", i, worst)
		}
	}
}

// TestTwoModelsShareACardThroughTheRegistry: two models, each with its own
// tier, on one card whose shared pool (tier.WithPoolRegistry) cannot hold
// both: the second places what the card has left; when the first brings its
// blocks home the second takes the room. Without the registry each tier keeps
// the budget it opened with and the room the first gave back stays out of the
// second's reach -- the orchestrator's room to make.
func TestTwoModelsShareACardThroughTheRegistry(t *testing.T) {
	a, _ := moveDevices(t)
	for _, shared := range []bool{true, false} {
		t.Run(fmt.Sprintf("shared=%v", shared), func(t *testing.T) {
			var reg *tier.PoolRegistry
			opts := []tier.Option{tier.WithDevices(a + "=400M"), tier.WithDeviceTune(tier.TuneOff)}
			if shared {
				reg = tier.NewPoolRegistry()
				opts = append(opts, tier.WithPoolRegistry(reg))
			}
			ga, err := openTier(t, opts...)
			if err != nil || ga == nil {
				noDevice(t, a, err)
			}
			defer ga.Close()
			gb, err := openTier(t, opts...)
			if err != nil {
				t.Fatal(err)
			}
			defer gb.Close()
			ma, err := Open(graphModel(t, "Qwen3-0.6B-Q8_0"))
			if err != nil {
				t.Fatal(err)
			}
			defer ma.Close()
			mb, err := Open(graphModel(t, "Qwen3-0.6B-Q8_0"))
			if err != nil {
				t.Fatal(err)
			}
			defer mb.Close()
			sa, sb := ma.NewState(64), mb.NewState(64)
			defer sa.Close()
			defer sb.Close()
			if err := sa.SetDevice(ga); err != nil {
				t.Fatal(err)
			}
			if err := sb.SetDevice(gb); err != nil {
				t.Fatal(err)
			}
			na, nb := sa.GPULayers(), sb.GPULayers()
			if na == 0 {
				t.Skipf("the first model took no block of %s: %s", a, ga.Err())
			}
			if shared && nb == mb.Cfg.NLayer {
				t.Fatalf("both models fit a shared 400M card whole (%d+%d blocks): the gate cannot see the pool", na, nb)
			}
			if _, err := sa.ReleaseBlocks(0, ma.Cfg.NLayer); err != nil || sa.GPULayers() != 0 {
				t.Fatalf("the first model kept %d blocks: %v", sa.GPULayers(), err)
			}
			if _, err := sb.PlaceBlocks(0, mb.Cfg.NLayer); err != nil {
				t.Fatal(err)
			}
			grew := sb.GPULayers() > nb
			t.Logf("shared=%v: first model %d blocks, second %d, then %d once the first went home (%s)", shared, na, nb, sb.GPULayers(), gb.Err())
			if shared && !grew {
				t.Fatalf("the second model took no room the first gave back (%d blocks before and after)", nb)
			}
		})
	}
}
