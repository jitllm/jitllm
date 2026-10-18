package model

import (
	"os"
	"slices"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestABatchLeavesAnotherSessionAlone puts two States on one tier: a single
// sequence A and a three-row batch B, whose rows need three times A's
// positions on every block. A device's KV capacity is each session's own, so
// B's reservation must not grow A's caches, and when the card cannot hold B's
// rows on every block, B takes blocks home for itself alone: A keeps every
// block on the card, its history byte for byte, and the logits it gives with
// nobody beside it.
func TestABatchLeavesAnotherSessionAlone(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(jlmOf(t, p), noTune, noGEMM)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const seq, rows, before, after = 256, 3, 3, 5
	open := func(t *testing.T, spec string) *tier.GPU {
		g, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || g == nil {
			noDevice(t, spec, err)
		}
		t.Cleanup(g.Close)
		return g
	}
	prompt := m.Vocab.Encode("The capital of France is", true)
	bPrompts := [][]int32{
		m.Vocab.Encode("Once upon a time, in a small village by the sea, there lived", true),
		m.Vocab.Encode("1, 2, 3,", true),
		m.Vocab.Encode("Water boils at", true),
	}
	bGens := []int{10, 6, 8}

	// attachA places A alone and prefills it; step decodes one greedy token
	// and returns a copy of the logits it came from.
	attachA := func(t *testing.T, g *tier.GPU) (*State, func() []float32) {
		a := m.NewState(seq)
		t.Cleanup(func() { a.Close() })
		if err := a.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if a.devCount() != m.Cfg.NLayer || !a.HeadOnDevice() {
			t.Fatalf("A placed %d of %d blocks, head on the card %v: %s",
				a.devCount(), m.Cfg.NLayer, a.HeadOnDevice(), g.Err())
		}
		lg, err := a.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		next := Greedy(lg)
		return a, func() []float32 {
			lg, err := a.Forward(next)
			if err != nil {
				t.Fatal(err)
			}
			next = Greedy(lg)
			return slices.Clone(lg)
		}
	}
	held := func(s *State) uint64 { return s.ld.(nn.Session).HeldBytes() }

	// A alone: the logits every later arm must reproduce exactly.
	// A subtest, so its tier is closed before the next arm opens another on
	// the same card.
	var want [][]float32
	var usedA uint64
	t.Run("alone", func(t *testing.T) {
		g := open(t, "gpu:0")
		_, step := attachA(t, g)
		usedA = g.Budgets()[0].Used
		for i := 0; i < before+after; i++ {
			want = append(want, step())
		}
	})
	if len(want) == 0 {
		t.Fatal("A alone did not run, so there is nothing to hold the shared arms to")
	}

	// B's tokens, from the host alone; a device run is held to them up to a
	// tie (batchTie), since the two tiers round differently.
	bIDs := func(t *testing.T, d *tier.GPU) ([][]int32, *State) {
		sc := NewScheduler(m, rows, seq)
		t.Cleanup(sc.Close)
		if d != nil {
			if err := sc.st.SetDevice(d); err != nil {
				t.Fatal(err)
			}
		}
		seqs := make([]*Seq, len(bPrompts))
		for i := range bPrompts {
			seqs[i] = &Seq{Prompt: bPrompts[i], MaxTokens: bGens[i]}
			sc.Submit(seqs[i])
		}
		for done, steps := 0, 0; done < len(seqs); steps++ {
			if steps > 200 {
				t.Fatalf("B made no progress: %d of %d retired", done, len(seqs))
			}
			d, err := sc.Step()
			if err != nil {
				t.Fatal(err)
			}
			done += len(d)
		}
		out := make([][]int32, len(seqs))
		for i, s := range seqs {
			if s.Err != nil {
				t.Fatalf("B row %d: %v", i, s.Err)
			}
			out[i] = s.Out
		}
		return out, sc.st
	}
	bWant, _ := bIDs(t, nil)
	sameB := func(t *testing.T, got [][]int32) {
		t.Helper()
		for i := range bWant {
			for j := range bWant[i] {
				if got[i][j] == bWant[i][j] {
					continue
				}
				if gap := hostGap(t, m, bPrompts[i], bWant[i][:j], bWant[i][j], got[i][j]); gap > batchTie {
					t.Fatalf("B row %d token %d: %d beside A, %d on the host, %.4f apart",
						i, j, got[i][j], bWant[i][j], gap)
				}
				break
			}
		}
	}
	// run interleaves: A decodes, B runs to completion on the same tier, A
	// decodes on. It checks A's history and logits and returns B's State.
	run := func(t *testing.T, g *tier.GPU) *State {
		a, step := attachA(t, g)
		var got [][]float32
		for i := 0; i < before; i++ {
			got = append(got, step())
		}
		hA, placed := held(a), g.Placed()
		bOut, b := bIDs(t, g)
		if h := held(a); h != hA {
			t.Errorf("B's rows moved A's history on the card: %d -> %d bytes", hA, h)
		}
		if a.devCount() != m.Cfg.NLayer || !slices.Equal(g.Placed(), placed) {
			t.Errorf("A's blocks moved: %d of %d on the card, placement %v -> %v",
				a.devCount(), m.Cfg.NLayer, placed, g.Placed())
		}
		for i := 0; i < after; i++ {
			got = append(got, step())
		}
		for i := range want {
			if !slices.Equal(got[i], want[i]) {
				t.Fatalf("A's logits at step %d differ from A alone on the card", i)
			}
		}
		sameB(t, bOut)
		return b
	}

	// Room for both: B's rows grow B's caches and nobody else's.
	var bFull uint64
	t.Run("room", func(t *testing.T) {
		b := run(t, open(t, "gpu:0"))
		if b.devCount() != m.Cfg.NLayer {
			t.Fatalf("with room for both, B placed %d of %d blocks", b.devCount(), m.Cfg.NLayer)
		}
		bFull = held(b)
		t.Logf("A alone charged %d bytes; B holds %d for %d rows of %d", usedA, bFull, rows, seq)
	})
	if bFull == 0 {
		t.Fatal("the roomy arm did not run, so the tight one has no budget to derive")
	}

	// Room for A and half of B's rows: B takes blocks home for itself, and A
	// keeps every one.
	t.Run("tight", func(t *testing.T) {
		b := run(t, open(t, testdevices.Like(t, "cuda:0")+"="+strconv.FormatUint(usedA+bFull/2, 10)))
		if n := b.devCount(); n == 0 || n == m.Cfg.NLayer {
			t.Fatalf("B placed %d of %d blocks: the budget did not make B split, so this "+
				"compared nothing", n, m.Cfg.NLayer)
		}
		t.Logf("B kept %d of %d blocks on the card; A kept all of them", b.devCount(), m.Cfg.NLayer)
	})
}
