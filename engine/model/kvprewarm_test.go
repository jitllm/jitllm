package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestFirstPromptDoesNotGrowTheKVPool: a State that places every block on a
// device grows the device's KV pools ahead (PrewarmKV), so its first prompt
// of one chunk grows none of them -- Stats.KVGrows does not move across the
// Prefill -- and answers exactly as the same State without the prewarm,
// whose first prompt does grow them (the violation arm, which must move the
// counter or the gate proves nothing). With a budget that leaves no room past
// the blocks, the prewarm takes nothing and places the same blocks as no
// prewarm: it grows into what placement left, never into a block's room.
func TestFirstPromptDoesNotGrowTheKVPool(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm")
	if src := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf"); fileExists(src) {
		p = jlmOf(t, src)
	}
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	ids := func(m *Model) []int32 {
		return m.Vocab.Encode(strings.Repeat("The river rose every spring until the bridges were islands. ", 40), true)[:400]
	}
	type arm struct {
		logits         []float32
		grows, prewarm int
		placed         int
		used           uint64 // the budget placement spent, before the prompt
	}
	run := func(prewarm bool, budget uint64) arm {
		m, err := Open(p, noTune, WithKVF16(false), WithKVPrewarm(prewarm))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		opts := []tier.Option{tier.WithDeviceTune(tier.TuneOff)}
		if budget > 0 {
			opts = append(opts, tier.WithBudget(budget))
		}
		g, err := openTier(t, opts...)
		if err != nil || g == nil {
			noDevice(t, "device", err)
		}
		defer g.Close()
		in := ids(m)
		st := m.NewState(len(in) + 16)
		defer st.Close()
		st.SetDevice(g)
		before := g.Stats()
		used := before.BudgetUsed
		lg, err := st.Prefill(in)
		if err != nil {
			t.Fatal(err)
		}
		after := g.Stats()
		return arm{append([]float32(nil), lg...), after.KVGrows - before.KVGrows, before.KVPrewarmPages, st.GPULayers(), used}
	}
	on, off := run(true, 0), run(false, 0)
	t.Logf("prewarm: %d page(s) grown ahead, %d growth(s) in the first prompt; without: %d growth(s); %d and %d blocks placed",
		on.prewarm, on.grows, off.grows, on.placed, off.placed)
	if on.placed == 0 {
		t.Skip("CARD TOO SMALL: no block placed -- this gate proved nothing on this device")
	}
	if off.grows == 0 {
		t.Fatalf("the first prompt grew no pool without the prewarm either: the gate cannot tell the arms apart")
	}
	if on.prewarm == 0 || on.grows != 0 {
		t.Fatalf("the prewarm grew %d page(s) ahead and the first prompt still grew a pool %d time(s)", on.prewarm, on.grows)
	}
	if on.placed != off.placed {
		t.Fatalf("%d blocks placed with the prewarm, %d without", on.placed, off.placed)
	}
	for i := range on.logits {
		if on.logits[i] != off.logits[i] && !(math.IsNaN(float64(on.logits[i])) && math.IsNaN(float64(off.logits[i]))) {
			t.Fatalf("logit %d: %v with the prewarm, %v without: a pool's size changed the answer", i, on.logits[i], off.logits[i])
		}
	}
	// A budget exactly what placement spent without the prewarm, and a page's
	// room more: the prewarm finds no room, takes nothing, and the State
	// places what the same budget places without it -- it grows into what placement left,
	// never into a block's room or past the budget.
	tight := off.used + 1<<16
	ton, toff := run(true, tight), run(false, tight)
	t.Logf("budget %d bytes (placement's own %d): %d and %d blocks placed, %d page(s) grown ahead",
		tight, off.used, ton.placed, toff.placed, ton.prewarm)
	if ton.placed != toff.placed || ton.prewarm != 0 {
		t.Fatalf("at placement's own budget the prewarm grew %d page(s) and placed %d blocks against %d", ton.prewarm, ton.placed, toff.placed)
	}
	for i := range ton.logits {
		if ton.logits[i] != toff.logits[i] {
			t.Fatalf("logit %d at the tight budget: %v with the prewarm, %v without", i, ton.logits[i], toff.logits[i])
		}
	}
}
