package engine

import (
	"github.com/jitllm/jitllm/engine/device/cpudev"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestThePoolParksOnlyWhileTokensPage is the spin decision. After a decode
// token that faulted pages in, the pool's workers park: the next token is
// likely to wait on reads too, and a spinning worker would burn a core through
// each. After one that read nothing, they spin: its regions follow each other
// closely, and a wake-up at every one cost Qwen3-30B 2.6x when the
// decision was made from the model's size against the budget instead.
//
// olmoe under a budget one expert page short of the whole model is the case
// that tells the two rules apart: the pager may evict throughout, and yet
// some tokens route only to experts already resident and read nothing.
func TestThePoolParksOnlyWhileTokensPage(t *testing.T) {
	p, ok := existingModel(testmodels.Path("olmoe-1b-7b-0924-instruct-Q4_K_M.gguf"))
	if !ok {
		testmodels.Missing(t, "MODEL MISSING: olmoe-1b-7b-0924-instruct-Q4_K_M (set JITLLM_MODELS) -- this gate proved nothing")
	}
	m, err := model.Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	c := m.Container
	var total uint64
	for i := 0; i < c.NPages(); i++ {
		total += c.PageBytes(i)
	}
	st := testInstance(t, m, 96, WithPageBudget(total-c.H.ExpPageSize))
	defer st.Close()
	if !st.fr.CanEvict() {
		t.Fatal("a budget one page short of the model and the pager cannot evict: this proved nothing")
	}
	lg, err := st.Prefill(m.Vocab.Encode("The capital of France is", true))
	if err != nil {
		t.Fatal(err)
	}
	var quiet, paged int
	for i := 0; i < 60 && (quiet == 0 || paged == 0); i++ {
		_, in0, _ := m.PageStats(st.fr)
		if lg, err = st.Forward(Greedy(lg)); err != nil {
			t.Fatal(err)
		}
		_, in1, _ := m.PageStats(st.fr)
		switch spin := cpudev.PoolOf(st.Devices()).Spinning(); {
		case in1 != in0 && spin:
			t.Fatalf("token %d faulted %d page(s) and left the pool spinning", i, in1-in0)
		case in1 == in0 && !spin:
			t.Fatalf("token %d read nothing and left the pool parked: the decision followed the budget, not the reads", i)
		case in1 == in0:
			quiet++
		default:
			paged++
		}
	}
	if quiet == 0 || paged == 0 {
		t.Fatalf("%d tokens read nothing and %d paged: both are needed for this to prove anything", quiet, paged)
	}
}
