package model

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestOffHeapFramesMatchTheHeap is the gate on the off-heap weights: a model
// whose frames and dense region are anonymous mappings decodes bit-identically
// to the same model on the Go heap (WithGCPlan, OnHeap), under a page budget of
// one frame so every block is evicted, re-read and re-bound into recycled
// mappings. It checks the mappings were SELECTED (the model maps bytes once it
// opens, and none on the heap) and RETURNED (none left once it closes).
func TestOffHeapFramesMatchTheHeap(t *testing.T) {
	paths := []string{benchModel(t)}
	if p, ok := existingModel(testmodels.Path("Qwen3-MOE-4x0.6B-Q4_K_M.gguf")); ok {
		paths = append(paths, p)
	}
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) { offHeapGate(t, jlmOf(t, path)) })
	}
}

func offHeapGate(t *testing.T, path string) {
	decode := func(off bool) [][]float32 {
		m, err := Open(path, WithPageBudget(1), WithGCPlan(sched.GCPlan{OnHeap: !off}))
		if err != nil {
			t.Fatal(err)
		}
		c := m.container
		if got := c.MappedBytes() > 0; got != off {
			m.Close()
			t.Fatalf("off-heap %v, but the model mapped %d bytes", off, c.MappedBytes())
		}
		s := m.NewInstance(64)
		ids := m.Vocab.Encode("The capital of France is", true)
		lg, err := s.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		out := [][]float32{append([]float32(nil), lg...)}
		for i := 0; i < 8; i++ {
			if lg, err = s.Forward(argmaxID(lg)); err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		s.Close()
		m.Close()
		if n := c.MappedBytes(); n != 0 {
			t.Fatalf("%d off-heap bytes still mapped after Close", n)
		}
		return out
	}
	want, got := decode(false), decode(true)
	for step := range want {
		for i := range want[step] {
			if math.Float32bits(got[step][i]) != math.Float32bits(want[step][i]) {
				t.Fatalf("step %d logit %d: %v off the heap, %v on it", step, i, got[step][i], want[step][i])
			}
		}
	}
	t.Logf("%d steps bit-identical with every frame off the heap at a one-frame budget", len(want))
}

// TestOffHeapMemoryStaysWithinTheBudget is the ledger on off-heap frames, which
// no collector frees. Decoding a mixture under a budget of a few pages -- block
// pages and expert pages are different sizes, so a frame evicted to make room
// is often not one a page-in can reuse -- every mapped byte is resident, free
// or the dense region, and free frames take only room the budget has left
// (resident itself may pass the budget while a token's pages are pinned). After
// TrimFree it is exactly resident plus dense, and after Close nothing this
// model mapped is left.
func TestOffHeapMemoryStaysWithinTheBudget(t *testing.T) {
	p, ok := existingModel(testmodels.Path("Qwen3-MOE-4x0.6B-Q4_K_M.gguf"))
	if !ok {
		t.Skip("MODEL MISSING: Qwen3-MOE-4x0.6B-Q4_K_M (set JITLLM_MODELS) -- this gate proved nothing")
	}
	m, err := Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	c := m.container
	if c.MappedBytes() == 0 {
		t.Skip("this platform maps nothing off the heap -- this gate proved nothing")
	}
	budget := 3 * m.PageSize()
	m.SetPageBudget(budget)
	st := m.NewInstance(64)
	lg, err := st.Prefill(m.Vocab.Encode("The capital of France is", true))
	if err != nil {
		t.Fatal(err)
	}
	worst := uint64(0)
	for i := 0; i < 12; i++ {
		if lg, err = st.Forward(Greedy(lg)); err != nil {
			t.Fatal(err)
		}
		mb, res, fb := c.MappedBytes(), c.ResidentBytes(), c.FreeBytes()
		worst = max(worst, mb)
		if acct := res + fb + c.H.DenseLen; mb > acct+1<<16 || mb < acct {
			t.Fatalf("step %d: %d bytes mapped, but resident %d + free %d + dense %d is %d: a frame is mapped and owned by nothing",
				i, mb, res, fb, c.H.DenseLen, acct)
		}
		if res+fb > max(budget, res) {
			t.Fatalf("step %d: %d free bytes beside %d resident, past the %d-byte budget", i, fb, res, budget)
		}
	}
	if _, faults, _ := m.PageStats(); faults == 0 {
		t.Fatal("nothing was paged: the budget held the model, so this proved nothing")
	}
	c.TrimFree()
	if fb := c.FreeBytes(); fb != 0 {
		t.Fatalf("%d free bytes after TrimFree", fb)
	}
	if mb, want := c.MappedBytes(), c.ResidentBytes()+c.H.DenseLen; mb > want+1<<16 || mb < want {
		t.Fatalf("after TrimFree %d bytes mapped, want the %d resident plus dense", mb, want)
	}
	st.Close()
	m.Close()
	if after := c.MappedBytes(); after != 0 {
		t.Fatalf("%d off-heap bytes still mapped after the model closed", after)
	}
	t.Logf("budget %d, worst %d mapped (dense %d)", budget, worst, c.H.DenseLen)
}

// TestTwoPagedModelsShareTheProcess is the multi-model case off the heap: two
// models open in one process, each under a budget of a few pages so both page,
// decoding at the same time. Each must produce what it produces alone, each
// File's ledger must hold throughout (every mapped byte resident, free or
// dense), closing one must return exactly its mappings while the other keeps
// its own and decodes, and closing both must leave neither mapping anything.
func TestTwoPagedModelsShareTheProcess(t *testing.T) {
	var paths []string
	for _, name := range []string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Qwen3-MOE-4x0.6B-Q4_K_M.gguf"} {
		p, ok := existingModel(testmodels.Path(name))
		if !ok {
			t.Skipf("MODEL MISSING: %s (set JITLLM_MODELS) -- this gate proved nothing", name)
		}
		paths = append(paths, jlmOf(t, p))
	}
	const steps = 10
	decode := func(m *Model) []int32 {
		st := m.NewInstance(64)
		defer st.Close()
		lg, err := st.Prefill(m.Vocab.Encode("The capital of France is", true))
		if err != nil {
			t.Error(err)
			return nil
		}
		var out []int32
		for i := 0; i < steps; i++ {
			id := Greedy(lg)
			out = append(out, id)
			if lg, err = st.Forward(id); err != nil {
				t.Error(err)
				return nil
			}
			c := m.container
			if mb, acct := c.MappedBytes(), c.ResidentBytes()+c.FreeBytes()+c.H.DenseLen; mb > acct+1<<16 || mb < acct {
				t.Errorf("%s step %d: %d mapped against %d accounted", c.Path, i, mb, acct)
			}
		}
		return out
	}
	open := func(p string) *Model {
		m, err := Open(p)
		if err != nil {
			t.Fatal(err)
		}
		m.SetPageBudget(3 * m.PageSize())
		return m
	}
	// Each alone, for the reference.
	var want [2][]int32
	for i, p := range paths {
		m := open(p)
		want[i] = decode(m)
		m.Close()
	}
	ms := [2]*Model{open(paths[0]), open(paths[1])}
	if ms[0].container.MappedBytes() == 0 {
		ms[0].Close()
		ms[1].Close()
		t.Skip("this platform maps nothing off the heap -- this gate proved nothing")
	}
	var got [2][]int32
	done := make(chan int, 2)
	for i := range ms {
		go func() { got[i] = decode(ms[i]); done <- i }()
	}
	<-done
	<-done
	for i := range ms {
		if fmt.Sprint(got[i]) != fmt.Sprint(want[i]) {
			t.Fatalf("model %d beside the other read %v, alone %v", i, got[i], want[i])
		}
		if _, faults, _ := ms[i].PageStats(); faults == 0 {
			t.Fatalf("model %d never paged: its budget held it, so this proved nothing", i)
		}
	}
	// Closing the first returns its mappings and only its; the second still runs.
	second := ms[1].container.MappedBytes()
	ms[0].Close()
	if first, now := ms[0].container.MappedBytes(), ms[1].container.MappedBytes(); first != 0 || now != second {
		t.Fatalf("after closing the first model it maps %d bytes and the second %d, want 0 and the second's %d",
			first, now, second)
	}
	if again := decode(ms[1]); fmt.Sprint(again) != fmt.Sprint(want[1]) {
		t.Fatalf("the second model after the first closed read %v, want %v", again, want[1])
	}
	ms[1].Close()
	if a, b := ms[0].container.MappedBytes(), ms[1].container.MappedBytes(); a+b != 0 {
		t.Fatalf("both models closed and they still map %d and %d off-heap bytes", a, b)
	}
}
