package model

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestOffHeapFramesMatchTheHeap is the gate on jlm.SetOffHeap: a model whose
// frames and dense region are anonymous mappings decodes bit-identically to
// the same model on the Go heap, under a page budget of one frame so every
// block is evicted, re-read and re-bound into recycled mappings. It checks
// the mappings were SELECTED (OffHeapBytes rises once the model opens) and
// RETURNED (back to where it started once the model closes).
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
		base := jlm.OffHeapBytes()
		opts := []Option{noTune, WithPageBudget(1)}
		if !off {
			opts = append(opts, WithHeapWeights())
		}
		m, err := Open(path, opts...)
		if err != nil {
			t.Fatal(err)
		}
		if got := jlm.OffHeapBytes() > base; got != off {
			m.Close()
			t.Fatalf("off-heap %v, but the model mapped %d bytes", off, jlm.OffHeapBytes()-base)
		}
		s := m.NewState(64)
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
		if n := jlm.OffHeapBytes() - base; n != 0 {
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
	before := jlm.OffHeapBytes()
	m, err := Open(jlmOf(t, p), noTune)
	if err != nil {
		t.Fatal(err)
	}
	c := m.container
	if c.MappedBytes() == 0 {
		t.Skip("this platform maps nothing off the heap -- this gate proved nothing")
	}
	budget := 3 * m.PageSize()
	m.SetPageBudget(budget)
	st := m.NewState(64)
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
	if after := jlm.OffHeapBytes(); after != before {
		t.Fatalf("%d off-heap bytes before the model, %d after it closed", before, after)
	}
	t.Logf("budget %d, worst %d mapped (dense %d)", budget, worst, c.H.DenseLen)
}

// TestTwoPagedModelsShareTheProcess is the multi-model case off the heap: two
// models open in one process, each under a budget of a few pages so both page,
// decoding at the same time. Each must produce what it produces alone, each
// File's ledger must hold throughout (every mapped byte resident, free or
// dense), closing one must return exactly its mappings while the other keeps
// decoding, and closing both must bring the process back to where it started.
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
		st := m.NewState(64)
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
		m, err := Open(p, noTune)
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
	before := jlm.OffHeapBytes()
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
	if now := jlm.OffHeapBytes(); now != before+int64(second) {
		t.Fatalf("after closing the first model %d off-heap bytes, want the second's %d over the %d before",
			now, second, before)
	}
	if again := decode(ms[1]); fmt.Sprint(again) != fmt.Sprint(want[1]) {
		t.Fatalf("the second model after the first closed read %v, want %v", again, want[1])
	}
	ms[1].Close()
	if now := jlm.OffHeapBytes(); now != before {
		t.Fatalf("both models closed and %d off-heap bytes remain over the %d before", now-before, before)
	}
}
