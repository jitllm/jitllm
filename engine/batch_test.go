package engine

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestBatchMatchesForward holds ForwardBatch to the decode path it batches.
//
// The rows carry different tokens: identical sequences agree even when every
// row reads sequence 0's KV cache. Teacher-forced: both sides consume the same
// token stream, so a disagreement is localised to its position.
func TestBatchMatchesForward(t *testing.T) { batchMatchesForward(t) }

// batchMatchesForward is TestBatchMatchesForward with opts on every model it opens.
func batchMatchesForward(t *testing.T, opts ...Option) {
	for name, path := range models() {
		if name == "tinyllama" && os.Getenv("JITLLM_SLOW") == "" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			// A container-only host keeps the .jlm and not its GGUF.
			if p, ok := existingModel(path); ok {
				path = p
			} else {
				t.Skipf("model not present: %s (nor its container)", path)
			}
			// Two modes, as TestPrefillMatchesForward has. On the composed
			// CPU's float forms a step's rows and a row alone are apart by
			// their folding order, so the batch is held to floatFormsNMSE. The
			// shipped forms round their int8 activations apart, which flips
			// small-margin argmaxes, so that mode is judged by margin and a
			// gross-error tripwire.
			t.Run("exact-gemm", func(t *testing.T) {
				batchVsForward(t, path, true, append([]Option{withCPUFloat()}, opts...)...)
			})
			t.Run("shipped", func(t *testing.T) {
				batchVsForward(t, path, false, opts...)
			})
			// The single sequences and the batch on the composed CPU's
			// blocks: held to the shipped mode's bar.
			t.Run("composed-cpu", func(t *testing.T) {
				batchVsForward(t, path, false, opts...)
			})
		})
	}
	// The mixtures' batches on the composed CPU, each call holding the
	// experts its rows pick.
	for _, file := range cpuMixtureRowsModels() {
		t.Run("mixture/"+file+"/composed-cpu", func(t *testing.T) {
			batchVsForward(t, testmodels.Path(file), false, opts...)
		})
	}
}

func batchVsForward(t *testing.T, path string, exact bool, opts ...Option) {
	instOpts := opts
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const nseq, steps = 3, 12
	// Distinct, in-range, and deliberately not adjacent ids.
	streams := make([][]int32, nseq)
	for i := range streams {
		streams[i] = make([]int32, steps)
		for j := range streams[i] {
			streams[i][j] = int32((7 + i*101 + j*13) % m.Cfg.NVocab)
		}
	}

	// The single-sequence answer, one instance per row, and on the composed
	// CPU the block steps of a step it served, summed over the rows, and
	// what it declined.
	want := make([][][]float32, nseq)
	decServed, decDeclined := 0, [composer.NumDeclines]int{}
	for i := range streams {
		s := testInstance(t, m, steps+1, instOpts...)
		for _, id := range streams[i] {
			lg, err := s.Forward(id)
			if err != nil {
				s.Close()
				t.Fatal(err)
			}
			want[i] = append(want[i], append([]float32(nil), lg...))
		}
		{
			served, declined := s.CPUBlockSteps()
			decServed += served / steps
			for j, n := range declined {
				decDeclined[j] += n
			}
			total := served
			for j, n := range declined {
				total += n
				if n > 0 {
					t.Logf("row %d: %d block steps declined: %s", i, n, composer.DeclineName(j))
				}
			}
			if total != steps*m.Cfg.NLayer {
				s.Close()
				t.Fatalf("the census counts %d block steps on the CPU, where the decode ran %d", total, steps*m.Cfg.NLayer)
			}
			if i == 0 {
				t.Logf("the composed CPU served %d of %d block steps", served, total)
			}
		}
		s.Close()
	}

	b := newSessBatch(m, nseq, steps+1, instOpts...)
	defer b.Close()
	judge := logitJudge{exact: exact, what: "ForwardBatch and Forward"}
	for j := 0; j < steps; j++ {
		row := make([]int32, nseq)
		for i := range streams {
			row[i] = streams[i][j]
		}
		if _, err := b.ForwardBatch(row); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < nseq; i++ {
			judge.step(t, b.BatchLogits(i), want[i][j], fmt.Sprintf("seq %d step %d", i, j))
		}
	}
	judge.finish(t)
	// The batch's rows on the composed CPU's blocks: every step of every
	// block the decode served, served, and declined only by a reason the
	// decode declined the block by.
	{
		served, declined := b.CPUBlockSteps()
		for i, n := range declined {
			if n > 0 && decDeclined[i] == 0 {
				t.Errorf("the batch declined %d block steps the decode served: %s (%v)", n, composer.DeclineName(i), b.CPUFormErr())
			}
		}
		if want := steps * decServed / nseq; served != want {
			t.Errorf("the composed CPU served %d of the batch's block steps, where the decode served %d", served, want)
		}
	}
}

// logitJudge holds a batched path's logits to the single-sequence ones in one
// of the two modes the batch gates share. Exact: the composed CPU's float
// forms, a step's against a row's, apart by their folding order alone
// (floatFormsNMSE). Shipped: the batched GEMM may sum a k-quant
// super-block in integers (cpu.StationaryIntAcc: arm64, pre-VNNI amd64), a
// reassociation of NMSE ~1e-12 a matmul that int8 re-quantization of the
// activations amplifies into the 0.2-0.9 logit band, so a flip is judged at
// its own position against the size of the perturbation, and a gross error
// by an NMSE tripwire.
type logitJudge struct {
	exact    bool
	what     string
	sse, sy2 float64
}

// step judges one row's logits. Teacher-forced, so a flip is judged at its own
// position: where both argmaxes won by less than the whole perturbation the
// model is undecided; a confident flip is a defect.
func (j *logitJudge) step(t *testing.T, got, want []float32, where string) {
	t.Helper()
	if g, w := Greedy(got), Greedy(want); g != w {
		mA, mB, d := margin(want), margin(got), maxAbsDiff(want, got)
		if mA > d || mB > d {
			t.Errorf("%s: batch %d (margin %.4f), single %d (margin %.4f), "+
				"max|dlogit| %.4f", where, g, mB, w, mA, d)
		} else {
			t.Logf("%s: batch %d, single %d on margins %.4f / %.4f against "+
				"a %.4f perturbation -- undecided, not a defect", where, g, w, mB, mA, d)
		}
	}
	for x, v := range got {
		d := float64(v - want[x])
		j.sse += d * d
		j.sy2 += float64(want[x]) * float64(want[x])
	}
}

func (j *logitJudge) finish(t *testing.T) {
	t.Helper()
	nmse := 0.0
	if j.sy2 > 0 {
		nmse = j.sse / j.sy2
	}
	t.Logf("logits NMSE %.3e over every step", nmse)
	switch {
	case math.IsNaN(nmse):
		t.Error("NaN logits")
	case j.exact && !(nmse <= floatFormsNMSE):
		t.Errorf("NMSE %.3e on the float forms -- %s differ past their folding order (bound %.0e)",
			nmse, j.what, floatFormsNMSE)
	case nmse > 1e-2:
		// Two orders below the NMSE ~1 a wrong cache offset produces.
		t.Errorf("NMSE %.3e between %s -- too far to be reassociation", nmse, j.what)
	}
}
func TestRaggedBatchMatchesForward(t *testing.T) { raggedBatchMatchesForward(t) }

// raggedBatchMatchesForward is TestRaggedBatchMatchesForward with opts on every model it opens.
func raggedBatchMatchesForward(t *testing.T, opts ...Option) {
	for name, path := range models() {
		if name == "tinyllama" && os.Getenv("JITLLM_SLOW") == "" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			if p, ok := existingModel(path); ok {
				path = p
			} else {
				t.Skipf("model not present: %s (nor its container)", path)
			}
			t.Run("exact-gemm", func(t *testing.T) {
				raggedVsForward(t, path, true, append([]Option{withCPUFloat()}, opts...)...)
			})
			t.Run("shipped", func(t *testing.T) {
				raggedVsForward(t, path, false, opts...)
			})
		})
	}
}

func raggedVsForward(t *testing.T, path string, exact bool, opts ...Option) {
	instOpts := opts
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const nseq, steps = 3, 12
	// Row 0 runs straight through; rows 1 and 2 are retired mid-flight at
	// different steps, so no two rows share a depth afterwards.
	retireAt := [nseq]int{-1, 4, 7}
	streams := make([][]int32, nseq)
	for i := range streams {
		streams[i] = make([]int32, steps)
		for j := range streams[i] {
			streams[i][j] = int32((5 + i*211 + j*17) % m.Cfg.NVocab)
		}
	}

	// The oracle: one instance per row, Reset at the step the batch retires that
	// row.
	want := make([][][]float32, nseq)
	for i := range streams {
		s := testInstance(t, m, steps+1, instOpts...)
		for j, id := range streams[i] {
			if j == retireAt[i] {
				s.Reset()
			}
			lg, err := s.Forward(id)
			if err != nil {
				s.Close()
				t.Fatal(err)
			}
			want[i] = append(want[i], append([]float32(nil), lg...))
		}
		s.Close()
	}

	b := newSessBatch(m, nseq, steps+1, instOpts...)
	defer b.Close()
	judge := logitJudge{exact: exact, what: "a ragged ForwardBatch and Forward"}
	ragged := false
	for j := 0; j < steps; j++ {
		for i := 0; i < nseq; i++ {
			if j == retireAt[i] {
				b.Retire(i)
			}
		}
		seen := map[int]bool{}
		for i := 0; i < nseq; i++ {
			seen[b.SeqPos(i)] = true
		}
		if len(seen) > 1 {
			ragged = true
		}
		row := make([]int32, nseq)
		for i := range streams {
			row[i] = streams[i][j]
		}
		if _, err := b.ForwardBatch(row); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < nseq; i++ {
			judge.step(t, b.BatchLogits(i), want[i][j],
				fmt.Sprintf("row %d (depth %d) step %d", i, b.SeqPos(i)-1, j))
		}
	}
	// Assert the test actually went ragged, or it proves nothing.
	if !ragged {
		t.Fatal("no step had rows at different depths; this test proved nothing")
	}
	judge.finish(t)
}

// TestPrefillSeqMatchesForward is the gate on admitting a prompt into one row
// of a running batch.
//
// The other rows must hold different, non-empty history, or slot-0 addressing
// passes: a prompt admitted into row 1 without a sequence base would overwrite
// row 0's keys and read them back.
func TestPrefillSeqMatchesForward(t *testing.T) { prefillSeqMatchesForward(t) }

// prefillSeqMatchesForward is TestPrefillSeqMatchesForward with opts on every model it opens.
func prefillSeqMatchesForward(t *testing.T, opts ...Option) {
	for name, path := range models() {
		if name == "tinyllama" && os.Getenv("JITLLM_SLOW") == "" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			if p, ok := existingModel(path); ok {
				path = p
			} else {
				t.Skipf("model not present: %s (nor its container)", path)
			}
			// Token equality needs the exact arm, as TestBatchMatchesForward's
			// exact mode: on arm64 the batch's rows run MatMulPacked's fused
			// kernel, which sums a k-quant super-block in integers, where the
			// single instance decodes through the tiled kernel's float chain
			// (TuneOff pins it); qwen2vl's admitted row parted at step 5.
			instOpts := opts
			m, err := model.Open(jlmOf(t, path))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()

			const nseq, plen, steps = 3, 11, 6
			const admit = 1 // the row the prompt goes into
			maxSeq := plen + steps + 4
			id := func(a, b int) int32 { return int32((3 + a*307 + b*29) % m.Cfg.NVocab) }
			prompt := make([]int32, plen)
			for j := range prompt {
				prompt[j] = id(admit, j)
			}
			after := make([]int32, steps)
			for j := range after {
				after[j] = id(admit, plen+j)
			}

			// The oracle: one instance, the same prompt, the same continuation.
			one := testInstance(t, m, maxSeq, instOpts...)
			pl, err := one.Prefill(prompt)
			if err != nil {
				one.Close()
				t.Fatal(err)
			}
			want := []int32{Greedy(pl)}
			wantLg := [][]float32{append([]float32(nil), pl...)}
			for _, tk := range after {
				lg, err := one.Forward(tk)
				if err != nil {
					one.Close()
					t.Fatal(err)
				}
				want = append(want, Greedy(lg))
				wantLg = append(wantLg, append([]float32(nil), lg...))
			}
			one.Close()

			// Give the other rows real history first.
			b := newSessBatch(m, nseq, maxSeq, instOpts...)
			defer b.Close()
			for j := 0; j < 4; j++ {
				row := make([]int32, nseq)
				for i := range row {
					row[i] = id(i+7, j)
				}
				if _, err := b.ForwardBatch(row); err != nil {
					t.Fatal(err)
				}
			}
			// Retire the row to admit into, the legal way to reuse a slot.
			b.Retire(admit)
			if b.SeqPos(admit) != 0 {
				t.Fatalf("retired row is at position %d, want 0", b.SeqPos(admit))
			}
			lg, err := b.PrefillSeq(admit, prompt)
			if err != nil {
				t.Fatal(err)
			}
			if b.SeqPos(admit) != plen {
				t.Errorf("after a %d-token prompt the row is at position %d", plen, b.SeqPos(admit))
			}
			got := []int32{Greedy(lg)}
			gotLg := [][]float32{append([]float32(nil), lg...)}
			for _, tk := range after {
				row := make([]int32, nseq)
				for i := range row {
					row[i] = id(i+7, 99) // the other rows carry on with their own stream
				}
				row[admit] = tk
				if _, err := b.ForwardBatch(row); err != nil {
					t.Fatal(err)
				}
				got = append(got, Greedy(b.BatchLogits(admit)))
				gotLg = append(gotLg, append([]float32(nil), b.BatchLogits(admit)...))
			}
			// Teacher-forced, so every step is held to the band the row's
			// forms read apart (hostStepNMSE, RULE 8): its logits, not its
			// argmax, which a tie inside the band turns either way.
			for j := range want {
				if nm := logitNMSE(gotLg[j], wantLg[j]); !(nm <= hostStepNMSE) {
					t.Errorf("step %d: the admitted row's logits read NMSE %.3e from the single instance's (bound %.0e)", j, nm, hostStepNMSE)
				} else if got[j] != want[j] {
					t.Logf("step %d: admitted row %d, single instance %d, a tie inside the band (NMSE %.3e)", j, got[j], want[j], nm)
				}
			}
		})
	}
}
