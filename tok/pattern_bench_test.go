package tok

import (
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/tok/pretok"
)

// benchStages are the splits most vocabularies run, as the stage kinds an
// older container carries for them, each with its own splitter.
func benchStages() []struct {
	name string
	op   pretok.Op
} {
	return []struct {
		name string
		op   pretok.Op
	}{
		{"llama-bpe", pretok.Op{Kind: pretok.OpSplit, Split: llama3Params(3)}},
		{"qwen2", pretok.Op{Kind: pretok.OpSplit, Split: llama3Params(1)}},
		{"gpt-4o", pretok.Op{Kind: pretok.OpSplit, Split: pretok.SplitParams{Contractions: true, CaseFold: true, DigitGroup: 3,
			CaseRuns: true, SuffixContract: true, PunctSlash: true}}},
		{"gpt-2", pretok.Op{Kind: pretok.OpByteLevel, UseRegex: true}},
	}
}

// longText is the corpus's prose and code, every script in it, repeated to
// a quarter of a mebibyte: what a long prompt asks of a split.
func longText(b testing.TB) string {
	var s strings.Builder
	base := strings.Join(patternCorpus(b)[:40], "\n")
	for s.Len() < 256<<10 {
		s.WriteString(base)
	}
	return s.String()
}

func splitCount(each func(string, *runes, func(string)), text string, sc *runes) int {
	n := 0
	each(text, sc, func(string) { n++ })
	return n
}

func stageEach(op pretok.Op) func(string, *runes, func(string)) {
	return func(s string, sc *runes, emit func(string)) { applyOp(op, s, sc, emit) }
}

// BenchmarkSplitStage times each split over the long text, as its own splitter
// and as the pattern engine, one at a time: under perf stat at one thread the
// instruction counts of the two arms are the work.
func BenchmarkSplitStage(b *testing.B) {
	text := longText(b)
	for _, st := range benchStages() {
		pat, beh, _ := st.op.AsPattern()
		sp, err := compileSplit(pat, beh, false)
		if err != nil {
			b.Fatal(err)
		}
		for _, arm := range []struct {
			name string
			each func(string, *runes, func(string))
		}{{"stage", stageEach(st.op)}, {"pattern", sp.each}} {
			b.Run(st.name+"/"+arm.name, func(b *testing.B) {
				var sc runes
				b.SetBytes(int64(len(text)))
				for b.Loop() {
					splitCount(arm.each, text, &sc)
				}
			})
		}
	}
}

// BenchmarkSplitPaired is the wall-time comparison, paired and interleaved in
// one process: per round the stage then the engine (the order alternating),
// 21 rounds, and the median of the per-round ratios engine/stage with its
// interquartile range.
func BenchmarkSplitPaired(b *testing.B) {
	text := longText(b)
	for _, st := range benchStages() {
		pat, beh, _ := st.op.AsPattern()
		sp, err := compileSplit(pat, beh, false)
		if err != nil {
			b.Fatal(err)
		}
		b.Run(st.name, func(b *testing.B) {
			var sa, sb runes
			stage := stageEach(st.op)
			for b.Loop() {
				var ratios []float64
				for r := range 21 {
					arms := [2]func(string, *runes, func(string)){stage, sp.each}
					var took [2]time.Duration
					for k := range 2 {
						i := (k + r) % 2
						sc := &sa
						if i == 1 {
							sc = &sb
						}
						t0 := time.Now()
						splitCount(arms[i], text, sc)
						took[i] = time.Since(t0)
					}
					ratios = append(ratios, float64(took[1])/float64(took[0]))
				}
				slices.Sort(ratios)
				med := ratios[len(ratios)/2]
				b.ReportMetric(med, "engine/stage")
				b.ReportMetric((ratios[3*len(ratios)/4]-ratios[len(ratios)/4])/med, "iqr/median")
			}
		})
	}
}
