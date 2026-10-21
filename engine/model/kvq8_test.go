package model

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/cpu"
)

// kvq8Models are the models the q8 cache gate runs: the smallest real llama
// (stories15M), a real Llama 3 (GQA, 64-wide heads), a real hybrid whose
// attention layers sit between gated delta-rule layers (Qwen3.5-0.8B), a short
// convolution hybrid (LFM2), and Gemma 4's two head geometries (GeomSplit:
// the sliding layers attend at a width of their own, and its E-model shares
// KV between layers).
var kvq8Models = []string{
	"stories15M-q8_0.gguf",
	"Llama-3.2-1B-Instruct-Q4_K_M.gguf",
	"Qwen3.5-0.8B-f16.gguf",
	"lfm2/LFM2-350M-Q4_K_M.gguf",
	"synth-gemma4.gguf",
	"synth-gemma4-e.gguf",
}

// kvTeacherForce runs ids through a fresh State at t and returns every
// position's logits. setup, when not nil, sees the State before its first
// token.
func kvTeacherForce(t *testing.T, m *Model, kt KVType, ids []int32, setup func(*State)) (*State, [][]float32) {
	t.Helper()
	if err := m.SetKVType(kt); err != nil {
		t.Fatal(err)
	}
	defer m.ClearKVF16()
	s := m.NewState(len(ids) + 1)
	if setup != nil {
		setup(s)
	}
	var outs [][]float32
	for _, id := range ids {
		out, err := s.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		outs = append(outs, append([]float32(nil), out...))
	}
	return s, outs
}

// q8Twin makes s's f32 cache store every row rounded through q8_0
// (kvLayout.q8Twin): the control arm a q8 cache must equal bit for bit.
func q8Twin(s *State) {
	c := s.c
	hd := max(c.HeadDim, c.HeadDimSWA)
	s.kvl.jit = s.jit
	s.jit.ReserveKVQuant(hd)
	s.kvl.q8Twin = make([]float32, (max(c.NKVHead, c.NKVHeadSWA)+1)*cpu.KVRowBytes(cpu.KVQ8, hd)/4)
}

// TestKVQ8IsSelectedAndRuns holds the q8_0 cache end to end, teacher-forced
// over 48 tokens, against two arms of the same model:
//
//   - its twin, an f32 cache whose every stored row was rounded through q8_0
//     and widened back (q8Twin): the q8 kernels read d*q exactly as the f32
//     ones read the widened row, in the same order, so the logits must be
//     BIT-IDENTICAL at every position. Any difference is the q8 cache's own
//     addressing (a wrong block's scale, head, row or page), which an NMSE
//     band would hide inside the rounding;
//   - the plain f32 cache, which it must NOT equal (else the q8 cache never
//     ran) and whose distance is the q8_0 rounding's cost, logged per model.
//
// It checks the cache is SELECTED (State.KVType and the page's row format),
// and counts the KV bytes a position costs in each format against the row
// format's arithmetic.
func TestKVQ8IsSelectedAndRuns(t *testing.T) {
	for _, name := range kvq8Models {
		t.Run(name, func(t *testing.T) {
			// Untuned: the arms are held bit for bit, and a kernel pick timed
			// mid-run (the arm64 per-shape choice) moves the reduction order
			// between them -- the logits then differ by up to a few percent.
			m, err := Open(jlmOf(t, testmodels.Path(name)), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := make([]int32, 48)
			for i := range ids {
				ids[i] = int32((i*7919 + 13) % m.Cfg.NVocab)
			}
			s32, a := kvTeacherForce(t, m, KVF32, ids, nil)
			defer s32.Close()
			s8, b := kvTeacherForce(t, m, KVQ8_0, ids, nil)
			defer s8.Close()
			tw, c := kvTeacherForce(t, m, KVF32, ids, q8Twin)
			defer tw.Close()
			if s8.KVType() != KVQ8_0 || s8.kvl.fmt != cpu.KVQ8 {
				t.Fatalf("asked for a q8_0 cache and got %v", s8.KVType())
			}
			if s32.KVType() != KVF32 || tw.KVType() != KVF32 {
				t.Fatalf("asked for f32 caches and got %v and %v", s32.KVType(), tw.KVType())
			}
			for p := range ids {
				for i := range b[p] {
					if math.Float32bits(b[p][i]) != math.Float32bits(c[p][i]) {
						t.Fatalf("position %d logit %d: %v on the q8_0 cache, %v on the f32 cache of its rows",
							p, i, b[p][i], c[p][i])
					}
				}
			}
			var worst, floor float64
			floor = math.Inf(1)
			agree := 0
			for p := 0; p < len(ids); p++ {
				e := logitNMSE(b[p], a[p])
				if math.IsNaN(e) {
					t.Fatalf("position %d: a non-finite logit NMSE", p)
				}
				worst = math.Max(worst, e)
				floor = math.Min(floor, e)
				if argmaxF(a[p]) == argmaxF(b[p]) {
					agree++
				}
			}
			if worst == 0 {
				t.Fatalf("the q8_0 cache's logits equal the f32 cache's at every position: it did not run")
			}
			// The twin shares the widening load with the q8 kernels (a scale
			// read from the wrong block reads wrong in both, and the twin
			// still agrees), which the kernel gates hold against an
			// independent dequantization (cpu.TestAttnQ8*). The band against
			// f32 closes it end to end: clean, the worst position's logit
			// NMSE is 6e-4 to 0.053 across these models and the top-1 agrees
			// at 35 of 48 positions or more; with every scale read from the
			// row's first block, Llama-3.2-1B reads 5.13 and agrees at 1.
			if worst > 0.25 || agree < len(ids)/2 {
				t.Errorf("against the f32 cache: worst logit NMSE %.3g, top-1 at %d of %d positions -- past "+
					"what q8_0 rounding costs", worst, agree, len(ids))
			}
			// Bytes: every layer that keeps history, K and V, at the row
			// format's bytes.
			b32, b8 := s32.KVBytesPerToken(), s8.KVBytesPerToken()
			want := 0
			for li := 0; li < m.Cfg.NLayer; li++ {
				if s8.kv.layers[li].p == 0 || m.Cfg.KVShared(li) {
					continue
				}
				l := s8.kvlAt(li)
				want += 2 * l.nKV * cpu.KVRowBytes(cpu.KVQ8, l.headDim)
			}
			if b8 != want || b8 >= b32 {
				t.Errorf("q8_0 costs %d bytes a position (want %d) against f32's %d", b8, want, b32)
			}
			t.Logf("%-36s KV bytes/token f32 %d q8_0 %d (%.3fx); against f32: logit NMSE worst %.3g best %.3g, top-1 %d/%d",
				name, b32, b8, float64(b8)/float64(b32), worst, floor, agree, len(ids))
		})
	}
}

// argmaxF is the index of the largest logit.
func argmaxF(x []float32) int {
	j := 0
	for i := range x {
		if x[i] > x[j] {
			j = i
		}
	}
	return j
}

// TestKVQ8RefusesByName: the caches a q8 row cannot express are refused at
// Open, naming the reason, and through SetKVType on an open model.
func TestKVQ8RefusesByName(t *testing.T) {
	for _, c := range []struct{ name, why string }{
		{"synth-deepseek-lite", "MLA"}, // the directory: jlmOf converts it fresh, a .jlm named here is opened as it stands
		{"synth-deepseek32.gguf", "lightning indexer"},
		{"synth-minimaxm3.gguf", "MiniMax Sparse Attention"},
		{"synth-deepseek4.gguf", "compressor"},
	} {
		t.Run(c.name, func(t *testing.T) {
			src, ok := existingModel(testmodels.Path(c.name))
			if !ok {
				testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path(c.name))
			}
			p := jlmOf(t, src)
			if _, err := Open(p, WithKVType(KVQ8_0)); err == nil || !strings.Contains(err.Error(), c.why) {
				t.Fatalf("Open with a q8_0 cache: %v, want a refusal naming %q", err, c.why)
			}
			m, err := Open(p)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if err := m.SetKVType(KVQ8_0); err == nil || !strings.Contains(err.Error(), c.why) {
				t.Fatalf("SetKVType(q8_0): %v, want a refusal naming %q", err, c.why)
			}
		})
	}
}
