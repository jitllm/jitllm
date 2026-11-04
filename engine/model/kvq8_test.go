package model

import (
	"encoding/binary"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestKVQ8RefusesByName: the caches a q8 row cannot express are refused at
// Open, naming the reason, and through SetKVType on an open model.
func TestKVQ8RefusesByName(t *testing.T) {
	for _, c := range []struct{ name, why string }{
		{"synth-deepseek-lite", "MLA"}, // the directory: jlmOf converts it fresh, a .jlm named here is opened as it stands
		{"synth-deepseek32.gguf", "lightning indexer"},
		{"synth-minimaxm3.gguf", "MiniMax Sparse Attention"},
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

// TestCPUConvertsAQ8History runs a q8 history's composed conversions through
// the instance, as a migration does: rows of every head width its layers
// hold, past whole units of the forms so the last rows are staged, narrowed
// into the history's bytes word for word as oracle.QuantizeQ8 writes a
// zero-padded row, then widened back to each element's d*q. No q8 history
// can be asked for while the composer forms no attention over one
// (KVTypeRefusal), so the instance is given one directly, on every model
// whose history a q8 row can hold.
func TestCPUConvertsAQ8History(t *testing.T) {
	ran := 0
	for _, name := range principleModels() {
		t.Run(name, func(t *testing.T) {
			m, err := Open(jlmOf(t, testmodels.Path(name)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if m.Cfg.kvQ8Inexpressible() != nil {
				return
			}
			s := m.NewInstance(8)
			defer s.Close()
			s.setKVFmt(cpu.KVQ8)
			if s.KVType() != KVQ8_0 {
				t.Fatal("a q8_0 cache was asked for and not selected")
			}
			rng := rand.New(rand.NewSource(5))
			widths := map[int]bool{}
			for li := s.lo; li < s.hi; li++ {
				widths[s.kvlAt(li).headDim] = true
			}
			for hd := range widths {
				n := 2*cpuRowsCap + 3
				nb := (hd + cpu.KVQ8Block - 1) / cpu.KVQ8Block
				rs := cpu.KVRowBytes(cpu.KVQ8, hd)
				f := make([]float32, n*hd)
				for i := range f {
					f[i] = float32(rng.NormFloat64())
				}
				h := make([]byte, n*rs)
				before := s.cpu.kvconv.ran
				if !s.cpuKV(true, hd, f, h) {
					t.Fatalf("head width %d: the q8 rows are not served", hd)
				}
				pad := make([]float32, nb*cpu.KVQ8Block)
				q := make([]int8, len(pad))
				pairs := make([]float32, 2*nb)
				for r := 0; r < n; r++ {
					clear(pad[copy(pad, f[r*hd:(r+1)*hd]):])
					if err := oracle.QuantizeQ8(q, pairs, pad, 128); err != nil {
						t.Fatal(err)
					}
					row := h[r*rs : (r+1)*rs]
					for i, c := range q {
						if int8(row[i]) != c {
							t.Fatalf("head width %d: row %d code %d is %d, want %d", hd, r, i, int8(row[i]), c)
						}
					}
					for b, p := range pairs {
						if got := math.Float32frombits(binary.LittleEndian.Uint32(row[len(q)+4*b:])); math.Float32bits(got) != math.Float32bits(p) {
							t.Fatalf("head width %d: row %d pair word %d is %v, want %v", hd, r, b, got, p)
						}
					}
				}
				back := make([]float32, len(f))
				if !s.cpuKV(false, hd, back, h) {
					t.Fatalf("head width %d: the q8 rows are not widened", hd)
				}
				for r := 0; r < n; r++ {
					row := h[r*rs : (r+1)*rs]
					for i := 0; i < hd; i++ {
						d := math.Float32frombits(binary.LittleEndian.Uint32(row[nb*cpu.KVQ8Block+8*(i/cpu.KVQ8Block):]))
						if want := d * float32(int8(row[i])); math.Float32bits(back[r*hd+i]) != math.Float32bits(want) {
							t.Fatalf("head width %d: row %d element %d is %v, want %v", hd, r, i, back[r*hd+i], want)
						}
					}
				}
				if got := s.cpu.kvconv.ran - before; got != 2 {
					t.Fatalf("head width %d: %d conversions counted, want 2", hd, got)
				}
				t.Logf("head width %d: %d rows narrowed and widened", hd, n)
			}
			ran++
		})
	}
	if ran == 0 {
		t.Fatal("no model holds a q8 history")
	}
}
