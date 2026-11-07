package engine

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
)

// TestCPUConvertsAQ8History runs a q8 history's composed conversions through
// the instance, as a migration does: rows of every head width its layers
// hold, past whole units of the forms so the last rows are staged, narrowed
// into the history's bytes word for word as oracle.QuantizeQ8 writes a
// zero-padded row, then widened back to each element's d*q, on every model
// whose history a q8 row can hold.
func TestCPUConvertsAQ8History(t *testing.T) {
	ran := 0
	for _, name := range principleModels() {
		t.Run(name, func(t *testing.T) {
			m, err := model.Open(jlmOf(t, testmodels.Path(name)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			s := testInstance(t, m, 8, WithKVType(model.KVQ8_0))
			defer s.Close()
			if s.KVType() != model.KVQ8_0 {
				t.Fatal("a q8_0 cache was asked for and not selected")
			}
			rng := rand.New(rand.NewSource(5))
			widths := map[int]bool{}
			for li := s.lo; li < s.hi; li++ {
				widths[s.kvlAt(li).headDim] = true
			}
			for hd := range widths {
				n := 2*cpuRowsCap + 3
				nb := (hd + nn.KVQ8Block - 1) / nn.KVQ8Block
				rs := nn.KVRowBytes(nn.KVQ8, hd)
				f := make([]float32, n*hd)
				for i := range f {
					f[i] = float32(rng.NormFloat64())
				}
				h := make([]byte, n*rs)
				before := kvRan(s)
				if !s.convertKV(true, hd, f, h) {
					t.Fatalf("head width %d: the q8 rows are not served", hd)
				}
				pad := make([]float32, nb*nn.KVQ8Block)
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
				if !s.convertKV(false, hd, back, h) {
					t.Fatalf("head width %d: the q8 rows are not widened", hd)
				}
				for r := 0; r < n; r++ {
					row := h[r*rs : (r+1)*rs]
					for i := 0; i < hd; i++ {
						d := math.Float32frombits(binary.LittleEndian.Uint32(row[nb*nn.KVQ8Block+8*(i/nn.KVQ8Block):]))
						if want := d * float32(int8(row[i])); math.Float32bits(back[r*hd+i]) != math.Float32bits(want) {
							t.Fatalf("head width %d: row %d element %d is %v, want %v", hd, r, i, back[r*hd+i], want)
						}
					}
				}
				if got := kvRan(s) - before; got != 2 {
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

// TestCPUAttendsAQ8History runs a prompt and its decode on the composed CPU
// with a q8_0 history and with an f32 one, on every model with an attention
// history: every block step is served composed over the q8 pages, and the
// first attending block -- whose keys and values read no history before they
// are written, so both instances write the same rows -- holds each row of a
// prompt inside its window as oracle.QuantizeQ8 quantizes the f32 history's
// row, widened to d*q, bit for bit. The logits must be finite
// and not the f32 history's (which they would be were the rows held at f32);
// how far they move is the model's sensitivity, logged and not judged:
// attention reading the rows is TestComposedCPUAttendQ8's.
func TestCPUAttendsAQ8History(t *testing.T) {
	const decode = 6
	ran := 0
	for _, name := range principleModels() {
		t.Run(name, func(t *testing.T) {
			m, err := model.Open(jlmOf(t, testmodels.Path(name)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			c := m.Cfg
			li := -1
			for i := range c.NLayer {
				if c.LayerKind(i).Attends() {
					li = i
					break
				}
			}
			if li < 0 {
				return
			}
			ids := m.Vocab.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 6), true)
			n := len(ids) - decode
			type run struct {
				logits [][]float32
				k, v   []float32
				hd     int
			}
			do := func(kv model.KVType) run {
				s := testInstance(t, m, len(ids)+1, WithKVType(kv))
				defer s.Close()
				if s.KVType() != kv {
					t.Fatalf("a %v cache was asked for and %v selected", kv, s.KVType())
				}
				lg, err := s.Prefill(ids[:n])
				if err != nil {
					t.Fatal(err)
				}
				r := run{logits: [][]float32{append([]float32(nil), lg...)}}
				for i := range decode {
					if lg, err = s.Forward(ids[n+i]); err != nil {
						t.Fatal(err)
					}
					r.logits = append(r.logits, append([]float32(nil), lg...))
				}
				cpuServed(t, s, fmt.Sprintf("%v: a prompt of %d", kv, n), (1+decode)*c.NLayer, true)
				h := 16
				if c.SWA(li) && c.SWAWindow > 0 {
					h = min(h, c.SWAWindow)
				}
				hs := testInstance(t, m, h+1, WithKVType(kv))
				defer hs.Close()
				if _, err := hs.Prefill(ids[:h]); err != nil {
					t.Fatal(err)
				}
				l := hs.kvlAt(li)
				_, _, r.k, r.v = kvMigBufs(l, h)
				if l.fmt == nn.KVF32 {
					r.k, r.v = make([]float32, h*l.kvDim()), make([]float32, h*l.kvDim())
				}
				if !hs.kv.MigrateKV(li, r.k, r.v, h, false, hs.kvConv) {
					t.Fatalf("%v: block %d's history was not gathered", kv, li)
				}
				r.hd = l.headDim
				return r
			}
			f, q := do(model.KVF32), do(model.KVQ8_0)
			nb := (f.hd + nn.KVQ8Block - 1) / nn.KVQ8Block
			pad, codes, pairs := make([]float32, nb*nn.KVQ8Block), make([]int8, nb*nn.KVQ8Block), make([]float32, 2*nb)
			for _, x := range [][2][]float32{{f.k, q.k}, {f.v, q.v}} {
				for r := 0; r < len(x[0])/f.hd; r++ {
					clear(pad[copy(pad, x[0][r*f.hd:(r+1)*f.hd]):])
					if err := oracle.QuantizeQ8(codes, pairs, pad, 128); err != nil {
						t.Fatal(err)
					}
					for i := range f.hd {
						want := pairs[2*(i/nn.KVQ8Block)] * float32(codes[i])
						if got := x[1][r*f.hd+i]; math.Float32bits(got) != math.Float32bits(want) {
							t.Fatalf("block %d's history row %d element %d is %v, the f32 row quantized is %v", li, r, i, got, want)
						}
					}
				}
			}
			worst, differs := 0.0, false
			for i := range f.logits {
				e, _ := nmseOf(q.logits[i], f.logits[i])
				if math.IsNaN(e) || math.IsInf(e, 0) {
					t.Fatalf("row %d: NMSE %v", i, e)
				}
				worst = max(worst, e)
				differs = differs || e > 0
			}
			if !differs {
				t.Fatal("the q8 history's logits equal the f32 history's: its rows were not quantized")
			}
			t.Logf("block %d's %d history rows quantized exactly; %d prompt rows and %d decoded: worst NMSE %.3e against the f32 history",
				li, len(f.k)/f.hd, n, decode, worst)
			ran++
		})
	}
	if ran == 0 {
		t.Fatal("no model holds a q8 history")
	}
}
