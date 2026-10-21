package model

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestNoTensorIsInGGUFLayout is the assertion behind "the container carries the
// paging layout and nothing else". A wrong layout costs only bandwidth, so an
// output oracle cannot catch it; this asserts the layout. The embedding is also
// checked by name because it has a second reader (embedRow).
func TestNoTensorIsInGGUFLayout(t *testing.T) {
	for _, name := range []string{
		"tinyllama-1.1b-q3_K_M.gguf",        // tied head, Q6_K embedding
		"Llama-3.2-1B-Instruct-Q4_K_M.gguf", // tied head
		"Qwen3-MOE-4x0.6B-Q4_K_M.gguf",      // expert banks and an F32 router
	} {
		t.Run(name, func(t *testing.T) {
			m, err := Open(jlmOf(t, testmodels.Path(name)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if !m.Container {
				t.Fatal("not a container -- this gate proved nothing")
			}

			packed, verbatim := 0, 0
			check := func(what string, tn *tensor) {
				if tn.e == nil || len(tn.data) == 0 {
					return
				}
				if _, ok := kernels.QuantOf(tn.typ); !ok {
					// F32/F16 norms, biases and routers have no packed layout.
					verbatim++
					return
				}
				if tn.packed == nil || len(tn.packed.QS) == 0 {
					t.Errorf("%s (%s) is in GGUF layout inside a container", what, tn.typ)
					return
				}
				packed++
			}
			check("token_embd.weight", &m.embd)
			check("output.weight", &m.output)
			for i := range m.layers {
				l := &m.layers[i]
				for _, p := range []struct {
					n string
					t *tensor
				}{{"attn_q", &l.wq}, {"attn_k", &l.wk}, {"attn_v", &l.wv}, {"attn_output", &l.wo},
					{"ffn_gate", &l.gate}, {"ffn_up", &l.up}, {"ffn_down", &l.down}, {"ffn_gate_inp", &l.router}} {
					check("blk."+itoa(i)+"."+p.n, p.t)
				}
				for e := range l.experts {
					check("blk."+itoa(i)+".exp_gate", &l.experts[e].gate)
					check("blk."+itoa(i)+".exp_up", &l.experts[e].up)
					check("blk."+itoa(i)+".exp_down", &l.experts[e].down)
				}
			}
			if packed == 0 {
				t.Fatal("nothing was packed -- this gate proved nothing")
			}

			// The embedding lookup must take the packed branch.
			if _, ok := kernels.QuantOf(m.embd.typ); ok && m.embd.packed == nil {
				t.Fatal("the embedding is quantised and not packed")
			}
			dst := make([]float32, m.Cfg.NEmbd)
			if err := m.embedRow(dst, 1); err != nil {
				t.Fatalf("embedRow: %v", err)
			}
			var nz int
			for _, v := range dst {
				if v != 0 {
					nz++
				}
			}
			if nz == 0 {
				t.Fatal("the embedding row came back all zero")
			}
			t.Logf("%s: %d packed, %d verbatim (%s embedding, %d non-zero of %d)",
				name, packed, verbatim, m.embd.typ, nz, len(dst))
		})
	}

	// And the vision tower, which holds its own tensors.
	t.Run("tower", func(t *testing.T) {
		if _, err := os.Stat(towerPath); err != nil {
			t.Skipf("MODEL MISSING: %s (%v) (set JITLLM_MODELS to the model directory) -- this gate proved nothing", towerPath, err)
		}
		vlm, tw := openTower(t)
		defer vlm.Close()
		if tw.container == nil {
			t.Fatal("the tower did not open a container -- this gate proved nothing")
		}
		packed := 0
		for _, w := range tw.weights() {
			if w.e == nil || len(w.data) == 0 {
				continue
			}
			if _, ok := kernels.QuantOf(w.typ); !ok {
				continue
			}
			if w.packed == nil || len(w.packed.QS) == 0 {
				t.Errorf("tower %s (%s) is in GGUF layout inside a container", w.e.Name, w.typ)
				continue
			}
			packed++
		}
		if packed == 0 {
			t.Fatal("no tower weight was packed -- this gate proved nothing")
		}
		t.Logf("tower: %d packed weights", packed)
	})
}
