package engine

import (
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/layout"
	"github.com/jitllm/jitllm/internal/testmodels"
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
			m, err := model.Open(jlmOf(t, testmodels.Path(name)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if m.Container == nil {
				t.Fatal("not a container -- this gate proved nothing")
			}
			s := testInstance(t, m, 64)
			defer s.Close()

			packed, verbatim := 0, 0
			check := func(what string, tn *model.Tensor) {
				if tn.E == nil || len(tn.Data) == 0 {
					return
				}
				if _, ok := layout.QuantOf(tn.Typ); !ok {
					// F32/F16 norms, biases and routers have no packed layout.
					verbatim++
					return
				}
				if tn.Packed == nil || len(tn.Packed.QS) == 0 {
					t.Errorf("%s (%s) is in GGUF layout inside a container", what, tn.Typ)
					return
				}
				packed++
			}
			check("token_embd.weight", &s.dense.Embd)
			check("output.weight", &s.dense.Output)
			for i := range m.Layers {
				l, err := s.pageIn(i)
				if err != nil {
					t.Fatal(err)
				}
				for _, p := range []struct {
					n string
					t *model.Tensor
				}{{"attn_q", &l.Wq}, {"attn_k", &l.Wk}, {"attn_v", &l.Wv}, {"attn_output", &l.Wo},
					{"ffn_gate", &l.Gate}, {"ffn_up", &l.Up}, {"ffn_down", &l.Down}, {"ffn_gate_inp", &l.Router}} {
					check("blk."+model.Itoa(i)+"."+p.n, p.t)
				}
				for e := range l.Experts {
					check("blk."+model.Itoa(i)+".exp_gate", &l.Experts[e].Gate)
					check("blk."+model.Itoa(i)+".exp_up", &l.Experts[e].Up)
					check("blk."+model.Itoa(i)+".exp_down", &l.Experts[e].Down)
				}
			}
			if packed == 0 {
				t.Fatal("nothing was packed -- this gate proved nothing")
			}

			// The embedding lookup must take the packed branch.
			if _, ok := layout.QuantOf(s.dense.Embd.Typ); ok && s.dense.Embd.Packed == nil {
				t.Fatal("the embedding is quantised and not packed")
			}
			dst := make([]float32, m.Cfg.NEmbd)
			if err := embedRowRef(&s.dense.Embd, dst, 1); err != nil {
				t.Fatalf("embedRowRef: %v", err)
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
				name, packed, verbatim, m.Embd.Typ, nz, len(dst))
		})
	}
}
