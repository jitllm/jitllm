//go:build jitllmbench

// This file is a measurement instrument, not a gate, so it is opt-in by build
// tag rather than by a skip in every default run:
//
//	go test -tags jitllmbench -run TestQ8RouterMargins ./engine/model
//
// JITLLM_* variables select its parameters.

package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestQ8RouterMargins converts the safetensors fixture JITLLM_Q8_ROUTER names
// (one with a golden in testdata/golden/hf) twice, float and WithQ8, decodes
// the golden's ids on both, and logs every mixture block's selection with its
// scores (sigmoid plus the selection bias, the order the selection is made in)
// and the margin between the k-th and the next. A position where the two
// containers select different experts is named. It is what attributed
// synth-glm4moe-hf's position 7 to a flip (see hfQ8NMSE).
func TestQ8RouterMargins(t *testing.T) {
	name := os.Getenv("JITLLM_Q8_ROUTER")
	if name == "" {
		t.Skip("set JITLLM_Q8_ROUTER to a fixture, e.g. synth-glm4moe-hf")
	}
	b, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "hf", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var gold struct {
		IDs []int32 `json:"ids"`
	}
	if err := json.Unmarshal(b, &gold); err != nil {
		t.Fatal(err)
	}
	dir := testmodels.Path(name)
	type rec struct {
		sel    map[int][]int32
		logits map[int][]float32
	}
	run := func(q8 bool) []rec {
		dst := filepath.Join(t.TempDir(), name+jlm.Ext)
		var opts []convert.Option
		if q8 {
			opts = append(opts, convert.WithQ8())
		}
		if _, err := convert.FromSafetensors(dir, dst, jlm.Fingerprint{Host: "test"}, opts...); err != nil {
			t.Fatal(err)
		}
		m, err := Open(dst, WithKVF16(false))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		st := m.NewInstance(len(gold.IDs) + 1)
		defer st.Close()
		var out []rec
		for _, id := range gold.IDs {
			r := rec{map[int][]int32{}, map[int][]float32{}}
			m.Trace(func(l int, n string, v []float32) {
				switch n {
				case "moe_logits":
					r.logits[l] = append([]float32(nil), v...)
				case "moe_topk":
					s := make([]int32, len(v))
					for i, x := range v {
						s[i] = int32(x)
					}
					r.sel[l] = s
				}
			})
			if _, err := st.Forward(id); err != nil {
				t.Fatal(err)
			}
			m.Trace(nil)
			out = append(out, r)
		}
		// selection scores
		for p, r := range out {
			for l, lg := range r.logits {
				bias := m.layers[l].vector(jlm.RoleExpProbsB)
				sc := make([]float64, len(lg))
				idx := make([]int, len(lg))
				for e, x := range lg {
					sc[e] = 1 / (1 + math.Exp(-float64(x)))
					if bias != nil {
						sc[e] += float64(bias[e])
					}
					idx[e] = e
				}
				sort.Slice(idx, func(a, b int) bool { return sc[idx[a]] > sc[idx[b]] })
				k := len(r.sel[l])
				t.Logf("q8=%v pos %d layer %d sel %v  order %v  scores %s  margin(k,k+1) %.5f",
					q8, p, l, r.sel[l], idx[:k+2], fmtScores(sc, idx[:k+2]), sc[idx[k-1]]-sc[idx[k]])
			}
		}
		return out
	}
	f := run(false)
	q := run(true)
	for p := range f {
		for l := range f[p].sel {
			a, b := fmt.Sprint(f[p].sel[l]), fmt.Sprint(q[p].sel[l])
			if a != b {
				t.Logf("pos %d layer %d: f32 selects %s, Q8 %s", p, l, a, b)
			}
		}
	}
}

func fmtScores(sc []float64, idx []int) string {
	s := ""
	for _, i := range idx {
		s += fmt.Sprintf("%d:%.5f ", i, sc[i])
	}
	return s
}
