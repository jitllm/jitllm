package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestSafetensorsQ8MatchTransformers is TestSafetensorsMatchTransformers with
// convert.WithQ8: the same goldens, the same fixed ids, a bound for Q8_0
// instead of for float, and a check that the matrices really were quantized
// (a WithQ8 that quantized nothing would pass the float gate's bound too).
func TestSafetensorsQ8MatchTransformers(t *testing.T) {
	golds, _ := filepath.Glob(filepath.Join("..", "..", "testdata", "golden", "hf", "*.json"))
	if len(golds) == 0 {
		t.Fatal("no goldens -- run scripts/hfgold.py; this gate proved nothing")
	}
	for _, g := range golds {
		name := strings.TrimSuffix(filepath.Base(g), ".json")
		t.Run(name, func(t *testing.T) {
			dir := testmodels.Path(name)
			if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
				t.Skipf("MODEL MISSING: %s (set JITLLM_MODELS to the model directory) -- this gate proved nothing", dir)
			}
			b, err := os.ReadFile(g)
			if err != nil {
				t.Fatal(err)
			}
			var gold struct {
				IDs []int32 `json:"ids"`
				Pos []struct {
					Head []float64 `json:"head"`
				} `json:"pos"`
			}
			if err := json.Unmarshal(b, &gold); err != nil {
				t.Fatal(err)
			}
			dst := filepath.Join(t.TempDir(), name+jlm.Ext)
			if _, err := convert.FromSafetensors(dir, dst, jlm.Fingerprint{Host: "test"}, convert.WithQ8()); err != nil {
				t.Fatalf("convert %s: %v", dir, err)
			}
			f, err := jlm.Open(dst)
			if err != nil {
				t.Fatal(err)
			}
			q8, float := 0, 0
			for _, e := range f.Entries() {
				switch {
				case e.Type == jlm.TypeQ8:
					q8++
				case e.NDim >= 2 && e.Dims[0]%32 == 0 && e.Role != jlm.RoleRouter && e.Role != jlm.RoleShRouter:
					float++
				}
			}
			f.Close()
			if q8 == 0 && float == 0 {
				t.Skipf("no matrix here has k a multiple of 32, so there is nothing to quantize -- this gate proved nothing")
			}
			if q8 == 0 || float != 0 {
				t.Fatalf("WithQ8 left %d quantizable matrices float and quantized %d", float, q8)
			}
			// A position over the bound is excused only where some mixture
			// block chose other experts than the float container did there: a
			// top-k is the one discontinuity a rounding may legitimately cross.
			q8Logits, q8Sel := runTraced(t, dst, gold.IDs)
			var fSel [][]string
			worst, over := 0.0, 0
			for p := range gold.IDs {
				var num, den float64
				for i, w := range gold.Pos[p].Head {
					d := float64(q8Logits[p][i]) - w
					num, den = num+d*d, den+w*w
				}
				nmse := num / den
				if math.IsNaN(nmse) || math.IsInf(nmse, 0) {
					t.Fatalf("pos %d: NMSE %v", p, nmse)
				}
				worst = max(worst, nmse)
				if nmse <= hfQ8NMSE {
					continue
				}
				over++
				if fSel == nil {
					fdst := filepath.Join(t.TempDir(), name+"-float"+jlm.Ext)
					if _, err := convert.FromSafetensors(dir, fdst, jlm.Fingerprint{Host: "test"}); err != nil {
						t.Fatalf("convert %s: %v", dir, err)
					}
					_, fSel = runTraced(t, fdst, gold.IDs)
				}
				flipped := ""
				for l := range q8Sel[p] {
					if q8Sel[p][l] != fSel[p][l] {
						flipped += fmt.Sprintf(" block %d float %s, Q8 %s;", l, fSel[p][l], q8Sel[p][l])
					}
				}
				if flipped == "" {
					t.Errorf("pos %d: NMSE %.3e against transformers (bound %.0e), and every mixture "+
						"block chose the float container's experts there, so it is not a top-k flip",
						p, nmse, hfQ8NMSE)
					continue
				}
				t.Logf("pos %d: NMSE %.3e against transformers (bound %.0e), a top-k flip:%s",
					p, nmse, hfQ8NMSE, flipped)
			}
			t.Logf("%d matrices Q8_0, %d positions, worst NMSE %.2e, %d over the bound", q8, len(gold.IDs), worst, over)
		})
	}
}

// runTraced decodes ids one at a time on the container at path, returning
// every position's logits and, per block, the experts its router selected
// ("" for a block with no mixture).
func runTraced(t *testing.T, path string, ids []int32) ([][]float32, [][]string) {
	t.Helper()
	m, err := Open(path, noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	st := m.NewInstance(len(ids) + 1)
	defer st.Close()
	var logits [][]float32
	var sels [][]string
	for _, id := range ids {
		sel := make([]string, m.Cfg.NLayer)
		m.Trace(func(l int, n string, v []float32) {
			if n == "moe_topk" {
				sel[l] = fmt.Sprint(v)
			}
		})
		lg, err := st.Forward(id)
		m.Trace(nil)
		if err != nil {
			t.Fatal(err)
		}
		logits = append(logits, append([]float32(nil), lg...))
		sels = append(sels, sel)
	}
	return logits, sels
}

// hfQ8NMSE is measured: clean arms 4.4e-07 to 9.3e-03 (the random fixtures
// amplify rounding), while a stored scale 1.27x its codes reads 0.11 to 1.24.
// It catches a wrong layout, type or scale, not a rounding mode, and each of
// those moves every position. A mixture's top-k is discontinuous, so a
// position whose selection the rounding flipped is excused, and only that one:
// synth-glm4moe-hf's position 7 reads 1.47e-01 against 1.1e-03..3.1e-03
// elsewhere because block 2 picks experts [5 2] where the float path picks
// [5 0] -- expert 0's selection score (sigmoid plus bias) led expert 2's by
// 0.0298 in float and trails it by 0.0021 in Q8_0.
const hfQ8NMSE = 5e-2
