//go:build amd64 || arm64

package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// mtpGolden is scripts/mtpgold.py's record: the trunk's and the prediction
// block's logits at every position of a fixed id sequence, from transformers'
// own Qwen3_5DecoderLayer wired as vLLM and llama.cpp wire the block.
type mtpGolden struct {
	IDS         []int32     `json:"ids"`
	NLogit      int         `json:"nlogit"`
	Trunk       [][]float64 `json:"trunk"`
	MTP         [][]float64 `json:"mtp"`
	MTPArgmax   []int32     `json:"mtp_argmax"`
	TrunkArgmax []int32     `json:"trunk_argmax"`
}

// TestMTPMatchesReference holds the prediction block to the reference on its
// own logits: the trunk's hidden state at every position of the golden's ids,
// then the draft over rows (x_q, h_{q-1}), every row's logits read back and
// compared. The synthetic fixtures are F32 end to end and held to the bound
// the safetensors gate uses; the real model runs its Q8_0 file against a bf16
// reference and is held to its top-1 and a gross-error bound.
//
// A violation of each half of the block's input -- the hidden state shifted to
// the wrong row, the embedding of the wrong token -- runs beside it and must
// fail, so the gate is known to see the pairing and the concatenation.
func TestMTPMatchesReference(t *testing.T) {
	for _, tc := range []struct {
		golden, model string
		exact         bool
	}{
		{"synth-qwen35-hybrid-mtp", "synth-qwen35-hybrid-mtp.gguf", true},
		{"synth-qwen35moe-hybrid-mtp", "synth-qwen35moe-hybrid-mtp.gguf", true},
		{"synth-deepseek-mtp", "synth-deepseek-mtp", true},
		// scripts/qwen4expgold.py: the hyper-connected block, held to
		// transformers' modules wired as llama.cpp's graph_mtp.
		{"synth-qwen4exp-mtp", "synth-qwen4exp-mtp.gguf", true},
		{"Qwen3.5-0.8B", "qwen35/Qwen3.5-0.8B-MTP-Q8_0.gguf", false},
	} {
		t.Run(tc.golden, func(t *testing.T) {
			b, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "mtp", tc.golden+".json"))
			if err != nil {
				t.Fatalf("no golden (run scripts/mtpgold.py %s): %v", tc.golden, err)
			}
			var g mtpGolden
			if err := json.Unmarshal(b, &g); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(testmodels.Path(tc.model)); err != nil {
				t.Skipf("MODEL MISSING: %v (scripts/mtpgold.py builds it) -- this gate proved nothing", err)
			}
			m := openSpecModel(t, tc.model, noTune, WithKVF16(false))
			defer m.Close()
			clean := mtpLogits(t, m, g.IDS, 0)
			worst, agree := judgeMTP(t, m, g, clean, tc.exact, "clean")
			t.Logf("%s: worst NMSE %.3e, top-1 %d of %d", tc.golden, worst, agree, len(g.IDS))
			// The violations: the hidden state one row late (the pairing), and
			// each row's own token's embedding replaced by the next one's.
			for _, v := range []struct {
				name  string
				shift int
			}{{"hidden-one-row-late", 1}, {"next-token-embedding", 2}} {
				bad := mtpLogits(t, m, g.IDS, v.shift)
				w, _ := nmseRows(m, g, bad)
				if w < 1e-3 {
					t.Errorf("violation %s reads worst NMSE %.3e: the gate cannot see it", v.name, w)
				}
				t.Logf("violation %s: worst NMSE %.3e -- caught", v.name, w)
			}
		})
	}
}

// mtpLogits runs the trunk over ids and the draft over the rows the block
// reads, returning every row's logits. fault 1 pairs row q with h_{q-2} (the
// hidden state a row late); fault 2 embeds x_{q+1} in row q.
func mtpLogits(t *testing.T, m *Model, ids []int32, fault int) [][]float32 {
	t.Helper()
	c := m.Cfg
	st := m.NewState(len(ids) + 8)
	defer st.Close()
	sp, err := st.Speculate(WithSpecDraft(1))
	if err != nil {
		t.Fatal(err)
	}
	defer sp.Close()
	_, hid, err := sp.trunkPrompt(ids)
	if err != nil {
		t.Fatal(err)
	}
	n := len(ids)
	toks := append([]int32(nil), ids...)
	hw := c.specHidW()
	hs := make([]float32, n*hw)
	lag := 1
	if fault == 1 {
		lag = 2
	}
	for q := lag; q < n; q++ {
		copy(hs[q*hw:], hid[(q-lag)*hw:(q-lag+1)*hw])
	}
	if fault == 2 {
		copy(toks, ids[1:])
	}
	out := &rowsOut{from: 0, logits: make([]float32, n*c.NVocab), hidden: make([]float32, n*hw)}
	if err := sp.draftRows(toks, hs, out); err != nil {
		t.Fatal(err)
	}
	rows := make([][]float32, n)
	for q := range rows {
		rows[q] = out.logits[q*c.NVocab : (q+1)*c.NVocab]
	}
	return rows
}

// nmseRows is the worst per-row NMSE of got against the golden's MTP logits
// over the recorded prefix, and how many rows agree on the argmax.
func nmseRows(m *Model, g mtpGolden, got [][]float32) (worst float64, agree int) {
	for q, want := range g.MTP {
		var se, s2 float64
		for i, w := range want {
			d := float64(got[q][i]) - w
			se += d * d
			s2 += w * w
		}
		e := se / s2
		if math.IsNaN(e) || math.IsInf(e, 0) {
			return math.Inf(1), 0
		}
		worst = max(worst, e)
		if Greedy(got[q]) == g.MTPArgmax[q] {
			agree++
		}
	}
	return worst, agree
}

func judgeMTP(t *testing.T, m *Model, g mtpGolden, got [][]float32, exact bool, what string) (float64, int) {
	t.Helper()
	worst, agree := nmseRows(m, g, got)
	// The F32 fixtures are bound as the safetensors gate is: 1e-9 with an f32
	// cache. The Q8_0 model against bf16 weights widened to f32 is bound by
	// its quantization: a gross-error bound, and the argmax at most rows.
	bound, need := 1e-9, len(g.IDS)
	if !exact {
		bound, need = 2e-2, len(g.IDS)*3/4
	}
	if worst > bound || agree < need {
		t.Fatalf("%s: worst NMSE %.3e (bound %.0e), top-1 agrees at %d of %d (need %d)",
			what, worst, bound, agree, len(g.IDS), need)
	}
	return worst, agree
}
