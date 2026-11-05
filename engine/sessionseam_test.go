package engine

import (
	"os"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestSessionsFollowTheirInstanceAcrossSeamMoves: an instance's blocks
// moving between the card and the CPU mid-decode carry the history of every
// session leasing it, not the one it ran for last. Three sessions at three
// positions (two retired and opened again along the way), teacher-forced and
// stepped together, against the same sessions on a CPU-only instance. The
// arms: the seam shrinks (every block on the card, then half, then none) and
// grows back, twice; and an instance that ran on the CPU grows onto the card.
func TestSessionsFollowTheirInstanceAcrossSeamMoves(t *testing.T) {
	for _, c := range []struct {
		file  string
		bound float64
	}{{"Llama-3.2-1B-Instruct-Q4_K_M.gguf", 1e-1}, {"qwen35/Qwen3.5-0.8B-Q4_K_M.gguf", 3e-2}} {
		t.Run(c.file, func(t *testing.T) { sessionSeamMoves(t, testmodels.Path(c.file), c.bound) })
	}
}

func sessionSeamMoves(t *testing.T, p string, bound float64) {
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	instOpts := []Option{WithKVPrewarm(false)}
	m, err := model.Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	cfg := m.Cfg
	const rows, steps, seq = 3, 21, 64
	texts := []string{
		"Once upon a time, in a small village by the sea, there lived an old fisherman who had a small boat and a very old dog",
		"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps",
		"1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,",
	}
	stream := make([][]int32, rows)
	for i, s := range texts {
		if stream[i] = m.Vocab.Encode(s, true); len(stream[i]) < steps {
			t.Fatalf("row %d's text is %d tokens, the script needs %d", i, len(stream[i]), steps)
		}
	}
	half := cfg.NLayer / 2
	for half > 0 && !kvLegalCut(cfg, half) {
		half--
	}
	type move struct{ at, n int }
	// drive steps the sessions on in, the instance's seam moving as moves
	// say, and returns every step's logits, a row each.
	drive := func(in *Instance, moves []move) [][]float32 {
		pool := NewPool()
		if err := pool.Serve(in, rows); err != nil {
			t.Fatal(err)
		}
		defer pool.Unserve(in)
		sess := make([]*Session, rows)
		pos := make([]int, rows)
		open := func(r int) {
			if sess[r] != nil {
				sess[r].Close()
			}
			s, err := pool.NewSession(seq)
			if err != nil {
				t.Fatal(err)
			}
			if err := s.Acquire(t.Context(), PhaseGenerate); err != nil {
				t.Fatal(err)
			}
			sess[r], pos[r] = s, 0
		}
		for r := range sess {
			open(r)
		}
		defer func() {
			for _, s := range sess {
				s.Close()
			}
		}()
		out := make([][]float32, 0, steps)
		runs := make([]Run, rows)
		for step := 0; step < steps; step++ {
			switch step {
			case 3:
				open(1)
			case 6:
				open(2)
			}
			for _, mv := range moves {
				if mv.at == step {
					in.With(func(in *Instance) { in.SetGPULayers(mv.n) })
				}
			}
			for r := range runs {
				runs[r] = Run{Session: sess[r], Tokens: stream[r][pos[r] : pos[r]+1], Logits: true}
				pos[r]++
			}
			lg, err := StepRuns(runs)
			if err != nil {
				t.Fatalf("step %d: %v", step, err)
			}
			row := make([]float32, 0, rows*cfg.NVocab)
			for r := range lg {
				row = append(row, lg[r]...)
			}
			out = append(out, row)
		}
		return out
	}
	host := testInstance(t, m, seq, instOpts...)
	want := drive(host, nil)
	host.Close()
	const first = 8
	for _, arm := range []struct {
		name  string
		start int
		moves []move
	}{
		{"shrink-then-grow", -1, []move{{first, half}, {10, 0}, {12, cfg.NLayer}, {14, half}, {16, 0}, {18, cfg.NLayer}}},
		{"grow", 0, []move{{first, cfg.NLayer}}},
	} {
		t.Run(arm.name, func(t *testing.T) {
			g, err := openGPU(t, gpu.WithDevices("gpu:0"), gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || g == nil {
				noDevice(t, "device", err)
			}
			defer g.Close()
			in := testInstance(t, m, seq, instOpts...)
			defer in.Close()
			if err := in.SetDeviceLayers(g, arm.start); err != nil {
				t.Fatal(err)
			}
			got := drive(in, arm.moves)
			moved := 0
			for _, mv := range arm.moves {
				if mv.at < steps {
					moved++
				}
			}
			if moved == 0 || g.Stats().SessionRows == 0 {
				t.Fatalf("no move ran or no row ran on the card across sessions (%d moves, %d rows)", moved, g.Stats().SessionRows)
			}
			for step := range got {
				if e := logitNMSE(got[step], want[step]); !(e <= bound) {
					t.Fatalf("step %d: the sessions' logits read NMSE %.3e against the CPU's, past %.0e", step, e, bound)
				}
			}
			t.Logf("%d steps of %d sessions within %.0e of the CPU across %d seam moves", steps, rows, bound, moved)
		})
	}
}
