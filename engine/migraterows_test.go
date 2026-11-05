package engine

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"reflect"
	"runtime"
	"testing"
	"time"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
)

// TestBatchSeamMovesCarryEveryRow: a batch's blocks moving between the device
// and the host mid-decode must carry every row's history, not row 0's. The
// device keeps a batch's rows in one cache per block (row r at r*maxSeq+pos)
// and one recurrent state per row; the host keeps every row in each page and
// per-row rconv/rstate. A migration that moves slot 0 alone leaves the other
// rows attending over nothing after the move: fluent, and wrong.
//
// Three rows at three different positions (two retired and restarted along
// the way), teacher-forced, against a host-only batch driven identically. The
// arms: the seam shrinks (every block on the device, then half, then none) and
// grows back; a batch that starts on the host grows onto the device; and
// relocation hands a block home because the batch's history does not fit, and
// reclaims it with every row's history once the room comes back.
func TestBatchSeamMovesCarryEveryRow(t *testing.T) {
	w := fusedRunsCheck(t, false)
	defer w.done()
	batchSeamMoves(t, testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf"), 1e-1, true, w)
}

// TestBatchSeamMovesCarryEveryRowHybrid is the same on a hybrid, where a linear
// block's history is a per-row recurrent summary (MigrateRec) beside the
// attention blocks' KV.
//
// Without the relocation arm: relocation prices the KV alone (ReserveKV), and
// a hybrid batch's first step also grows every linear block's state to the
// batch (growSeat), so a budget short of the KV refuses that growth instead of
// relocating. The moves themselves are the other arms'.
func TestBatchSeamMovesCarryEveryRowHybrid(t *testing.T) {
	w := fusedRunsCheck(t, false)
	defer w.done()
	batchSeamMoves(t, testmodels.Path("qwen35/Qwen3.5-0.8B-Q4_K_M.gguf"), 3e-2, false, w)
}

// TestBatchSeamMovesCarryEveryRowMoE is the same on a mixture, whose blocks
// carry a router, expert banks and their grouped scratch on the device.
func TestBatchSeamMovesCarryEveryRowMoE(t *testing.T) {
	batchSeamMoves(t, testmodels.Path("Qwen3-MOE-4x0.6B-Q4_K_M.gguf"), 1e-1, false, nil)
}

// TestBatchSeamMovesCarryEveryRowMoEFamilies is the same on the mixture
// families' fixtures (moeFixtures): a dense lead block (glm4moe, ernie4_5), a
// gated shared expert (qwen2moe), a selection bias, DeepSeek V3.2's indexer key
// (carried in the latent row, so a move carries it), and relocation. They are
// F32, so the device's own arithmetic is far under the bound.
func TestBatchSeamMovesCarryEveryRowMoEFamilies(t *testing.T) {
	for _, name := range []string{"synth-glm4moe", "synth-qwen2moe", "synth-ernie45moe", "synth-minimaxm2", "synth-gemma4-moe",
		"synth-deepseek32", "synth-minimaxm3",
		"synth-bailingmoe2", "synth-dots1", "synth-phimoe", "synth-kimik3"} {
		t.Run(name, func(t *testing.T) {
			p, ok := existingModel(testmodels.Path(name + ".gguf"))
			if !ok {
				testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path(name+".gguf")+
					" (set JITLLM_MODELS to the model directory) -- run "+moeGoldScript+" "+name+" (RULE 11)")
			}
			batchSeamMoves(t, p, 1e-3, true, nil)
		})
	}
}

// TestBatchSeamMovesCarryEveryRowHunyuan is the same on Hunyuan, whose k is
// cached unweighted on every tier (jlm.FlagQKNormPostRope folds its weight into
// q's): a history written by the device must be read by the host and back
// again, so a seam move is where a tier caching another k would show. The
// fixtures are F32 and take the families' bound (1.1e-06 and 2.1e-06 before
// the first move, 8.5e-07 and 1.5e-06 after); the real Hunyuan-0.5B's own
// device arithmetic is wider than Llama's -- 3.0e-02 to 4.5e-02 before the
// first move and 6.2e-02 to 7.1e-02 after on an RTX 3050 Ti -- and its bound
// leaves a V100's twofold spread room under the 0.73 a lost history read on
// Llama.
func TestBatchSeamMovesCarryEveryRowHunyuan(t *testing.T) {
	for _, c := range []struct {
		name  string
		bound float64
	}{{"synth-hunyuan.gguf", 1e-3}, {"synth-hunyuanmoe.gguf", 1e-3},
		{"hunyuan/tencent_Hunyuan-0.5B-Instruct-Q4_K_M.gguf", 3e-1}} {
		t.Run(c.name, func(t *testing.T) {
			p, ok := existingModel(testmodels.Path(c.name))
			if !ok {
				testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path(c.name)+
					" (set JITLLM_MODELS to the model directory) -- RULE 11")
			}
			batchSeamMoves(t, p, c.bound, true, nil)
		})
	}
}

// batchSeamMoves runs the arms on the model at p. bound is the logit NMSE a
// row may differ from the host-only batch by after a move, set by the
// violations: before the first move the device's own arithmetic is the whole
// difference (int8 activations, another reduction order), and a row that lost
// its history is a step change on top of it. The worst row, after the first
// move, on an RTX 3050 Ti (a V100's clean arms in brackets):
//
//	                 clean              rows' KV lost    rows' summary lost
//	Llama-3.2-1B     8.0e-03 (1.5e-02)  1.6 - 2.9        --
//	  relocation     8.7e-03 (1.3e-02)  0.73             --
//	Qwen3.5-0.8B     1.8e-03 (2.6e-03)  7.8e-02 - 0.14   1.6 - 2.6
//
// The V100's device arithmetic alone reads 1.3e-02 on Llama and 8.3e-03 on the
// hybrid before any move, which is what the bounds sit above.
func batchSeamMoves(t *testing.T, p string, bound float64, relocate bool, w *fuseWatch) {
	batchSeamMovesWith(t, p, bound, relocate, nil, nil, w)
}

// batchSeamMovesWith is batchSeamMoves with model options added and a check
// handed each arm's device stats once its steps have run -- the count that
// says the configuration under test was the one that ran (RULE 10).
func batchSeamMovesWith(t *testing.T, p string, bound float64, relocate bool, mopts []Option,
	check func(t *testing.T, st gpu.Stats), w *fuseWatch) {
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	// No KV prewarm: the squeezed arm measures what attaching spends and
	// needs the rows' growth to outrun it, which pages grown ahead would cover.
	instOpts := append([]Option{WithKVPrewarm(false)}, mopts...)
	m, err := model.Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	instOpts = append(instOpts, w.opt())
	c := m.Cfg
	const rows, steps = 3, 21
	texts := []string{
		"Once upon a time, in a small village by the sea, there lived an old fisherman who had a small boat and a very old dog",
		"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps",
		"1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,",
	}
	stream := make([][]int32, rows)
	for i, s := range texts {
		stream[i] = m.Vocab.Encode(s, true)
		if len(stream[i]) < steps {
			t.Fatalf("row %d's text is %d tokens, the script needs %d", i, len(stream[i]), steps)
		}
	}
	half := c.NLayer / 2
	// A KV-sharing model's seam may not part a source from its readers
	// (fixKVGroups): the half is the nearest cut below that does not.
	for half > 0 && !kvLegalCut(c, half) {
		half--
	}
	type move struct{ at, n int }
	// The first move is at step 8 in every arm; before it the device's own
	// arithmetic is the whole difference.
	const first = 8
	for _, arm := range []struct {
		name  string
		start int // blocks placed at attach
		moves []move
		// squeeze attaches under a budget that holds every block but only half
		// the batch's history, so the attach's reservation hands blocks to the
		// host for this instance (reserveRows); at step `first` the budget comes
		// back and a step reclaims them, sending the rows' history up.
		squeeze bool
		// seq is each row's capacity. The squeezed arm's is wider, so the
		// history that does not fit costs several blocks rather than one.
		seq int
	}{
		// Down to half (a split: the top half comes home), then none, then
		// every block back up with three rows of history -- twice, so the
		// second round trip can be held to the first (see again).
		{"shrink-then-grow", -1, []move{{first, half}, {10, 0}, {12, c.NLayer},
			{14, half}, {16, 0}, {18, c.NLayer}}, false, 64},
		// A batch that ran on the host moves onto the device mid-decode.
		{"grow", 0, []move{{first, c.NLayer}}, false, 64},
		{"relocate-then-reclaim", -1, nil, true, 1024},
	} {
		if arm.squeeze && !relocate {
			continue
		}
		t.Run(arm.name, func(t *testing.T) {
			gr0 := armGoroutines()
			// Retires put the rows at different positions: row 1 restarts at
			// step 3, row 2 at step 6, so at the first move they sit at 8, 5
			// and 2.
			drive := func(st *sessBatch, each func(step int)) [][]float32 {
				out := make([][]float32, 0, steps)
				for step := 0; step < steps; step++ {
					switch step {
					case 3:
						st.Retire(1)
					case 6:
						st.Retire(2)
					}
					each(step)
					toks := make([]int32, rows)
					for r := range toks {
						toks[r] = stream[r][st.SeqPos(r)]
					}
					lg, err := st.ForwardBatch(toks)
					if err != nil {
						t.Fatalf("step %d: %v", step, err)
					}
					out = append(out, append([]float32(nil), lg[:rows*c.NVocab]...))
				}
				return out
			}
			host := newSessBatch(m, rows, arm.seq, instOpts...)
			// A forced cache format is the configuration under test: one that
			// was not selected would run the f32 arms again.
			if host.opt.kvTypeSet && host.KVType() != host.opt.kvType {
				t.Fatalf("asked for a %v KV cache and the batch holds %v", host.opt.kvType, host.KVType())
			}
			want := drive(host, func(int) {})
			host.Close()

			open := func(opts ...gpu.Option) *gpu.GPU {
				g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices("gpu:0"),
					gpu.WithDeviceTune(gpu.TuneOff)}, opts...)...)
				if err != nil || g == nil {
					noDevice(t, "device", err)
				}
				return g
			}
			// The squeezed budget is measured: every block and half the rows'
			// history, from an attach with room for all of it.
			var squeezed, limit uint64
			paged := false
			if arm.squeeze {
				pg := open()
				ps := newSessBatch(m, rows, arm.seq, instOpts...)
				if err := ps.SetDeviceLayers(pg, arm.start); err != nil {
					t.Fatal(err)
				}
				squeezed, limit = pg.Bytes()-pg.Stats().KVBytes/2, pg.Budgets()[0].Limit
				if sd, ok := ps.ld.(nn.SeqKVDevice); ok && sd.PerSequenceKV() {
					// Paged: nothing is reserved at attach, so the squeeze is the
					// attach exactly -- every block and the pool's working set --
					// and the rows' first pages do not fit. The relocation then
					// happens during a step, not at the attach.
					squeezed, paged = pg.Bytes(), true
				}
				ps.Close()
				pg.Close()
			}
			var g *gpu.GPU
			if arm.squeeze {
				g = open(gpu.WithBudget(squeezed))
			} else {
				g = open()
			}
			st := newSessBatch(m, rows, arm.seq, instOpts...)
			// The arm closes both itself, as part of its leak checks; this is
			// for a failure before that, which would otherwise hold the card
			// for every test after it.
			closed := false
			t.Cleanup(func() {
				if !closed {
					st.Close()
					g.Close()
				}
			})
			if err := st.SetDeviceLayers(g, arm.start); err != nil {
				t.Fatal(err)
			}
			if arm.squeeze && !paged {
				// Relocation is the configuration under test: the attach must
				// have handed blocks home, and the rest of the rows' history
				// must then fit.
				if st.Relocations() == 0 || st.devCount() == c.NLayer || st.devCount() == 0 {
					t.Fatalf("the squeezed attach relocated %d block(s), %d of %d placed: "+
						"the arm tested no relocation (%q)", st.Relocations(), st.devCount(), c.NLayer, g.Err())
				}
				t.Logf("attach: %d block(s) relocated, %d of %d placed", st.Relocations(), st.devCount(), c.NLayer)
			}
			if want := max(arm.start, 0); arm.start >= 0 && st.devCount() != want {
				t.Fatalf("attached with %d blocks placed, asked for %d", st.devCount(), want)
			}
			if arm.start < 0 && !arm.squeeze && (st.devCount() != c.NLayer || !st.HeadOnDevice()) {
				t.Fatalf("placed %d of %d blocks, head on a device %v: %q -- the full-device arm "+
					"would test a split", st.devCount(), c.NLayer, st.HeadOnDevice(), g.Err())
			}
			// Steady state before the first move (every block placed, three
			// rows grown) and after the first round trip.
			const again = 13
			var s0, s1 gpu.Stats
			got := drive(st, func(step int) {
				if len(arm.moves) > 0 && arm.start < 0 {
					switch step {
					case first - 1:
						s0 = g.Stats()
					case again:
						s1 = g.Stats()
					}
				}
				// A squeezed card holds what it was given: a step's scratch
				// charged past the budget is room taken that relocation was
				// there to make (a 3-row step's per-row logits over the
				// 512-row prompt scratch put synth-deepseek32 at 3.6 times
				// its budget, and nothing relocated).
				if arm.squeeze && step > 0 && step <= first {
					if b := g.Bytes(); b > squeezed {
						t.Fatalf("step %d: the device charges %d bytes against a budget of %d (%q)",
							step, b, squeezed, g.Err())
					}
				}
				if arm.squeeze && step == first {
					if paged && st.Relocations() == 0 {
						t.Fatalf("step %d: the rows outgrew a card squeezed to the attach and no "+
							"block relocated: the arm tested no relocation (%q)", step, g.Err())
					}
					t.Logf("step %d: %d placed, rows at %d %d %d; the budget comes back",
						step, st.devCount(), st.SeqPos(0), st.SeqPos(1), st.SeqPos(2))
					if _, err := g.SetBudget(limit); err != nil {
						t.Fatal(err)
					}
				}
				for _, mv := range arm.moves {
					if step != mv.at {
						continue
					}
					pos := []int{st.SeqPos(0), st.SeqPos(1), st.SeqPos(2)}
					lin := 0
					for _, li := range st.placedFrom(0) {
						if c.LayerKind(li).Recurrent() {
							lin++
						}
					}
					if got := st.SetGPULayers(mv.n); got != mv.n || st.devCount() != mv.n {
						t.Fatalf("step %d: SetGPULayers(%d) left %d placed (%d on devices): %q",
							step, mv.n, got, st.devCount(), g.Err())
					}
					ls := g.Stats()
					t.Logf("step %d: seam -> %d with rows at %v (%d linear block(s) were placed); "+
						"device KV %d B, rec %d B, %d B in use", step, mv.n, pos, lin, ls.KVBytes, ls.RecBytes, g.Bytes())
					if mv.n == 0 && (ls.KVBytes != 0 || ls.RecBytes != 0) {
						t.Errorf("step %d: every block came home and the device still holds "+
							"%d B of KV and %d B of recurrent state", step, ls.KVBytes, ls.RecBytes)
					}
				}
			})
			if check != nil {
				check(t, g.Stats())
			}
			if arm.squeeze {
				// The step after the budget came back reclaimed the blocks it
				// could, sending the rows' history up with them.
				if st.Reclaims() == 0 {
					t.Fatalf("no block was reclaimed after the budget came back (%d placed): "+
						"the arm tested no move onto the device (%q)", st.devCount(), g.Err())
				}
				t.Logf("%d block(s) reclaimed, %d of %d placed at the end", st.Reclaims(), st.devCount(), c.NLayer)
			}
			if len(arm.moves) > 0 && arm.start < 0 {
				// A round trip ends where it started: every block placed, three
				// rows grown, and no buffer or charge a move left behind. The
				// first may leave the device's own staging grown: while the
				// blocks are home the host runs them, and a mixture's router
				// is a lone matvec the device serves (GPU.MatVec), sizing a
				// partial buffer the placed blocks never asked for. Staging
				// only grows to the widest shape asked, so the first trip is
				// held exactly outside the scratch and the second exactly
				// everywhere, the driver's rounding included.
				// The composed prompt arenas are scratch too: a width's
				// buffers every prompt form shares, grown to the widest.
				outside := func(s gpu.Stats) (alloc, charged uint64) {
					return s.Allocated - s.ScratchBytes - s.ComposedArena, s.BudgetUsed - s.RoundingBytes - s.ScratchBytes - s.ComposedArena
				}
				a0, c0 := outside(s0)
				a1, c1 := outside(s1)
				if a1 != a0 || c1 != c0 {
					t.Errorf("after the first round trip the device's buffers outside its scratch hold %d B "+
						"(%+d) and its charges %d B (%+d): a move left a buffer or a charge behind (moved: %s)",
						a1, int64(a1)-int64(a0), c1, int64(c1)-int64(c0), statsMoved(s0, s1))
				}
				if ds := int64(s1.ScratchBytes) - int64(s0.ScratchBytes); ds != 0 {
					t.Logf("the first round trip grew the device's own scratch by %d B", ds)
				}
				if b := g.Bytes(); b != s1.BudgetUsed {
					t.Errorf("device bytes after the second round trip %d, after the first %d (%+d)",
						b, s1.BudgetUsed, int64(b)-int64(s1.BudgetUsed))
				}
			}
			closed = true
			if err := st.Close(); err != nil {
				t.Fatal(err)
			}
			if ls := g.Stats(); ls.KVBytes != 0 || ls.RecBytes != 0 {
				t.Errorf("after the instance closed the device still holds %d B of KV and %d B "+
					"of recurrent state", ls.KVBytes, ls.RecBytes)
			}
			g.Close()
			// Nothing the moves started is left running.
			deadline := time.Now().Add(3 * time.Second)
			for armGoroutines() > gr0 && time.Now().Before(deadline) {
				time.Sleep(20 * time.Millisecond)
			}
			if n := armGoroutines(); n > gr0 {
				buf := make([]byte, 1<<16)
				t.Errorf("%d goroutine(s) before the arm, %d after:\n%s", gr0, n, buf[:runtime.Stack(buf, true)])
			}

			worst, wat, wrow := 0.0, -1, -1
			before := 0.0
			for step := range want {
				for r := 0; r < rows; r++ {
					a := want[step][r*c.NVocab : (r+1)*c.NVocab]
					b := got[step][r*c.NVocab : (r+1)*c.NVocab]
					var num, den float64
					for i := range a {
						if math.IsNaN(float64(b[i])) || math.IsInf(float64(b[i]), 0) {
							t.Fatalf("step %d row %d logit %d is %g: not finite", step, r, i, b[i])
						}
						d := float64(b[i] - a[i])
						num += d * d
						den += float64(a[i]) * float64(a[i])
					}
					e := num / den
					if step < first {
						before = max(before, e)
						continue
					}
					if e > worst {
						worst, wat, wrow = e, step, r
					}
				}
			}
			t.Logf("worst logit NMSE: %.3e before the first move, %.3e after (step %d row %d)",
				before, worst, wat, wrow)
			if worst > bound {
				t.Errorf("step %d row %d: logit NMSE %.3e after a seam move, bound %.0e -- "+
					"a row's history did not move with its blocks", wat, wrow, worst, bound)
			}
		})
	}
}

// kvLegalCut reports that a seam at n keeps every KV-sharing block on the
// side of its source.
func kvLegalCut(c *model.Config, n int) bool {
	for li := c.NLayer - c.NKVShared; li < c.NLayer; li++ {
		if (li < n) != (c.KVSource(li) < n) {
			return false
		}
	}
	return true
}

// TestBatchSeamMovesCarryEveryRowSSM is TestBatchSeamMovesCarryEveryRowHybrid
// on the Mamba-2 fixtures: three rows decoding while the seam moves under
// them, each row's convolution window and state carried across by MigrateRec.
// They are F32, so the device's own arithmetic is far under the bound.
func TestBatchSeamMovesCarryEveryRowSSM(t *testing.T) {
	w := fusedRunsCheck(t, false)
	defer w.done()
	for _, fx := range ssmFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			p, ok := existingModel(testmodels.Path(fx.name + ".gguf"))
			if !ok {
				testmodels.Missing(t, "%s", "MODEL MISSING: "+fx.name+" -- run "+ssmGoldScript)
			}
			batchSeamMoves(t, p, 1e-4, false, w)
		})
	}
}

// armGoroutines is the goroutines running but the container's page-I/O
// workers: the file starts them at its first partial read, wherever that
// falls -- inside an arm on a host whose budget holds the whole model -- and
// they end with its Close, not with a move.
func armGoroutines() int {
	buf := make([]byte, 1<<20)
	buf = buf[:runtime.Stack(buf, true)]
	return runtime.NumGoroutine() - bytes.Count(buf, []byte("format/jlm.(*File).startIO.func1("))
}

// statsMoved names the integer fields of two snapshots that differ, with
// the change.
func statsMoved(a, b gpu.Stats) string {
	va, vb := reflect.ValueOf(a), reflect.ValueOf(b)
	var out []byte
	for i := range va.NumField() {
		fa, fb := va.Field(i), vb.Field(i)
		var d int64
		switch fa.Kind() {
		case reflect.Int, reflect.Int64, reflect.Int32:
			d = fb.Int() - fa.Int()
		case reflect.Uint, reflect.Uint64, reflect.Uint32, reflect.Uintptr:
			d = int64(fb.Uint()) - int64(fa.Uint())
		default:
			continue
		}
		if d != 0 {
			out = fmt.Appendf(out, "%s %+d; ", va.Type().Field(i).Name, d)
		}
	}
	return string(out)
}
