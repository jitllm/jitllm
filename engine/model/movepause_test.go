package model

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// TestMovePause measures what a sequence waits when every block moves from
// one device of a tier to another mid-reply and back: the direct path
// (prepared while generating, the history copied device to device) against
// the host path (WithDirectMoves(false): the history home and up again, the
// target prepared inside the pause), alternated in one process. A measurement,
// not a gate (RULE 4): JITLLM_MOVE_PAUSE_MODEL names the model and opts in,
// JITLLM_TEST_DEVICES the two devices, JITLLM_MOVE_ROUNDS the rounds (3).
//
// Every arm is teacher-forced on an unmoved control's tokens in the same
// round and held to it: no prompt row computed again, and the logits bit for
// bit where both devices run the same kernels.
func TestMovePause(t *testing.T) {
	name := os.Getenv("JITLLM_MOVE_PAUSE_MODEL")
	if name == "" {
		t.Skip("JITLLM_MOVE_PAUSE_MODEL is unset: a measurement, run on request")
	}
	a, b := moveDevices(t)
	rounds := 3
	if v, err := strconv.Atoi(os.Getenv("JITLLM_MOVE_ROUNDS")); err == nil && v > 0 {
		rounds = v
	}
	path := jlmOf(t, testmodels.Path(name))
	const gen, moveAt, backAt = 240, 80, 160

	topics := []string{"tides", "volcanoes", "the printing press", "honeybees", "glaciers", "the telegraph",
		"coral reefs", "bridges", "sourdough", "lighthouses", "the monsoon", "mitochondria", "chess openings",
		"railways", "aurora", "tea", "the abacus", "cathedrals", "owls", "canals", "vaccines", "typefaces",
		"deserts", "the compass", "bread ovens", "satellites", "orchids", "clocks", "wind turbines", "caves"}
	var sb strings.Builder
	sb.WriteString("The following is a long conversation between a curious User and a helpful Assistant.\n\n")
	for i, tp := range topics {
		fmt.Fprintf(&sb, "User: Question %d. Tell me one surprising fact about %s, and why it matters.\n", i+1, tp)
		fmt.Fprintf(&sb, "Assistant: Fact %d: %s have a history that most people never hear about; "+
			"what matters is how %s shaped the people who depended on them, and how little of it was planned.\n\n", i+1, tp, tp)
	}
	sb.WriteString("User: Now, using everything above, list the first five topics we discussed, in order, and say which one you found most surprising.\nAssistant:")

	same := strings.SplitN(a, ":", 2)[0] == strings.SplitN(b, ":", 2)[0]
	// run is one arm on a fresh tier: direct or not, moving when move is set,
	// teacher-forced on ids when given.
	run := func(direct, move bool, ids []int32) ([]int32, [][]float32, []BlockMove, [][2]int, []time.Duration) {
		g, err := openTier(t, tier.WithDevices(a+","+b), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || g == nil {
			noDevice(t, a+","+b, err)
		}
		defer g.Close()
		distinctCards(t, g)
		m, err := Open(path, WithPlacement(Placement{Rest: &Place{On: a}, Head: &Place{On: a}}), WithDirectMoves(direct))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		in := m.Vocab.Encode(sb.String(), true)
		st := m.NewState(len(in) + gen + 8)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		placed := st.placedOn(a)
		if len(placed) != m.Cfg.NLayer {
			t.Fatalf("%s took %d of %d blocks: %s", a, len(placed), m.Cfg.NLayer, g.Err())
		}
		logits, err := st.Prefill(in)
		if err != nil {
			t.Fatal(err)
		}
		var out []int32
		var lg [][]float32
		var moves []BlockMove
		var during [][2]int // the tokens generated while each move prepared
		var took []time.Duration
		var pm *PendingMove
		n := 0
		for i := 0; i < gen; i++ {
			lg = append(lg, append([]float32(nil), logits...))
			next := Greedy(logits)
			if ids != nil {
				next = ids[i]
			}
			out = append(out, next)
			if move && (i == moveAt || i == backAt) {
				to := b
				if i == backAt {
					to = a
				}
				if pm, err = st.PrepareMove(0, m.Cfg.NLayer, to); err != nil {
					t.Fatal(err)
				}
				n = 0
			}
			if pm != nil && (pm.Ready() || i == backAt-1 || i == gen-1) {
				rows := st.PrefillRows()
				mv, err := pm.Commit()
				if err != nil {
					t.Fatal(err)
				}
				if got := st.PrefillRows() - rows; got != 0 {
					t.Fatalf("the move computed %d prompt rows again", got)
				}
				pm = nil
				moves, during = append(moves, mv), append(during, [2]int{i - n, i})
			}
			n++
			t0 := time.Now()
			if logits, err = st.Forward(next); err != nil {
				t.Fatal(err)
			}
			took = append(took, time.Since(t0))
		}
		return out, lg, moves, during, took
	}
	// mean is the mean token time over tokens [lo, hi), in ms.
	mean := func(d []time.Duration, lo, hi int) float64 {
		if hi <= lo {
			return 0
		}
		var sum time.Duration
		for _, x := range d[lo:hi] {
			sum += x
		}
		return float64(sum.Microseconds()) / 1000 / float64(hi-lo)
	}

	for r := 0; r < rounds; r++ {
		ctl, ctlLogits, _, _, ctlTook := run(true, false, nil)
		// The two arms alternate their order round to round.
		arms := []bool{true, false}
		if r%2 == 1 {
			arms = []bool{false, true}
		}
		for _, direct := range arms {
			_, lg, moves, during, took := run(direct, true, ctl)
			var worst float64
			for i := range ctlLogits {
				for j, v := range ctlLogits[i] {
					worst = max(worst, math.Abs(float64(v-lg[i][j])))
				}
			}
			if math.IsNaN(worst) || same && worst != 0 {
				t.Fatalf("round %d direct=%v: the moved logits differ from the control's by %g", r, direct, worst)
			}
			arm := "host  "
			if direct {
				arm = "direct"
			}
			for i, mv := range moves {
				lo, hi := during[i][0], during[i][1]
				t.Logf("round %d %s move %d: pause %8.1f ms, %d blocks, KV %d B, copied %d B via %q, head %v, "+
					"%d staged in %.1f ms beside %d tokens (%.2f ms/token, unmoved %.2f); first token after %.2f ms "+
					"(unmoved %.2f); max |dlogit| %g",
					r, arm, i, float64(mv.Pause.Microseconds())/1000, len(mv.Moved), mv.KVBytes, mv.Copied,
					mv.Path, mv.Head, mv.Staged, float64(mv.Prepare.Microseconds())/1000, hi-lo,
					mean(took, lo, hi), mean(ctlTook, lo, hi), mean(took, hi, hi+1), mean(ctlTook, hi, hi+1), worst)
			}
			t.Logf("round %d %s: ms/token %.2f before, %.2f between, %.2f after the moves (unmoved %.2f, %.2f, %.2f)",
				r, arm, mean(took, 10, moveAt), mean(took, during[0][1]+8, backAt), mean(took, during[1][1]+8, gen),
				mean(ctlTook, 10, moveAt), mean(ctlTook, during[0][1]+8, backAt), mean(ctlTook, during[1][1]+8, gen))
		}
	}
}
