package engine

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestTwoInstancesRunABlockOnTwoCards places two instances of one model on one
// card of a two-card tier, both prefilled and decoding, and moves the upper
// half of the second's blocks to the other card mid-generation. The first
// instance's sessions keep
// their history on the copy where it is, so the tier gives the second a copy
// of its own: each instance names its own card for every block, the tier
// holds the weights twice, and each decodes its own control's tokens,
// teacher-forced -- the unmoved first bit for bit, the second bit for bit
// between two cards running the same kernels and in the reduction-order band
// between two backends. The second's copy goes when it closes, the second
// card back to what it held before the move, and the first decodes on. The
// violation is the tier's one route a block: before it ran per session, this
// move was refused ("another session is using it").
func TestTwoInstancesRunABlockOnTwoCards(t *testing.T) {
	for _, name := range append([]string{"Llama-3.2-1B-Instruct-Q4_K_M.gguf"}, relocModels()...) {
		path := jlmOf(t, testmodels.Path(name))
		t.Run(name, func(t *testing.T) { twoInstancesRunABlockOnTwoCards(t, path) })
	}
}

func twoInstancesRunABlockOnTwoCards(t *testing.T, path string) {
	a, b := moveDevices(t)
	g, err := openGPU(t, gpu.WithDevices(a+","+b), gpu.WithDeviceTune(gpu.TuneOff))
	if err != nil || g == nil {
		noDevice(t, a+","+b, err)
	}
	defer g.Close()
	distinctCards(t, g)
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	opts := []Option{WithPlacement(Placement{Rest: &Place{On: a}, Head: &Place{On: a}})}
	const ntok, moveAt = 10, 3
	prompts := [2][]int32{
		m.Vocab.Encode("The capital of France is a city that", true),
		m.Vocab.Encode("Once upon a time, in a small village by the sea", true),
	}
	maxSeq := max(len(prompts[0]), len(prompts[1])) + ntok + 1
	open := func() *Instance {
		t.Helper()
		st := testInstance(t, m, maxSeq, opts...)
		if err := st.SetDevice(g); err != nil {
			st.Close()
			t.Fatal(err)
		}
		if len(st.placedOn(a)) == 0 {
			st.Close()
			t.Skipf("%s took no block of this model: %s", a, g.Err())
		}
		return st
	}
	// Each prompt alone, never moved: the controls.
	var ctl [2][]int32
	var ctlLogits [2][][]float32
	for i, p := range prompts {
		st := open()
		lg, err := st.Prefill(p)
		for j := 0; err == nil && j < ntok; j++ {
			ctlLogits[i] = append(ctlLogits[i], slices.Clone(lg))
			ctl[i] = append(ctl[i], Greedy(lg))
			lg, err = st.Forward(ctl[i][j])
		}
		st.Close()
		if err != nil {
			t.Fatal(err)
		}
	}
	x, y := open(), open()
	defer x.Close()
	yOpen := true
	defer func() {
		if yOpen {
			y.Close()
		}
	}()
	lx, err := x.Prefill(prompts[0])
	if err != nil {
		t.Fatal(err)
	}
	ly, err := y.Prefill(prompts[1])
	if err != nil {
		t.Fatal(err)
	}
	// The upper half: the projection stays with the lower one, on a for
	// both instances (the tier holds one projection).
	placed := y.placedOn(a)
	if len(placed) < 2 {
		t.Skipf("%s took %d block(s); the upper half of one is all of it", a, len(placed))
	}
	placed = placed[len(placed)/2:]
	lo, hi := placed[0], placed[len(placed)-1]+1
	for _, li := range placed {
		glo, ghi := y.kvSpan(li)
		lo, hi = min(lo, glo), max(hi, ghi)
	}
	one, onB := g.Bytes(), g.DevStats()[1].BudgetUsed
	var got [2][][]float32
	for j := range ntok {
		got[0] = append(got[0], slices.Clone(lx))
		got[1] = append(got[1], slices.Clone(ly))
		if j == moveAt {
			mv, err := y.MoveBlocks(lo, hi, b)
			if err != nil {
				t.Fatalf("blocks %d..%d of the second instance to %s: %v", lo, hi-1, b, err)
			}
			if len(mv.Moved) == 0 {
				t.Fatalf("no block of the second instance moved to %s: %+v", b, mv)
			}
			for _, li := range placed {
				if at, _ := y.DeviceOf(li); at != b {
					t.Fatalf("the second instance's block %d is on %q after its move to %s", li, at, b)
				}
				if at, _ := x.DeviceOf(li); at != a {
					t.Fatalf("the first instance's block %d is on %q: the second's move took it from %s", li, at, a)
				}
			}
			if two := g.Bytes(); two <= one {
				t.Fatalf("the tier holds %d bytes of weights after the move, %d before: no second copy", two, one)
			}
			t.Logf("%d block(s) moved to %s, %d KV bytes by %q; weights %d -> %d bytes", len(mv.Moved), b,
				mv.KVBytes, mv.Path, one, g.Bytes())
		}
		if lx, err = x.Forward(ctl[0][j]); err != nil {
			t.Fatal(err)
		}
		if ly, err = y.Forward(ctl[1][j]); err != nil {
			t.Fatal(err)
		}
	}
	same := strings.SplitN(a, ":", 2)[0] == strings.SplitN(b, ":", 2)[0]
	for i := range 2 {
		var band float64
		for j := range ctlLogits[i] {
			var worst float64
			for k, v := range ctlLogits[i][j] {
				worst = max(worst, math.Abs(float64(v-got[i][j][k])))
			}
			band = max(band, worst)
			switch {
			case math.IsNaN(worst):
				t.Fatalf("instance %d, token %d: a non-finite logit", i, j)
			case (i == 0 || j <= moveAt || same) && worst != 0:
				t.Fatalf("instance %d, token %d differs from its control by %g", i, j, worst)
			case worst > 2.5:
				t.Fatalf("instance %d, token %d: %g from its control, outside the band", i, j, worst)
			}
		}
		t.Logf("instance %d: largest logit difference from its control %g", i, band)
	}
	// The second's copy goes with it; the first decodes on.
	y.Close()
	yOpen = false
	// Only the second instance's moved blocks were ever on b, and b gives up
	// what it held for the placement with its last block (dropIfEmpty): it
	// ends at no more than before the move. A copy left behind is past it.
	// The first card also drops the second's own history and scratch, so it
	// has no bound.
	if left := g.DevStats()[1].BudgetUsed; left > onB {
		t.Fatalf("%s holds %d bytes after the second instance closed, past the %d before its move: the second's copy stayed", b, left, onB)
	}
	if _, err := x.Forward(ctl[0][ntok-1]); err != nil {
		t.Fatal(err)
	}
}
