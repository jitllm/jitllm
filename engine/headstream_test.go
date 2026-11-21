package engine

import (
	"math"
	"os"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
)

// TestTheHeadStreams places tinyllama's blocks and its output projection on
// the card, the projection streamed (head=cuda:0~) beside streamed blocks on a
// budget of a few slots, and decodes the same ids as the projection resident
// on a card with room and as the CPU.
//
// The streamed projection is a page: in for the call that reads it, and a
// victim when a block's page-in wants its room, so on this budget every token
// pages it in again (Stats.HeadPageIns). Its logits are the resident head's
// (the same forms over the same bytes, so the same greedy token everywhere),
// and against the CPU it stays in the band the resident head reads.
// tinyllama ties nothing: its projection is no input row's table.
func TestTheHeadStreams(t *testing.T) {
	testdevices.Require(t, "cuda:0")
	p := testmodels.Path("tinyllama-1.1b-q3_K_M.gguf")
	if _, err := os.Stat(p); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	path := jlmOf(t, p)
	ids := []int32{1, 450, 7483, 310, 3444, 338, 3681, 29889, 450, 7483}
	run := func(t *testing.T, spec string, pl Placement) ([][]float32, gpu.Stats) {
		t.Helper()
		g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
		if err != nil || g == nil {
			noDevice(t, spec, err)
		}
		defer g.Close()
		m, err := model.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		if m.Cfg.TiedEmbd {
			t.Fatal("tinyllama ties its embedding: its projection is an input row's table and does not stream")
		}
		st := testInstance(t, m, 64, WithPlacement(pl))
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatalf("%s: %v (%s)", spec, err, g.Err())
		}
		if n, ok := st.ld.(nn.NamedDevice).HeadDeviceName(); !ok || n != "cuda:0" {
			t.Fatalf("%s: the head is on %q (%v), placed on cuda:0", spec, n, ok)
		}
		if st.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("%s: %d of %d blocks on the card (%s)", spec, st.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		var out [][]float32
		for _, id := range ids {
			l, err := st.Forward(id)
			if err != nil {
				t.Fatalf("%s: %v (%s)", spec, err, g.Err())
			}
			out = append(out, append([]float32(nil), l...))
		}
		if st.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("%s: %d of %d blocks left on the card after the decode (%s)", spec, st.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		if n, ok := st.ld.(nn.NamedDevice).HeadDeviceName(); !ok || n != "cuda:0" {
			t.Fatalf("%s: the head went to %q (%v) during the decode", spec, n, ok)
		}
		return out, g.Stats()
	}

	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	host := testInstance(t, m, 64)
	var want [][]float32
	for _, id := range ids {
		l, err := host.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, append([]float32(nil), l...))
	}
	// Half the weights: two slots as wide as the projection's planes, so the
	// blocks stream and the projection takes its turn among them.
	budget := m.WeightBytes() / 2
	host.Close()
	m.Close()

	res, rs := run(t, "cuda:0", Placement{Strict: true, Rest: &Place{On: "cuda:0"}, Head: &Place{On: "cuda:0"}})
	if rs.HeadsStreamed != 0 {
		t.Fatalf("the resident arm placed %d streamed projection(s)", rs.HeadsStreamed)
	}
	spec := "cuda:0=" + strconv.FormatUint(budget, 10)
	str, ss := run(t, spec, Placement{Strict: true, Rest: &Place{On: "cuda:0", Stream: true}, Head: &Place{On: "cuda:0", Stream: true}})
	t.Logf("streamed on %s: %d slot(s), %d head page-in(s) and %d page-out(s), %d block page-in(s) over %d tokens",
		spec, ss.Slots, ss.HeadPageIns, ss.HeadPageOuts, ss.PageIns, len(ids))
	if ss.HeadsStreamed == 0 {
		t.Fatal("the head placed streamed is not a page: this arm ran a resident head")
	}
	// The first token finds it in from placement; every later one pages it
	// in again, or the arm measured a head that never left.
	if ss.HeadPageIns < len(ids)-1 || ss.HeadPageOuts < len(ids)-1 {
		t.Errorf("%d head page-in(s) and %d page-out(s) over %d tokens: the budget did not make it stream",
			ss.HeadPageIns, ss.HeadPageOuts, len(ids))
	}
	nmse := func(a, b []float32) float64 {
		var num, den float64
		for i := range b {
			d := float64(a[i]) - float64(b[i])
			num += d * d
			den += float64(b[i]) * float64(b[i])
		}
		return num / den
	}
	var resCPU, strCPU, strRes float64
	for p := range ids {
		for i, v := range str[p] {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("token %d logit %d is %v on the streamed head: not finite", p, i, v)
			}
		}
		if Greedy(str[p]) != Greedy(res[p]) {
			t.Errorf("token %d: %d on the streamed head, %d on the resident one", p, Greedy(str[p]), Greedy(res[p]))
		}
		resCPU = max(resCPU, nmse(res[p], want[p]))
		strCPU = max(strCPU, nmse(str[p], want[p]))
		strRes = max(strRes, nmse(str[p], res[p]))
	}
	t.Logf("worst token NMSE: streamed against resident %.3e, resident against the CPU %.3e, streamed against the CPU %.3e",
		strRes, resCPU, strCPU)
	// The device's band is the resident head's against the CPU (RULE 11c);
	// the streamed head reads the same planes through the same forms.
	if lim := math.Max(2*resCPU, 1e-6); strCPU > lim {
		t.Fatalf("streamed head NMSE %.3e against the CPU, past %.3e (twice the resident head's %.3e)", strCPU, lim, resCPU)
	}
	if strRes > math.Max(resCPU/10, 1e-7) {
		t.Fatalf("streamed head NMSE %.3e against the resident head: not the same projection", strRes)
	}
}
