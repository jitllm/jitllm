//go:build jitllmfault

package engine

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
)

const faultModel = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

func faultTier(t *testing.T, extra ...gpu.Option) *gpu.GPU {
	t.Helper()
	o := append(append([]gpu.Option{gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...), extra...)
	if testdevices.Selected() {
		o = append(o, gpu.WithDevices(testdevices.One("")))
	}
	g, err := openGPU(t, o...)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	return g
}

type faultRun struct {
	placed    int
	why       string
	cause     nn.FailCause
	toks      []int32
	err       error
	ev        Evacuations
	demotions int
}

// faultGreedy prefills ids and takes three greedy tokens.
func faultGreedy(st *Instance, ids []int32) ([]int32, error) {
	logits, err := st.Prefill(ids)
	var toks []int32
	for i := 0; err == nil && i < 3; i++ {
		id := argmaxID(logits)
		toks = append(toks, id)
		logits, err = st.Forward(id)
	}
	return toks, err
}

// buildFaultRun places the blocks on a device whose composer cannot form a
// matvec -- the failure reported as a build failure or, for the violation, as
// the device failing -- then prefills a prompt and steps a few tokens.
func buildFaultRun(t *testing.T, m *model.Model, ids []int32, asDevice bool, io ...Option) faultRun {
	t.Helper()
	var f gpu.Faults
	f.SetFormFault(true, graph.OpMatVec, asDevice)
	g := faultTier(t, gpu.WithFaults(&f))
	defer g.Close()
	st := testInstance(t, m, len(ids)+8, io...)
	defer st.Close()
	st.SetDevice(g)
	r := faultRun{placed: st.GPULayers(), why: g.Err(), cause: g.ErrCause()}
	r.toks, r.err = faultGreedy(st, ids)
	r.ev, r.demotions = st.Evacuations(), st.DeviceDemotions()
	return r
}

// TestBuildFailureMovesNothing fails the composer's form of every matvec (a
// rewrite of its registry). Forms are built where the blocks are placed, never
// at a step, so the placement meets the failure: every block is declined
// there, named as a form that does not build, the device records a build
// failure, and the prompt and its steps run on the CPU device with its
// answer -- nothing demoted, relocated or refused. The violation reports the
// same failure as the device failing, which a call meeting it would demote
// for: the gate must tell the two apart (RULE 10).
func TestBuildFailureMovesNothing(t *testing.T) {
	p := testmodels.Path(faultModel)
	instOpts := []Option{WithKVF16(false)}
	m, err := model.Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode("The capital of France is Paris, and the capital of Germany is", true)
	host := testInstance(t, m, len(ids)+8, instOpts...)
	want, err := faultGreedy(host, ids)
	host.Close()
	if err != nil {
		t.Fatal(err)
	}

	r := buildFaultRun(t, m, ids, false, instOpts...)
	if r.placed != 0 || !strings.Contains(r.why, "does not build") {
		t.Fatalf("a form that does not build: %d blocks placed, the device said %q", r.placed, r.why)
	}
	if r.cause != nn.FailBuild {
		t.Fatalf("a form that does not build is recorded as %v, want %v", r.cause, nn.FailBuild)
	}
	if r.err != nil || !slices.Equal(r.toks, want) {
		t.Fatalf("the declined blocks on the CPU device: %v, tokens %v, the CPU's %v", r.err, r.toks, want)
	}
	if r.demotions != 0 || r.ev != (Evacuations{}) {
		t.Fatalf("a build failure moved work: %+v, %d demotions", r.ev, r.demotions)
	}
	t.Logf("declined at the placement: %s", r.why)

	v := buildFaultRun(t, m, ids, true, instOpts...)
	if v.cause == nn.FailBuild {
		t.Fatalf("the violation is recorded as a build failure (%q): the gate cannot tell a build failure from a "+
			"device fault", v.why)
	}
	t.Logf("the violation is recorded as %v", v.cause)
}

// TestDeviceFaultAtDecodeEvacuatesAndIsCounted fails the device's first call,
// a decode token, and demands the host finish it with the host's logits and
// the demotion be counted as a fault.
func TestDeviceFaultAtDecodeEvacuatesAndIsCounted(t *testing.T) {
	p := testmodels.Path(faultModel)
	instOpts := []Option{WithKVF16(false)}
	m, err := model.Open(jlmOf(t, p))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const token = 1000
	host := testInstance(t, m, 8, instOpts...)
	defer host.Close()
	want, err := host.Forward(token)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)

	t.Setenv("JITLLM_DEVICE_FAIL_AT", "1")
	g := faultTier(t)
	defer g.Close()
	dev := testInstance(t, m, 8, instOpts...)
	defer dev.Close()
	dev.SetDevice(g)
	if dev.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("the device took %d of %d blocks", dev.GPULayers(), m.Cfg.NLayer)
	}
	got, err := dev.Forward(token)
	if err != nil {
		t.Fatal(err)
	}
	ev := dev.Evacuations()
	if ev.DemotedFault != 1 || ev.DemotedDecline != 0 || ev.BuildRefused != 0 || dev.GPULayers() != 0 {
		t.Fatalf("a device fault at decode: %+v, %d blocks left on the device", ev, dev.GPULayers())
	}
	var num, den float64
	for i := range want {
		d := float64(got[i] - want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	if argmaxID(got) != argmaxID(want) || math.IsNaN(num/den) || num/den > 1e-9 {
		t.Fatalf("after the demotion: argmax %d (host %d), logit NMSE %.3e", argmaxID(got), argmaxID(want), num/den)
	}
}
