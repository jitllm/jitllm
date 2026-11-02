//go:build jitllmfault

package model

import (
	"errors"
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
)

const faultModel = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

func faultTier(t *testing.T, parts gpu.Parts) *gpu.GPU {
	t.Helper()
	o := append([]gpu.Option{gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)
	if testdevices.Selected() {
		o = append(o, gpu.WithDevices(testdevices.One("")))
	}
	g, err := gpu.OpenWith(o...)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	return g
}

type faultRun struct {
	err           error
	ev            Evacuations
	demotions     int
	before, after int
	spills        int
}

// buildFaultRun prefills a prompt and steps a few tokens on a device whose
// composer cannot form a matvec, the failure reported as a build failure or,
// for the violation, as the device failing.
func buildFaultRun(t *testing.T, m *Model, parts gpu.Parts, asDevice bool) faultRun {
	t.Helper()
	gpu.SetFormFault(true, graph.OpMatVec, asDevice)
	defer gpu.SetFormFault(false, 0, false)
	g := faultTier(t, parts)
	defer g.Close()
	ids := m.Vocab.Encode("The capital of France is Paris, and the capital of Germany is", true)
	st := m.NewInstance(len(ids) + 8)
	defer st.Close()
	st.SetDevice(g)
	r := faultRun{before: st.GPULayers()}
	if r.before == 0 {
		t.Fatalf("no block on the device: %s", g.Err())
	}
	logits, err := st.Prefill(ids)
	for i := 0; err == nil && i < 3; i++ {
		_, err = st.Forward(argmaxID(logits))
	}
	r.err = err
	r.ev, r.demotions = st.Evacuations(), st.DeviceDemotions()
	r.after, r.spills = st.GPULayers(), st.Spills()
	return r
}

// TestBuildFailureMovesNothing fails the composer's form of every matvec (a
// rewrite of its registry) and demands the call that needed the form return
// the named nn.BuildError with no block moved, none demoted, none relocated.
// The violation reports the same failure as the device failing: the instance must
// then demote, which proves the gate can tell the two apart (RULE 10).
func TestBuildFailureMovesNothing(t *testing.T) {
	p := testmodels.Path(faultModel)
	m, err := Open(jlmOf(t, p), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, c := range []struct {
		name  string
		parts gpu.Parts
	}{{"block", gpu.ComposedBlock}, {"head", gpu.ComposedHead}} {
		t.Run(c.name, func(t *testing.T) {
			r := buildFaultRun(t, m, c.parts, false)
			var be *nn.BuildError
			if !errors.As(r.err, &be) {
				t.Fatalf("a form that does not build gave %v, want an nn.BuildError", r.err)
			}
			t.Logf("refused: %v", r.err)
			if r.ev.BuildRefused == 0 {
				t.Fatalf("the refusal was not counted: %+v", r.ev)
			}
			if r.demotions != 0 || r.ev.DemotedFault != 0 || r.ev.DemotedDecline != 0 || r.ev.Relocated != 0 ||
				r.spills != 0 || r.after != r.before {
				t.Fatalf("a build failure moved work: %+v, demotions %d, spills %d, %d of %d blocks left on the device",
					r.ev, r.demotions, r.spills, r.after, r.before)
			}

			v := buildFaultRun(t, m, c.parts, true)
			if v.ev.DemotedFault == 0 || v.after != 0 {
				t.Fatalf("the violation ran without evacuating (%+v, %d blocks left): the gate cannot tell a build "+
					"failure from a device fault", v.ev, v.after)
			}
			t.Logf("the violation evacuated: %+v", v.ev)
		})
	}
}

// TestDeviceFaultAtDecodeEvacuatesAndIsCounted fails the device's first call,
// a decode token, and demands the host finish it with the host's logits and
// the demotion be counted as a fault.
func TestDeviceFaultAtDecodeEvacuatesAndIsCounted(t *testing.T) {
	p := testmodels.Path(faultModel)
	m, err := Open(jlmOf(t, p), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const token = 1000
	host := m.NewInstance(8)
	defer host.Close()
	want, err := host.Forward(token)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)

	t.Setenv("JITLLM_DEVICE_FAIL_AT", "1")
	g := faultTier(t, 0)
	defer g.Close()
	dev := m.NewInstance(8)
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
