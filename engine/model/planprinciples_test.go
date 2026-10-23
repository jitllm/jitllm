package model

import (
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
)

// The five principles on the composer's plan. Each gate is the old engine
// path's own, pointed at a planned model; one that cannot run fails, it does
// not skip.

// planModels are the models every plan principle gate runs: a dense block,
// a hybrid's, and a real checkpoint.
var planModels = []string{"synth-ernie45.jlm", "synth-lfm2.jlm", "Qwen3-0.6B-Q8_0.gguf"}

// planDevices is every device the plan runs on here, closed when the test
// ends; none fails the test, since a gate with no device proved nothing.
func planDevices(t *testing.T) ([]backend.Device, jit.Device) {
	t.Helper()
	devs := backend.Open()
	t.Cleanup(func() {
		for _, d := range devs {
			d.Close()
		}
	})
	if len(devs) == 0 {
		t.Fatal("no device: this plan gate proved nothing")
	}
	return devs, jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}
}

// planTeacherForced runs prompt then ids teacher-forced on a fresh state of
// m, calling at(i, s) before step i, and returns each step's logits.
func planTeacherForced(t *testing.T, m *Model, prompt, ids []int32, at func(i int, s *State)) [][]float32 {
	t.Helper()
	s := m.NewState(len(prompt) + len(ids) + 8)
	defer s.Close()
	lg, err := s.Prefill(prompt)
	if err != nil {
		t.Fatal(err)
	}
	out := [][]float32{append([]float32(nil), lg...)}
	for i, id := range ids {
		if at != nil {
			at(i, s)
		}
		if lg, err = s.Forward(id); err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out
}

// planHeld holds moved logits to the unmoved arm's: a history lost in the
// move reads O(1), a device's reduction order its band.
func planHeld(t *testing.T, moved, still [][]float32) {
	t.Helper()
	for i := range still {
		if nmse, worst := nmseOf(moved[i], still[i]); !(nmse <= 1e-3) {
			t.Errorf("step %d: NMSE %.3g against the unmoved arm, worst %.3g", i, nmse, worst)
		}
	}
}

// TestPlanRelocatesToAnotherDevice is TestLayerRelocatesToAnotherDevice on
// the plan: a sequence moves to another device after its first token, its
// history with it, and the steps after hold to the arm that stayed.
func TestPlanRelocatesToAnotherDevice(t *testing.T) {
	devs, gpu := planDevices(t)
	if len(devs) < 2 {
		t.Fatalf("%d device(s): a sequence has nowhere to move, so this gate proved nothing", len(devs))
	}
	for _, name := range planModels {
		t.Run(name, func(t *testing.T) {
			m, err := Open(jlmOf(t, testmodels.Path(name)), noTune, WithPlanDevice(devs[0], gpu))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			prompt := m.Vocab.Encode("The capital of France is", true)
			ids := []int32{5, 9, 17, 33, 65, 129, 7, 11}
			still := planTeacherForced(t, m, prompt, ids, nil)
			moved := planTeacherForced(t, m, prompt, ids, func(i int, s *State) {
				if i == 1 {
					if err := s.SetPlanDevice(devs[1], gpu); err != nil {
						t.Fatal(err)
					}
				}
			})
			planHeld(t, moved, still)
		})
	}
}

// TestPlanRelocatesBlocksToTheCPU is the seam's move on the plan
// (TestRelocatedBlocksComeBack): half the blocks go to the CPU device
// mid-sequence and come back, their KV pages with them, and every step holds
// to the arm that never moved.
func TestPlanRelocatesBlocksToTheCPU(t *testing.T) {
	devs, gpu := planDevices(t)
	for _, name := range planModels {
		for _, dev := range devs {
			t.Run(name+"/"+dev.API(), func(t *testing.T) {
				m, err := Open(jlmOf(t, testmodels.Path(name)), noTune, WithPlanDevice(dev, gpu))
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				prompt := m.Vocab.Encode("The capital of France is", true)
				ids := []int32{5, 9, 17, 33, 65, 129, 7, 11}
				still := planTeacherForced(t, m, prompt, ids, nil)
				half := m.Cfg.NLayer / 2
				moved := planTeacherForced(t, m, prompt, ids, func(i int, s *State) {
					switch i {
					case 2:
						if err := s.SetPlanLayers(half); err != nil {
							t.Fatal(err)
						}
						if got := s.PlanLayers(); got != half {
							t.Fatalf("%d of %d blocks on the device after placing %d", got, m.Cfg.NLayer, half)
						}
					case 5:
						if err := s.SetPlanLayers(m.Cfg.NLayer); err != nil {
							t.Fatal(err)
						}
						if got := s.PlanLayers(); got != m.Cfg.NLayer {
							t.Fatalf("%d of %d blocks on the device after bringing them back", got, m.Cfg.NLayer)
						}
					}
				})
				planHeld(t, moved, still)
			})
		}
	}
}

// TestPlanDoesNotAllocate is TestDecodeDoesNotAllocate on the plan: a warm
// decode token of a planned model makes no engine heap allocation, on every
// device (decodeAllocs, its counter and verdict).
func TestPlanDoesNotAllocate(t *testing.T) {
	devs, gpu := planDevices(t)
	for _, name := range planModels {
		for _, dev := range devs {
			t.Run(name+"/"+dev.API(), func(t *testing.T) {
				m, err := Open(jlmOf(t, testmodels.Path(name)), noTune, WithPageBudget(1<<40), WithPlanDevice(dev, gpu))
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				// The selection check: the decode measured is the plan's.
				st := m.NewState(8)
				if _, err := st.Forward(5); err != nil || st.PlanLayers() != m.Cfg.NLayer {
					t.Fatalf("the state is not on the plan (%d of %d blocks, %v): this gate would measure the old engine path", st.PlanLayers(), m.Cfg.NLayer, err)
				}
				st.Close()
				decodeAllocs(t, m, nil)
			})
		}
	}
}

// TestPlanResetForgetsTheSequence prefills a prompt on a planned model,
// resets the state and prefills another: the logits are a fresh state's,
// a hybrid's recurrent summary zeroed with the rest.
func TestPlanResetForgetsTheSequence(t *testing.T) {
	devs, gpu := planDevices(t)
	for _, name := range planModels {
		for _, dev := range devs {
			t.Run(name+"/"+dev.API(), func(t *testing.T) {
				m, err := Open(jlmOf(t, testmodels.Path(name)), noTune, WithPlanDevice(dev, gpu))
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				first := m.Vocab.Encode("The capital of Spain is Madrid, and of Italy", true)
				second := m.Vocab.Encode("The capital of France is", true)
				fresh := m.NewState(64)
				want, err := fresh.Prefill(second)
				if err != nil {
					t.Fatal(err)
				}
				want = append([]float32(nil), want...)
				fresh.Close()
				s := m.NewState(64)
				defer s.Close()
				if _, err := s.Prefill(first); err != nil {
					t.Fatal(err)
				}
				s.Reset()
				got, err := s.Prefill(second)
				if err != nil {
					t.Fatal(err)
				}
				if nmse, worst := nmseOf(got, want); worst != 0 {
					t.Errorf("after a reset: NMSE %.3g against a fresh state, worst %.3g: the state kept the first prompt", nmse, worst)
				}
			})
		}
	}
}

