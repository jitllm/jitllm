package engine

import (
	"context"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestSessionMovesBetweenInstances is the design's per-phase lease: a
// session prefills on one instance and generates on another the scheduler
// serves for that phase alone, its history moving with it, against one
// instance prefilling where the first ran and then placed where the second
// does: on the CPU then a card, and from one card to another (two backends'
// kernels, so the reference crosses the same two). Each lease must have been
// the instance it names.
func TestSessionMovesBetweenInstances(t *testing.T) {
	path, ok := existingModel(testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
	if !ok {
		t.Fatal("model not present")
	}
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	devs := testdevices.Take(t, 2, "cuda:0,vulkan:1")
	if len(devs) != 2 {
		t.Fatalf("two devices wanted, got %v", devs)
	}
	card := func(name string) *gpu.GPU {
		g, err := openGPU(t, gpu.WithDevices(name), gpu.WithDeviceTune(gpu.TuneOff))
		if err != nil || g == nil {
			noDevice(t, name, err)
		}
		return g
	}
	g0, g1 := card(devs[0]), card(devs[1])
	defer g0.Close()
	defer g1.Close()
	prompt := m.Vocab.Encode("The capital of France is", true)
	const gen, seq = 16, 128
	// decode runs gen greedy tokens on in after lg.
	decode := func(in *Instance, lg []float32) []int32 {
		var out []int32
		for range gen {
			tok := int32(argmax(lg))
			out = append(out, tok)
			if lg, err = in.Forward(tok); err != nil {
				t.Fatal(err)
			}
		}
		return out
	}
	for _, arm := range []struct {
		name         string
		prefill, gen *gpu.GPU
	}{{"cpu-then-card", nil, g0}, {"card-then-card", g0, g1}} {
		t.Run(arm.name, func(t *testing.T) {
			ref := testInstance(t, m, seq)
			if arm.prefill != nil {
				ref.SetDevice(arm.prefill)
			}
			lg, err := ref.Prefill(prompt)
			if err != nil {
				t.Fatal(err)
			}
			ref.SetDevice(arm.gen)
			if ref.GPULayers() != m.Cfg.NLayer {
				t.Fatalf("the reference placed %d of %d blocks: %q", ref.GPULayers(), m.Cfg.NLayer, arm.gen.Err())
			}
			want := decode(ref, lg)
			ref.Close()

			pool := NewPool()
			p := testInstance(t, m, seq)
			defer p.Close()
			if arm.prefill != nil {
				p.SetDevice(arm.prefill)
			}
			g := testInstance(t, m, seq)
			defer g.Close()
			g.SetDevice(arm.gen)
			if g.GPULayers() != m.Cfg.NLayer {
				t.Fatalf("the generation instance placed %d of %d blocks: %q", g.GPULayers(), m.Cfg.NLayer, arm.gen.Err())
			}
			if err := pool.Serve(p, 0, PhasePrefill); err != nil {
				t.Fatal(err)
			}
			if err := pool.Serve(g, 0, PhaseGenerate); err != nil {
				t.Fatal(err)
			}
			s, err := pool.NewSession(seq)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			ctx := context.Background()
			if err := s.Run(ctx, PhasePrefill, func(in *Instance) error {
				if in != p {
					t.Errorf("the prefill leased another instance than the one served for it")
				}
				lg, err = in.Prefill(prompt)
				return err
			}); err != nil {
				t.Fatal(err)
			}
			var got []int32
			if err := s.Run(ctx, PhaseGenerate, func(in *Instance) error {
				if in != g {
					t.Errorf("the generation leased another instance than the one served for it")
				}
				got = decode(in, lg)
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, want) {
				t.Fatalf("moved between instances %v, one instance %v", got, want)
			}
			t.Logf("%d tokens equal after the history moved %s", gen, arm.name)
		})
	}
}
