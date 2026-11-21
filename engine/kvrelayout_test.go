package engine

import (
	"context"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// TestHistoryTakesTheLeasingInstancesFormat: a session whose history is in
// one format leases an instance whose history is in another, and the
// instance carries what it wrote into its own, through the CPU device's
// conversions composed with it -- nothing composed at the lease.
//
//   - Round trip: a binary16 history widened to float32 on the generation
//     instance and narrowed home to the prefill one is the same halves, so the
//     session's next prompt row there is bit for bit a session's that never
//     left.
//   - The violation: a history scribbled on while it is away comes home as
//     what was written, and the round trip parts.
func TestHistoryTakesTheLeasingInstancesFormat(t *testing.T) {
	for _, name := range []string{"stories15M-q8_0.gguf", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"} {
		t.Run(name, func(t *testing.T) {
			m, err := model.Open(jlmOf(t, testmodels.Path(name)))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			const seq = 64
			prompt := make([]int32, 24)
			for i := range prompt {
				prompt[i] = int32(300 + 7*i)
			}
			next := []int32{1000}
			ctx := context.Background()

			solo := testInstance(t, m, seq, WithKVF16(true))
			defer solo.Close()
			if _, err := solo.Prefill(prompt); err != nil {
				t.Fatal(err)
			}
			want, err := solo.Prefill(next)
			if err != nil {
				t.Fatal(err)
			}
			want = slices.Clone(want)

			run := func(scribble bool) []float32 {
				pool := NewPool()
				p := testInstance(t, m, seq, WithKVF16(true))
				defer p.Close()
				g := testInstance(t, m, seq, WithKVF16(false))
				defer g.Close()
				if err := pool.Serve(p, 0, PhasePrefill); err != nil {
					t.Fatal(err)
				}
				if err := pool.Serve(g, 0, PhaseGenerate); err != nil {
					t.Fatal(err)
				}
				// Served after the prefill instance, the generation one took its
				// binary16; a device attached after serving steps an empty cache
				// down to float32 (SetDeviceLayers), as this does.
				g.With(func(in *Instance) { in.stepDownKVToF32() })
				if p.kvFmt != nn.KVF16 || g.kvFmt != nn.KVF32 {
					t.Fatalf("the instances' histories are %v and %v, want f16 and f32: the formats never differ", p.kvFmt, g.kvFmt)
				}
				s, err := pool.NewSession(seq)
				if err != nil {
					t.Fatal(err)
				}
				defer s.Close()
				if err := s.Run(ctx, PhasePrefill, func(in *Instance) error {
					_, err := in.Prefill(prompt)
					return err
				}); err != nil {
					t.Fatal(err)
				}
				conv := func(in *Instance) int { return in.kvConvs[nn.KVF16].(*cpudev.KVConv).Ran }
				ran := conv(g)
				asked := g.Account(nn.CPU).Asks
				if err := s.Run(ctx, PhaseGenerate, func(in *Instance) error {
					if in != g {
						t.Fatal("the generation leased another instance than the one served for it")
					}
					if s.kvFormat != nn.KVF32 {
						t.Fatalf("on the float32 instance the session's history is %v", s.kvFormat)
					}
					if scribble {
						for li := range s.kv.layers {
							if k := s.kv.layers[li].mem.K(0); k != nil {
								k[0] += 3
							}
						}
					}
					return nil
				}); err != nil {
					t.Fatal(err)
				}
				if conv(g) == ran {
					t.Fatal("the binary16 history reached the float32 instance through none of its conversions")
				}
				if n := g.Account(nn.CPU).Asks - asked; n != 0 {
					t.Fatalf("leasing the float32 instance asked the CPU for %d form(s)", n)
				}
				var home []float32
				if err := s.Run(ctx, PhasePrefill, func(in *Instance) error {
					if s.kvFormat != nn.KVF16 {
						t.Fatalf("back on the binary16 instance the session's history is %v", s.kvFormat)
					}
					lg, err := in.Prefill(next)
					home = slices.Clone(lg)
					return err
				}); err != nil {
					t.Fatal(err)
				}
				return home
			}
			home := run(false)
			if !slices.Equal(home, want) {
				e, _ := nmseOf(home, want)
				t.Fatalf("a history that went to the float32 instance and came home parts from one that never left (NMSE %.3g)", e)
			}
			bad := run(true)
			if slices.Equal(bad, want) {
				t.Fatal("a history scribbled on while away came home unchanged: the round trip does not read the carried pages")
			}
			t.Logf("the history went f16 -> f32 -> f16 and the next row is bit for bit the one that never left")
		})
	}
}
