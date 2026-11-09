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

// TestASessionStepsOnEitherInstanceOfOnePlacement is the server's decode
// groups' lease: instances of one placement, and a session leasing whichever
// the pool gives it for each step -- the history is the session's, so which
// of them steps it changes nothing. Its tokens, decoded on two instances in
// turn, are those it decodes on one throughout: on the CPU, on a card, and
// split over two cards.
func TestASessionStepsOnEitherInstanceOfOnePlacement(t *testing.T) {
	path, ok := existingModel(testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
	if !ok {
		t.Fatal("model not present")
	}
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	nb := m.Cfg.NLayer
	d := testdevices.Take(t, 2, "cuda:0,vulkan:1")
	prompt := m.Vocab.Encode("The clerk counted barrels of salt on the upper floor, and", true)
	const gen, seq = 24, 128
	type arm struct {
		name string
		tier func(t *testing.T) *gpu.GPU
		opts []Option
	}
	first := map[int]Place{}
	for li := range nb / 2 {
		first[li] = Place{On: d[0]}
	}
	arms := []arm{
		{"cpu", nil, nil},
		{"one-card", func(t *testing.T) *gpu.GPU {
			g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(d[0])}, testTierOpts(t)...)...)
			if err != nil || g == nil {
				noDevice(t, d[0], err)
			}
			return g
		}, nil},
		{"two-cards", func(t *testing.T) *gpu.GPU {
			g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(d[0] + "," + d[1])}, testTierOpts(t)...)...)
			if err != nil || g == nil {
				noDevice(t, d[0]+","+d[1], err)
			}
			return g
		}, []Option{WithStreamTrial(false), WithPlacement(Placement{Blocks: first, Rest: &Place{On: d[1]}, Strict: true})}},
	}
	for _, a := range arms {
		t.Run(a.name, func(t *testing.T) {
			var g *gpu.GPU
			if a.tier != nil {
				g = a.tier(t)
				defer g.Close()
			}
			inst := func() *Instance {
				in := testInstance(t, m, seq, a.opts...)
				if g != nil {
					if err := in.SetDevice(g); err != nil {
						t.Fatal(err)
					}
					if in.GPULayers() != nb {
						t.Fatalf("%d of %d blocks on the cards", in.GPULayers(), nb)
					}
				}
				return in
			}
			ia, ib := inst(), inst()
			defer ia.Close()
			defer ib.Close()
			ctx := context.Background()
			// decode is gen greedy tokens of a session prefilled on ia,
			// step i leased from on(i).
			decode := func(on func(i int) *Instance) []int32 {
				pool := NewPool()
				defer pool.Close()
				if err := pool.Serve(ia, 0); err != nil {
					t.Fatal(err)
				}
				s, err := pool.NewSession(seq)
				if err != nil {
					t.Fatal(err)
				}
				defer s.Close()
				var lg []float32
				if err := s.Run(ctx, PhasePrefill, func(in *Instance) error {
					lg, err = in.Prefill(prompt)
					return err
				}); err != nil {
					t.Fatal(err)
				}
				served := ia
				var out []int32
				for i := range gen {
					if want := on(i); want != served {
						s.Release()
						if err := pool.Unserve(served); err != nil {
							t.Fatal(err)
						}
						if err := pool.Serve(want, 0); err != nil {
							t.Fatal(err)
						}
						served = want
					}
					tok := int32(argmax(lg))
					out = append(out, tok)
					if err := s.Run(ctx, PhaseGenerate, func(in *Instance) error {
						if in != served {
							t.Errorf("step %d leased an instance the pool does not serve", i)
						}
						lg, err = in.Forward(tok)
						return err
					}); err != nil {
						t.Fatal(err)
					}
				}
				s.Release()
				if err := pool.Unserve(served); err != nil {
					t.Fatal(err)
				}
				return out
			}
			want := decode(func(int) *Instance { return ia })
			got := decode(func(i int) *Instance {
				if i%2 == 0 {
					return ib
				}
				return ia
			})
			if !slices.Equal(got, want) {
				t.Fatalf("decoded on two instances in turn %v, on one %v", got, want)
			}
			t.Logf("%d tokens equal decoded on two instances in turn", gen)
			if g != nil {
				return
			}
			// The violation: the second instance reads the step's row
			// wrongly. The comparison must see it.
			setFault(ib, cpuFaultRow)
			bad := decode(func(i int) *Instance {
				if i%2 == 0 {
					return ib
				}
				return ia
			})
			setFault(ib, cpuFaultNone)
			if slices.Equal(bad, want) {
				t.Error("a row fault on the second instance decodes the same tokens: the comparison cannot see it")
			}
		})
	}
}
