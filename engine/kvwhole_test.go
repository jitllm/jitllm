package engine

import (
	"context"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestARepeatedPromptOfWholePagesRestoresAllButItsLast: a prompt of whole
// pages that a session before it ran, restored by a session that runs its
// last position (RestorePrefix, the step loop's restore), takes every
// position but that one from the store, and leaves it there for the next, the last page at the fill under
// it, and answers as it does alone. It does so from a shared store and from
// a copying one (a server's memory cache), on the CPU and with every block
// on a card, the card paging as the host does (a server's geometry) and
// apart from it, after a session that decoded past its prompt and after one
// that stopped at its prompt's end and closed (a server's request of one
// token), each repeat but the last doing the same. Against a restore that gives up the page its bound falls inside,
// it restores a page short; against one that stops at the chunk under the
// bound, a chunk short.
func TestARepeatedPromptOfWholePagesRestoresAllButItsLast(t *testing.T) {
	const page, pages, gen = 64, 2, 8
	m, err := model.Open(jlmOf(t, testmodels.Path(shareModels()[1])))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompt := make([]int32, page*pages)
	for j := range prompt {
		prompt[j] = int32((300 + j*7) % m.Cfg.NVocab)
	}
	for _, where := range []string{"cpu", "card", "card-apart"} {
		for _, first := range []int{gen, 0} {
			card := where != "cpu"
			name := where
			if first == 0 {
				name += "/stopped-and-closed"
			}
			t.Run(name, func(t *testing.T) {
				opts := []Option{sharedCPU(t), withKVPage(page)}
				var g *gpu.GPU
				if card {
					var err error
					var o []gpu.Option
					if where == "card" {
						o = append(o, gpu.WithConfig(func(c *gpu.Config) { c.KVPage = page }))
					}
					if g, err = openGPU(t, o...); err != nil || g == nil {
						testmodels.Missing(t, "no card: %v", err)
					}
					t.Cleanup(func() { g.Close() })
				}
				run := func(store KVStore, share bool, steps int) (int, []int32) {
					t.Helper()
					s := testInstance(t, m, len(prompt)+gen+16, opts...)
					if steps == 0 {
						defer s.Close()
					} else {
						t.Cleanup(func() { s.Close() })
					}
					if g != nil {
						if err := s.SetDeviceLayers(g, -1); err != nil {
							t.Fatal(err)
						}
						if n := s.GPULayers(); n != m.Cfg.NLayer {
							t.Fatalf("the card took %d of %d blocks: %v", n, m.Cfg.NLayer, s.DeviceDeclines())
						}
					}
					n := 0
					if store != nil {
						mustKey(t, s, "whole-pages")
						if share {
							if err := s.ShareKV(store.(*SharedStore)); err != nil {
								t.Fatal(err)
							}
						} else {
							s.SetKVStore(store)
						}
						var err error
						if n, err = s.RestorePrefix(prompt); err != nil {
							t.Fatal(err)
						}
					}
					lg, err := s.Prefill(prompt[n:])
					if err != nil {
						t.Fatal(err)
					}
					if store != nil {
						if err := s.SealPrompt(lg); err != nil {
							t.Fatal(err)
						}
					}
					var toks []int32
					for range steps {
						id := Greedy(lg)
						toks = append(toks, id)
						if lg, err = s.Forward(id); err != nil {
							t.Fatal(err)
						}
					}
					return n, toks
				}
				_, solo := run(nil, false, gen)
				for _, share := range []bool{true, false} {
					var st KVStore = NewMemStore()
					kind := "copying"
					if share {
						st, kind = NewSharedStore(), "shared"
					}
					if n, _ := run(st, share, first); n != 0 {
						t.Fatalf("the first session restored %d positions from an empty %s store", n, kind)
					}
					for r := 1; r <= 3; r++ {
						steps := first
						if r == 3 {
							steps = gen
						}
						n, toks := run(st, share, steps)
						if n != len(prompt)-1 {
							t.Fatalf("repeat %d of the prompt of %d positions restored %d from the %s store, want %d", r, len(prompt), n, kind, len(prompt)-1)
						}
						if steps == gen {
							sameTokens(t, "restored all but its last from the "+kind+" store, against solo", solo, toks)
						}
					}
				}
			})
		}
	}
}

// TestPooledRequestsOfWholePagesRestoreAllButTheirLast is the server's
// request of one token on the shared store, repeated: each a session of the
// pool on one instance with every block on a card, which shares the store,
// restores, runs the rest of its prompt in a step, seals, and closes. Every
// request after the first restores all but the prompt's last position.
func TestPooledRequestsOfWholePagesRestoreAllButTheirLast(t *testing.T) {
	const page, pages = 256, 2
	m, err := model.Open(jlmOf(t, testmodels.Path(shareModels()[1])))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompt := make([]int32, page*pages)
	for j := range prompt {
		prompt[j] = int32((300 + j*7) % m.Cfg.NVocab)
	}
	for _, where := range []string{"card", "card-apart"} {
		t.Run(where, func(t *testing.T) {
			var o []gpu.Option
			if where == "card" {
				o = append(o, gpu.WithConfig(func(c *gpu.Config) { c.KVPage = page }))
			}
			g, err := openGPU(t, o...)
			if err != nil || g == nil {
				testmodels.Missing(t, "no card: %v", err)
			}
			defer g.Close()
			in := testInstance(t, m, len(prompt)+16, sharedCPU(t), withKVPage(page))
			defer in.Close()
			if err := in.SetDeviceLayers(g, -1); err != nil {
				t.Fatal(err)
			}
			if n := in.GPULayers(); n != m.Cfg.NLayer {
				t.Fatalf("the card took %d of %d blocks: %v", n, m.Cfg.NLayer, in.DeviceDeclines())
			}
			pool := NewPool()
			if err := pool.Serve(in, 0); err != nil {
				t.Fatal(err)
			}
			defer pool.Unserve(in)
			ctx := context.Background()
			store := NewSharedStore()
			var want int32
			for r := range 5 {
				s, err := pool.NewSession(len(prompt) + 16)
				if err != nil {
					t.Fatal(err)
				}
				n := 0
				if err := s.Run(ctx, PhasePrefill, func(in *Instance) error {
					mustKey(t, in, "whole-pages")
					if err := in.ShareKV(store); err != nil {
						return err
					}
					n, err = in.RestorePrefix(prompt)
					return err
				}); err != nil {
					t.Fatal(err)
				}
				out, err := StepRuns([]Run{{Session: s, Tokens: prompt[n:], Logits: true}})
				if err != nil {
					t.Fatal(err)
				}
				id := Greedy(out[0])
				if err := s.Run(ctx, PhasePrefill, func(in *Instance) error { return in.SealPrompt(out[0]) }); err != nil {
					t.Fatal(err)
				}
				s.Close()
				if wantN := min(r, 1) * (len(prompt) - 1); n != wantN {
					t.Fatalf("request %d restored %d positions, want %d", r, n, wantN)
				}
				if r == 0 {
					want = id
				} else if id != want {
					t.Fatalf("request %d answered %d, the first %d", r, id, want)
				}
			}
		})
	}
}
