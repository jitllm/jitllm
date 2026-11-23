package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestARepeatedPromptOfWholePagesRestoresAllButItsLast: a prompt of whole
// pages that a session before it ran, restored by a session that runs its
// last position (RestorePrefix, the step loop's restore), takes every
// position but that one from the shared store, the last page at the fill
// under it, and answers as it does alone -- on the CPU and with every block
// on a card, after a session that decoded past its prompt and after one
// that stopped at its prompt's end and closed (a server's request of one
// token, which runs no step past its prompt and closes its session). Against a restore that gives up the page its bound falls inside, it
// restores a page short; against one that stops at the chunk under the
// bound, a chunk short. A copying store, which names no page by a fill it
// was not sealed at, restores the whole pages under the bound.
func TestARepeatedPromptOfWholePagesRestoresAllButItsLast(t *testing.T) {
	const page, pages, gen = 32, 2, 8
	m, err := model.Open(jlmOf(t, testmodels.Path(shareModels()[0])))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompt := make([]int32, page*pages)
	for j := range prompt {
		prompt[j] = int32((300 + j*7) % m.Cfg.NVocab)
	}
	for _, card := range []bool{false, true} {
		for _, first := range []int{gen, 0} {
			name := "cpu"
			if card {
				name = "card"
			}
			if first == 0 {
				name += "/stopped-and-closed"
			}
			t.Run(name, func(t *testing.T) {
				opts := []Option{sharedCPU(t), withKVPage(page)}
				var g *gpu.GPU
				if card {
					var err error
					if g, err = openGPU(t); err != nil || g == nil {
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
				st := NewSharedStore()
				if n, _ := run(st, true, first); n != 0 {
					t.Fatalf("the first session restored %d positions from an empty store", n)
				}
				n, toks := run(st, true, gen)
				if n != len(prompt)-1 {
					t.Fatalf("the repeated prompt of %d positions restored %d from the shared store, want %d", len(prompt), n, len(prompt)-1)
				}
				sameTokens(t, "restored all but its last against solo", solo, toks)
				if card {
					return
				}
				mem := NewMemStore()
				run(mem, false, first)
				n, toks = run(mem, false, gen)
				if n != len(prompt)-page {
					t.Fatalf("the repeated prompt restored %d positions from a copying store, want the %d of its whole pages under the bound", n, len(prompt)-page)
				}
				sameTokens(t, "copied against solo", solo, toks)
			})
		}
	}
}
