package engine

import (
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
)

// TestParkTakesTheCardHistoryOffAndBack: a session with every block on
// the device parks (its history comes home; the instance keeps its blocks on
// the card), resumes onto the device, and decodes the tokens its solo run on the
// device decodes. A session sharing its history (ShareKV) parks as a server
// parks it, into a pinned view of a memory cache (ParkInto), while another
// session runs the same prompt and the shared store is trimmed to nothing.
func TestParkTakesTheCardHistoryOffAndBack(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm")
	if src := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf"); fileExists(src) {
		p = jlmOf(t, src)
	}
	m, err := model.Open(p)
	if err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v -- this gate proved nothing", err)
	}
	defer m.Close()
	g := swaTier(t)
	defer g.Close()
	const half, n = 20, 40
	for _, share := range []bool{false, true} {
		name := map[bool]string{false: "own", true: "shared"}[share]
		t.Run(name, func(t *testing.T) {
			// Long enough that the history seals pages before the park.
			prompt := m.Vocab.Encode(strings.Repeat("Once upon a time there was a little fox. ", 24), true)
			shared := NewSharedStore()
			start := func(store bool) *Instance {
				st := testInstance(t, m, 1024)
				st.SetDevice(g)
				if st.GPULayers() != m.Cfg.NLayer {
					st.Close()
					t.Skipf("CARD TOO SMALL: placed %d of %d blocks (%s) -- this gate proved nothing here",
						st.GPULayers(), m.Cfg.NLayer, g.Err())
				}
				if store {
					if err := st.SetCacheKey("park/" + name); err != nil {
						t.Fatal(err)
					}
					if err := st.ShareKV(shared); err != nil {
						t.Fatal(err)
					}
				}
				return st
			}
			solo := start(false)
			lg, err := solo.Prefill(prompt)
			if err != nil {
				t.Fatal(err)
			}
			want, _ := decodeIDs(t, solo, Greedy(lg), n)
			solo.Close()

			st := start(share)
			defer st.Close()
			if share {
				lg, err = st.PrefillCached(prompt)
			} else {
				lg, err = st.Prefill(prompt)
			}
			if err != nil {
				t.Fatal(err)
			}
			got, next := decodeIDs(t, st, Greedy(lg), half)
			var ps ParkStats
			if share {
				ps, err = st.ParkInto(NewMemStore().Pinned())
			} else {
				ps, err = st.Park()
			}
			if err != nil {
				t.Fatal(err)
			}
			// The session's history comes home; the instance's blocks stay on the
			// card, for every session leasing it.
			if ps.Blocks != m.Cfg.NLayer || heldOnDevice(st) != 0 || st.GPULayers() != m.Cfg.NLayer {
				t.Fatalf("park brought %d blocks' history home, left %d bytes of it on the device and %d of %d blocks placed",
					ps.Blocks, heldOnDevice(st), st.GPULayers(), m.Cfg.NLayer)
			}
			if share {
				// Another session on the prompt while this one is parked, and
				// the store trimmed to what sessions hold.
				other := start(true)
				if _, err := other.PrefillCached(prompt); err != nil {
					t.Fatal(err)
				}
				other.Close()
				shared.Trim(0)
			}
			if _, err := st.Resume(); err != nil {
				t.Fatal(err)
			}
			if heldOnDevice(st) == 0 {
				t.Fatalf("resume left the session's history off the device (%v)", st.DeviceDeclines())
			}
			rest, _ := decodeIDs(t, st, next, n-half)
			got = append(got, rest...)
			if !slices.Equal(got, want) {
				t.Fatalf("parked and resumed on the device %v\nsolo %v", got, want)
			}
			t.Logf("%s: %d positions, %d blocks home and back, %d pages out; %d tokens equal to the solo run",
				name, len(prompt)+n, ps.Blocks, ps.PagesOut, n)
		})
	}
}

// heldOnDevice is what st's session holds on its device now.
func heldOnDevice(st *Instance) uint64 {
	if d, ok := st.ld.(nn.Session); ok {
		return d.HeldBytes()
	}
	return 0
}
