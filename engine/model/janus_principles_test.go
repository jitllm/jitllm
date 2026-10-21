package model

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// Janus-Pro's tower under the five principles: it pages under a budget with
// the same bits, relocates host <-> device, and a decode after a picture
// allocates nothing.

func janusEncodePhoto(t *testing.T, tw *Tower, s *State) []float32 {
	t.Helper()
	px, err := s.PreprocessImage(decodeImage(t, mcvPhoto))
	if err != nil {
		t.Fatal(err)
	}
	out, err := s.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), out...)
}

func TestJanusTowerPagesUnderABudget(t *testing.T) {
	m := openJanus(t)
	vpage := m.container.PageBytes(int(m.container.H.NBlocks))
	s := m.Tower().testState()
	full := janusEncodePhoto(t, m.Tower(), s)
	s.Close()
	m.Close()
	for _, k := range []uint64{1, 3} {
		mb := openJanus(t, WithPageBudget(k*vpage))
		s := mb.Tower().testState()
		got := janusEncodePhoto(t, mb.Tower(), s)
		s.Close()
		_, ev := mb.container.Faults()
		mb.Close()
		if ev == 0 {
			t.Fatalf("a budget of %d vision page(s) evicted nothing", k)
		}
		for i := range full {
			if got[i] != full[i] {
				t.Fatalf("budget of %d vision page(s): element %d is %v, want %v", k, i, got[i], full[i])
			}
		}
		t.Logf("a budget of %d vision page(s): %d eviction(s), every element identical", k, ev)
	}
}

func TestJanusTowerRelocates(t *testing.T) {
	m := openJanus(t)
	defer m.Close()
	tw := m.Tower()
	ran := 0
	for _, spec := range stepDevices() {
		g, err := openTier(t, tier.WithDevices(spec))
		if err != nil || g == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer g.Close()
			s := tw.testState()
			defer s.Close()
			s.SetDevice(g)
			if s.GPUBlocks() != tw.Cfg.NLayer {
				t.Fatalf("%d of %d blocks placed", s.GPUBlocks(), tw.Cfg.NLayer)
			}
			dev1 := janusEncodePhoto(t, tw, s)
			s.SetDevice(nil)
			host := janusEncodePhoto(t, tw, s)
			s.SetDevice(g)
			dev2 := janusEncodePhoto(t, tw, s)
			for i := range dev1 {
				if dev1[i] != dev2[i] {
					t.Fatalf("element %d: %v on the first visit, %v on the second", i, dev1[i], dev2[i])
				}
			}
			n := visNMSE(t, "device against host", dev1, host)
			t.Logf("%s -> host -> %s: the device encodes identical, NMSE %.3e against the host's", spec, spec, n)
			if n > 2e-2 { // 24 blocks of two tiers rounding int8 apart; one block reads ~1e-6 (TestJanusOnDeviceMatchesHost)
				t.Fatalf("NMSE %.3e between the device's and the host's", n)
			}
		})
	}
	if ran == 0 {
		t.Skip("no GPU backend on this host")
	}
}

func TestJanusDecodeAfterAPictureDoesNotAllocate(t *testing.T) {
	m := openJanus(t, noTune)
	defer m.Close()
	tw := m.Tower()
	run := func(t *testing.T, g *tier.GPU) {
		ts := tw.testState()
		if g != nil {
			ts.SetDevice(g)
		}
		emb := janusEncodePhoto(t, tw, ts)
		ts.Close()
		msgs := []ChatMessage{{Role: "user", Content: "What is this?", Images: 1}}
		spans, err := m.ChatSpans(msgs, [][]float32{emb}, true)
		if err != nil {
			t.Fatal(err)
		}
		const warm, n = 96, 48
		s := m.NewState(SpanPositions(spans, m.Cfg.NEmbd) + warm + n + 8)
		defer s.Close()
		if g != nil {
			if err := s.SetDevice(g); err != nil {
				t.Fatal(err)
			}
			if s.GPULayers() == 0 {
				t.Skip("no text block placed on the device -- this arm proved nothing")
			}
		}
		if _, err := s.PrefillMixed(spans...); err != nil {
			t.Fatal(err)
		}
		for i := range warm {
			if _, err := s.Forward(int32(5 + i)); err != nil {
				t.Fatal(err)
			}
		}
		var c0 tier.Stats
		if g != nil {
			c0 = g.Stats()
		}
		r0 := m.container.Reads()
		w := countAllocs(func() {
			for i := range n {
				if _, err := s.Forward(int32(3 + i%64)); err != nil {
					t.Fatal(err)
				}
			}
		})
		captures := 0
		if g != nil {
			captures = g.Stats().Captures - c0.Captures
		}
		allocVerdict(t, fmt.Sprintf("after a picture, %d of %d blocks placed", s.GPULayers(), m.Cfg.NLayer),
			w, n, m.container.Reads()-r0, captures)
	}
	t.Run("host", func(t *testing.T) { run(t, nil) })
	for _, spec := range testdevices.Specs("cuda", "vulkan", "metal") {
		t.Run(spec, func(t *testing.T) {
			g, err := openTier(t, append([]tier.Option{tier.WithDevices(spec)}, testTierOpts(t)...)...)
			if err != nil || g == nil {
				noDevice(t, spec, err)
			}
			defer g.Close()
			run(t, g)
		})
	}
}
