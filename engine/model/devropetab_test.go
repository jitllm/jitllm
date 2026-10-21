package model

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestRopeTableOnDeviceMatchesTheUpload gates the device-built rotary table end
// to end. The kernel gate (backend.TestRopeTableOnDeviceMatchesHost) cannot say
// the tier launched it before the rotation, gave it the local planes for a local
// layer, or wrote this token's position into the buffer it reads.
//
// Both arms run in one process against one copy of the weights, with the tier
// opened with TuneOff (the split is the reduction order), so the bar is
// identical token ids and near-identical logits.
func TestRopeTableOnDeviceMatchesTheUpload(t *testing.T) {
	for _, c := range []struct {
		file  string
		quirk string
	}{
		// NEOX rotation, 500000 base, one rotary configuration.
		{"Llama-3.2-1B-Instruct-Q4_K_M.jlm", "neox, one base"},
		// Two bases: gemma3's local layers train at 10000 and its global ones at
		// the file's base, so the device must build both tables and pick per
		// block. The wrong planes for a local layer give fluent wrong text.
		{"gemma-3-1b-it-Q4_K_M.jlm", "two rotary bases, local and global"},
	} {
		t.Run(c.file, func(t *testing.T) {
			p := testmodels.Path(c.file)
			if _, err := os.Stat(p); err != nil {
				t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
			}
			m, err := Open(p)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The capital of France is", true)

			// Poisoned scratch on both arms: position 0's sine plane is zero, so
			// an unwritten table would read as correct from a zeroed buffer.
			g, err := openTier(t, tier.WithDeviceTune(tier.TuneOff),
				tier.WithPoisonScratch(true))
			if err != nil || g == nil {
				noDevice(t, "device", err)
			}
			defer g.Close()

			run := func(host bool) ([]int32, [][]float32, tier.Stats) {
				g.RopeTableHost = host
				before := g.Stats()
				s := m.NewState(64)
				defer s.Close()
				s.SetDevice(g)
				if s.GPULayers() != m.Cfg.NLayer {
					t.Skipf("the device took %d of %d blocks (%v) -- this gate is about "+
						"EVERY block being placed", s.GPULayers(), m.Cfg.NLayer, g.Err())
				}
				if _, err := s.Prefill(ids[:len(ids)-1]); err != nil {
					t.Fatal(err)
				}
				var out []int32
				var logits [][]float32
				id := ids[len(ids)-1]
				for i := 0; i < 24; i++ {
					l, err := s.Forward(id)
					if err != nil {
						t.Fatal(err)
					}
					logits = append(logits, append([]float32(nil), l...))
					id = argmaxID(l)
					out = append(out, id)
				}
				now := g.Stats()
				return out, logits, tier.Stats{
					RopeTables:       now.RopeTables - before.RopeTables,
					RopeTableUploads: now.RopeTableUploads - before.RopeTableUploads,
					RopeTableWhy:     now.RopeTableWhy,
				}
			}

			// The host arm first, so its counters are the control.
			hostIDs, hostLg, hostSt := run(true)
			devIDs, devLg, devSt := run(false)

			// The selection check: the arms agree by design, so identical ids
			// say nothing about which ran without tier.Stats.RopeTables.
			if devSt.RopeTableWhy != "" {
				t.Fatalf("the device never built a table: %s", devSt.RopeTableWhy)
			}
			if devSt.RopeTables == 0 {
				t.Fatalf("the device arm built 0 tables (%d uploaded) -- the configuration "+
					"under test was never selected", devSt.RopeTableUploads)
			}
			if hostSt.RopeTables != 0 {
				t.Fatalf("the HOST arm built %d table(s) on the device -- the two arms are "+
					"not different configurations", hostSt.RopeTables)
			}
			if hostSt.RopeTableUploads == 0 {
				t.Fatal("the host arm uploaded no table at all, so it is not the control " +
					"this comparison needs")
			}

			for i := range hostIDs {
				if devIDs[i] != hostIDs[i] {
					t.Fatalf("token %d: the device built %d where the upload gave %d\n"+
						"  device %v\n  upload %v", i, devIDs[i], hostIDs[i], devIDs, hostIDs)
				}
			}
			// And the logits too: an argmax hides everything below the margin.
			//
			// The bound is not zero: the device and host tables are two
			// spellings of one reduction (the device rounds half-up in
			// integers, the host half-even; spirv lowers OpFma unfused), one ulp
			// apart. 1e-3 is far over an ulp and far under what a real defect
			// produces (a wrong base or quadrant moves logits by 0.1 or more).
			const band = 1e-3
			var worst float64
			for i := range hostLg {
				for j := range hostLg[i] {
					if d := math.Abs(float64(devLg[i][j]) - float64(hostLg[i][j])); d > worst {
						worst = d
					}
				}
			}
			if worst > band {
				t.Fatalf("the logits differ by %.3e, over the %.0e band -- the two arms run "+
					"every other kernel identically, so this is the table and not arithmetic "+
					"drift", worst, band)
			}
			t.Logf("%-32s %s\n    device arm: %d built, %d uploaded    host arm: %d built, %d uploaded\n"+
				"    %d tokens identical, worst |dlogit| %.3e",
				c.file, c.quirk, devSt.RopeTables, devSt.RopeTableUploads,
				hostSt.RopeTables, hostSt.RopeTableUploads, len(devIDs), worst)
		})
	}
}

func argmaxID(l []float32) int32 {
	best, at := float32(math.Inf(-1)), 0
	for i, v := range l {
		if v > best {
			best, at = v, i
		}
	}
	return int32(at)
}

// TestRopeTablePrefillChunkMatchesTheUpload is the batched half, which the
// decode gate cannot reach: a prefill builds one table per position and the row
// count is baked, so it is a different kernel. It is also the only path where a
// chunk can be ragged, a case the tier deliberately uploads.
func TestRopeTablePrefillChunkMatchesTheUpload(t *testing.T) {
	p := testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M.jlm")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode(repeatWords("the quick brown fox jumps over the lazy dog ", 120), true)
	// Two whole chunks, so the last one's table is the device's (a ragged tail
	// is uploaded). The chunk is the device's width, not the host's.
	const chunk = nn.MaxDevicePrefillChunk
	if len(ids) < 2*chunk {
		t.Fatalf("the prompt is %d tokens, too short for two chunks", len(ids))
	}
	ids = ids[:2*chunk]

	g, err := openTier(t, tier.WithDeviceTune(tier.TuneOff), tier.WithPoisonScratch(true))
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	defer g.Close()

	var devRows []float32
	run := func(host bool) ([]float32, tier.Stats) {
		g.RopeTableHost = host
		before := g.Stats()
		s := m.NewState(2*chunk + 1)
		defer s.Close()
		s.SetDevice(g)
		if s.GPULayers() != m.Cfg.NLayer {
			t.Skipf("the device took %d of %d blocks (%v)", s.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		l, err := s.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		if !host {
			if devRows, err = g.RopeRows(chunk); err != nil {
				t.Fatal(err)
			}
		}
		now := g.Stats()
		return append([]float32(nil), l...), tier.Stats{
			RopeTables:       now.RopeTables - before.RopeTables,
			RopeTableUploads: now.RopeTableUploads - before.RopeTableUploads,
			RopeTableWhy:     now.RopeTableWhy,
		}
	}
	hostL, hostSt := run(true)
	devL, devSt := run(false)

	if hostSt.RopeTables != 0 {
		t.Fatalf("the HOST arm built %d table(s) on the device -- the two arms are not "+
			"different configurations", hostSt.RopeTables)
	}
	if devSt.RopeTables == 0 {
		t.Fatalf("the device built 0 tables over a %d-token prefill (%d uploaded, why %q) -- "+
			"the batched kernel was never selected", len(ids), devSt.RopeTableUploads, devSt.RopeTableWhy)
	}
	// The ragged chunk is reported, not demanded: whether a prompt ends
	// mid-chunk depends on its length, which the gate does not control.
	t.Logf("prefill of %d tokens\n    device arm: %d built, %d uploaded    host arm: %d built, %d uploaded",
		len(ids), devSt.RopeTables, devSt.RopeTableUploads,
		hostSt.RopeTables, hostSt.RopeTableUploads)

	// Compare the table itself, not the logits it feeds: over a two-chunk
	// prefill int8 requantization amplifies a one-ulp table difference to a
	// few tenths of a logit, the same size as a table built one position late.
	nrot := m.Cfg.NRot
	host := make([]float32, nrot)
	var worstTab float64
	worstAt := 0
	for r := 0; r < chunk; r++ {
		m.rope.Table(host, chunk+r)
		for i := range host {
			if d := math.Abs(float64(devRows[r*nrot+i]) - float64(host[i])); d > worstTab {
				worstTab, worstAt = d, chunk+r
			}
		}
	}
	// Four ulps of a unit-magnitude value.
	if worstTab > 2.4e-7 {
		t.Fatalf("the second chunk's device table differs from the host's by %.3e at position %d",
			worstTab, worstAt)
	}
	var worst float64
	for i := range hostL {
		worst = math.Max(worst, math.Abs(float64(devL[i])-float64(hostL[i])))
	}
	t.Logf("second chunk's table within %.3e of the host's; the logits within %.3e", worstTab, worst)
	if argmaxID(devL) != argmaxID(hostL) {
		t.Fatalf("the prefill's own argmax differs: %d against %d", argmaxID(devL), argmaxID(hostL))
	}
	t.Logf("prefill worst |dlogit| %.3e, argmax %d", worst, argmaxID(devL))
}

func repeatWords(s string, n int) string {
	out := ""
	for i := 0; i < n; i++ {
		out += s
	}
	return fmt.Sprint(out)
}
