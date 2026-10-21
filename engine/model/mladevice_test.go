package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestMLAOnEveryDevice runs Multi-head Latent Attention on the device, end to
// end, against the host. Kernel gates prove the kernels, not that the tier asks
// for the right ones, so this runs MLA models with every block placed.
//
// It asserts the block count, not that the forward succeeded (a run that fell
// to the host would also answer correctly). The split tuner is pinned because
// it varies the reduction order per process.
func TestMLAOnEveryDevice(t *testing.T) {
	ids := []int32{5, 11, 23, 41, 67, 89, 101, 127}
	ran := 0
	for _, name := range []string{"synth-deepseek-dense", "synth-deepseek", "synth-deepseek-lite"} {
		t.Run(name, func(t *testing.T) {
			src := hfContainer(t, name)
			m, err := Open(src, noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if !m.Cfg.MLA() {
				t.Fatalf("%s is not an MLA model -- this gate would prove nothing", name)
			}
			want := teacherForce(t, m, ids)
			// Every block, including DeepSeek mixture blocks (the V3 router
			// runs on the device).
			wantPlaced := m.Cfg.NLayer
			// Count the mixture blocks so a fixture that stopped being a
			// mixture cannot leave this gate covering MLA alone.
			moe := 0
			for li := 0; li < m.Cfg.NLayer; li++ {
				if m.Cfg.MoEAt(li) {
					moe++
				}
			}
			t.Logf("%d of %d block(s) are mixtures", moe, m.Cfg.NLayer)
			if name == "synth-deepseek" && moe == 0 {
				t.Fatalf("%s carries no mixture block, so the V3 router never runs "+
					"and this gate covers MLA alone", name)
			}
			if wantPlaced == 0 {
				t.Fatalf("%s places no block at all -- this gate would prove nothing", name)
			}
			for _, spec := range stepDevices() {
				gpu, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
				if err != nil || gpu == nil {
					t.Logf("%s: not present (%v)", spec, err)
					continue
				}
				ran++
				t.Run(spec, func(t *testing.T) {
					defer gpu.Close()
					st := m.NewState(len(ids) + 1)
					defer st.Close()
					st.SetDeviceLayers(gpu, -1)
					if n := st.GPULayers(); n != wantPlaced {
						t.Fatalf("%s took %d of %d placeable block(s) (%d in the model): %v",
							spec, n, wantPlaced, m.Cfg.NLayer, declineList(st))
					}
					got := runIDs(t, st, ids)
					if n := st.GPULayers(); n != wantPlaced {
						t.Fatalf("the session fell to the host (%d blocks left): %v",
							n, gpu.Err())
					}
					worst, worstD := compare(t, want, got)
					// The latent history is paged like every other: a run that
					// read the contiguous cache is not the path that ships.
					if ps := gpu.Stats(); ps.PagedLaunches == 0 {
						t.Fatalf("%s decoded %s through no paged attention: the latent cache was not paged",
							name, spec)
					}
					t.Logf("%s: %d of %d blocks on %s, %d positions, worst logit NMSE %.3e, "+
						"worst |dlogit| %.3f, %d paged launches (%d staged)", name, wantPlaced, m.Cfg.NLayer, spec,
						len(want), worst, worstD, gpu.Stats().PagedLaunches, gpu.Stats().PagedStagedLaunches)
					// Print the decline reasons: a missing feature and a full
					// card want opposite responses.
					for _, d := range declineList(st) {
						t.Logf("  declined x%d: %s", d.Blocks, d.Why)
					}
				})
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

// TestMLADeviceFaultsAreCaught runs the gate above against three DELIBERATE
// violations of latent attention's invariants and requires every one of them to
// be seen. Each is invisible to shape checks and kernel gates (swapped banks
// can share a shape, a wrong offset reads finite values, an off-by-one sheet
// reads a real head); only the whole model against the host sees them.
func TestMLADeviceFaultsAreCaught(t *testing.T) {
	ids := []int32{5, 11, 23, 41, 67, 89}
	src := hfContainer(t, "synth-deepseek-dense")
	m, err := Open(src, noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	want := teacherForce(t, m, ids)
	ran := 0
	for _, spec := range stepDevices() {
		if g, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff)); err == nil && g != nil {
			g.Close()
		} else {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		for _, f := range []struct {
			name string
			kind tier.MLAFault
		}{
			{"swap-the-absorb-banks", tier.MLAFaultSwapBanks},
			{"value-head-major-in-the-row", tier.MLAFaultValueHeadMajor},
			{"per-head-sheet-off-by-one", tier.MLAFaultSheetOff},
		} {
			t.Run(spec+"/"+f.name, func(t *testing.T) {
				gpu, err := openTier(t, tier.WithDevices(spec),
					tier.WithDeviceTune(tier.TuneOff), tier.WithMLAFault(f.kind))
				if err != nil || gpu == nil {
					t.Fatalf("%s: %v", spec, err)
				}
				defer gpu.Close()
				st := m.NewState(len(ids) + 1)
				defer st.Close()
				st.SetDeviceLayers(gpu, -1)
				// The fault must be selected: a fault arm that declined every
				// block would run the host and pass.
				if n := st.GPULayers(); n != m.Cfg.NLayer {
					t.Fatalf("the fault arm placed %d of %d blocks, so it ran the host: %v",
						n, m.Cfg.NLayer, declineList(st))
				}
				got := runIDs(t, st, ids)
				worst, worstD := worstOf(want, got)
				if worst < mlaDeviceNMSE {
					t.Fatalf("%s survived: worst logit NMSE %.3e, worst |dlogit| %.3f -- "+
						"the gate does not see it", f.name, worst, worstD)
				}
				t.Logf("%s: caught at logit NMSE %.3e, |dlogit| %.3f", f.name, worst, worstD)
			})
		}
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

// mlaDeviceNMSE is the bound both the passing and the fault gates use. It is
// the f32 reduction-order band between device and host, which grows with the
// number of blocks below the seam; the violations land orders above it.
const mlaDeviceNMSE = 1e-2

func runIDs(t *testing.T, st *State, ids []int32) [][]float32 {
	t.Helper()
	out := make([][]float32, 0, len(ids))
	for _, id := range ids {
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out
}

// worstOf is the largest per-position logit NMSE and the largest absolute logit
// difference. A non-finite arm reads as +Inf rather than as a pass, since
// NaN > x is false.
func worstOf(want, got [][]float32) (float64, float64) {
	worst, worstD := 0.0, 0.0
	for p := range want {
		var num, den float64
		for i := range want[p] {
			d := float64(got[p][i] - want[p][i])
			if math.IsNaN(d) || math.IsInf(d, 0) {
				return math.Inf(1), math.Inf(1)
			}
			num, den = num+d*d, den+float64(want[p][i])*float64(want[p][i])
			worstD = math.Max(worstD, math.Abs(d))
		}
		if den == 0 {
			return math.Inf(1), math.Inf(1)
		}
		nmse := num / den
		if math.IsNaN(nmse) {
			return math.Inf(1), math.Inf(1)
		}
		worst = math.Max(worst, nmse)
	}
	return worst, worstD
}

func compare(t *testing.T, want, got [][]float32) (float64, float64) {
	t.Helper()
	worst, worstD := worstOf(want, got)
	if !(worst < mlaDeviceNMSE) {
		t.Fatalf("worst logit NMSE %.3e against the host, worst |dlogit| %.3f", worst, worstD)
	}
	return worst, worstD
}

func declineList(st *State) []DeviceDecline { return st.DeviceDeclines() }

// TestMLASeamMovesWithItsHistory moves a latent KV history across the seam in
// both directions mid-sequence. Without MigrateKV a moved block attends over
// the wrong cache and stays fluent. Both directions are exercised because they
// fail differently: demotion reads into host pages whose k and v alias
// (kvPages.latent), promotion writes at the stride the scores kernel bakes.
func TestMLASeamMovesWithItsHistory(t *testing.T) {
	defer fusedRunsCheck(t, true)()
	ids := []int32{5, 11, 23, 41, 67, 89, 101, 127}
	m, err := Open(hfContainer(t, "synth-deepseek-dense"), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	want := teacherForce(t, m, ids)
	ran := 0
	for _, spec := range stepDevices() {
		gpu, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			st := m.NewState(len(ids) + 1)
			defer st.Close()
			st.SetDeviceLayers(gpu, -1)
			if n := st.GPULayers(); n != m.Cfg.NLayer {
				t.Fatalf("%s took %d of %d blocks: %v", spec, n, m.Cfg.NLayer, declineList(st))
			}
			got := make([][]float32, len(ids))
			// A third on the card, a third on the host, a third back on the card:
			// the history crosses in both directions with live positions behind it.
			a, b := len(ids)/3, 2*len(ids)/3
			step := func(lo, hi int) {
				for i := lo; i < hi; i++ {
					lg, err := st.Forward(ids[i])
					if err != nil {
						t.Fatalf("token %d: %v", i, err)
					}
					got[i] = append([]float32(nil), lg...)
				}
			}
			step(0, a)
			if n := st.SetGPULayers(0); n != 0 {
				t.Fatalf("SetGPULayers(0) left %d blocks placed", n)
			}
			step(a, b)
			if n := st.SetGPULayers(m.Cfg.NLayer); n != m.Cfg.NLayer {
				t.Fatalf("regrowing the seam took %d of %d blocks: %v",
					n, m.Cfg.NLayer, gpu.Err())
			}
			step(b, len(ids))
			worst, worstD := compare(t, want, got)
			t.Logf("%s: %d on the card, %d on the host, %d back on the card; "+
				"worst logit NMSE %.3e, worst |dlogit| %.3f", spec, a, b-a, len(ids)-b, worst, worstD)
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

// TestMLAOnADeviceWithPoisonedBuffers runs MLA with NaN in every fresh device
// buffer, so a read of memory nothing wrote cannot hide behind a zeroed
// allocation. Under MLA the cache row is written by two kernels at two offsets,
// so a gap between them would be read by every later position.
func TestMLAOnADeviceWithPoisonedBuffers(t *testing.T) {
	ids := []int32{5, 11, 23, 41, 67, 89}
	m, err := Open(hfContainer(t, "synth-deepseek-dense"), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	want := teacherForce(t, m, ids)
	ran := 0
	for _, spec := range stepDevices() {
		gpu, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff),
			tier.WithPoisonKV(true), tier.WithPoisonScratch(true))
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			st := m.NewState(len(ids) + 1)
			defer st.Close()
			st.SetDeviceLayers(gpu, -1)
			if n := st.GPULayers(); n != m.Cfg.NLayer {
				t.Fatalf("%s took %d of %d blocks: %v", spec, n, m.Cfg.NLayer, declineList(st))
			}
			got := runIDs(t, st, ids)
			worst, worstD := compare(t, want, got)
			t.Logf("%s: NaN in every fresh buffer, worst logit NMSE %.3e, |dlogit| %.3f",
				spec, worst, worstD)
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

// TestMLAStepsTheKVDownAtTheRightWidths covers the SHIPPING KV width on an MLA
// model (f16, the amd64 default), which every gate beside it pins to f32.
//
// SetDeviceLayers steps an f16 cache down to f32 and re-emits the host
// attention kernels; they must use MLA's widths (KVLoraRank+NRot scores,
// KVLoraRank values), not HeadDim. Those kernels only run on host blocks, so
// this places one block and leaves the rest home. It fails against
// stepDownKVToF32 going through AddAttn.
func TestMLAStepsTheKVDownAtTheRightWidths(t *testing.T) {
	ids := []int32{5, 11, 23, 41, 67, 89}
	ran := 0
	for _, name := range []string{"synth-deepseek-dense", "synth-deepseek"} {
		t.Run(name, func(t *testing.T) {
			src := hfContainer(t, name)
			// The REFERENCE is f16 too: this gate is about the step-down
			// emitting the right kernels, not about what an f16 cache costs.
			m, err := Open(src, noTune, WithKVF16(true))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if !m.Cfg.MLA() {
				t.Fatalf("%s is not an MLA model -- this gate would prove nothing", name)
			}
			if m.Cfg.NLayer < 2 {
				t.Fatalf("%s has %d block(s): a partial seam needs at least two",
					name, m.Cfg.NLayer)
			}
			want := teacherForce(t, m, ids)
			for _, spec := range stepDevices() {
				gpu, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
				if err != nil || gpu == nil {
					continue
				}
				ran++
				t.Run(spec, func(t *testing.T) {
					defer gpu.Close()
					st := m.NewState(len(ids) + 1)
					defer st.Close()
					if !st.KVIsF16() {
						t.Fatal("the cache is not f16, so no step-down happens and " +
							"this gate proved nothing")
					}
					// ONE block: the rest stay on the host and run the attention
					// kernels the step-down emitted.
					st.SetDeviceLayers(gpu, 1)
					if n := st.GPULayers(); n != 1 {
						t.Skipf("%s placed %d block(s), not 1", spec, n)
					}
					if st.KVIsF16() {
						t.Fatal("the cache is still f16 after SetDeviceLayers -- the " +
							"step-down did not run and this gate proved nothing")
					}
					got := runIDs(t, st, ids)
					worst, worstD := compare(t, want, got)
					t.Logf("%s: 1 of %d block(s) on %s, worst logit NMSE %.3e, worst |dlogit| %.4f",
						name, m.Cfg.NLayer, spec, worst, worstD)
				})
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

// TestNoPEMLAOnEveryDevice is TestMLAOnEveryDevice for a model whose attention
// applies no positional encoding, on a hybrid (Kimi-Linear: MLA full blocks and
// Kimi Delta Attention linear blocks), with every block of both kinds placed.
//
// Kimi-Linear keeps qk_rope_head_dim channels and never rotates them. A tier
// that rotates them anyway is exact at position 0 (rotation by 0 is the
// identity) and wrong after, so the gate runs several positions. Where the
// rotation is skipped the device must copy those channels instead (mlaRopeK and
// mlaRopeQ write into the cache row and the absorbed query).
func TestNoPEMLAOnEveryDevice(t *testing.T) {
	ids := []int32{5, 11, 23, 41, 67, 89, 101, 127}
	const name = "synth-kimilinear"
	src := hfContainer(t, name)
	m, err := Open(src, noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	switch {
	case !m.Cfg.MLA():
		t.Fatalf("%s is not an MLA model -- this gate would prove nothing", name)
	case !m.Cfg.NoPosEnc:
		t.Fatalf("%s does not set NoPosEnc, so the rotation this gate is about "+
			"still runs and a green line would mean nothing", name)
	case !m.Cfg.Hybrid():
		t.Fatalf("%s is not a hybrid -- see the comment above", name)
	}
	// Every block and both kinds: a tier that declined either kind would still
	// answer correctly from the host.
	wantPlaced, linear := m.Cfg.NLayer, 0
	for li := 0; li < m.Cfg.NLayer; li++ {
		if m.Cfg.LayerKind(li).Recurrent() {
			linear++
		}
	}
	if linear == 0 || linear == m.Cfg.NLayer {
		t.Fatalf("%s has %d linear block(s) of %d; this gate needs both kinds",
			name, linear, m.Cfg.NLayer)
	}
	want := teacherForce(t, m, ids)
	ran := 0
	for _, spec := range stepDevices() {
		gpu, err := openTier(t, tier.WithDevices(spec), tier.WithDeviceTune(tier.TuneOff))
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			st := m.NewState(len(ids) + 1)
			defer st.Close()
			st.SetDeviceLayers(gpu, -1)
			if n := st.GPULayers(); n != wantPlaced {
				for _, d := range declineList(st) {
					t.Logf("  declined x%d: %s", d.Blocks, d.Why)
				}
				t.Fatalf("%s took %d of %d block(s), %d of them linear: %v (tier: %q)",
					spec, n, wantPlaced, linear, declineList(st), gpu.Err())
			}
			got := runIDs(t, st, ids)
			if n := st.GPULayers(); n != wantPlaced {
				t.Fatalf("the session fell to the host (%d blocks left): %v", n, gpu.Err())
			}
			// The recurrence must have advanced: a linear block whose kernels
			// never ran leaves the summary at zero and the model still answers.
			if rs := st.recStepped(); rs != linear {
				t.Errorf("%s: recSteps is %d with %d linear block(s) placed -- every "+
					"linear block advances its recurrence once per call", spec, rs, linear)
			} else {
				t.Logf("%s: recSteps %d", spec, rs)
			}
			// The question a caller asks before re-running a failed step (the
			// server's joint step does) reads the same session's count.
			if !st.RecurrenceAdvanced() {
				t.Errorf("%s: RecurrenceAdvanced is false after %d linear block(s) stepped", spec, linear)
			}
			worst, worstD := compare(t, want, got)
			t.Logf("%s: %d of %d blocks on %s (%d linear), %d positions, worst logit "+
				"NMSE %.3e, worst |dlogit| %.3f", name, wantPlaced, m.Cfg.NLayer, spec,
				linear, len(want), worst, worstD)
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}
