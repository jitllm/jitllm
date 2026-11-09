package engine

import (
	"context"
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/nn"
)

// TestRoutedExpertsRunOnTheCPUBesideTheCard places mixture fixtures, with a
// shared expert and without, on a card with their routed experts on the CPU
// device (WithExperts("cpu")): every mixture block runs cut (cutBlock), its
// first half on the card and its experts on the CPU device, each served
// there at every step of the prompt and the decode; the logits are finite
// and as far from the CPU device's alone as the whole model on the card is,
// within a factor. A cut block whose second half ran on nothing would leave
// the experts out and part from both. Within the same band: a cut block moved
// to the CPU device whole and back cut mid-decode, and two sessions stepped
// as one routed step; a warm decode with the blocks cut allocates nothing;
// and the placement trial's arm of experts on the CPU device places the
// blocks cut.
func TestRoutedExpertsRunOnTheCPUBesideTheCard(t *testing.T) {
	ids := []int32{1, 5, 9, 13, 2}
	toks := []int32{3, 7, 11, 4}
	for _, name := range []string{"synth-qwen2moe", "synth-glm4moe", "synth-phimoe"} {
		for _, dev := range stepDevices() {
			t.Run(name+"/"+dev, func(t *testing.T) {
				m, _, openOpts := openMoE(t, name)
				defer m.Close()
				want := cpuDecode(t, m, ids, toks, openOpts...)
				card, _, _ := composedDecode(t, m, slices.Clip(openOpts), dev, ids, toks, nn.ComposedBlock, false, false)

				g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)...)
				if err != nil || g == nil {
					noDevice(t, dev, err)
				}
				defer g.Close()
				s := testInstance(t, m, len(ids)+len(toks)+4, append(slices.Clip(openOpts), WithExperts("cpu"), WithStreamTrial(false))...)
				defer s.Close()
				s.SetDeviceFallback(false)
				if err := s.SetDevice(g); err != nil {
					t.Fatal(err)
				}
				if s.GPULayers() != m.Cfg.NLayer {
					t.Fatalf("%s took %d of %d blocks: %s", dev, s.GPULayers(), m.Cfg.NLayer, g.Err())
				}
				mixtures := 0
				for li := range m.Cfg.NLayer {
					if !m.Cfg.MoEAt(li) {
						continue
					}
					if !s.cutAt(li) {
						t.Fatalf("mixture block %d is not cut, its experts placed on the CPU device (card: %q, declines: %v, prep: %v)",
							li, g.Err(), s.DeviceDeclines(), s.prepErr())
					}
					mixtures++
				}
				if mixtures == 0 {
					t.Fatal("the fixture has no mixture block")
				}
				cpu0, _ := s.Account(nn.CPU).Steps.Of(nn.ComposedBlock)
				st0 := g.Stats()
				l, err := s.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				var got [][]float32
				for _, tok := range toks {
					got = append(got, append([]float32(nil), l...))
					if l, err = s.Forward(tok); err != nil {
						t.Fatal(err)
					}
				}
				got = append(got, append([]float32(nil), l...))
				if s.DeviceDemotions() != 0 {
					t.Fatalf("the device failed %d time(s) and its blocks moved home", s.DeviceDemotions())
				}
				cpu1, declined := s.Account(nn.CPU).Steps.Of(nn.ComposedBlock)
				for i, n := range declined {
					if n > 0 {
						t.Fatalf("the CPU device declined %d block steps: %s", n, nn.DeclineName(i))
					}
				}
				// The prompt is a call of its rows, every token one.
				calls := 1 + len(toks)
				if served := cpu1 - cpu0; served != calls*mixtures {
					t.Fatalf("the CPU device served %d block steps, where %d calls over %d mixtures run %d", served, calls, mixtures, calls*mixtures)
				}
				if n := servedSince(st0, g.Stats(), nn.ComposedBlock); n == 0 {
					t.Fatal("the card served no block step")
				}
				worst, cardWorst := 0.0, 0.0
				for i := range want {
					e, c := logitNMSE(want[i], got[i]), logitNMSE(want[i], card[i])
					if math.IsNaN(e) || math.IsInf(e, 0) {
						t.Fatalf("step %d: non-finite logits (NMSE %v)", i, e)
					}
					worst, cardWorst = max(worst, e), max(cardWorst, c)
				}
				t.Logf("%d mixtures cut: worst NMSE %.3g from the CPU device's logits, the card's whole %.3g", mixtures, worst, cardWorst)
				if worst > 4*max(cardWorst, 1e-7) {
					t.Fatalf("the cut blocks part from the CPU device by %.3g, past 4x the whole card's %.3g", worst, cardWorst)
				}

				// A cut block moves to the CPU device whole mid-decode, its history
				// with it, and back cut, with the same answer within the band.
				li0 := -1
				for li := range m.Cfg.NLayer {
					if m.Cfg.MoEAt(li) {
						li0 = li
						break
					}
				}
				mv := testInstance(t, m, len(ids)+len(toks)+4, append(slices.Clip(openOpts), WithExperts("cpu"), WithStreamTrial(false))...)
				defer mv.Close()
				mv.SetDeviceFallback(false)
				if err := mv.SetDevice(g); err != nil {
					t.Fatal(err)
				}
				l, err = mv.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				moved := 0.0
				for i, tok := range toks {
					moved = max(moved, logitNMSE(want[i], l))
					switch i {
					case 1:
						if _, err := mv.MoveBlocks(li0, li0+1, nn.CPU); err != nil {
							t.Fatal(err)
						}
						if mv.cutAt(li0) || mv.devAt(li0) {
							t.Fatalf("block %d moved to the CPU device is still cut or on the card", li0)
						}
					case 2:
						if _, err := mv.MoveBlocks(li0, li0+1, dev); err != nil {
							t.Fatal(err)
						}
						if !mv.cutAt(li0) {
							t.Fatalf("block %d moved back to %s is not cut: its experts were placed on the CPU device", li0, dev)
						}
					}
					if l, err = mv.Forward(tok); err != nil {
						t.Fatal(err)
					}
				}
				moved = max(moved, logitNMSE(want[len(toks)], l))

				// Two sessions leasing one cut instance step as rows of one routed
				// step (stepRouted: no one device steps a cut instance whole),
				// each within the band.
				in := testInstance(t, m, len(ids)+len(toks)+4, append(slices.Clip(openOpts), WithExperts("cpu"), WithStreamTrial(false))...)
				defer in.Close()
				in.SetDeviceFallback(false)
				if err := in.SetDevice(g); err != nil {
					t.Fatal(err)
				}
				if err := in.StepRefusal(); err == nil {
					t.Fatal("a cut instance takes the joint arm, whose one device steps every block whole")
				}
				pool := NewPool()
				if err := pool.Serve(in, 0); err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := pool.Unserve(in); err != nil {
						t.Error(err)
					}
				}()
				ctx := context.Background()
				var sess [2]*Session
				for k := range sess {
					if sess[k], err = pool.NewSession(len(ids) + len(toks) + 4); err != nil {
						t.Fatal(err)
					}
					defer sess[k].Close()
					if err := sess[k].Run(ctx, PhasePrefill, func(in *Instance) error {
						_, err := in.Prefill(ids)
						return err
					}); err != nil {
						t.Fatal(err)
					}
				}
				rows0, joint := in.RoutedStepRows(), 0.0
				runs := make([]Run, len(sess))
				for i, tok := range toks {
					for k, se := range sess {
						if err := se.Acquire(ctx, PhaseGenerate); err != nil {
							t.Fatal(err)
						}
						runs[k] = Run{Session: se, Tokens: []int32{tok}, Logits: true}
					}
					out, err := StepRuns(runs)
					if err != nil {
						t.Fatal(err)
					}
					for _, lg := range out {
						joint = max(joint, logitNMSE(want[i+1], lg))
					}
				}
				if rows := in.RoutedStepRows() - rows0; rows != int64(2*len(toks)) {
					t.Fatalf("the routed step ran %d rows, where %d steps of two sessions run %d", rows, len(toks), 2*len(toks))
				}
				// A warm decode with blocks cut allocates nothing (RULE 2f).
				decodeAllocs(t, m, g, append(slices.Clip(openOpts), WithExperts("cpu"), WithStreamTrial(false))...)

				// The placement trial's arm with the experts on the CPU device
				// (initStreamTrial) places the blocks cut, from an instance placed
				// whole on the card.
				tr := testInstance(t, m, len(ids)+len(toks)+4, append(slices.Clip(openOpts), WithStreamTrial(false))...)
				defer tr.Close()
				tr.SetDeviceFallback(false)
				if err := tr.SetDevice(g); err != nil {
					t.Fatal(err)
				}
				if tr.anyCut() {
					t.Fatal("an instance placed with its experts where the card puts them runs cut")
				}
				tu := &seamTuner{trial: true, arms: []trialArm{{blocks: m.Cfg.NLayer}, {blocks: 0}, {blocks: m.Cfg.NLayer, experts: nn.CPU}}}
				if !tu.place(tr, 2) {
					t.Fatalf("the trial's arm of experts on the CPU device was not placed: %s", g.Err())
				}
				for li := range m.Cfg.NLayer {
					if m.Cfg.MoEAt(li) && !tr.cutAt(li) {
						t.Fatalf("the trial's arm of experts on the CPU device left mixture block %d whole", li)
					}
				}
				if _, err := tr.Prefill(ids); err != nil {
					t.Fatal(err)
				}
				t.Logf("two sessions stepped as one routed step: worst NMSE %.3g", joint)
				if math.IsNaN(joint) || joint > 4*max(cardWorst, 1e-7) {
					t.Fatalf("the routed step parts from the CPU device by %.3g, past 4x the whole card's %.3g", joint, cardWorst)
				}
				t.Logf("block %d moved to the CPU device and back cut: worst NMSE %.3g", li0, moved)
				if math.IsNaN(moved) || moved > 4*max(cardWorst, 1e-7) {
					t.Fatalf("the moves part from the CPU device by %.3g, past 4x the whole card's %.3g", moved, cardWorst)
				}
			})
		}
	}
}
