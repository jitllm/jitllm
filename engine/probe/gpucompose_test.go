package probe

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestComposedGPUProbeMatchesTheBuilt holds the composed probe's numbers to
// the probe built from IR by hand, the two run in turn in one process: each
// number a composed form measures within a band of its median ratio over
// the rounds, twice the built probe's own spread over them and never under
// 10%. The numbers both arms still measure with built kernels are logged
// beside them, the harness's own control.
func TestComposedGPUProbeMatchesTheBuilt(t *testing.T) {
	devs := backend.OpenWith(backend.Opts{Admit: testdevices.Admit()})
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Fatal("no device: the gate proved nothing")
	}
	const rounds = 6
	for _, dev := range devs {
		t.Run(dev.API(), func(t *testing.T) {
			if _, ok := Timer(dev); !ok {
				t.Logf("%s: %s has no device clock: both arms time the wall", dev.Name(), dev.API())
			}
			var built, composed []cost.Device
			for range rounds {
				b, err := oldGPU(dev)
				if err != nil {
					t.Fatal(err)
				}
				c, err := GPU(dev)
				if err != nil {
					t.Fatal(err)
				}
				built, composed = append(built, b), append(composed, c)
			}
			names := [cost.NumClasses]string{"float", "special", "int", "index", "lanes", "group", "mem"}
			_, clocked := Timer(dev)
			type field struct {
				name     string
				of       func(cost.Device) float64
				composed bool
				// apart, where set, is why a composed number is not held to
				// the built one: the two measure different things.
				apart string
			}
			fields := []field{
				{"bandwidth", func(d cost.Device) float64 { return d.Bandwidth }, true, ""},
				{"floor", func(d cost.Device) float64 { return d.Floor }, true, ""},
				{"launch", func(d cost.Device) float64 { return d.Launch }, true, ""},
				{"L1 rate", func(d cost.Device) float64 { return d.Caches[0].Rate.Total }, false, ""},
				{"L2 rate", func(d cost.Device) float64 { return d.Caches[1].Rate.Total }, false, ""},
				{"overlap", func(d cost.Device) float64 { return d.Overlap }, false, ""},
				{"latency", func(d cost.Device) float64 { return d.Latency }, false, ""},
			}
			for cl := range cost.ClassMem {
				_, _, composed := gpuStep(cl)
				apart := ""
				switch {
				case composed && !clocked:
					apart = "the built arm times one launch's wall, its submission with it, which the composed arm's two depths cancel"
				case cl == cost.ClassSpecial:
					apart = "the device's JIT of the composed exponentials, straight code, against the built loop's: the same form through SPIR-V reaches the built rate on the same card"
				}
				// A thread and a chain run in an iteration's loop, not straight
				// code: held to the built loop on every class that composes.
				looped := ""
				if composed && !clocked {
					looped = "the built arm times one launch's wall, its submission with it, which the composed arm's two trip counts cancel"
				}
				fields = append(fields,
					field{names[cl] + " all", func(d cost.Device) float64 { return d.Rates[cl].Total }, composed, apart},
					field{names[cl] + " a thread", func(d cost.Device) float64 { return d.Rates[cl].Thread }, composed, looped},
					field{names[cl] + " delay", func(d cost.Device) float64 { return d.Delay[cl] }, composed, looped})
			}
			checked := 0
			for _, f := range fields {
				var b, ratios []float64
				for i := range rounds {
					x, y := f.of(built[i]), f.of(composed[i])
					if !(x > 0) || !(y > 0) || math.IsInf(x, 0) || math.IsInf(y, 0) {
						if f.composed {
							t.Errorf("%s: built %.4g, composed %.4g: not a measurement", f.name, x, y)
						}
						continue
					}
					b, ratios = append(b, x), append(ratios, y/x)
				}
				if len(ratios) < rounds {
					continue
				}
				slices.Sort(b)
				slices.Sort(ratios)
				band := max(0.10, 2*(b[len(b)-1]/b[0]-1))
				r := ratios[len(ratios)/2]
				arm := "built in both"
				switch {
				case f.apart != "":
					arm = "composed apart"
				case f.composed:
					arm = "composed"
				}
				t.Logf("%-16s %-13s built %.4g (spread %.3f), composed/built %.3f [%.3f, %.3f], band %.3f", f.name, arm, b[len(b)/2], b[len(b)-1]/b[0], r, ratios[0], ratios[len(ratios)-1], band)
				if f.apart != "" {
					t.Logf("%-16s not held to the built: %s", f.name, f.apart)
				}
				if !f.composed || f.apart != "" {
					continue
				}
				if math.Abs(math.Log(r)) > math.Log1p(band) {
					t.Errorf("%s: composed/built %.3f past the band %.3f", f.name, r, band)
				}
				checked++
			}
			if checked == 0 {
				t.Fatal("no number compared: the gate proved nothing")
			}
		})
	}
}
