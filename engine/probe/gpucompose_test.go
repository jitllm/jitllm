package probe

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/engine/cost"
	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestComposedGPUProbeMatchesTheBuilt holds the composed probe's numbers to
// the probe built from IR by hand, the two run in turn in one process: each
// number a composed form measures within a band of its median ratio over
// the rounds, twice the built probe's own spread over them and never under
// 10%. A number the built measures wrong is held by bounds that say why:
// its median ratio not past one plus the band, and in every round over
// another number of the composed probe's by the band. A number the two arms
// take different work for is held to a range whose reason says how (took). The numbers both arms
// still measure with one built kernel are the harness's own control, held
// to the band too: the two arms timing one kernel apart is the harness's
// fault, whatever the composed numbers read.
func TestComposedGPUProbeMatchesTheBuilt(t *testing.T) {
	quiet(t)
	devs := backend.OpenWith(backend.Opts{Admit: testdevices.Admit()})
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		testmodels.Missing(t, "no device: the gate proved nothing")
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
				// bound, where set, is why a composed number is held under
				// the built one and over floor's number of its own round.
				bound string
				floor func(cost.Device) float64
			}
			walked := ""
			if !clocked {
				walked = "the built arm times one launch's wall, its submission with it, which the composed arm's two walks cancel"
			}
			fields := []field{
				{"bandwidth", func(d cost.Device) float64 { return d.Bandwidth }, true, "", "", nil},
				{"floor", func(d cost.Device) float64 { return d.Floor }, true, "", "", nil},
				{"launch", func(d cost.Device) float64 { return d.Launch }, true, "", "", nil},
				{"L1 rate", func(d cost.Device) float64 { return d.Caches[0].Rate.Total }, true, "", "", nil},
				{"L2 rate", func(d cost.Device) float64 { return d.Caches[1].Rate.Total }, true, "",
					"the built arm's workgroups each read the whole set from the same offsets, so a multiprocessor's workgroups read each other's lines from its own cache and the built counts those hits (on a V100 past the last cache's own rate); the composed arm's warps each stream the set from a phase of their own, so a multiprocessor reads more than its cache holds: under the built's rate, over the memory's bandwidth",
					func(d cost.Device) float64 { return d.Bandwidth }},
				{"overlap", func(d cost.Device) float64 { return d.Overlap }, true, "", "", nil},
				{"latency", func(d cost.Device) float64 { return d.Latency }, true, walked, "", nil},
				{"L1 latency", func(d cost.Device) float64 { return d.Caches[0].Latency }, true, walked, "", nil},
				{"L1 bytes", func(d cost.Device) float64 { return float64(d.Caches[0].Bytes) }, true, "", "", nil},
				{"L2 latency", func(d cost.Device) float64 { return d.Caches[1].Latency }, true, walked, "", nil},
			}
			// took is, by number, the range composed/built is held to where
			// the two arms take different work, and why.
			type within struct {
				lo, hi float64
				why    string
			}
			took := map[string]within{}
			issue := "a composed sum issues its add beside its exchange, a subgroup's issue the two share, where the built issues exchanges alone"
			feeds := "a composed exchange is timed feeding an add, as a butterfly's does, the add's own wait taken off (the estimator adds each node's along a path), where the built's feeds the next exchange"
			twice := "a composed sum over the workgroup waits on it twice a trip, its slots read before the next trip writes them (RULE 13), where the built waits once and lets the next trip's writes race the reads"
			switch {
			case clocked && dev.API() == "ptx":
				took["lanes a thread"] = within{0.6, 0.95, "(" + issue + "; measured 0.82 on a V100)"}
				took["lanes delay"] = within{1, 1.35, "(" + feeds + ": an add's result reaches an exchange later than an exchange's does; measured 1.21 on a V100)"}
				took["group all"] = within{1, 1.25, "(the built times one launch of 256 trips, the kernel's floor in it, which the composed arm's two trip counts cancel; measured 1.11 on a V100)"}
				took["group a thread"] = within{0.8, 1, "(" + twice + "; measured 0.90 on a V100)"}
				took["group delay"] = within{1, 1.25, "(" + twice + "; measured 1.10 on a V100)"}
			case clocked && dev.API() == "msl":
				took["lanes all"] = within{0.8, 1, "(" + issue + "; measured 0.90 on an M4)"}
				took["lanes a thread"] = within{0.6, 0.85, "(" + issue + "; measured 0.71 on an M4)"}
				took["lanes delay"] = within{0.6, 0.85, "(" + feeds + ": an exchange's result reaches an add sooner than it reaches an exchange; measured 0.71 on an M4)"}
				took["group all"] = within{0.7, 0.95, "(" + twice + "; measured 0.82 on an M4)"}
				took["group a thread"] = within{0.6, 0.85, "(" + twice + "; measured 0.72 on an M4)"}
				took["group delay"] = within{1.2, 1.6, "(" + twice + "; measured 1.41 on an M4)"}
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
					field{names[cl] + " all", func(d cost.Device) float64 { return d.Rates[cl].Total }, composed, apart, "", nil},
					field{names[cl] + " a thread", func(d cost.Device) float64 { return d.Rates[cl].Thread }, composed, looped, "", nil},
					field{names[cl] + " delay", func(d cost.Device) float64 { return d.Delay[cl] }, composed, looped, "", nil})
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
				case f.bound != "":
					arm = "composed bound"
				case f.composed:
					arm = "composed"
				}
				t.Logf("%-16s %-13s built %.4g (spread %.3f), composed/built %.3f [%.3f, %.3f], band %.3f", f.name, arm, b[len(b)/2], b[len(b)-1]/b[0], r, ratios[0], ratios[len(ratios)-1], band)
				if f.apart != "" {
					t.Logf("%-16s not held to the built: %s", f.name, f.apart)
				}
				if f.apart != "" {
					continue
				}
				if w, ok := took[f.name]; ok && f.composed {
					t.Logf("%-16s held to [%.2f, %.2f]: %s", f.name, w.lo, w.hi, w.why)
					if r < w.lo || r > w.hi {
						t.Errorf("%s: composed/built %.3f outside [%.2f, %.2f]", f.name, r, w.lo, w.hi)
					}
					checked++
					continue
				}
				if !f.composed {
					if math.Abs(math.Log(r)) > math.Log1p(band) {
						t.Errorf("%s: the arms time one built kernel %.3f apart, past the band %.3f: the harness's control", f.name, r, band)
					}
					continue
				}
				if f.bound != "" {
					t.Logf("%-16s held by bounds: %s", f.name, f.bound)
					if r > 1+band {
						t.Errorf("%s: composed/built %.3f past 1 plus the band %.3f", f.name, r, band)
					}
					for i, c := range composed {
						if y, fl := f.of(c), f.floor(c); !(y > (1+band)*fl) {
							t.Errorf("%s: round %d's composed %.4g not over %.4g by the band %.3f", f.name, i, y, fl, band)
						}
					}
					checked++
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
