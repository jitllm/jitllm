package jit

import (
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// Estimate is placement p's cost at step s, from the IR its passes leave.
func (r *Registry) Estimate(f *jlm.File, s Step, p Placement) (Cost, error) {
	parts, err := front(f)
	if err != nil {
		return Cost{}, err
	}
	run := make([]PartPlan, len(parts))
	for i, part := range parts {
		run[i] = PartPlan{Part: part, Device: p(part)}
	}
	var total Cost
	var errs []error
	for i := 0; i < len(run); {
		j := i + 1
		for j < len(run) && run[j].Device == run[i].Device {
			j++
		}
		ks, err := r.passRun(f, run[i:j], i, s)
		errs = append(errs, err)
		m := costModel{d: run[i].Device, s: s}
		for _, k := range ks {
			total = total.add(m.cost(k))
		}
		i = j
	}
	for i := 1; i < len(run); i++ {
		total = total.add(seam(run[i-1], run[i], s))
	}
	return total, errors.Join(errs...)
}

// CodeCost is one call's estimate at a step, beside the call and what the
// estimate assumed of it: the words it passes by value, by field, and for
// each word that holds an address the bytes the call reaches past it.
type CodeCost struct {
	Code   Code
	Part   int
	Device Device
	Cost   Cost
	Values map[int]uint64
	Reach  map[int]int64
}

// EstimateCodes is each call's estimate when placement p runs at step s,
// in the order the step makes them, in the steady state of decoding: the
// caches as a step before this one left them.
func (r *Registry) EstimateCodes(f *jlm.File, s Step, p Placement) ([]CodeCost, error) {
	parts, err := front(f)
	if err != nil {
		return nil, err
	}
	run := make([]PartPlan, len(parts))
	for i, part := range parts {
		run[i] = PartPlan{Part: part, Device: p(part)}
	}
	var out []CodeCost
	var errs []error
	for i := 0; i < len(run); {
		j := i + 1
		for j < len(run) && run[j].Device == run[i].Device {
			j++
		}
		d := run[i].Device
		ks, err := r.passRun(f, run[i:j], i, s)
		errs = append(errs, err)
		m := costModel{d: d, s: s}
		for _, k := range ks {
			m.cost(k)
		}
		for _, k := range ks {
			c := m.cost(k)
			codes, err := r.lower([]*Kernel{k}, d)
			if err != nil || len(codes) != 1 {
				errs = append(errs, err)
				continue
			}
			e := estimator{k: k, s: s, trips: map[int]float64{}, homes: k.homes()}
			cc := CodeCost{Code: codes[0].Code, Part: k.Of, Device: d, Cost: c, Values: map[int]uint64{}, Reach: map[int]int64{}}
			for _, a := range k.Args {
				switch {
				case a.Buf < 0:
					if v, ok := e.eval(a.Node-1, 0); ok {
						cc.Values[a.Field] = uint64(v)
					}
				case k.Bufs[a.Buf].Word > 0:
					w := 0.0
					if off := k.Bufs[a.Buf].Offset; off > 0 {
						if v, ok := e.eval(off-1, 0); ok {
							w = v
						}
					}
					if v, ok := stepWord(k.Bufs[a.Buf].Operand, int(w), s); ok {
						cc.Values[a.Field] = uint64(v)
					}
				default:
					cc.Reach[a.Field] = max(cc.Reach[a.Field], m.reach[a.Buf])
				}
			}
			out = append(out, cc)
		}
		i = j
	}
	return out, errors.Join(errs...)
}

// seam is what carrying a part's boundary value to the next part costs
// where the two sit on different devices.
func seam(a, b PartPlan, s Step) Cost {
	if a.Device == b.Device || len(a.Part.Graph.Out) == 0 {
		return Cost{}
	}
	n := int64(4 * s.Rows * a.Part.Graph.Values[a.Part.Graph.Out[0]].Width)
	c := Cost{Carried: n}
	if l := min(a.Device.Link, b.Device.Link); l > 0 {
		c.Seconds = float64(n) / l
	}
	return c
}

// Suggest is plan mode: a placement over devices at step s, each part on
// the device that makes the model's estimate least, the seams' transfers
// counted.
func (r *Registry) Suggest(f *jlm.File, s Step, devices []Device) (Placement, Cost, error) {
	if len(devices) == 0 {
		return nil, Cost{}, fmt.Errorf("jit: no device to place on")
	}
	parts, err := front(f)
	if err != nil {
		return nil, Cost{}, err
	}
	n, m := len(parts), len(devices)
	own := make([][]Cost, n)
	for i, part := range parts {
		own[i] = make([]Cost, m)
		for j, d := range devices {
			ks, err := r.passRun(f, []PartPlan{{Part: part, Device: d}}, i, s)
			if err != nil {
				own[i][j] = Cost{Seconds: 1e30}
				continue
			}
			m := costModel{d: d, s: s}
			for _, k := range ks {
				own[i][j] = own[i][j].add(m.cost(k))
			}
		}
	}
	best := make([][]Cost, n)
	from := make([][]int, n)
	for i := range best {
		best[i], from[i] = make([]Cost, m), make([]int, m)
		for j := range devices {
			if i == 0 {
				best[i][j] = own[i][j]
				continue
			}
			from[i][j] = -1
			for q := range devices {
				c := best[i-1][q].add(seam(PartPlan{Part: parts[i-1], Device: devices[q]}, PartPlan{Part: parts[i], Device: devices[j]}, s)).add(own[i][j])
				if from[i][j] < 0 || c.Seconds < best[i][j].Seconds {
					best[i][j], from[i][j] = c, q
				}
			}
		}
	}
	end := 0
	for j := range devices {
		if best[n-1][j].Seconds < best[n-1][end].Seconds {
			end = j
		}
	}
	pick := make([]int, n)
	pick[n-1] = end
	for i := n - 1; i > 0; i-- {
		pick[i-1] = from[i][pick[i]]
	}
	at := map[[2]int]Device{}
	for i, part := range parts {
		at[[2]int{int(part.Kind), part.Block}] = devices[pick[i]]
	}
	return func(p Part) Device { return at[[2]int{int(p.Kind), p.Block}] }, best[n-1][end], nil
}
