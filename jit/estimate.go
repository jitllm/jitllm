package jit

import (
	"errors"
	"fmt"

	"github.com/jitllm/jitllm/format/jlm"
)

// Cost is an estimate of a placement from its final IR, nothing lowered:
// the bytes its kernels move, their launches and the bytes carried between
// devices, and the seconds the devices' descriptions make of them.
type Cost struct {
	Bytes, Carried int64
	Launches       int
	Seconds        float64
}

func (c Cost) add(o Cost) Cost {
	return Cost{Bytes: c.Bytes + o.Bytes, Carried: c.Carried + o.Carried, Launches: c.Launches + o.Launches, Seconds: c.Seconds + o.Seconds}
}

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
		for _, k := range ks {
			total = total.add(k.cost(run[i].Device))
		}
		i = j
	}
	for i := 1; i < len(run); i++ {
		total = total.add(seam(run[i-1], run[i], s))
	}
	return total, errors.Join(errs...)
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

// cost is kernel k's estimate on device d: each buffer a word at every
// address its widest load or store takes, bounded by its shape.
func (k *Kernel) cost(d Device) Cost {
	per := map[int]int64{}
	note := func(b int, f int64) {
		if sh := k.Bufs[b].Shape; sh != nil {
			n := int64(1)
			for _, e := range sh {
				n *= int64(e)
			}
			f = min(f, n)
		}
		per[b] = max(per[b], f)
	}
	live := k.live()
	for i, nd := range k.Nodes {
		if nd.Op == OpLoad && live[i] && k.Bufs[nd.Buf].Kind != BufShared {
			note(nd.Buf, k.footprint(i))
		}
	}
	for _, s := range k.Stores {
		if s.Barrier || s.Run || k.Bufs[s.Buf].Kind == BufShared {
			continue
		}
		f := int64(1)
		if s.Ax != nil {
			for _, a := range s.Ax {
				f *= k.footprint(a)
			}
		} else {
			f = k.footprint(s.Ix)
		}
		note(s.Buf, f)
	}
	var words int64
	for _, f := range per {
		words += f
	}
	c := Cost{Bytes: 4 * words, Launches: 1}
	if d.Bandwidth > 0 {
		c.Seconds = float64(c.Bytes)/d.Bandwidth + d.Launch
	}
	return c
}

// footprint is how many values node i takes over a launch: the product of
// the extents of the coordinates, thread indices and loops it reads.
func (k *Kernel) footprint(i int) int64 {
	seen := map[int]bool{}
	axes, loops := map[int]bool{}, map[int]bool{}
	thread := false
	var walk func(i int)
	walk = func(i int) {
		if seen[i] {
			return
		}
		seen[i] = true
		nd := k.Nodes[i]
		switch nd.Op {
		case OpCoord:
			axes[nd.A] = true
		case OpIter:
			for c := nd.At; c >= 0; c = k.Loops[c].Parent {
				loops[c] = true
			}
		case OpThread, OpLocal, OpFlat:
			thread = true
		}
		if nd.Op == OpLoad && nd.Ax == nil {
			walk(nd.A)
			return
		}
		for _, a := range k.args(&nd) {
			walk(*a)
		}
	}
	walk(i)
	f := int64(1)
	for a := range axes {
		f *= int64(k.Space[a])
	}
	for c := range loops {
		f *= int64(max(1, k.Loops[c].Trip))
	}
	if thread {
		f *= int64(max(1, k.Group))
	}
	return f
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
			for _, k := range ks {
				own[i][j] = own[i][j].add(k.cost(d))
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
