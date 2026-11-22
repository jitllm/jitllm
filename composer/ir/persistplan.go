package ir

import "fmt"

// The plan of a block's launches as persistent kernels: which launches share
// one, at what grid, ordered how. The device collects the launches, compiles
// what comes back and states its grid limit; every choice of the kernel's
// shape is made here.

// Joins reports whether launch p can join a persistent kernel of width w,
// and why not: PersistJoins on its kernel, and no buffer bound to two of its
// parameters, one read and one written -- an in-place launch whose phases
// Persist cannot see inside.
func (p Part) Joins(w int) error {
	if p.K == nil {
		return fmt.Errorf("no IR")
	}
	if err := PersistJoins(p.K, w); err != nil {
		return err
	}
	if len(p.Bind) != len(p.K.Params) {
		return fmt.Errorf("ir: %s binds %d of %d parameters", p.K.Name, len(p.Bind), len(p.K.Params))
	}
	rs, ws := p.K.access()
	for _, r := range rs {
		for _, x := range ws {
			if r != x && p.Bind[r] == p.Bind[x] {
				return fmt.Errorf("%s binds one buffer as an input and an output", p.K.Name)
			}
		}
	}
	return nil
}

// PersistRun is launches [Lo, Hi) of a sequence: composed into one
// persistent kernel of width W when they are two or more, else issued alone,
// Why saying why a launch could not join even a run of its own.
type PersistRun struct {
	Lo, Hi, W int
	Why       error
}

// Composed reports whether the run is composed rather than issued alone.
func (r PersistRun) Composed() bool { return r.Hi-r.Lo >= 2 }

// PersistRuns cuts a sequence of launches into runs: each run takes the
// width of its first launch and every following launch that joins it.
func PersistRuns(parts []Part) []PersistRun {
	var runs []PersistRun
	for i := 0; i < len(parts); {
		w := parts[i].Width
		j := i
		for j < len(parts) && parts[j].Joins(w) == nil {
			j++
		}
		r := PersistRun{Lo: i, Hi: max(j, i+1), W: w}
		if j == i {
			r.Why = parts[i].Joins(w)
		}
		runs = append(runs, r)
		i = r.Hi
	}
	return runs
}

// PersistGrid is the grid a run is first composed at: its widest launch's
// workgroups, capped by limit when limit is positive.
func PersistGrid(parts []Part, limit int) int {
	most := 0
	for _, p := range parts {
		most = max(most, p.Groups)
	}
	if limit > 0 {
		most = min(most, limit)
	}
	return most
}

// DepRegion is the words of the dep buffer, [Base, End), a kernel ordered by
// dependency counters may take. The zero region orders with grid barriers.
type DepRegion struct{ Base, End int }

// PersistAt composes a run's launches at width w on grid workgroups: with
// dependency counters from dep.Base when dep is not empty and the counters
// fit before dep.End, else with grid barriers (Persisted.Dep says which).
func PersistAt(parts []Part, w, grid int, dep DepRegion) (*Persisted, error) {
	if dep.End > dep.Base {
		pk, err := PersistDeps(fmt.Sprintf("persistdep%d_%d", len(parts), grid), parts, w, grid, dep.Base)
		if err != nil {
			return nil, err
		}
		if dep.Base+pk.Words <= dep.End {
			pk.Dep = true
			for _, o := range pk.K.Ops {
				if o.Kind == OpDepWait && o.Imm&DepTile != 0 {
					pk.Tiles++
				}
			}
			return pk, nil
		}
	}
	return Persist(fmt.Sprintf("persist%d_%d", len(parts), grid), parts, w, grid)
}
