package gpu

import (
	"fmt"

	"github.com/jitllm/jitllm/composer"
)

// entriesMigrate moves block li's entries plane of session sid between host
// and device: e is the entries pos positions closed, in order, an entry
// kvRow wide, entry j at slot ((j+1)*period-1)%P/period of the page of
// position (j+1)*period-1 (composer.History.Period). The rows' move
// (pagedMigrate) ranged the sequence first; pages at home are written and
// read there. Callers hold g.mu.
func (g *devTier) entriesMigrate(sid uint64, li int, e []float32, pos int, toDevice bool) error {
	kp := g.kvPages()
	pl, ok := kp.layers[entKey(li)]
	if !ok {
		return fmt.Errorf("block %d keeps no entries on this device", li)
	}
	s := seqID{sid, 0}
	P, per, w := kp.p, pl.geom.period, pl.geom.kvRow
	n := pos / per
	if n*w > len(e) {
		return fmt.Errorf("%d entries of %d floats do not fit the host's %d", n, w, len(e))
	}
	if n == 0 {
		return nil
	}
	npages := (pos + P - 1) / P
	kw := composer.History{Period: per}.Slots(P) * w
	ev := min(len(pl.home[s]), npages)
	if toDevice {
		if ev > 0 && len(pl.owned[s]) == 0 {
			if err := g.ensureRange(kp, s, ev); err != nil {
				return err
			}
			pl.owned[s] = make([]uint32, ev)
			if err := pl.tab.WriteAt(kp.ranges[s].off*4, u32view(pl.owned[s])); err != nil {
				return err
			}
		}
		if have := len(pl.owned[s]); npages > have {
			if err := g.allocPages(sid, kp, pl, s, npages-have); err != nil {
				return err
			}
		}
		if err := g.flushTabs(); err != nil {
			return err
		}
	}
	ids := pl.owned[s]
	if len(ids) < npages {
		return fmt.Errorf("the sequence holds %d entry page(s) and %d positions need %d", len(ids), pos, npages)
	}
	pages := make([][]float32, npages)
	if toDevice {
		for j := range pages {
			pages[j] = make([]float32, kw)
		}
		for j := 0; j < n; j++ {
			q := (j+1)*per - 1
			at := q % P / per * w
			copy(pages[q/P][at:at+w], e[j*w:(j+1)*w])
		}
		for j, page := range pages {
			if ids[j] == 0 {
				if pl.home == nil {
					pl.home = map[seqID][][]float32{}
				}
				for len(pl.home[s]) <= j {
					pl.home[s] = append(pl.home[s], nil)
				}
				pl.home[s][j] = page
				continue
			}
			if err := pl.k.WriteAt(int(ids[j])*kw*4, f32b(page)); err != nil {
				return err
			}
		}
		return nil
	}
	all, err := g.readPages(pl, pl.k, ids[ev:npages], kw)
	if err != nil {
		return err
	}
	for j := range pages {
		if j < ev {
			if j >= len(pl.home[s]) || pl.home[s][j] == nil {
				return fmt.Errorf("entry page %d is at home and holds nothing", j)
			}
			pages[j] = pl.home[s][j][:kw]
			continue
		}
		pages[j] = all[(j-ev)*kw : (j-ev+1)*kw]
	}
	for j := 0; j < n; j++ {
		q := (j+1)*per - 1
		at := q % P / per * w
		copy(e[j*w:(j+1)*w], pages[q/P][at:at+w])
	}
	return nil
}
