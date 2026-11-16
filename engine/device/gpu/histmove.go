package gpu

import (
	"fmt"
)

// historyMigrate moves block li's history h past its rows, of session sid,
// between host and device: e is the entries pos positions closed, in order,
// an entry its kvRow keys and, where its values are not its keys, its kvRow
// values after them, entry j at slot ((j+1)*period-1)%P/period of the page
// of position (j+1)*period-1 (composer.History.Period). A page is its keys
// and its values after them, as eviction sends it home. The rows' move
// (pagedMigrate) ranged the sequence first; pages at home are written and
// read there. Callers hold g.mu.
func (g *devTier) historyMigrate(sid uint64, li, h int, e []float32, pos int, toDevice bool) error {
	kp := g.kvPages()
	pl, ok := kp.layers[histKey(li, h)]
	if h < 1 || !ok {
		return fmt.Errorf("block %d keeps no history %d on this device", li, h)
	}
	s := seqID{sid, 0}
	P, per, w := kp.p, max(1, pl.geom.period), pl.geom.kvRow
	kw, vw := pl.geom.kWords(P), pl.geom.vWords(P)
	ew := w
	if vw > 0 {
		ew = 2 * w
	}
	n := pos / per
	if n*ew > len(e) {
		return fmt.Errorf("%d entries of %d floats do not fit the host's %d", n, ew, len(e))
	}
	if n == 0 {
		return nil
	}
	npages := (pos + P - 1) / P
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
			pages[j] = make([]float32, kw+vw)
		}
		for j := 0; j < n; j++ {
			q := (j+1)*per - 1
			at := q % P / per * w
			copy(pages[q/P][at:at+w], e[j*ew:j*ew+w])
			if vw > 0 {
				copy(pages[q/P][kw+at:kw+at+w], e[j*ew+w:(j+1)*ew])
			}
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
			if err := pl.k.WriteAt(int(ids[j])*kw*4, f32b(page[:kw])); err != nil {
				return err
			}
			if vw > 0 {
				if err := pl.v.WriteAt(int(ids[j])*vw*4, f32b(page[kw:])); err != nil {
					return err
				}
			}
		}
		return nil
	}
	keys, err := g.readPages(pl, pl.k, ids[ev:npages], kw)
	if err != nil {
		return err
	}
	vals, err := g.readPages(pl, pl.v, ids[ev:npages], vw)
	if err != nil {
		return err
	}
	for j := range pages {
		if j < ev {
			if j >= len(pl.home[s]) || len(pl.home[s][j]) < kw+vw {
				return fmt.Errorf("entry page %d is at home and holds nothing", j)
			}
			pages[j] = pl.home[s][j][:kw+vw]
			continue
		}
		pages[j] = append(keys[(j-ev)*kw:(j-ev+1)*kw:(j-ev+1)*kw], vals[(j-ev)*vw:(j-ev+1)*vw]...)
	}
	for j := 0; j < n; j++ {
		q := (j+1)*per - 1
		at := q % P / per * w
		copy(e[j*ew:j*ew+w], pages[q/P][at:at+w])
		if vw > 0 {
			copy(e[j*ew+w:(j+1)*ew], pages[q/P][kw+at:kw+at+w])
		}
	}
	return nil
}
