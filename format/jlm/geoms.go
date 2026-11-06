package jlm

import "fmt"

// olderGeoms states, for a container an older writer made, the geometries it
// carried in fields of their own: the sliding layers' (swa, zero where the
// model's) and a double-width feed-forward on the KV-sharing blocks
// (Flag2DoubleFFN). A container that states Geoms carries neither.
func (c *Config) olderGeoms(swa Geom) error {
	wide := c.Flags2.Has(Flag2DoubleFFN)
	if swa == (Geom{}) && !wide {
		return nil
	}
	if len(c.Geoms) != 0 {
		return fmt.Errorf("jlm: a container states its blocks' geometries twice")
	}
	if swa != (Geom{}) && (c.SWAWindow == 0 || c.SWAPeriod == 0) {
		return fmt.Errorf("jlm: a sliding-layer geometry with no sliding layers")
	}
	c.Flags2 &^= Flag2DoubleFFN
	gs := make([]Geom, c.NLayer)
	for b := range gs {
		g := c.GeomAt(b)
		if c.swaOr(b, 1, 0) == 1 {
			g.HeadDim = c.swaOr(b, swa.HeadDim, g.HeadDim)
			g.NKVHead = c.swaOr(b, swa.NKVHead, g.NKVHead)
			g.NRot = c.swaOr(b, swa.NRot, g.NRot)
		}
		if wide && c.KVShared(b) {
			g.NFFN *= 2
		}
		gs[b] = g
	}
	c.Geoms = gs
	return nil
}

// geomsValid refuses a geometry no block can run: one per block or none, kv
// heads that group the query heads, a rotary inside its head.
func (c *Config) geomsValid() error {
	if len(c.Geoms) == 0 {
		return nil
	}
	if len(c.Geoms) != int(c.NLayer) {
		return fmt.Errorf("jlm: %d blocks' geometries for %d blocks", len(c.Geoms), c.NLayer)
	}
	for b, g := range c.Geoms {
		switch {
		case g.NKVHead == 0 || c.NHead%g.NKVHead != 0:
			return fmt.Errorf("jlm: block %d's %d kv heads for %d query heads", b, g.NKVHead, c.NHead)
		case g.HeadDim == 0 || g.NRot > g.HeadDim || g.NRot%2 != 0:
			return fmt.Errorf("jlm: block %d's rotary width %d on a %d-wide head", b, g.NRot, g.HeadDim)
		case g.NFFN == 0 && c.NFFN != 0:
			return fmt.Errorf("jlm: block %d has no feed-forward width", b)
		}
	}
	return nil
}
