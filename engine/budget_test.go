package engine

import (
	"fmt"
	"os"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestDeviceBudgetAccounting prints what a base-resident placement would cost,
// separating the three quantities a placement decision needs and that a single
// "bytes per token" figure runs together.
//
// A uniform page is sized by the max base over both layer kinds, not an average,
// and the three quantities are kept apart because comparing one against another
// is how a nonexistent saving gets claimed:
//
//	LOGICAL  what the arithmetic of a token touches (BytesPerToken)
//	STORAGE  what a residency decision has to hold
//	PCIe     what a device that does not hold it has to be sent
//
// JITLLM_BUDGET=<path-to-.jlm> selects the model. It prints and asserts nothing,
// so it is a measurement and not a gate.
func TestDeviceBudgetAccounting(t *testing.T) {
	path := os.Getenv("JITLLM_BUDGET")
	if path == "" {
		t.Skip("set JITLLM_BUDGET=<model.jlm> to print the placement accounting")
	}
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	c := m.Cfg
	mib := func(u uint64) string { return fmt.Sprintf("%9.2f MiB", float64(u)/(1<<20)) }
	gib := func(u uint64) string { return fmt.Sprintf("%7.3f GiB", float64(u)/(1<<30)) }

	// STORAGE: base bytes per block, and the max a uniform page must take.
	type acct struct{ base, exp uint64 }
	per := make([]acct, c.NLayer)
	var dense uint64
	for i := range m.Container.Entries() {
		e := &m.Container.Entries()[i]
		qs, d, sc := m.Container.SpanLen(e)
		n := qs + d + sc
		if e.Block == jlm.DenseBlock {
			dense += n
			continue
		}
		if int(e.Block) >= len(per) {
			continue // a vision block; not part of this accounting
		}
		if e.NDim == 3 || e.Index >= 0 {
			per[e.Block].exp += n
		} else {
			per[e.Block].base += n
		}
	}
	var maxBase, sumBase, sumExp uint64
	nAttn, nLin := 0, 0
	for li := range per {
		sumBase += per[li].base
		sumExp += per[li].exp
		if per[li].base > maxBase {
			maxBase = per[li].base
		}
		if c.LayerKind(li).Recurrent() {
			nLin++
		} else {
			nAttn++
		}
	}
	uniform := maxBase * uint64(c.NLayer)
	t.Logf("%d layers (%d attention, %d linear), page %s",
		c.NLayer, nAttn, nLin, mib(m.PageSize()))
	t.Logf("STORAGE  base per block max %s -> %s for all %d, uniform page",
		mib(maxBase), gib(uniform), c.NLayer)
	t.Logf("STORAGE  expert banks %s   dense region %s   total %s",
		gib(sumExp), mib(dense), gib(sumBase+sumExp+dense))

	// STORAGE: a streamed placement (gpu.Config.StreamExperts) costs the
	// container base plus the compact bank of NExpertUsed sheets of gate, up and
	// down that the device holds for the routed experts.
	var sheet uint64
	if c.NExpert > 0 && c.NExpertUsed > 0 {
		for i := range m.Container.Entries() {
			e := &m.Container.Entries()[i]
			if e.Block != 0 {
				continue
			}
			switch e.Role {
			case jlm.RoleExpGateBank, jlm.RoleExpUpBank, jlm.RoleExpDownBank:
				qs, d, sc, sheets, err := m.Container.SheetSpan(e)
				if err != nil || sheets == 0 {
					continue
				}
				sheet += qs + d + sc
			}
		}
	}
	compact := sheet * uint64(c.NExpertUsed)
	streamed := (maxBase + compact) * uint64(c.NLayer)
	t.Logf("STREAMED one sheet of gate+up+down %s, x%d routed = %s a block",
		mib(sheet), c.NExpertUsed, mib(compact))
	t.Logf("STREAMED a placed block is base %s + bank %s = %s; x%d = %s",
		mib(maxBase), mib(compact), mib(maxBase+compact), c.NLayer, gib(streamed))

	var expPerTok uint64
	if c.NExpert > 0 {
		expPerTok = sumExp / uint64(c.NExpert) * uint64(c.NExpertUsed)
	}

	// What the pager actually reads for one token of a fully placed model. The
	// residency unit is jlm.Chunk (1 MiB), so a 16 KiB plane costs a whole
	// chunk; this counts the distinct chunks a token's plan touches against the
	// bytes it wants.
	if c.NExpert > 0 && c.NExpertUsed > 0 {
		sel := make([]uint32, c.NExpertUsed)
		for i := range sel {
			// Spread across the bank as a router does; 0..k-1 would share
			// chunks and flatter the count.
			sel[i] = uint32(i * c.NExpert / c.NExpertUsed)
		}
		type span struct{ off, n uint64 }
		var wantB uint64
		chunks := map[uint64]bool{}
		add := func(off, n uint64) {
			if n == 0 {
				return
			}
			wantB += n
			for ch := off / jlm.Chunk; ch <= (off+n-1)/jlm.Chunk; ch++ {
				chunks[ch] = true
			}
		}
		var baseB uint64
		baseChunks := map[uint64]bool{}
		for i := range m.Container.Entries() {
			e := &m.Container.Entries()[i]
			if e.Block != 0 || e.Role.Vision() || e.Role.Expanded() {
				continue
			}
			switch e.Role {
			case jlm.RoleExpGateBank, jlm.RoleExpUpBank, jlm.RoleExpDownBank:
				qs, d, sc, sheets, err := m.Container.SheetSpan(e)
				if err != nil || sheets == 0 {
					continue
				}
				for _, x := range sel {
					o := uint64(x)
					add(e.QSOff+o*qs, qs)
					add(e.DOff+o*d, d)
					add(e.SCOff+o*sc, sc)
				}
			default:
				q, d2, s2 := m.Container.SpanLen(e)
				baseB += q + d2 + s2
				for _, sp := range []span{{e.QSOff, q}, {e.DOff, d2}, {e.SCOff, s2}} {
					if sp.n == 0 {
						continue
					}
					for ch := sp.off / jlm.Chunk; ch <= (sp.off+sp.n-1)/jlm.Chunk; ch++ {
						baseChunks[ch] = true
					}
				}
			}
		}
		rd := uint64(len(chunks)) * jlm.Chunk
		bs := uint64(len(baseChunks)) * jlm.Chunk
		t.Logf("READ     one block's routed sheets: want %s, %d chunk(s) = %s (%.1fx)",
			mib(wantB), len(chunks), mib(rd), float64(rd)/float64(wantB))
		t.Logf("READ     one block's base: want %s, %d chunk(s) = %s",
			mib(baseB), len(baseChunks), mib(bs))
		t.Logf("READ     a token over %d blocks: sheets %s + base %s = %s "+
			"(against %s of experts consumed)",
			c.NLayer, gib(rd*uint64(c.NLayer)), gib(bs*uint64(c.NLayer)),
			gib((rd+bs)*uint64(c.NLayer)), gib(expPerTok))
	}

	// The layout inside one page: if every non-bank tensor sits below every bank
	// tensor, a frame could be allocated to the base extent alone and the sheets
	// read elsewhere. If they interleave, that needs a container change.
	{
		base0, _ := m.Container.PageBounds(0)
		var baseEnd, bankStart uint64
		bankStart = ^uint64(0)
		for i := range m.Container.Entries() {
			e := &m.Container.Entries()[i]
			if e.Block != 0 || e.Role.Vision() || e.Role.Expanded() {
				continue
			}
			q, d2, s2 := m.Container.SpanLen(e)
			lo, hi := e.QSOff, e.QSOff+q
			for _, sp := range [][2]uint64{{e.DOff, e.DOff + d2}, {e.SCOff, e.SCOff + s2}} {
				if sp[1] == sp[0] {
					continue
				}
				if sp[0] < lo {
					lo = sp[0]
				}
				if sp[1] > hi {
					hi = sp[1]
				}
			}
			switch e.Role {
			case jlm.RoleExpGateBank, jlm.RoleExpUpBank, jlm.RoleExpDownBank:
				if lo < bankStart {
					bankStart = lo
				}
			default:
				if hi > baseEnd {
					baseEnd = hi
				}
			}
		}
		t.Logf("LAYOUT   page starts at %d; base ends at +%s, bank starts at +%s -- %s",
			base0, mib(baseEnd-base0), mib(bankStart-base0),
			map[bool]string{true: "SEPARABLE: a frame could be the base extent",
				false: "INTERLEAVED: the bank is not a suffix"}[baseEnd <= bankStart])
	}

	// STORAGE: what else a session charges the same card.
	one := uint64(0)
	if nLin > 0 {
		g := c.Delta()
		one = uint64(g.DeltaStateLen()+g.ConvStateLen()) * 4
	}
	rec := 2 * one * uint64(nLin)
	t.Logf("STORAGE  recurrent, ONE session: %s (2 buffers x %d linear x %s)",
		mib(rec), nLin, mib(one))
	for _, ctx := range []int{512, 2048, 8192} {
		kv := uint64(ctx) * uint64(c.KVDim()) * 4 * 2 * uint64(nAttn)
		t.Logf("STORAGE  KV at ctx %5d, ONE session: %s", ctx, mib(kv))
	}

	// LOGICAL / PCIe: what a decode token touches.
	t.Logf("LOGICAL  a decode token: base %s + routed experts %s + head %s",
		gib(sumBase), gib(expPerTok), mib(m.HeadReadBytes()))
	t.Logf("LOGICAL  BytesPerToken() (a weights-CONSUMED estimate): %s",
		gib(m.BytesPerToken()))
	t.Logf("PCIe     with base RESIDENT, a token sends only the routed experts: %s",
		gib(expPerTok))
	t.Logf("PCIe     with nothing resident, a token sends: %s", gib(sumBase+expPerTok))

	// The decision this is for, at card sizes in the CLI's units (GiB after
	// headroom): a 4 GB card reports about 1.9.
	for _, card := range []uint64{1930 << 20, 3500 << 20, 7000 << 20} {
		// Against the streamed cost, not the base alone: a placed block holds
		// its compact bank for the whole run.
		fit := int(c.NLayer)
		if per := maxBase + compact; per > 0 {
			if n := int((card - rec) / per); n < fit {
				fit = n
			}
		}
		t.Logf("FITS?    card %s STREAMED: %d of %d blocks at %s each (+%s recurrent)",
			gib(card), fit, c.NLayer, mib(maxBase+compact), mib(rec))
		left := int64(card) - int64(uniform) - int64(rec)
		var slots int64
		if c.NExpert > 0 && left > 0 {
			slots = left / int64(sumExp/uint64(c.NExpert)/uint64(c.NLayer))
		}
		t.Logf("FITS?    card %s: base+recurrent %s, %s left = %d expert slot(s) "+
			"(a token routes %d, one layer at a time)",
			gib(card), gib(uniform+rec), gib(uint64(max64i(left, 0))), slots,
			int(c.NExpertUsed)*int(c.NLayer))
	}
}

func max64i(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
