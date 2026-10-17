package tier

import (
	"fmt"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/backend"
)

// Experts on a second card.
//
// Config.ExpertCard names a device that holds no blocks: it keeps whole expert
// banks for the mixture blocks the other cards run. A block's expert cache on
// its own card misses an expert; instead of the host reading the expert's
// pages and sending the sheets across the host bus, the sheet is copied card
// to card (backend.PeerCopier: NVLink or PCIe peer access, staged through the
// host by the driver where the cards have neither) out of the expert card's
// resident bank into the cache slot. The router, the hit path and the kernels
// are the ones without an expert card, and so are the answers: the slot holds
// the same bytes the host would have sent.
//
// A bank goes to the expert card at the end of a placement (attachExpertCard),
// in block order, for as many blocks as the card has room for; a block it has
// no room for keeps being served by the host. The bank is the block's: it is
// freed with the block (releaseBlock), and its bytes are the expert card's
// resident charge.

// peerBank is one block's gate, up and down banks, whole, on the expert card.
type peerBank struct {
	card *devTier
	r    [3]*resident // nil for a matrix the block does not have (ungated experts)
}

// attachExpertCard gives every streamed block with an expert cache a bank on
// the expert card. It runs at the end of a placement, after the caches are
// sized. A block that cannot be attached stays host-served; the first
// failure's reason is kept in LastErr.
func (g *GPU) attachExpertCard() {
	xd := g.xdev
	if xd == nil {
		return
	}
	g.mu.Lock()
	devs := append([]*devTier(nil), g.devs...)
	g.mu.Unlock()
	for _, d := range devs {
		d.attachPeers(xd)
	}
}

// attachPeers attaches the expert card's banks for the blocks this device
// holds, in block order.
func (g *devTier) attachPeers(xd *devTier) {
	if _, ok := g.dev.(backend.PeerCopier); !ok || g.dev.API() != xd.dev.API() {
		return
	}
	g.mu.Lock()
	var todo []*layer
	for _, l := range g.layers {
		if l != nil && l.stream != nil && l.stream.cached() && l.stream.peer == nil && l.stream.nExpert > 0 {
			todo = append(todo, l)
		}
	}
	g.mu.Unlock()
	for _, l := range todo {
		pb, err := g.peerBankFor(xd, l)
		if err != nil {
			g.mu.Lock()
			g.PeerDeclined++
			g.LastErr = fmt.Sprintf("block %d: no bank on the expert card: %v", l.stream.li, err)
			g.mu.Unlock()
			continue
		}
		g.mu.Lock()
		l.stream.peer = pb
		g.PeerBlocks++
		g.mu.Unlock()
	}
}

// peerBankFor makes block l's banks on xd and fills them from the host's
// sheets, expert by expert.
func (g *devTier) peerBankFor(xd *devTier, l *layer) (*peerBank, error) {
	st := l.stream
	pb := &peerBank{card: xd}
	cache := [3]*resident{l.gate, l.up, l.down}
	ws := [3]nn.Weight{st.w.Gate, st.w.Up, st.w.Down}
	slots := len(st.owner)
	if slots == 0 {
		return nil, fmt.Errorf("no expert cache")
	}
	for m := 0; m < 3; m++ {
		c := cache[m]
		if m == 0 && c == nil && ws[0].Packed == nil && ws[0].Data == nil {
			continue // ungated experts have no gate bank
		}
		if c == nil || !c.ok || ws[m].Packed == nil {
			pb.free()
			return nil, fmt.Errorf("matrix %d has no host side", m)
		}
		xd.mu.Lock()
		r, sh, ok := xd.residentCompact(c.t, c.nrows/slots, c.k, st.nExpert)
		xd.mu.Unlock()
		if !ok {
			pb.free()
			return nil, fmt.Errorf("no room for matrix %d (%d experts) on %s", m, st.nExpert, xd.dev.Name())
		}
		pb.r[m] = r
		if sh != st.sh[m] {
			pb.free()
			return nil, fmt.Errorf("matrix %d sheets are %v bytes, the cache's are %v", m, sh, st.sh[m])
		}
		if err := pb.fill(m, ws[m].Packed, st.nExpert, sh); err != nil {
			pb.free()
			return nil, err
		}
	}
	return pb, nil
}

// fill writes every expert's planes of matrix m from the host.
func (pb *peerBank) fill(m int, pk *nn.Packed, nExpert int, sh [3]int) error {
	dst := [3]backend.Buf{pb.r[m].qs, pb.r[m].d, pb.r[m].sc}
	for e := 0; e < nExpert; e++ {
		var planes [3][]byte
		release := func() {}
		if pk.Sheet != nil {
			sp, rel, err := pk.Sheet(e)
			release = rel
			if err != nil {
				release()
				return err
			}
			planes = [3][]byte{sp.QS, sp.D, sp.SC}
		} else {
			for pl, whole := range [3][]byte{pk.QS, pk.D, pk.SC} {
				if sh[pl] == 0 {
					continue
				}
				if (e+1)*sh[pl] > len(whole) {
					return fmt.Errorf("expert %d plane %d wants bytes %d..%d of %d", e, pl, e*sh[pl], (e+1)*sh[pl], len(whole))
				}
				planes[pl] = whole[e*sh[pl] : (e+1)*sh[pl]]
			}
		}
		for pl := 0; pl < 3; pl++ {
			if sh[pl] == 0 {
				continue
			}
			if len(planes[pl]) != sh[pl] {
				release()
				return fmt.Errorf("expert %d plane %d is %d bytes, want %d", e, pl, len(planes[pl]), sh[pl])
			}
			if err := dst[pl].WriteAt(e*sh[pl], planes[pl]); err != nil {
				release()
				return err
			}
		}
		release()
	}
	return nil
}

// free gives the banks back to the expert card.
func (pb *peerBank) free() {
	if pb == nil {
		return
	}
	xd := pb.card
	xd.mu.Lock()
	defer xd.mu.Unlock()
	for i, r := range pb.r {
		if r == nil {
			continue
		}
		n := r.bytes()
		for _, b := range [3]backend.Buf{r.qs, r.d, r.sc} {
			if b != nil {
				b.Free()
			}
		}
		xd.refund(n)
		pb.r[i] = nil
	}
}

// peerServes is the copier for matrix m's slots when the block has a peer bank
// for it and this device copies card to card, nil otherwise.
func (st *streamBank) peerServes(g *devTier, m int) backend.PeerCopier {
	if st.peer == nil || st.peer.r[m] == nil {
		return nil
	}
	pc, _ := g.dev.(backend.PeerCopier)
	return pc
}

// peerSheet copies expert e's plane pl of matrix m from the expert card into
// dst at slot, n bytes. It reports false (and copies nothing) when the block
// has no peer bank for the matrix.
func (st *streamBank) peerSheet(g *devTier, m, pl, e, slot, n int, dst backend.Buf) (bool, error) {
	pc := st.peerServes(g, m)
	if pc == nil {
		return false, nil
	}
	r := st.peer.r[m]
	src := [3]backend.Buf{r.qs, r.d, r.sc}[pl]
	e2 := e
	if peerFaulted("wrong") {
		// The violation: the neighbour's sheet.
		e2 = (e + 1) % st.nExpert
	}
	direct, err := pc.CopyPeer(dst, slot*n, st.peer.card.dev, src, e2*n, n)
	if err != nil {
		return false, err
	}
	g.PeerBytes += int64(n)
	if pl == 0 {
		g.PeerSheets++
	}
	if !direct {
		g.PeerStaged++
	}
	return true, nil
}

// peerAll reports whether every matrix the block has is served by its peer
// bank, in which case the host need not read the sheets' pages.
func (st *streamBank) peerAll(g *devTier) bool {
	if st.peer == nil {
		return false
	}
	for m := 0; m < 3; m++ {
		if st.peer.r[m] != nil && st.peerServes(g, m) == nil {
			return false
		}
	}
	return true
}
