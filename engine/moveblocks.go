package engine

import (
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/jitllm/jitllm/nn"
)

// BlockMove is what MoveBlocks did.
type BlockMove struct {
	// Moved is the blocks that changed device; Declined the ones the target
	// refused, which stay where they were with their history.
	Moved    []int
	Declined []int
	// KVBytes is the history the moved blocks carried, at the position the
	// move was made.
	KVBytes uint64
	// RecBytes is the recurrent summaries the moved linear blocks carried.
	RecBytes uint64
	// Path is how the history travelled, the slowest any block took: "peer"
	// card to card, "staged" through a host buffer the tier holds, "cpu"
	// back to the CPU device's pages and up again; "" when there was none.
	Path string
	// Copied is the bytes the tier copied between the devices (the history
	// in the devices' own page layout).
	Copied uint64
	// Head is set when the output projection followed the blocks.
	Head bool
	// Staged is the blocks prepared on the target while the sequence kept
	// generating (PrepareMove), and Prepare how long that took: time the
	// sequence did not wait.
	Staged  int
	Prepare time.Duration
	// Pause is the commit: what the sequence waited between two tokens.
	Pause time.Duration
}

// PendingMove is a move between devices being prepared while the instance keeps
// generating: the target takes each block's weights and builds its kernels
// on another goroutine (nn.BlockMover.StageBlock), and Commit, at a token
// boundary, copies the history and switches the route. Ready says the
// preparation is done, so a caller that commits only then never waits on it.
type PendingMove struct {
	s      *Instance
	lo, hi int
	to     string
	// blocks are the ones to move, plans their plans, staged which the
	// target prepared ahead.
	blocks []int
	plans  []nn.LayerPlan
	staged []bool
	done   chan struct{}
	prep   time.Duration
	closed bool
	// headStaged is the projection prepared on the target as well.
	headStaged bool
}

// MoveBlocks moves blocks [lo, hi) to the device named to, the CPU (nn.CPU)
// as any other, while the sequence keeps its place: each block's history (its
// KV pages, a linear block's recurrent summary) goes with it, so the next
// token reads exactly the rows it would have read where it was, and no row of
// the prompt is computed again. It is the relocation of the Scope's "move
// layers and pages between them", reachable by a caller rather than only
// through SetDevice, which moves everything to the CPU first.
//
// Between two cards it is PrepareMove and Commit at once: the target is
// prepared inside the call. A caller that can keep generating meanwhile
// prepares first and commits at a later token.
//
// Call it between tokens: like every placement call it must not overlap a
// Forward of this Instance. Blocks already on to are left alone, and a block the
// target refuses stays where it was and is reported. When no block of this
// Instance is left on the device holding the output projection, the projection
// follows them; a move to the CPU brings it to the CPU with them.
//
// A pinned block refuses the call before anything moves. Between two cards a
// range that does not hold a KV-sharing group whole is refused, since the
// group's readers attend to its source's history wherever it is; a group the
// move to or from the CPU splits ends where its source is (fixKVGroups).
func (s *Instance) MoveBlocks(lo, hi int, to string) (BlockMove, error) {
	defer s.settleHist()
	if lo < s.lo || hi > s.hi || lo >= hi {
		return BlockMove{}, fmt.Errorf("model: blocks %d..%d are outside this instance's %d..%d", lo, hi-1, s.lo, s.hi-1)
	}
	if s.pending != nil {
		return BlockMove{}, fmt.Errorf("model: a move of blocks %d..%d to %s is pending",
			s.pending.lo, s.pending.hi-1, s.pending.to)
	}
	if onCPU(to) {
		return s.moveOffCards(lo, hi)
	}
	if err := s.vacate(lo, hi); err != nil {
		return BlockMove{}, err
	}
	var cpu []int
	for li := lo; li < hi; li++ {
		if !s.devAt(li) {
			cpu = append(cpu, li)
			continue
		}
		if err := s.movable(li, lo, hi); err != nil {
			return BlockMove{}, err
		}
	}
	var mv BlockMove
	t0 := time.Now()
	if len(cpu) > 0 {
		placed, err := s.placeOnto(lo, hi, to)
		for _, li := range placed {
			mv.KVBytes += s.blockHistoryAt(li, s.pos)
			mv.RecBytes += s.recBytesOf(li)
		}
		mv.Moved = placed
		if err != nil {
			return mv, err
		}
	}
	placing := time.Since(t0)
	p, err := s.prepareMove(lo, hi, to, true)
	if err != nil {
		return mv, err
	}
	dm, err := p.Commit()
	mv.Moved = append(mv.Moved, dm.Moved...)
	slices.Sort(mv.Moved)
	mv.Declined = dm.Declined
	for _, li := range cpu {
		if !s.devAt(li) {
			mv.Declined = append(mv.Declined, li)
		}
	}
	slices.Sort(mv.Declined)
	mv.KVBytes += dm.KVBytes
	mv.RecBytes += dm.RecBytes
	mv.Path, mv.Copied, mv.Head = dm.Path, dm.Copied, dm.Head
	mv.Staged, mv.Prepare = dm.Staged, dm.Prepare
	mv.Pause = dm.Pause + placing
	return mv, err
}

// movable refuses a move of placed block li inside [lo, hi): a block sharing
// its history beyond the range, or one the placement pins where it is.
func (s *Instance) movable(li, lo, hi int) error {
	if glo, ghi := s.kvSpan(li); glo < lo || ghi > hi {
		return fmt.Errorf("model: block %d shares its history with blocks %d..%d, "+
			"which move together", li, glo, ghi-1)
	}
	if s.pinnedOnDev(li) {
		return fmt.Errorf("model: block %d is pinned where it is by the placement", li)
	}
	return nil
}

// PrepareMove starts a move of placed blocks [lo, hi) to the card named to
// (see MoveBlocks) and returns at once: the target prepares the blocks while
// the instance keeps generating on the source, and nothing the instance runs
// changes until Commit. Call it, Commit and Cancel between tokens; an instance
// holds one pending move at a time. A block on the CPU is refused: there is
// nothing on a card to keep generating from, and MoveBlocks places it.
func (s *Instance) PrepareMove(lo, hi int, to string) (*PendingMove, error) {
	if onCPU(to) {
		return nil, fmt.Errorf("model: a move to the CPU is not prepared ahead; MoveBlocks makes it")
	}
	return s.prepareMove(lo, hi, to, false)
}

// prepareMove is PrepareMove; with skipCPU the blocks of the range on the
// CPU, the ones the target refused when MoveBlocks offered them, are left out
// rather than refused.
func (s *Instance) prepareMove(lo, hi int, to string, skipCPU bool) (*PendingMove, error) {
	nd, ok := s.ld.(nn.NamedDevice)
	if !ok {
		return nil, fmt.Errorf("model: no device that places blocks by name is attached, " +
			"so there is no device to move blocks to")
	}
	if s.pending != nil {
		return nil, fmt.Errorf("model: a move of blocks %d..%d to %s is pending",
			s.pending.lo, s.pending.hi-1, s.pending.to)
	}
	if lo < s.lo || hi > s.hi || lo >= hi {
		return nil, fmt.Errorf("model: blocks %d..%d are outside this instance's %d..%d", lo, hi-1, s.lo, s.hi-1)
	}
	for li := lo; li < hi; li++ {
		if !s.devAt(li) {
			if skipCPU {
				continue
			}
			return nil, fmt.Errorf("model: block %d is on the CPU; a move prepared ahead "+
				"moves placed blocks, and MoveBlocks places one", li)
		}
		if err := s.movable(li, lo, hi); err != nil {
			return nil, err
		}
	}
	if err := s.vacate(lo, hi); err != nil {
		return nil, err
	}
	p := &PendingMove{s: s, lo: lo, hi: hi, to: to, done: make(chan struct{})}
	for li := lo; li < hi; li++ {
		if !s.devAt(li) {
			continue
		}
		if at, _ := nd.DeviceOf(li); at == nn.DeviceName(to) {
			continue
		}
		p.blocks = append(p.blocks, li)
		p.plans = append(p.plans, *s.planFor(li))
	}
	p.staged = make([]bool, len(p.blocks))
	s.pending = p
	mover, ok := s.ld.(nn.BlockMover)
	// Staging reads the block's page and hands the tier its weights while the
	// Instance runs: the instance's own vision and encoder segments, and a mixture's
	// expert placement, are decided by the offer path, so they are prepared at
	// the commit instead.
	if !ok || s.opt.hostMoves || s.c.MoE() || len(p.blocks) == 0 {
		close(p.done)
		return p, nil
	}
	// The projection is prepared too when the move leaves none of the instance's
	// blocks beside it (Commit then moves it).
	var head *nn.Head
	if was, on := nd.HeadDeviceName(); on && s.head != nil && was != nn.DeviceName(to) {
		follows := true
		for li := s.lo; li < s.hi; li++ {
			if at, ok := nd.DeviceOf(li); ok && s.devAt(li) && at == was && (li < lo || li >= hi) {
				follows = false
			}
		}
		if follows {
			head = s.newHead()
		}
	}
	// Room for the history at the commit: the tokens the preparation runs
	// beside add to it.
	pos := s.pos + 256
	go func() {
		defer close(p.done)
		t0 := time.Now()
		all := true
		for i, li := range p.blocks {
			w, err := s.blockWeights(li)
			if err != nil {
				all = false
				break
			}
			p.staged[i] = mover.StageBlock(li, to, &p.plans[i], &w, pos)
			all = all && p.staged[i]
		}
		if all && head != nil {
			p.headStaged = mover.StageHead(to, head)
		}
		p.prep = time.Since(t0)
	}()
	return p, nil
}

// Ready reports that the target has finished preparing, so Commit will not
// wait for it.
func (p *PendingMove) Ready() bool {
	select {
	case <-p.done:
		return true
	default:
		return false
	}
}

// Cancel abandons a move that was not committed, freeing what the target
// prepared for it.
func (p *PendingMove) Cancel() {
	if p.closed {
		return
	}
	<-p.done
	p.closed = true
	p.s.pending = nil
	if mover, ok := p.s.ld.(nn.BlockMover); ok {
		mover.Unstage(p.lo, p.hi)
	}
}

// Commit makes the move, between two tokens: each block's history goes to
// the target and the route switches. It waits for the preparation if it is
// still running.
func (p *PendingMove) Commit() (mv BlockMove, err error) {
	s := p.s
	defer s.settleHist()
	if p.closed {
		return mv, fmt.Errorf("model: the move was already committed or cancelled")
	}
	<-p.done
	p.closed = true
	s.pending = nil
	t0 := time.Now()
	mv.Prepare = p.prep
	for _, st := range p.staged {
		if st {
			mv.Staged++
		}
	}
	defer func() { mv.Pause = time.Since(t0) }()
	nd := s.ld.(nn.NamedDevice)
	headWas, headOn := nd.HeadDeviceName()
	headOn = headOn && s.head != nil
	mover, direct := s.ld.(nn.BlockMover)
	direct = direct && !s.opt.hostMoves
	for i, li := range p.blocks {
		if at, _ := nd.DeviceOf(li); at == nn.DeviceName(p.to) {
			continue
		}
		path, copied, moved := "", uint64(0), false
		err = nn.ErrNotDirect
		if direct {
			path, copied, moved, err = s.moveBlockDirect(mover, li, p.to, &p.plans[i])
		}
		if errors.Is(err, nn.ErrNotDirect) {
			path = nn.CPU
			moved, err = s.moveThroughMemory(nd, li, p.to)
		}
		if err != nil {
			if mover != nil {
				mover.Unstage(p.lo, p.hi)
			}
			return mv, err
		}
		if !moved {
			mv.Declined = append(mv.Declined, li)
			continue
		}
		mv.Moved = append(mv.Moved, li)
		mv.KVBytes += s.blockHistoryAt(li, s.pos)
		mv.RecBytes += s.recBytesOf(li)
		mv.Copied += copied
		mv.Path = slowerPath(mv.Path, path)
	}
	if mover != nil {
		// A declined block's staged copy, if the target kept one, and what
		// the moved blocks left on their sources.
		mover.Unstage(p.lo, p.hi)
		defer mover.RetireMoved()
	}
	// The projection follows once nothing of this instance is left beside it: a
	// head alone on a card costs a crossing a token and holds the card.
	if headOn && headWas != nn.DeviceName(p.to) && len(mv.Moved) > 0 && !s.anyBlockOn(nd, headWas) {
		if mv.Head, err = nd.PrepHeadOn(p.to, s.newHead()); err != nil {
			return mv, err
		}
	}
	return mv, nil
}

// slowerPath is the slower of two history paths, "" being none.
func slowerPath(a, b string) string {
	rank := map[string]int{"": 3, "peer": 2, "staged": 1, nn.CPU: 0}
	if rank[b] < rank[a] {
		return b
	}
	return a
}

// moveBlockDirect moves block li through the tier: the history is copied from
// the source's pages to the target's and never passes through host memory. A linear block's
// summary, a few kilobytes, is read back and goes up as it does on every other
// move. nn.ErrNotDirect changed nothing.
func (s *Instance) moveBlockDirect(mover nn.BlockMover, li int, to string, plan *nn.LayerPlan) (string, uint64, bool, error) {
	if !s.migrateRec(li, false) {
		return "", 0, false, fmt.Errorf("model: block %d's recurrent summary would not come home", li)
	}
	res, err := mover.MoveBlock(li, to, plan, nil, s.pos)
	if errors.Is(err, nn.ErrNotStaged) {
		var w nn.LayerWeights
		if w, err = s.blockWeights(li); err != nil {
			return "", 0, false, err
		}
		res, err = mover.MoveBlock(li, to, plan, &w, s.pos)
	}
	if err != nil || !res.Moved {
		return "", 0, false, err
	}
	if !s.migrateRec(li, true) {
		s.ld.ReleaseLayers(li, li+1)
		s.unmarkOnDev(li)
		return "", 0, false, fmt.Errorf("model: block %d reached %s but its recurrent summary did not; "+
			"it runs on the host", li, to)
	}
	s.releasePlacedPage(s.ld, li)
	return res.Path, res.Bytes, true, nil
}

// moveThroughMemory moves block li to the device named to through this instance's
// own cache: the history is read back to the session's cache (MigrateKV copies, so the source keeps
// its rows until the target has them), the target takes the block, which
// frees it on the source, and the history goes up to the target. False means
// the target refused the block, which then stays where it was.
func (s *Instance) moveThroughMemory(nd nn.NamedDevice, li int, to string) (bool, error) {
	if !s.migrateKV(li, s.pos, false) || !s.migrateRec(li, false) {
		return false, fmt.Errorf("model: block %d's history would not come home from its device", li)
	}
	plan, w, err := s.blockOffer(li)
	if err != nil {
		return false, err
	}
	took, err := nd.PrepLayerOn(to, li, &plan, &w)
	if err != nil || !took {
		return false, err
	}
	if !s.migrateKV(li, s.pos, true) || !s.migrateRec(li, true) {
		// The weights are on the target with no history; the history is in host memory,
		// so the block runs there instead of attending over empty pages.
		s.ld.ReleaseLayers(li, li+1)
		s.unmarkOnDev(li)
		return false, fmt.Errorf("model: block %d reached %s but its history did not; "+
			"it runs on the host", li, to)
	}
	s.releasePlacedPage(s.ld, li)
	return true, nil
}

// anyBlockOn reports whether a block of this instance is on the named device.
func (s *Instance) anyBlockOn(nd nn.NamedDevice, name string) bool {
	for li := s.lo; li < s.hi; li++ {
		if !s.devAt(li) {
			continue
		}
		if at, ok := nd.DeviceOf(li); ok && at == name {
			return true
		}
	}
	return false
}

// DeviceOf is the name of the device holding block li, and whether a device
// holds it.
func (s *Instance) DeviceOf(li int) (string, bool) {
	if !s.devAt(li) {
		return "", false
	}
	if nd, ok := s.named(); ok {
		return nd.DeviceOf(li)
	}
	return "", true
}

// named is the device that names where the instance's blocks are: its own
// session's view, which every placement runs through (Place) -- another
// instance on the tier may run a block on another card, each view routed to
// its own copy -- else the bound session's, or the device the instance is
// placed on where neither has one.
func (s *Instance) named() (nn.NamedDevice, bool) {
	if s.own != nil {
		if nd, ok := s.own.ld.(nn.NamedDevice); ok {
			return nd, true
		}
	}
	if nd, ok := s.ld.(nn.NamedDevice); ok {
		return nd, true
	}
	nd, ok := s.ldCand.(nn.NamedDevice)
	return nd, ok
}

// blockHistoryAt is historyAt for block li alone.
func (s *Instance) blockHistoryAt(li, pos int) uint64 {
	if li < s.kv.lo || li >= len(s.kv.layers) {
		return 0
	}
	pg := &s.kv.layers[li]
	if pg.p == 0 {
		return 0
	}
	n := pos
	if pg.win > 0 && n > pg.win+pg.p {
		n = pg.win + pg.p
	}
	per := uint64(pg.pp) * 4 / uint64(pg.p) // a key's and a value's row
	if pg.latent {
		return uint64(n) * per
	}
	return 2 * uint64(n) * per
}

// recBytesOf is block li's recurrent summary in bytes, every row's.
func (s *Instance) recBytesOf(li int) uint64 {
	if !s.keepsRec(li) {
		return 0
	}
	nc, ns := s.recLens()
	return uint64(nc+ns) * 4
}

// BlockHistory is what block li's history costs, so a caller can say how much
// of it a move carries at any position: a recurrent summary's bytes (fixed)
// and the bytes each kept position adds, which a windowed block stops adding
// once `window` positions are held (zero keeps every position). It is
// blockHistoryAt's arithmetic in its two parts.
func (s *Instance) BlockHistory(li int) (perPos uint64, window int, state uint64) {
	state = s.recBytesOf(li)
	if li < s.kv.lo || li >= len(s.kv.layers) {
		return 0, 0, state
	}
	pg := &s.kv.layers[li]
	if pg.p == 0 {
		return 0, 0, state
	}
	perPos = uint64(pg.pp) * 4 / uint64(pg.p)
	if !pg.latent {
		perPos *= 2
	}
	if pg.win > 0 {
		window = pg.win + pg.p
	}
	return perPos, window, state
}
