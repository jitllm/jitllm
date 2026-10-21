package kernels

import (
	"os"
	"sync"
	"sync/atomic"
	"unsafe"
)

// Arena is the host-resident home of packed weights, in the device layout.
//
// Packing costs roughly three times an upload,
// so a block evicted from a card and wanted again is a DMA if its packed bytes
// still exist on the host and a repack plus DMA if they do not. The arena keeps
// the pack. Where the device can unpack a format itself from the raw bytes
// (kernels.Unpack), tier's resident() asks that first; the arena serves the
// formats it cannot, and a hit uploads packed bytes with no kernel at all.
//
// On a unified device the pack is the device buffer (importPacked wraps its
// pointer), so resident() asks importing() before either.
//
// It is a second copy of a block's bytes, and the device layout is wider than
// the file's for some formats, so retention is budgeted and defaults to zero;
// see NewArena. The bytes are meaningless to jit_old/cpu (transposed and
// normalised to value = scale*q - bias).
//
// It does not know what a transformer is: an entry is one tensor, keyed on the
// bytes it was packed from, and `block` is an opaque tag the caller uses to
// release entries together.
//
// The zero Arena is not usable; a nil *Arena is, and every method on one is the
// "off" behaviour, so a caller with no arena and one with no budget take the
// same code path.
type Arena struct {
	mu    sync.Mutex
	limit uint64
	used  uint64
	ents  map[akey]*Packed
	// loose are entries Forget unkeyed that something may still read: a
	// device importing them, or a block whose release frees them. They stay
	// charged until then.
	loose []*Packed
	// packing is every pack running outside the lock, by its source address,
	// so Forget can spoil one whose bytes it covers.
	packing []*inflight

	packs    atomic.Int64
	hits     atomic.Int64
	declines atomic.Int64
}

// akey identifies one packed tensor by the bytes it came from.
//
// The shape is part of the key: an expert bank and its expert 0 begin at the
// same byte, and handing one's packed arrays to the other is an out-of-bounds
// device read that does not fault.
type akey struct {
	p        *byte
	nrows, k int
	t        Quant
}

// Packed is one tensor in the device layout, living in host memory that a
// device can be uploaded from more than once.
//
// QS, D and SC are views into one aligned region, each starting on a
// HostAlign() boundary and padded to a whole multiple of it, so each can be
// the target of a host-pointer import as well as of a copy (both drivers
// require the length as well as the address; see backend.HostImport). They are
// the same three arrays PackWeights returns, with the same contents to the bit.
type Packed struct {
	QS, D, SC []uint32
	T         Quant
	Rows, K   int
	// Block is the caller's tag, or -1. The arena attaches no meaning to it
	// beyond Release grouping.
	Block int

	mem []byte // the whole aligned region
	own bool   // mem is a mapping and must be unmapped, not left to the GC
	n   uint64 // the region's size, as charged against the budget
	// imports counts the device buffers aliasing this region (Import), and
	// loose says Forget has unkeyed it. Guarded by the arena's mu.
	imports int
	loose   bool
}

// Bytes is what this entry costs the arena, including its alignment slack.
func (p *Packed) Bytes() uint64 {
	if p == nil {
		return 0
	}
	return p.n
}

// HostAlign is the alignment every packed array starts on, and the multiple its
// length is padded to.
//
// A copy works at any alignment; an import does not: VK_EXT_external_memory_host
// wants minImportedHostPointerAlignment (0x1000 here) and Metal's
// newBufferWithBytesNoCopy: wants whole pages. It is the larger of the two
// because the second and third arrays start at offsets inside the region, and
// Apple Silicon's page is 16 KiB: a constant 4096 made Metal refuse two arrays
// of three and silently fall back to copying.
func HostAlign() uint64 {
	if a := uint64(os.Getpagesize()); a > hostAlignFloor {
		return a
	}
	return hostAlignFloor
}

// hostAlignFloor is minImportedHostPointerAlignment as the Vulkan devices here
// report it; a host with smaller pages still lays its arrays out on it.
const hostAlignFloor = 4096

func alignUp(n uint64) uint64 { a := HostAlign(); return (n + a - 1) &^ (a - 1) }

// NewArena is a packed arena that may retain `retain` bytes.
//
// retain == 0 means retain nothing, and it is the default. That is the
// opposite of tier.Pool's convention (0 means unbounded) on purpose: an
// unbounded arena is a second copy of the whole model in host RAM, so a
// forgotten limit must mean "off". A model that fits pays a nil map lookup: its
// blocks are uploaded once and never paged in.
func NewArena(retain uint64) *Arena {
	return &Arena{limit: retain, ents: map[akey]*Packed{}}
}

// SetLimit changes the retention budget. Raising it admits new entries; lowering
// it does NOT evict what is already held -- an entry that is out is a dangling
// pointer to whoever holds it (see Release), so shrinking only stops the arena
// growing and the caller drops what it no longer wants by block.
func (a *Arena) SetLimit(n uint64) {
	if a == nil {
		return
	}
	a.mu.Lock()
	a.limit = n
	a.mu.Unlock()
}

// Headroom is how many more bytes this arena will accept before it starts
// declining. Zero when it is off, and zero when it is full.
//
// It prices a block on a unified device, where the arena is what pays: a tensor
// it takes costs the device nothing (the device wraps these bytes), and one it
// declines costs a full copy. A caller must know which before packing.
func (a *Arena) Headroom() uint64 {
	if a == nil {
		return 0
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.limit <= a.used {
		return 0
	}
	return a.limit - a.used
}

func (a *Arena) Limit() uint64 {
	if a == nil {
		return 0
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.limit
}

// Get is the packed form of this tensor if the arena already holds it.
func (a *Arena) Get(t Quant, src []byte, nrows, k int) *Packed {
	if a == nil || len(src) == 0 {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.ents[akey{p: &src[0], nrows: nrows, k: k, t: t}]
}

// Pack returns this tensor in the device layout, packing it the first time and
// handing back the same host bytes every time after.
//
// nil, nil means the arena is not holding this tensor (it is off or full) and
// the caller must pack it itself and own the result. That is not an error.
//
// The pack runs outside the lock so it can overlap a token (see tier.Prewarm).
// Two callers racing on one tensor both pack; the loser frees its region and
// returns the winner's, which is byte-identical. The loser's work is counted in
// Packs, so the counter does not hide duplicate work.
func (a *Arena) Pack(block int, t Quant, src []byte, nrows, k int) (*Packed, error) {
	if a == nil || len(src) == 0 {
		return nil, nil
	}
	key := akey{p: &src[0], nrows: nrows, k: k, t: t}
	a.mu.Lock()
	if p := a.ents[key]; p != nil {
		a.mu.Unlock()
		a.hits.Add(1)
		return p, nil
	}
	if a.limit == 0 {
		a.mu.Unlock()
		a.declines.Add(1)
		return nil, nil
	}
	nq, nd, nsc, err := PackedWords(t, nrows, k)
	if err != nil {
		a.mu.Unlock()
		return nil, err
	}
	need := regionBytes(nq, nd, nsc)
	if a.used+need > a.limit {
		a.mu.Unlock()
		a.declines.Add(1)
		return nil, nil
	}
	// Reserved before the pack, so two concurrent packs cannot both decide they
	// fit in the same last megabyte.
	a.used += need
	fl := &inflight{at: uintptr(unsafe.Pointer(key.p))}
	a.packing = append(a.packing, fl)
	a.mu.Unlock()

	p, err := newPacked(t, nrows, k, block, nq, nd, nsc, need)
	if err == nil {
		err = PackWeightsInto(t, src, nrows, k, p.QS, p.D, p.SC)
		a.packs.Add(1)
	}
	if err != nil {
		if p != nil {
			p.free()
		}
		a.mu.Lock()
		a.used -= need
		a.done(fl)
		a.mu.Unlock()
		return nil, err
	}

	a.mu.Lock()
	a.done(fl)
	// Forgotten while it packed: the bytes it read may be either tensor's, so
	// it is not kept, and the caller packs for itself.
	if fl.spoiled {
		a.used -= need
		a.mu.Unlock()
		p.free()
		a.declines.Add(1)
		return nil, nil
	}
	if q := a.ents[key]; q != nil { // lost the race; both answers are the same bytes
		a.used -= need
		a.mu.Unlock()
		p.free()
		a.hits.Add(1)
		return q, nil
	}
	a.ents[key] = p
	a.mu.Unlock()

	// No page-cache eviction here: dropping a source page is jlm's job
	// (File.DropPage), decided by the page budget.
	return p, nil
}

// Import records that a device buffer aliases p's region, and Unimport
// that one no longer does. An unkeyed entry (Forget) is freed when the last
// one lets go: nothing will ever ask for it by its bytes again.
func (a *Arena) Import(p *Packed) {
	if a == nil || p == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	p.imports++
}

func (a *Arena) Unimport(p *Packed) {
	if a == nil || p == nil {
		return
	}
	a.mu.Lock()
	if p.imports > 0 {
		p.imports--
	}
	gone := p.loose && p.imports == 0
	if gone {
		a.unloose(p)
	}
	a.mu.Unlock()
	if gone {
		p.free()
	}
}

// Forget unkeys every entry packed from bytes inside [lo, hi): the bytes there
// are about to be another tensor's, and a Pack or Get of whatever lands at the
// same address must not be handed this one. An entry no device imports is
// freed at once (a copy uploaded from it is the device's own); one a device
// imports waits for its Unimport, or for Release of its block.
func (a *Arena) Forget(lo, hi uintptr) {
	if a == nil {
		return
	}
	a.mu.Lock()
	for _, fl := range a.packing {
		if fl.at >= lo && fl.at < hi {
			fl.spoiled = true
		}
	}
	var gone []*Packed
	for k, p := range a.ents {
		if at := uintptr(unsafe.Pointer(k.p)); at < lo || at >= hi {
			continue
		}
		delete(a.ents, k)
		if p.imports == 0 {
			a.used -= min(a.used, p.n)
			gone = append(gone, p)
			continue
		}
		p.loose = true
		a.loose = append(a.loose, p)
	}
	a.mu.Unlock()
	for _, p := range gone {
		p.free()
	}
}

// inflight is one pack running outside the lock: where its source bytes start,
// and whether Forget covered them meanwhile.
type inflight struct {
	at      uintptr
	spoiled bool
}

// done takes fl off the in-flight list. Callers hold a.mu.
func (a *Arena) done(fl *inflight) {
	for i, f := range a.packing {
		if f == fl {
			a.packing = append(a.packing[:i], a.packing[i+1:]...)
			return
		}
	}
}

// unloose takes p off the loose list and refunds it. Callers hold a.mu.
func (a *Arena) unloose(p *Packed) {
	for i, q := range a.loose {
		if q == p {
			a.loose = append(a.loose[:i], a.loose[i+1:]...)
			break
		}
	}
	p.loose = false
	a.used -= min(a.used, p.n)
}

// Release drops every entry tagged with this block and returns the bytes given
// back. A block of -1 releases the untagged entries.
//
// It unmaps, so a *Packed handed out earlier becomes a dangling pointer and
// touching it segfaults. The caller must already have dropped the device
// buffers uploaded from these bytes, which a page-out does first.
func (a *Arena) Release(block int) uint64 {
	if a == nil {
		return 0
	}
	a.mu.Lock()
	var freed uint64
	var gone []*Packed
	for k, p := range a.ents {
		if p.Block == block {
			delete(a.ents, k)
			freed += p.n
			gone = append(gone, p)
		}
	}
	// An unkeyed entry of this block is released with it, as it would have
	// been keyed.
	kept := a.loose[:0]
	for _, p := range a.loose {
		if p.Block == block {
			p.loose = false
			freed += p.n
			gone = append(gone, p)
			continue
		}
		kept = append(kept, p)
	}
	a.loose = kept
	if a.used >= freed {
		a.used -= freed
	} else {
		a.used = 0
	}
	a.mu.Unlock()
	for _, p := range gone {
		p.free()
	}
	return freed
}

// Close drops every entry. See Release for what that does to handles.
func (a *Arena) Close() {
	if a == nil {
		return
	}
	a.mu.Lock()
	gone := make([]*Packed, 0, len(a.ents)+len(a.loose))
	for k, p := range a.ents {
		delete(a.ents, k)
		gone = append(gone, p)
	}
	for _, p := range a.loose {
		p.loose = false
		gone = append(gone, p)
	}
	a.loose = nil
	a.used = 0
	a.mu.Unlock()
	for _, p := range gone {
		p.free()
	}
}

// ArenaStats is what the arena has done. It is reported once per tier, not per
// device: two devices paging one model share its entries.
type ArenaStats struct {
	// Packs is PackWeights calls the arena made; it must not grow with the
	// number of page-ins.
	Packs int64
	// Hits is page-ins served from bytes that were already packed.
	Hits int64
	// Declines is packs the arena did not take, because it is off or full.
	// The caller packed those itself and counts them in tier.Stats.Packs, so
	// "the pack ran once" is checkable as a sum.
	Declines int64
	// Bytes and Entries are what is retained; Loose is how many of
	// the retained entries Forget unkeyed and a reader still holds.
	Bytes   uint64
	Entries int
	Loose   int
	Limit   uint64
}

func (a *Arena) Stats() ArenaStats {
	if a == nil {
		return ArenaStats{}
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return ArenaStats{
		Packs: a.packs.Load(), Hits: a.hits.Load(), Declines: a.declines.Load(),
		Bytes: a.used, Entries: len(a.ents), Loose: len(a.loose), Limit: a.limit,
	}
}

// regionBytes is what one tensor occupies in the arena: the three arrays, each
// rounded up to HostAlign() so each begins and ends on an importable boundary.
// The slack is at most three pages per tensor.
func regionBytes(nq, nd, nsc int) uint64 {
	n := alignUp(uint64(nq) * 4)
	n += alignUp(uint64(nd) * 4)
	n += alignUp(uint64(nsc) * 4)
	return n
}

func newPacked(t Quant, nrows, k, block, nq, nd, nsc int, n uint64) (*Packed, error) {
	mem, own, err := mapAligned(int(n))
	if err != nil {
		return nil, err
	}
	p := &Packed{T: t, Rows: nrows, K: k, Block: block, mem: mem, own: own, n: n}
	off := uint64(0)
	p.QS, off = words(mem, off, nq)
	p.D, off = words(mem, off, nd)
	p.SC, _ = words(mem, off, nsc)
	return p, nil
}

// words carves one HostAlign()-aligned uint32 view out of the region and returns
// the offset of the next one.
func words(mem []byte, off uint64, n int) ([]uint32, uint64) {
	if n == 0 {
		return nil, off
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&mem[off])), n), off + alignUp(uint64(n)*4)
}

func (p *Packed) free() {
	if p == nil {
		return
	}
	if p.own {
		unmapAligned(p.mem)
	}
	p.QS, p.D, p.SC, p.mem = nil, nil, nil, nil
}

// Addr is the host address of a packed array as an integer, for alignment
// checks.
func Addr(v []uint32) uintptr {
	if len(v) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&v[0]))
}

// Ptr is the same address as a pointer, which is what an importing backend is
// handed. It is separate from Addr so the address never round-trips through a
// uintptr, the pattern `go vet` flags.
func Ptr(v []uint32) unsafe.Pointer {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Pointer(&v[0])
}

// PaddedBytes is how many bytes one packed array of n words occupies in the
// arena: its own bytes, rounded up to a whole HostAlign(). A caller handing an
// array to backend.HostImport must pass this padded length (both drivers need
// a multiple of the alignment); the padding is inside the arena's own region.
func PaddedBytes(n int) uint64 {
	if n <= 0 {
		return 0
	}
	return alignUp(uint64(n) * 4)
}
