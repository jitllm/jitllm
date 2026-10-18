package model

import (
	"fmt"
	"io"
	"sync"

	"github.com/jitllm/jitllm/engine/nn"
)

// Prefix sharing: sessions whose prompts share a prefix hold its KV pages
// once.
//
// A sealed page (one the sequence has written past) never changes, and under
// a namespace it is named by every token up to its end (kvCache.pageKey), so
// two sessions whose prompts agree that far would compute the same bytes.
// SharedStore keeps one copy of each such page and hands every session that
// reaches it the SAME memory: a session's page table (kvPages.k and .v) points
// at it, refcounted, instead of at a page of its own. Nothing else about the
// cache changes -- the attention kernels read a page wherever it is.
//
// The rules that keep it correct:
//
//   - Only sealed, whole pages are shared. The page a session is writing into
//     is its own; the first page where two prompts diverge is computed by
//     each (its key covers the diverging tokens, so it never matches).
//   - A shared page is never written in place. Every write path (the token's
//     append, a migration scattering rows home) copies a shared page into a
//     private one first and drops its reference: copy-on-write.
//   - A reference is dropped wherever the session lets a page go (eviction, a
//     restore's truncation, a window's release, Close), never handed to the
//     session's page pool, which would recycle memory others still read.
//   - A shared page does not move to a device for one of its holders: the
//     block stays on the host (State.migrateKV refuses, counted by
//     KVSharedRefusals). A device page shared between sessions is not
//     implemented; a session with blocks on a device does not share
//     (State.ShareKV).
//   - A recurrent layer's summary is not paged by position: it is kept as a
//     restore point beside the pages (recpoint.go), a copy of a state that
//     moves on, through the store's Get and Set. A batch is refused: its
//     pages hold every row.
//   - A page a prompt ends inside is held too, named by its fill, and every
//     page answers to the chunks of history it holds (alias): a prompt that
//     shares only part of a page aliases it and copies it on its first write.
//
// SharedStore is also an ordinary KVStore: Get and Set copy, so a session that
// does not share (SetKVStore alone) reads and writes it as it would a
// MemStore, and a sharing session finds the pages it published.
type SharedStore struct {
	mu    sync.Mutex
	pages map[kvStoreKey]*sharedPage
	held  map[*float32]*sharedPage
	bytes uint64
	// order is the held pages oldest first, for Trim; a forgotten one is
	// passed over there.
	order []*sharedPage
}

// sharedPage is one page's single copy: refs sessions alias it. key is the
// name it was published under; aka are the other names it answers to (the
// shorter prefixes it holds, alias), forgotten with it.
type sharedPage struct {
	key  kvStoreKey
	aka  []kvStoreKey
	b    []float32
	refs int
	// gone means Drop or Prune took it out of the index: no session can
	// acquire it again, and it is forgotten once the last holder lets go.
	gone bool
}

// lenReader is a page whose size is known before it is read (bytes.Reader).
type lenReader interface {
	io.Reader
	Len() int
}

// NewSharedStore returns an empty SharedStore.
func NewSharedStore() *SharedStore {
	return &SharedStore{pages: map[kvStoreKey]*sharedPage{}, held: map[*float32]*sharedPage{}}
}

// Get writes a copy of the held page into page, or returns ErrNoPage.
func (m *SharedStore) Get(cacheId string, layer, index int, page io.Writer) error {
	m.mu.Lock()
	e := m.pages[kvStoreKey{cacheId, layer, index}]
	m.mu.Unlock()
	if e == nil {
		return ErrNoPage
	}
	// The bytes of a held page never change (no session writes one in place),
	// so they are read outside the lock.
	_, err := page.Write(kvBytesOf(e.b))
	return err
}

// Set holds a copy of the page under its key, replacing one no session holds.
// A key a session aliases keeps its page: those bytes are what it reads.
func (m *SharedStore) Set(cacheId string, layer, index int, page io.Reader) error {
	// A page offered again at its size and held by no session is read in
	// place: a hybrid's restore points, re-stored every prompt, allocate
	// nothing.
	if r, ok := page.(lenReader); ok {
		m.mu.Lock()
		e := m.pages[kvStoreKey{cacheId, layer, index}]
		if e != nil && e.refs == 0 && len(e.b)*4 == r.Len() && e.key == (kvStoreKey{cacheId, layer, index}) {
			defer m.mu.Unlock()
			_, err := io.ReadFull(r, kvBytesOf(e.b))
			return err
		}
		m.mu.Unlock()
	}
	b, err := io.ReadAll(page)
	if err != nil {
		return err
	}
	if len(b)%4 != 0 {
		return fmt.Errorf("model: a kv page of %d bytes is not float32 slots", len(b))
	}
	f := make([]float32, len(b)/4)
	copy(kvBytesOf(f), b)
	m.mu.Lock()
	defer m.mu.Unlock()
	k := kvStoreKey{cacheId, layer, index}
	if e := m.pages[k]; e != nil {
		if e.refs > 0 {
			return nil
		}
		m.forget(e)
	}
	m.add(k, f)
	return nil
}

// Drop forgets the pages held under one cache id. A page a session still
// aliases stays with it until it lets go.
func (m *SharedStore) Drop(cacheId string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for k, e := range m.pages {
		if k.cache == cacheId {
			m.retire(e)
		}
	}
	return nil
}

// Prune forgets every page no session holds, which is what bounds the store:
// a page is kept for the next session to find until Prune (or Drop) lets it
// go.
func (m *SharedStore) Prune() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, e := range m.pages {
		if e.refs == 0 {
			m.retire(e)
		}
	}
}

// Bytes is what the store holds: every page once, however many sessions
// alias it.
func (m *SharedStore) Bytes() uint64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.bytes
}

// Refs is how many session references page `index` of `layer` under cacheId
// has, -1 when the store does not hold it.
func (m *SharedStore) Refs(cacheId string, layer, index int) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	if e := m.pages[kvStoreKey{cacheId, layer, index}]; e != nil {
		return e.refs
	}
	return -1
}

// publish hands the store a sealed page a session computed. When the store
// already holds that page, the session gets the held copy (and keeps
// ownership of b, which it gives back to its pool); otherwise b becomes the
// held copy. Either way the session holds one reference to what is returned.
func (m *SharedStore) publish(k kvStoreKey, b []float32) []float32 {
	m.mu.Lock()
	defer m.mu.Unlock()
	if e := m.pages[k]; e != nil {
		if len(e.b) == len(b) {
			e.refs++
			return e.b
		}
		// Another shape under one key is a stale page from another geometry
		// (the key digests it, so only a collision): the new one replaces it.
		m.retire(e)
	}
	e := m.add(k, b)
	e.refs++
	return b
}

// acquire is the held page under k with one more reference, or nil.
func (m *SharedStore) acquire(k kvStoreKey, n int) []float32 {
	m.mu.Lock()
	defer m.mu.Unlock()
	e := m.pages[k]
	if e == nil || len(e.b) != n {
		return nil
	}
	e.refs++
	return e.b
}

// release drops one session reference to the held page b.
func (m *SharedStore) release(b []float32) {
	if len(b) == 0 {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	e := m.held[&b[0]]
	if e == nil {
		panic("model: a shared kv page released that the store does not hold")
	}
	e.refs--
	if e.refs < 0 {
		panic("model: a shared kv page released more often than it was taken")
	}
	if e.refs == 0 && e.gone {
		m.forget(e)
	}
}

// alias makes the held page b answer to k as well: the page a prefix ends
// inside, named by that shorter prefix (a chunk or a fill key, kvpage.go),
// so a prompt sharing only that much aliases it and copies it on its first
// write (kvPages.cow). A name already held is left as it is.
func (m *SharedStore) alias(k kvStoreKey, b []float32) {
	if len(b) == 0 {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	e := m.held[&b[0]]
	if e == nil || e.gone {
		return
	}
	if _, ok := m.pages[k]; ok {
		return
	}
	m.pages[k] = e
	e.aka = append(e.aka, k)
}

// Trim forgets pages no session holds, oldest first, until the store holds
// at most limit bytes: a bound for a store a server keeps for a model.
func (m *SharedStore) Trim(limit uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	keep := m.order[:0]
	for _, e := range m.order {
		if m.held[&e.b[0]] != e {
			continue // forgotten already
		}
		if e.refs == 0 && m.bytes > limit {
			m.retire(e)
			continue
		}
		keep = append(keep, e)
	}
	clear(m.order[len(keep):])
	m.order = keep
}

// add indexes a new held page. Callers hold m.mu.
func (m *SharedStore) add(k kvStoreKey, b []float32) *sharedPage {
	e := &sharedPage{key: k, b: b}
	m.pages[k] = e
	m.held[&b[0]] = e
	m.order = append(m.order, e)
	m.bytes += 4 * uint64(len(b))
	return e
}

// retire takes e out of the index, forgetting it now if nobody holds it.
// Callers hold m.mu.
func (m *SharedStore) retire(e *sharedPage) {
	m.unindex(e)
	e.gone = true
	if e.refs == 0 {
		m.forget(e)
	}
}

// unindex takes every name of e out of the index. Callers hold m.mu.
func (m *SharedStore) unindex(e *sharedPage) {
	for _, k := range append(e.aka, e.key) {
		if m.pages[k] == e {
			delete(m.pages, k)
		}
	}
	e.aka = e.aka[:0]
}

// forget drops e entirely. Callers hold m.mu.
func (m *SharedStore) forget(e *sharedPage) {
	m.unindex(e)
	if m.held[&e.b[0]] == e {
		delete(m.held, &e.b[0])
		m.bytes -= 4 * uint64(len(e.b))
	}
}

// ShareKV makes this session hold the sealed pages of its history once with
// every other session sharing st: what it seals is published to st, and a
// prefix PrefillCached finds there is aliased, not copied (SharedStore has
// the rules). It sets st as the session's store; SetCacheKey must name the
// namespace first, since a page is named by its tokens only inside one.
//
// Refused by name: a batch (its pages hold every row's slots), DeepSeek V4 (its
// compressed entries are a second history keyed by entry), and a session with
// blocks on a device (a device page shared between sessions is not
// implemented).
// UnshareKV ends ShareKV: the session's references go, and it keeps no store
// and no namespace. The sequence starts over (Reset).
func (s *State) UnshareKV() {
	if s.kv.share == nil {
		return
	}
	s.Reset()
	s.kv.unshareAll()
	s.kv.store, s.kv.share = NoStore{}, nil
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		s.kv.layers[li].sh = nil
	}
	s.kv.ns = ""
}

func (s *State) ShareKV(st *SharedStore) error {
	switch {
	case st == nil:
		return fmt.Errorf("model: ShareKV wants a store")
	case s.nseq > 1:
		return fmt.Errorf("model: a %d-row batch cannot share kv pages: one page holds every row's slots", s.nseq)
	case len(s.kv.ent) > 0:
		return fmt.Errorf("model: %s keeps compressed entries beside its pages: kv page sharing is refused on it", s.c.Arch)
	case s.kv.ns == "":
		return fmt.Errorf("model: ShareKV wants a namespace (SetCacheKey): a page is named by its tokens only inside one")
	case s.ld != nil && s.devCount() > 0 && !s.kvShareDevice():
		return fmt.Errorf("model: %d block(s) of this session are on a device that does not name its kv "+
			"pages (nn.KVShareDevice)", s.devCount())
	case s.pos > 0:
		return fmt.Errorf("model: ShareKV wants a fresh sequence, this one is at position %d", s.pos)
	}
	s.kv.store, s.kv.share = st, st
	if s.kv.devAt == nil {
		s.kv.devAt = s.kvDevAt
	}
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		s.kv.layers[li].sh = st
	}
	return nil
}

// KVSharedBytes is what this session reads from pages it shares: held once
// in its SharedStore, and not counted by KVBytes.
func (s *State) KVSharedBytes() uint64 {
	var n uint64
	for li := s.kv.lo; li < len(s.kv.layers); li++ {
		n += s.kv.layers[li].sharedBytes()
	}
	return n
}

// KVSharedRefusals is how many times a block holding shared pages was kept
// off a device: a shared page does not move for one of its holders.
func (s *State) KVSharedRefusals() int64 { return s.kv.shareRefused }

// isShared says page n aliases a SharedStore's page.
func (s *kvPages) isShared(n int) bool { return n < len(s.shared) && s.shared[n] }

// anyShared says some page of this layer is shared.
func (s *kvPages) anyShared() bool {
	for _, x := range s.shared {
		if x {
			return true
		}
	}
	return false
}

// markShared records that page n aliases the store's.
func (s *kvPages) markShared(n int, on bool) {
	for len(s.shared) <= n {
		s.shared = append(s.shared, false)
	}
	s.shared[n] = on
}

// unshare drops this session's references to page n's shared memory and
// leaves the slots for the caller to fill or nil. The bytes are untouched:
// other sessions read them.
func (s *kvPages) unshare(n int) {
	if !s.isShared(n) {
		return
	}
	s.sh.release(s.k[n])
	if !s.latent {
		s.sh.release(s.v[n])
	}
	s.shared[n] = false
}

// cow gives page n private copies of its shared memory before a write: the
// write lands in this session's page and every other holder keeps reading
// what it read.
func (s *kvPages) cow(n int) {
	if !s.isShared(n) {
		return
	}
	if s.cowFault {
		return // the violation: write into the shared page in place
	}
	// From the layer's spare first, as a fresh page is (page0): the copy
	// overwrites it whole.
	k := s.page0()
	copy(k, s.k[n])
	v := k
	if !s.latent {
		v = s.page0()
		copy(v, s.v[n])
	}
	s.unshare(n)
	s.k[n], s.v[n] = k, v
}

// sharedBytes is what this layer's shared pages hold.
func (s *kvPages) sharedBytes() uint64 {
	var n uint64
	for i := range s.shared {
		if s.shared[i] && i < len(s.k) && s.k[i] != nil {
			n += uint64(s.pp) * 4
			if !s.latent {
				n += uint64(s.pp) * 4
			}
		}
	}
	return n
}

// publishSealed hands page n, sealed, to the store: it becomes the store's
// copy, or the store's existing copy replaces it and the session's buffers go
// back to its pool.
func (kc *kvCache) publishSealed(li, n int, key string) {
	s := &kc.layers[li]
	if kc.devAt(li) != nil {
		kc.queuePublish(li, n, key) // named where it lives, on the card
		return
	}
	if s.isShared(n) {
		return // restored from the store: already the held copy
	}
	// A page the store holds already: the session reads the held copy, and
	// its own memory goes to its spare, where the next page it needs is taken
	// from (page0).
	k := kc.share.publish(kvStoreKey{key, li * 2, n}, s.k[n])
	if &k[0] != &s.k[n][0] {
		s.spare = append(s.spare, s.k[n])
	}
	v := k
	if !s.latent {
		v = kc.share.publish(kvStoreKey{key, li*2 + 1, n}, s.v[n])
		if &v[0] != &s.v[n][0] {
			s.spare = append(s.spare, s.v[n])
		}
	}
	s.k[n], s.v[n] = k, v
	s.markShared(n, true)
}

// faultShared aliases page n of layer li from the store, or reports a miss.
func (kc *kvCache) faultShared(li, n int, key string) error {
	s := &kc.layers[li]
	// A block on a device holds its history there: the page key names is
	// put into the session's table on the card (nn.KVShareDevice.MapKV), no
	// byte moved.
	if d := kc.devAt(li); d != nil {
		if !d.MapKV(li, n, key, kc.mapEnd) {
			return ErrNoPage
		}
		kc.mapped += 2
		return nil
	}
	k := kc.share.acquire(kvStoreKey{key, li * 2, n}, s.pp)
	if k == nil {
		return ErrNoPage
	}
	v := k
	if !s.latent {
		if v = kc.share.acquire(kvStoreKey{key, li*2 + 1, n}, s.pp); v == nil {
			kc.share.release(k)
			return ErrNoPage
		}
	}
	s.k[n], s.v[n] = k, v
	s.out[n] = false
	s.markShared(n, true)
	kc.fetched += 2
	return nil
}

// unshareAll drops every shared page's reference, for Close: the pool must
// never take a page others read.
func (kc *kvCache) unshareAll() {
	for li := kc.lo; li < len(kc.layers); li++ {
		s := &kc.layers[li]
		for n := range s.shared {
			if s.shared[n] {
				s.unshare(n)
				s.k[n], s.v[n] = nil, nil
			}
		}
	}
}

// kvShareDevice says the session's device names and maps its kv pages.
func (s *State) kvShareDevice() bool {
	_, ok := s.ld.(nn.KVShareDevice)
	return ok
}

// kvDevAt is the device holding block li's history when it names its pages
// (nn.KVShareDevice), nil when the block is on the host.
func (s *State) kvDevAt(li int) nn.KVShareDevice {
	if !s.devAt(li) {
		return nil
	}
	d, _ := s.ld.(nn.KVShareDevice)
	return d
}
