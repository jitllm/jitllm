package model

import (
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
	"time"
)

// FileStore keeps KV pages on disk, one file per page.
//
// It lives in core because it carries no transport dependency; remote stores
// belong outside. One file per page needs no header or seek, makes Drop an
// os.RemoveAll, and makes a partial write unobservable (see Set). It is where
// a sealed page goes when no tier holds it, not the RAM prefill buffer.
type FileStore struct {
	dir string
	// max bounds the store in bytes. Nothing here expires, so without a cap
	// the disk fills; zero is unbounded, for a caller managing the directory.
	max     uint64
	mu      sync.Mutex
	used    uint64 // bytes, counted lazily on the first Set and maintained after
	known   bool
	evicted int64
}

// NewFileStoreLimit roots a store at dir and bounds it to max bytes, evicting
// the least recently used pages when a write would exceed it.
//
// LRU is by the file's own mtime (Get refreshes it), which needs no index and
// survives a restart. It evicts a page, not a prefix: dropping a middle page
// leaves the later ones unreachable until they age out too, which is rare
// because a prefix's pages are written, and so age, together.
func NewFileStoreLimit(dir string, max uint64) (*FileStore, error) {
	f, err := NewFileStore(dir)
	if err != nil {
		return nil, err
	}
	f.max = max
	return f, nil
}

// NewFileStore roots a store at dir, creating it if it does not exist. It is
// unbounded; NewFileStoreLimit is the one to reach for unless the caller is
// managing the directory themselves.
func NewFileStore(dir string) (*FileStore, error) {
	if dir == "" {
		return nil, errors.New("model: FileStore needs a directory")
	}
	// 0700: a KV page is the model's state for a specific prompt, and the
	// per-user namespace is a privacy boundary a world-readable directory
	// would defeat.
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	return &FileStore{dir: dir}, nil
}

// cacheDir is one sequence's own directory, so Drop is a single removal and two
// sessions cannot collide on a page id.
//
// The free-form key is url.PathEscaped so no key can name a directory outside
// f.dir; the "kv-" prefix closes the case escaping leaves open (".." passes
// through PathEscape, and "kv-.." is just a name).
func (f *FileStore) cacheDir(cacheId string) string {
	return filepath.Join(f.dir, "kv-"+url.PathEscape(cacheId))
}

func (f *FileStore) page(cacheId string, layer, index int) string {
	return filepath.Join(f.cacheDir(cacheId),
		strconv.Itoa(layer)+"-"+strconv.Itoa(index))
}

// Get streams the page into the caller's sink.
//
// A missing file is ErrNoPage ("recompute it") and anything else is an error
// ("stop"). The page's length is not checked here: the store does not know the
// geometry, and kvCache.fault asserts it.
func (f *FileStore) Get(cacheId string, layer, index int, page io.Writer) error {
	fh, err := os.Open(f.page(cacheId, layer, index))
	if err != nil {
		if os.IsNotExist(err) {
			return ErrNoPage
		}
		return err
	}
	defer fh.Close()
	if _, err := io.Copy(page, fh); err != nil {
		return err
	}
	// Touched on read so a page every request hits is not the first evicted.
	// Best effort: a read-only mount still serves.
	if f.max > 0 {
		now := time.Now()
		os.Chtimes(f.page(cacheId, layer, index), now, now)
	}
	return nil
}

// Set writes the page, atomically.
//
// It is written beside itself and renamed, so a page is either absent or
// whole: a page present but short would be read back as history the model
// never wrote.
func (f *FileStore) Set(cacheId string, layer, index int, page io.Reader) error {
	dir := f.cacheDir(cacheId)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	final := f.page(cacheId, layer, index)
	tmp, err := os.CreateTemp(dir, ".part-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name) // a no-op once the rename below has succeeded
	if _, err := io.Copy(tmp, page); err != nil {
		tmp.Close()
		return err
	}
	// Sync before the rename, or a crash can leave the renamed file present
	// with its bytes still in flight.
	sz, _ := tmp.Seek(0, io.SeekCurrent)
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(name, final); err != nil {
		return err
	}
	f.account(uint64(sz))
	return nil
}

// account adds n bytes to the running total and evicts if that puts the store
// over its limit.
func (f *FileStore) account(n uint64) {
	if f.max == 0 {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.known {
		f.used, f.known = f.scan(), true
	}
	f.used += n
	if f.used <= f.max {
		return
	}
	// Evict to 90% of the cap, not the cap, or every later Set at the bound
	// would walk the directory again.
	f.evictTo(f.max * 9 / 10)
}

type kvFileEnt struct {
	path string
	size uint64
	mod  int64
}

// scan totals the store and is the only walk of the directory. It runs once per
// process, on the first bounded write.
func (f *FileStore) scan() uint64 {
	var n uint64
	for _, e := range f.walk() {
		n += e.size
	}
	return n
}

func (f *FileStore) walk() []kvFileEnt {
	dirs, err := os.ReadDir(f.dir)
	if err != nil {
		return nil
	}
	var out []kvFileEnt
	for _, d := range dirs {
		if !d.IsDir() {
			continue
		}
		sub := filepath.Join(f.dir, d.Name())
		ents, err := os.ReadDir(sub)
		if err != nil {
			continue
		}
		for _, e := range ents {
			if e.IsDir() || len(e.Name()) == 0 || e.Name()[0] == '.' {
				continue
			}
			fi, err := e.Info()
			if err != nil {
				continue
			}
			out = append(out, kvFileEnt{filepath.Join(sub, e.Name()), uint64(fi.Size()), fi.ModTime().UnixNano()})
		}
	}
	return out
}

// evictTo removes the least recently used pages until the store is under want.
// The caller holds f.mu.
func (f *FileStore) evictTo(want uint64) {
	ents := f.walk()
	sort.Slice(ents, func(i, j int) bool { return ents[i].mod < ents[j].mod })
	used := uint64(0)
	for _, e := range ents {
		used += e.size
	}
	for _, e := range ents {
		if used <= want {
			break
		}
		if os.Remove(e.path) == nil {
			used -= e.size
			f.evicted++
		}
	}
	f.used = used
}

// Evicted is how many page files this store has deleted to stay inside its
// limit. A cache that is thrashing and one that is sized right look identical
// without it.
func (f *FileStore) Evicted() int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.evicted
}

// Drop removes everything held for one cache. A sequence that ended leaves
// nothing behind, which is the difference between a store and a leak.
func (f *FileStore) Drop(cacheId string) error {
	err := os.RemoveAll(f.cacheDir(cacheId))
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// Pages is how many page files this store holds for one cache, for the gates
// that assert a write happened and Drop emptied it.
func (f *FileStore) Pages(cacheId string) (int, error) {
	ents, err := os.ReadDir(f.cacheDir(cacheId))
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	n := 0
	for _, e := range ents {
		if !e.IsDir() && len(e.Name()) > 0 && e.Name()[0] != '.' {
			n++ // a .part-* is an unfinished Set, not a page
		}
	}
	return n, nil
}

// Total is every page file this store holds, across all keys.
//
// Once pages are content-addressed a session's pages spread over one key per
// page (so sessions can share them), and "how much is here" is the useful
// question rather than Pages(cacheId).
func (f *FileStore) Total() (int, error) {
	dirs, err := os.ReadDir(f.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	n := 0
	for _, d := range dirs {
		if !d.IsDir() {
			continue
		}
		ents, err := os.ReadDir(filepath.Join(f.dir, d.Name()))
		if err != nil {
			return 0, err
		}
		for _, e := range ents {
			if !e.IsDir() && len(e.Name()) > 0 && e.Name()[0] != '.' {
				n++
			}
		}
	}
	return n, nil
}

// errPageMismatch is "these bytes are not a page of this shape". It is distinct
// from an I/O failure on purpose: a mismatch is a stale or foreign cache
// directory and must cost a prefill, while a broken disk must stop the request.
// See kvCache.miss.
const errPageMismatch modelError = "model: kv page does not match this geometry"

// shortPage is what a store returning the wrong number of bytes gets. It wraps
// errPageMismatch so the fault path treats it as a miss rather than as a
// failure -- see kvCache.fault.
func shortPage(li, n, got, want int) error {
	return fmt.Errorf("%w: came back short: layer %d page %d is %d bytes, the geometry says %d",
		errPageMismatch, li, n, got, want)
}
