package model

import (
	"container/list"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"sync"
)

// A picture's identity, and the two caches that read it.
//
// An image is named by what the tower READS, not by the file it came from:
// the family's preprocessing configuration, the grid the rows cover, and the
// preprocessed pixels themselves. That name exists before a single block runs,
// so it can name the rows the picture becomes in two places --
//
//	the image cache    the vision segment's output rows, keyed by the name, so
//	                   a picture seen before runs no tower block (EncodeImage);
//	the prefix cache   every row of a prompt has an id the page keys hash
//	                   (kvCache.seq): a token's own, and for an image row the
//	                   name, the row's index and its rotary position, so the
//	                   pages after a picture are named as text pages are.
//
// A name over the projector's OUTPUT values would be neither: it is known only
// after the tower has run, and on a device the split tuner picks its reduction
// order per process, so one picture encodes to different bits in two
// processes. See docs/design/vision-as-blocks.md, section 6.

// ImageKey names one run of the vision segment: sha256 over the tower's
// preprocessing configuration, the patch grid and the preprocessed pixels.
// The zero key names nothing.
type ImageKey [32]byte

func (k ImageKey) zero() bool { return k == ImageKey{} }

// imageKeyVersion changes whenever what a key covers changes, so a cache
// written under one definition is never read under another.
const imageKeyVersion = "jitllm/image/1"

// faultKeyAliases is a gate's violation: the configuration and the grid left
// out of the name, so two pictures with the same pixels and another layout
// alias. False in every real run.

// imageKey is the name of the pixels px laid out as a gh x gw patch grid,
// preprocessed by c's family.
func (c TowerConfig) imageKey(px []float32, gh, gw int) ImageKey {
	h := sha256.New()
	var b [8]byte
	put := func(v int) {
		binary.LittleEndian.PutUint64(b[:], uint64(int64(v)))
		h.Write(b[:])
	}
	putf := func(v float64) {
		binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
		h.Write(b[:])
	}
	h.Write([]byte(imageKeyVersion))
	if c.fault != faultKeyAliases {
		h.Write([]byte(c.Projector))
		for _, v := range []int{int(c.Kind), c.NLayer, c.NEmbd, c.ImageSz, c.PatchSz, c.Scale,
			c.MinTiles, c.MaxTiles, c.PosBuckets, c.Queries, c.MinPixels, c.MaxPixels,
			c.WinPattern, c.WinSize, c.ProjDim, b2i(c.CLS), b2i(c.Rope), b2i(c.RMS), b2i(c.PreLN)} {
			put(v)
		}
		for i := 0; i < 3; i++ {
			putf(c.Mean[i])
			putf(c.Std[i])
		}
		put(gh)
		put(gw)
	}
	put(len(px))
	buf := make([]byte, 4*1024)
	for i := 0; i < len(px); {
		n := 0
		for ; n+4 <= len(buf) && i < len(px); i, n = i+1, n+4 {
			binary.LittleEndian.PutUint32(buf[n:], math.Float32bits(px[i]))
		}
		h.Write(buf[:n])
	}
	var k ImageKey
	h.Sum(k[:0])
	return k
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// imageCache is the vision segment's output rows by ImageKey: a model's, so
// every State of it shares what any of them encoded. Least recently used
// entries go first once the bytes pass the budget; an entry is immutable, and
// what a caller gets is a copy.
type imageCache struct {
	mu      sync.Mutex
	budget  int64
	bytes   int64
	entries map[ImageKey]*list.Element
	lru     list.List // front is the most recent
	hits    int64
	misses  int64
}

type imageEntry struct {
	key  ImageKey
	rows []float32
	grid ImageGrid
}

// defaultImageCache is the image cache's budget when nothing sets one: a few
// hundred pictures of a small tower, a few dozen of a large one.
const defaultImageCache = 256 << 20

func (c *imageCache) get(k ImageKey) (*imageEntry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[k]
	if !ok {
		c.misses++
		return nil, false
	}
	c.hits++
	c.lru.MoveToFront(e)
	return e.Value.(*imageEntry), true
}

func (c *imageCache) put(k ImageKey, rows []float32, grid ImageGrid) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil {
		c.entries = map[ImageKey]*list.Element{}
	}
	if _, ok := c.entries[k]; ok {
		return
	}
	size := int64(4 * len(rows))
	if size > c.budget {
		return
	}
	for c.bytes+size > c.budget {
		old := c.lru.Back()
		oe := old.Value.(*imageEntry)
		c.lru.Remove(old)
		delete(c.entries, oe.key)
		c.bytes -= int64(4 * len(oe.rows))
	}
	c.entries[k] = c.lru.PushFront(&imageEntry{key: k, rows: append([]float32(nil), rows...), grid: grid})
	c.bytes += size
}

// ImageCacheStats is how many encodes the model's image cache answered and
// how many ran the tower.
func (m *Model) ImageCacheStats() (hits, misses int64) {
	m.imgCache.mu.Lock()
	defer m.imgCache.mu.Unlock()
	return m.imgCache.hits, m.imgCache.misses
}

// EncodeImage is Encode through the model's image cache: the picture px --
// preprocessed, its grid the one this State last preprocessed or was told --
// is named (ImageKey), and a name seen before returns its rows without a
// block running. The rows are the caller's.
//
// Encode itself always runs the tower: a gate comparing two placements of the
// same picture must not be answered by the cache.
func (s *State) EncodeImage(px []float32) (Image, error) {
	if s.vis == nil {
		return Image{}, fmt.Errorf("model: EncodeImage on a State that is not the vision segment's")
	}
	gh, gw := s.patchGrid()
	rows, k, err := s.encodeNamed(px, gh, gw, func() ([]float32, error) { return s.Encode(px) })
	if err != nil {
		return Image{}, err
	}
	g := s.Grid()
	embd, deep := splitDeep(rows, g.Rows(), s.vis.t.Cfg.ProjDim)
	return Image{Embd: embd, Grid: g, Key: &k, Deep: deep}, nil
}

// encodeNamed names the pixels px of a gh x gw grid and answers from the image
// cache, or runs encode and keeps what it returns. The rows are a copy.
func (s *State) encodeNamed(px []float32, gh, gw int, encode func() ([]float32, error)) ([]float32, ImageKey, error) {
	return s.encodeNamedKey(s.vis.t.Cfg.imageKey(px, gh, gw), gh, gw, encode)
}

// encodeNamedKey is encodeNamed for pixels already named k.
func (s *State) encodeNamedKey(k ImageKey, gh, gw int, encode func() ([]float32, error)) ([]float32, ImageKey, error) {
	// A gate's violation breaks the tower on purpose and must run it: the
	// cache would answer with the clean rows.
	if s.vis.fault != towerFaultNone || s.vis.t.Cfg.fault != towerFaultNone {
		rows, err := encode()
		return append([]float32(nil), rows...), k, err
	}
	if e, ok := s.m.imgCache.get(k); ok {
		s.vis.cacheHits++
		return append([]float32(nil), e.rows...), k, nil
	}
	rows, err := encode()
	if err != nil {
		return nil, k, err
	}
	sc := max(s.vis.t.Cfg.Scale, 1)
	s.m.imgCache.put(k, rows, s.vis.t.Cfg.gridOf(gh/sc, gw/sc))
	return append([]float32(nil), rows...), k, nil
}
