package jlm_test

import (
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestChunkIsPerFile: two Files open on one container in one process, one
// given its own fill granularity, in both orders. Each fills at its own, and
// the granularity cannot change once a page has been read.
func TestChunkIsPerFile(t *testing.T) {
	src, _ := expertSource(t)
	dst := filepath.Join(t.TempDir(), "m.jlm")
	if _, err := jlm.Write(dst, src, jlm.Fingerprint{Host: "t"}); err != nil {
		t.Fatal(err)
	}
	open := func() *jlm.Frames { return openFrames(t, dst) }
	header := open().ChunkBytes()
	want := uint64(4096)
	if header == want {
		want = 8192
	}
	chunks := func(c *jlm.Frames) int64 {
		before := c.ChunksIn()
		if err := c.EnsurePage(0); err != nil {
			t.Fatal(err)
		}
		return c.ChunksIn() - before
	}
	for _, setFirst := range []bool{true, false} {
		var set, plain *jlm.Frames
		if setFirst {
			set = open()
			if err := set.SetChunk(want); err != nil {
				t.Fatal(err)
			}
			plain = open()
		} else {
			plain = open()
			set = open()
			if err := set.SetChunk(want); err != nil {
				t.Fatal(err)
			}
		}
		if set.ChunkBytes() != want || plain.ChunkBytes() != header {
			t.Fatalf("set first=%v: the Files fill at %d and %d, want %d and the header's %d",
				setFirst, set.ChunkBytes(), plain.ChunkBytes(), want, header)
		}
		page := set.File().PageBytes(0)
		for _, c := range []struct {
			f     *jlm.Frames
			chunk uint64
		}{{set, want}, {plain, header}} {
			if n, w := chunks(c.f), int64((page+c.chunk-1)/c.chunk); n != w {
				t.Errorf("set first=%v: a %d-byte page read %d chunk(s) at %d bytes, want %d",
					setFirst, page, n, c.chunk, w)
			}
		}
		if err := set.SetChunk(want * 2); err == nil {
			t.Errorf("set first=%v: SetChunk after a page was read succeeded", setFirst)
		}
	}
}
