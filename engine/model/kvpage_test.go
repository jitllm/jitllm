package model

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/cpu"
)

// TestKVSpansCoverTheWindowExactly: the page walk must visit every position of
// [w0, w0+want) once, in increasing order, and never step outside a page.
//
// Order is as load-bearing as coverage: the accumulate kernel seeds from Out for
// every span after the first, so the sum is bit-identical to one contiguous call
// only if spans arrive in increasing position order.
func TestKVSpansCoverTheWindowExactly(t *testing.T) {
	for _, P := range []int{32, 64, 256} {
		s := kvPages{p: P}
		for _, w0 := range []int{0, 1, P - 1, P, P + 1, 3*P + 7} {
			for _, want := range []int{1, 2, P - 1, P, P + 1, 5*P + 3} {
				spans := s.spans(nil, w0, want)
				if len(spans) == 0 {
					t.Fatalf("P=%d w0=%d want=%d: no spans", P, w0, want)
				}
				pos := w0
				for i, sp := range spans {
					if sp.n <= 0 {
						t.Fatalf("P=%d w0=%d want=%d: span %d is empty", P, w0, want, i)
					}
					if sp.first < 0 || sp.first+sp.n > P {
						t.Fatalf("P=%d w0=%d want=%d: span %d runs [%d,%d) of a %d-position page",
							P, w0, want, i, sp.first, sp.first+sp.n, P)
					}
					gotPage, gotOff := s.page(pos)
					if sp.page != gotPage || sp.first != gotOff {
						t.Fatalf("P=%d w0=%d want=%d: span %d says page %d offset %d, position %d is page %d offset %d -- the walk is not contiguous",
							P, w0, want, i, sp.page, sp.first, pos, gotPage, gotOff)
					}
					if i > 0 && sp.page <= spans[i-1].page {
						t.Fatalf("P=%d w0=%d want=%d: span %d is page %d after page %d -- not increasing",
							P, w0, want, i, sp.page, spans[i-1].page)
					}
					pos += sp.n
				}
				if pos != w0+want {
					t.Fatalf("P=%d w0=%d want=%d: the walk covered [%d,%d), short by %d",
						P, w0, want, w0, pos, w0+want-pos)
				}
			}
		}
	}
	// A layer with no pages has no spans, which is what a linear-attention block
	// relies on rather than a nil check at every reader.
	none := kvPages{p: 0}
	if got := none.spans(nil, 0, 100); got != nil {
		t.Fatalf("a layer with no pages produced %d span(s)", len(got))
	}
}

// TestKVPageGeometryFollowsTheLayerKind pins the three answers the per-layer page
// size exists to give, and what newKVCache does with them.
func TestKVPageGeometryFollowsTheLayerKind(t *testing.T) {
	// Ordinary attention: a kernel-aligned run, and a power of two so that
	// pos/P is a shift.
	c := &Config{NLayer: 4, NCtx: 4096}
	for li := 0; li < c.NLayer; li++ {
		p := kvPagePositions(c, li, 0)
		if p <= 0 {
			t.Fatalf("layer %d: attention layer got %d positions per page", li, p)
		}
		if p%kvKeyTile != 0 {
			t.Fatalf("layer %d: page of %d positions is not a multiple of the %d key tile; "+
				"the last tile of every page would straddle the boundary", li, p, kvKeyTile)
		}
		if p&(p-1) != 0 {
			t.Errorf("layer %d: page of %d positions is not a power of two, so pos/P is a "+
				"divide rather than a shift", li, p)
		}
	}
	// A context shorter than a page does not get a page bigger than the context.
	short := &Config{NLayer: 1, NCtx: 64}
	if p := kvPagePositions(short, 0, 0); p > align(64, kvKeyTile) {
		t.Errorf("NCtx=64 gave a %d-position page", p)
	}
	// A sliding-window layer is one page of its window, forever.
	swa := &Config{NLayer: 2, NCtx: 8192, SWAWindow: 512, SWAPeriod: 2}
	if !swa.SWA(0) {
		t.Fatal("the SWA arm is unreachable in this config; the gate proves nothing about it")
	}
	if p := kvPagePositions(swa, 0, 0); p != 512 {
		t.Errorf("a 512-key local layer got a %d-position page", p)
	}

	// A linear-attention layer keeps no pages at all, asserted through
	// newKVCache: its history is a recurrent summary outside this cache, and a
	// page would be coverage that restore() counts.
	hyb := &Config{NLayer: 3, NCtx: 4096, LayerKinds: []jlm.LayerKind{
		jlm.LayerLinearAttn, jlm.LayerFullAttn, jlm.LayerLinearAttn,
	}}
	if hyb.LayerKind(0) != jlm.LayerLinearAttn {
		t.Fatal("the linear arm is unreachable in this config; the gate proves nothing about it")
	}
	kc := newKVCache(hyb, "geom", 1, kvLayout{}, nil, 0)
	for li, want := range []bool{false, true, false} {
		got := kvPagePositions(hyb, li, 0) > 0
		if got != want {
			t.Errorf("layer %d (%v): pages=%v, want %v", li, hyb.LayerKind(li), got, want)
		}
		if (kc.layers[li].p > 0) != want {
			t.Errorf("layer %d: newKVCache gave p=%d", li, kc.layers[li].p)
		}
		if !want && len(kc.layers[li].k) != 0 {
			t.Errorf("layer %d: a linear layer allocated %d page(s)", li, len(kc.layers[li].k))
		}
	}
}

// TestRelocationCrossesPages covers the gather/scatter pair over more than one
// page, which is how a block's history moves to a device and back (migrateKV,
// driven by SetGPULayers and seamtune). scatter must allocate every page it
// writes, since a placed block's host pages are all nil on demotion, and gather
// must refuse a page trim evicted in place.
func TestRelocationCrossesPages(t *testing.T) {
	const p, kvDim, n = 8, 4, 20 // 20 positions over 8 per page = 3 pages
	l := kvLayout{nKV: 1, headDim: kvDim, maxSeq: p}
	pg := kvPages{p: p, pp: l.slots(1 * p * kvDim)}

	// A history arriving from a device: nothing is resident, every page is new.
	k := make([]float32, l.slots(n*kvDim))
	v := make([]float32, l.slots(n*kvDim))
	for i := range k {
		k[i], v[i] = float32(i+1), float32(-(i + 1))
	}
	if !pg.scatter(l, 0, n, k, v) {
		t.Fatal("scatter refused a fresh multi-page history")
	}
	if last, _ := pg.page(n - 1); last < 2 {
		t.Fatalf("this config only reaches page %d; the gate needs more than one page "+
			"or it tests the single-page path twice", last)
	}
	for i := 0; i <= 2; i++ {
		if !pg.resident(i) {
			t.Fatalf("page %d is not resident after scatter, so scatter wrote into nil", i)
		}
	}

	// And it comes back byte for byte.
	gk := make([]float32, l.slots(n*kvDim))
	gv := make([]float32, l.slots(n*kvDim))
	if !pg.gather(l, 0, n, gk, gv) {
		t.Fatal("gather refused a fully resident history")
	}
	for i := range k {
		if gk[i] != k[i] || gv[i] != v[i] {
			t.Fatalf("round trip differs at %d: k %v vs %v, v %v vs %v",
				i, gk[i], k[i], gv[i], v[i])
		}
	}

	// An evicted page is a refusal, not a panic: trim nils a slot in place, so
	// "addressable" and "resident" are different questions.
	pg.k[1], pg.v[1] = nil, nil
	if pg.gather(l, 0, n, gk, gv) {
		t.Fatal("gather accepted a history with an evicted page: the caller would have " +
			"migrated whatever the destination buffer already held")
	}
}

// TestPageKeySeparatesAliasingGeometries: a page's byte count cannot tell these
// apart, so the key must.
//
// A page is nseq * p * (NKVHead*HeadDim) * elem bytes, so tinyllama (4 x 64),
// Qwen2-1.5B (2 x 128) and gemma-2b (1 x 256) all give byte-identical pages, and
// head-major is the same bytes permuted. A byte-count check cannot catch any of
// them loading into another.
func TestPageKeySeparatesAliasingGeometries(t *testing.T) {
	toks := []int32{5, 6, 7, 8, 9, 10, 11, 12}
	base := &Config{NLayer: 1, NCtx: 4096, NKVHead: 4, HeadDim: 64, NEmbd: 2048,
		NHead: 32, NRot: 64, NFFN: 5632, NVocab: 32000, RopeBase: 10000, Arch: "llama"}

	keyOf := func(c *Config, nseq int, l kvLayout) string {
		kc := newKVCache(c, "id", nseq, l, nil, 0)
		kc.ns, kc.seq = "ns", toks
		kc.layers[0].p = 8
		return kc.pageKey(0, 0)
	}
	l32 := kvLayout{nKV: 4, headDim: 64}
	want := keyOf(base, 1, l32)
	if want == "" {
		t.Fatal("the reference key is empty, so every comparison below is vacuous")
	}

	// Each variant keeps the page byte count identical where it can.
	qwen := *base
	qwen.NKVHead, qwen.HeadDim = 2, 128 // kvDim 256, same as base
	gemma := *base
	gemma.NKVHead, gemma.HeadDim = 1, 256 // kvDim 256, same as base
	fewer := *base
	fewer.NVocab = 128256 // same geometry, different model
	for _, tc := range []struct {
		name string
		c    *Config
		nseq int
		l    kvLayout
	}{
		{"nKV 2 x hd 128 (same kvDim)", &qwen, 1, kvLayout{nKV: 2, headDim: 128}},
		{"nKV 1 x hd 256 (same kvDim)", &gemma, 1, kvLayout{nKV: 1, headDim: 256}},
		{"head-major (same bytes, permuted)", base, 1, kvLayout{nKV: 4, headDim: 64, headMajor: true}},
		{"f16 cache", base, 1, kvLayout{fmt: cpu.KVF16, nKV: 4, headDim: 64}},
		{"nseq 2", base, 2, l32},
		{"a different vocabulary", &fewer, 1, l32},
	} {
		if got := keyOf(tc.c, tc.nseq, tc.l); got == want {
			t.Errorf("%s produced the SAME page key as the reference: its pages would be "+
				"loaded and reinterpreted, which no length check can catch", tc.name)
		}
	}

	// The same geometry with the same tokens must agree, or nothing is ever
	// reused and the gate above would pass with the key randomised.
	if again := keyOf(base, 1, l32); again != want {
		t.Fatalf("the same geometry and tokens gave %q then %q: no page could ever be "+
			"found again", want, again)
	}
	// A different token prefix separates too, which is the solver's own premise.
	kc := newKVCache(base, "id", 1, l32, nil, 0)
	kc.ns, kc.seq, kc.layers[0].p = "ns", []int32{5, 6, 7, 8, 9, 10, 11, 99}, 8
	if got := kc.pageKey(0, 0); got == want {
		t.Error("a different last token produced the same key")
	}
}

// TestChunkKeysAreTheirDigests holds chunkKey's one-pass keys to the digest
// each names, across a growing sequence, a rollback and a rewritten tail.
func TestChunkKeysAreTheirDigests(t *testing.T) {
	kc := &kvCache{ns: "t", geom: []byte("g"), bh: map[int][]string{}}
	check := func() {
		t.Helper()
		for e := kvChunk; e <= len(kc.seq); e += kvChunk {
			if got, want := kc.chunkKey(e), kc.digest(-kvChunk, 0, e); got != want {
				t.Fatalf("chunk ending at %d of %d: %q, its digest %q", e, len(kc.seq), got, want)
			}
		}
	}
	ids := make([]int32, 100)
	for i := range ids {
		ids[i] = int32(i * 7)
	}
	kc.note(0, ids[:40]...)
	check()
	kc.note(40, ids[40:]...)
	check()
	kc.note(33, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9) // a rollback rewrites the tail
	check()
	if len(kc.ck) != len(kc.seq)/kvChunk {
		t.Fatalf("%d chunk keys kept for %d positions", len(kc.ck), len(kc.seq))
	}
}
