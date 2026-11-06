package engine

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/nn"
)

// TestPagedKVIsTokenIdenticalToOnePage: splitting the history into pages must
// not change a single token.
//
// The control arm is this same code with the page size raised to maxSeq, which
// gives one page whose bytes are the flat cache, so there is no second
// implementation to drift. It asserts token identity rather than an NMSE bound
// (a page-boundary bug hides inside the cross-tier logit band); equality is
// reachable because nn.AttnAccInto reproduces the contiguous FMA order exactly.
func TestPagedKVIsTokenIdenticalToOnePage(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const maxSeq, gen = 96, 24
	prompt := []int32{1, 450, 7483, 310, 3444, 338}

	run := func(page int) ([]int32, int, int64) {
		s := testInstance(t, m, maxSeq, withKVPage(page))
		defer s.Close()
		out := make([]int32, 0, gen)
		var lg []float32
		for _, id := range prompt {
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		for i := 0; i < gen; i++ {
			id := int32(argmax(lg))
			out = append(out, id)
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		// Pages committed for an attention layer: whether the arm exercised a
		// boundary at all.
		pages := 0
		for li := range s.kv.layers {
			if n := s.kv.layers[li].mem.Len(); n > pages {
				pages = n
			}
		}
		return out, pages, s.AttnPaired()
	}

	// The control: one page, which is the unpaged cache by construction.
	want, wantPages, wantPaired := run(maxSeq)
	if wantPages != 1 {
		t.Fatalf("the one-page control committed %d pages -- it is not the control it "+
			"claims to be", wantPages)
	}

	for _, P := range []int{8, 16} {
		got, pages, paired := run(P)
		// The arm must have crossed a boundary, or it tests the control again.
		if pages < 2 {
			t.Fatalf("P=%d committed %d page(s) over %d positions -- the walk never "+
				"crossed a boundary and the gate proved nothing",
				P, pages, len(prompt)+gen)
		}
		if len(got) != len(want) {
			t.Fatalf("P=%d produced %d tokens, want %d", P, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("P=%d (%d pages): token %d is %d, one page gives %d\n got %v\nwant %v",
					P, pages, i, got[i], want[i], got, want)
			}
		}
		// The paired kernel must still run across pages; tokens alone cannot see
		// it declining, since two single passes give the same answer.
		if wantPaired > 0 && paired == 0 {
			t.Errorf("P=%d: the one-page arm paired %d head(s) and the paged arm paired none "+
				"-- the paired walk declined across the boundary", P, wantPaired)
		}
		t.Logf("P=%-4d %d pages, %d tokens identical to the one-page cache, %d paired",
			P, pages, len(got), paired)
	}
}

func argmax(v []float32) int {
	best := 0
	for i := range v {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

// TestKVStoreRoundTripChangesNoToken: a page evicted to the store and faulted
// back must give the same tokens as one that never left.
//
// Runs the same sequence three ways: resident-only, written through to a
// MemStore but never evicted, and evicted hard enough that the read path fetches
// pages back. The counters are asserted too, so a budget that never evicted
// cannot pass by testing the resident path twice.
func TestKVStoreRoundTripChangesNoToken(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const maxSeq, gen, page = 96, 24, 8
	prompt := []int32{1, 450, 7483, 310, 3444, 338}

	run := func(store KVStore, budget uint64) ([]int32, int64, int64, int64) {
		s := testInstance(t, m, maxSeq, withKVPage(page))
		defer s.Close()
		if store != nil {
			s.SetKVStore(store)
		}
		mustBudget(t, s, budget)
		var lg []float32
		for _, id := range prompt {
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		out := make([]int32, 0, gen)
		for i := 0; i < gen; i++ {
			id := int32(argmax(lg))
			out = append(out, id)
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		st, fe, ev := s.KVPageStats()
		return out, st, fe, ev
	}

	want, _, _, _ := run(nil, 0)

	// Write-through, nothing evicted: the store is exercised and the read path
	// never needs it.
	got, stored, fetched, evicted := run(NewMemStore(), 0)
	if stored == 0 {
		t.Fatal("no page was written through to the store -- the gate proved nothing")
	}
	if evicted != 0 || fetched != 0 {
		t.Errorf("an unbudgeted session evicted %d and fetched %d pages", evicted, fetched)
	}
	sameTokens(t, "write-through", want, got)

	// A budget small enough that sealed pages have to go.
	got, stored, fetched, evicted = run(NewMemStore(), 1<<10)
	if evicted == 0 {
		t.Fatal("a 1 KiB budget evicted nothing -- the fault path never ran")
	}
	if fetched == 0 {
		t.Fatalf("%d page(s) evicted and none fetched back: the read walked over pages "+
			"that are not there", evicted)
	}
	sameTokens(t, "evict-and-fault", want, got)
	t.Logf("evicting arm: %d stored, %d evicted, %d faulted back -- tokens identical",
		stored, evicted, fetched)
}

func sameTokens(t *testing.T, arm string, want, got []int32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d tokens, want %d", arm, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: token %d is %d, resident-only gives %d\n got %v\nwant %v",
				arm, i, got[i], want[i], got, want)
		}
	}
}

// TestFileStoreRoundTripChangesNoToken: pages that went to DISK and came back
// must give the same tokens as pages that never left memory.
//
// The file backend is the one that serialises a page and reads it back through
// a different path. Counters are asserted so an arm that never evicted, or a
// Drop that removed nothing, cannot pass.
func TestFileStoreRoundTripChangesNoToken(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const maxSeq, gen, page = 96, 24, 8
	prompt := []int32{1, 450, 7483, 310, 3444, 338}

	run := func(store KVStore, budget uint64) ([]int32, int64, int64, int64, string) {
		s := testInstance(t, m, maxSeq, withKVPage(page))
		defer s.Close()
		if store != nil {
			s.SetKVStore(store)
		}
		mustBudget(t, s, budget)
		var lg []float32
		for _, id := range prompt {
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		out := make([]int32, 0, gen)
		for i := 0; i < gen; i++ {
			id := int32(argmax(lg))
			out = append(out, id)
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		st, fe, ev := s.KVPageStats()
		return out, st, fe, ev, s.kv.id
	}

	want, _, _, _, _ := run(nil, 0)

	fs, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	got, stored, fetched, evicted, id := run(fs, 1<<10)
	if stored == 0 || evicted == 0 || fetched == 0 {
		t.Fatalf("stored %d, evicted %d, faulted %d -- the disk round trip did not happen",
			stored, evicted, fetched)
	}
	sameTokens(t, "file-store", want, got)

	// The pages are really on disk, and there are no unfinished .part files.
	onDisk, err := fs.Pages(id)
	if err != nil {
		t.Fatal(err)
	}
	if onDisk == 0 {
		t.Fatal("the store reports no page files after a run that stored pages")
	}
	t.Logf("%d stored, %d evicted, %d faulted back, %d page file(s) on disk -- tokens identical",
		stored, evicted, fetched, onDisk)

	// Drop actually empties: a store that forgets its index and keeps the bytes
	// leaks disk.
	if err := fs.Drop(id); err != nil {
		t.Fatal(err)
	}
	if n, err := fs.Pages(id); err != nil || n != 0 {
		t.Fatalf("after Drop the store holds %d page(s) (err %v)", n, err)
	}
}

// TestFileStoreRefusesAShortPage: a page that comes back with fewer bytes than
// the geometry says must be an ERROR, not zeros.
//
// A truncated file reads back short into a freshly allocated buffer, so the
// missing tail is zeros: keys the model never wrote, attended to as a plausible
// history.
func TestFileStoreRefusesAShortPage(t *testing.T) {
	dir := t.TempDir()
	fs, err := NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	full := make([]byte, 4096)
	for i := range full {
		full[i] = byte(i)
	}
	if err := fs.Set("7", 0, 0, bytes.NewReader(full)); err != nil {
		t.Fatal(err)
	}
	// Truncate it behind the store's back, which is what a full disk leaves.
	p := fs.page("7", 0, 0)
	if err := os.Truncate(p, 1024); err != nil {
		t.Fatal(err)
	}
	w := newSliceWriter(make([]byte, 4096))
	if err := fs.Get("7", 0, 0, w); err != nil {
		t.Fatalf("Get on a truncated page: %v", err)
	}
	// The store cannot know the geometry, so it returns what is there; the
	// caller knows the length and is where the refusal belongs.
	if w.n == len(w.b) {
		t.Fatal("a truncated page filled the whole buffer")
	}
	// Assert that the fault refuses, not that a constructor builds an error.
	kc := &kvCache{id: "7", store: fs, layers: []kvPages{{p: 8, pp: len(w.b) / 4, mem: heldPages(len(w.b) / 4)}}}
	err = kc.fault(0, 0)
	// A miss, not a failure, and the key is dropped on the way past: a page left
	// by a differently-configured process should cost a prefill, not brick that
	// prefix forever.
	if !errors.Is(err, ErrNoPage) {
		t.Fatalf("faulting a truncated page returned %v, want ErrNoPage", err)
	}
	if kc.mismatched == 0 {
		t.Fatal("the mismatch was not counted, so an operator cannot tell a stale cache " +
			"directory from a cold one")
	}
	if n, _ := fs.Pages("7"); n != 0 {
		t.Fatalf("the offending key still holds %d page(s): the next request would hit "+
			"it again and the prefix stays bricked", n)
	}
	if kc.layers[0].resident(0) {
		t.Fatal("a refused fault left the page resident, so a reader would see its zeros")
	}
	t.Logf("a short page is refused at the fault: %v", err)
	// A missing page is a miss, not a failure.
	if err := fs.Get("7", 0, 99, newSliceWriter(make([]byte, 16))); !errors.Is(err, ErrNoPage) {
		t.Fatalf("a page that was never written returned %v, want ErrNoPage", err)
	}
}

// truncStore is a FileStore that hands back a page one byte short, as a full
// disk or a killed writer leaves.
type truncStore struct {
	*FileStore
	bad map[kvStoreKey]bool
}

func (t *truncStore) Get(cacheId string, layer, index int, page io.Writer) error {
	if t.bad[kvStoreKey{cacheId, layer, index}] {
		var buf bytes.Buffer
		if err := t.FileStore.Get(cacheId, layer, index, &buf); err != nil {
			return err
		}
		b := buf.Bytes()
		if len(b) > 0 {
			b = b[:len(b)-1]
		}
		_, err := page.Write(b)
		return err
	}
	return t.FileStore.Get(cacheId, layer, index, page)
}

// TestAShortPageFailsTheRequest: a page the store cannot return in full must
// make Forward return an error, not produce a token.
//
// A short page leaves zeros in a fresh buffer, and the answer is fluent and
// wrong, so counting it and carrying on is not enough. Against the violation
// (kvFault recording nothing) this test produces tokens.
func TestAShortPageFailsTheRequest(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	fs, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	ts := &truncStore{FileStore: fs, bad: map[kvStoreKey]bool{}}

	s := testInstance(t, m, 96, withKVPage(8))
	defer s.Close()
	s.SetKVStore(ts)
	mustBudget(t, s, 1<<10) // small enough that sealed pages are evicted

	// Run far enough that pages seal, spill and start being faulted back.
	for i := 0; i < 20; i++ {
		if _, err := s.Forward(int32(5 + i)); err != nil {
			t.Fatalf("clean run failed at token %d: %v", i, err)
		}
	}
	stored, fetched, _ := s.KVPageStats()
	if stored == 0 || fetched == 0 {
		t.Fatalf("stored %d, faulted %d -- nothing round-tripped, so nothing can be "+
			"truncated and the gate would prove nothing", stored, fetched)
	}

	// Now corrupt every page of layer 0 and demand a refusal.
	for n := 0; n < 8; n++ {
		ts.bad[kvStoreKey{s.kv.id, 0, n}] = true
		ts.bad[kvStoreKey{s.kv.id, 1, n}] = true
	}
	// Drop what is resident so the next read must go to the store.
	s.kv.budget = 1
	s.kv.trim(s.pos)

	var lastErr error
	for i := 0; i < 8 && lastErr == nil; i++ {
		_, lastErr = s.Forward(int32(30 + i))
	}
	if lastErr == nil {
		t.Fatal("a truncated page produced tokens: the engine answered from zeros " +
			"where its own history should have been")
	}
	// ErrNoPage: a geometry refusal is a miss, but a miss on an evicted page
	// mid-generation leaves nothing to recompute from, so the request still
	// fails. The mismatch counter keeps "corrupt" distinct from "absent".
	if !errors.Is(lastErr, ErrNoPage) {
		t.Fatalf("got %v, want the page to be treated as absent", lastErr)
	}
	if s.KVStoreMismatches() == 0 {
		t.Fatal("the truncated page was not counted as a mismatch, so a stale cache " +
			"directory is indistinguishable from a cold one in the counters")
	}
	t.Logf("refused, as it must: %v (%d mismatch(es) counted)", lastErr, s.KVStoreMismatches())
}

// TestPrefixCacheSkipsWhatTheStoreHolds is the reuse gate: a second session that
// names the same key must attend over the first one's pages instead of computing
// them, and must produce the same tokens doing it.
//
// Asserts both halves: positions restored > 0 and fewer tokens prefilled
// (otherwise it tests the resident path twice), and identical output (a skip
// with different tokens is attention over history the model never wrote).
func TestPrefixCacheSkipsWhatTheStoreHolds(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const maxSeq, gen, page = 256, 16, 8
	// Long enough that several pages seal, or the gate tests an empty cache.
	prompt := make([]int32, 0, 60)
	for i := 0; i < 60; i++ {
		prompt = append(prompt, int32(300+i*7))
	}
	tail := []int32{338, 3681, 310}

	fs, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	const key = "tinyllama/q3_K_M/f32"

	// run returns the generated tokens and how many positions came from the store.
	run := func(cached bool) ([]int32, int) {
		s := testInstance(t, m, maxSeq, withKVPage(page))
		defer s.Close()
		s.SetKVStore(fs)
		mustKey(t, s, key)
		full := append(append([]int32{}, prompt...), tail...)
		var lg []float32
		restored := 0
		if cached {
			if lg, err = s.PrefillCached(full); err != nil {
				t.Fatal(err)
			}
			restored = s.KVRestored()
		} else {
			if lg, err = s.Prefill(full); err != nil {
				t.Fatal(err)
			}
		}
		out := make([]int32, 0, gen)
		for i := 0; i < gen; i++ {
			id := int32(argmax(lg))
			out = append(out, id)
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return out, restored
	}

	// Pass 1 fills the store. Pass 2 must find it.
	want, restored := run(false)
	if restored != 0 {
		t.Fatalf("the first pass restored %d positions from an empty store", restored)
	}
	// Content-addressed pages live one key per page, so the count is across the
	// whole store rather than under the namespace -- see FileStore.Total.
	if n, err := fs.Total(); err != nil || n == 0 {
		t.Fatalf("the first pass left %d page file(s) (err %v) -- nothing to reuse", n, err)
	}

	got, reused := run(true)
	if reused == 0 {
		t.Fatal("the second pass restored nothing, so it recomputed the prefix and this " +
			"gate tested the ordinary prefill twice")
	}
	if reused >= len(prompt)+len(tail) {
		t.Fatalf("restored %d positions of a %d-token prompt -- nothing was left to run",
			reused, len(prompt)+len(tail))
	}
	sameTokens(t, "prefix-cache", want, got)
	t.Logf("%d of %d prompt positions came from the store; tokens identical",
		reused, len(prompt)+len(tail))

	// A different key must miss: reuse keyed on nothing is reuse of anything.
	func() {
		s := testInstance(t, m, maxSeq, withKVPage(page), withKVPage(page))
		defer s.Close()
		s.SetKVStore(fs)
		mustKey(t, s, key+"-other")
		if _, err := s.PrefillCached(append(append([]int32{}, prompt...), tail...)); err != nil {
			t.Fatal(err)
		}
		if s.KVRestored() != 0 {
			t.Fatalf("a different key restored %d positions", s.KVRestored())
		}
	}()
}

// TestEvictionActuallyLowersKVBytes: eviction must lower KVBytes, and trim must
// stop at the budget. A bytes() that counted the page slice's length stayed
// flat across an eviction, so trim evicted every sealed page; token and counter
// gates cannot see that, since an over-eager evictor refaults and answers
// identically, only slower.
func TestEvictionActuallyLowersKVBytes(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const maxSeq, page = 128, 8
	s := testInstance(t, m, maxSeq, withKVPage(page))
	defer s.Close()
	s.SetKVStore(NewMemStore())

	// Reach several sealed pages per layer, so there is something to evict and
	// something to keep.
	for i := 0; i < 80; i++ {
		if _, err := s.Forward(int32(300 + i)); err != nil {
			t.Fatal(err)
		}
	}
	full := s.KVBytes()
	if full == 0 {
		t.Fatal("an 80-token session holds no KV bytes")
	}

	// Ask for roughly half and see what actually happens.
	budget := full / 2
	mustBudget(t, s, budget)
	s.kv.trim(s.pos)
	left := s.KVBytes()

	if left >= full {
		t.Fatalf("KVBytes was %d before the eviction and %d after: evicting a page did "+
			"not lower it, so nothing can ever satisfy a budget", full, left)
	}
	if left > budget {
		t.Fatalf("KVBytes is %d against a budget of %d -- trim stopped short", left, budget)
	}
	// This half catches the overshoot: a trim that sees its own progress stops
	// at the budget, so a fair fraction must still be resident.
	if left < budget/2 {
		t.Fatalf("trim left %d bytes against a budget of %d (was %d): it kept evicting "+
			"past the point it was satisfied", left, budget, full)
	}
	t.Logf("%d -> %d bytes against a budget of %d", full, left, budget)

	// The session still answers: the eviction was a budget decision, not damage.
	if _, err := s.Forward(int32(400)); err != nil {
		t.Fatal(err)
	}
}

// TestBatchSealsAgainstTheSlowestRow: with rows at different positions, no page
// a lagging row can still write may be offered to the store or evicted.
//
// A page holds every row's slots, so sealing against the high-water mark (as
// advance() once did while admitting a new row) offers a page the new row is
// still writing, and its next write re-creates it zeroed, taking the other rows'
// history with it. The assertion is on the seal point rather than on tokens,
// which diverge only when eviction and a re-fault happen to line up.
func TestBatchSealsAgainstTheSlowestRow(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const maxSeq, page, nseq = 64, 8, 2
	s := newSessBatch(m, nseq, maxSeq, withKVPage(page))
	defer s.Close()
	s.SetKVStore(NewMemStore())

	// Row 0 runs past several pages; row 1 stays at zero, as a newly admitted
	// row does.
	prompt := make([]int32, 40)
	for i := range prompt {
		prompt[i] = int32(300 + i)
	}
	if _, err := s.PrefillSeq(0, prompt); err != nil {
		t.Fatal(err)
	}
	if s.SeqPos(0) < 4*page {
		t.Fatalf("row 0 reached %d, under four %d-position pages -- there is no page for "+
			"the high-water mark to wrongly seal", s.SeqPos(0), page)
	}
	if s.SeqPos(1) != 0 {
		t.Fatalf("row 1 is at %d; this gate needs a lagging row", s.SeqPos(1))
	}

	// Nothing may be sealed at all: row 1 will write into page 0 onwards.
	stored, _, evicted := s.KVPageStats()
	if stored != 0 || evicted != 0 {
		t.Fatalf("with row 1 at position 0, %d page(s) were stored and %d evicted: "+
			"seal ran against the furthest row (%d), not the slowest",
			stored, evicted, s.SeqPos(0))
	}
	for li := range s.kv.layers {
		if n := s.kv.layers[li].lastSealed; n != 0 {
			t.Fatalf("layer %d marked %d page(s) sealed while row 1 can still write them",
				li, n)
		}
	}

	// Bring the lagging row up and the pages become sealable, so the gate
	// measures the seal point and not a store that never works.
	if _, err := s.PrefillSeq(1, prompt); err != nil {
		t.Fatal(err)
	}
	if stored, _, _ := s.KVPageStats(); stored == 0 {
		t.Fatal("no page was stored even with both rows past several pages, so the " +
			"assertion above would hold with sealing removed entirely")
	}
}

// TestPrefixCacheSurvivesTheProcess: one process writes the pages and exits, a
// second one reads them off disk. In-process, nothing forces the second reader
// to be genuinely cold; re-executing this test binary with -test.run and an
// environment variable naming the directory does.
func TestPrefixCacheSurvivesTheProcess(t *testing.T) {
	dir := os.Getenv("JITLLM_XPROC_DIR")
	if dir == "" {
		// Parent: make a directory, run the child twice, compare.
		xprocModelHere(t)
		dir = t.TempDir()
		write := runXProc(t, dir, "write")
		read := runXProc(t, dir, "read")
		if n := strings.Count(write.tokens, ",") + 1; n < 16 {
			t.Fatalf("the writer reported %d token(s) of %d: the parser truncated them and the "+
				"equality below would compare a prefix", n, 16)
		}
		if write.tokens == "" || read.tokens == "" {
			t.Fatalf("a child produced no tokens (write %q, read %q)", write.tokens, read.tokens)
		}
		if write.restored != 0 {
			t.Fatalf("the WRITING process restored %d positions from an empty directory",
				write.restored)
		}
		if write.files == 0 {
			t.Fatal("the writing process left no page files, so the second process had " +
				"nothing to find and the restore below proves nothing")
		}
		if read.restored == 0 {
			t.Fatal("the second PROCESS restored nothing: the pages did not survive, so " +
				"this proved only that one process can read its own memory")
		}
		if write.tokens != read.tokens {
			t.Fatalf("a restored prefix changed the output:\n  cold %s\n  warm %s",
				write.tokens, read.tokens)
		}
		t.Logf("process 1 wrote %d page file(s); process 2 restored %d position(s) and "+
			"generated identical tokens", write.files, read.restored)
		return
	}

	// Child: one process, one job, then exit.
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	fs, err := NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	s := testInstance(t, m, 256, withKVPage(8))
	defer s.Close()
	s.SetKVStore(fs)
	mustKey(t, s, "tinyllama/q3_K_M/xproc")

	prompt := make([]int32, 0, 63)
	for i := 0; i < 60; i++ {
		prompt = append(prompt, int32(300+i*7))
	}
	prompt = append(prompt, 338, 3681, 310)

	lg, err := s.PrefillCached(prompt)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]int32, 0, 16)
	for i := 0; i < 16; i++ {
		id := int32(argmax(lg))
		out = append(out, id)
		if lg, err = s.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	// Total(), not Pages(a session key): pages are content-addressed, one key
	// per page.
	n, err := fs.Total()
	if err != nil {
		t.Fatal(err)
	}
	// The parent parses this line; nothing else crosses the boundary. Tokens are
	// comma-separated because Sscanf's %s stops at a space.
	ids := make([]string, len(out))
	for i, id := range out {
		ids[i] = strconv.Itoa(int(id))
	}
	fmt.Printf("XPROC restored=%d files=%d tokens=%s\n", s.KVRestored(), n, strings.Join(ids, ","))
}

// mustBudget fails the test when a budget is refused, which is how a gate that
// forgot to attach a store finds out rather than silently measuring no eviction.
func mustBudget(t *testing.T, s *Instance, b uint64) {
	t.Helper()
	if err := s.SetKVBudget(b); err != nil {
		t.Fatal(err)
	}
}

func mustKey(t *testing.T, s *Instance, ns string) {
	t.Helper()
	if err := s.SetCacheKey(ns); err != nil {
		t.Fatal(err)
	}
}

type xprocResult struct {
	restored, files int
	tokens          string
}

// xprocModelHere is a parent's check for the model its children open: a child
// whose model is missing skips and prints no XPROC line, which the parent would
// otherwise report as a broken child rather than a missing model.
func xprocModelHere(t *testing.T) {
	t.Helper()
	p := testmodels.Path("tinyllama-1.1b-q3_K_M.gguf")
	if _, ok := existingModel(p); !ok {
		testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS to the model directory) -- the child processes open it", p)
	}
}

func runXProc(t *testing.T, dir, role string) xprocResult {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run", "^TestPrefixCacheSurvivesTheProcess$", "-test.v")
	cmd.Env = append(os.Environ(), "JITLLM_XPROC_DIR="+dir)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s child: %v\n%s", role, err, out)
	}
	var r xprocResult
	for _, line := range strings.Split(string(out), "\n") {
		if !strings.HasPrefix(line, "XPROC ") {
			continue
		}
		var toks string
		if _, err := fmt.Sscanf(line, "XPROC restored=%d files=%d tokens=%s",
			&r.restored, &r.files, &toks); err != nil {
			t.Fatalf("%s child: cannot parse %q: %v", role, line, err)
		}
		r.tokens = toks
	}
	if r.tokens == "" {
		t.Fatalf("%s child printed no XPROC line:\n%s", role, out)
	}
	return r
}

// TestPrefixSolverFindsTheLongestSharedPrefix: session A runs a system prompt,
// session B arrives with a different, longer prompt that begins with it, and
// nobody tells B anything. Naming a page by a hash of the tokens that produced
// it lets the store answer: a hit proves every token to that boundary matches.
// Three arms: a shared prefix is found, a divergent prompt finds only the common
// part, an unrelated prompt finds nothing.
func TestPrefixSolverFindsTheLongestSharedPrefix(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const maxSeq, page, gen = 256, 8, 12
	fs, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	const ns = "tinyllama/q3_K_M/f32"

	system := make([]int32, 48) // the shared "system prompt"
	for i := range system {
		system[i] = int32(300 + i*7)
	}
	with := func(tail ...int32) []int32 {
		return append(append([]int32{}, system...), tail...)
	}

	// run returns the generated tokens and how many positions the solver found.
	run := func(prompt []int32, useNS bool) ([]int32, int) {
		s := testInstance(t, m, maxSeq, withKVPage(page))
		defer s.Close()
		s.SetKVStore(fs)
		if useNS {
			mustKey(t, s, ns)
		}
		lg, err := s.PrefillCached(prompt)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]int32, 0, gen)
		for i := 0; i < gen; i++ {
			id := int32(argmax(lg))
			out = append(out, id)
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return out, s.KVRestored()
	}

	// Session A fills the store. (run returns KVRestored() on both arms, so the
	// zero below is a measurement: kvSkipped is written only by PrefillCached.)
	if _, n := run(with(1, 2, 3), true); n != 0 {
		t.Fatalf("the first session solved %d positions from an empty store", n)
	}

	// Session B: a different prompt sharing the system prefix.
	wantB, _ := run(with(9, 9, 9, 9), false) // the cold control, no namespace
	gotB, solved := run(with(9, 9, 9, 9), true)
	if solved == 0 {
		t.Fatal("the solver found nothing for a prompt sharing 48 tokens of prefix: it " +
			"is still matching whole sessions rather than prefixes")
	}
	if solved > len(system) {
		t.Fatalf("the solver claimed %d positions of a %d-token shared prefix -- it "+
			"reused pages past the point the prompts agree", solved, len(system))
	}
	sameTokens(t, "shared-prefix", wantB, gotB)
	t.Logf("a prompt nobody registered reused %d of its %d shared prefix positions",
		solved, len(system))

	// A prompt that shares nothing must find nothing: the solver matches
	// content, not whatever the store holds.
	other := make([]int32, 48)
	for i := range other {
		other[i] = int32(2000 + i*3)
	}
	if _, n := run(other, true); n != 0 {
		t.Fatalf("an unrelated prompt solved %d positions", n)
	}
}

// TestDefaultCacheIDIsNotGuessableAcrossProcesses: a session with NO namespace
// must not address another process's pages.
//
// A per-process counter ("session-1") made a second process's first instance find
// the first process's pages on disk, mixing two histories silently.
func TestDefaultCacheIDIsNotGuessableAcrossProcesses(t *testing.T) {
	dir := os.Getenv("JITLLM_XPROC_ANON")
	if dir == "" {
		xprocModelHere(t)
		dir = t.TempDir()
		a := runXProcAnon(t, dir)
		b := runXProcAnon(t, dir)
		if a.files == 0 {
			t.Fatal("the first process stored nothing, so the second could not have " +
				"collided with it and this gate proves nothing")
		}
		if b.restored != 0 {
			t.Fatalf("a second process with NO namespace restored %d position(s) of "+
				"another process's history", b.restored)
		}
		t.Logf("process 1 stored %d page file(s); process 2 with no namespace restored %d",
			a.files, b.restored)
		return
	}

	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	fs, err := NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	s := testInstance(t, m, 128, withKVPage(8))
	defer s.Close()
	s.SetKVStore(fs) // and deliberately NO SetCacheKey
	prompt := make([]int32, 40)
	for i := range prompt {
		prompt[i] = int32(300 + i)
	}
	if _, err := s.PrefillCached(prompt); err != nil {
		t.Fatal(err)
	}
	n, err := fs.Total()
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("XPROC restored=%d files=%d tokens=[]\n", s.KVRestored(), n)
}

func runXProcAnon(t *testing.T, dir string) xprocResult {
	t.Helper()
	cmd := exec.Command(os.Args[0],
		"-test.run", "^TestDefaultCacheIDIsNotGuessableAcrossProcesses$", "-test.v")
	cmd.Env = append(os.Environ(), "JITLLM_XPROC_ANON="+dir)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("child: %v\n%s", err, out)
	}
	var r xprocResult
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, "XPROC ") {
			var toks string
			fmt.Sscanf(line, "XPROC restored=%d files=%d tokens=%s", &r.restored, &r.files, &toks)
			r.tokens = "x"
		}
	}
	if r.tokens == "" {
		t.Fatalf("child printed no XPROC line:\n%s", out)
	}
	return r
}

// TestNamespaceSurvivesTheF16StepDown: naming a namespace and THEN attaching a
// device must not silently disable the cache.
//
// The f16 step-down rebuilds the cache with newKVCache, and the rebuild once
// dropped ns, so pages went to a name no later run could find.
func TestNamespaceSurvivesTheF16StepDown(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	s := testInstance(t, m, 128, withKVPage(8))
	defer s.Close()
	// A budget needs somewhere to spill, and the budget is half of what must
	// survive the rebuild.
	s.SetKVStore(NewMemStore())
	mustKey(t, s, "tinyllama/q3_K_M/stepdown")

	// The real path: stepDownKVToF32 is the body SetDeviceLayers runs, extracted
	// so a gate can reach it without a card.
	mustBudget(t, s, 1<<20)
	s.stepDownKVToF32()
	if got := s.CacheNamespace(); got != "tinyllama/q3_K_M/stepdown" {
		t.Fatalf("the namespace is %q after the step-down: pages would go to the "+
			"process-local session id and no later run could find them", got)
	}
	if s.kv.budget != 1<<20 {
		t.Fatalf("the KV budget is %d after the step-down: SetKVBudget reported success "+
			"and trim would never evict for the life of the session", s.kv.budget)
	}
	// And it still names pages, which is what the namespace exists for.
	s.kv.seq, s.kv.layers[0].p = []int32{1, 2, 3, 4, 5, 6, 7, 8}, 8
	if k := s.kv.pageKey(0, 0); k == "" || !strings.HasPrefix(k, "tinyllama/q3_K_M/stepdown/") {
		t.Fatalf("page key after the rebuild is %q", k)
	}
}

// TestBatchRefusesACacheNamespace: a page holds every row's slots, so no single
// sequence's tokens can name it.
func TestBatchRefusesACacheNamespace(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	b := newSessBatch(m, 2, 32)
	defer b.Close()
	if err := b.SetCacheKey("tinyllama/q3_K_M/f32"); err == nil {
		t.Fatal("a 2-row batch accepted a cache namespace: it would fill a store with " +
			"pages named after row 0's tokens that hold every row's slots, and which " +
			"PrefillCached refuses to read back")
	}
	// A single-sequence instance still accepts one, or the refusal is not about
	// batching.
	s := testInstance(t, m, 32)
	defer s.Close()
	if err := s.SetCacheKey("tinyllama/q3_K_M/f32"); err != nil {
		t.Fatalf("a single-sequence instance was refused a namespace: %v", err)
	}
}

// TestSealDoesNotClaimPagesItCouldNotStore: the seal mark means stored, not
// walked past. A placed block's host pages are never written, so advancing the
// mark past them stored nothing and a later pull-down from the card found the
// mark already past. trim treats everything below the mark as evictable, so a
// page never stored must never be a candidate.
func TestSealDoesNotClaimPagesItCouldNotStore(t *testing.T) {
	store := NewMemStore()
	kc := &kvCache{id: "s", store: store, bh: map[int][]string{},
		layers: []kvPages{{p: 4, pp: 8}}}
	pg := &kc.layers[0]

	// Pages 0 and 1 exist; page 2 was never written, as a device layer's are not.
	pg.mem = heldPages(8, true, true, false, true)
	pg.written = 8 // positions 0..7, i.e. pages 0 and 1, are genuinely full

	kc.seal(16) // cur = 4, so it would like to seal pages 0..3
	if pg.lastSealed != 2 {
		t.Fatalf("lastSealed is %d after a gap at page 2: seal claimed pages it did not "+
			"store, so trim may evict the only copy and a later seal will skip them",
			pg.lastSealed)
	}
	if kc.stored != 4 { // two pages x (k and v)
		t.Fatalf("stored %d entries, want 4 (pages 0 and 1, k and v)", kc.stored)
	}

	// Fill the gap -- which is what a pull-down from the card does -- and the
	// mark must now advance past it.
	pg.mem.Set(2, make([]float32, 8), make([]float32, 8))
	pg.written = 16
	kc.seal(16)
	if pg.lastSealed != 4 {
		t.Fatalf("lastSealed is %d after the gap was filled: the pages a device handed "+
			"back can never be stored", pg.lastSealed)
	}
	if kc.stored != 8 {
		t.Fatalf("stored %d entries after the gap was filled, want 8", kc.stored)
	}

	// A page resident but only partly written is not sealed: scatter allocates
	// up to the page holding pos-1, so the page straddling the end of a run has
	// a tail of zeros, and sealing it stores zeros under a key claiming them.
	kc2 := &kvCache{id: "s", store: NewMemStore(), bh: map[int][]string{},
		layers: []kvPages{{p: 4, pp: 8}}}
	p2 := &kc2.layers[0]
	p2.mem = heldPages(8, true, true)
	p2.written = 6 // page 1 holds positions 4..7 and only 4 and 5 were written
	kc2.seal(8)    // cur = 2, so it would like to seal pages 0 and 1
	if p2.lastSealed != 1 {
		t.Fatalf("lastSealed is %d with only 6 of 8 positions written: a page whose tail "+
			"is zeros was stored under a key asserting it holds them", p2.lastSealed)
	}

	// The mark falls on a rollback, or trim treats a page the write path is
	// about to use as a candidate.
	kc.seal(4)
	if pg.lastSealed != 1 {
		t.Fatalf("lastSealed is %d after the position rolled back to 4", pg.lastSealed)
	}
}

// mirrorDevice is a LayerDevice that keeps its own KV and hands it back: enough
// to test that the engine pulls the history down and seals it, without a card.
type mirrorDevice struct {
	kv       map[int][]float32
	refuse   bool // MigrateKV(toDevice=false) fails, as the f16 V cache does
	corrupt  bool // hand back a history that is not what was given
	migrated int
}

func newMirrorDevice() *mirrorDevice {
	return &mirrorDevice{kv: map[int][]float32{}}
}

func (d *mirrorDevice) MatVec([]float32, quant.Type, []byte, []float32, int, int) bool {
	return false
}
func (d *mirrorDevice) PrepLayer(int, *nn.LayerPlan, *nn.LayerWeights) bool { return true }
func (d *mirrorDevice) Layers(int, int, int, int, []float32, []float32, []float32, *nn.Head) bool {
	return false // decline the block: the host computes, which is what we compare against
}
func (d *mirrorDevice) PrepHead(*nn.Head) bool { return false }
func (d *mirrorDevice) HeadResident() bool     { return false }
func (d *mirrorDevice) MigrateKV(li int, k, v []float32, pos int, toDevice bool) bool {
	if pos <= 0 {
		return true
	}
	d.migrated++
	if toDevice {
		d.kv[li*2] = append([]float32(nil), k...)
		d.kv[li*2+1] = append([]float32(nil), v...)
		return true
	}
	if d.refuse {
		return false
	}
	copy(k, d.kv[li*2])
	copy(v, d.kv[li*2+1])
	if d.corrupt {
		for i := range k {
			k[i] *= 1.5
		}
		// And v, negated and scaled: a model whose scores barely move with
		// k's scale (Gemma 4's fixture, whose normed k and unit scale keep
		// the softmax near flat over a short history) still reads every
		// value. Not a negation alone, which is its own inverse: a block
		// placeHead gives back for the head's room comes home and goes up
		// again, so a history can cross twice before the gate reads it.
		for i := range v {
			v[i] *= -1.5
		}
	}
	return true
}

// MigrateRec: this fake keeps no recurrent state, so nothing to move is success.
func (d *mirrorDevice) MigrateRec(int, []float32, []float32, bool) bool { return true }
func (d *mirrorDevice) ReserveKV(int) bool                              { return true }
func (d *mirrorDevice) ReleaseLayers(int, int)                          {}
func (d *mirrorDevice) Reserve(int, int) bool                           { return true }
func (d *mirrorDevice) PrewarmLayer(int, *nn.LayerPlan, *nn.LayerWeights) bool {
	return false
}

// TestDeviceHistoryIsPulledDownAndSealed covers syncKVFromDevice. attnPrep runs
// only over the host block range, so a placed block writes no host page and seal
// would store nothing while every counter reports success.
func TestDeviceHistoryIsPulledDownAndSealed(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	// syncKVFromDevice is driven directly: a fake that declines Layers makes the
	// engine write host pages itself and never call MigrateKV, and one that
	// accepts Layers would have to compute the block.
	run := func(refuse bool) (files int, syncFails int64, migrated int) {
		fs, err := NewFileStore(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		s := testInstance(t, m, 128, withKVPage(8))
		defer s.Close()
		s.SetKVStore(fs)
		mustKey(t, s, "tinyllama/q3_K_M/devsync")

		// A sequence the device holds and the host does not: what a placed
		// block leaves behind.
		dev := newMirrorDevice()
		dev.refuse = refuse
		const pos = 40
		kvDim := s.kvl.slots(pos * s.kvl.kvDim())
		for li := 0; li < 4; li++ {
			k, v := make([]float32, kvDim), make([]float32, kvDim)
			for i := range k {
				k[i], v[i] = float32(li*1000+i), float32(-(li*1000 + i))
			}
			dev.kv[li*2], dev.kv[li*2+1] = k, v
		}
		// The placement is the set the instance marks, not a count.
		s.setLD(dev)
		for li := 0; li < 4; li++ {
			s.markOnDev(li)
		}
		s.kv.seq = make([]int32, pos)
		for i := range s.kv.seq {
			s.kv.seq[i] = int32(300 + i)
		}
		s.pos, s.bpos[0] = pos, pos

		if err := s.syncKVFromDevice(); err != nil {
			t.Fatal(err)
		}
		n, err := fs.Total()
		if err != nil {
			t.Fatal(err)
		}
		return n, s.KVDeviceSyncFailures(), dev.migrated
	}

	files, fails, migrated := run(false)
	if migrated == 0 {
		t.Fatal("MigrateKV was never called, so nothing about the device path was tested")
	}
	if files == 0 {
		t.Fatal("nothing was stored: a placed block's history was never pulled down, so " +
			"the cache is silently dead on every offloaded run")
	}
	if fails != 0 {
		t.Fatalf("%d sync failure(s) on a device that hands its history back", fails)
	}
	t.Logf("%d page file(s) stored from 4 placed block(s), %d migration(s)", files, migrated)

	// A device that refuses is counted, not swallowed: the f16 V cache cannot be
	// copied as float32, and that must read as "not cached", not a failing store.
	files, fails, _ = run(true)
	if fails == 0 {
		t.Fatal("a device that refused to hand its history back was not counted, so an " +
			"operator sees an empty cache with no reason")
	}
	if files != 0 {
		t.Fatalf("%d page file(s) stored although the device refused: zeros were sealed "+
			"as history", files)
	}
}

// TestHybridReusesBothHalvesOfItsHistory: a hybrid's past lives in two places
// and prefix reuse must carry both.
//
// A gated delta net keeps a running summary instead of a per-position cache, so
// restoring attention pages alone resumes the delta blocks from a summary of
// nothing: fluent and wrong. The summary is a function of the tokens the page
// key names and constant in context length, so it is stored too. The third arm:
// with the recurrent half missing the prefix must be given up, not used.
func TestHybridReusesBothHalvesOfItsHistory(t *testing.T) {
	m, openOpts := hybridModel(t)
	defer m.Close()
	if !m.Cfg.Hybrid() {
		t.Fatal("the fixture is not a hybrid, so this gate is about nothing")
	}

	const maxSeq, page, gen = 256, 8, 10
	prompt := make([]int32, 0, 40)
	for i := 0; i < 40; i++ {
		prompt = append(prompt, int32(3+i%7))
	}

	run := func(store KVStore, ns string) ([]int32, int) {
		s := testInstance(t, m, maxSeq, append(append([]Option(nil), openOpts...), withKVPage(page))...)
		defer s.Close()
		if store != nil {
			s.SetKVStore(store)
			if err := s.SetCacheKey(ns); err != nil {
				t.Fatal(err)
			}
		}
		lg, err := s.PrefillCached(prompt)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]int32, 0, gen)
		for i := 0; i < gen; i++ {
			id := int32(argmax(lg))
			out = append(out, id)
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return out, s.KVRestored()
	}

	want, _ := run(nil, "")

	st := NewMemStore()
	if _, n := run(st, "hy/v1"); n != 0 {
		t.Fatalf("the first pass restored %d positions from an empty store", n)
	}
	got, reused := run(st, "hy/v1")
	if reused == 0 {
		t.Fatal("a hybrid reused nothing: its recurrent half was never stored, so prefix " +
			"reuse is still refused for every architecture that has one")
	}
	sameTokens(t, "hybrid-prefix", want, got)
	t.Logf("a hybrid reused %d of %d prompt positions, both halves, tokens identical",
		reused, len(prompt))

	// Without the recurrent half the prefix is given up. dropRecurrentStore
	// serves the KV pages and refuses the summaries, like a cache written before
	// summaries were stored.
	half := &dropRecurrentStore{MemStore: st, from: 2 * m.Cfg.NLayer}
	if out, n := run(half, "hy/v1"); n != 0 {
		t.Fatalf("a hybrid reused %d positions with its summaries missing: the delta "+
			"blocks resumed from a summary of nothing (tokens %v)", n, out)
	}
}

// dropRecurrentStore serves pages and refuses every recurrent snapshot.
type dropRecurrentStore struct {
	*MemStore
	from int
}

func (d *dropRecurrentStore) Get(cacheId string, layer, index int, page io.Writer) error {
	if layer >= d.from {
		return ErrNoPage
	}
	return d.MemStore.Get(cacheId, layer, index, page)
}

// TestFileStoreStaysInsideItsLimit: a bounded store must evict, stay under, and
// still answer for what it kept.
//
// Nothing here expires, so without a cap the only thing that removes a page is
// a person noticing a full disk.
func TestFileStoreStaysInsideItsLimit(t *testing.T) {
	dir := t.TempDir()
	const page, limit = 4096, 40 * 4096
	fs, err := NewFileStoreLimit(dir, limit)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, page)
	for i := 0; i < 200; i++ {
		if err := fs.Set(fmt.Sprintf("ns/%d", i), 0, 0, bytes.NewReader(buf)); err != nil {
			t.Fatal(err)
		}
	}
	n, err := fs.Total()
	if err != nil {
		t.Fatal(err)
	}
	if n == 0 {
		t.Fatal("the store is empty after 200 writes")
	}
	if uint64(n)*page > limit {
		t.Fatalf("%d page(s) of %d bytes is %d, over the %d-byte limit: the cap does nothing",
			n, page, uint64(n)*page, limit)
	}
	if fs.Evicted() == 0 {
		t.Fatal("nothing was evicted although 200 pages were written into a 40-page store, " +
			"so the assertion above would hold with the limit ignored")
	}
	t.Logf("200 writes into a %d-page store left %d page(s), %d evicted",
		limit/page, n, fs.Evicted())

	// What survived is the most recent, which makes it a cache rather than a
	// random subset.
	w := newSliceWriter(make([]byte, page))
	if err := fs.Get("ns/199", 0, 0, w); err != nil {
		t.Fatalf("the most recently written page was evicted: %v", err)
	}

	// An unbounded store keeps everything, or the arms above are not about the
	// limit.
	un, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 60; i++ {
		if err := un.Set(fmt.Sprintf("ns/%d", i), 0, 0, bytes.NewReader(buf)); err != nil {
			t.Fatal(err)
		}
	}
	if n, _ := un.Total(); n != 60 {
		t.Fatalf("an unbounded store holds %d of 60 pages", n)
	}
}

// TestKVPagesRelocateWithTheLayer: a block that changes tiers mid-conversation
// takes its history with it, and the tokens do not move.
//
// The failure is not a crash: a block whose weights moved and whose keys and
// values did not attends over an empty cache and answers fluently. It runs
// anywhere: the fake carries the history rather than computing with it, since
// what is under test is the gather/migrate/scatter seam in model/.
func TestKVPagesRelocateWithTheLayer(t *testing.T) {
	w := fusedRunsCheck(t, true)
	defer w.done()
	for _, name := range relocModels() {
		for _, c := range []struct {
			name string
			kt   model.KVType
		}{{"f32", model.KVF32}, {"f16", model.KVF16}} {
			t.Run(name+"/"+c.name, func(t *testing.T) { kvPagesRelocateWithTheLayer(t, jlmOf(t, testmodels.Path(name)), c.kt, w) })
		}
	}
}

// kvPagesRelocateWithTheLayer is TestKVPagesRelocateWithTheLayer on one model.
func kvPagesRelocateWithTheLayer(t *testing.T, path string, kt model.KVType, w *fuseWatch) {
	// Untuned: the logit sum below is held bit for bit, and a kernel pick
	// timed mid-run (the arm64 per-shape choice) moves the reduction order
	// between the two arms.
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.Cfg.KVTypeRefusal(kt); err != nil {
		t.Skipf("refused by name: %v", err)
	}
	// A model every block of which is recurrent (plain Mamba-2) keeps no
	// attention history at all: there is no page to carry, and its summary's
	// relocation is TestBatchSeamMovesCarryEveryRowSSM's and
	// TestLayerRelocatesToAnotherDevice's.
	attends := false
	for li := 0; li < m.Cfg.NLayer; li++ {
		attends = attends || m.Cfg.LayerKind(li).Attends()
	}
	if !attends {
		t.Skip("every block is recurrent: no KV page exists to relocate")
	}

	const maxSeq, page, gen = 128, 8, 12
	warm := 40
	prompt := make([]int32, warm)
	for i := range prompt {
		prompt[i] = int32(300 + i)
	}

	// move==false is the control: the same session, never relocated.
	// sum is every logit of the generation, summed: tokens alone cannot see a
	// history whose model's argmax rides the input token (Gemma 4's fixture,
	// a tied head behind a large embedding scale), where its logits move.
	var sum float64
	run := func(move, corrupt bool) ([]int32, int) {
		sum = 0
		// A binary16 history travels converted, widened on its way out and
		// narrowed home through the CPU device's composed conversions
		// (migrateKVAs): exact both ways, so the logits are held bit for bit
		// at every width.
		s := testInstance(t, m, maxSeq, WithKVType(kt), withKVPage(page), w.opt())
		defer s.Close()
		if s.KVType() != kt {
			t.Fatalf("asked for a %v cache and got %v", kt, s.KVType())
		}
		var lg []float32
		for _, id := range prompt {
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		// The longest history of any layer: a sliding layer (synth-exaone4's
		// window of 4) keeps a ring of pages, and a global one carries the rest.
		most := 0
		for i := range s.kv.layers {
			most = max(most, s.kv.layers[i].mem.Len())
		}
		if most < 3 {
			t.Fatalf("only %d page(s) after %d tokens: a single-page history cannot show "+
				"that relocation carries the pages", most, warm)
		}
		dev := newMirrorDevice()
		dev.corrupt = corrupt
		moved := 0
		if move {
			// Up gathers the host pages and hands them over; down hands them
			// back and scatters them in, both at a live position.
			s.SetDeviceLayers(dev, 4)
			if s.gpuLayers == 0 {
				t.Fatalf("no block was placed, so nothing relocated and this gate is "+
					"vacuous (kvF16=%v)", s.KVIsF16())
			}
			// Straight back down with nothing run in between, so what returns
			// must be exactly what left.
			s.SetGPULayers(0)
			moved = dev.migrated
			if moved == 0 {
				t.Fatal("MigrateKV was never called, so the history did not travel")
			}
			if kt == model.KVF16 && kvRan(s) == 0 {
				t.Fatal("the binary16 history travelled without the CPU device's composed conversions")
			}
		}
		out := make([]int32, 0, gen)
		for i := 0; i < gen; i++ {
			id := int32(argmax(lg))
			out = append(out, id)
			for _, v := range lg {
				sum += float64(v)
			}
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return out, moved
	}

	want, _ := run(false, false)
	wantSum := sum
	got, moved := run(true, false)
	sameTokens(t, "relocate", want, got)
	if sum != wantSum {
		t.Fatalf("relocate: the logits moved (sum %v against %v) with the tokens identical", sum, wantSum)
	}
	t.Logf("%d block migration(s) over a %d-page history; tokens and logits identical", moved, warm/page)

	// And the gate is sensitive: if the history comes back wrong the tokens
	// (or, where the argmax rides the input, the logits) must move, or this
	// asserts that relocation is ignored, not lossless.
	bad, _ := run(true, true)
	if equalTokens(want, bad) && sum == wantSum {
		t.Fatal("a corrupted history produced the same tokens: the relocated pages are " +
			"not what attention reads, so this gate proves nothing about them")
	}
}

func equalTokens(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestKVBudgetWithoutAStoreIsRefused: a cap with nowhere to spill must say so.
//
// trim only evicts a page the store already holds, and with NoStore nothing is
// ever sealed, so a cap without a store would silently cap nothing.
func TestKVBudgetWithoutAStoreIsRefused(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	// The page size is read when the cache is built, so it must be in effect
	// before NewInstance or nothing ever seals.
	s := testInstance(t, m, 128, withKVPage(8))
	defer s.Close()

	if err := s.SetKVBudget(1 << 20); err == nil {
		t.Fatal("a KV budget was accepted with no store: it would cap nothing, evict " +
			"nothing and report nothing")
	}
	// Zero is "no cap" and must stay free of the requirement.
	if err := s.SetKVBudget(0); err != nil {
		t.Fatalf("clearing the budget was refused: %v", err)
	}
	// With a store it is accepted and it works.
	s.SetKVStore(NewMemStore())
	if err := s.SetKVBudget(1 << 14); err != nil {
		t.Fatalf("a budget with a store was refused: %v", err)
	}
	for i := 0; i < 60; i++ {
		if _, err := s.Forward(int32(300 + i)); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, ev := s.KVPageStats(); ev == 0 {
		t.Fatal("nothing was evicted under a budget a 60-token session exceeds")
	}
}

// TestPagedKVIsTokenIdenticalOnEveryModel sweeps every container: mixtures,
// grouped-query geometries, a vision-carrying container and a hybrid reach
// cases tinyllama cannot. The control is the same binary at P >= maxSeq (one
// page, the flat cache) against P = 8, so any difference is the page walk.
func TestPagedKVIsTokenIdenticalOnEveryModel(t *testing.T) { pagedKVIsTokenIdenticalOnEveryModel(t) }

// pagedKVIsTokenIdenticalOnEveryModel is TestPagedKVIsTokenIdenticalOnEveryModel with opts on every model it opens.
func pagedKVIsTokenIdenticalOnEveryModel(t *testing.T, opts ...Option) {
	if testing.Short() {
		t.Skip("opens every model on the box")
	}
	// Containers, not GGUFs: the container is what the engine reads, and some
	// models exist only as one.
	paths := testmodels.Glob("*.jlm")
	if len(paths) == 0 {
		t.Skip("no containers")
	}
	pick := os.Getenv("JITLLM_KVSWEEP_MODEL")
	const maxSeq, gen = 96, 12
	// Long enough to cross several page boundaries; the page count is asserted
	// below so a model reaching one page cannot pass.
	prompt := make([]int32, 0, 40)
	for i := 0; i < 40; i++ {
		prompt = append(prompt, int32(2+i%6))
	}

	// A container this build cannot read is excluded rather than skipped (jlm has
	// no backward compatibility, so it cannot be opened at all), and named so it
	// stays visible.
	var stale []string
	var declined []string
	var usable []string
	for _, p := range paths {
		name := filepath.Base(p)
		if strings.Contains(name, "mmproj") {
			continue
		}
		m, err := model.Open(p)
		if err != nil && strings.Contains(err.Error(), "this build reads") {
			// A stale container whose GGUF remains is converted again, as
			// every other gate's jlmOf does: excluding it ran the sweep on
			// none of the models after a format bump.
			// In a subtest of its own, so a conversion this build declines
			// skips that model and not the sweep.
			if src := strings.TrimSuffix(p, jlm.Ext) + ".gguf"; fileExists(src) {
				fresh := ""
				t.Run("convert/"+name, func(t *testing.T) { fresh = jlmOf(t, src) })
				if fresh != "" {
					p = fresh
					m, err = model.Open(p)
				}
			}
		}
		if err != nil {
			if strings.Contains(err.Error(), "this build reads") {
				stale = append(stale, name)
				continue
			}
			// A current container whose architecture is not implemented yet:
			// excluded by name and counted (see declinedRefusal).
			if declinedRefusal(err) {
				declined = append(declined, name)
				continue
			}
			t.Fatalf("%s: %v", name, err)
		} else {
			m.Close()
		}
		usable = append(usable, p)
	}
	if len(stale) > 0 {
		t.Logf("%d container(s) predate this build's format and were excluded (no GGUF "+
			"source remains for them): %s", len(stale), strings.Join(stale, ", "))
	}
	if len(declined) > 0 {
		t.Logf("%d container(s) convert and are declined by name because their architecture "+
			"is not implemented yet: %s", len(declined), strings.Join(declined, ", "))
	}
	if len(usable) == 0 {
		t.Fatal("no readable container on this box")
	}

	ran := 0
	for _, p := range usable {
		name := filepath.Base(p)
		if pick != "" && !strings.Contains(name, pick) {
			continue
		}
		fi, err := os.Stat(p)
		if err != nil || (pick == "" && fi.Size() > 2<<30) {
			continue // the big ones need JITLLM_KVSWEEP_MODEL, one at a time
		}
		t.Run(name, func(t *testing.T) {
			instOpts3 := opts
			m, err := model.Open(p)
			if err != nil {
				t.Fatalf("open: %v", err)
			}
			defer m.Close()

			run := func(page int) []int32 {
				s := testInstance(t, m, maxSeq, append(append([]Option(nil), instOpts3...), withKVPage(page))...)
				defer s.Close()
				var lg []float32
				for _, id := range prompt {
					if int(id) >= m.Cfg.NVocab {
						t.Skipf("prompt does not fit this vocabulary (%d)", m.Cfg.NVocab)
					}
					if lg, err = s.Forward(id); err != nil {
						t.Fatal(err)
					}
				}
				out := make([]int32, 0, gen)
				for i := 0; i < gen; i++ {
					id := int32(argmax(lg))
					out = append(out, id)
					if lg, err = s.Forward(id); err != nil {
						t.Fatal(err)
					}
				}
				return out
			}

			whole := run(maxSeq) // one page: the flat cache paging replaced
			paged := run(8)      // ceil(n/8) pages over the same window

			// The page count is asserted, or a model that collapses to one page
			// (a linear layer keeps none; a short context clamps the size) passes
			// while testing nothing.
			func() {
				s := testInstance(t, m, maxSeq, append(append([]Option(nil), instOpts3...), withKVPage(8))...)
				defer s.Close()
				for _, id := range prompt {
					if _, err := s.Forward(id); err != nil {
						t.Fatal(err)
					}
				}
				most := 0
				for li := range s.kv.layers {
					if n := s.kv.layers[li].mem.Len(); n > most {
						most = n
					}
				}
				if most < 2 {
					t.Skipf("no layer reaches more than %d page(s); this model cannot show "+
						"a page walk at this depth", most)
				}
			}()
			sameTokens(t, "paged-vs-one-page", whole, paged)
			ran++
		})
	}
	if ran == 0 {
		t.Fatal("no model was compared: the sweep proved nothing")
	}
}

// TestPrefixCacheHoldsAPromptShorterThanAPage is the gate on a chat turn, a
// prompt that never reaches a page boundary. sealTail stores the page the
// position is inside, named by how many positions it holds, and restore probes
// for it. The assertion that matters is that the tokens match an uncached run.
func TestPrefixCacheHoldsAPromptShorterThanAPage(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const maxSeq, gen, page = 256, 12, 64
	// Shorter than one page, so only the tail can carry it.
	prompt := make([]int32, 0, 20)
	for i := 0; i < 20; i++ {
		prompt = append(prompt, int32(300+i*7))
	}
	fs, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	run := func() ([]int32, int) {
		s := testInstance(t, m, maxSeq, withKVPage(page))
		defer s.Close()
		s.SetKVStore(fs)
		mustKey(t, s, "short/prompt")
		lg, err := s.PrefillCached(prompt)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]int32, 0, gen)
		for i := 0; i < gen; i++ {
			id := int32(argmax(lg))
			out = append(out, id)
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return out, s.KVRestored()
	}

	want, restored := run()
	if restored != 0 {
		t.Fatalf("the first pass restored %d positions from an empty store", restored)
	}
	// The store has to hold something, or the second pass just prefills again.
	if n, err := fs.Total(); err != nil || n == 0 {
		t.Fatalf("a %d-token prompt left %d page file(s) (err %v): nothing was "+
			"sealed, so a prompt shorter than a page still caches nothing",
			len(prompt), n, err)
	}

	got, reused := run()
	if reused == 0 {
		held, _ := fs.Total()
		t.Fatalf("the second pass restored nothing of a %d-token prompt with %d "+
			"page file(s) in the store", len(prompt), held)
	}
	// The whole prompt is a good answer: sealLogits stores the row the prompt
	// ended on, so a fully cached prompt runs nothing (a hybrid cannot shorten
	// its summary by one), and the tokens check it is the right nothing.
	if reused > len(prompt) {
		t.Fatalf("restored %d positions of a %d-token prompt", reused, len(prompt))
	}
	sameTokens(t, "short-prompt-cache", want, got)
	t.Logf("%d of %d positions came from a PARTIAL page; tokens identical",
		reused, len(prompt))
}

// TestPrefixCacheReusesASharedPrefix is the case a prefix cache is for: two
// prompts sharing a prefix and then diverging. Reuse is addressed at kvChunk
// positions, independent of the compute page (kvPageTarget), as vLLM, SGLang
// and LMCache also keep them apart. The assertion is the tokens.
func TestPrefixCacheReusesASharedPrefix(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const maxSeq, gen, page = 256, 10, 128
	// Two prompts that share a prefix and then diverge (the chat shape).
	// Neither reaches a page boundary.
	shared := make([]int32, 0, 40)
	for i := 0; i < 40; i++ {
		shared = append(shared, int32(300+i*7))
	}
	first := append(append([]int32{}, shared...), 611, 612, 613)
	second := append(append([]int32{}, shared...), 907, 908, 909, 910, 911)

	fs, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	run := func(p []int32, cached bool) ([]int32, int) {
		s := testInstance(t, m, maxSeq, withKVPage(page))
		defer s.Close()
		if cached {
			s.SetKVStore(fs)
			mustKey(t, s, "shared/prefix")
		}
		lg, err := s.PrefillCached(p)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]int32, 0, gen)
		for i := 0; i < gen; i++ {
			id := int32(argmax(lg))
			out = append(out, id)
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return out, s.KVRestored()
	}

	// The truth for the second prompt, computed with no store at all.
	want, _ := run(second, false)

	// Turn one fills the cache; turn two must find the shared part of it.
	if _, n := run(first, true); n != 0 {
		t.Fatalf("the first pass restored %d positions from an empty store", n)
	}
	got, reused := run(second, true)
	if reused == 0 {
		t.Fatalf("a %d-token prompt sharing %d tokens with the cached one reused "+
			"NOTHING: the cache is addressing at the page (%d) rather than at a "+
			"chunk (%d)", len(second), len(shared), page, kvChunk)
	}
	if reused > len(shared) {
		t.Fatalf("restored %d positions of a %d-token shared prefix: the cache "+
			"served positions the two prompts do not agree on", reused, len(shared))
	}
	sameTokens(t, "shared-prefix", want, got)
	t.Logf("%d of %d shared positions reused at chunk %d; tokens identical to an "+
		"uncached run", reused, len(shared), kvChunk)
}

// fillCountingStore is a MemStore that counts the SetFills it is offered, so a
// test can say the shared-copy path ran.
type fillCountingStore struct {
	*MemStore
	fills, keys int
}

func (c *fillCountingStore) SetFills(layer, index int, src []byte, keys []string, lens []int) error {
	c.fills++
	c.keys += len(keys)
	return c.MemStore.SetFills(layer, index, src, keys, lens)
}

// TestMemStoreFillsServeASharedPrefix is TestPrefixCacheReusesASharedPrefix on
// a MemStore, the store a server keeps: the partial last page goes in as fills
// over one copy (SetFills), and a second prompt that shares part of it, and
// the identical prompt again, restore from those fills with the tokens an
// uncached run gives.
func TestMemStoreFillsServeASharedPrefix(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	const maxSeq, gen, page = 256, 10, 128
	shared := make([]int32, 0, 40)
	for i := 0; i < 40; i++ {
		shared = append(shared, int32(300+i*7))
	}
	first := append(append([]int32{}, shared...), 611, 612, 613)
	second := append(append([]int32{}, shared...), 907, 908, 909, 910, 911)

	ms := NewMemStore()
	defer ms.Close()
	store := &fillCountingStore{MemStore: ms}
	run := func(p []int32, cached bool) ([]int32, int) {
		s := testInstance(t, m, maxSeq, withKVPage(page))
		defer s.Close()
		if cached {
			s.SetKVStore(store)
			mustKey(t, s, "fills/prefix")
		}
		lg, err := s.PrefillCached(p)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]int32, 0, gen)
		for i := 0; i < gen; i++ {
			id := int32(argmax(lg))
			out = append(out, id)
			if lg, err = s.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return out, s.KVRestored()
	}

	wantSecond, _ := run(second, false)
	wantFirst, _ := run(first, false)

	if _, n := run(first, true); n != 0 {
		t.Fatalf("the first pass restored %d positions from an empty store", n)
	}
	if store.fills == 0 || store.keys <= store.fills {
		t.Fatalf("the partial page went to the store as %d SetFills call(s) of %d key(s): "+
			"the shared-copy path did not run, or carried one fill", store.fills, store.keys)
	}
	got, reused := run(second, true)
	if reused == 0 || reused > len(shared) {
		t.Fatalf("a prompt sharing %d tokens restored %d positions from the fills", len(shared), reused)
	}
	sameTokens(t, "fills shared-prefix", wantSecond, got)
	again, whole := run(first, true)
	if whole == 0 {
		t.Fatal("the identical prompt restored nothing from its exact-end fill")
	}
	sameTokens(t, "fills identical-prompt", wantFirst, again)
	t.Logf("%d SetFills of %d keys; %d of %d shared positions and %d of %d identical-prompt positions restored; tokens identical",
		store.fills, store.keys, reused, len(shared), whole, len(first))
}

// heldPages is a block's page memory of pp floats a page, page n resident
// where held[n] says so, for a gate that builds its history by hand.
func heldPages(pp int, held ...bool) *cpudev.Pages {
	p := cpudev.NewView(1).Pages(0, pp, false, nil)
	for n, h := range held {
		p.Extend(n)
		if h {
			p.Set(n, make([]float32, pp), make([]float32, pp))
		}
	}
	return p
}
