package jlm_test

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
	"github.com/jitllm/jitllm/tok"
)

// model returns a small GGUF to convert, or skips loudly (RULE 10): the
// message says which path it looked at rather than printing "ok" in 0.001s.
func model(t *testing.T, name string) string {
	t.Helper()
	p := testmodels.Path(name)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %s (%v) (set JITLLM_MODELS to the model directory) -- this gate proved nothing", p, err)
	}
	return p
}

// TestConvertedBytesEqualThePacker: the conversion is correct if and only if
// it produces the same bytes kernels.PackWeights produces, since a page-in
// hands them to the driver untouched. Anything weaker admits a subtly
// different layout that computes plausible wrong numbers.
func TestConvertedBytesEqualThePacker(t *testing.T) {
	for _, name := range []string{"tinyllama-1.1b-q3_K_M.gguf", "SmolLM2-360M-Instruct-Q8_0.gguf"} {
		t.Run(name, func(t *testing.T) {
			src := model(t, name)
			dst := filepath.Join(t.TempDir(), "m.jlm")
			if _, err := convert.FromGGUF(src, dst, jlm.Fingerprint{Host: "test"}); err != nil {
				t.Fatalf("convert: %v", err)
			}
			c, err := jlm.Open(dst)
			if err != nil {
				t.Fatalf("open: %v", err)
			}
			defer c.Close()
			// Open reads no page, so a span is nil until the block is in.
			if err := c.LoadAll(); err != nil {
				t.Fatalf("load: %v", err)
			}
			g, err := gguf.Open(src)
			if err != nil {
				t.Fatal(err)
			}
			defer g.Close()

			checked, packed := 0, 0
			for i := range c.Entries() {
				e := &c.Entries()[i]
				gt, ok := g.Get(e.Name)
				if !ok {
					t.Fatalf("%s: in the container, not in the source", e.Name)
				}
				qs, d, sc := c.Span(e)
				want := g.Bytes(gt)
				// Nothing is exempt: token_embd and output are packed like every
				// other quantised tensor.
				q, quantised := jlm.Packer(e.Type)
				if !quantised {
					if !bytes.Equal(qs[:len(want)], want) {
						t.Fatalf("%s: unquantised bytes differ", e.Name)
					}
					checked++
					continue
				}
				rows := e.Rows()
				for dd := 2; dd < int(e.NDim); dd++ {
					rows *= int(e.Dims[dd])
				}
				wq, wd, wsc, err := kernels.PackWeights(q, want, rows, e.K())
				if err != nil {
					t.Fatalf("%s: pack: %v", e.Name, err)
				}
				for _, p := range []struct {
					what string
					got  []byte
					want []uint32
				}{{"qs", qs, wq}, {"d", d, wd}, {"sc", sc, wsc}} {
					n := len(p.want) * 4
					if n == 0 {
						continue
					}
					if len(p.got) < n {
						t.Fatalf("%s %s: container span is %d bytes, the packer produced %d",
							e.Name, p.what, len(p.got), n)
					}
					if !bytes.Equal(p.got[:n], jlm.U32Bytes(p.want)) {
						t.Fatalf("%s %s: the container and kernels.PackWeights disagree "+
							"(%d bytes, rows=%d k=%d %s)", e.Name, p.what, n, rows, e.K(), q)
					}
				}
				checked++
				packed++
			}
			if checked == 0 || packed == 0 {
				t.Fatalf("compared %d tensors, %d of them packed -- this gate proved nothing", checked, packed)
			}
			t.Logf("%s: %d tensors byte-identical, %d of them packed; page %d bytes, %d blocks",
				name, checked, packed, c.H.PageSize, c.H.NBlocks)
		})
	}
}

// TestEveryBlockFitsItsPage: block i must live entirely inside
// [DataOff+i*P, DataOff+(i+1)*P), which is what makes a page-in one multiply
// and a slice.
func TestEveryBlockFitsItsPage(t *testing.T) {
	src := model(t, "tinyllama-1.1b-q3_K_M.gguf")
	dst := filepath.Join(t.TempDir(), "m.jlm")
	if _, err := convert.FromGGUF(src, dst, jlm.Fingerprint{}); err != nil {
		t.Fatal(err)
	}
	c, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if err := c.LoadAll(); err != nil {
		t.Fatalf("load: %v", err)
	}

	blocks, dense := 0, 0
	for i := range c.Entries() {
		e := &c.Entries()[i]
		if e.Block == jlm.DenseBlock {
			continue
		}
		// A block tensor lives in one of two regions: per-block vectors (norms,
		// biases, per-head scalars) are in the dense region (alwaysResident),
		// still found by (role, block); matrices are in the page. The failure
		// caught is a span in neither: placed by one rule, resolved by another.
		lo := c.H.DataOff + uint64(e.Block)*c.H.PageSize
		hi := lo + c.H.PageSize
		if e.Role.Expanded() {
			lo, hi = c.H.DenseOff, c.H.DenseOff+c.H.DenseLen
			dense++
		}
		qs, d, sc := c.Span(e)
		for _, s := range []struct {
			what string
			off  uint64
			n    int
		}{{"qs", e.QSOff, len(qs)}, {"d", e.DOff, len(d)}, {"sc", e.SCOff, len(sc)}} {
			if s.n == 0 {
				continue
			}
			if s.off < lo || s.off+uint64(s.n) > hi {
				where := "block " + itoaT(int(e.Block)) + "'s page"
				if e.Role.Expanded() {
					where = "the dense region"
				}
				t.Fatalf("%s %s: [%d,%d) is outside %s [%d,%d)",
					e.Name, s.what, s.off, s.off+uint64(s.n), where, lo, hi)
			}
			if s.off%jlm.Align != 0 {
				t.Fatalf("%s %s: offset %d is not %d-aligned", e.Name, s.what, s.off, jlm.Align)
			}
		}
		blocks++
	}
	if blocks == 0 {
		t.Fatal("no block tensors were checked -- this gate proved nothing")
	}
	t.Logf("%d block tensors inside their pages, page %d bytes", blocks, c.H.PageSize)
}

// TestContainerCarriesTheTokenizer: self-sufficiency is a gate, since a user
// may delete the source after converting. That includes
// tokenizer.chat_template, which nothing on the decode path consumes and is
// therefore the field that gets forgotten.
func TestContainerCarriesTheTokenizer(t *testing.T) {
	src := model(t, "tinyllama-1.1b-q3_K_M.gguf")
	dst := filepath.Join(t.TempDir(), "m.jlm")
	if _, err := convert.FromGGUF(src, dst, jlm.Fingerprint{}); err != nil {
		t.Fatal(err)
	}
	c, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	g, err := gguf.Open(src)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()

	// Both arms go through the container's schema: a Vocab built from the
	// parsed GGUF in memory against one decoded from the written bytes, so
	// the round trip (codec, merge pairs, special ids, stages) is checked.
	direct, err := convert.VocabOf(g)
	if err != nil {
		t.Fatal(err)
	}
	want, err := tok.New(direct)
	if err != nil {
		t.Fatal(err)
	}
	got, err := tok.New(c.Vocab())
	if err != nil {
		t.Fatalf("tok.New on the container: %v -- the tokenizer did not survive conversion", err)
	}
	probes := []string{
		"The capital of France is", "héllo wörld", "1234567890",
		"def f(x):\n    return x*2\n", "  leading and trailing  ", "🙂 emoji",
	}
	for _, p := range probes {
		a, b := want.Encode(p, true), got.Encode(p, true)
		if len(a) != len(b) {
			t.Fatalf("%q: %d ids from the GGUF, %d from the container", p, len(a), len(b))
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("%q: id %d is %d from the GGUF and %d from the container", p, i, a[i], b[i])
			}
		}
	}
	// Anything the engine reaches for that lives only in the source file
	// breaks the first time someone frees disk.
	if _, ok := g.KV["tokenizer.chat_template"]; ok {
		if _, has := c.Vocab().Template(""); !has {
			t.Fatal("the source has a chat template and the container does not -- " +
				"a caller who deletes the source loses it")
		}
	}
	if c.Config().Arch == jlm.ArchNone {
		t.Fatal("the container carries no architecture")
	}
	v := c.Vocab()
	t.Logf("tokenizer round-tripped over %d probes; %d tokens, %d merges, %d pre stages, %d templates",
		len(probes), len(v.Tokens), len(v.Merges), len(v.Pre), len(v.Templates))
}

// TestVersionIsRefusedNotMisparsed: the magic says "is this ours", the version
// "can we read it", so they are separate fields. There is no backward
// compatibility, so the refusal has to be exact and say what to do.
func TestVersionIsRefusedNotMisparsed(t *testing.T) {
	src := model(t, "tinyllama-1.1b-q3_K_M.gguf")
	dst := filepath.Join(t.TempDir(), "m.jlm")
	if _, err := convert.FromGGUF(src, dst, jlm.Fingerprint{}); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	if got := string(raw[0:8]); got != "JITLLM\x00\x00" {
		t.Fatalf("magic is %q, want %q", got, "JITLLM\x00\x00")
	}
	if v := binary.LittleEndian.Uint32(raw[8:]); v != jlm.Version {
		t.Fatalf("version field is %d, want %d", v, jlm.Version)
	}

	// A different version: refused, and the message names both numbers.
	bumped := filepath.Join(t.TempDir(), "bumped.jlm")
	b := append([]byte(nil), raw...)
	binary.LittleEndian.PutUint32(b[8:], jlm.Version+1)
	if err := os.WriteFile(bumped, b, 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := jlm.Open(bumped)
	if err == nil {
		f.Close()
		t.Fatal("a future version opened -- the version field is not checked")
	}
	if !strings.Contains(err.Error(), "convert") {
		t.Fatalf("refused with %q, which does not say how to fix it", err)
	}

	// A different magic: refused as not ours, not as a version problem.
	alien := filepath.Join(t.TempDir(), "alien.jlm")
	b = append([]byte(nil), raw...)
	copy(b[0:8], "GGUF\x00\x00\x00\x00")
	if err := os.WriteFile(alien, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if f, err := jlm.Open(alien); err == nil {
		f.Close()
		t.Fatal("a foreign magic opened")
	} else if !strings.Contains(err.Error(), "not a jlm container") {
		t.Fatalf("refused with %q, want the magic to be the complaint", err)
	}
}

func itoaT(v int) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

// TestPagesArePhysicallyPaddedAndIndexed is the gate on the layout's central
// promise: page size is decided at conversion and every block occupies exactly
// that, so reading block i is DataOff + i*PageSize and nothing else (an
// index, not a running sum carried on the hot path). Three things must hold:
//
//   - every block's tensors fit inside its page (TestEveryBlockFitsItsPage);
//   - PageSize is the maximum block, so no block is short-changed;
//   - the file is long enough for NBlocks whole pages, so reading the last
//     one is a full read rather than a short one at EOF.
func TestPagesArePhysicallyPaddedAndIndexed(t *testing.T) {
	src := model(t, "tinyllama-1.1b-q3_K_M.gguf")
	dst := filepath.Join(t.TempDir(), "m"+jlm.Ext)
	if _, err := convert.FromGGUF(src, dst, jlm.Fingerprint{}); err != nil {
		t.Fatal(err)
	}
	c, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	h := c.H
	if h.NBlocks == 0 {
		t.Skip("no block pages in this container")
	}

	fi, err := os.Stat(dst)
	if err != nil {
		t.Fatal(err)
	}
	want := int64(h.DataOff) + int64(h.NBlocks)*int64(h.PageSize)
	if fi.Size() < want {
		t.Fatalf("file is %d bytes, but block %d ends at %d: the last page is short and "+
			"a full read of it runs off the end", fi.Size(), h.NBlocks-1, want)
	}

	// PageSize is the widest block, so the widest block exactly fills its page.
	used := make([]uint64, h.NBlocks)
	for i := range c.Entries() {
		e := &c.Entries()[i]
		if e.Block == jlm.DenseBlock || e.Role.Expanded() {
			continue
		}
		lo := h.DataOff + uint64(e.Block)*h.PageSize
		qs, d, sc := c.SpanLen(e)
		for _, sp := range []struct{ off, n uint64 }{{e.QSOff, qs}, {e.DOff, d}, {e.SCOff, sc}} {
			if sp.n == 0 {
				continue
			}
			if end := sp.off + sp.n - lo; end > used[e.Block] {
				used[e.Block] = end
			}
		}
	}
	var widest uint64
	for _, u := range used {
		if u > h.PageSize {
			t.Fatalf("a block uses %d bytes of a %d-byte page", u, h.PageSize)
		}
		if u > widest {
			widest = u
		}
	}
	if widest == 0 {
		t.Fatal("no block used any of its page; this gate proved nothing")
	}
	// The widest block must be within one alignment unit of the page: if it were
	// far short, PageSize came from somewhere other than the blocks.
	if h.PageSize-widest >= jlm.Align {
		t.Errorf("widest block is %d bytes but PageSize is %d: page size is not the "+
			"maximum block", widest, h.PageSize)
	}
	t.Logf("%d pages of %d bytes, widest block %d, file %d = DataOff %d + %d*%d",
		h.NBlocks, h.PageSize, widest, fi.Size(), h.DataOff, h.NBlocks, h.PageSize)
}

// TestWeightsStayRelocatable is the gate on what may leave a block's page.
//
// The MoE router is F32 and would be swept out of its page by a rule of "not
// packed lives in the dense region", making it permanently host-resident.
// Routing data must not be evicted but must stay relocatable, which is a
// residency policy over pages. Anything a matvec reads in place stays a page;
// only tensors expanded at load (Role.Expanded) may live elsewhere.
func TestWeightsStayRelocatable(t *testing.T) {
	var src string
	for _, n := range []string{"Qwen3-MOE-4x0.6B-Q4_K_M.gguf", "olmoe-1b-7b-Q4_K_M.gguf"} {
		p := testmodels.Path(n)
		if _, err := os.Stat(p); err == nil {
			src = p
			break
		}
	}
	if src == "" {
		t.Skip("MODEL MISSING: no mixture in " + testmodels.Dir() + " (set JITLLM_MODELS to the model directory) -- this gate proved nothing")
	}
	dst := filepath.Join(t.TempDir(), "m"+jlm.Ext)
	if _, err := convert.FromGGUF(src, dst, jlm.Fingerprint{}); err != nil {
		t.Fatal(err)
	}
	c, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	h := c.H

	routers, weights := 0, 0
	for i := range c.Entries() {
		e := &c.Entries()[i]
		if e.Block == jlm.DenseBlock || e.Role.Expanded() {
			continue
		}
		inDense := e.QSOff >= h.DenseOff && e.QSOff < h.DenseOff+h.DenseLen
		if inDense {
			t.Errorf("%s (%v) is in the DENSE region: a weight a matvec reads must stay "+
				"a page, or it can never be placed on a device or moved off one",
				e.Name, e.Role)
		}
		weights++
		if e.Role == jlm.RoleRouter {
			routers++
		}
	}
	if routers == 0 {
		t.Fatal("no router found in a mixture -- this gate proved nothing")
	}
	t.Logf("%d block weights stay pages, %d of them routers", weights, routers)
}
