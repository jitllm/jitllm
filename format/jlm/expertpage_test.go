package jlm_test

import (
	"bytes"
	"encoding/binary"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// expertSource is a three-block mixture: blocks 0 and 1 carry expert banks
// (two Q4_0, one F32 to cover a verbatim sheet), block 2 is dense and must get
// no expert pages at all.
func expertSource(t *testing.T) (*jlm.Source, map[jlm.Role][2][]byte) {
	t.Helper()
	const k, rows, nExp = 64, 32, 4
	rng := rand.New(rand.NewPCG(7, 9))
	q4 := func(n int) []byte { // n rows of k as Q4_0 source blocks
		b := make([]byte, n*k/32*18)
		for i := 0; i < len(b); i += 18 {
			binary.LittleEndian.PutUint16(b[i:], 0x3000+uint16(rng.IntN(0x800)))
			for j := 2; j < 18; j++ {
				b[i+j] = byte(rng.Uint32())
			}
		}
		return b
	}
	f32 := func(n int) []byte {
		b := make([]byte, n*4)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(b[4*i:], 0x3F000000+rng.Uint32()&0xFFFF)
		}
		return b
	}
	var ts []jlm.Tensor
	add := func(role jlm.Role, block int32, ty jlm.Type, dims []uint64, b []byte) {
		e := jlm.Tensor{Role: role, Block: block, Index: -1, Type: ty,
			NDim: uint8(len(dims)), Data: b, Name: role.String()}
		copy(e.Dims[:], dims)
		ts = append(ts, e)
	}
	banks := map[jlm.Role][2][]byte{}
	add(jlm.RoleTokenEmbd, jlm.DenseBlock, jlm.TypeQ4, []uint64{k, 8}, q4(8))
	for b := int32(0); b < 3; b++ {
		add(jlm.RoleAttnQ, b, jlm.TypeQ4, []uint64{k, k}, q4(k))
		if b == 2 {
			continue
		}
		gate, down := q4(rows*nExp), f32(k*rows*nExp)
		add(jlm.RoleExpGateBank, b, jlm.TypeQ4, []uint64{k, rows, nExp}, gate)
		add(jlm.RoleExpDownBank, b, jlm.TypeF32, []uint64{rows, k, nExp}, down)
		if b == 1 {
			banks[jlm.RoleExpGateBank] = [2][]byte{gate}
			banks[jlm.RoleExpDownBank] = [2][]byte{down}
		}
	}
	return &jlm.Source{
		Config: &jlm.Config{Arch: jlm.ArchQwen3MoE, NLayer: 3, NEmbd: k, NHead: 1, NKVHead: 1,
			HeadDim: k, NRot: k, NFFN: k, NVocab: 8, NCtx: 128, RMSEps: 1e-5, RopeBase: 10000,
			EmbdScale: 1, AttnFactor: 1, NExpert: nExp, NExpertUsed: 2},
		Vocab: &jlm.Vocab{Kind: jlm.VocabBPE, Tokens: jlm.NewStrings([]string{"a", "b", "c", "d", "e", "f", "g", "h"}),
			Kinds: make([]jlm.TokenKind, 8), BOS: -1, EOS: -1, Unk: -1, Pad: -1, Sep: -1, Mask: -1},
		Tensors: ts,
	}, banks
}

// TestExpertPagesHoldEachExpertsSheets writes a mixture and reads every expert
// of block 1 back through its own expert page: the bytes must be exactly what
// the packer makes of that expert's source sheet (or the sheet itself, for a
// verbatim bank), the bank must have no contiguous span, the dense block must
// own no expert page, and eviction must take the least recently used expert.
func TestExpertPagesHoldEachExpertsSheets(t *testing.T) {
	src, banks := expertSource(t)
	dst := filepath.Join(t.TempDir(), "m.jlm")
	h, err := jlm.Write(dst, src, jlm.Fingerprint{Host: "t"})
	if err != nil {
		t.Fatal(err)
	}
	if h.NExpPages != 2*4 || h.ExpPageSize == 0 {
		t.Fatalf("%d expert pages of %d; two mixture blocks of four experts want 8", h.NExpPages, h.ExpPageSize)
	}
	c, err := jlm.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	checked := 0
	for _, role := range []jlm.Role{jlm.RoleExpGateBank, jlm.RoleExpDownBank} {
		e, ok := c.Find(role, 1, -1)
		if !ok {
			t.Fatalf("no %v in block 1", role)
		}
		if !c.ExpertPaged(e) {
			t.Fatalf("%v is not in expert pages", role)
		}
		if qs, _, _ := c.Span(e); qs != nil {
			t.Fatalf("%v has a contiguous span; a split bank must be reached by Sheet", role)
		}
		raw := banks[role][0]
		sheets := int(e.Dims[2])
		per := len(raw) / sheets
		for x := 0; x < sheets; x++ {
			p := c.ExpertPage(e, x)
			if want := int(h.NBlocks) + 4 + x; p != want {
				t.Fatalf("%v expert %d is page %d, want %d (block 1 follows block 0's four)", role, x, p, want)
			}
			if err := c.EnsurePage(p); err != nil {
				t.Fatal(err)
			}
			qs, d, sc := c.Sheet(e, x)
			sheet := raw[x*per : (x+1)*per]
			q, packed := jlm.Packer(e.Type)
			if !packed {
				if !bytes.Equal(qs, sheet) {
					t.Fatalf("%v expert %d: verbatim sheet differs", role, x)
				}
				checked++
				continue
			}
			wq, wd, wsc, err := kernels.PackWeights(q, sheet, int(e.Dims[1]), int(e.Dims[0]))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(qs, jlm.U32Bytes(wq)) || !bytes.Equal(d, jlm.U32Bytes(wd)) ||
				!bytes.Equal(sc, jlm.U32Bytes(wsc)) {
				t.Fatalf("%v expert %d: sheet bytes are not the packer's", role, x)
			}
			checked++
		}
	}
	if checked != 8 {
		t.Fatalf("checked %d sheets, want 8", checked)
	}
	// The dense block's attention is in its block page and has no expert page.
	if e, _ := c.Find(jlm.RoleAttnQ, 2, -1); c.ExpertPage(e, 0) != -1 {
		t.Fatal("a dense weight resolved to an expert page")
	}

	// LRU over experts: room for exactly two expert pages, touch A, B, C.
	c.ReleasePages()
	c.SetBudget(2 * h.ExpPageSize)
	e, _ := c.Find(jlm.RoleExpGateBank, 0, -1)
	for x := 0; x < 3; x++ {
		if err := c.EnsurePage(c.ExpertPage(e, x)); err != nil {
			t.Fatal(err)
		}
	}
	if c.Resident(c.ExpertPage(e, 0)) || !c.Resident(c.ExpertPage(e, 1)) || !c.Resident(c.ExpertPage(e, 2)) {
		t.Fatalf("after A, B, C in room for two: A=%v B=%v C=%v, want the least recently used (A) evicted",
			c.Resident(c.ExpertPage(e, 0)), c.Resident(c.ExpertPage(e, 1)), c.Resident(c.ExpertPage(e, 2)))
	}
	// Touch B, bring A back: C is now the oldest.
	if err := c.EnsurePage(c.ExpertPage(e, 1)); err != nil {
		t.Fatal(err)
	}
	if err := c.EnsurePage(c.ExpertPage(e, 0)); err != nil {
		t.Fatal(err)
	}
	if c.Resident(c.ExpertPage(e, 2)) || !c.Resident(c.ExpertPage(e, 1)) {
		t.Fatal("a re-used expert was evicted ahead of an older one")
	}
}
