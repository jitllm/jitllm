package jlm

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// Source is a model, described in this format's own terms, ready to be
// written: what the model is, what its tokenizer is, and a tensor table whose
// entries carry their own bytes. Anything that can fill one is a converter,
// so a new source format is a new file in convert/, not a change here.
type Source struct {
	Config  *Config
	Vision  *Vision // a tower only
	Vocab   *Vocab
	Tensors []Tensor
	// Origin names the bytes the tensors come from, immutably: a pinned
	// repository commit and its files, or local files with their sizes and
	// times. A write of a named origin can be resumed (see Write); "" is a
	// source that cannot name its bytes, whose write always starts over.
	Origin string
}

// Tensor is one weight on its way in. Data is the source bytes; the writer
// packs them into this format's layout (see Write).
//
// Load, when set, replaces Data and is called once, when the writer reaches
// the tensor. The layout pass needs only Type and Dims, so a source too big to
// hold (or streamed, not on disk at all) hands over a function and the peak
// is one tensor instead of the model.
type Tensor struct {
	Role  Role
	Block int32 // DenseBlock for a non-block tensor
	Index int32 // -1 unless the role is stored per expert
	Type  Type
	NDim  uint8
	Dims  [4]uint64
	Data  []byte
	Load  func() ([]byte, error)
	Name  string // diagnostics only
}

// Bytes is the tensor's source bytes, from Data or from Load.
func (t *Tensor) Bytes() ([]byte, error) {
	if t.Load != nil {
		return t.Load()
	}
	return t.Data, nil
}

// packerOf maps a container type onto the device packer's, and reports false
// for a type carried verbatim.
//
// A type that is not packed is not an error: F32/F16 norms, biases and
// routers are carried verbatim, since the container must be self-sufficient.
func packerOf(t Type) (kernels.Quant, bool) {
	switch t {
	case TypeQ4:
		return kernels.Q4_0, true
	case TypeQ8:
		return kernels.Q8_0, true
	case TypeQ5:
		return kernels.Q5_0, true
	case TypeQ51:
		return kernels.Q5_1, true
	case TypeMX4:
		return kernels.MXFP4, true
	case TypeTQ2:
		return kernels.TQ2, true
	case TypeTQ2F:
		return kernels.TQ2F, true
	case TypeQ3S:
		return kernels.Q3_K, true
	case TypeQ4S:
		return kernels.Q4_K, true
	case TypeQ5S:
		return kernels.Q5_K, true
	case TypeQ6S:
		return kernels.Q6_K, true
	}
	return 0, false
}

// Packed reports whether a container stores this type in the transposed plane
// layout. Exported for the reading side, which needs to know whether a span is
// a payload or a row-major array.
func Packed(t Type) bool { _, ok := packerOf(t); return ok }

// Packer is packerOf, exported so a gate can pack a tensor the same way the
// writer did and compare the bytes.
func Packer(t Type) (kernels.Quant, bool) { return packerOf(t) }

// verbatimBytes is how many bytes a non-packed type occupies for n elements.
func verbatimBytes(t Type, n uint64) (uint64, error) {
	switch t {
	case TypeF32:
		return n * 4, nil
	case TypeF16, TypeBF16:
		return n * 2, nil
	case TypeIQ4NL:
		if n%32 != 0 {
			return 0, fmt.Errorf("jlm: %d iq4nl elements is not a whole number of 32-wide blocks", n)
		}
		return n / 32 * 18, nil
	}
	return 0, fmt.Errorf("jlm: type %v is neither packed nor verbatim", t)
}

// sheetsOf is how many independent matrices an entry holds: one for an
// ordinary weight, n_expert for a mixture's bank.
//
// An expert bank is n_expert sheets, not one tall one. The packed layout is
// [sub-block][word][row] with rows interleaved, so packing a bank as one sheet
// scatters expert e across the whole bank and reading one expert is thousands
// of strided reads. Packed per sheet, expert e owns one contiguous range. The
// total bytes are identical (PackedWords is linear in nrows).
func sheetsOf(dims [4]uint64, ndim uint8) (sheets, rows uint64) {
	sheets, rows = 1, 1
	if ndim >= 2 {
		rows = dims[1]
	}
	for i := 2; i < int(ndim); i++ {
		sheets *= dims[i]
	}
	return sheets, rows
}

// ExpertBank reports whether a role is a mixture's expert bank, which a v26
// container stores in expert pages rather than in its block page: one page per
// (block, expert), holding that expert's sheet of every bank in the block.
func ExpertBank(r Role) bool {
	return r == RoleExpGateBank || r == RoleExpUpBank || r == RoleExpDownBank
}

// expertPaged is ExpertBank for an entry that actually has sheets to split.
func expertPaged(e *Entry) bool {
	return ExpertBank(e.Role) && e.Block != DenseBlock && !alwaysResident(e) && e.NDim >= 3
}

// sheetPlanes is one sheet's unaligned length in each plane, packed or not. A
// verbatim bank is one plane of rows*k elements a sheet.
func sheetPlanes(e *Entry) (qs, d, sc, sheets uint64, err error) {
	if _, ok := packerOf(e.Type); ok {
		return sheetSpan(e)
	}
	sh, rows := sheetsOf(e.Dims, e.NDim)
	n, err := verbatimBytes(e.Type, rows*e.Dims[0])
	return n, 0, 0, sh, err
}

// SheetSpan is ONE expert sheet's length in each plane, plus how many sheets
// the entry holds. It is the unaligned per-sheet size, so sheet i of a plane
// begins at that plane's offset plus i*qs (or i*d, i*sc) exactly.
//
// It lives here because spans() decides a bank's layout, and a second copy
// of that arithmetic is how a reader fetches another expert's bytes. spans()
// rounds each plane up to Align; the sheets inside it have no padding
// between them.
//
// sheets is 1 for an ordinary 2-D weight, where the "sheet" is the whole thing.
func (f *File) SheetSpan(e *Entry) (qs, d, sc, sheets uint64, err error) {
	return sheetSpan(e)
}

// sheetSpan is SheetSpan without a File, so the writer can ask it too.
func sheetSpan(e *Entry) (qs, d, sc, sheets uint64, err error) {
	sh, rows := sheetsOf(e.Dims, e.NDim)
	q, ok := packerOf(e.Type)
	if !ok {
		// Verbatim types have no planes to split; the caller wants the whole
		// span and SpanLen already gives it.
		return 0, 0, 0, 0, fmt.Errorf("jlm: %v is %v, which is not packed", e.Role, e.Type)
	}
	pq, pd, psc, err := kernels.PackedWords(q, int(rows), int(e.Dims[0]))
	if err != nil {
		return 0, 0, 0, 0, err
	}
	return uint64(pq) * 4, uint64(pd) * 4, uint64(psc) * 4, sh, nil
}

// minGroupBytes is the smallest PCIe transfer a stream group should make.
//
// A group is a separate WriteAt per plane and a small host-to-device copy is
// priced per call, so splitting too far costs more than the overlap buys.
// 64 KiB is where that turned over on the PCIe link measured; it is
// the one number here that belongs to the machine rather than the model.
const minGroupBytes = 64 << 10

// streamGroupsFor is how many pieces a streamed block's selection should be
// uploaded in, so that group g's transfer overlaps group g+1's read.
//
// The limit falls out of the model's geometry, which is why it can be
// stored: the smallest plane of one sheet bounds splitting (a group moves
// ceil(k/n) of them per plane), so the rule is the largest n whose group
// still clears minGroupBytes. On Qwen3-Next-80B that gives three, which is
// where the measured dose peaked. A container is written where it runs (see
// Fingerprint); tier.Config.StreamGroups overrides it.
func streamGroupsFor(es []Entry, used int) uint32 {
	if used < 2 {
		return 1
	}
	smallest := uint64(0)
	for i := range es {
		e := &es[i]
		if e.Block == DenseBlock || e.Role.Expanded() {
			continue
		}
		qs, d, sc, sheets, err := sheetSpan(e)
		if err != nil || sheets < 2 {
			continue // not a bank: nothing here is uploaded per sheet
		}
		for _, n := range [3]uint64{qs, d, sc} {
			if n > 0 && (smallest == 0 || n < smallest) {
				smallest = n
			}
		}
	}
	if smallest == 0 {
		return 1 // no routed bank, so nothing streams
	}
	n := uint32(1)
	for try := uint32(2); int(try) <= used; try++ {
		per := (uint64(used) + uint64(try) - 1) / uint64(try)
		if per*smallest < minGroupBytes {
			break
		}
		n = try
	}
	return n
}

// chunkFor is the fill granularity this model's geometry asks for: the smallest
// span any reader requests on its own, rounded down to a power of two.
//
// It is computed at conversion and stored in the header because the right
// chunk is a property of the model: a package global must be set before Open
// and silently does nothing afterwards, while a header value cannot be applied
// too late and is visible in `jitllm info`.
//
// The quantity is the smallest independently-requested span, since that is
// what the rounding is paid on: a routed bank is asked for per (sheet,
// plane), and a 1 MiB chunk rounded Qwen3-Next-80B's 16 KiB d planes up 64x.
// Going finer buys nothing, and coarser costs a contiguous read nothing
// (planRuns coalesces adjacent chunks), so this is a floor, not a tuning.
//
// Expanded and dense-region tensors are excluded: they are never paged.
func chunkFor(es []Entry) uint64 {
	smallest := uint64(1) << 20
	for i := range es {
		e := &es[i]
		// An expert page is read whole, so its planes are never requested on
		// their own and must not drag the block pages' granularity down.
		if e.Block == DenseBlock || e.Role.Expanded() || expertPaged(e) {
			continue
		}
		if qs, d, sc, sheets, err := sheetSpan(e); err == nil && sheets > 0 {
			for _, n := range [3]uint64{qs, d, sc} {
				if n > 0 && n < smallest {
					smallest = n
				}
			}
			continue
		}
		if qs, d, sc, err := spans(e.Type, e.Dims, e.NDim); err == nil {
			if n := qs + d + sc; n > 0 && n < smallest {
				smallest = n
			}
		}
	}
	// Down to a power of two: an unaligned chunk makes an unaligned page read,
	// which the direct handle answers with EINVAL.
	c := uint64(1) << 20
	for c > Align && c > smallest {
		c >>= 1
	}
	if c < Align {
		c = Align
	}
	return c
}

// spans is how many bytes each of a tensor's three regions occupies, already
// rounded up to Align; a zero length means the region does not exist. Every
// packable type is packed, including the embedding and head; F32/F16 norms,
// biases and routers are verbatim because there is no device layout for them.
func spans(t Type, dims [4]uint64, ndim uint8) (qs, d, sc uint64, err error) {
	k := dims[0]
	sheets, rows := sheetsOf(dims, ndim)
	if q, ok := packerOf(t); ok {
		pq, pd, psc, err := kernels.PackedWords(q, int(rows), int(k))
		if err != nil {
			return 0, 0, 0, err
		}
		return alignUp(uint64(pq) * 4 * sheets),
			alignUp(uint64(pd) * 4 * sheets),
			alignUp(uint64(psc) * 4 * sheets), nil
	}
	rows *= sheets
	n, err := verbatimBytes(t, rows*k)
	if err != nil {
		return 0, 0, 0, err
	}
	return alignUp(n), 0, 0, nil
}

// StoredBytes is how many bytes a tensor of this type and shape occupies in a
// container, its planes aligned as Write lays them.
func StoredBytes(t Type, dims [4]uint64, ndim uint8) (uint64, error) {
	qs, d, sc, err := spans(t, dims, ndim)
	return qs + d + sc, err
}

// alwaysResident says a tensor lives in the dense region rather than in its
// block's page, so it is readable whether or not that block is anywhere.
//
// Norms live in the dense region rather than in their block's page, so opening
// a model does not fault every block into host frames before placement can
// send most of them to a device. It is decided by role, through
// Role.Expanded: a tensor read into a fresh slice at load may live anywhere,
// and a tensor a matvec reads in place must stay a page. See Role.Expanded for
// why the MoE router is not here. The
func alwaysResident(e *Entry) bool {
	return !RowTable(e) && (e.Block == DenseBlock || e.Role.Expanded())
}

// RowTable says a tensor is a row table: rows read one at a time from the
// file, never resident whole and never in the dense region or a page. It
// sits after every page array, verbatim and row-major, so row r is
// RowBytes(e) bytes at QSOff + r*RowBytes(e). Qwen3.8-Flash-Next's n-gram
// table is one: 320 million rows, of which a token reads sixteen.
func RowTable(e *Entry) bool { return e.Role == RoleNGramTable }

// RowBytes is one row of a row table: Dims[0] elements of its type.
func RowBytes(e *Entry) (uint64, error) { return verbatimBytes(e.Type, e.Dims[0]) }

// Write lays src out as dst, a jlm container.
//
// src is a *Source, not a path: the container format depends on no source
// format, and a new input format is a new file in convert/.
//
// Two passes, and the first allocates nothing: the packed form of a large
// model does not fit in host RAM, so the layout is decided from arithmetic
// (kernels.PackedWords) before any tensor is packed.
func Write(dst string, src *Source, fp Fingerprint) (*Header, error) {
	l, err := layoutOf(src, fp)
	if err != nil {
		return nil, err
	}
	return l.write(dst, src)
}

// Plan is what Write would write for src, decided without reading a tensor's
// bytes: the header (page sizes, counts, offsets) and the container's size.
// It is the same layout Write takes, so a conversion that plans cleanly
// cannot fail an hour in on anything the shapes decide.
func Plan(src *Source, fp Fingerprint) (*Header, uint64, error) {
	l, err := layoutOf(src, fp)
	if err != nil {
		return nil, 0, err
	}
	return l.h, l.total, nil
}

// layout is pass 1's result: every entry placed, the encoded sections, and
// the file's size.
type layout struct {
	es                    []Entry
	h                     *Header
	cfgb, visb, vocb, fpb []byte
	expPageSize, total    uint64
}

func layoutOf(src *Source, fp Fingerprint) (*layout, error) {
	if src == nil || src.Config == nil {
		return nil, fmt.Errorf("jlm: Write needs a Source with a Config")
	}
	// --- pass 1: shapes and sizes, no packing ------------------------------
	es := make([]Entry, 0, len(src.Tensors))
	// Two page arrays, one file: a vision block gets its own array only
	// because one page size is wrong in both directions (see Role.Vision).
	// Nothing downstream branches on which array a tensor came from.
	perBlock := map[int32]uint64{}
	perVis := map[int32]uint64{}
	maxBlock, maxVis := int32(-1), int32(-1)
	for i := range src.Tensors {
		t := &src.Tensors[i]
		if t.NDim == 0 || int(t.NDim) > len(t.Dims) {
			return nil, fmt.Errorf("jlm: %v has %d dimensions", t.Role, t.NDim)
		}
		if !t.Role.Valid() {
			return nil, fmt.Errorf("jlm: tensor %d has role code %d, which this format does not define", i, t.Role)
		}
		if !t.Type.Valid() {
			return nil, fmt.Errorf("jlm: %v has type code %d, which this format does not define", t.Role, t.Type)
		}
		e := Entry{Role: t.Role, Block: t.Block, Index: t.Index, Type: t.Type, NDim: t.NDim, Name: t.Name}
		e.Dims = t.Dims
		qs, d, sc, err := spans(t.Type, e.Dims, e.NDim)
		if err != nil {
			return nil, fmt.Errorf("jlm: %v: %w", t.Role, err)
		}
		if e.Role.Vision() {
			if e.Block > maxVis {
				maxVis = e.Block
			}
		} else if e.Block > maxBlock {
			maxBlock = e.Block
		}
		if !alwaysResident(&e) && !expertPaged(&e) && !RowTable(&e) {
			if e.Role.Vision() {
				perVis[e.Block] += qs + d + sc
			} else {
				perBlock[e.Block] += qs + d + sc
			}
		}
		es = append(es, e)
	}

	type roleAt struct {
		r Role
		b int32
	}
	present := map[roleAt]bool{}
	for i := range es {
		if !es[i].Role.Vision() {
			present[roleAt{es[i].Role, es[i].Block}] = true
		}
	}
	if err := src.Config.CheckQKNorm(int(maxBlock+1), func(r Role, b int32) bool {
		return present[roleAt{r, b}]
	}); err != nil {
		return nil, err
	}

	// P = max block bytes, and the padding is physical: every block occupies
	// exactly P bytes so the file is the page array (block i at DataOff + i*P).
	// The waste is only the block-size variance, paid once here rather than as
	// runtime raggedness every token.
	var pageSize uint64
	for _, n := range perBlock {
		if n > pageSize {
			pageSize = n
		}
	}
	pageSize = alignUp(pageSize)
	nBlocks := uint32(maxBlock + 1)
	if len(perBlock) == 0 {
		nBlocks, pageSize = 0, Align
	}
	var visPageSize uint64
	for _, n := range perVis {
		if n > visPageSize {
			visPageSize = n
		}
	}
	visPageSize = alignUp(visPageSize)
	nVisBlocks := uint32(maxVis + 1)
	if len(perVis) == 0 {
		nVisBlocks, visPageSize = 0, Align
	}

	// The expert pages. Each mixture block contributes NExpert of them, in
	// block order; page p holds, for every expert bank of its block in table
	// order, that expert's sheet, each plane Align'd for device import and
	// O_DIRECT. inner[i] is a bank's plane offsets inside any of its block's
	// expert pages, so one set of entry offsets addresses them all: expert x
	// is x*expPageSize further on.
	type planeOffs struct{ qs, d, sc uint64 }
	inner := map[int]planeOffs{}
	expOrd := map[int32]uint64{} // block -> its first expert page
	var expPageSize, nExpert uint64
	{
		perExp := map[int32]uint64{}
		var blocks []int32
		for i := range es {
			e := &es[i]
			if !expertPaged(e) {
				continue
			}
			qs, d, sc, sheets, err := sheetPlanes(e)
			if err != nil {
				return nil, fmt.Errorf("jlm: %s: %w", e.Name, err)
			}
			if nExpert == 0 {
				nExpert = sheets
			}
			if sheets != nExpert {
				return nil, fmt.Errorf("jlm: %s has %d experts and another bank has %d; expert "+
					"pages are indexed by (block, expert) and need one count", e.Name, sheets, nExpert)
			}
			if _, seen := perExp[e.Block]; !seen {
				blocks = append(blocks, e.Block)
			}
			cur := perExp[e.Block]
			var po planeOffs
			po.qs, cur = cur, cur+alignUp(qs)
			po.d, cur = cur, cur+alignUp(d)
			po.sc, cur = cur, cur+alignUp(sc)
			inner[i] = po
			perExp[e.Block] = cur
			expPageSize = max(expPageSize, cur)
		}
		sort.Slice(blocks, func(a, b int) bool { return blocks[a] < blocks[b] })
		for o, b := range blocks {
			expOrd[b] = uint64(o) * nExpert
		}
	}
	nExpPages := uint64(len(expOrd)) * nExpert
	expPageSize = alignUp(expPageSize)

	// --- layout -------------------------------------------------------------
	cfgb := encodeConfig(src.Config)
	visb := encodeVision(src.Vision)
	vocb := encodeVocab(src.Vocab)
	fpb := encodeFP(fp)
	// The table's size does not depend on the offsets it holds -- they are
	// fixed-width -- so it can be encoded once for length and again for content.
	tabLen := uint64(len(encodeTable(es)))

	h := &Header{PageSize: pageSize, NBlocks: nBlocks, NTensors: uint32(len(es)),
		VisPageSize: visPageSize, NVisBlocks: nVisBlocks, Chunk: chunkFor(es),
		StreamGroups: streamGroupsFor(es, int(src.Config.NExpertUsed))}
	h.CfgOff = HeaderBytes
	h.CfgLen = uint64(len(cfgb))
	h.VisOff = h.CfgOff + h.CfgLen
	h.VisLen = uint64(len(visb))
	h.VocOff = h.VisOff + h.VisLen
	h.VocLen = uint64(len(vocb))
	h.TabOff = h.VocOff + h.VocLen
	h.TabLen = tabLen
	h.FPOff = h.TabOff + h.TabLen
	h.FPLen = uint64(len(fpb))
	h.DenseOff = alignUp(h.FPOff + h.FPLen)

	dense := h.DenseOff
	for i := range es {
		if !alwaysResident(&es[i]) {
			continue
		}
		qs, d, sc, _ := spans(es[i].Type, es[i].Dims, es[i].NDim)
		es[i].QSOff, dense = dense, dense+qs
		es[i].DOff, dense = dense, dense+d
		es[i].SCOff, dense = dense, dense+sc
	}
	h.DenseLen = dense - h.DenseOff
	h.DataOff = alignUp(dense)

	h.VisDataOff = alignUp(h.DataOff + uint64(nBlocks)*pageSize)
	h.ExpDataOff = alignUp(h.VisDataOff + uint64(nVisBlocks)*visPageSize)
	h.NExpPages, h.ExpPageSize = uint32(nExpPages), expPageSize
	if nExpPages == 0 {
		h.ExpDataOff, h.ExpPageSize = 0, 0
	}

	// One cursor per (array, block). The array is chosen by the role, the same
	// way the dense/page split is chosen by alwaysResident -- a tensor's
	// address is a property of what it is, not of who is reading it.
	type slot struct {
		vis   bool
		block int32
	}
	cursor := make(map[slot]uint64, len(perBlock)+len(perVis))
	for i := range es {
		// Same predicate as the dense pass above, or this one overwrites the
		// offsets it just assigned and a norm ends up inside a page again.
		if alwaysResident(&es[i]) || RowTable(&es[i]) {
			continue
		}
		if po, ok := inner[i]; ok {
			page0 := h.ExpDataOff + expOrd[es[i].Block]*expPageSize
			es[i].QSOff, es[i].DOff, es[i].SCOff = page0+po.qs, page0+po.d, page0+po.sc
			continue
		}
		k := slot{es[i].Role.Vision(), es[i].Block}
		base, ok := cursor[k]
		if !ok {
			if k.vis {
				base = h.VisDataOff + uint64(k.block)*visPageSize
			} else {
				base = h.DataOff + uint64(k.block)*pageSize
			}
		}
		qs, d, sc, _ := spans(es[i].Type, es[i].Dims, es[i].NDim)
		es[i].QSOff, base = base, base+qs
		es[i].DOff, base = base, base+d
		es[i].SCOff, base = base, base+sc
		cursor[k] = base
	}

	total := h.VisDataOff + uint64(nVisBlocks)*visPageSize
	if nExpPages > 0 {
		total = h.ExpDataOff + nExpPages*expPageSize
	}
	// The row tables, after everything a pager reads.
	for i := range es {
		if !RowTable(&es[i]) {
			continue
		}
		if es[i].Block != DenseBlock || es[i].NDim != 2 || Packed(es[i].Type) {
			return nil, fmt.Errorf("jlm: %s: a row table is a model-level 2-D verbatim tensor", es[i].Name)
		}
		qs, _, _, err := spans(es[i].Type, es[i].Dims, es[i].NDim)
		if err != nil {
			return nil, fmt.Errorf("jlm: %s: %w", es[i].Name, err)
		}
		es[i].QSOff = alignUp(total)
		es[i].DOff, es[i].SCOff = es[i].QSOff+qs, es[i].QSOff+qs
		total = es[i].QSOff + qs
	}
	return &layout{es: es, h: h, cfgb: cfgb, visb: visb, vocb: vocb, fpb: fpb,
		expPageSize: expPageSize, total: total}, nil
}

// write is pass 2: the bytes, one tensor at a time.
//
// A container is written beside itself and renamed, so an interrupted
// conversion leaves no container at all. Written in place, a failure after
// the header (ENOSPC, a kill) leaves a file that opens cleanly, carries the
// right version and Fingerprint, and is zeros where the weights should be
// (a full disk produces exactly that). The rename is in the same directory,
// so it is atomic.
//
// The next tensor's bytes are fetched while this one is packed and written:
// for a streamed source the fetch is the network, and the two overlap. Two
// Loads never run at once, so a Load need not be safe against another.
//
// A source that names its bytes (Source.Origin) is written resumably: every
// journalEvery bytes the part is synced and a journal beside it records how
// many tensors are down. A failed or killed write keeps both, and the next
// Write of the same layout from the same origin by the same build picks up
// at the journal's count. Anything else starts over.
func (l *layout) write(dst string, src *Source) (*Header, error) {
	es, h, expPageSize, total := l.es, l.h, l.expPageSize, l.total
	cfgb, visb, vocb, fpb := l.cfgb, l.visb, l.vocb, l.fpb
	hdr := make([]byte, HeaderBytes)
	h.encode(hdr)
	tab := encodeTable(es)

	tmp := dst + ".part"
	jn := tmp + journalSuffix
	journal := src.Origin != ""
	sum := resumeDigest(src.Origin, hdr, cfgb, visb, vocb, tab, fpb)
	start := 0
	if journal {
		start = resumeAt(tmp, jn, sum, total, len(es))
	}
	var out *os.File
	var err error
	if start > 0 {
		out, err = os.OpenFile(tmp, os.O_RDWR, 0)
	} else {
		os.Remove(jn)
		out, err = os.Create(tmp)
	}
	if err != nil {
		return nil, err
	}
	// keep is set once a journal records progress: from then on a failure
	// leaves the part for the next Write to resume.
	keep := start > 0
	defer func() {
		out.Close()
		if !keep {
			os.Remove(tmp) // a no-op once the rename below has happened
		}
	}()

	if start == 0 {
		if err := out.Truncate(int64(total)); err != nil {
			return nil, fmt.Errorf("jlm: sizing %s to %d: %w", dst, total, err)
		}
	}
	for _, w := range []struct {
		off uint64
		b   []byte
	}{{0, hdr}, {h.CfgOff, cfgb}, {h.VisOff, visb}, {h.VocOff, vocb},
		{h.TabOff, tab}, {h.FPOff, fpb}} {
		if _, err := out.WriteAt(w.b, int64(w.off)); err != nil {
			return nil, err
		}
	}

	type fetched struct {
		b   []byte
		err error
	}
	next := make(chan fetched, 1)
	fetch := func(i int) {
		b, err := src.Tensors[i].Bytes()
		next <- fetched{b, err}
	}
	if start < len(es) {
		go fetch(start)
	}
	var dirty uint64
	every := journalSpacing(total)
	for i := start; i < len(es); i++ {
		e := &es[i]
		got := <-next
		if got.err != nil {
			return nil, fmt.Errorf("jlm: %s (tensor %d of %d): %w", e.Name, i, len(es), got.err)
		}
		if i+1 < len(es) {
			go fetch(i + 1)
		}
		n, err := writeTensor(out, e, got.b, expPageSize)
		if err != nil {
			return nil, err
		}
		dirty += n
		if journal && (dirty >= every || i+1 == len(es)) {
			// The count is written only after the bytes it vouches for are
			// on the disk, so a journal never claims a tensor a crash lost.
			if err := out.Sync(); err != nil {
				return nil, err
			}
			if err := writeJournal(jn, sum, i+1); err != nil {
				return nil, err
			}
			dirty, keep = 0, true
		}
	}
	if err := out.Sync(); err != nil {
		return nil, err
	}
	if err := out.Close(); err != nil {
		return nil, fmt.Errorf("jlm: closing %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, dst); err != nil {
		return nil, fmt.Errorf("jlm: publishing %s: %w", dst, err)
	}
	os.Remove(jn)
	return h, nil
}

// writeTensor packs one tensor's source bytes into its place and reports how
// many bytes it wrote.
func writeTensor(out *os.File, e *Entry, raw []byte, expPageSize uint64) (uint64, error) {
	var n uint64
	put := func(b []byte, off uint64) error {
		if _, err := out.WriteAt(b, int64(off)); err != nil {
			return fmt.Errorf("jlm: %v: %w", e.Role, err)
		}
		n += uint64(len(b))
		return nil
	}
	q, packed := packerOf(e.Type)
	// A sheet's bytes go to its expert page when the bank is split out,
	// and back to back inside the entry otherwise.
	sheets, rows := sheetsOf(e.Dims, e.NDim)
	split := expertPaged(e)
	if !packed {
		if !split {
			return n, put(raw, e.QSOff)
		}
		per := len(raw) / int(sheets)
		for sh := 0; sh < int(sheets); sh++ {
			if err := put(raw[sh*per:(sh+1)*per], e.QSOff+uint64(sh)*expPageSize); err != nil {
				return n, err
			}
		}
		return n, nil
	}
	// One sheet at a time, back to back: see sheetsOf. A dense weight is
	// one sheet and takes the same path, so there is no second shape here.
	srcSheet := len(raw) / int(sheets)
	nq, nd, nsc, err := kernels.PackedWords(q, int(rows), e.K())
	if err != nil {
		return n, fmt.Errorf("jlm: %v: %w", e.Role, err)
	}
	for sh := 0; sh < int(sheets); sh++ {
		qs, dw, sc, err := kernels.PackWeights(q, raw[sh*srcSheet:(sh+1)*srcSheet], int(rows), e.K())
		if err != nil {
			return n, fmt.Errorf("jlm: %v sheet %d: %w", e.Role, sh, err)
		}
		sq, sd, ssc := uint64(sh*nq*4), uint64(sh*nd*4), uint64(sh*nsc*4)
		if split {
			sq, sd, ssc = uint64(sh)*expPageSize, uint64(sh)*expPageSize, uint64(sh)*expPageSize
		}
		for _, w := range []struct {
			off uint64
			v   []uint32
		}{
			{e.QSOff + sq, qs},
			{e.DOff + sd, dw},
			{e.SCOff + ssc, sc},
		} {
			if len(w.v) == 0 {
				continue
			}
			if err := put(U32Bytes(w.v), w.off); err != nil {
				return n, err
			}
		}
	}
	return n, nil
}

// journalEvery is how many bytes are written between journal points: the
// most a resumed write does twice. A variable so a gate can journal every
// tensor.
var journalEvery uint64 = 1 << 30

// journalSpacing is journalEvery for a container of total bytes, closer for
// a small one: a write journalled only at its end is never resumed, and a
// model of a few hundred MB is still minutes of a slow link. A thirty-second
// of the container keeps the syncs few.
func journalSpacing(total uint64) uint64 {
	return min(journalEvery, max(total/32, 64<<10))
}

// journalSuffix follows the part's name: out.jlm.part.resume.
const journalSuffix = ".resume"

const journalMagic = "jlm resume v1"

// resumeDigest names everything a resumed write must share with the one that
// wrote the journal: where the bytes come from, the layout every offset is
// taken from, and the build that wrote them (in the fingerprint), so a
// changed converter never finishes another's container.
func resumeDigest(origin string, parts ...[]byte) string {
	h := sha256.New()
	fmt.Fprintf(h, "%d:%s", len(origin), origin)
	for _, p := range parts {
		fmt.Fprintf(h, "%d:", len(p))
		h.Write(p)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func writeJournal(jn, sum string, done int) error {
	tmp := jn + ".tmp"
	if err := os.WriteFile(tmp, []byte(fmt.Sprintf("%s\n%s\n%d\n", journalMagic, sum, done)), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, jn)
}

// readJournal is a journal's digest and count; ok is false for anything that
// is not a whole journal.
func readJournal(jn string) (sum string, done int, ok bool) {
	b, err := os.ReadFile(jn)
	if err != nil || len(b) > 4096 {
		return "", 0, false
	}
	f := strings.Split(strings.TrimSuffix(string(b), "\n"), "\n")
	if len(f) != 3 || f[0] != journalMagic {
		return "", 0, false
	}
	done, err = strconv.Atoi(f[2])
	if err != nil || done < 0 {
		return "", 0, false
	}
	return f[1], done, true
}

// resumeAt is the tensor a write resumes at: the journal's count when the
// journal matches this write and the part is the size this layout gives it,
// else 0.
func resumeAt(tmp, jn, sum string, total uint64, tensors int) int {
	got, done, ok := readJournal(jn)
	if !ok || got != sum || done > tensors {
		return 0
	}
	fi, err := os.Stat(tmp)
	if err != nil || fi.Size() != int64(total) {
		return 0
	}
	return done
}

// Resumable reports how many tensors an interrupted write of dst left on the
// disk, by its journal. Whether the next write resumes from them is decided
// by Write: it must be the same layout, from the same origin, by the same
// build.
func Resumable(dst string) (done int, ok bool) {
	_, done, ok = readJournal(dst + ".part" + journalSuffix)
	return done, ok
}
