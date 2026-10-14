// Package jlm is jitllm's own model container: weights already in the
// device layout, one page per block, the tokenizer travelling with them.
//
// The point is that the runtime stops computing anything about layout. Every
// kernel wants weights transposed so element u of consecutive rows is adjacent
// (kernels/pack.go); repacking a GGUF at page-in ran far slower than the upload
// and dominated inference on a model too big for the card. Converting once
// removes the term: a page-in is a write of a slice of the file.
//
// One page is one block, and the file is the page array: block i occupies
// exactly PageSize bytes at DataOff + i*PageSize, physically padded at
// conversion, so addressing is a multiply and the slot arithmetic has no
// estimate in it.
//
// The device layout is backend-independent: PackWeights takes no target, so
// one container serves CUDA, Vulkan, Metal and the host (scripts/three-way.sh
// runs one model across ptx and spirv devices at once, token-identical to the
// CPU). Formats not yet at native width cost extra bytes over their GGUF; see
// oracle.TestPackedPayloadIsNativeWidth.
package jlm

import (
	"encoding/binary"
	"fmt"
	"runtime/debug"
)

// Ext is the container's file extension, and the only weight format the engine
// runs. A source format reaches convert/ and nothing else.
const Ext = ".jlm"

// Magic is the first 8 bytes of every container and never changes.
//
// The version is a separate field, not part of the magic: "is this ours" and
// "can we read it" are different questions, and a magic that changed per
// revision reported a real container as "not a jlm container".
var Magic = [8]byte{'J', 'I', 'T', 'L', 'L', 'M', 0, 0}

// Version is the container revision, stored as a u32 at offset 8. Bump it for
// any layout change: there is no backward compatibility, because a container
// is rebuilt from its source in seconds. The version covers where bytes live as
// well as what they mean: a structure read with the wrong placement rules finds
// an empty span, not an error (v8/v9 moved tensors into the dense region).
//
// Revisions, newest first, each of which a previous reader would misread:
//
//	v29  a row table (RowTable) after the page arrays, read a row at a time,
//	     and TypeIQ4NL
//	v28  multi-token-prediction blocks kept after the trunk (Config.NMTP) and
//	     the roles that make one (RoleNextnEHProj and its siblings)
//	v27  Q4_K/Q5_K sub-block scales and minima packed as a 96-bit stream, the
//	     GGUF's own 12 bytes (kernels.ScStream)
//	v26  a mixture's experts split out of the block page into a third page
//	     array, one page per (block, expert), evicted LRU
//	v25  a fourth bool per pre-tokenizer stage: HanRuns (kimi-k2)
//	v24  MXFP4's E8M0 block scale stored as a byte
//	v22  YaRN's four numbers and a linear rotary factor in the config record
//	v21  StreamGroups in the header
//	v20  the pager's fill granularity in the header (see chunkFor)
//	v19  three bools per pre-tokenizer stage: CaseRuns, SuffixContract,
//	     PunctSlash (o200k)
//	v18  gemma2's two logit softcaps in the config record
//	v16  the CLIP vision roles and the native Q5_K payload, merged
//	v14  Q4_0/Q8_0 pair two rows' f16 scales in one d word, so the half is
//	     known at emit time from the row index (kernels.DIndex)
//	v6   the container speaks only its own vocabulary: type codes, (role,
//	     block, index) identity, a typed config and pre-tokenizer stages
//	v5   Q6_K's payload is a 4-bit plus a 2-bit plane
//	v3   no tensor is in GGUF layout
const Version = 29

// HeaderBytes is the fixed header. Everything after it is located by offset.
// It is 192 because every offset in this header is u64; squeezing offsets
// into u32 silently truncates a container over 4 GiB.
const HeaderBytes = 192

// Align is the alignment of every span in the file.
//
// 4096 matches Vulkan's minImportedHostPointerAlignment on common devices,
// so an aligned buffer imports with no copy, and page-granular eviction can
// drop a span without straddling.
const Align = 4096

// DenseBlock is the block index of a tensor that belongs to no block -- the
// embedding, the output projection, the final norm.
//
// They are read every token whatever the pager does, so they live in their
// own dense region outside the page array rather than in a block page.
const DenseBlock = -1

// Header is the fixed prefix. All offsets are absolute file offsets.
type Header struct {
	PageSize uint64 // bytes per block page; block i is at DataOff + i*PageSize
	NBlocks  uint32
	NTensors uint32
	CfgOff   uint64 // jlm.Config, the typed model description
	CfgLen   uint64
	VisOff   uint64 // jlm.Vision, present only for a tower
	VisLen   uint64
	VocOff   uint64 // jlm.Vocab, the tokenizer
	VocLen   uint64
	TabOff   uint64 // the tensor table
	TabLen   uint64
	DataOff  uint64 // start of the TEXT page array, Align-aligned
	// The vision tower is a second page array in the same file: vision block i
	// is at VisDataOff + i*VisPageSize. A second array rather than a second
	// size because one page size is wrong in both directions; see Role.Vision.
	VisDataOff  uint64
	VisPageSize uint64
	NVisBlocks  uint32
	DenseOff    uint64 // non-block weights
	DenseLen    uint64
	FPOff       uint64 // what this file was tuned against; see Fingerprint
	FPLen       uint64
	// Chunk is the pager's fill granularity for this container, computed from
	// its own geometry at conversion. See jlm.chunkFor for why it is a stored
	// property of the model rather than a package default.
	Chunk uint64
	// StreamGroups is how many pieces a streamed block's routed selection
	// should be uploaded in. See jlm.streamGroupsFor.
	StreamGroups uint32
	// The third page array, one page per (mixture block, expert). Expert
	// page p is at ExpDataOff + p*ExpPageSize and holds that expert's sheet of
	// every bank in its block; a bank entry's offsets address expert 0's page
	// and expert x is x*ExpPageSize further on. Zero for a dense model.
	NExpPages   uint32
	ExpDataOff  uint64
	ExpPageSize uint64
}

// Fingerprint records what the conversion was tuned for.
//
// Containers are written where they run, so tuning to the machine is
// legitimate, but a copied file must produce a warning and a correct run.
// Compared on load, never enforced.
type Fingerprint struct {
	Host   string
	Device string
	VRAM   uint64
	// Writer identifies the build that produced this container: the module
	// version and the Go toolchain, from the embedded build info. Most of a
	// container's content is converter decisions (e.g. add_bos resolution) that
	// the layout version cannot catch, so a stale converter is otherwise
	// invisible.
	Writer string
}

// WriterID is the build identity Convert stamps into a container.
func WriterID() string {
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		return "unknown"
	}
	rev, dirty := "", ""
	for _, s := range bi.Settings {
		switch s.Key {
		case "vcs.revision":
			if len(s.Value) > 12 {
				rev = s.Value[:12]
			} else {
				rev = s.Value
			}
		case "vcs.modified":
			if s.Value == "true" {
				dirty = "+dirty"
			}
		}
	}
	if rev == "" {
		rev = bi.Main.Version
	}
	return bi.GoVersion + " " + rev + dirty
}

// Entry is one tensor.
//
// Entry is one tensor, identified by three numbers: Role says what it is,
// Block which repeating unit (DenseBlock for none), and Index which expert
// for a per-expert role (-1 otherwise). Name is a diagnostics label.
//
// Lengths are derived, not stored: a quantised tensor's three spans come
// from kernels.PackedWords (held to PackWeights by TestPackedWordsMatchesPack),
// a verbatim one's from its dims and type. Unquantised tensors (norms, biases)
// are carried verbatim, since the container must be self-sufficient.
type Entry struct {
	Role  Role
	Block int32 // DenseBlock for a non-block tensor
	Index int32 // -1 unless the role is stored per expert
	Type  Type
	NDim  uint8
	Dims  [4]uint64 // fastest-varying first: k = Dims[0], rows = Dims[1]
	QSOff uint64    // absolute, Align-aligned. The only span when verbatim.
	DOff  uint64
	SCOff uint64
	Name  string // diagnostics only
}

// Rows and K are ne1 and ne0, matching model.tensor.
func (e *Entry) Rows() int {
	if e.NDim < 2 {
		return 1
	}
	return int(e.Dims[1])
}
func (e *Entry) K() int { return int(e.Dims[0]) }

// The header is magic, version, then offsets:
//
//	0   8   magic "JITLLM\0\0"
//	8   4   version (u32)
//	12  4   reserved, zero
//	16  8   page size ... and so on to HeaderBytes
func (h *Header) encode(b []byte) {
	copy(b[0:8], Magic[:])
	le := binary.LittleEndian
	le.PutUint32(b[8:], Version)
	le.PutUint32(b[12:], 0)
	le.PutUint64(b[16:], h.PageSize)
	le.PutUint32(b[24:], h.NBlocks)
	le.PutUint32(b[28:], h.NTensors)
	le.PutUint64(b[32:], h.CfgOff)
	le.PutUint64(b[40:], h.CfgLen)
	le.PutUint64(b[48:], h.TabOff)
	le.PutUint64(b[56:], h.TabLen)
	le.PutUint64(b[64:], h.DataOff)
	le.PutUint64(b[72:], h.DenseOff)
	le.PutUint64(b[80:], h.DenseLen)
	le.PutUint64(b[88:], h.FPOff)
	le.PutUint64(b[96:], h.FPLen)
	le.PutUint64(b[104:], h.VocOff)
	le.PutUint64(b[112:], h.VocLen)
	le.PutUint64(b[120:], h.VisOff)
	le.PutUint64(b[128:], h.VisLen)
	le.PutUint64(b[136:], h.VisDataOff)
	le.PutUint64(b[144:], h.VisPageSize)
	le.PutUint32(b[152:], h.NVisBlocks)
	le.PutUint64(b[160:], h.Chunk)
	le.PutUint32(b[168:], h.StreamGroups)
	le.PutUint32(b[172:], h.NExpPages)
	le.PutUint64(b[176:], h.ExpDataOff)
	le.PutUint64(b[184:], h.ExpPageSize)
}

// decodeHeader reads and validates the header. A container is untrusted input
// (RULE 9), so every offset and length is checked against the file size
// before anything slices with it: a malformed container is an error, never a
// panic.
func decodeHeader(b []byte, size uint64) (*Header, error) {
	if len(b) < HeaderBytes {
		return nil, fmt.Errorf("jlm: %d bytes is shorter than the %d-byte header", len(b), HeaderBytes)
	}
	if [8]byte(b[0:8]) != Magic {
		return nil, fmt.Errorf("jlm: not a jlm container (magic %q)", b[0:8])
	}
	if v := binary.LittleEndian.Uint32(b[8:]); v != Version {
		return nil, fmt.Errorf("jlm: container is version %d and this build reads %d; "+
			"re-run `jitllm convert` (there is no backward compatibility, on purpose)", v, Version)
	}
	le := binary.LittleEndian
	h := &Header{
		PageSize: le.Uint64(b[16:]),
		NBlocks:  le.Uint32(b[24:]),
		NTensors: le.Uint32(b[28:]),
		CfgOff:   le.Uint64(b[32:]),
		CfgLen:   le.Uint64(b[40:]),
		TabOff:   le.Uint64(b[48:]),
		TabLen:   le.Uint64(b[56:]),
		DataOff:  le.Uint64(b[64:]),
		DenseOff: le.Uint64(b[72:]),
		DenseLen: le.Uint64(b[80:]),
		FPOff:    le.Uint64(b[88:]),
		FPLen:    le.Uint64(b[96:]),
		VocOff:   le.Uint64(b[104:]),
		VocLen:   le.Uint64(b[112:]),
		VisOff:   le.Uint64(b[120:]),
		VisLen:   le.Uint64(b[128:]),

		VisDataOff:   le.Uint64(b[136:]),
		VisPageSize:  le.Uint64(b[144:]),
		NVisBlocks:   le.Uint32(b[152:]),
		Chunk:        le.Uint64(b[160:]),
		StreamGroups: le.Uint32(b[168:]),
		NExpPages:    le.Uint32(b[172:]),
		ExpDataOff:   le.Uint64(b[176:]),
		ExpPageSize:  le.Uint64(b[184:]),
	}
	if h.PageSize == 0 || h.PageSize%Align != 0 {
		return nil, fmt.Errorf("jlm: page size %d is not a non-zero multiple of %d", h.PageSize, Align)
	}
	// Refused rather than repaired: a chunk that is not a power of two of at
	// least Align makes unaligned reads, which O_DIRECT answers with EINVAL. The
	// ceiling is the largest page array, not PageSize: a tower-only container has
	// a degenerate PageSize of Align and megabyte vision pages.
	page := h.PageSize
	page = max(page, h.VisPageSize, h.ExpPageSize)
	if h.Chunk < Align || h.Chunk&(h.Chunk-1) != 0 || (page > 0 && h.Chunk > page) {
		return nil, fmt.Errorf("jlm: chunk %d is not a power of two in [%d, %d]",
			h.Chunk, Align, page)
	}
	if h.DataOff%Align != 0 {
		return nil, fmt.Errorf("jlm: data offset %d is not %d-aligned", h.DataOff, Align)
	}
	for _, s := range []struct {
		name     string
		off, len uint64
	}{
		{"config", h.CfgOff, h.CfgLen},
		{"vocab", h.VocOff, h.VocLen},
		{"vision", h.VisOff, h.VisLen},
		{"table", h.TabOff, h.TabLen},
		{"dense", h.DenseOff, h.DenseLen},
		{"fingerprint", h.FPOff, h.FPLen},
	} {
		if s.off > size || s.len > size || s.off+s.len > size {
			return nil, fmt.Errorf("jlm: %s span [%d,%d) runs past the %d-byte file", s.name, s.off, s.off+s.len, size)
		}
	}
	// Each page array is checked as a whole: its size is a product of two
	// header fields, so it is checked for overflow before it is compared.
	for _, a := range []struct {
		name       string
		n          uint64
		off, psize uint64
	}{
		{"block", uint64(h.NBlocks), h.DataOff, h.PageSize},
		{"vision", uint64(h.NVisBlocks), h.VisDataOff, h.VisPageSize},
		{"expert", uint64(h.NExpPages), h.ExpDataOff, h.ExpPageSize},
	} {
		if a.n == 0 {
			continue
		}
		if a.psize == 0 || a.psize%Align != 0 || a.off%Align != 0 {
			return nil, fmt.Errorf("jlm: %s pages of %d at %d are not %d-aligned", a.name, a.psize, a.off, Align)
		}
		if a.psize > (^uint64(0))/a.n {
			return nil, fmt.Errorf("jlm: %d %s pages of %d bytes overflows", a.n, a.name, a.psize)
		}
		if end := a.off + a.n*a.psize; end > size || end < a.off {
			return nil, fmt.Errorf("jlm: %d %s pages of %d from %d run past the %d-byte file", a.n, a.name, a.psize, a.off, size)
		}
	}
	return h, nil
}

// alignUp rounds n up to the next multiple of Align.
func alignUp(n uint64) uint64 { return (n + Align - 1) &^ uint64(Align-1) }
