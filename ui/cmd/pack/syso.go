package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode/utf16"
)

// Resource types (winuser.h).
const (
	rtIcon      = 3
	rtGroupIcon = 14
	rtVersion   = 16
)

// langEnUS is the language every resource is filed under, with code page
// 1200 (UTF-16) in the version's translation table.
const langEnUS = 0x0409

// icoSizes() are the images the Windows icon carries: what Explorer, the task
// bar and Alt-Tab ask for at 100-200% scale.
func icoSizes() []int {
	return []int{16, 24, 32, 48, 64, 128, 256}
}

// resource is one leaf of the resource tree.
type resource struct {
	typ, id uint16
	data    []byte
}

// syso is a COFF object holding one .rsrc section: the resource tree with the
// icon images, the icon group that names them, and the version information.
// The Go linker merges a .syso's .rsrc section into the executable's resource
// directory (cmd/link/internal/loadpe).
func syso(arch, version string) ([]byte, error) {
	var machine, relType uint16
	switch arch {
	case "amd64":
		machine, relType = 0x8664, 3 // IMAGE_REL_AMD64_ADDR32NB
	case "arm64":
		machine, relType = 0xAA64, 2 // IMAGE_REL_ARM64_ADDR32NB
	default:
		return nil, fmt.Errorf("no Windows resource for GOARCH %q", arch)
	}
	vi, err := versionInfo(version)
	if err != nil {
		return nil, err
	}

	res := []resource{{rtVersion, 1, vi}}
	var group bytes.Buffer
	binary.Write(&group, binary.LittleEndian, [3]uint16{0, 1, uint16(len(icoSizes()))}) // GRPICONDIR
	for i, s := range icoSizes() {
		p, err := iconPNG(s)
		if err != nil {
			return nil, err
		}
		id := uint16(i + 1)
		res = append(res, resource{rtIcon, id, p})
		dim := uint8(s) // 256 is written as 0
		group.Write([]byte{dim, dim, 0, 0})
		binary.Write(&group, binary.LittleEndian, struct {
			Planes, BitCount uint16
			Bytes            uint32
			ID               uint16
		}{1, 32, uint32(len(p)), id})
	}
	res = append(res, resource{rtGroupIcon, 1, group.Bytes()})

	sec, relocs := rsrcSection(res)
	return coff(machine, relType, sec, relocs), nil
}

// rsrcSection lays out the three-level resource directory (type, id,
// language) followed by the data entries and the data. It returns the section
// and the offsets of the data entries' RVA fields, which the linker relocates.
func rsrcSection(res []resource) ([]byte, []uint32) {
	sort.Slice(res, func(i, j int) bool {
		if res[i].typ != res[j].typ {
			return res[i].typ < res[j].typ
		}
		return res[i].id < res[j].id
	})
	var types []uint16
	byType := map[uint16][]resource{}
	for _, r := range res {
		if len(byType[r.typ]) == 0 {
			types = append(types, r.typ)
		}
		byType[r.typ] = append(byType[r.typ], r)
	}

	const dirHdr, entry, dataEntry = 16, 8, 16
	// Sizes first: the root, a directory per type, a directory per leaf.
	off := uint32(dirHdr + entry*len(types))
	typeDir := map[uint16]uint32{}
	for _, t := range types {
		typeDir[t] = off
		off += uint32(dirHdr + entry*len(byType[t]))
	}
	langDir := make([]uint32, len(res))
	for i := range res {
		langDir[i] = off
		off += dirHdr + entry
	}
	dataEnt := make([]uint32, len(res))
	for i := range res {
		dataEnt[i] = off
		off += dataEntry
	}
	dataAt := make([]uint32, len(res))
	for i, r := range res {
		off = (off + 7) &^ 7
		dataAt[i] = off
		off += uint32(len(r.data))
	}

	b := make([]byte, off)
	le := binary.LittleEndian
	dir := func(at uint32, n int) uint32 {
		le.PutUint16(b[at+14:], uint16(n)) // NumberOfIdEntries
		return at + dirHdr
	}
	put := func(at uint32, id uint16, target uint32, sub bool) {
		le.PutUint32(b[at:], uint32(id))
		if sub {
			target |= 0x80000000
		}
		le.PutUint32(b[at+4:], target)
	}
	e := dir(0, len(types))
	for _, t := range types {
		put(e, t, typeDir[t], true)
		e += entry
	}
	i := 0
	var relocs []uint32
	for _, t := range types {
		e := dir(typeDir[t], len(byType[t]))
		for _, r := range byType[t] {
			put(e, r.id, langDir[i], true)
			e += entry
			put(dir(langDir[i], 1), langEnUS, dataEnt[i], false)
			le.PutUint32(b[dataEnt[i]:], dataAt[i]) // RVA: the addend, relocated
			le.PutUint32(b[dataEnt[i]+4:], uint32(len(r.data)))
			relocs = append(relocs, dataEnt[i])
			copy(b[dataAt[i]:], r.data)
			i++
		}
	}
	return b, relocs
}

// coff wraps the section in an object file: the header, one section header,
// the data, its relocations against the section's own symbol, and a symbol
// table with that one symbol and an empty string table.
func coff(machine, relType uint16, sec []byte, relocs []uint32) []byte {
	const hdr, shdr, rel, sym = 20, 40, 10, 18
	dataAt := uint32(hdr + shdr)
	relAt := dataAt + uint32(len(sec))
	symAt := relAt + uint32(rel*len(relocs))

	var b bytes.Buffer
	w := func(v any) { binary.Write(&b, binary.LittleEndian, v) }
	w(machine)
	w(uint16(1)) // NumberOfSections
	w(uint32(0)) // TimeDateStamp: none, so the object is reproducible
	w(symAt)     // PointerToSymbolTable
	w(uint32(1)) // NumberOfSymbols
	w(uint16(0)) // SizeOfOptionalHeader
	w(uint16(0)) // Characteristics
	b.WriteString(".rsrc\x00\x00\x00")
	w(uint32(0)) // VirtualSize
	w(uint32(0)) // VirtualAddress
	w(uint32(len(sec)))
	w(dataAt)
	w(relAt)
	w(uint32(0)) // PointerToLinenumbers
	w(uint16(len(relocs)))
	w(uint16(0))          // NumberOfLinenumbers
	w(uint32(0x40000040)) // IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ
	b.Write(sec)
	for _, r := range relocs {
		w(r)         // VirtualAddress: the field's offset in the section
		w(uint32(0)) // SymbolTableIndex
		w(relType)
	}
	b.WriteString(".rsrc\x00\x00\x00")
	w(uint32(0))   // Value
	w(uint16(1))   // SectionNumber
	w(uint16(0))   // Type
	b.WriteByte(3) // IMAGE_SYM_CLASS_STATIC
	b.WriteByte(0) // NumberOfAuxSymbols
	w(uint32(4))   // the string table: its own length, nothing in it
	return b.Bytes()
}

// versionInfo is the VS_VERSIONINFO resource Explorer's Details tab and the
// task manager read the program's name and version from.
func versionInfo(version string) ([]byte, error) {
	v, err := quad(version)
	if err != nil {
		return nil, err
	}
	var fixed bytes.Buffer
	binary.Write(&fixed, binary.LittleEndian, [13]uint32{
		0xFEEF04BD, 0x00010000, // signature, structure version
		uint32(v[0])<<16 | uint32(v[1]), uint32(v[2])<<16 | uint32(v[3]), // file version
		uint32(v[0])<<16 | uint32(v[1]), uint32(v[2])<<16 | uint32(v[3]), // product version
		0x3F, 0, // flags mask, flags
		0x00040004, // VOS_NT_WINDOWS32
		1, 0,       // VFT_APP, no subtype
		0, 0, // no date
	})
	strs := [][2]string{
		{"CompanyName", "jitllm"},
		{"FileDescription", "jitllm"},
		{"FileVersion", version},
		{"InternalName", "jitllm-ui"},
		{"LegalCopyright", "Apache-2.0"},
		{"OriginalFilename", "jitllm-ui.exe"},
		{"ProductName", "jitllm"},
		{"ProductVersion", version},
	}
	var kids [][]byte
	for _, s := range strs {
		val := wstr(s[1])
		kids = append(kids, verNode(s[0], 1, val, uint16(len(val)/2), nil))
	}
	table := verNode(fmt.Sprintf("%04X04B0", langEnUS), 1, nil, 0, kids)
	sfi := verNode("StringFileInfo", 1, nil, 0, [][]byte{table})
	var tr bytes.Buffer
	binary.Write(&tr, binary.LittleEndian, [2]uint16{langEnUS, 1200})
	vfi := verNode("VarFileInfo", 1, nil, 0, [][]byte{verNode("Translation", 0, tr.Bytes(), uint16(tr.Len()), nil)})
	return verNode("VS_VERSION_INFO", 0, fixed.Bytes(), uint16(fixed.Len()), [][]byte{sfi, vfi}), nil
}

// verNode is one node of the version tree: its length, its value's length (in
// characters for a string, bytes otherwise), its type, its key, then the value
// and the children, each starting on a 32-bit boundary.
func verNode(key string, typ uint16, value []byte, valueLen uint16, kids [][]byte) []byte {
	b := make([]byte, 6)
	binary.LittleEndian.PutUint16(b[2:], valueLen)
	binary.LittleEndian.PutUint16(b[4:], typ)
	b = append(b, wstr(key)...)
	pad := func() {
		for len(b)%4 != 0 {
			b = append(b, 0)
		}
	}
	pad()
	b = append(b, value...)
	for _, k := range kids {
		pad()
		b = append(b, k...)
	}
	binary.LittleEndian.PutUint16(b, uint16(len(b)))
	return b
}

// wstr is s in UTF-16LE with its terminating zero.
func wstr(s string) []byte {
	u := utf16.Encode([]rune(s))
	b := make([]byte, 2*len(u)+2)
	for i, c := range u {
		binary.LittleEndian.PutUint16(b[2*i:], c)
	}
	return b
}

// quad is a version as Windows states one, four 16-bit numbers. A goreleaser
// version's suffix ("1.2.4-next", "1.2.3-rc1") is dropped from the numbers
// and kept in the strings.
func quad(version string) ([4]uint16, error) {
	var q [4]uint16
	core, _, _ := strings.Cut(strings.TrimPrefix(version, "v"), "-")
	parts := strings.Split(core, ".")
	if len(parts) > 4 {
		return q, fmt.Errorf("version %q has more than four numbers", version)
	}
	for i, p := range parts {
		n, err := strconv.ParseUint(p, 10, 16)
		if err != nil {
			return q, fmt.Errorf("version %q: %v", version, err)
		}
		q[i] = uint16(n)
	}
	return q, nil
}
