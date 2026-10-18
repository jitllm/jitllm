package jlm

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// Container migrations. The reader opens one version (Version); a container
// written by an older build is brought forward by a chain of migrations, one
// per version, each rewriting only what that version changed and carrying
// every other byte across. A container is self-sufficient -- its source may be
// gone -- so this chain is how an old one survives a format change.
//
// A migration that changes no byte but the version word runs in place. One
// that changes what tensors exist, or their sizes, rewrites the container
// through Write from its own entries, their stored planes carried verbatim
// (Tensor.Stored), so the result is byte for byte the container a fresh
// conversion at the new version writes (TestUpgradeIsAFreshConversion).

// migrations[v] rewrites the version v container at path as version v+1.
// Every Version bump adds the entry for the version before it
// (TestEveryVersionHasItsMigration).
// Filled in init: a migration opens the container, and Open's refusal asks
// which versions upgrade.
var migrations = map[uint32]func(path string) error{}

func init() {
	migrations[26] = migrate26
	migrations[27] = migrate27
	migrations[28] = migrate28
	migrations[29] = migrate29
}

// Upgradable reports whether a chain of migrations reaches Version from v.
func Upgradable(v uint32) bool {
	if v >= Version {
		return false
	}
	for ; v < Version; v++ {
		if migrations[v] == nil {
			return false
		}
	}
	return true
}

// ReadVersion is the version word of the container at path.
func ReadVersion(path string) (uint32, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	var b [12]byte
	if _, err := f.ReadAt(b[:], 0); err != nil {
		return 0, fmt.Errorf("jlm: reading %s: %w", path, err)
	}
	if string(b[0:8]) != Magic {
		return 0, fmt.Errorf("jlm: %s is not a jlm container", path)
	}
	return binary.LittleEndian.Uint32(b[8:]), nil
}

// Upgrade brings the container at path to Version through the migrations,
// and reports the version it started at. A current container is left as it
// is; one no chain reaches is refused, naming the convert command.
func Upgrade(path string) (from uint32, err error) {
	from, err = ReadVersion(path)
	if err != nil {
		return 0, err
	}
	if from == Version {
		return from, nil
	}
	if !Upgradable(from) {
		return from, fmt.Errorf("jlm: %s is version %d and no migration reaches %d from it; "+
			"re-run `jitllm convert` on its source", path, from, Version)
	}
	for v := from; v < Version; v++ {
		if err := migrations[v](path); err != nil {
			return from, fmt.Errorf("jlm: upgrading %s from version %d to %d: %w", path, v, v+1, err)
		}
		if got, err := ReadVersion(path); err != nil || got != v+1 {
			return from, fmt.Errorf("jlm: the migration from version %d left %s at version %d (%v)", v, path, got, err)
		}
	}
	return from, nil
}

// setVersion writes v into path's version word in place.
func setVersion(path string, v uint32) error {
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		return err
	}
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	if _, err := f.WriteAt(b[:], 8); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// storedBytes reads entry e's bytes as the container stores them: a verbatim
// type's bytes, every sheet back to back; a packed type's planes, every
// sheet's QS, then D, then SC (Tensor.Stored's layout).
func (f *File) storedBytes(e *Entry) ([]byte, error) {
	sheets, rows := sheetsOf(e.Dims, e.NDim)
	split := expertPaged(e)
	// The exact bytes of one sheet's plane, as writeTensor puts them (spans'
	// lengths carry the alignment padding).
	type plane struct{ off, per uint64 }
	var planes []plane
	if q, ok := packerOf(e.Type); ok {
		nq, nd, nsc, err := kernels.PackedWords(q, int(rows), e.K())
		if err != nil {
			return nil, err
		}
		if f.ver < 27 && kernels.ScStream(q) {
			// v26 spent a word on two sub-blocks' pairs, four a super-block.
			nsc = int(rows) * e.K() / 256 * scWords26
		}
		planes = []plane{{e.QSOff, uint64(nq) * 4}, {e.DOff, uint64(nd) * 4}, {e.SCOff, uint64(nsc) * 4}}
	} else {
		per, err := verbatimBytes(e.Type, rows*e.Dims[0])
		if err != nil {
			return nil, err
		}
		planes = []plane{{e.QSOff, per}}
	}
	var total uint64
	for _, pl := range planes {
		total += pl.per * sheets
	}
	out := make([]byte, total)
	at := uint64(0)
	for _, pl := range planes {
		per := pl.per
		if per == 0 {
			continue
		}
		for sh := uint64(0); sh < sheets; sh++ {
			src := pl.off + sh*per
			if split {
				src = pl.off + sh*f.H.ExpPageSize
			}
			if _, err := f.f.ReadAt(out[at:at+per], int64(src)); err != nil {
				return nil, fmt.Errorf("jlm: reading %v block %d: %w", e.Role, e.Block, err)
			}
			at += per
		}
	}
	return out, nil
}

// migrate29 is version 29 to 30: the audio tower's depthwise bias folded into
// its BatchNorm shift (RoleABNShift becomes dwB*scale + shift, RoleADWB goes).
// A container with no audio tower changes only its version word, in place.
func migrate29(path string) error {
	f, err := openAt(path, 29)
	if err != nil {
		return err
	}
	if f.vis == nil || f.vis.Audio == nil {
		f.Close()
		return setVersion(path, 30)
	}
	defer f.Close()
	f32 := func(e *Entry) ([]float32, error) {
		if e.Type != TypeF32 {
			return nil, fmt.Errorf("%v block %d is %v, not f32", e.Role, e.Block, e.Type)
		}
		b, err := f.storedBytes(e)
		if err != nil {
			return nil, err
		}
		v := make([]float32, len(b)/4)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		return v, nil
	}
	src := &Source{Config: f.cfg, Vision: f.vis, Vocab: f.voc}
	for i := range f.tab {
		e := &f.tab[i]
		if e.Role == RoleADWB {
			continue
		}
		t := f.carried(e)
		if e.Role == RoleABNShift {
			dw, ok := f.Find(RoleADWB, e.Block, e.Index)
			sc, ok2 := f.Find(RoleABNScale, e.Block, e.Index)
			if !ok || !ok2 {
				return fmt.Errorf("audio block %d carries its BatchNorm shift without the depthwise bias and scale", e.Block)
			}
			bias, err := f32(dw)
			if err != nil {
				return err
			}
			scale, err := f32(sc)
			if err != nil {
				return err
			}
			shift, err := f32(e)
			if err != nil {
				return err
			}
			if len(bias) != len(shift) || len(scale) != len(shift) {
				return fmt.Errorf("audio block %d: depthwise bias %d, BatchNorm %d and %d wide",
					e.Block, len(bias), len(scale), len(shift))
			}
			b := make([]byte, 4*len(shift))
			for c := range shift {
				binary.LittleEndian.PutUint32(b[4*c:], math.Float32bits(bias[c]*scale[c]+shift[c]))
			}
			t.Load, t.Data = nil, b
		}
		src.Tensors = append(src.Tensors, t)
	}
	return f.publish(path, src, 30)
}

// carried is entry e as Write takes it back: its stored planes, read when the
// writer reaches it, placed without packing.
func (f *File) carried(e *Entry) Tensor {
	return Tensor{Role: e.Role, Block: e.Block, Index: e.Index, Type: e.Type, NDim: e.NDim,
		Dims: e.Dims, Name: e.Name, Stored: true, Load: func() ([]byte, error) { return f.storedBytes(e) }}
}

// publish writes src as a version ver container in place of the one f was opened from,
// beside it, and moves it over path once f is closed: Windows refuses to
// replace a file that is still open. One streaming copy; the peak is a
// tensor.
func (f *File) publish(path string, src *Source, ver uint32) error {
	next := path + ".upgrade"
	if _, err := writeAt(next, src, f.FP, ver); err != nil {
		f.Close()
		os.Remove(next)
		return err
	}
	f.Close()
	if err := os.Rename(next, path); err != nil {
		os.Remove(next)
		return fmt.Errorf("jlm: publishing %s: %w", path, err)
	}
	return nil
}

// migrate28 is version 28 to 29. v29 added the row table (placed after every
// page array, and only for RoleNGramTable, which no v28 writer knew) and
// TypeIQ4NL (a type code no v28 container carries). Nothing a v28 container
// holds moved or changed meaning, so only the version word changes, in place.
func migrate28(path string) error {
	f, err := openAt(path, 28)
	if err != nil {
		return err
	}
	f.Close()
	return setVersion(path, 29)
}

// migrate27 is version 27 to 28. v28 put the count of multi-token-prediction
// blocks (Config.NMTP) into the config record after NExpertGroupUsed; a v27
// container has none, so it is zero. The record grows by four bytes, which
// moves every offset after it, so the container is rewritten with every
// tensor's planes carried as they are.
func migrate27(path string) error {
	f, err := openAt(path, 27)
	if err != nil {
		return err
	}
	src := &Source{Config: f.cfg, Vision: f.vis, Vocab: f.voc}
	for i := range f.tab {
		src.Tensors = append(src.Tensors, f.carried(&f.tab[i]))
	}
	return f.publish(path, src, 28)
}

// scWords26 is the SC words a row's super-block held in a v26 Q4_K or Q5_K
// tensor: eight sub-blocks' 6-bit scale and minimum, a byte each, two
// sub-blocks a word.
const scWords26 = 4

// migrate26 is version 26 to 27. v27 packs Q4_K's and Q5_K's sub-block scales
// and minima as a 96-bit stream, three words a super-block (kernels.ScStream),
// where v26 gave each 6-bit field a byte, four words. The fields are the same
// numbers, so the SC plane of every such tensor is re-laid from the v26 one;
// its QS and D planes, and every other tensor, are carried as they are. The
// planes shrink, which moves every offset after the first, so the container
// is rewritten.
func migrate26(path string) error {
	f, err := openAt(path, 26)
	if err != nil {
		return err
	}
	src := &Source{Config: f.cfg, Vision: f.vis, Vocab: f.voc}
	for i := range f.tab {
		e := &f.tab[i]
		t := f.carried(e)
		if q, ok := packerOf(e.Type); ok && kernels.ScStream(q) {
			t.Load = func() ([]byte, error) { return f.restream26(e, q) }
		}
		src.Tensors = append(src.Tensors, t)
	}
	return f.publish(path, src, 27)
}

// restream26 is a v26 Q4_K or Q5_K entry's stored planes with every sheet's SC
// plane re-laid as v27's stream. v26 word w of a row (sc[w*rows+r]) held
// sub-blocks 2w and 2w+1: the scale in bits 0-7 and 16-23, the minimum in
// 8-15 and 24-31. v27 word w of super-block s is sc[(s*3+w)*rows+r], each
// field where kernels.ScField puts it.
func (f *File) restream26(e *Entry, q kernels.Quant) ([]byte, error) {
	b, err := f.storedBytes(e)
	if err != nil {
		return nil, err
	}
	sheets, rows := sheetsOf(e.Dims, e.NDim)
	nq, nd, nsc, err := kernels.PackedWords(q, int(rows), e.K())
	if err != nil {
		return nil, err
	}
	nsuper := e.K() / 256
	old := int(rows) * nsuper * scWords26
	head := (nq + nd) * 4 * int(sheets)
	if len(b) != head+old*4*int(sheets) || nsc != int(rows)*nsuper*kernels.ScStreamWords {
		return nil, fmt.Errorf("jlm: %s: %d stored bytes for a v26 %v of %d sheets of %d rows",
			e.Name, len(b), e.Type, sheets, rows)
	}
	out := make([]byte, head+nsc*4*int(sheets))
	copy(out, b[:head])
	le := binary.LittleEndian
	for sh := 0; sh < int(sheets); sh++ {
		in := b[head+sh*old*4:]
		dst := out[head+sh*nsc*4:]
		for r := 0; r < int(rows); r++ {
			for s := 0; s < nsuper; s++ {
				var ws [kernels.ScStreamWords]uint32
				for sub := 0; sub < 8; sub++ {
					si := s*8 + sub
					v := le.Uint32(in[4*((si/2)*int(rows)+r):])
					lane := uint(si%2) * 16
					scPut26(&ws, sub, false, uint8(v>>lane))
					scPut26(&ws, sub, true, uint8(v>>(lane+8)))
				}
				for w, v := range ws {
					le.PutUint32(dst[4*((s*kernels.ScStreamWords+w)*int(rows)+r):], v)
				}
			}
		}
	}
	return out, nil
}

// scPut26 ORs a 6-bit field into a super-block's stream words, where
// kernels.ScField places it.
func scPut26(ws *[kernels.ScStreamWords]uint32, sub int, min bool, v uint8) {
	w, b, hi := kernels.ScField(sub, min)
	ws[w] |= uint32(v&63) << uint(b)
	if hi > 0 {
		ws[w+1] |= uint32(v&63) >> uint(6-hi)
	}
}
