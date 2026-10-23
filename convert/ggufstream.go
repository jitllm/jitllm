package convert

import (
	"errors"
	"fmt"
	"io"
	"strings"
	"unsafe"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/format/jlm"
)

// Remote is a source file that is not on this disk, read by byte range: a
// Hub file (hf.Remote). Name must name its bytes immutably -- a pinned
// commit -- because it is what a resumed write checks against.
type Remote interface {
	io.ReaderAt
	Size() int64
	Name() string
}

// StreamGGUFs is FromGGUFs for a GGUF, and optionally its vision tower,
// read by range where it lives: nothing of either is written to disk, so the
// container is the only file the conversion grows, and the writer's fetch of
// the next tensor is the network, overlapped with packing this one.
//
// The converter's GGUF path reads some tensors' values -- to split a fused
// matrix, fold a scale, compare two tables -- and passes most through. Which
// ones is a property of the architecture, so it is not listed: the source is
// built over a reservation of the file (gguf.OpenAt) in which only the
// tensors read so far hold their bytes, the pages the build touched are
// asked of the kernel, those tensors are read and the build runs again, until
// a build touches nothing it was not given. What it never touched is read
// when the writer reaches it. A build that needs more than maxStreamHeld of
// the model in memory is refused rather than run out of memory.
//
// The write is resumable (jlm.Source.Origin): a cancelled or failed stream
// keeps its journal, and the next call for the same files picks up there.
// gguf.ErrNoStream is the answer on a platform that cannot tell what was
// read; the caller downloads instead.
func StreamGGUFs(src, mmproj Remote, dst string, fp jlm.Fingerprint, opts ...Option) (*jlm.Header, error) {
	var o options
	for _, fn := range opts {
		fn(&o)
	}
	if o.q8 {
		return nil, fmt.Errorf("convert: WithQ8 quantizes a safetensors model; %s is a GGUF", src.Name())
	}
	if o.head != "" {
		return nil, fmt.Errorf("convert: WithHead reads a downloaded head; %s is streamed", src.Name())
	}
	var files []*streamFile
	defer func() {
		for _, sf := range files {
			sf.f.Close()
		}
	}()
	open := func(r Remote) (*streamFile, error) {
		f, err := gguf.OpenAt(r.Name(), r, r.Size())
		if err != nil {
			return nil, err
		}
		sf := &streamFile{f: f, r: r, held: make([]bool, len(f.Tensors))}
		files = append(files, sf)
		return sf, nil
	}
	text, err := open(src)
	if err != nil {
		return nil, err
	}
	var vf *meta.File
	mmName := ""
	if mmproj != nil {
		tower, err := open(mmproj)
		if err != nil {
			return nil, err
		}
		vf, mmName = tower.f, mmproj.Name()
	}

	// Small tensors -- norms, biases, scales, tables -- are what a build reads
	// most, and too small for a page of their own to show it: held from the
	// start.
	for _, sf := range files {
		for i := range sf.f.Tensors {
			if sf.f.Tensors[i].NBytes < streamSmall {
				sf.held[i] = true
			}
		}
	}
	var s *jlm.Source
	for {
		var held uint64
		for _, sf := range files {
			n, err := sf.fill()
			if err != nil {
				return nil, err
			}
			held += n
		}
		if held > maxStreamHeld {
			return nil, fmt.Errorf("convert: %s reads %d MB of its weights to convert: %w",
				src.Name(), held>>20, ErrStreamHolds)
		}
		var berr error
		s, berr = buildCaught(func() (*jlm.Source, error) {
			s, err := buildGGUF(text.f, vf, src.Name(), mmName, o)
			if err != nil {
				return nil, err
			}
			// A build may defer reading to the writer (a stacked bank's
			// Load joins its experts' bytes then): each is run here too, so
			// what it reads is held by the time the writer runs it.
			for i := range s.Tensors {
				if s.Tensors[i].Load != nil {
					if _, err := s.Tensors[i].Load(); err != nil {
						return nil, err
					}
				}
			}
			return s, nil
		})
		grew := false
		for _, sf := range files {
			g, err := sf.touched()
			if err != nil {
				return nil, err
			}
			grew = grew || g
		}
		if !grew {
			if berr != nil {
				return nil, berr
			}
			break
		}
	}

	// What the build handed over untouched is read when the writer gets to it.
	for i := range s.Tensors {
		t := &s.Tensors[i]
		for _, sf := range files {
			if off, ok := sf.within(t.Data); ok && !sf.allHeld(off, uint64(len(t.Data))) {
				r, n := sf.r, len(t.Data)
				t.Data, t.Load = nil, func() ([]byte, error) {
					b := make([]byte, n)
					if _, err := r.ReadAt(b, int64(off)); err != nil {
						return nil, err
					}
					return b, nil
				}
				break
			}
		}
	}
	var origin strings.Builder
	for _, sf := range files {
		fmt.Fprintf(&origin, "gguf %s %d\n", sf.r.Name(), sf.r.Size())
	}
	s.Origin = origin.String()
	if fp.Writer == "" {
		fp.Writer = jlm.WriterID()
	}
	if err := stateTower(s); err != nil {
		return nil, err
	}
	return jlm.Write(dst, s, fp)
}

// ErrStreamHolds is StreamGGUFs refusing a model whose conversion reads more
// of its weights than a stream holds in memory (maxStreamHeld) -- a GGUF
// that ships each expert as its own matrix, stacked at conversion. Nothing
// is written; the caller downloads instead.
var ErrStreamHolds = errors.New("more than a streamed conversion holds in memory")

// streamSmall is the size below which a streamed tensor is read up front.
const streamSmall = 64 << 10

// maxStreamHeld bounds the bytes of a streamed model a build may need in
// memory at once.
const maxStreamHeld = 4 << 30

// streamFile is one streamed GGUF and which of its tensors hold their bytes.
type streamFile struct {
	f    *meta.File
	r    Remote
	held []bool
}

// fill reads every held tensor into the reservation, again on every build:
// a build may have written into one it was given. It returns the bytes held.
func (sf *streamFile) fill() (uint64, error) {
	var n uint64
	for i := range sf.f.Tensors {
		if !sf.held[i] {
			continue
		}
		t := &sf.f.Tensors[i]
		if _, err := sf.r.ReadAt(sf.f.Bytes(t), int64(sf.f.DataStart+t.Off)); err != nil {
			return 0, err
		}
		n += t.NBytes
	}
	return n, nil
}

// touched marks held every tensor a build read or wrote a page of, and
// reports whether there was one. Only the pages wholly inside a tensor are
// asked about: a page it shares with its neighbour says nothing about which
// of the two was read.
func (sf *streamFile) touched() (bool, error) {
	in, err := gguf.Resident(sf.f.Data)
	if err != nil {
		return false, err
	}
	ps := uint64(gguf.PageSize())
	grew := false
	for i := range sf.f.Tensors {
		if sf.held[i] {
			continue
		}
		t := &sf.f.Tensors[i]
		lo := sf.f.DataStart + t.Off
		for p := (lo + ps - 1) / ps; (p+1)*ps <= lo+t.NBytes; p++ {
			if in[p] {
				sf.held[i], grew = true, true
				break
			}
		}
	}
	return grew, nil
}

// within reports where b lies in the file, when it is a slice of the
// reservation.
func (sf *streamFile) within(b []byte) (uint64, bool) {
	if len(b) == 0 || len(sf.f.Data) == 0 {
		return 0, false
	}
	base := uintptr(unsafe.Pointer(unsafe.SliceData(sf.f.Data)))
	p := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	if p < base || p-base+uintptr(len(b)) > uintptr(len(sf.f.Data)) {
		return 0, false
	}
	return uint64(p - base), true
}

// allHeld reports whether bytes [off, off+n) of the file are all held
// tensors' bytes, which a build may have changed in place and must be
// written as they are.
func (sf *streamFile) allHeld(off, n uint64) bool {
	for i := range sf.f.Tensors {
		t := &sf.f.Tensors[i]
		lo := sf.f.DataStart + t.Off
		if lo+t.NBytes > off && lo < off+n && !sf.held[i] {
			return false
		}
	}
	return off >= sf.f.DataStart
}

// buildCaught runs one build over a partly filled reservation, where a
// tensor not yet read is zeros and a build may fail on them -- even panic,
// dividing by a zero it read. That failure is the build asking for bytes:
// the pages it touched say which, and the next build has them.
func buildCaught(build func() (*jlm.Source, error)) (s *jlm.Source, err error) {
	defer func() {
		if r := recover(); r != nil {
			s, err = nil, fmt.Errorf("convert: %v", r)
		}
	}()
	return build()
}
