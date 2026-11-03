package gguf

import (
	"fmt"
	"io"

	"github.com/jitllm/jitllm/convert/meta"
)

// ErrNoStream is OpenAt's answer on a platform where a streamed GGUF cannot
// be told which of its bytes the converter read (see Streamable).
const ErrNoStream streamError = "gguf: this platform cannot convert a GGUF while it streams"

type streamError string

func (e streamError) Error() string { return string(e) }

// OpenAt parses a GGUF read through r, a file of size bytes that is not on
// this disk -- a Hub file read by range. Data is a reservation of the file's
// size in anonymous memory, untouched except for the header: a tensor's bytes
// are there only once its caller fills them (r.ReadAt into Bytes(t)), and
// Resident tells which pages anything has touched. Close releases it.
//
// The header is read in growing prefixes until it parses, so a tokenizer's
// megabytes are read once and the weights not at all.
func OpenAt(name string, r io.ReaderAt, size int64) (*meta.File, error) {
	if !Streamable() {
		return nil, ErrNoStream
	}
	if size <= 0 {
		return nil, fmt.Errorf("gguf: %s: %d bytes", name, size)
	}
	whole, err := reserve(int(size))
	if err != nil {
		return nil, fmt.Errorf("gguf: %s: %w", name, err)
	}
	release := func() error { return unreserve(whole) }
	var got int64
	for n := min(int64(1<<20), size); ; n = min(2*n, size) {
		if _, err := r.ReadAt(whole[got:n], got); err != nil {
			release()
			return nil, fmt.Errorf("gguf: %s: %w", name, err)
		}
		got = n
		f, err := parse(whole[:n], whole, name, release)
		if err == nil {
			return f, nil
		}
		// A header longer than the prefix fails to parse; at maxHeader, or at
		// the whole file, the failure is the file's.
		if n == size || n >= maxHeader {
			release()
			return nil, err
		}
	}
}

// maxHeader bounds how much of a streamed file is read looking for the end of
// its header: a vocabulary of a quarter million tokens is a few tens of MB.
const maxHeader = 512 << 20

// Streamable reports whether this platform can tell which bytes of an
// anonymous reservation were read: a page read for the first time is
// resident afterwards and an unread one is not. It is asked of the kernel,
// once, rather than assumed.
func Streamable() bool { return probe() }

func probe() bool {
	ps := PageSize()
	b, err := reserve(4 * ps)
	if err != nil {
		return false
	}
	defer unreserve(b)
	first := b[ps+1]
	in, err := Resident(b)
	return err == nil && first == 0 && len(in) == 4 && !in[0] && in[1] && !in[2] && !in[3]
}
