package jlm

// SetJournalEvery sets how many bytes a resumable write puts down between
// journal points, and returns the call that puts it back.
func SetJournalEvery(n uint64) func() {
	old := journalEvery
	journalEvery = n
	return func() { journalEvery = old }
}

// SetReadHook runs h before every file read with the read's offset; a non-nil
// result fails the read, and the read proper follows otherwise. It is how a
// gate holds, counts or fails reads.
func (f *File) SetReadHook(h func(off int64) error) {
	f.readHook = func(dst []byte, off int64) error {
		if err := h(off); err != nil {
			return err
		}
		_, err := f.f.ReadAt(dst, off)
		return err
	}
}

// PageOffset is where page p's bytes start in the file and how many there are.
func (f *File) PageOffset(p int) (int64, uint64) {
	off, n := f.pageAt(p)
	return int64(off), n
}

// SpeculationHeld is how many frames hold a guess nobody has demanded.
func (f *File) SpeculationHeld() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.specHeld
}
