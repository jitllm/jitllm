package tier

// A chunk of a stream-major residual (AltUp, DeepSeek V4, Kimi-K3, Qwen4Exp)
// holds stream k of row r at (k*n+r)*d. Rows [off, off+m) of it are not one
// slice but m rows of every stream, so a fallback that runs a chunk in parts
// (sub-chunks, or a row at a time) gathers them into a chunk of m rows,
// stream-major again, and puts the result back.

// cutRows is rows [off, off+m) of the n-row chunk x of `streams` streams of
// width d: a slice of x itself for one stream, otherwise a gathered copy in a
// buffer taken from the tier (returned by uncutRows). The buffer is kept, so
// a warm cut allocates nothing.
func (g *devTier) cutRows(x []float32, streams, d, n, off, m int) []float32 {
	if streams <= 1 {
		return x[off*d : (off+m)*d]
	}
	g.mu.Lock()
	buf := g.cutBuf
	g.cutBuf = nil
	g.mu.Unlock()
	buf = growF32(buf, streams*m*d)
	for k := 0; k < streams; k++ {
		at := off
		if k > 0 && rowsFaulted("streamcutrows") {
			// The violation: the streams after the first are cut one row on.
			at = (off + 1) % (n - m + 1)
		}
		copy(buf[k*m*d:(k+1)*m*d], x[(k*n+at)*d:(k*n+at+m)*d])
	}
	return buf
}

// uncutRows puts the rows cutRows gathered back into the chunk and hands the
// buffer back to the tier. For one stream the cut was a view of x and there
// is nothing to do. The violation streamcutrows puts the later streams one
// row on, as cutRows takes them.
func (g *devTier) uncutRows(x, v []float32, streams, d, n, off, m int) {
	if streams <= 1 {
		return
	}
	for k := 0; k < streams; k++ {
		at := off
		if k > 0 && rowsFaulted("streamcutrows") {
			at = (off + 1) % (n - m + 1)
		}
		copy(x[(k*n+at)*d:(k*n+at+m)*d], v[k*m*d:(k+1)*m*d])
	}
	g.mu.Lock()
	g.cutBuf = v[:0]
	g.mu.Unlock()
}

// rowInputs are a session's per-row inputs for its next call: the rows'
// per-layer embeddings, token ids, n-gram rows and rotary marks
// (devSess.pleHost, tokIDs, ngHost, qsaMarks). A call over rows [off, off+m)
// of an n-row chunk reads them from its own first row, so a fallback that
// runs the chunk in parts hands each part its rows' share.
type rowInputs struct {
	ple, ng []float32
	ids     []int32
	marks   []int32
}

// cutInputs replaces session sid's per-row inputs with those of rows
// [off, off+m) of an n-row chunk and returns what they were, for
// restoreInputs. An input the chunk did not carry stays absent.
func (g *devTier) cutInputs(sid uint64, n, off, m int) rowInputs {
	g.mu.Lock()
	defer g.mu.Unlock()
	ds := g.sessOf(sid)
	was := rowInputs{ple: ds.pleHost, ng: ds.ngHost, ids: ds.tokIDs, marks: ds.qsaMarks}
	if off == 0 && m == n {
		return was
	}
	ds.pleHost = cutRowsOf(was.ple, n, off, m)
	ds.ngHost = cutRowsOf(was.ng, n, off, m)
	if len(was.ids) >= n {
		ds.tokIDs = was.ids[off : off+m]
	}
	if len(was.marks) > 0 {
		at := 0
		for r := 0; r < off && at < len(was.marks); r++ {
			at += 1 + 2*int(was.marks[at])
		}
		from := min(at, len(was.marks))
		for r := 0; r < m && at < len(was.marks); r++ {
			at += 1 + 2*int(was.marks[at])
		}
		ds.qsaMarks = was.marks[from:min(at, len(was.marks))]
	}
	return was
}

// restoreInputs puts back what cutInputs replaced.
func (g *devTier) restoreInputs(sid uint64, was rowInputs) {
	g.mu.Lock()
	ds := g.sessOf(sid)
	ds.pleHost, ds.ngHost, ds.tokIDs, ds.qsaMarks = was.ple, was.ng, was.ids, was.marks
	g.mu.Unlock()
}

// cutRowsOf is rows [off, off+m) of v, n rows of equal width; v when it is not
// n whole rows (the input was not given).
func cutRowsOf(v []float32, n, off, m int) []float32 {
	if n == 0 || len(v) < n || len(v)%n != 0 {
		return v
	}
	w := len(v) / n
	return v[off*w : (off+m)*w]
}
