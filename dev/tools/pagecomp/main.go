// Command pagecomp is the first filter for lossless page compression: it
// samples real .jlm pages (dense block, expert, head) and measures, per stored
// format and plane, what general codecs and a byte-plane split recover, what
// decoding a tile costs, and what the per-tile metadata adds.
//
// It measures the current stored representation only and changes no format.
// Tiles are compressed independently (random access), and a tile that does not
// shrink stays plain, so the ratio r includes the offset table and the plain
// fallback. The decoders are Go's, on one host thread: they price the codecs,
// not a generated decoder, and the break-even D > B/(1-r) is printed against
// the link bandwidths given on the command line so the reader can compare.
package main

import (
	"bytes"
	"compress/flate"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"time"

	"github.com/jitllm/jitllm/format/jlm"
)

// stat accumulates one (class, type, plane, variant) cell.
type stat struct {
	raw, comp, meta int64
	decNs           int64
	decBytes        int64
	entropy         float64 // order-0 entropy bound, in bytes
}

type key struct {
	class, typ, plane, variant string
}

// options are the command's flags.
type options struct {
	tileSize, maxTiles, nPages, minSpan int
	disk, pcie, hbm                     float64
}

func parseOptions() *options {
	o := &options{}
	flag.IntVar(&o.tileSize, "tile", 16<<10, "independently decodable tile size in bytes")
	flag.IntVar(&o.maxTiles, "tiles", 64, "tiles sampled per tensor plane")
	flag.IntVar(&o.nPages, "pages", 6, "pages sampled per class")
	flag.IntVar(&o.minSpan, "minspan", 64<<10, "skip planes smaller than this")
	flag.Float64Var(&o.disk, "disk", 3.0, "disk bandwidth, GB/s")
	flag.Float64Var(&o.pcie, "pcie", 12.0, "PCIe bandwidth, GB/s")
	flag.Float64Var(&o.hbm, "hbm", 800.0, "device memory bandwidth, GB/s")
	flag.Parse()
	return o
}

const metaPerTile = 5 // 4-byte offset + 1-byte codec flag

func main() {
	o := parseOptions()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: pagecomp [flags] <model.jlm>")
		os.Exit(2)
	}
	f, err := jlm.Open(flag.Arg(0))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()
	fr, err := jlm.NewFrames(f)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer fr.Close()
	cells := map[key]*stat{}
	nb, nv := int(f.H.NBlocks), int(f.H.NVisBlocks)
	np := f.NPages()
	var blockPages, expertPages []int
	for li := 0; li < np; li++ {
		switch {
		case li < nb:
			blockPages = append(blockPages, li)
		case li >= nb+nv:
			expertPages = append(expertPages, li)
		}
	}
	fmt.Printf("%s: %d block pages, %d expert pages, page %d KiB, tile %d KiB\n",
		flag.Arg(0), len(blockPages), len(expertPages), f.H.PageSize>>10, o.tileSize>>10)

	// head: the dense region.
	for i := range f.Entries() {
		e := &f.Entries()[i]
		if f.InDense(e) {
			qs, d, sc := fr.Span(e)
			measure(o, cells, "head", e.Type, qs, d, sc)
		}
	}
	for _, li := range pick(blockPages, o.nPages) {
		if err := fr.EnsurePage(li); err != nil {
			fmt.Fprintln(os.Stderr, err)
			continue
		}
		for i := range f.Entries() {
			e := &f.Entries()[i]
			if f.InDense(e) || f.ExpertPaged(e) || f.BlockOf(e) != li {
				continue
			}
			qs, d, sc := fr.Span(e)
			measure(o, cells, "dense", e.Type, qs, d, sc)
		}
		fr.DropPage(li)
	}
	for _, li := range pick(expertPages, o.nPages) {
		if err := fr.EnsurePage(li); err != nil {
			fmt.Fprintln(os.Stderr, err)
			continue
		}
		for i := range f.Entries() {
			e := &f.Entries()[i]
			if !f.ExpertPaged(e) {
				continue
			}
			x := li - nb - nv - int((e.QSOff-f.H.ExpDataOff)/f.H.ExpPageSize)
			if x < 0 {
				continue
			}
			qs, d, sc := fr.Sheet(e, x)
			measure(o, cells, "expert", e.Type, qs, d, sc)
		}
		fr.DropPage(li)
	}
	report(o, cells)
}

// pick returns n indices spread evenly over xs.
func pick(xs []int, n int) []int {
	if len(xs) <= n {
		return xs
	}
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, xs[i*len(xs)/n])
	}
	return out
}

func measure(o *options, cells map[key]*stat, class string, t jlm.Type, qs, d, sc []byte) {
	for _, p := range []struct {
		name string
		b    []byte
	}{{"qs", qs}, {"d", d}, {"sc", sc}} {
		if len(p.b) < o.minSpan {
			continue
		}
		for _, v := range variants(t, p.name) {
			k := key{class, t.String(), p.name, v.name}
			s := cells[k]
			if s == nil {
				s = &stat{}
				cells[k] = s
			}
			tiles(o, p.b, func(tile []byte) { v.run(s, tile) })
		}
	}
}

// tiles calls fn on up to maxTiles evenly spaced tiles of b.
func tiles(o *options, b []byte, fn func([]byte)) {
	n := (len(b) + o.tileSize - 1) / o.tileSize
	take := n
	if take > o.maxTiles {
		take = o.maxTiles
	}
	for i := 0; i < take; i++ {
		t := i * n / take
		lo := t * o.tileSize
		hi := lo + o.tileSize
		if hi > len(b) {
			hi = len(b)
		}
		fn(b[lo:hi])
	}
}

type variant struct {
	name string
	run  func(*stat, []byte)
}

// variants are the codecs tried on a plane. "split" moves the plane's
// byte lanes apart first: 2 lanes for 16-bit elements, 4 for 32-bit words, and
// nibble lanes for a 4-bit payload plane.
func variants(t jlm.Type, plane string) []variant {
	vs := []variant{
		{"flate1", func(s *stat, b []byte) { codec(s, b, 1, nil) }},
		{"flate9", func(s *stat, b []byte) { codec(s, b, 9, nil) }},
		{"order0", func(s *stat, b []byte) { entropy(s, b, 1) }},
	}
	if sp := splitter(t, plane); sp != nil {
		vs = append(vs,
			variant{"split+flate1", func(s *stat, b []byte) { codec(s, b, 1, sp) }},
			variant{"split+flate9", func(s *stat, b []byte) { codec(s, b, 9, sp) }},
			variant{"split+order0", func(s *stat, b []byte) { entropy(s, sp.fwd(b), sp.n) }})
	}
	return vs
}

type lanes struct {
	n int // lanes the split produces

	fwd func([]byte) []byte
	inv func(dst, src []byte)
}

func splitter(t jlm.Type, plane string) *lanes {
	switch {
	case plane == "qs" && (t == jlm.TypeF16 || t == jlm.TypeBF16):
		return strideLanes(2)
	case plane == "qs" && t == jlm.TypeF32:
		return strideLanes(4)
	case plane == "d" || plane == "sc":
		return strideLanes(4)
	case plane == "qs" && (t == jlm.TypeQ4 || t == jlm.TypeQ4S || t == jlm.TypeQ5S || t == jlm.TypeQ5):
		return nibbleLanes()
	}
	return nil
}

func strideLanes(w int) *lanes {
	return &lanes{
		n: w,
		fwd: func(b []byte) []byte {
			n := len(b) / w
			out := make([]byte, len(b))
			for i := 0; i < n; i++ {
				for l := 0; l < w; l++ {
					out[l*n+i] = b[i*w+l]
				}
			}
			copy(out[n*w:], b[n*w:])
			return out
		},
		inv: func(dst, src []byte) {
			n := len(src) / w
			for i := 0; i < n; i++ {
				for l := 0; l < w; l++ {
					dst[i*w+l] = src[l*n+i]
				}
			}
			copy(dst[n*w:], src[n*w:])
		},
	}
}

func nibbleLanes() *lanes {
	return &lanes{
		n: 2,
		fwd: func(b []byte) []byte {
			h := len(b) / 2
			out := make([]byte, len(b))
			for i := 0; i < h; i++ {
				out[i] = b[2*i]&15 | b[2*i+1]<<4
				out[h+i] = b[2*i]>>4 | b[2*i+1]&0xf0
			}
			copy(out[2*h:], b[2*h:])
			return out
		},
		inv: func(dst, src []byte) {
			h := len(src) / 2
			for i := 0; i < h; i++ {
				dst[2*i] = src[i]&15 | src[h+i]<<4
				dst[2*i+1] = src[i]>>4 | src[h+i]&0xf0
			}
			copy(dst[2*h:], src[2*h:])
		},
	}
}

// codec compresses one tile with flate, keeps it plain when that is not
// smaller, charges the tile's metadata, and times decoding it.
func codec(s *stat, b []byte, level int, sp *lanes) {
	in := b
	if sp != nil {
		in = sp.fwd(b)
	}
	var buf bytes.Buffer
	w, _ := flate.NewWriter(&buf, level)
	w.Write(in)
	w.Close()
	s.raw += int64(len(b))
	s.meta += metaPerTile
	if buf.Len() >= len(b) {
		s.comp += int64(len(b)) // stays plain: no decode
		return
	}
	s.comp += int64(buf.Len())
	dst := make([]byte, len(in))
	out := make([]byte, len(b))
	reps := 0
	rd := flate.NewReader(bytes.NewReader(buf.Bytes()))
	t0 := time.Now()
	for time.Since(t0) < 2*time.Millisecond || reps < 3 {
		rd.(flate.Resetter).Reset(bytes.NewReader(buf.Bytes()), nil)
		io.ReadFull(rd, dst)
		if sp != nil {
			sp.inv(out, dst)
		}
		reps++
	}
	s.decNs += time.Since(t0).Nanoseconds()
	s.decBytes += int64(reps) * int64(len(b))
}

// entropy adds the order-0 bound for a tile cut into n equal lanes, each with
// its own byte histogram: what a per-lane Huffman or ANS coder could reach
// before its table. n is 1 for an unsplit tile.
func entropy(s *stat, b []byte, n int) {
	seg := len(b) / n
	var total float64
	for l := 0; l < n; l++ {
		part := b[l*seg : (l+1)*seg]
		if l == n-1 {
			part = b[l*seg:]
		}
		var h [256]int
		for _, c := range part {
			h[c]++
		}
		var bits float64
		for _, c := range h {
			if c > 0 {
				bits -= float64(c) * math.Log2(float64(c)/float64(len(part)))
			}
		}
		e := bits / 8
		if e > float64(len(part)) {
			e = float64(len(part))
		}
		total += e
	}
	s.raw += int64(len(b))
	s.meta += int64(metaPerTile + 128*n) // offsets plus a coarse 128-byte code-length table per lane
	s.entropy += total
}

// summary folds the planes of each (class, type) into one row per codec: a
// plane with no split variant contributes its unsplit result to the split rows,
// since the converter would store it unsplit.
func summary(o *options, cells map[key]*stat) {
	type ck struct{ class, typ string }
	seen := map[ck]bool{}
	for k := range cells {
		seen[ck{k.class, k.typ}] = true
	}
	var cks []ck
	for c := range seen {
		cks = append(cks, c)
	}
	sort.Slice(cks, func(i, j int) bool {
		if cks[i].class != cks[j].class {
			return cks[i].class < cks[j].class
		}
		return cks[i].typ < cks[j].typ
	})
	fmt.Printf("\nwhole tensor (all planes, sampled bytes): r and decode rate\n")
	fmt.Printf("%-7s %-5s %-13s %9s %7s %9s %8s %8s %9s\n", "class", "type", "codec", "MiB", "r", "dec GB/s", "disk>", "pcie>", "hbm>")
	for _, c := range cks {
		for _, v := range []string{"flate1", "flate9", "order0", "split+flate1", "split+flate9", "split+order0"} {
			var raw, out, dn, db float64
			for _, pl := range []string{"qs", "d", "sc"} {
				st := cells[key{c.class, c.typ, pl, v}]
				base := v
				if st == nil && len(v) > 6 && v[:6] == "split+" {
					base = v[6:]
					st = cells[key{c.class, c.typ, pl, base}]
				}
				if st == nil {
					continue
				}
				raw += float64(st.raw)
				if v == "order0" || v == "split+order0" {
					out += st.entropy + float64(st.meta)
				} else {
					out += float64(st.comp + st.meta)
					dn += float64(st.decNs)
					db += float64(st.decBytes)
				}
			}
			if raw == 0 {
				continue
			}
			r := out / raw
			dec := "-"
			if dn > 0 {
				dec = fmt.Sprintf("%.2f", db/dn)
			}
			be := func(b float64) string {
				if r >= 1 {
					return "never"
				}
				return fmt.Sprintf("%.1f", b/(1-r))
			}
			fmt.Printf("%-7s %-5s %-13s %9.2f %7.4f %9s %8s %8s %9s\n", c.class, c.typ, v, raw/(1<<20), r, dec, be(o.disk), be(o.pcie), be(o.hbm))
		}
	}
}

func report(o *options, cells map[key]*stat) {
	summary(o, cells)
	keys := make([]key, 0, len(cells))
	for k := range cells {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := keys[i], keys[j]
		if a.class != b.class {
			return a.class < b.class
		}
		if a.typ != b.typ {
			return a.typ < b.typ
		}
		if a.plane != b.plane {
			return a.plane < b.plane
		}
		return a.variant < b.variant
	})
	fmt.Printf("\nbreak-even D > B/(1-r), B = disk %.1f, pcie %.1f, hbm %.0f GB/s\n", o.disk, o.pcie, o.hbm)
	fmt.Printf("%-7s %-5s %-3s %-13s %9s %7s %9s %8s %8s %9s\n",
		"class", "type", "pl", "codec", "MiB", "r", "dec GB/s", "disk>", "pcie>", "hbm>")
	for _, k := range keys {
		s := cells[k]
		var r float64
		dec := "-"
		if k.variant == "order0" || k.variant == "split+order0" {
			r = (s.entropy + float64(s.meta)) / float64(s.raw)
		} else {
			r = float64(s.comp+s.meta) / float64(s.raw)
			if s.decNs > 0 {
				dec = fmt.Sprintf("%.2f", float64(s.decBytes)/float64(s.decNs))
			}
		}
		be := func(b float64) string {
			if r >= 1 {
				return "never"
			}
			return fmt.Sprintf("%.1f", b/(1-r))
		}
		fmt.Printf("%-7s %-5s %-3s %-13s %9.2f %7.4f %9s %8s %8s %9s\n",
			k.class, k.typ, k.plane, k.variant, float64(s.raw)/(1<<20), r, dec,
			be(o.disk), be(o.pcie), be(o.hbm))
	}
}
