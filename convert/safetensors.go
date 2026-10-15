package convert

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/jitllm/jitllm/convert/safetensors"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
)

// The second input format, producing the same container gguf.go does. A
// safetensors file holds weights only; config.json, tokenizer.json and
// tokenizer_config.json live beside it, so the unit this converter takes is a
// directory, and naming one .safetensors file means "this shard, and the JSON
// beside it". The tokenizer comes from the model's own tokenizer.json rather
// than a label (RULE 7m); see hfVocabOf.

// FromSafetensors writes src as dst, a jlm container.
//
// src is either a directory holding config.json and one or more .safetensors
// files, or one .safetensors file whose directory holds the JSON. A sharded
// model is found through model.safetensors.index.json.
func FromSafetensors(src, dst string, fp jlm.Fingerprint, opts ...Option) (*jlm.Header, error) {
	dir, shards, err := hfShards(src)
	if err != nil {
		return nil, err
	}
	var open []*safetensors.File
	defer func() {
		for _, f := range open {
			f.Close()
		}
	}()
	for _, p := range shards {
		f, err := safetensors.Open(p)
		if err != nil {
			return nil, err
		}
		open = append(open, f)
	}
	return FromShards(os.DirFS(dir), dir, open, dst, fp, opts...)
}

// IndexShards is the shard file names a model.safetensors.index.json names,
// sorted and each once. name is for messages.
func IndexShards(b []byte, name string) ([]string, error) {
	var m struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", name, err)
	}
	seen := map[string]bool{}
	var shards []string
	for _, f := range m.WeightMap {
		// A shard name is untrusted text and must be a plain file name in
		// this directory.
		if f != filepath.Base(f) || f == "" {
			return nil, fmt.Errorf("convert: %s names shard %q, which is not a plain file name", name, f)
		}
		if !seen[f] {
			seen[f] = true
			shards = append(shards, f)
		}
	}
	if len(shards) == 0 {
		return nil, fmt.Errorf("convert: %s has an empty weight_map", name)
	}
	sort.Strings(shards)
	return shards, nil
}

// FromShards writes a container from shards already open, with meta holding
// config.json and the tokenizer -- read once, first, into the container's own
// config and vocabulary. The shards may be local files or, through
// safetensors.OpenAt, anything else that can read a byte range; meta may be
// a directory (os.DirFS) or the files in memory (fstest.MapFS).
func FromShards(meta fs.FS, name string, open []*safetensors.File, dst string, fp jlm.Fingerprint, opts ...Option) (*jlm.Header, error) {
	s, err := hfSourceOf(hfDir{meta, name}, open)
	if err != nil {
		return nil, err
	}
	var o options
	for _, fn := range opts {
		fn(&o)
	}
	if o.head != "" {
		return nil, fmt.Errorf("convert: WithHead joins a GGUF head to a GGUF; %s is safetensors", name)
	}
	if o.q8 {
		quantizeMatrices(s.Tensors)
	}
	if err := o.applyChatTemplate(s); err != nil {
		return nil, err
	}
	// Same stamp the GGUF path applies; see jlm.Fingerprint.Writer.
	if fp.Writer == "" {
		fp.Writer = jlm.WriterID()
	}
	s.Origin = originOf(name, open)
	return jlm.Write(dst, s, fp)
}

// originOf names a model's shards for a resumed write (jlm.Source.Origin). A
// streamed shard's name carries the commit it was pinned at, which is
// immutable; a local one is named with its size and modification time.
func originOf(name string, open []*safetensors.File) string {
	var b strings.Builder
	fmt.Fprintf(&b, "safetensors %s", name)
	for _, f := range open {
		fmt.Fprintf(&b, "\n%s %d", f.Path, len(f.Tensors))
		if fi, err := os.Stat(f.Path); err == nil {
			fmt.Fprintf(&b, " %d %d", fi.Size(), fi.ModTime().UnixNano())
		}
	}
	return b.String()
}

// Option changes how a model is converted.
type Option func(*options)

type options struct {
	q8           bool   // WithQ8, safetensors only
	chatTemplate string // WithChatTemplate
	head         string // WithHead, GGUF only
}

// WithHead converts a GGUF together with its multi-token-prediction head,
// published as a GGUF of its own: the head's blocks become the model's
// prediction blocks after the trunk (withHead), exactly as if the trunk's
// file had carried them.
func WithHead(path string) Option { return func(o *options) { o.head = path } }

// WithQ8 stores every weight matrix as Q8_0 instead of carrying its float
// bytes. It is the only lossy step this path takes and it is opt-in: the
// default stays exact, which the transformers goldens are held to. A bf16
// checkpoint otherwise widens to f32 (about 4x the Q8_0 size).
func WithQ8() Option { return func(o *options) { o.q8 = true } }

// quantizeMatrices rewrites every float matrix whose rows are whole Q8_0
// blocks, at the point its bytes are read. Routers stay float, as llama.cpp
// keeps them: a flipped top-k selection is a different model, not a noisier
// one. Vectors, convolution taps and ragged k are carried as they are.
func quantizeMatrices(ts []jlm.Tensor) {
	for i := range ts {
		t := &ts[i]
		switch {
		case t.NDim < 2, t.Dims[0]%q8Elems != 0,
			t.Role == jlm.RoleRouter, t.Role == jlm.RoleShRouter:
			continue
		}
		src, ok := sourceType(t.Type)
		if !ok || jlm.Packed(t.Type) {
			continue
		}
		k, name := int(t.Dims[0]), t.Name
		load, data := t.Load, t.Data
		t.Type, t.Data = jlm.TypeQ8, nil
		t.Load = func() ([]byte, error) {
			b := data
			if load != nil {
				var err error
				if b, err = load(); err != nil {
					return nil, err
				}
			}
			vals := make([]float32, uint64(len(b))/src.BlockBytes()*src.BlockElems())
			if err := quant.Dequant32(src, b, vals); err != nil {
				return nil, fmt.Errorf("convert: %s: %w", name, err)
			}
			return quantizeQ8Rows(vals, k, len(vals)/k), nil
		}
	}
}

// IsSafetensors reports whether src names a safetensors model -- a directory
// holding one, or a .safetensors file. It is what cmd/jitllm routes on.
func IsSafetensors(src string) bool {
	if strings.HasSuffix(src, ".safetensors") {
		return true
	}
	fi, err := os.Stat(src)
	if err != nil || !fi.IsDir() {
		return false
	}
	_, shards, err := hfShards(src)
	return err == nil && len(shards) > 0
}

// hfShards resolves src to a directory and the .safetensors files that make up
// one model.
//
// An index is the authority when there is one, and a glob is not a fallback
// for it: a directory can also hold another model's shards or an adapter. So:
// index if present, else model.safetensors, else a single .safetensors; several
// files with no index is a refusal.
func hfShards(src string) (dir string, shards []string, err error) {
	fi, err := os.Stat(src)
	if err != nil {
		return "", nil, err
	}
	if !fi.IsDir() {
		if !strings.HasSuffix(src, ".safetensors") {
			return "", nil, fmt.Errorf("convert: %s is not a .safetensors file", src)
		}
		return filepath.Dir(src), []string{src}, nil
	}
	dir = src
	idx := filepath.Join(dir, "model.safetensors.index.json")
	if b, err := os.ReadFile(idx); err == nil {
		names, err := IndexShards(b, idx)
		if err != nil {
			return "", nil, err
		}
		for _, n := range names {
			shards = append(shards, filepath.Join(dir, n))
		}
		return dir, shards, nil
	}
	if p := filepath.Join(dir, "model.safetensors"); fileExists(p) {
		return dir, []string{p}, nil
	}
	all, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	switch len(all) {
	case 0:
		return "", nil, fmt.Errorf("convert: %s holds no .safetensors file", dir)
	case 1:
		return dir, all, nil
	}
	return "", nil, fmt.Errorf("convert: %s holds %d .safetensors files and no "+
		"model.safetensors.index.json to say which belong to one model", dir, len(all))
}

func fileExists(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && !fi.IsDir()
}

// hfLoc is where one source tensor's bytes are: a safetensors entry, or for a
// compressed-tensors MXFP4 weight its codes (f, t) and scales (sf, st).
type hfLoc struct {
	f  *safetensors.File
	t  *safetensors.Tensor
	sf *safetensors.File
	st *safetensors.Tensor
}

// mx reports whether this is an MXFP4 weight (convert/hfmxfp4.go).
func (l hfLoc) mx() bool { return l.st != nil }

// hfDir is where a HuggingFace model's JSON and tokenizer are read from: a
// directory on disk, or the few files held in memory for a streamed
// conversion. name is for messages.
type hfDir struct {
	fs   fs.FS
	name string
}

// dirFiles is a model directory on the disk.
func dirFiles(dir string) hfDir { return hfDir{os.DirFS(dir), dir} }

func (d hfDir) read(f string) ([]byte, error) { return fs.ReadFile(d.fs, f) }
func (d hfDir) has(f string) bool             { _, err := fs.Stat(d.fs, f); return err == nil }
func (d hfDir) path(f string) string          { return filepath.Join(d.name, f) }
func (d hfDir) String() string                { return d.name }

// hfSourceOf translates a HuggingFace directory into the container's own terms.
func hfSourceOf(dir hfDir, shards []*safetensors.File) (*jlm.Source, error) {
	hc, err := readHFConfig(dir)
	if err != nil {
		return nil, err
	}
	ha, err := hfArchOf(hc)
	if err != nil {
		return nil, err
	}

	// One directory over every shard, refusing a name carried twice (a broken
	// download; taking either silently converts half a model).
	all := map[string]hfLoc{}
	for _, f := range shards {
		for i := range f.Tensors {
			n := f.Tensors[i].Name
			if prev, dup := all[n]; dup {
				return nil, fmt.Errorf("convert: %q is in both %s and %s",
					n, filepath.Base(prev.f.Path), filepath.Base(f.Path))
			}
			all[n] = hfLoc{f: f, t: &f.Tensors[i]}
		}
	}
	// compressed-tensors MXFP4: each weight's codes and scales become one
	// weight of the container's own MXFP4 (convert/hfmxfp4.go).
	mx, err := compressedTensorsOf(hc)
	if err != nil {
		return nil, err
	}
	if err := mergeMXFP4(all, mx); err != nil {
		return nil, err
	}

	names := all2names(all)
	sort.Strings(names)

	cfg, err := ha.config(hc, names)
	if err != nil {
		return nil, err
	}
	s := &jlm.Source{Config: cfg}

	voc, err := hfVocabOf(dir, cfg.NVocab)
	if err != nil {
		return nil, err
	}
	if voc.Stop, err = hfStops(dir, voc); err != nil {
		return nil, err
	}
	s.Vocab = voc

	s.Tensors = make([]jlm.Tensor, 0, len(names)+len(hc.extra))
	s.Tensors = append(s.Tensors, hc.extra...)
	var consumed map[string]bool
	if ha.gather != nil {
		read := func(name string) ([]float32, error) {
			loc, ok := all[name]
			if !ok {
				return nil, fmt.Errorf("convert: %s: no tensor %q", dir, name)
			}
			if loc.mx() {
				return nil, fmt.Errorf("convert: %s is MXFP4, and its values are folded at conversion: %w",
					name, ErrNotImplemented)
			}
			ty, widen, err := hfTypeOf(loc.t.DType)
			if err != nil {
				return nil, fmt.Errorf("convert: %s: %w", name, err)
			}
			data, err := loc.f.ReadTensor(loc.t)
			if err != nil {
				return nil, err
			}
			if widen {
				data = widenBF16(data)
			}
			var out []float32
			if ty == jlm.TypeF16 {
				for i := 0; i+2 <= len(data); i += 2 {
					out = append(out, float32(quant.DecodeHalf(binary.LittleEndian.Uint16(data[i:]))))
				}
				return out, nil
			}
			for i := 0; i+4 <= len(data); i += 4 {
				out = append(out, math.Float32frombits(binary.LittleEndian.Uint32(data[i:])))
			}
			return out, nil
		}
		ts, used, err := ha.gather(cfg, names, read)
		if err != nil {
			return nil, err
		}
		s.Tensors = append(s.Tensors, ts...)
		consumed = used
	}
	banks := map[bankKey][]*jlm.Tensor{}
	fuses := map[bankKey][]*jlm.Tensor{}
	for _, name := range names {
		if consumed[name] {
			continue
		}
		loc := all[name]
		role, block, expert, ignored, err := ha.identify(name, cfg.LayerKinds)
		if err != nil {
			return nil, err
		}
		if ignored {
			continue
		}
		t := loc.t
		shape := t.Shape
		if len(shape) == 0 || len(shape) > 4 {
			return nil, fmt.Errorf("convert: %s has %d dimensions", name, len(shape))
		}
		var ty jlm.Type
		var read func() ([]byte, error)
		var rt *hfRetyped
		if ha.retype != nil && !loc.mx() {
			readNamed := func(n string) ([]byte, *safetensors.Tensor, error) {
				l, ok := all[n]
				if !ok || l.mx() {
					return nil, nil, fmt.Errorf("convert: %s: no tensor %q", dir, n)
				}
				b, err := l.f.ReadTensor(l.t)
				return b, l.t, err
			}
			if rt, err = ha.retype(hc, role, name, t, readNamed); err != nil {
				return nil, err
			}
		}
		if rt != nil {
			ty, shape, read = rt.ty, rt.shape, rt.read
		} else if loc.mx() {
			// [rows, cols] of MXFP4 values, read as GGUF's blocks.
			ty, shape, read = jlm.TypeMX4, []uint64{t.Shape[0], 2 * t.Shape[1]}, loc.readMXFP4
		} else {
			var widen bool
			if ty, widen, err = hfTypeOf(t.DType); err != nil {
				return nil, fmt.Errorf("convert: %s: %w", name, err)
			}
			read = func() ([]byte, error) {
				data, err := loc.f.ReadTensor(t)
				if err == nil && widen {
					data = widenBF16(data)
				}
				return data, err
			}
		}
		e := jlm.Tensor{Role: role, Block: block, Index: -1, Type: ty,
			NDim: uint8(len(shape)), Name: name}
		// The shape is reversed and no byte moves: safetensors lists axes
		// slowest-first, C-contiguous, so a Linear weight is [out, in]; this
		// container's Dims are ne0-fastest, so the same bytes are [in, out].
		for i, d := range shape {
			e.Dims[len(shape)-1-i] = d
		}
		parts := []jlm.Tensor{e}
		roles, isFused := ha.fused[role]
		sp := ha.split[ha.blockRest(name)]
		// A split or a fused cut reads elements, which a block of MXFP4 is
		// not; only a plain matrix, a bank member or a row concatenation may
		// arrive packed.
		if loc.mx() && (sp != nil || isFused) {
			return nil, fmt.Errorf("convert: %s is MXFP4 and is cut at conversion: %w", name, ErrNotImplemented)
		}
		// A plain matrix is read when the writer reaches it; everything else is
		// read immediately, because a split or row cut sizes its parts from the data and a
		// fixup may change a vector's type. Those are small; the matrices are
		// what make a model tens of GB.
		lazy := sp == nil && !isFused && e.NDim >= 2
		var data []byte
		if !lazy {
			if data, err = read(); err != nil {
				return nil, err
			}
		}
		switch {
		case sp != nil:
			if parts, err = sp(e, data, cfg); err != nil {
				return nil, err
			}
		case isFused:
			if parts, err = splitRows(e, data, roles, cfg); err != nil {
				return nil, err
			}
		default:
			parts[0].Data = data
		}
		for i := range parts {
			if lazy {
				parts[i].Load = lazyFixup(ha, parts[i], read, cfg)
			} else if err := ha.setData(&parts[i], parts[i].Data, cfg); err != nil {
				return nil, err
			}
			if expert >= 0 {
				// The index comes from a tensor name, so it is bounded by the
				// config before it sizes anything (RULE 9).
				if uint32(expert) >= cfg.NExpert {
					return nil, fmt.Errorf("convert: %s: expert %d of %d", name, expert, cfg.NExpert)
				}
				k := bankKey{parts[i].Role, block}
				for int(expert) >= len(banks[k]) {
					banks[k] = append(banks[k], nil)
				}
				if banks[k][expert] != nil {
					return nil, fmt.Errorf("convert: %s: expert %d twice", name, expert)
				}
				banks[k][expert] = &parts[i]
				continue
			}
			if order, ok := ha.fuseOrder[parts[i].Role]; ok {
				pos := indexOf(order, ha.blockRest(name))
				if pos < 0 {
					return nil, fmt.Errorf("convert: %s resolves to %v, which is built by "+
						"concatenation, and it is not one of its %d sources",
						name, parts[i].Role, len(order))
				}
				k := bankKey{parts[i].Role, block}
				if fuses[k] == nil {
					fuses[k] = make([]*jlm.Tensor, len(order))
				}
				if fuses[k][pos] != nil {
					return nil, fmt.Errorf("convert: %s: %v source %d twice", name, parts[i].Role, pos)
				}
				fuses[k][pos] = &parts[i]
				continue
			}
			if err := hfCheckShape(&parts[i], cfg); err != nil {
				return nil, err
			}
			s.Tensors = append(s.Tensors, parts[i])
		}
	}
	bs, err := stackBanks(banks, cfg)
	if err != nil {
		return nil, err
	}
	s.Tensors = append(s.Tensors, bs...)
	fs, err := concatRows(fuses, ha, cfg)
	if err != nil {
		return nil, err
	}
	s.Tensors = append(s.Tensors, fs...)
	if s.Tensors, err = nextnTensors(s.Tensors, cfg); err != nil {
		return nil, err
	}
	if err := foldPostRopeQKNorm(s); err != nil {
		return nil, err
	}
	if len(s.Tensors) == 0 {
		return nil, fmt.Errorf("convert: %s: no tensor in this model has a role", dir)
	}
	return s, nil
}

// lazyFixup is setData deferred to the writer: read the bytes, run the
// architecture's fixup, and hand them over. A fixup that changed a matrix's
// type or shape at that point would have been laid out wrong already, so it
// is refused rather than written.
func lazyFixup(ha *hfArch, e jlm.Tensor, read func() ([]byte, error), c *jlm.Config) func() ([]byte, error) {
	return func() ([]byte, error) {
		data, err := read()
		if err != nil {
			return nil, err
		}
		t := e
		if err := ha.setData(&t, data, c); err != nil {
			return nil, err
		}
		if t.Type != e.Type || t.Dims != e.Dims || t.NDim != e.NDim {
			return nil, fmt.Errorf("convert: %s: the fixup changed a matrix's type or shape "+
				"after it was laid out", e.Name)
		}
		return t.Data, nil
	}
}

func indexOf(xs []string, x string) int {
	for i, v := range xs {
		if v == x {
			return i
		}
	}
	return -1
}

// concatRows builds the roles an architecture declares in fuseOrder by joining
// their sources' rows, in the declared order. A source is [rows, k] with k
// contiguous, so appending bytes gives [sum(rows), k]. Every source must agree
// on k and type, and a missing source is a refusal: Kimi-Linear's mixed
// projection is sliced by arithmetic, so a missing v would be read past the
// end and a missing q would silently promote k into q's rows.
func concatRows(fuses map[bankKey][]*jlm.Tensor, ha *hfArch, c *jlm.Config) ([]jlm.Tensor, error) {
	keys := make([]bankKey, 0, len(fuses))
	for k := range fuses {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].block != keys[j].block {
			return keys[i].block < keys[j].block
		}
		return keys[i].role < keys[j].role
	})
	out := make([]jlm.Tensor, 0, len(keys))
	for _, k := range keys {
		parts, order := fuses[k], ha.fuseOrder[k.role]
		first := parts[0]
		if first == nil {
			return nil, fmt.Errorf("convert: block %d %v: source %q is missing",
				k.block, k.role, order[0])
		}
		t := jlm.Tensor{Role: k.role, Block: k.block, Index: -1, Type: first.Type,
			NDim: first.NDim, Dims: first.Dims, Name: first.Name}
		var rows uint64
		for i, p := range parts {
			switch {
			case p == nil:
				return nil, fmt.Errorf("convert: block %d %v: source %q is missing",
					k.block, k.role, order[i])
			case p.Type != first.Type:
				return nil, fmt.Errorf("convert: block %d %v: source %q is %v and %q is %v",
					k.block, k.role, order[i], p.Type, order[0], first.Type)
			case p.NDim != first.NDim || p.Dims[0] != first.Dims[0]:
				return nil, fmt.Errorf("convert: block %d %v: source %q has ne0 %d and %q has %d",
					k.block, k.role, order[i], p.Dims[0], order[0], first.Dims[0])
			}
			rows += p.Dims[1]
		}
		t.Dims[1] = rows
		t.Load = joined(parts)
		if err := hfCheckShape(&t, c); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, nil
}

// joined is the bytes of ts back to back, each taken from its Data or its
// Load. The loads run a few at a time: a streamed bank is hundreds of ranged
// reads, and one at a time pays each one's round trip in series.
func joined(ts []*jlm.Tensor) func() ([]byte, error) {
	return func() ([]byte, error) {
		bufs := make([][]byte, len(ts))
		errs := make([]error, len(ts))
		sem := make(chan struct{}, joinFanout)
		var wg sync.WaitGroup
		for i, t := range ts {
			if t.Load == nil {
				bufs[i] = t.Data
				continue
			}
			wg.Add(1)
			sem <- struct{}{}
			go func() {
				defer wg.Done()
				bufs[i], errs[i] = t.Load()
				<-sem
			}()
		}
		wg.Wait()
		n := 0
		for i, b := range bufs {
			if errs[i] != nil {
				return nil, errs[i]
			}
			n += len(b)
		}
		out := make([]byte, 0, n)
		for _, b := range bufs {
			out = append(out, b...)
		}
		return out, nil
	}
}

// joinFanout is hf.DefaultConns: a streamed expert is a few MB, one range, so a
// bank keeps as many connections busy as loads run at once, and one carried
// about 60 MB/s from the Hub's CDN. The hf client bounds the
// connections; a local file is unaffected.
const joinFanout = 16

// bankKey names one expert bank: a role in a block.
type bankKey struct {
	role  jlm.Role
	block int32
}

// stackBanks turns HuggingFace's per-expert matrices into the container's
// banks. Expert e's [rows, k] matrix is C-contiguous, so E of them back to
// back are the [E, rows, k] GGUF bank shape the engine already reads
// (model.bankExpert slices it). A missing expert is a refusal: the router has
// NExpert rows.
func stackBanks(banks map[bankKey][]*jlm.Tensor, c *jlm.Config) ([]jlm.Tensor, error) {
	keys := make([]bankKey, 0, len(banks))
	for k := range banks {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].block != keys[j].block {
			return keys[i].block < keys[j].block
		}
		return keys[i].role < keys[j].role
	})
	out := make([]jlm.Tensor, 0, len(keys))
	for _, k := range keys {
		sheets := banks[k]
		if uint32(len(sheets)) != c.NExpert {
			return nil, fmt.Errorf("convert: block %d %v: %d expert(s), want %d",
				k.block, k.role, len(sheets), c.NExpert)
		}
		first := sheets[0]
		if first == nil || first.NDim != 2 {
			return nil, fmt.Errorf("convert: block %d %v: expert 0 missing or not a matrix", k.block, k.role)
		}
		b := jlm.Tensor{Role: k.role, Block: k.block, Index: -1, Type: first.Type, NDim: 3,
			Dims: [4]uint64{first.Dims[0], first.Dims[1], uint64(len(sheets))},
			Name: strings.Replace(first.Name, "experts.0.", "experts.*.", 1)}
		for e, x := range sheets {
			if x == nil || x.Type != first.Type || x.Dims != first.Dims || len(x.Data) != len(first.Data) {
				return nil, fmt.Errorf("convert: block %d %v: expert %d is missing or unlike expert 0",
					k.block, k.role, e)
			}
		}
		b.Load = joined(sheets)
		if err := hfCheckShape(&b, c); err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return out, nil
}

// splitRows cuts a fused projection into the tensors the container stores.
// The source is row-major [out, in], so a row range is a byte range. The row
// counts come from the config and must sum to the tensor's: parts sized by a
// wrong geometry are plausible wrong tensors.
func splitRows(e jlm.Tensor, data []byte, roles []jlm.Role, c *jlm.Config) ([]jlm.Tensor, error) {
	if e.NDim != 2 {
		return nil, fmt.Errorf("convert: %s is fused and %d-dimensional", e.Name, e.NDim)
	}
	var want uint64
	rows := make([]uint64, len(roles))
	for i, r := range roles {
		rows[i] = hfRows(r, c)
		want += rows[i]
	}
	if e.Dims[1] != want || want == 0 || uint64(len(data))%want != 0 {
		return nil, fmt.Errorf("convert: %s has %d rows, and its parts %v want %d",
			e.Name, e.Dims[1], roles, want)
	}
	rowBytes := uint64(len(data)) / want
	out := make([]jlm.Tensor, len(roles))
	var at uint64
	for i, r := range roles {
		p := e
		p.Role, p.Dims[1] = r, rows[i]
		p.Name = fmt.Sprintf("%s[%v]", e.Name, r)
		p.Data = data[at*rowBytes : (at+rows[i])*rowBytes]
		at += rows[i]
		out[i] = p
	}
	return out, nil
}

// hfHeadDimV is the value head's width. jlm.Config stores zero where it equals
// HeadDim, so every reader has to ask rather than read the field.
func hfHeadDimV(c *jlm.Config) uint32 {
	if c.HeadDimV != 0 {
		return c.HeadDimV
	}
	return c.HeadDim
}

// hfRows is a projection's output width, from the config alone.
func hfRows(r jlm.Role, c *jlm.Config) uint64 {
	switch r {
	case jlm.RoleAttnQ:
		return uint64(c.NHead) * uint64(c.HeadDim)
	case jlm.RoleAttnK, jlm.RoleAttnV:
		return uint64(c.NKVHead) * uint64(c.HeadDim)
	case jlm.RoleFFNGate, jlm.RoleFFNUp:
		return uint64(c.NFFN)
	}
	return 0
}

func all2names[T any](m map[string]T) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// hfTypeOf maps a safetensors dtype onto a container type.
//
// A dtype with no path is refused by name: there is no quantiser here, so an
// unquantised checkpoint stays unquantised.
//
// BF16 is widened to F32 on purpose: jit/cpu emits native matvecs for F32 and
// F16 only (cpu.SupportedNative), and widening is exact (bf16 is f32's top 16
// bits) at 2x the bytes. A native BF16 kernel would remove both.
func hfTypeOf(d safetensors.DType) (t jlm.Type, widen bool, err error) {
	switch d {
	case safetensors.F32:
		return jlm.TypeF32, false, nil
	case safetensors.F16:
		return jlm.TypeF16, false, nil
	case safetensors.BF16:
		return jlm.TypeF32, true, nil
	}
	return 0, false, fmt.Errorf("dtype %s has no path into this container "+
		"(F32, F16 and BF16 are carried; there is no quantiser here)", d)
}

// widenBF16 turns little-endian bf16 into little-endian f32 by shifting each
// value into the high half of a word. Exact: bf16 is the top 16 bits of an
// IEEE binary32.
func widenBF16(src []byte) []byte {
	n := len(src) / 2
	dst := make([]byte, n*4)
	for i := 0; i < n; i++ {
		dst[i*4+2] = src[i*2]
		dst[i*4+3] = src[i*2+1]
	}
	return dst
}

// ---------------------------------------------------------------------------
// config.json

// hfConfig is the subset of config.json this converter reads, plus every field
// it must refuse on. The refusals are declared rather than implied, since an
// absent field here is a key nobody read.
type hfConfig struct {
	Architectures         []string        `json:"architectures"`
	ModelType             string          `json:"model_type"`
	HiddenSize            uint32          `json:"hidden_size"`
	IntermediateSize      uint32          `json:"intermediate_size"`
	NumHiddenLayers       uint32          `json:"num_hidden_layers"`
	NumAttentionHeads     uint32          `json:"num_attention_heads"`
	NumKeyValueHeads      *uint32         `json:"num_key_value_heads"`
	HeadDim               *uint32         `json:"head_dim"`
	MaxPositionEmbeddings uint32          `json:"max_position_embeddings"`
	RMSNormEps            float64         `json:"rms_norm_eps"`
	RopeTheta             float64         `json:"rope_theta"`
	RopeScaling           json.RawMessage `json:"rope_scaling"`
	TieWordEmbeddings     *bool           `json:"tie_word_embeddings"`
	VocabSize             uint32          `json:"vocab_size"`
	HiddenAct             string          `json:"hidden_act"`
	AttentionBias         bool            `json:"attention_bias"`
	MLPBias               bool            `json:"mlp_bias"`
	PartialRotaryFactor   *float64        `json:"partial_rotary_factor"`
	SlidingWindow         *uint32         `json:"sliding_window"`
	UseSlidingWindow      *bool           `json:"use_sliding_window"`
	TorchDtype            string          `json:"torch_dtype"`
	HiddenActivation      string          `json:"hidden_activation"`
	// transformers 5 carries rope_theta (and any scaling) here instead.
	RopeParameters json.RawMessage `json:"rope_parameters"`
	// extra holds tensors the config states and the checkpoint does not ship
	// (LongRoPE's factor vectors); the arch's config reader fills it.
	extra []jlm.Tensor

	// A mixture. num_experts is qwen3moe/olmoe's spelling, num_local_experts
	// mixtral's.
	NumExperts                   uint32   `json:"num_experts"`
	NumLocalExperts              uint32   `json:"num_local_experts"`
	NumExpertsPerTok             uint32   `json:"num_experts_per_tok"`
	MoEIntermediateSize          hfUint   `json:"moe_intermediate_size"`
	NormTopKProb                 *bool    `json:"norm_topk_prob"`
	DecoderSparseStep            *uint32  `json:"decoder_sparse_step"`
	MLPOnlyLayers                []uint32 `json:"mlp_only_layers"`
	SharedExpertIntermediateSize uint32   `json:"shared_expert_intermediate_size"`
	ClipQKV                      *float64 `json:"clip_qkv"`

	// Multi-head Latent Attention (DeepSeek-V2/V3). See jlm.Config's MLA block.
	// q_lora_rank is a plain uint32: V2-Lite's "q_lora_rank": null means a
	// one-step query projection, which jlm.Config.QLoraRank spells as zero too.
	QLoraRank     uint32 `json:"q_lora_rank"`
	KVLoraRank    uint32 `json:"kv_lora_rank"`
	QKNopeHeadDim uint32 `json:"qk_nope_head_dim"`
	QKRopeHeadDim uint32 `json:"qk_rope_head_dim"`
	VHeadDim      uint32 `json:"v_head_dim"`

	// DeepSeek's mixture. Its expert count is n_routed_experts (a third
	// spelling), plus a leading dense run, a shared expert sized as a multiple
	// of the routed width, grouped selection, and a scale on the weights.
	FirstKDenseReplace  uint32  `json:"first_k_dense_replace"`
	NRoutedExperts      uint32  `json:"n_routed_experts"`
	NSharedExperts      uint32  `json:"n_shared_experts"`
	NGroup              uint32  `json:"n_group"`
	TopkGroup           uint32  `json:"topk_group"`
	RoutedScalingFactor float64 `json:"routed_scaling_factor"`
	ScoringFunc         string  `json:"scoring_func"`
	// NumNextnPredictLayers is how many multi-token-prediction blocks follow
	// the trunk (DeepSeek-V3, GLM-4.7-Flash): model.layers.{NLayer+i}.
	NumNextnPredictLayers uint32 `json:"num_nextn_predict_layers"`

	// Kimi-Linear: the recurrence's geometry and the per-layer kind maps.
	// It re-spells mixture keys that already have other spellings
	// (num_experts_per_token, num_shared_experts, num_expert_group,
	// moe_renormalize). encoding/json leaves an unread key at zero, so a missed
	// spelling fails silently; each is read into its own field so two spellings
	// in one file can be compared rather than one silently winning.
	LayerTypes          []string `json:"layer_types"`
	MLPLayerTypes       []string `json:"mlp_layer_types"`
	LinearNumHeads      uint32   `json:"linear_num_heads"`
	LinearHeadDim       uint32   `json:"linear_head_dim"`
	LinearConvKernelDim uint32   `json:"linear_conv_kernel_dim"`
	NumExpertsPerToken  uint32   `json:"num_experts_per_token"`
	NumSharedExperts    uint32   `json:"num_shared_experts"`
	NumExpertGroup      uint32   `json:"num_expert_group"`
	MoERenormalize      *bool    `json:"moe_renormalize"`
	// Moonshot's own configuration_kimi.py -- what the published checkpoint
	// ships -- says the same things in another shape; see kimiNormalize.
	LinearAttnConfig *struct {
		KDALayers      []uint32 `json:"kda_layers"`
		FullAttnLayers []uint32 `json:"full_attn_layers"`
		NumHeads       uint32   `json:"num_heads"`
		HeadDim        uint32   `json:"head_dim"`
		ShortConv      uint32   `json:"short_conv_kernel_size"`
	} `json:"linear_attn_config"`
	MoELayerFreq     *uint32 `json:"moe_layer_freq"`
	MLAUseNope       *bool   `json:"mla_use_nope"`
	MoERouterActFunc string  `json:"moe_router_activation_func"`

	// ERNIE 4.5's mixture, under its own names (Ernie4_5_MoeConfig).
	MoENumExperts       uint32 `json:"moe_num_experts"`
	MoEK                uint32 `json:"moe_k"`
	MoENumSharedExperts uint32 `json:"moe_num_shared_experts"`
	MoELayerStartIndex  *int   `json:"moe_layer_start_index"`
	MoELayerEndIndex    *int   `json:"moe_layer_end_index"`
	MoELayerInterval    uint32 `json:"moe_layer_interval"`

	// Hunyuan's, per layer as lists (hfUint), and its two switches.
	MoETopK         hfUint `json:"moe_topk"`
	NumSharedExpert hfUint `json:"num_shared_expert"`
	UseCLA          *bool  `json:"use_cla"`
	// SmolLM3's NoPE layers, and EXAONE 4's window pattern (an int, or a
	// string transformers cannot read without layer_types). See
	// convert/safetensors_dense.go.
	NoRopeLayers         []int           `json:"no_rope_layers"`
	NoRopeLayerInterval  *uint32         `json:"no_rope_layer_interval"`
	SlidingWindowPattern json.RawMessage `json:"sliding_window_pattern"`

	// nextn is NumNextnPredictLayers where the architecture keeps them, set
	// by its config function before the shared one reads it.
	nextn uint32

	// keys is every top-level key config.json carries. An absent key takes
	// its CLASS's default, which differs per class (SeedOssConfig's head_dim
	// is 128, SmolLM3Config's num_key_value_heads 4), and a pointer field
	// cannot tell absent from null.
	keys map[string]json.RawMessage

	path string // for error messages
}

// absent reports whether config.json does not carry key at all.
func (c *hfConfig) absent(key string) bool {
	_, ok := c.keys[key]
	return !ok
}

func readHFConfig(dir hfDir) (*hfConfig, error) {
	p := dir.path("config.json")
	b, err := dir.read("config.json")
	if err != nil {
		// Not optional: without it there is no block count, head count or
		// rotary base, and defaults produce confident nonsense.
		return nil, fmt.Errorf("convert: %s: a safetensors model keeps its "+
			"hyperparameters in config.json beside the weights, and this one has none: %w", dir, err)
	}
	c := &hfConfig{path: p}
	if err := json.Unmarshal(b, c); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", p, err)
	}
	if err := json.Unmarshal(b, &c.keys); err != nil {
		return nil, fmt.Errorf("convert: %s: %w", p, err)
	}
	// transformers 5 moved rope_theta (and the scaling spec) into
	// rope_parameters; reading only the old key would default the base.
	if len(c.RopeParameters) > 0 && string(c.RopeParameters) != "null" {
		var rp struct {
			RopeTheta           float64  `json:"rope_theta"`
			PartialRotaryFactor *float64 `json:"partial_rotary_factor"`
		}
		if err := json.Unmarshal(c.RopeParameters, &rp); err != nil {
			return nil, fmt.Errorf("convert: %s: rope_parameters: %w", p, err)
		}
		if c.RopeTheta == 0 {
			c.RopeTheta = rp.RopeTheta
		}
		if c.PartialRotaryFactor == nil {
			c.PartialRotaryFactor = rp.PartialRotaryFactor
		}
		if len(c.RopeScaling) == 0 || string(c.RopeScaling) == "null" {
			c.RopeScaling = c.RopeParameters
		}
	}
	return c, nil
}

// act is the feed-forward activation. hidden_activation (transformers 4) is
// the corrected key for gemma's mislabelled "gelu" (trained as the tanh
// approximation); where present it is the answer, even though transformers 5
// reads hidden_act alone.
func (c *hfConfig) act() string {
	if c.HiddenActivation != "" {
		return c.HiddenActivation
	}
	return c.HiddenAct
}

// ---------------------------------------------------------------------------
// the architecture list

// hfArch is one architecture's HuggingFace vocabulary: its tensor names, its
// config reader, and whatever it does to a tensor on the way in.
//
// It is a list and adding one is a decision (RULE 7): a name mapping that is
// 95% right loads and is wrong, so each entry has a fixture in
// testdata/golden/hf that model.TestSafetensorsMatchTransformers compares
// against transformers.
type hfArch struct {
	arch jlm.Arch
	// model holds whole-name, model-level tensors.
	model map[string]jlm.Role
	// block holds the suffix after "model.layers.N.".
	block map[string]jlm.Role
	// blockLinear overrides block for layers whose kind is LayerLinearAttn,
	// and is consulted first for those: in a hybrid one name means two things
	// (Kimi-Linear's self_attn.q_proj is the recurrent block's query in one
	// layer and MLA's in another, with different shapes). The kind comes from
	// Config.LayerKinds, read before identify runs.
	blockLinear map[string]jlm.Role
	// ignore names tensors that are derived rather than trained. It is an
	// explicit list, not a skip: an unrecognised name is still a refusal.
	ignore map[string]bool
	// ignorePrefix names whole subtrees the container does not carry -- a
	// vision tower this converter does not build -- by their prefix. Like
	// ignore it is a list, so a name outside it is still identified or refused.
	ignorePrefix []string
	// blockPrefix is the per-layer name prefix, "model.layers." for llama.
	blockPrefix string
	// expertPrefix is the block suffix before an expert's index, and expert
	// maps what follows the index onto a bank role: HuggingFace ships one
	// matrix per expert, the container one bank per role (stackBanks).
	expertPrefix string
	expert       map[string]jlm.Role
	// fused names a role whose tensor is several projections' rows back to
	// back, and the roles they split into, in order (splitRows).
	fused map[jlm.Role][]jlm.Role
	// split cuts one source tensor into several container tensors where the cut
	// is not a plain row range. It is keyed by the block-relative name, not the
	// role: MLA's up-projections arrive either as one kv_b_proj to cut or
	// already as two halves, both producing the same two roles, so a role key
	// would re-cut a tensor that is already whole.
	split map[string]hfSplit
	// fuseOrder is split run backwards: it names, in row order, the
	// block-relative source tensors that concatenate into one container role
	// (several sources map to the same role in block/blockLinear). It is an
	// order because the rows are positional: deltaGeom slices the mixed
	// projection q|k|v by arithmetic, so any other order has the right shape
	// and every head wrong.
	fuseOrder map[jlm.Role][]string
	config    func(*hfConfig, []string) (*jlm.Config, error)
	// gather folds source tensors whose VALUES the converter needs into
	// tensors of its own before anything is written (Apertus's four xIELU
	// scalars a block become one RoleXIELU vector). It reads through read and
	// returns what it made and the source names it consumed, which the
	// identify loop then skips. Optional.
	gather func(c *jlm.Config, names []string, read func(name string) ([]float32, error)) ([]jlm.Tensor, map[string]bool, error)
	// fixup rewrites a tensor's bytes on the way in. Optional: an architecture
	// whose weights are already in the container's layout leaves it nil and
	// setData carries the bytes through.
	fixup func(*jlm.Tensor, []byte, *jlm.Config) error
	// retype gives a tensor a container type its source dtype does not map
	// to, before anything is laid out: the type, the [rows, cols] shape and
	// the read that produces its bytes (BitNet's ternary projections, from
	// packed codes or latent weights). nil for a tensor it leaves alone; read
	// returns any source tensor's bytes. Optional.
	retype func(hc *hfConfig, role jlm.Role, name string, t *safetensors.Tensor,
		read func(name string) ([]byte, *safetensors.Tensor, error)) (*hfRetyped, error)
}

// hfRetyped is a tensor retype gave its own container type: its shape is
// safetensors' [rows, cols], and read yields the container's bytes.
type hfRetyped struct {
	ty    jlm.Type
	shape []uint64
	read  func() ([]byte, error)
}

// hfSplit cuts one source tensor into the container tensors it holds. Each
// returned tensor carries its own Data; the caller does not touch them again
// except to check their shapes.
type hfSplit func(jlm.Tensor, []byte, *jlm.Config) ([]jlm.Tensor, error)

// blockRest is the block-relative name identify matched -- the key `split`
// uses -- or name itself for a model-level tensor. identify has already
// validated the index by the time this runs, so a parse failure here can only
// mean the caller asked about a name identify rejected.
func (a *hfArch) blockRest(name string) string {
	if a.blockPrefix == "" || !strings.HasPrefix(name, a.blockPrefix) {
		return name
	}
	_, rest, err := leadingIndex(name, name[len(a.blockPrefix):])
	if err != nil {
		return name
	}
	return rest
}

// setData runs the architecture's fixup, or carries the bytes through.
func (a *hfArch) setData(e *jlm.Tensor, data []byte, c *jlm.Config) error {
	if a.fixup == nil {
		e.Data = data
		return nil
	}
	return a.fixup(e, data, c)
}

// hfArchTable maps the class name in config.json's "architectures" onto an
// entry. See hfArch.
var hfArchTable = map[string]*hfArch{
	"LlamaForCausalLM":    llamaHF,
	"Qwen2ForCausalLM":    qwen2HF,
	"Qwen3ForCausalLM":    qwen3HF,
	"Qwen3MoeForCausalLM": qwen3moeHF,
	"OlmoeForCausalLM":    olmoeHF,
	"MixtralForCausalLM":  mixtralHF,
	"GemmaForCausalLM":    gemmaHF,
	"Phi3ForCausalLM":     phi3HF,
	// One graph, two class names: V2 and V3 differ in the router, and every
	// difference is a key config.json states. See deepseek2HF.
	"DeepseekV2ForCausalLM": deepseek2HF,
	"DeepseekV3ForCausalLM": deepseek2HF,
	// Glm4MoeLiteForCausalLM (GLM-4.7-Flash) carries exactly the names this
	// entry maps, over kv_lora_rank 512 and q_lora_rank 768. Matching names are
	// not a matching graph, so it is gated by scripts/hfgold.py's
	// synth-glm4moelite in model.TestSafetensorsMatchTransformers. Its sibling
	// Glm4MoeForCausalLM (GLM-4.5-Air) has no kv_lora_rank and is not MLA, so
	// it is deliberately absent.
	"Glm4MoeLiteForCausalLM": deepseek2HF,
	"KimiLinearForCausalLM":  kimiLinearHF,
	// Kimi-K3's text model, its experts compressed-tensors MXFP4 in the
	// release (convert/hfkimik3.go, convert/hfmxfp4.go).
	"KimiK3ForConditionalGeneration": kimiK3HF,
	// The mixture families, each to the container its GGUF converts to. See
	// convert/hfmoefamily.go.
	"Glm4MoeForCausalLM":        glm4moeHF,
	"Qwen2MoeForCausalLM":       qwen2moeHF,
	"Ernie4_5ForCausalLM":       ernie45HF,
	"Ernie4_5_MoeForCausalLM":   ernie45moeHF,
	"HunYuanDenseV1ForCausalLM": hunyuanHF,
	"HunYuanMoEV1ForCausalLM":   hunyuanHF,
	// The modern text group (convert/hfmoefamily2.go).
	"Dots1ForCausalLM": dots1HF,
	// BitNet b1.58 (convert/hfbitnet.go).
	"BitNetForCausalLM": bitnetHF,
	// Phi-3.5-MoE and its smaller siblings name the class as their remote
	// code did; transformers saves its own spelling (convert/hfphimoe.go).
	"PhiMoEForCausalLM": phimoeHF,
	"PhimoeForCausalLM": phimoeHF,
	// Apertus (convert/hfapertus.go).
	"ApertusForCausalLM": apertusHF,
	// The dense llama family (convert/safetensors_dense.go). OLMo 3 is
	// ArchOLMo2 with a window, as on the GGUF side. Ministral3ForCausalLM is
	// mistral3's text model; Mistral3ForConditionalGeneration, which nests it
	// under a vision tower, is not here.
	"SmolLM3ForCausalLM":    smollm3HF,
	"ArceeForCausalLM":      arceeHF,
	"SeedOssForCausalLM":    seedOssHF,
	"Olmo2ForCausalLM":      olmo2HF,
	"Olmo3ForCausalLM":      olmo3HF,
	"Exaone4ForCausalLM":    exaone4HF,
	"Ministral3ForCausalLM": ministral3HF,
}

func hfArchOf(c *hfConfig) (*hfArch, error) {
	for _, n := range c.Architectures {
		if a, ok := hfArchTable[n]; ok {
			return a, nil
		}
	}
	// architectures is the authority and model_type is not a fallback: one
	// model_type ("llama") covers several classes, and only the causal LM one
	// has this graph.
	have := strings.Join(c.Architectures, ", ")
	if have == "" {
		have = "none, and model_type is " + strconv.Quote(c.ModelType)
	}
	return nil, fmt.Errorf("convert: %s: architectures %s is not implemented from "+
		"safetensors (implemented: %s)", c.path, have, strings.Join(HFClasses(), ", "))
}

// HFClasses is every Hugging Face class (config.json's architectures) the
// safetensors path converts -- hfArchTable, which is the list -- sorted.
func HFClasses() []string {
	names := make([]string, 0, len(hfArchTable))
	for n := range hfArchTable {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// HFClassGraph is the graph a Hugging Face class converts to, and whether the
// class is in the list. As on the GGUF side, the config can refine it (OLMo 3
// with a window is jlm.ArchOLMo3); this is the class's own entry.
func HFClassGraph(class string) (jlm.Arch, bool) {
	a, ok := hfArchTable[class]
	if !ok {
		return jlm.ArchNone, false
	}
	return a.arch, true
}

// identify turns a HuggingFace tensor name into the triple a container stores.
// An unrecognised name is a refusal, as on the GGUF side. expert is the
// expert's index for a per-expert matrix and -1 otherwise. kinds is
// Config.LayerKinds, empty for a non-hybrid; it selects blockLinear for the
// recurrent layers.
func (a *hfArch) identify(name string, kinds []jlm.LayerKind) (role jlm.Role, block, expert int32, ignored bool, err error) {
	if a.ignore[name] {
		return 0, 0, -1, true, nil
	}
	for _, p := range a.ignorePrefix {
		if strings.HasPrefix(name, p) {
			return 0, 0, -1, true, nil
		}
	}
	rest := name
	block = jlm.DenseBlock
	if strings.HasPrefix(rest, a.blockPrefix) {
		n, r, err := leadingIndex(name, rest[len(a.blockPrefix):])
		if err != nil {
			return 0, 0, -1, false, err
		}
		block, rest = n, r
		if a.ignore[rest] {
			return 0, 0, -1, true, nil
		}
		if int(block) < len(kinds) && kinds[block] == jlm.LayerLinearAttn {
			if rr, ok := a.blockLinear[rest]; ok {
				return rr, block, -1, false, nil
			}
		}
		if rr, ok := a.block[rest]; ok {
			return rr, block, -1, false, nil
		}
		if a.expertPrefix != "" && strings.HasPrefix(rest, a.expertPrefix) {
			e, r, err := leadingIndex(name, rest[len(a.expertPrefix):])
			if err != nil {
				return 0, 0, -1, false, err
			}
			if rr, ok := a.expert[r]; ok {
				return rr, block, e, false, nil
			}
		}
		return 0, 0, -1, false, fmt.Errorf("convert: tensor %q: %q has no role in this "+
			"container's vocabulary; a tensor dropped at conversion is a model that "+
			"loads and is wrong, so this is a refusal", name, rest)
	}
	if rr, ok := a.model[rest]; ok {
		return rr, jlm.DenseBlock, -1, false, nil
	}
	return 0, 0, -1, false, fmt.Errorf("convert: tensor %q has no role in this container's "+
		"vocabulary; a tensor dropped at conversion is a model that loads and is wrong, "+
		"so this is a refusal", name)
}

// leadingIndex splits "N.rest" into N and rest.
func leadingIndex(name, s string) (int32, string, error) {
	i := strings.IndexByte(s, '.')
	if i <= 0 {
		return 0, "", fmt.Errorf("convert: %q: no index", name)
	}
	n, err := strconv.Atoi(s[:i])
	if err != nil || n < 0 || n > math.MaxInt32 {
		return 0, "", fmt.Errorf("convert: %q: index %q", name, s[:i])
	}
	return int32(n), s[i+1:], nil
}

// hfCheckShape re-derives every tensor's shape from the config and refuses a
// mismatch. It is the gate on the name table: a q projection wired to k has
// the same shape under MHA and answers fluently, so the mapping can only be
// checked against arithmetic (gate/up are NFFN rows, q is NHead*HeadDim...).
func hfCheckShape(e *jlm.Tensor, c *jlm.Config) error {
	want := func(dims ...uint64) error {
		if int(e.NDim) != len(dims) {
			return fmt.Errorf("convert: %s is %d-dimensional, and %v is %d",
				e.Name, e.NDim, e.Role, len(dims))
		}
		for i, d := range dims {
			if e.Dims[i] != d {
				return fmt.Errorf("convert: %s is %v and %v wants %v",
					e.Name, e.Dims[:e.NDim], e.Role, dims)
			}
		}
		return nil
	}
	qrows := uint64(c.NHead) * uint64(c.HeadDim)
	kvrows := uint64(c.NKVHead) * uint64(c.HeadDim)
	// The output projection reads the value head, which differs from the key
	// head only under MLA (HeadDimV is zero elsewhere).
	vrows := uint64(c.NHead) * uint64(hfHeadDimV(c))
	embd, ffn, vocab := uint64(c.NEmbd), uint64(c.NFFN), uint64(c.NVocab)
	exp, fexp := uint64(c.NExpert), uint64(c.NFFNExp)
	lat, nope := uint64(c.KVLoraRank), uint64(c.HeadDim)-uint64(c.NRot)
	heads, qlat := uint64(c.NHead), uint64(c.QLoraRank)
	// A latent mixture (Kimi-K3) runs its routed experts at ExpertLatent,
	// behind a down- and an up-projection.
	expIn := embd
	if c.ExpertLatent != 0 {
		expIn = uint64(c.ExpertLatent)
	}
	switch e.Role {
	case jlm.RoleFFNRoutedDown:
		return want(embd, expIn)
	case jlm.RoleFFNRoutedUp:
		return want(expIn, embd)
	case jlm.RoleFFNRoutedNorm:
		return want(expIn)
	case jlm.RoleAttnResScore, jlm.RoleFFNResScore, jlm.RoleOutputResScore:
		return want(embd)
	case jlm.RoleAttnQNorm:
		// The width discriminates qwen3's per-head norm from olmoe's
		// whole-projection one.
		if c.Flags.Has(jlm.FlagQKNormWide) {
			return want(qrows)
		}
		return want(uint64(c.HeadDim))
	case jlm.RoleAttnKNorm:
		if c.Flags.Has(jlm.FlagQKNormWide) {
			return want(kvrows)
		}
		return want(uint64(c.HeadDim))
	case jlm.RoleRouter:
		return want(embd, exp)
	case jlm.RoleExpGateBank, jlm.RoleExpUpBank:
		return want(expIn, fexp, exp)
	case jlm.RoleExpDownBank:
		return want(fexp, expIn, exp)
	case jlm.RoleTokenEmbd, jlm.RoleOutput:
		return want(embd, vocab)
	case jlm.RoleOutputNorm, jlm.RoleAttnNorm, jlm.RoleFFNNorm, jlm.RolePostAttnNorm, jlm.RolePostFFNNorm:
		return want(embd)
	case jlm.RoleAttnQ:
		return want(embd, qrows)
	case jlm.RoleAttnK, jlm.RoleAttnV:
		return want(embd, kvrows)
	case jlm.RoleAttnOut:
		return want(vrows, embd)

	// Multi-head Latent Attention. The fixture makes three of these widths
	// equal on purpose, so these say which is which.
	case jlm.RoleAttnQA:
		return want(embd, qlat)
	case jlm.RoleAttnQANorm:
		return want(qlat)
	case jlm.RoleAttnQB:
		return want(qlat, qrows)
	case jlm.RoleAttnKVA:
		return want(embd, lat+uint64(c.NRot))
	case jlm.RoleAttnKVANorm:
		return want(lat)
	case jlm.RoleAttnKB:
		// The fixture has qk_nope == kv_lora_rank == v_head_dim, so this cannot
		// see the transpose; the arithmetic gate in deepseek_test.go does.
		return want(nope, lat, heads)
	case jlm.RoleAttnVB:
		return want(lat, uint64(hfHeadDimV(c)), heads)

	case jlm.RoleExpProbsB, jlm.RoleRouterBias:
		// ERNIE's class keeps its correction bias as [1, n], and llama.cpp's
		// converter writes that shape unchanged: the same bytes either way.
		if e.NDim == 2 {
			return want(exp, 1)
		}
		return want(exp)
	case jlm.RoleShExpGate, jlm.RoleShExpUp:
		return want(embd, uint64(c.NFFNShExp))
	case jlm.RoleShExpDown:
		return want(uint64(c.NFFNShExp), embd)

	case jlm.RoleAttnQBias:
		return want(qrows)
	case jlm.RoleAttnKBias, jlm.RoleAttnVBias:
		return want(kvrows)
	case jlm.RoleAttnOutBias:
		return want(embd)
	case jlm.RoleFFNGate, jlm.RoleFFNUp:
		return want(embd, ffn)
	case jlm.RoleFFNDown:
		return want(ffn, embd)

	case jlm.RoleNextnEHProj:
		return want(2*embd, embd)
	case jlm.RoleNextnENorm, jlm.RoleNextnHNorm, jlm.RoleNextnHeadNorm:
		return want(embd)
	case jlm.RoleNextnHead, jlm.RoleNextnEmbd:
		return want(embd, vocab)
	}
	return nil
}

// ---------------------------------------------------------------------------
// llama

var llamaHF = &hfArch{
	arch:        jlm.ArchLlama,
	blockPrefix: "model.layers.",
	model: map[string]jlm.Role{
		"model.embed_tokens.weight": jlm.RoleTokenEmbd,
		"model.norm.weight":         jlm.RoleOutputNorm,
		"lm_head.weight":            jlm.RoleOutput,
	},
	block: map[string]jlm.Role{
		"input_layernorm.weight": jlm.RoleAttnNorm,
		// post_attention_layernorm is the FFN's input norm (ffn_norm), not a
		// post-norm; gemma2/gemma3's similarly named tensor is a real post-norm.
		// This is the trap retarget() handles on the GGUF side.
		"post_attention_layernorm.weight": jlm.RoleFFNNorm,

		"self_attn.q_proj.weight": jlm.RoleAttnQ,
		"self_attn.k_proj.weight": jlm.RoleAttnK,
		"self_attn.v_proj.weight": jlm.RoleAttnV,
		"self_attn.o_proj.weight": jlm.RoleAttnOut,
		"self_attn.q_proj.bias":   jlm.RoleAttnQBias,
		"self_attn.k_proj.bias":   jlm.RoleAttnKBias,
		"self_attn.v_proj.bias":   jlm.RoleAttnVBias,
		"self_attn.o_proj.bias":   jlm.RoleAttnOutBias,

		"mlp.gate_proj.weight": jlm.RoleFFNGate,
		"mlp.up_proj.weight":   jlm.RoleFFNUp,
		"mlp.down_proj.weight": jlm.RoleFFNDown,
	},
	// inv_freq is derived (a function of rope_theta and the head dim, which the
	// config carries), so it is named and ignored rather than stored as a
	// second source of truth.
	ignore: map[string]bool{
		"model.rotary_emb.inv_freq":     true,
		"self_attn.rotary_emb.inv_freq": true,
	},
	config: hfConfigWith(jlm.ArchLlama, 0, "silu"),
	fixup:  llamaHFFixup,
}

// qwen2 is llama plus the attention biases, which the block map already
// has: the file decides whether a bias exists. It is NEOX and unpermuted,
// as its GGUF is: convert_hf_to_gguf.py permutes only llama's rows. For
// qwen3 permuting would also misalign the per-head q/k norm weight.
var qwen2HF = &hfArch{
	arch:        jlm.ArchQwen2,
	blockPrefix: llamaHF.blockPrefix,
	model:       llamaHF.model,
	block:       llamaHF.block,
	ignore:      llamaHF.ignore,
	config:      hfConfigWith(jlm.ArchQwen2, jlm.FlagRopeNeox, "silu"),
}

// qwen3 is llama plus a per-head q/k norm of width [head_dim] (FlagQKNorm);
// olmoe ships the same names at [hidden_size] (FlagQKNormWide). hfCheckShape
// checks the width, so the other shape is refused.
var qwen3Block = blockPlus(llamaHF.block, map[string]jlm.Role{
	"self_attn.q_norm.weight": jlm.RoleAttnQNorm,
	"self_attn.k_norm.weight": jlm.RoleAttnKNorm,
})

var qwen3HF = &hfArch{
	arch:        jlm.ArchQwen3,
	blockPrefix: llamaHF.blockPrefix,
	model:       llamaHF.model,
	block:       qwen3Block,
	ignore:      llamaHF.ignore,
	config:      hfConfigWith(jlm.ArchQwen3, jlm.FlagRopeNeox|jlm.FlagQKNorm, "silu"),
}

// The HuggingFace mixture spelling qwen3moe and olmoe share: a router at
// mlp.gate and one SwiGLU per expert under mlp.experts.N.
var hfExperts = map[string]jlm.Role{
	"gate_proj.weight": jlm.RoleExpGateBank,
	"up_proj.weight":   jlm.RoleExpUpBank,
	"down_proj.weight": jlm.RoleExpDownBank,
}

// qwen3's attention with the dense FFN replaced by a router and experts.
var qwen3moeHF = &hfArch{
	arch:         jlm.ArchQwen3MoE,
	blockPrefix:  llamaHF.blockPrefix,
	model:        llamaHF.model,
	block:        blockPlus(qwen3Block, map[string]jlm.Role{"mlp.gate.weight": jlm.RoleRouter}),
	expertPrefix: "mlp.experts.",
	expert:       hfExperts,
	ignore:       llamaHF.ignore,
	config:       hfConfigWith(jlm.ArchQwen3MoE, jlm.FlagRopeNeox|jlm.FlagQKNorm, "silu"),
}

// olmoe is qwen3moe's names over two different graph facts: its q/k norm
// spans the whole projection (checked by shape), and its mixture weights are
// not renormalised (norm_topk_prob, read by hfMoEConfig).
var olmoeHF = &hfArch{
	arch:         jlm.ArchOLMoE,
	blockPrefix:  llamaHF.blockPrefix,
	model:        llamaHF.model,
	block:        qwen3moeHF.block,
	expertPrefix: "mlp.experts.",
	expert:       hfExperts,
	ignore:       llamaHF.ignore,
	config: func(c *hfConfig, names []string) (*jlm.Config, error) {
		// clip_qkv clamps q, k and v to [-x, x]; nothing here does.
		if c.ClipQKV != nil {
			return nil, fmt.Errorf("convert: %s: clip_qkv %v is not implemented", c.path, *c.ClipQKV)
		}
		cfg, err := hfConfigWith(jlm.ArchOLMoE, jlm.FlagRopeNeox|jlm.FlagQKNorm|jlm.FlagQKNormWide, "silu")(c, names)
		if err != nil {
			return nil, err
		}
		// The same refusal configOf makes: a whole-projection k norm under GQA
		// would need a width nothing states.
		if cfg.NKVHead != cfg.NHead {
			return nil, fmt.Errorf("convert: %s: a whole-projection q/k norm with GQA "+
				"(%d q heads, %d kv heads) is not implemented: %w", c.path, cfg.NHead, cfg.NKVHead, ErrNotImplemented)
		}
		return cfg, nil
	},
}

// mixtral is llama with the FFN swapped, as its GGUF names it, so it keeps
// llama's rotary permutation and Arch. Its router always renormalises the
// top-k; the class has no norm_topk_prob.
var mixtralHF = &hfArch{
	arch:        jlm.ArchLlama,
	blockPrefix: llamaHF.blockPrefix,
	model:       llamaHF.model,
	block: map[string]jlm.Role{
		"input_layernorm.weight":          jlm.RoleAttnNorm,
		"post_attention_layernorm.weight": jlm.RoleFFNNorm,
		"self_attn.q_proj.weight":         jlm.RoleAttnQ,
		"self_attn.k_proj.weight":         jlm.RoleAttnK,
		"self_attn.v_proj.weight":         jlm.RoleAttnV,
		"self_attn.o_proj.weight":         jlm.RoleAttnOut,
		"block_sparse_moe.gate.weight":    jlm.RoleRouter,
	},
	expertPrefix: "block_sparse_moe.experts.",
	expert: map[string]jlm.Role{
		"w1.weight": jlm.RoleExpGateBank,
		"w3.weight": jlm.RoleExpUpBank,
		"w2.weight": jlm.RoleExpDownBank,
	},
	ignore: llamaHF.ignore,
	config: func(c *hfConfig, names []string) (*jlm.Config, error) {
		cfg, err := hfConfigWith(jlm.ArchLlama, 0, "silu")(c, names)
		if err != nil {
			return nil, err
		}
		cfg.Flags &^= jlm.FlagNoExpertNorm
		return cfg, nil
	},
	fixup: llamaHFFixup,
}

// gemma differs from llama in four silent places: NEOX rotary with rows as
// shipped, a tanh-GELU gate, the embedding scaled by sqrt(hidden_size), and an
// RMSNorm that multiplies by (1 + weight). The +1 is folded into the weight
// here, as its GGUF carries it. The class defaults tie_word_embeddings to
// true, so an absent key means tied.
var gemmaHF = &hfArch{
	arch:        jlm.ArchGemma,
	blockPrefix: llamaHF.blockPrefix,
	model:       llamaHF.model,
	block:       llamaHF.block,
	ignore:      llamaHF.ignore,
	config: func(c *hfConfig, names []string) (*jlm.Config, error) {
		if c.TieWordEmbeddings == nil {
			t := true
			c.TieWordEmbeddings = &t
		}
		cfg, err := hfConfigWith(jlm.ArchGemma, jlm.FlagRopeNeox|jlm.FlagGELU, "gelu_pytorch_tanh")(c, names)
		if err != nil {
			return nil, err
		}
		cfg.EmbdScale = float32(math.Sqrt(float64(cfg.NEmbd)))
		return cfg, nil
	},
	fixup: gemmaHFFixup,
}

// phi3 fuses q/k/v into qkv_proj and gate/up into gate_up_proj; both are the
// file's packing and are split here, as convert.unfuse does on the GGUF side.
// Gate comes first: Phi3MLP multiplies the activation of the first half into
// the second. RoleFFNUp is the fused tensor's role, as the schema says.
var phi3HF = &hfArch{
	arch:        jlm.ArchPhi3,
	blockPrefix: llamaHF.blockPrefix,
	model:       llamaHF.model,
	block: map[string]jlm.Role{
		"input_layernorm.weight":          jlm.RoleAttnNorm,
		"post_attention_layernorm.weight": jlm.RoleFFNNorm,
		"self_attn.qkv_proj.weight":       jlm.RoleAttnQKV,
		"self_attn.o_proj.weight":         jlm.RoleAttnOut,
		"mlp.gate_up_proj.weight":         jlm.RoleFFNUp,
		"mlp.down_proj.weight":            jlm.RoleFFNDown,
	},
	fused: map[jlm.Role][]jlm.Role{
		jlm.RoleAttnQKV: {jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV},
		jlm.RoleFFNUp:   {jlm.RoleFFNGate, jlm.RoleFFNUp},
	},
	ignore: llamaHF.ignore,
	config: func(c *hfConfig, names []string) (*jlm.Config, error) {
		// phi3 slides every layer (allLocal), as the GGUF path does.
		win := uint32(0)
		if c.SlidingWindow != nil && (c.UseSlidingWindow == nil || *c.UseSlidingWindow) {
			win, c.SlidingWindow = *c.SlidingWindow, nil
		}
		cfg, err := hfConfigWith(jlm.ArchPhi3, jlm.FlagRopeNeox, "silu")(c, names)
		if err != nil {
			return nil, err
		}
		if win > 0 {
			cfg.SWAWindow, cfg.SWAPeriod = win, allLocal(cfg.NLayer)
		}
		return cfg, nil
	},
}

// ---------------------------------------------------------------------------
// deepseek2 -- DeepSeek-V2/V3/R1, and the models that do not carry the name
//
// One entry for two class names: V2 and V3 differ only in the router, and
// every difference is a key config.json states (scoring_func,
// e_score_correction_bias, n_group). Kimi-K2 declares DeepseekV3ForCausalLM.
//
// Dense and mixture blocks spell their feed-forward with disjoint names, so one
// flat table identifies both; deepseekLayerSets checks each name lands on a
// block first_k_dense_replace agrees with.
var deepseek2HF = &hfArch{
	arch:        jlm.ArchDeepseek2,
	blockPrefix: llamaHF.blockPrefix,
	model:       llamaHF.model,
	block: map[string]jlm.Role{
		"input_layernorm.weight":          jlm.RoleAttnNorm,
		"post_attention_layernorm.weight": jlm.RoleFFNNorm,

		// MLA. q_a_proj/q_a_layernorm/q_b_proj are the two-step query, absent
		// when q_lora_rank is null; q_proj is the one-step form. deepseekQueryForm
		// checks they are exclusive.
		"self_attn.q_a_proj.weight":           jlm.RoleAttnQA,
		"self_attn.q_a_layernorm.weight":      jlm.RoleAttnQANorm,
		"self_attn.q_b_proj.weight":           jlm.RoleAttnQB,
		"self_attn.q_proj.weight":             jlm.RoleAttnQ,
		"self_attn.kv_a_proj_with_mqa.weight": jlm.RoleAttnKVA,
		"self_attn.kv_a_layernorm.weight":     jlm.RoleAttnKVANorm,
		"self_attn.o_proj.weight":             jlm.RoleAttnOut,
		// The un-absorbed pair, cut by deepseekSplitKVB. A checkpoint that
		// ships the halves already separate names them k_b_proj / v_b_proj and
		// takes neither branch.
		"self_attn.kv_b_proj.weight": jlm.RoleAttnKB,
		"self_attn.k_b_proj.weight":  jlm.RoleAttnKB,
		"self_attn.v_b_proj.weight":  jlm.RoleAttnVB,

		// The dense leading block.
		"mlp.gate_proj.weight": jlm.RoleFFNGate,
		"mlp.up_proj.weight":   jlm.RoleFFNUp,
		"mlp.down_proj.weight": jlm.RoleFFNDown,

		// The mixture blocks. e_score_correction_bias is added to the router's
		// scores for selection only, not to the weight, hence a separate role.
		"mlp.gate.weight":                     jlm.RoleRouter,
		"mlp.gate.e_score_correction_bias":    jlm.RoleExpProbsB,
		"mlp.shared_experts.gate_proj.weight": jlm.RoleShExpGate,
		"mlp.shared_experts.up_proj.weight":   jlm.RoleShExpUp,
		"mlp.shared_experts.down_proj.weight": jlm.RoleShExpDown,

		// A multi-token-prediction block's own tensors (blocks past
		// num_hidden_layers; jlm.Config.NMTP). Its head and embedding are
		// copies of the trunk's in every published checkpoint, which
		// nextnTensors drops.
		"eh_proj.weight":          jlm.RoleNextnEHProj,
		"enorm.weight":            jlm.RoleNextnENorm,
		"hnorm.weight":            jlm.RoleNextnHNorm,
		"shared_head.norm.weight": jlm.RoleNextnHeadNorm,
		"shared_head.head.weight": jlm.RoleNextnHead,
		"embed_tokens.weight":     jlm.RoleNextnEmbd,
	},
	expertPrefix: "mlp.experts.",
	expert:       hfExperts,
	ignore:       llamaHF.ignore,
	split: map[string]hfSplit{
		"self_attn.kv_b_proj.weight": deepseekSplitKVB,
	},
	config: deepseekHFConfig,
}

func deepseekHFConfig(c *hfConfig, names []string) (*jlm.Config, error) {
	// The prediction blocks are this spelling's: Kimi-Linear, the other caller
	// of deepseekHFConfigFFN, ships none. DeepseekV3Config defaults the count
	// to 1, and transformers drops the block's tensors on load and so on save:
	// a checkpoint that declares a block and carries none of it has none.
	if c.NumNextnPredictLayers > 0 {
		first := fmt.Sprintf("model.layers.%d.", c.NumHiddenLayers)
		for _, n := range names {
			if strings.HasPrefix(n, first) {
				c.nextn = c.NumNextnPredictLayers
				break
			}
		}
	}
	return deepseekHFConfigFFN(c, names, "mlp.")
}

// deepseekHFConfigFFN is deepseekHFConfig with the feed-forward's name prefix
// as a parameter. Kimi-Linear carries the same MLA geometry and the same V3
// router under "block_sparse_moe." instead of "mlp.", and the layer-set check
// below is the one place that spelling reaches.
func deepseekHFConfigFFN(c *hfConfig, names []string, ffn string) (*jlm.Config, error) {
	bad := func(f string, a ...any) error {
		return fmt.Errorf("convert: "+c.path+": "+f, a...)
	}
	// Consumed before llamaHFConfig, as phi3 consumes sliding_window:
	// hfRopeScalingOK refuses every scaling, and this architecture applies
	// YaRN, so the key is read and removed rather than loosening the refusal.
	yarn, err := hfYarnConsume(c)
	if err != nil {
		return nil, err
	}
	if c.KVLoraRank == 0 || c.QKNopeHeadDim == 0 || c.QKRopeHeadDim == 0 || c.VHeadDim == 0 {
		return nil, bad("kv_lora_rank=%d qk_nope_head_dim=%d qk_rope_head_dim=%d v_head_dim=%d: "+
			"this architecture is multi-head latent attention and all four describe its geometry",
			c.KVLoraRank, c.QKNopeHeadDim, c.QKRopeHeadDim, c.VHeadDim)
	}
	cfg, err := hfConfigWith(jlm.ArchDeepseek2, 0, "silu")(c, names)
	if err != nil {
		return nil, err
	}
	// DeepseekV3Config sets head_dim = qk_rope_head_dim (the rotary half), so
	// the head width llamaHFConfig took must be overwritten.
	cfg.HeadDim = c.QKNopeHeadDim + c.QKRopeHeadDim
	cfg.NRot = c.QKRopeHeadDim
	cfg.HeadDimV = c.VHeadDim
	if cfg.HeadDimV == cfg.HeadDim {
		cfg.HeadDimV = 0 // stored only when it says something HeadDim does not
	}
	cfg.QLoraRank, cfg.KVLoraRank = c.QLoraRank, c.KVLoraRank
	// MLA has no GQA: the latent is shared by every head (the "mqa" in
	// kv_a_proj_with_mqa), so a disagreeing num_key_value_heads is refused.
	if cfg.NKVHead != cfg.NHead {
		return nil, bad("num_key_value_heads %d against %d attention heads: latent "+
			"attention shares one key-value latent across every head", cfg.NKVHead, cfg.NHead)
	}
	if err := deepseekQueryForm(c, cfg, names); err != nil {
		return nil, err
	}
	cfg.NMTP = c.nextn
	if err := deepseekMoE(c, cfg, names, ffn); err != nil {
		return nil, err
	}
	hfYarnApply(yarn, cfg)
	return cfg, nil
}

// deepseekQueryForm checks that the checkpoint ships exactly one of the two
// query spellings, and the one q_lora_rank names.
// Both forms are real and only the config says which: V2-Lite has
// "q_lora_rank": null and a one-step q_proj; larger models go through
// q_a_proj, a norm and q_b_proj. A mismatch would leave weights unread.
func deepseekQueryForm(c *hfConfig, cfg *jlm.Config, names []string) error {
	var oneStep, twoStep bool
	for _, n := range names {
		switch {
		case strings.HasSuffix(n, ".self_attn.q_proj.weight"):
			oneStep = true
		case strings.HasSuffix(n, ".self_attn.q_a_proj.weight"):
			twoStep = true
		}
	}
	switch {
	case cfg.QLoraRank == 0 && twoStep:
		return fmt.Errorf("convert: %s: q_lora_rank is null or absent and the weights "+
			"carry q_a_proj: the query is projected in two steps and nothing would read them", c.path)
	case cfg.QLoraRank != 0 && oneStep:
		return fmt.Errorf("convert: %s: q_lora_rank is %d and the weights carry q_proj: "+
			"the query is projected in one step and the rank is unread", c.path, cfg.QLoraRank)
	case !oneStep && !twoStep:
		return fmt.Errorf("convert: %s: no q_proj and no q_a_proj: this model has no "+
			"query projection", c.path)
	}
	return nil
}

// deepseekMoE reads the routed mixture, the shared expert and the leading dense
// run.
//
// hfMoEConfig has already run and seen no mixture, which is correct: DeepSeek
// spells its count n_routed_experts, and its knobs have no equivalent there.
func deepseekMoE(c *hfConfig, cfg *jlm.Config, names []string, ffn string) error {
	bad := func(f string, a ...any) error {
		return fmt.Errorf("convert: "+c.path+": "+f, a...)
	}
	if c.NRoutedExperts == 0 {
		// A dense DeepSeek is not a shape any checkpoint ships, and accepting
		// one silently would mean first_k_dense_replace went unread.
		return bad("n_routed_experts is 0: this architecture is a mixture")
	}
	if c.MoEIntermediateSize == 0 {
		return bad("moe_intermediate_size is 0")
	}
	cfg.NExpert, cfg.NExpertUsed, cfg.NFFNExp = c.NRoutedExperts, c.NumExpertsPerTok, uint32(c.MoEIntermediateSize)
	if cfg.NExpertUsed == 0 || cfg.NExpertUsed > cfg.NExpert {
		return bad("num_experts_per_tok %d of %d experts", c.NumExpertsPerTok, c.NRoutedExperts)
	}
	// DeepSeekV3MLP builds the shared expert at moe_intermediate_size *
	// n_shared_experts: one feed-forward, not several tensors.
	cfg.NFFNShExp = c.NSharedExperts * uint32(c.MoEIntermediateSize)
	// DeepseekV3Config defaults norm_topk_prob to true, the opposite of
	// hfMoEConfig's rule for qwen3moe, so only an explicit false disables it.
	if c.NormTopKProb != nil && !*c.NormTopKProb {
		cfg.Flags |= jlm.FlagNoExpertNorm
	}
	// An absent scoring_func is not one default for every class: DeepseekV2
	// implements softmax and carries no key, while DeepseekV3's, Glm4MoeLite's
	// and KimiLinear's routers hardcode sigmoid and never read it. GLM-4.7-Flash
	// omits the key, and converted as softmax it is fluent and wrong. See
	// hardcodesSigmoid.
	sigmoid := c.ScoringFunc == "sigmoid"
	switch c.ScoringFunc {
	case "":
		sigmoid = hardcodesSigmoid(c.Architectures)
	case "softmax", "sigmoid":
	default:
		return fmt.Errorf("convert: %s: scoring_func %q is not implemented: %w",
			c.path, c.ScoringFunc, ErrNotImplemented)
	}
	if sigmoid {
		cfg.Flags |= jlm.FlagExpertSigmoid
	}
	if c.RoutedScalingFactor != 0 {
		cfg.ExpertScale = float32(c.RoutedScalingFactor)
	}
	// Grouped selection. n_group 1 with topk_group 1 is the degenerate case
	// Kimi-K2 ships and it must reduce to the ungrouped path, so it is carried
	// rather than normalised away here.
	cfg.NExpertGroup, cfg.NExpertGroupUsed = c.NGroup, c.TopkGroup
	if cfg.NExpertGroup > 1 {
		switch {
		case cfg.NExpert%cfg.NExpertGroup != 0:
			return bad("%d experts do not divide into n_group %d", cfg.NExpert, cfg.NExpertGroup)
		case cfg.NExpertGroupUsed == 0 || cfg.NExpertGroupUsed > cfg.NExpertGroup:
			return bad("topk_group %d of n_group %d", cfg.NExpertGroupUsed, cfg.NExpertGroup)
		}
	}
	if c.FirstKDenseReplace > cfg.NLayer {
		return bad("first_k_dense_replace %d in a %d-block model", c.FirstKDenseReplace, cfg.NLayer)
	}
	cfg.NDenseLead = c.FirstKDenseReplace
	return deepseekLayerSets(c, cfg, names, ffn)
}

// deepseekLayerSets checks that every block's feed-forward is the kind
// first_k_dense_replace says it is.
//
// The names are disjoint, so without this check the converter would believe
// the key over the weights: a mismatched block would run a router with no
// bank, or never read its dense tensors.
func deepseekLayerSets(c *hfConfig, cfg *jlm.Config, names []string, ffn string) error {
	const pfx = "model.layers."
	dense := make(map[int32]bool, cfg.NLayer)
	moe := make(map[int32]bool, cfg.NLayer)
	for _, n := range names {
		if !strings.HasPrefix(n, pfx) {
			continue
		}
		li, rest, err := leadingIndex(n, n[len(pfx):])
		if err != nil {
			continue
		}
		// A block past the trunk is a multi-token-prediction block, and only
		// as many as num_nextn_predict_layers declares: an undeclared block
		// would be one nothing reads.
		if li >= int32(cfg.NLayer+cfg.NMTP) {
			return fmt.Errorf("convert: %s: %s is in block %d of a %d-block model with %d "+
				"prediction block(s) (num_nextn_predict_layers)", c.path, n, li, cfg.NLayer, cfg.NMTP)
		}
		// A dense block may spell its feed-forward "mlp." even where the
		// mixture is "block_sparse_moe." (Moonshot's modeling_kimi.py and the
		// published Kimi-Linear checkpoint); either is the same three matrices.
		denseAt := func(p string) bool {
			return rest == p+"gate_proj.weight" || rest == p+"up_proj.weight" ||
				rest == p+"down_proj.weight"
		}
		isDense := denseAt(ffn) || denseAt("mlp.")
		if !isDense && !strings.HasPrefix(rest, ffn) {
			continue
		}
		lead := li < int32(cfg.NDenseLead)
		if isDense != lead {
			kind := func(dense bool) string {
				if dense {
					return "dense"
				}
				return "a mixture"
			}
			return fmt.Errorf("convert: %s: block %d carries %s, which is %s, and "+
				"first_k_dense_replace %d makes that block %s", c.path, li, rest,
				kind(isDense), cfg.NDenseLead, kind(lead))
		}
		if isDense {
			dense[li] = true
		} else if rest == ffn+"gate.weight" {
			moe[li] = true
		}
	}
	// The positive half: the loop above only refuses what is present, and a
	// missing router or dense FFN is a silent absence. A prediction block is a
	// block of the same rule.
	for li := int32(0); li < int32(cfg.NLayer+cfg.NMTP); li++ {
		if li < int32(cfg.NDenseLead) {
			if !dense[li] {
				return fmt.Errorf("convert: %s: block %d is inside the leading dense run "+
					"of %d and has no %sgate_proj.weight", c.path, li, cfg.NDenseLead, ffn)
			}
			continue
		}
		if !moe[li] {
			return fmt.Errorf("convert: %s: block %d is a mixture block and has no "+
				"%sgate.weight to route with", c.path, li, ffn)
		}
	}
	return nil
}

// deepseekSplitKVB cuts one kv_b_proj into the two absorbed up-projections the
// container stores, per head.
//
// One of them is transposed and the other is not. Per head h the source holds
//
//	W_k^h : (qk_nope, kv_lora_rank)      W_v^h : (v_head_dim, kv_lora_rank)
//
// and the graph absorbs each into a different operand:
//
//	score = q_nope . (W_k c)  =  (W_k^T q_nope) . c    -> k_b is W_k transposed
//	out   = sum_j a_j (W_v c_j) = W_v (sum_j a_j c_j)  -> v_b is W_v as it is
//
// On the synth fixture every width is 16, so a swap or a missed transpose is
// invisible to shape checks; convert.TestDeepseekKVBSplitIsAbsorbed compares
// numbers. The outputs are sheeted one per head (jlm.sheetsOf) so the graph
// can address a head through SheetSpan; it is also llama.cpp's wk_b shape
// {qk_nope, kv_lora_rank, n_head}.
func deepseekSplitKVB(e jlm.Tensor, data []byte, c *jlm.Config) ([]jlm.Tensor, error) {
	if e.NDim != 2 {
		return nil, fmt.Errorf("convert: %s is %d-dimensional", e.Name, e.NDim)
	}
	nope, lat := uint64(c.HeadDim)-uint64(c.NRot), uint64(c.KVLoraRank)
	vdim, heads := uint64(hfHeadDimV(c)), uint64(c.NHead)
	if e.Dims[0] != lat || e.Dims[1] != heads*(nope+vdim) {
		return nil, fmt.Errorf("convert: %s is %v and %d head(s) of (%d nope + %d value) "+
			"rows over a %d-wide latent want [%d %d]",
			e.Name, e.Dims[:e.NDim], heads, nope, vdim, lat, lat, heads*(nope+vdim))
	}
	w, err := hfElemBytes(e.Type)
	if err != nil {
		return nil, fmt.Errorf("convert: %s: %w", e.Name, err)
	}
	if uint64(len(data)) != heads*(nope+vdim)*lat*uint64(w) {
		return nil, fmt.Errorf("convert: %s is %d bytes and its shape wants %d",
			e.Name, len(data), heads*(nope+vdim)*lat*uint64(w))
	}
	kb := make([]byte, heads*lat*nope*uint64(w))
	vb := make([]byte, heads*vdim*lat*uint64(w))
	for h := uint64(0); h < heads; h++ {
		head := data[h*(nope+vdim)*lat*uint64(w):]
		// W_k^h transposed: source element (i, j) -> sheet element (j, i).
		ks := kb[h*lat*nope*uint64(w):]
		for i := uint64(0); i < nope; i++ {
			for j := uint64(0); j < lat; j++ {
				s := (i*lat + j) * uint64(w)
				d := (j*nope + i) * uint64(w)
				copy(ks[d:d+uint64(w)], head[s:s+uint64(w)])
			}
		}
		// W_v^h verbatim: it already maps the latent to the value head.
		vs := vb[h*vdim*lat*uint64(w):]
		copy(vs, head[nope*lat*uint64(w):(nope+vdim)*lat*uint64(w)])
	}
	mk := func(role jlm.Role, dims [4]uint64, b []byte) jlm.Tensor {
		t := e
		t.Role, t.NDim, t.Dims, t.Data = role, 3, dims, b
		t.Name = fmt.Sprintf("%s[%v]", e.Name, role)
		return t
	}
	return []jlm.Tensor{
		mk(jlm.RoleAttnKB, [4]uint64{nope, lat, heads}, kb),
		mk(jlm.RoleAttnVB, [4]uint64{lat, vdim, heads}, vb),
	}, nil
}

// hfElemBytes is one element's width for the types this reader produces.
// BF16 is absent because hfTypeOf widens it before any of this runs.
func hfElemBytes(t jlm.Type) (int, error) {
	switch t {
	case jlm.TypeF32:
		return 4, nil
	case jlm.TypeF16:
		return 2, nil
	}
	return 0, fmt.Errorf("%v has no element width here", t)
}

// hfYarn is a YaRN rope_scaling as config.json spells it.
type hfYarn struct {
	Factor       float64  `json:"factor"`
	OrigCtx      uint32   `json:"original_max_position_embeddings"`
	BetaFast     *float64 `json:"beta_fast"`
	BetaSlow     *float64 `json:"beta_slow"`
	MScale       *float64 `json:"mscale"`
	MScaleAllDim *float64 `json:"mscale_all_dim"`
	AttnFactor   *float64 `json:"attention_factor"`
}

// hfYarnConsume reads a YaRN rope_scaling and removes it from the config, so
// llamaHFConfig's blanket refusal does not fire on it. Anything else is left
// for hfRopeScalingOK to refuse by name.
func hfYarnConsume(c *hfConfig) (*hfYarn, error) {
	if len(c.RopeScaling) == 0 || string(c.RopeScaling) == "null" {
		return nil, nil
	}
	var m map[string]any
	if err := json.Unmarshal(c.RopeScaling, &m); err != nil {
		return nil, fmt.Errorf("convert: %s: rope_scaling does not parse: %w", c.path, err)
	}
	kind, _ := m["rope_type"].(string)
	if kind == "" {
		kind, _ = m["type"].(string)
	}
	if kind != "yarn" {
		return nil, nil
	}
	var y hfYarn
	if err := json.Unmarshal(c.RopeScaling, &y); err != nil {
		return nil, fmt.Errorf("convert: %s: rope_scaling: %w", c.path, err)
	}
	c.RopeScaling = nil
	return &y, nil
}

// hfYarnApply writes a consumed YaRN into the container's fields, as
// transformers computes it.
//
// The magnitude on cos and sin is _compute_yarn_parameters' attention_factor:
// the config's own where it states one, else mscale(f, mscale) /
// mscale(f, mscale_all_dim) when both are set and non-zero, else mscale(f, 1),
// with mscale(f, k) = 0.1*k*ln(f) + 1. So a DeepSeek carrying mscale ==
// mscale_all_dim (every shipped one) leaves cos and sin at exactly one.
//
// The DeepSeek family's attention classes (DeepseekV2/V3, Glm4MoeLite)
// multiply the softmax scale by mscale(f, mscale_all_dim) squared
// (yarn_apply_mscale) whatever the rotary did; that is YarnLogMul, set only for
// ArchDeepseek2. No other class here scales the score.
func hfYarnApply(y *hfYarn, cfg *jlm.Config) {
	if y == nil || y.Factor <= 1 {
		return
	}
	cfg.YarnFactor = float32(y.Factor)
	cfg.YarnOrigCtx = y.OrigCtx
	if cfg.YarnOrigCtx == 0 {
		cfg.YarnOrigCtx = cfg.NCtx
	}
	cfg.YarnBetaFast, cfg.YarnBetaSlow = 32, 1
	if y.BetaFast != nil {
		cfg.YarnBetaFast = float32(*y.BetaFast)
	}
	if y.BetaSlow != nil {
		cfg.YarnBetaSlow = float32(*y.BetaSlow)
	}
	ms := func(k float64) float64 { return 0.1*k*math.Log(y.Factor) + 1 }
	// Python's truthiness: an absent key and a zero are both false.
	set := func(p *float64) bool { return p != nil && *p != 0 }
	switch {
	case y.AttnFactor != nil:
		cfg.AttnFactor *= float32(*y.AttnFactor)
	case set(y.MScale) && set(y.MScaleAllDim):
		cfg.AttnFactor *= float32(ms(*y.MScale) / ms(*y.MScaleAllDim))
	default:
		cfg.AttnFactor *= float32(ms(1))
	}
	if cfg.Arch == jlm.ArchDeepseek2 && set(y.MScaleAllDim) {
		cfg.YarnLogMul = float32(0.1 * *y.MScaleAllDim)
	}
}

// blockPlus is a copy of base with extra entries, so one architecture's table
// can extend another's without mutating it (a map is a reference).
func blockPlus(base, extra map[string]jlm.Role) map[string]jlm.Role {
	m := make(map[string]jlm.Role, len(base)+len(extra))
	for k, v := range base {
		m[k] = v
	}
	for k, v := range extra {
		m[k] = v
	}
	return m
}

// hfConfigWith is llamaHFConfig for one architecture: its flags, and the one
// activation its feed-forward computes.
func hfConfigWith(arch jlm.Arch, flags jlm.Flags, act string) func(*hfConfig, []string) (*jlm.Config, error) {
	return func(c *hfConfig, names []string) (*jlm.Config, error) {
		// The activation is a graph fact, so a different one is a refusal:
		// erf GELU and gelu_pytorch_tanh differ, and SiLU on a GELU checkpoint
		// runs fluently.
		if got := c.act(); got != "" && got != act && !(act == "silu" && got == "swish") {
			return nil, fmt.Errorf("convert: %s: hidden_act %q: this graph's feed-forward is %s-gated",
				c.path, got, act)
		}
		cfg, err := llamaHFConfig(arch, c, names)
		if err != nil {
			return nil, err
		}
		cfg.Flags |= flags
		return cfg, nil
	}
}

func llamaHFConfig(arch jlm.Arch, c *hfConfig, names []string) (*jlm.Config, error) {
	bad := func(f string, a ...any) error {
		return fmt.Errorf("convert: "+c.path+": "+f, a...)
	}
	if c.NumHiddenLayers == 0 || c.HiddenSize == 0 || c.NumAttentionHeads == 0 || c.IntermediateSize == 0 {
		return nil, bad("missing num_hidden_layers/hidden_size/num_attention_heads/intermediate_size")
	}
	// Required rather than derived from the embedding: hfCheckShape checks
	// the two against each other.
	if c.VocabSize == 0 {
		return nil, bad("missing vocab_size")
	}
	// Every refusal below is a key that changes the graph and has no place in
	// this container; defaulting any of them produces a model that runs.
	if c.MLPBias {
		return nil, bad("mlp_bias is true, and this container has no feed-forward bias role")
	}
	if err := hfRopeScalingOK(c); err != nil {
		return nil, err
	}
	if c.PartialRotaryFactor != nil && *c.PartialRotaryFactor != 1 {
		return nil, bad("partial_rotary_factor %v is not implemented", *c.PartialRotaryFactor)
	}
	// use_sliding_window is the switch, not sliding_window: Qwen2 declares a
	// window with use_sliding_window false.
	if c.SlidingWindow != nil && *c.SlidingWindow != 0 &&
		(c.UseSlidingWindow == nil || *c.UseSlidingWindow) {
		return nil, bad("sliding_window %d is in use and is not implemented from "+
			"safetensors", *c.SlidingWindow)
	}
	cfg := &jlm.Config{
		Arch:       arch,
		NLayer:     c.NumHiddenLayers,
		NEmbd:      c.HiddenSize,
		NHead:      c.NumAttentionHeads,
		NFFN:       c.IntermediateSize,
		NCtx:       c.MaxPositionEmbeddings,
		RMSEps:     1e-5,
		RopeBase:   10000,
		EmbdScale:  1,
		AttnFactor: 1,
	}
	if cfg.NCtx == 0 {
		cfg.NCtx = 2048
	}
	if c.RMSNormEps != 0 {
		cfg.RMSEps = float32(c.RMSNormEps)
	}
	if c.RopeTheta != 0 {
		cfg.RopeBase = float32(c.RopeTheta)
	}
	// Absent num_key_value_heads means MHA, not zero heads -- the same default
	// configOf applies to attention.head_count_kv.
	cfg.NKVHead = cfg.NHead
	if c.NumKeyValueHeads != nil && *c.NumKeyValueHeads != 0 {
		cfg.NKVHead = *c.NumKeyValueHeads
	}
	cfg.HeadDim = cfg.NEmbd / cfg.NHead
	if c.HeadDim != nil && *c.HeadDim != 0 {
		cfg.HeadDim = *c.HeadDim
	}
	// llama rotates the whole head dimension.
	cfg.NRot = cfg.HeadDim
	cfg.NVocab = c.VocabSize

	// Tied embeddings are the absence of the head, as on the GGUF side, and
	// config.json must agree: untied with no lm_head is a model whose shards
	// did not all arrive, not a tied one.
	hasHead := false
	for _, n := range names {
		if n == "lm_head.weight" {
			hasHead = true
			break
		}
	}
	tied := c.TieWordEmbeddings != nil && *c.TieWordEmbeddings
	switch {
	case !hasHead && !tied:
		return nil, bad("no lm_head.weight and tie_word_embeddings is not true: " +
			"this model's output head is missing, not tied")
	case hasHead && tied:
		return nil, bad("tie_word_embeddings is true and lm_head.weight is present; " +
			"which one is the head is not a question this converter may guess at")
	case !hasHead:
		cfg.Flags |= jlm.FlagTiedEmbd
	}
	if err := hfMoEConfig(c, cfg); err != nil {
		return nil, err
	}
	// attention_bias is not read: the biases are tensors, and the name table
	// matches them when the file ships them, so the file decides.
	return cfg, nil
}

// hfMoEConfig reads a mixture's shape. Absent expert keys leave cfg dense.
//
// Being a mixture is read off the keys, as configOf reads expert_count:
// mixtral is llama's Arch with an expert count.
func hfMoEConfig(c *hfConfig, cfg *jlm.Config) error {
	n := max(c.NumExperts, c.NumLocalExperts)
	if n == 0 {
		return nil
	}
	// A model whose layers are not all mixtures is refused by name: a
	// container with NExpert set makes every block a mixture.
	step := uint32(1)
	if c.DecoderSparseStep != nil {
		step = *c.DecoderSparseStep
	}
	if step > 1 || len(c.MLPOnlyLayers) > 0 {
		return fmt.Errorf("convert: %s: dense and sparse layers in one model "+
			"(decoder_sparse_step %d, mlp_only_layers %v) is not implemented",
			c.path, step, c.MLPOnlyLayers)
	}
	if c.SharedExpertIntermediateSize != 0 {
		return fmt.Errorf("convert: %s: a shared expert (shared_expert_intermediate_size %d) "+
			"is not implemented from safetensors", c.path, c.SharedExpertIntermediateSize)
	}
	cfg.NExpert, cfg.NExpertUsed = n, c.NumExpertsPerTok
	// moe_intermediate_size where the class has one; olmoe's and mixtral's
	// experts are intermediate_size wide.
	cfg.NFFNExp = uint32(c.MoEIntermediateSize)
	if cfg.NFFNExp == 0 {
		cfg.NFFNExp = c.IntermediateSize
	}
	if cfg.NExpertUsed == 0 || cfg.NExpertUsed > n {
		return fmt.Errorf("convert: %s: num_experts_per_tok %d of %d experts",
			c.path, c.NumExpertsPerTok, n)
	}
	// Qwen3MoeConfig and OlmoeConfig default norm_topk_prob to false, so an
	// absent key skips the division.
	if c.NormTopKProb == nil || !*c.NormTopKProb {
		cfg.Flags |= jlm.FlagNoExpertNorm
	}
	return nil
}

// hfRopeScalingOK refuses every rope_scaling but the one that means "none".
//
// It is a refusal by name, not a pass-through: applying no scaling where the
// weights expect one (llama3's, yarn) is fine for a hundred tokens and wrong
// after.
func hfRopeScalingOK(c *hfConfig) error {
	if len(c.RopeScaling) == 0 || string(c.RopeScaling) == "null" {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(c.RopeScaling, &m); err != nil {
		return fmt.Errorf("convert: %s: rope_scaling does not parse: %w", c.path, err)
	}
	// rope_parameters carries the base beside the scaling; the base is not
	// scaling, and partial_rotary_factor is refused on its own.
	delete(m, "rope_theta")
	delete(m, "partial_rotary_factor")
	if len(m) == 0 {
		return nil
	}
	kind, _ := m["rope_type"].(string)
	if kind == "" {
		kind, _ = m["type"].(string)
	}
	if kind == "default" {
		return nil
	}
	if kind == "" {
		kind = "unnamed"
	}
	return fmt.Errorf("convert: %s: rope_scaling %q is not implemented; the rotary "+
		"table it asks for is not the one this container's RopeBase describes", c.path, kind)
}

// llamaHFFixup rewrites q and k out of HuggingFace's rotary layout.
//
// This is the one place safetensors bytes are not container bytes.
// HuggingFace's Llama rotates with rotate_half, pairing dimension i with
// i+head_dim/2; this container's llama graph (no FlagRopeNeox) pairs
// (2i, 2i+1). convert_hf_to_gguf.py permutes q and k the same way, so a model
// converted from either format is the same container, which is what makes the
// cross-format gate possible and keeps one rotary layout in the engine. Getting
// it wrong still looks fine on short prompts.
func llamaHFFixup(e *jlm.Tensor, data []byte, c *jlm.Config) error {
	var heads uint32
	switch e.Role {
	case jlm.RoleAttnQ, jlm.RoleAttnQBias:
		heads = c.NHead
	case jlm.RoleAttnK, jlm.RoleAttnKBias:
		heads = c.NKVHead
	default:
		e.Data = data
		return nil
	}
	rows := uint64(heads) * uint64(c.HeadDim)
	// Dims[NDim-1] is the slowest axis: the row count for a matrix, the
	// only axis for a bias.
	if e.Dims[e.NDim-1] != rows {
		return fmt.Errorf("convert: %s has %d rows and %d heads of %d want %d",
			e.Name, e.Dims[e.NDim-1], heads, c.HeadDim, rows)
	}
	if uint64(len(data))%rows != 0 {
		return fmt.Errorf("convert: %s is %d bytes over %d rows", e.Name, len(data), rows)
	}
	rowBytes := uint64(len(data)) / rows
	if c.HeadDim%2 != 0 {
		return fmt.Errorf("convert: head_dim %d is odd; the rotary pairs rows", c.HeadDim)
	}
	e.Data = permuteHeadPairs(data, int(heads), int(c.HeadDim), int(rowBytes))
	return nil
}

// gemmaHFFixup folds gemma's (1 + weight) RMSNorm into the weight.
//
// Every norm, including the final one, as convert_hf_to_gguf.py does. An f16
// norm comes out f32: adding one in half precision would round differently
// from the reference.
func gemmaHFFixup(e *jlm.Tensor, data []byte, c *jlm.Config) error {
	switch e.Role {
	case jlm.RoleAttnNorm, jlm.RoleFFNNorm, jlm.RoleOutputNorm:
	default:
		e.Data = data
		return nil
	}
	out := make([]byte, 0, len(data)*2)
	switch e.Type {
	case jlm.TypeF32:
		for i := 0; i+4 <= len(data); i += 4 {
			v := math.Float32frombits(binary.LittleEndian.Uint32(data[i:])) + 1
			out = binary.LittleEndian.AppendUint32(out, math.Float32bits(v))
		}
	case jlm.TypeF16:
		for i := 0; i+2 <= len(data); i += 2 {
			v := float32(quant.DecodeHalf(binary.LittleEndian.Uint16(data[i:]))) + 1
			out = binary.LittleEndian.AppendUint32(out, math.Float32bits(v))
		}
		e.Type = jlm.TypeF32
	default:
		return fmt.Errorf("convert: %s: a %v norm", e.Name, e.Type)
	}
	e.Data = out
	return nil
}

// permuteHeadPairs moves row (h, j*hd/2 + i) to row (h, 2i + j), per head.
//
// It is a permutation of whole rows only, independent of the element type,
// and separate so a gate can call it.
func permuteHeadPairs(src []byte, heads, headDim, rowBytes int) []byte {
	dst := make([]byte, len(src))
	half := headDim / 2
	for h := 0; h < heads; h++ {
		base := h * headDim
		for j := 0; j < 2; j++ {
			for i := 0; i < half; i++ {
				s := (base + j*half + i) * rowBytes
				d := (base + 2*i + j) * rowBytes
				copy(dst[d:d+rowBytes], src[s:s+rowBytes])
			}
		}
	}
	return dst
}

// hardcodesSigmoid reports whether this checkpoint's own class gates with a
// sigmoid whatever config.json says, which decides the meaning of an absent
// scoring_func. It is a list of classes, not a default, because the
// references disagree (RULE 7m): DeepseekV2ForCausalLM softmaxes, the classes
// below sigmoid unconditionally. A class not named here keeps softmax.
func hardcodesSigmoid(archs []string) bool {
	if moe2HardcodesSigmoid(archs) {
		return true
	}
	for _, a := range archs {
		switch a {
		case "DeepseekV3ForCausalLM", "Glm4MoeLiteForCausalLM", "KimiLinearForCausalLM", "Glm4MoeForCausalLM":
			return true
		}
	}
	return false
}

// Kimi-Linear: ArchDeepseek2's attention and mixture, with Kimi Delta Attention
// in the layers layer_types calls "linear_attention". self_attn.q_proj and
// o_proj mean different things in linear and full layers, hence blockLinear.
var kimiLinearHF = &hfArch{
	arch:        jlm.ArchKimiLinear,
	blockPrefix: llamaHF.blockPrefix,
	model:       llamaHF.model,
	block: map[string]jlm.Role{
		"input_layernorm.weight":          jlm.RoleAttnNorm,
		"post_attention_layernorm.weight": jlm.RoleFFNNorm,

		// MLA. q_lora_rank is null on shipped checkpoints; the two-step names
		// are carried anyway since deepseekQueryForm decides from the weights.
		"self_attn.q_proj.weight":             jlm.RoleAttnQ,
		"self_attn.q_a_proj.weight":           jlm.RoleAttnQA,
		"self_attn.q_a_layernorm.weight":      jlm.RoleAttnQANorm,
		"self_attn.q_b_proj.weight":           jlm.RoleAttnQB,
		"self_attn.kv_a_proj_with_mqa.weight": jlm.RoleAttnKVA,
		"self_attn.kv_a_layernorm.weight":     jlm.RoleAttnKVANorm,
		"self_attn.kv_b_proj.weight":          jlm.RoleAttnKB,
		"self_attn.k_b_proj.weight":           jlm.RoleAttnKB,
		"self_attn.v_b_proj.weight":           jlm.RoleAttnVB,
		"self_attn.o_proj.weight":             jlm.RoleAttnOut,

		// The mixture, under block_sparse_moe. A dense layer is mlp.* in
		// Moonshot's checkpoint and block_sparse_moe.* when transformers saved it.
		"mlp.gate_proj.weight":                             jlm.RoleFFNGate,
		"mlp.up_proj.weight":                               jlm.RoleFFNUp,
		"mlp.down_proj.weight":                             jlm.RoleFFNDown,
		"block_sparse_moe.gate_proj.weight":                jlm.RoleFFNGate,
		"block_sparse_moe.up_proj.weight":                  jlm.RoleFFNUp,
		"block_sparse_moe.down_proj.weight":                jlm.RoleFFNDown,
		"block_sparse_moe.gate.weight":                     jlm.RoleRouter,
		"block_sparse_moe.gate.e_score_correction_bias":    jlm.RoleExpProbsB,
		"block_sparse_moe.shared_experts.gate_proj.weight": jlm.RoleShExpGate,
		"block_sparse_moe.shared_experts.up_proj.weight":   jlm.RoleShExpUp,
		"block_sparse_moe.shared_experts.down_proj.weight": jlm.RoleShExpDown,
	},
	blockLinear: map[string]jlm.Role{
		"input_layernorm.weight":          jlm.RoleAttnNorm,
		"post_attention_layernorm.weight": jlm.RoleFFNNorm,

		// q, k and v are three matrices here and one mixed projection
		// (qkBytes|qkBytes|Inner) in the container; fuseOrder joins them in the
		// geometry's order.
		"self_attn.q_proj.weight": jlm.RoleAttnQKV,
		"self_attn.k_proj.weight": jlm.RoleAttnQKV,
		"self_attn.v_proj.weight": jlm.RoleAttnQKV,
		// Three causal convolutions, likewise joined into one filter bank.
		"self_attn.q_conv1d.weight": jlm.RoleSSMConv1d,
		"self_attn.k_conv1d.weight": jlm.RoleSSMConv1d,
		"self_attn.v_conv1d.weight": jlm.RoleSSMConv1d,

		// b_proj is all of RoleSSMBA here: KDA's alpha comes from the low-rank
		// forget gate, so this is beta alone, where qwen3next interleaves both.
		"self_attn.b_proj.weight": jlm.RoleSSMBA,

		"self_attn.f_a_proj.weight": jlm.RoleSSMFA,
		"self_attn.f_b_proj.weight": jlm.RoleSSMFB,
		"self_attn.g_a_proj.weight": jlm.RoleSSMGA,
		"self_attn.g_b_proj.weight": jlm.RoleSSMGB,

		"self_attn.A_log":         jlm.RoleSSMA,
		"self_attn.dt_bias":       jlm.RoleSSMDtBias,
		"self_attn.o_norm.weight": jlm.RoleSSMNorm,
		"self_attn.o_proj.weight": jlm.RoleSSMOut,
	},
	fuseOrder: map[jlm.Role][]string{
		jlm.RoleAttnQKV:   {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"},
		jlm.RoleSSMConv1d: {"self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight"},
	},
	expertPrefix: "block_sparse_moe.experts.",
	expert: map[string]jlm.Role{
		"w1.weight": jlm.RoleExpGateBank,
		"w3.weight": jlm.RoleExpUpBank,
		"w2.weight": jlm.RoleExpDownBank,
	},
	ignore: llamaHF.ignore,
	split: map[string]hfSplit{
		"self_attn.kv_b_proj.weight": deepseekSplitKVB,
		"self_attn.q_conv1d.weight":  kimiSqueezeConv1d,
		"self_attn.k_conv1d.weight":  kimiSqueezeConv1d,
		"self_attn.v_conv1d.weight":  kimiSqueezeConv1d,
		"self_attn.A_log":            kimiExpandALog,
	},
	config: kimiLinearHFConfig,
}

// kimiSqueezeConv1d drops a causal convolution's middle axis.
//
// torch ships [channels, 1, taps] (a grouped Conv1d, one filter per
// channel), which reverses to ne = [taps, 1, channels]; the container's
// RoleSSMConv1d is [kernel, channels]. No byte moves. Left 3-D, concatRows
// would join ne1 (1 for each) and produce three channels, not 3*channels.
func kimiSqueezeConv1d(e jlm.Tensor, data []byte, c *jlm.Config) ([]jlm.Tensor, error) {
	if e.NDim != 3 || e.Dims[1] != 1 {
		return nil, fmt.Errorf("convert: %s: a causal convolution is [channels, 1, taps]; "+
			"this one has ne %v over %d dimension(s)", e.Name, e.Dims, e.NDim)
	}
	e.NDim, e.Dims = 2, [4]uint64{e.Dims[0], e.Dims[2], 0, 0}
	e.Data = data
	return []jlm.Tensor{e}, nil
}

// kimiExpandALog turns KDA's per-head log decay into the per-channel rate the
// gate kernel reads. The reference computes -exp(A_log) * softplus(g) and the
// gate kernel exp(a * softplus(z)), so a is -exp(A_log), which is also what
// llama.cpp writes into ssm_a for qwen3next. KDA's decay varies per channel
// while A_log is per head, so each head's rate is repeated linear_head_dim
// times here rather than broadcast every token.
//
// The source is [1, 1, heads, 1], which reverses to ne = [1, heads, 1, 1].
func kimiExpandALog(e jlm.Tensor, data []byte, c *jlm.Config) ([]jlm.Tensor, error) {
	heads, kDim := int(c.SSM.NHeadV), int(c.SSM.StateSize)
	if e.Type != jlm.TypeF32 {
		return nil, fmt.Errorf("convert: %s: A_log is %v; the decay is derived from it at "+
			"conversion and that arithmetic is float32", e.Name, e.Type)
	}
	n := len(data) / 4
	// Kimi-K3's release pads A_log to 128 entries for its 96 heads, the
	// padding zero; llama.cpp's converter takes the first n_head. A padding
	// that is not zero is a tensor of another geometry, and refused.
	if n > heads {
		for i := heads * 4; i < len(data); i++ {
			if data[i] != 0 {
				return nil, fmt.Errorf("convert: %s: %d element(s) for %d linear head(s), and the "+
					"ones past the heads are not zero padding", e.Name, n, heads)
			}
		}
		n, data = heads, data[:heads*4]
	}
	if n != heads {
		return nil, fmt.Errorf("convert: %s: %d element(s) for %d linear head(s)", e.Name, n, heads)
	}
	out := make([]byte, 0, heads*kDim*4)
	for h := 0; h < heads; h++ {
		a := math.Float32frombits(binary.LittleEndian.Uint32(data[h*4:]))
		rate := float32(-math.Exp(float64(a)))
		var w [4]byte
		binary.LittleEndian.PutUint32(w[:], math.Float32bits(rate))
		for i := 0; i < kDim; i++ {
			out = append(out, w[:]...)
		}
	}
	e.NDim, e.Dims, e.Data = 1, [4]uint64{uint64(heads * kDim), 0, 0, 0}, out
	return []jlm.Tensor{e}, nil
}

// kimiNormalize translates Moonshot's configuration_kimi.py form, which the
// published checkpoint uses, into the flat transformers form the rest of
// kimiLinearHFConfig reads (the synth fixture is the transformers form).
// Moonshot nests the recurrence in linear_attn_config, lists recurrent layers
// as kda_layers, and states the mixture by first_k_dense_replace and
// moe_layer_freq. The rules are modeling_kimi.py's:
//
//	is_kda_layer(i)   (i+1) in kda_layers       -- the lists are one-indexed
//	a mixture at i    i >= first_k_dense_replace and i % moe_layer_freq == 0
//
// A file that carries both forms and disagrees is refused.
func kimiNormalize(c *hfConfig) error {
	if c.MLAUseNope != nil && !*c.MLAUseNope {
		return fmt.Errorf("mla_use_nope is false, and this graph runs Kimi's latent " +
			"attention with no positional encoding")
	}
	if f := c.MoERouterActFunc; f != "" && f != "sigmoid" {
		return fmt.Errorf("moe_router_activation_func %q; KimiLinear routes with a sigmoid", f)
	}
	n := c.NumHiddenLayers
	if la := c.LinearAttnConfig; la != nil {
		for _, kv := range []struct {
			name string
			dst  *uint32
			v    uint32
		}{
			{"num_heads", &c.LinearNumHeads, la.NumHeads},
			{"head_dim", &c.LinearHeadDim, la.HeadDim},
			{"short_conv_kernel_size", &c.LinearConvKernelDim, la.ShortConv},
		} {
			if *kv.dst != 0 && *kv.dst != kv.v {
				return fmt.Errorf("linear_attn_config.%s is %d and the flat key says %d",
					kv.name, kv.v, *kv.dst)
			}
			*kv.dst = kv.v
		}
		types := make([]string, n)
		seen := make([]bool, n)
		for _, set := range []struct {
			list []uint32
			kind string
		}{{la.KDALayers, "linear_attention"}, {la.FullAttnLayers, "full_attention"}} {
			for _, l := range set.list {
				if l < 1 || l > n || seen[l-1] {
					return fmt.Errorf("linear_attn_config names layer %d (one-indexed) twice "+
						"or outside a %d-layer model", l, n)
				}
				seen[l-1], types[l-1] = true, set.kind
			}
		}
		for i, ok := range seen {
			if !ok {
				return fmt.Errorf("linear_attn_config names layer %d (one-indexed) in neither "+
					"kda_layers nor full_attn_layers", i+1)
			}
		}
		if c.LayerTypes != nil && !slices.Equal(c.LayerTypes, types) {
			return fmt.Errorf("layer_types %v and linear_attn_config %v disagree", c.LayerTypes, types)
		}
		c.LayerTypes = types
	}
	if c.MLPLayerTypes == nil && c.MoELayerFreq != nil {
		freq := *c.MoELayerFreq
		if freq == 0 {
			return fmt.Errorf("moe_layer_freq is 0")
		}
		c.MLPLayerTypes = make([]string, n)
		for i := uint32(0); i < n; i++ {
			c.MLPLayerTypes[i] = "dense"
			if i >= c.FirstKDenseReplace && i%freq == 0 {
				c.MLPLayerTypes[i] = "sparse"
			}
		}
	}
	return nil
}

// kimiLinearHFConfig reads Kimi-Linear's config.json: ArchDeepseek2's MLA and
// V3 router, plus the recurrence's geometry and the per-layer kind map. It
// first normalises the re-spelled mixture keys (see hfConfig); a checkpoint
// carrying two spellings with different values is refused, since there is no
// way to know which its weights were trained under (RULE 7m).
func kimiLinearHFConfig(c *hfConfig, names []string) (*jlm.Config, error) {
	bad := func(f string, a ...any) error {
		return fmt.Errorf("convert: "+c.path+": "+f, a...)
	}
	if err := kimiNormalize(c); err != nil {
		return nil, bad("%v", err)
	}
	pick := func(what string, canon *uint32, alt uint32, canonName, altName string) error {
		switch {
		case *canon != 0 && alt != 0 && *canon != alt:
			return bad("%s is %d under %q and %d under %q", what, *canon, canonName, alt, altName)
		case *canon == 0:
			*canon = alt
		}
		return nil
	}
	for _, p := range []struct {
		what               string
		canon              *uint32
		alt                uint32
		canonName, altName string
	}{
		{"the routed expert count", &c.NRoutedExperts, c.NumExperts, "n_routed_experts", "num_experts"},
		// KimiLinearTopkRouter reads num_local_experts, which
		// KimiLinearConfig aliases from num_experts, so either may appear.
		{"the routed expert count", &c.NRoutedExperts, c.NumLocalExperts, "n_routed_experts", "num_local_experts"},
		{"the selected expert count", &c.NumExpertsPerTok, c.NumExpertsPerToken, "num_experts_per_tok", "num_experts_per_token"},
		{"the shared expert count", &c.NSharedExperts, c.NumSharedExperts, "n_shared_experts", "num_shared_experts"},
		{"the expert group count", &c.NGroup, c.NumExpertGroup, "n_group", "num_expert_group"},
	} {
		if err := pick(p.what, p.canon, p.alt, p.canonName, p.altName); err != nil {
			return nil, err
		}
	}
	if c.NormTopKProb == nil {
		c.NormTopKProb = c.MoERenormalize
	} else if c.MoERenormalize != nil && *c.NormTopKProb != *c.MoERenormalize {
		return nil, bad("renormalisation is %v under \"norm_topk_prob\" and %v under \"moe_renormalize\"",
			*c.NormTopKProb, *c.MoERenormalize)
	}

	// The leading dense run is derived from mlp_layer_types. The container
	// states it as a length (first_k_dense_replace), so a dense block after a
	// sparse one cannot be expressed and is refused rather than approximated.
	if n := len(c.MLPLayerTypes); n != 0 {
		if uint32(n) != c.NumHiddenLayers {
			return nil, bad("mlp_layer_types has %d entries in a %d-block model", n, c.NumHiddenLayers)
		}
		lead := uint32(0)
		for _, k := range c.MLPLayerTypes {
			if k != "dense" {
				break
			}
			lead++
		}
		for i := int(lead); i < n; i++ {
			switch c.MLPLayerTypes[i] {
			case "sparse":
			case "dense":
				return nil, bad("mlp_layer_types has a dense block at %d after a sparse one; "+
					"this container says the dense run with a LENGTH and cannot express that: %w",
					i, ErrNotImplemented)
			default:
				return nil, bad("mlp_layer_types[%d] is %q, which is neither \"dense\" nor \"sparse\"",
					i, c.MLPLayerTypes[i])
			}
		}
		if c.FirstKDenseReplace != 0 && c.FirstKDenseReplace != lead {
			return nil, bad("first_k_dense_replace is %d and mlp_layer_types has %d leading dense block(s)",
				c.FirstKDenseReplace, lead)
		}
		c.FirstKDenseReplace = lead
	}

	cfg, err := deepseekHFConfigFFN(c, names, "block_sparse_moe.")
	if err != nil {
		return nil, err
	}
	cfg.Arch = jlm.ArchKimiLinear
	// The full blocks are MLA with NoPE: KimiLinearAttention never applies the
	// rotary and concatenates the qk_rope_head_dim channels into the key as is.
	cfg.Flags |= jlm.FlagNoPosEnc
	// Its latent norm is KimiLinearRMSNorm at the class default, 1e-6, not
	// rms_norm_eps (RULE 7m: transformers' class, as Kimi-K3's).
	cfg.LatentNormEps = k3LatentNormEps

	// The recurrence. Groups and NHeadV are both linear_num_heads: KDA has no
	// grouped-query sharing on the linear side, so rep is 1.
	if c.LinearNumHeads == 0 || c.LinearHeadDim == 0 || c.LinearConvKernelDim == 0 {
		return nil, bad("linear_num_heads=%d linear_head_dim=%d linear_conv_kernel_dim=%d: "+
			"all three describe the recurrent block's geometry",
			c.LinearNumHeads, c.LinearHeadDim, c.LinearConvKernelDim)
	}
	if c.LinearConvKernelDim < 2 {
		return nil, bad("linear_conv_kernel_dim %d: a causal convolution needs at least two taps",
			c.LinearConvKernelDim)
	}
	cfg.SSM = jlm.SSMConfig{
		ConvKernel: c.LinearConvKernelDim,
		Groups:     c.LinearNumHeads,
		NHeadV:     c.LinearNumHeads,
		StateSize:  c.LinearHeadDim,
		Inner:      c.LinearNumHeads * c.LinearHeadDim,
	}

	// layer_types is required, not defaulted: a hybrid with a wrong kind map
	// converts fluently wrong.
	if uint32(len(c.LayerTypes)) != cfg.NLayer {
		return nil, bad("layer_types has %d entries in a %d-block model", len(c.LayerTypes), cfg.NLayer)
	}
	cfg.LayerKinds = make([]jlm.LayerKind, cfg.NLayer)
	var nLinear int
	for i, k := range c.LayerTypes {
		switch k {
		case "linear_attention":
			cfg.LayerKinds[i] = jlm.LayerLinearAttn
			nLinear++
		case "full_attention":
			cfg.LayerKinds[i] = jlm.LayerFullAttn
		default:
			return nil, bad("layer_types[%d] is %q, which is neither \"linear_attention\" nor "+
				"\"full_attention\"", i, k)
		}
	}
	if nLinear == 0 {
		return nil, bad("layer_types names no linear_attention block, so this is not a hybrid")
	}
	return cfg, nil
}
