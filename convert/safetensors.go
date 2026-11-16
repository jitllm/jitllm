package convert

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
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
	var o options
	for _, fn := range opts {
		fn(&o)
	}
	s, err := hfSourceWith(hfDir{meta, name}, open, o.modelJSON, nil)
	if err != nil {
		return nil, err
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
	stateToolForm(s)
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
	q8           bool               // WithQ8, safetensors only
	chatTemplate string             // WithChatTemplate
	head         string             // WithHead, GGUF only
	modelConfig  string             // WithModelConfig, GGUF only
	pictureStep  func(*jlm.PreStep) // WithPictureStep, GGUF only
	longestEdge  int32              // the processor config's size.longest_edge, 0 for none
	modelJSON    string             // WithModelJSON, safetensors only
	taken        map[string]bool    // the features a conversion takes (FeaturesOf)
}

// WithPictureStep edits each picture step the tower's stater writes, before
// the converter states the picture's geometry from them and lowers them:
// a processor's step restated, as a gate restates one to see the geometry
// follow it.
func WithPictureStep(edit func(*jlm.PreStep)) Option {
	return func(o *options) { o.pictureStep = edit }
}

// WithModelConfig reads what a GGUF dropped from the publisher's config.json
// (path is the file or its model directory): a text model's LongRoPE
// factors, which llama.cpp's converter writes for no llama-arch model --
// MiniCPM-V 4's run from 1.0 to 31.0 across the rotary pairs, and its GGUF
// turns q and k with plain rope without them. The publisher's statement is
// the authority (RULE 7m), as WithChatTemplate's template is; a GGUF that
// carries its factors keeps them.
func WithModelConfig(path string) Option { return func(o *options) { o.modelConfig = path } }

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
	return hfSourceWith(dir, shards, "", nil)
}

// hfSourceWith is hfSourceOf by the caller's model.json, where there is one,
// each feature its blocks take added to taken where that is not nil.
func hfSourceWith(dir hfDir, shards []*safetensors.File, modelJSON string, taken map[string]bool) (*jlm.Source, error) {
	hc, err := readHFConfig(dir)
	if err != nil {
		return nil, err
	}
	ha, err := hfArchFor(dir, hc, modelJSON)
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

	hc.shapes = make(map[string][]uint64, len(all))
	for n, loc := range all {
		hc.shapes[n] = loc.t.Shape
	}
	cfg, err := ha.config(hc, names)
	if err != nil {
		return nil, err
	}
	if err := localAxes(cfg); err != nil {
		return nil, err
	}
	s := &jlm.Source{Config: cfg.Config}

	voc, err := hfVocabOf(dir, cfg.NVocab)
	if err != nil {
		return nil, err
	}
	if voc.Stop, err = hfStops(dir, voc); err != nil {
		return nil, err
	}
	if len(voc.Templates) == 0 && ha.template != "" {
		voc.Templates = []jlm.ChatTemplate{{Body: ha.template}}
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
		if ha.shape != nil && rt == nil && !loc.mx() {
			if err := ha.shape(&e); err != nil {
				return nil, err
			}
		}
		parts := []jlm.Tensor{e}
		sp := ha.split[ha.blockRest(name)]
		// A split reads elements, which a block of MXFP4 is not; only a plain
		// matrix, a bank member or a row concatenation may arrive packed.
		if loc.mx() && sp != nil {
			return nil, fmt.Errorf("convert: %s is MXFP4 and is cut at conversion: %w", name, ErrNotImplemented)
		}
		// A plain matrix is read when the writer reaches it; everything else is
		// read immediately, because a split sizes its parts from the data and a
		// fixup may change a vector's type. Those are small; the matrices are
		// what make a model tens of GB.
		lazy := sp == nil && e.NDim >= 2
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
	if len(s.Tensors) == 0 {
		return nil, fmt.Errorf("convert: %s: no tensor in this model has a role", dir)
	}
	if err := StateActivations(cfg, s); err != nil {
		return nil, err
	}
	g := ha.graph
	if g == nil {
		var err error
		if g, err = classGraph(ha); err != nil {
			return nil, err
		}
	}
	if err := stateGraphStreams(cfg.Config, s, g, hfKeys(hc)); err != nil {
		return nil, err
	}
	storeRowTables(cfg.Config, s)
	if err := StateSelection(cfg.Config, sourceStores(s)); err != nil {
		return nil, err
	}
	if err := expertsFit(cfg, s); err != nil {
		return nil, err
	}
	if err := valueWidthFits(cfg); err != nil {
		return nil, err
	}
	if err := geometryFits(cfg); err != nil {
		return nil, err
	}
	if err := ropeFits(cfg); err != nil {
		return nil, err
	}
	StateDelta(cfg.Config, sourceStores(s))
	if err := StateRevision(cfg); err != nil {
		return nil, err
	}
	if g != nil {
		if err := StateBlockGraphs(cfg, s, g, hfKeys(hc), taken); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// lazyFixup is setData deferred to the writer: read the bytes, run the
// architecture's fixup, and hand them over. A fixup that changed a matrix's
// type or shape at that point would have been laid out wrong already, so it
// is refused rather than written.
func lazyFixup(ha *hfArch, e jlm.Tensor, read func() ([]byte, error), c *Config) func() ([]byte, error) {
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
func concatRows(fuses map[bankKey][]*jlm.Tensor, ha *hfArch, c *Config) ([]jlm.Tensor, error) {
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
func stackBanks(banks map[bankKey][]*jlm.Tensor, c *Config) ([]jlm.Tensor, error) {
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

// hfHeadDimV is the value head's width. jlm.Config stores zero where it equals
// HeadDim, so every reader has to ask rather than read the field.
func hfHeadDimV(c *Config) uint32 {
	if c.HeadDimV != 0 {
		return c.HeadDimV
	}
	return c.HeadDim
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
// BF16 is widened to F32 on purpose: engine/device/cpu emits native matvecs for F32 and
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

// hfConfig is the subset of config.json the converter's Go reads; a class's
// statement reads the rest through keys.
type hfConfig struct {
	Architectures         []string        `json:"architectures"`
	ModelType             string          `json:"model_type"`
	NumKeyValueHeads      *uint32         `json:"num_key_value_heads"`
	HeadDim               *uint32         `json:"head_dim"`
	MaxPositionEmbeddings uint32          `json:"max_position_embeddings"`
	RMSNormEps            float64         `json:"rms_norm_eps"`
	RopeTheta             float64         `json:"rope_theta"`
	RopeScaling           json.RawMessage `json:"rope_scaling"`
	TieWordEmbeddings     *bool           `json:"tie_word_embeddings"`
	HiddenAct             string          `json:"hidden_act"`
	PartialRotaryFactor   *float64        `json:"partial_rotary_factor"`
	HiddenActivation      string          `json:"hidden_activation"`
	// transformers 5 carries rope_theta (and any scaling) here instead.
	RopeParameters json.RawMessage `json:"rope_parameters"`
	// extra holds tensors the config states and the checkpoint does not ship
	// (LongRoPE's factor vectors); the arch's config reader fills it.
	extra []jlm.Tensor

	// Multi-head Latent Attention (DeepSeek-V2/V3). See jlm.Config's MLA block.
	// q_lora_rank is a plain uint32: V2-Lite's "q_lora_rank": null means a
	// one-step query projection, which jlm.Config.QLoraRank spells as zero too.
	QLoraRank  uint32 `json:"q_lora_rank"`
	KVLoraRank uint32 `json:"kv_lora_rank"`

	// keys is every top-level key config.json carries. An absent key takes
	// its CLASS's default, which differs per class (SeedOssConfig's head_dim
	// is 128, SmolLM3Config's num_key_value_heads 4), and a pointer field
	// cannot tell absent from null.
	keys map[string]json.RawMessage

	path string // for error messages

	// shapes is each checkpoint tensor's shape in the format's order, for a
	// model.json's config, which hands configOf the tensor table it reads.
	shapes map[string][]uint64
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
	// name is the GGUF architecture the class converts as, whose library
	// file states its graph and container, or a Go-table class's own name
	// (classGraph).
	name string
	// template is the class's chat template where its release states none:
	// one that renders its chat in code ships no Jinja.
	template string
	// graph is the blocks' graph the class's model.json states, nil where
	// the engine builds it.
	graph *mjGraph
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
	config    func(*hfConfig, []string) (*Config, error)
	// gather folds source tensors whose VALUES the converter needs into
	// tensors of its own before anything is written (Apertus's four xIELU
	// scalars a block become one RoleXIELU vector). It reads through read and
	// returns what it made and the source names it consumed, which the
	// identify loop then skips. Optional.
	gather func(c *Config, names []string, read func(name string) ([]float32, error)) ([]jlm.Tensor, map[string]bool, error)
	// fixup rewrites a tensor's bytes on the way in. Optional: an architecture
	// whose weights are already in the container's layout leaves it nil and
	// setData carries the bytes through.
	fixup func(*jlm.Tensor, []byte, *Config) error
	// shape states the type and shape fixup gives a tensor, before the tensor
	// is laid out, so a transform that changes them still reads its source
	// when the writer reaches it. Optional.
	shape func(*jlm.Tensor) error
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
type hfSplit func(jlm.Tensor, []byte, *Config) ([]jlm.Tensor, error)

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
func (a *hfArch) setData(e *jlm.Tensor, data []byte, c *Config) error {
	if a.fixup == nil {
		e.Data = data
		return nil
	}
	return a.fixup(e, data, c)
}

// classRefusal refuses a class the library does not state.
func classRefusal(c *hfConfig) error {
	// architectures is the authority and model_type is not a fallback: one
	// model_type ("llama") covers several classes, and only the causal LM one
	// has this graph.
	have := strings.Join(c.Architectures, ", ")
	if have == "" {
		have = "none, and model_type is " + strconv.Quote(c.ModelType)
	}
	return fmt.Errorf("convert: %s: architectures %s is not implemented from "+
		"safetensors (implemented: %s)", c.path, have, strings.Join(HFClasses(), ", "))
}

// HFClasses is every Hugging Face class (config.json's architectures) the
// safetensors path converts -- the model.json library's, which is the list
// -- sorted. A library that does not parse adds nothing
// here and refuses every conversion; TestModelLibraryParses holds it.
func HFClasses() []string {
	var names []string
	if lib, err := modelLibrary(); err == nil {
		for n := range lib {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	return names
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
func hfCheckShape(e *jlm.Tensor, c *Config) error {
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
		// An output gate rides in the query projection, each head's gate after its query.
		if c.Flags.Has(jlm.FlagAttnOutGate) {
			return want(embd, 2*qrows)
		}
		return want(embd, qrows)
	case jlm.RoleAttnK:
		return want(embd, kvrows)
	case jlm.RoleAttnV:
		return want(embd, uint64(c.NKVHead)*uint64(hfHeadDimV(c)))
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

// hfYarn is a YaRN rope_scaling as config.json spells it.
type hfYarn struct {
	Factor       float64  `json:"factor"`
	OrigCtx      uint32   `json:"original_max_position_embeddings"`
	BetaFast     *float64 `json:"beta_fast"`
	BetaSlow     *float64 `json:"beta_slow"`
	MScale       *float64 `json:"mscale"`
	MScaleAllDim *float64 `json:"mscale_all_dim"`
	AttnFactor   *float64 `json:"attention_factor"`
	Truncate     *bool    `json:"truncate"`
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
// A class whose attention multiplies the softmax scale by a YaRN magnitude
// squared (DeepSeek's yarn_apply_mscale) states it as YarnLogMul: a model.json
// by its yarn_log_multiplier, hfYarnScoreScale for the Go reader that remains.
func hfYarnApply(y *hfYarn, cfg *Config) {
	if y == nil || y.Factor <= 1 {
		return
	}
	cfg.YarnFactor = float32(y.Factor)
	cfg.YarnOrigCtx = y.OrigCtx
	if cfg.YarnOrigCtx == 0 {
		cfg.YarnOrigCtx = cfg.NCtx
	}
	// Python's truthiness: an absent key and a zero are both false, so
	// `get("beta_fast") or 32` reads a zero as the default too.
	set := func(p *float64) bool { return p != nil && *p != 0 }
	cfg.YarnBetaFast, cfg.YarnBetaSlow = 32, 1
	if set(y.BetaFast) {
		cfg.YarnBetaFast = float32(*y.BetaFast)
	}
	if set(y.BetaSlow) {
		cfg.YarnBetaSlow = float32(*y.BetaSlow)
	}
	// truncate false takes the correction range as computed, unrounded.
	if y.Truncate != nil && !*y.Truncate {
		cfg.Flags |= jlm.FlagYarnExact
	}
	ms := func(k float64) float64 { return 0.1*k*math.Log(y.Factor) + 1 }
	switch {
	case y.AttnFactor != nil:
		cfg.AttnFactor *= float32(*y.AttnFactor)
	case set(y.MScale) && set(y.MScaleAllDim):
		cfg.AttnFactor *= float32(ms(*y.MScale) / ms(*y.MScaleAllDim))
	default:
		cfg.AttnFactor *= float32(ms(1))
	}
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
