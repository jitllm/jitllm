package main

import (
	"cmp"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/convert/hf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/localserver"
)

// convertCmd writes a GGUF (or a safetensors model) as a jlm container.
//
// The container stores weights in the layout the kernels read, so a page-in is
// a copy of bytes rather than a repack, which on a model larger than the card
// would otherwise dominate every token.
//
// The input may be a URL: `jitllm convert hf://owner/repo/x.gguf` downloads
// beside the models already here and converts; package hf carries the forms
// and the resume.
func convertCmd(args []string) error {
	fs := flag.NewFlagSet("convert", flag.ContinueOnError)
	outDir := fs.String("o", "", "directory the converted container (and a downloaded source) lands in")
	q8 := fs.Bool("q8", false, "safetensors only: store every weight matrix as Q8_0 "+
		"(lossy; the default carries the float bytes exactly, and bf16 widens to f32)")
	chatTpl := fs.String("chat-template", "", "store this file's chat template(s) in place of "+
		"the source's: a tokenizer_config.json, a chat_template.json, a .jinja file or a "+
		"model directory")
	head := fs.String("head", "", "a GGUF of the model's multi-token-prediction head, published "+
		"apart from it (unsloth's MTP/mtp-*.gguf): converted as the model's prediction block")
	plan := fs.Bool("plan", false, "safetensors only: read the shards' headers (by range, for a "+
		"repository) and print the container the conversion would write -- its size, tensor count "+
		"and anything it would refuse -- without converting")
	if err := fs.Parse(args); err != nil {
		return err
	}
	// Go's flag parser stops at the first positional, so `convert m.gguf -o
	// /tmp` would put "-o" where the mmproj goes.
	if err := misplacedFlag(fs); err != nil {
		return err
	}
	// The tower is an input, not a second output: one container holds the
	// text and vision blocks. See convert.FromGGUFs.
	src, mmproj, dst, err := convertArgs(libraryArgs(fs.Args()))
	if err != nil {
		return err
	}
	cl := hfClient()
	// A safetensors repository is converted where it lives: its shards are
	// read by range as the writer reaches each tensor and never stored. A GGUF
	// repository still downloads (openStreamed answers nil).
	var sm *streamed
	if r, remote, _ := hf.Parse(src); remote && r.File == "" && mmproj == "" {
		if sm, err = openStreamed(cl, r); err != nil {
			return err
		}
	}
	if *plan && sm == nil && !convert.IsSafetensors(src) {
		// A plan reads headers where the model lives; a GGUF repository would
		// download first.
		return fmt.Errorf("jitllm convert: -plan reads a safetensors model's headers, and %s is not one", src)
	}
	if sm != nil {
		src = sm.name
	} else if src, err = localise(cl, src, *outDir); err != nil {
		return err
	}
	if mmproj, err = localise(cl, mmproj, *outDir); err != nil {
		return err
	}
	// Which reader is decided here and nowhere else; both produce the same
	// jlm.Source. It is asked after the fetch: IsSafetensors stats the path,
	// and a remote reference is not a directory until it is on the disk.
	st := sm != nil || convert.IsSafetensors(src)
	// An mmproj is a GGUF-only contract, refused rather than ignored: a
	// HuggingFace vision model carries its tower in the same directory.
	if st && mmproj != "" {
		return fmt.Errorf("jitllm convert: %s is a safetensors model and %s is an "+
			"mmproj: the two-file contract is GGUF's, and a HuggingFace tower "+
			"lives in the model directory", src, mmproj)
	}
	// The default output is derived AFTER the fetch (see defaultDest).
	if dst == "" {
		if dst, err = defaultDest(src, *outDir); err != nil {
			return err
		}
	}
	if src == dst {
		return fmt.Errorf("jitllm convert: refusing to write %s over itself", src)
	}
	// The SOURCE bytes, which for a HuggingFace model are the shards rather
	// than the directory entry: a 4096-byte inode against a 1.4 GiB container
	// would print +36000000% and a rate of nothing.
	var srcBytes int64
	if sm != nil {
		srcBytes = sm.size()
	} else if srcBytes, err = sourceBytes(src); err != nil {
		return err
	}
	host, _ := os.Hostname()
	fp := jlm.Fingerprint{Host: host}

	if mmproj != "" {
		fmt.Fprintf(os.Stderr, "convert  %s + %s -> %s\n", src, mmproj, dst)
	} else {
		fmt.Fprintf(os.Stderr, "convert  %s -> %s\n", src, dst)
	}
	t0 := time.Now()
	var h *jlm.Header
	var opts []convert.Option
	if *chatTpl != "" {
		opts = append(opts, convert.WithChatTemplate(*chatTpl))
	}
	if *head != "" {
		opts = append(opts, convert.WithHead(*head))
	}
	if st {
		if *q8 {
			opts = append(opts, convert.WithQ8())
		}
		if *plan {
			return planConvert(sm, src, dst, *q8, opts)
		}
		if n, ok := jlm.Resumable(dst); ok {
			fmt.Fprintf(os.Stderr, "resume   %s.part's journal has %d tensor(s) written: the write "+
				"continues there if the layout, the source and this build match, and starts over if not\n", dst, n)
		}
		if sm != nil {
			stop := make(chan struct{})
			go sm.report(30*time.Second, stop)
			h, err = convert.FromShards(sm.meta, sm.name, sm.shards, dst, fp, opts...)
			close(stop)
		} else {
			h, err = convert.FromSafetensors(src, dst, fp, opts...)
		}
	} else {
		if *q8 {
			return fmt.Errorf("jitllm convert: -q8 quantizes a safetensors model; %s is a "+
				"GGUF, whose weights are already quantized", src)
		}
		h, err = convert.FromGGUFs(src, mmproj, dst, fp, opts...)
	}
	// Nothing is removed on failure: jlm.Write builds dst+".part" and renames
	// it over dst only when complete, so dst is still the previous container.
	// A safetensors write that journaled leaves the .part and its journal, and
	// the same command run again resumes from them.
	if err != nil {
		return err
	}
	di, err := os.Stat(dst)
	if err != nil {
		return err
	}
	el := time.Since(t0)
	gib := func(n int64) float64 { return float64(n) / (1 << 30) }
	fmt.Fprintf(os.Stderr, "         %.2f GiB -> %.2f GiB (%+.0f%%) in %s, %.2f GB/s\n",
		gib(srcBytes), gib(di.Size()),
		100*(float64(di.Size())/float64(srcBytes)-1), el.Round(time.Millisecond),
		float64(srcBytes)/el.Seconds()/1e9)
	fmt.Fprintf(os.Stderr, "         %d blocks of %.2f MiB, %d tensors; page i is at %d + i*%d\n",
		h.NBlocks, float64(h.PageSize)/(1<<20), h.NTensors, h.DataOff, h.PageSize)
	if h.NExpPages > 0 {
		fmt.Fprintf(os.Stderr, "         %d expert pages of %.2f MiB; expert page i is at %d + i*%d\n",
			h.NExpPages, float64(h.ExpPageSize)/(1<<20), h.ExpDataOff, h.ExpPageSize)
	}
	if h.NVisBlocks > 0 {
		fmt.Fprintf(os.Stderr, "         %d vision blocks of %.2f MiB; vision page i is at %d + i*%d\n",
			h.NVisBlocks, float64(h.VisPageSize)/(1<<20), h.VisDataOff, h.VisPageSize)
	}
	// Which build wrote it: a stale converter's container has a current
	// format version, so only this identifies it.
	fmt.Fprintf(os.Stderr, "         written by %s\n", jlm.WriterID())
	c, err := jlm.Open(dst)
	if err != nil {
		return fmt.Errorf("jitllm convert: wrote %s but cannot read it back: %w", dst, err)
	}
	defer c.Close()
	// Which chat template was stored and where it came from, read back from
	// the container: a GGUF's template is often older than its publisher's.
	from := "the source's own"
	if *chatTpl != "" {
		from = *chatTpl
	}
	if v := c.Vocab(); v != nil && len(v.Templates) > 0 {
		names := make([]string, len(v.Templates))
		for i, t := range v.Templates {
			names[i] = cmp.Or(t.Name, "(unnamed)")
		}
		fmt.Fprintf(os.Stderr, "         chat template from %s: %s\n", from, strings.Join(names, ", "))
	} else {
		fmt.Fprintf(os.Stderr, "         no chat template (a base model)\n")
	}
	// Padding is the design's only waste, so it is reported: it measures
	// block-size variance, not an error.
	if h.NBlocks > 0 || h.NVisBlocks > 0 {
		// SpanLen, not Span: Open reads no page, and Span answers nil for a
		// block that is out -- which made this report 100% on every model.
		var used uint64
		for i := range c.Entries() {
			e := &c.Entries()[i]
			// Expanded roles carry a block number but live in the dense
			// region (jlm.alwaysResident), so they are not page bytes.
			if e.Block == jlm.DenseBlock || e.Role.Expanded() {
				continue
			}
			qs, d, sc := c.SpanLen(e)
			used += qs + d + sc
		}
		// Expert pages are pages too: a bank's bytes are counted in `used` and
		// live in the third array.
		allPages := uint64(h.NBlocks)*h.PageSize + uint64(h.NVisBlocks)*h.VisPageSize +
			uint64(h.NExpPages)*h.ExpPageSize
		pad := allPages - min(used, allPages)
		fmt.Fprintf(os.Stderr, "         padding %.2f MiB of %.2f GiB in pages (%.1f%%), the block-size variance\n",
			float64(pad)/(1<<20), float64(allPages)/(1<<30),
			100*float64(pad)/float64(allPages))
	}
	aliasCatalogName(fs.Args(), dst)
	return nil
}

// aliasCatalogName gives a container converted from a catalog name that name
// as an alias in the models folder, so `jitllm convert qwen3-8b` is followed
// by `jitllm run qwen3-8b` rather than the file's own name. A name another
// container already answers to is left alone.
func aliasCatalogName(args []string, dst string) {
	if len(args) == 0 || libraryArgs(args)[0] == args[0] {
		return
	}
	dir := defaultModelDir()
	if filepath.Clean(filepath.Dir(dst)) != filepath.Clean(dir) {
		return
	}
	added, err := localserver.AddAlias(dir, filepath.Base(dst), args[0])
	switch {
	case err != nil:
		fmt.Fprintf(os.Stderr, "         alias %s not kept: %v\n", args[0], err)
	case added:
		fmt.Fprintf(os.Stderr, "alias    %s (jitllm run %s)\n", args[0], args[0])
	}
}

// splitPart is llama.cpp's gguf-split naming, NAME-00001-of-00003.gguf.
var splitPart = regexp.MustCompile(`^(.*)-(\d{5})-of-(\d{5})\.gguf$`)

// sourceBytes is how many bytes of model the input holds: the file's size for a
// GGUF or a single shard, and the sum of the weight files for a HuggingFace
// directory. The JSON beside them is a rounding error and is not counted.
func sourceBytes(src string) (int64, error) {
	fi, err := os.Stat(src)
	if err != nil {
		return 0, err
	}
	if !fi.IsDir() {
		// A split GGUF counts every part, not part 1 alone.
		if m := splitPart.FindStringSubmatch(src); m != nil && m[2] == "00001" {
			n, _ := strconv.Atoi(m[3])
			total := fi.Size()
			for i := 2; i <= n; i++ {
				p, err := os.Stat(fmt.Sprintf("%s-%05d-of-%s.gguf", m[1], i, m[3]))
				if err != nil {
					return 0, err
				}
				total += p.Size()
			}
			return total, nil
		}
		return fi.Size(), nil
	}
	shards, err := filepath.Glob(filepath.Join(src, "*.safetensors"))
	if err != nil {
		return 0, err
	}
	var n int64
	for _, p := range shards {
		if s, err := os.Stat(p); err == nil {
			n += s.Size()
		}
	}
	if n == 0 {
		return 0, fmt.Errorf("jitllm convert: %s holds no weight file", src)
	}
	return n, nil
}

// defaultDest is where a conversion with no output path writes: into -o's
// directory when one is given, whatever the source is (a local file as much as
// a download), otherwise into the models folder every front end and jitllmd
// read (defaultModelDir), created on first use, whatever disk the source is
// on.
func defaultDest(src, outDir string) (string, error) {
	dir, what := outDir, "-o "+outDir
	if dir == "" {
		dir = defaultModelDir()
		what = "the models folder " + dir
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("jitllm convert: %s: %w", what, err)
	}
	return convert.DestFor(src, dir), nil
}
