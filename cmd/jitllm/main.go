// Command jitllm is the jitllm CLI.
//
// One command serves CPU and GPU; device selection is a flag, not a build tag:
//
//	jitllm run -devices auto|all|cpu|gpu[:N]|cuda:0|vulkan:1[=BYTES],... [-vram BYTES] model.jlm prompt...
//
// jit/gpu is imported unconditionally. The drivers are dlopen'd rather than
// linked, which is what keeps the build free of cgo.
//
// The verify subcommand decodes the same prompt on both tiers, compares token
// ids, and runs the paired A/Bs.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/jitllm/jitllm/engine/sched"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/cmd/goheap"
	"github.com/jitllm/jitllm/internal/localserver"

	"os/exec"

	"github.com/jitllm/jitllm/dev/bench"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/jit/cpu"
	"github.com/jitllm/jitllm/jit/gpu/tier"
	"github.com/jitllm/jitllm/tok"
)

// alignGOMAXPROCS tells the Go runtime how wide this engine intends to be.
//
// sched.DecodeCores picks the physical P-cores and the pool starts one worker
// each, while GOMAXPROCS defaults to NumCPU. Several places size work off
// GOMAXPROCS (jlm's reader fan-out, jit/gpu/kernels/pack.go,
// jit/gpu/tier/tier.go), so the two must agree.
//
// It is set here rather than in sched because GOMAXPROCS is process-global and
// a library must not take that decision from whatever embeds it. An explicit
// GOMAXPROCS in the environment wins.
func alignGOMAXPROCS() {
	if os.Getenv("GOMAXPROCS") != "" {
		return
	}
	// The decode pool's width. A prefill pool wider than it (nn's
	// Config.PrefillCores) raises GOMAXPROCS for its own phase and puts it
	// back (nn.JIT.BeginPrefill).
	if n := len(sched.DecodeCores(schedOptions()...)); n > 0 {
		runtime.GOMAXPROCS(n)
	}
}

// numaNodes is the node list a model's weight memory is interleaved over: every
// node this process runs on, when there is more than one and nobody chose a
// policy already (numactl, a parent). JITLLM_NUMA=off leaves placement to the
// kernel's first touch. nil means no interleave.
func numaNodes() []int {
	if os.Getenv("JITLLM_NUMA") == "off" {
		return nil
	}
	if nodes := sched.NUMANodes(); len(nodes) > 1 && sched.MemPolicyDefault() {
		return nodes
	}
	return nil
}

// version is the release version, set at link time
// (-ldflags "-X main.version=v0.1.0"; see .goreleaser.yaml).
var version = "dev"

// procStart is as near process start as Go code runs: package variables are
// initialised before main. speed -ttft's load row counts from it, as
// llama.cpp's log timestamps count from its logger's start.
var procStart = time.Now()

func main() {
	alignGOMAXPROCS()
	// Page frames and the dense region live off the Go heap (jlm.SetOffHeap).
	// JITLLM_OFFHEAP=0 puts them back on it, with the budget's collector
	// reserve: the control arm for a measurement.
	if os.Getenv("JITLLM_OFFHEAP") == "0" {
		jlm.SetOffHeap(false)
		sched.SetWeightsOffHeap(false)
	}
	if os.Getenv("JITLLM_GCSTATS") == "1" {
		defer gcSummary()
	}
	// These subcommands take no model: they describe the machine, the build
	// or the catalogue rather than a file (doctor's container is optional).
	if len(os.Args) < 2 || (len(os.Args) < 3 && os.Args[1] != "hardware" && os.Args[1] != "library" && os.Args[1] != "version" &&
		os.Args[1] != "capabilities" && os.Args[1] != "doctor") {
		usage()
	}
	var err error
	switch os.Args[1] {
	case "version":
		fmt.Println(version)
	case "hardware":
		err = hardware()
	case "capabilities":
		err = capabilitiesCmd(os.Args[2:], os.Stdout)
	case "doctor":
		os.Exit(doctorCmd(os.Args[2:]))
	case "info":
		err = info(os.Args[2], len(os.Args) > 3 && os.Args[3] == "-kv")
	case "asm":
		err = dumpAsm(os.Args[2])
	case "bench":
		err = benchCmd(os.Args[2:])
	case "run":
		err = runCmd(os.Args[2:])
	case "batch":
		err = batchCmd(os.Args[2:])
	case "speed":
		err = speedCmd(os.Args[2:])
	case "verify":
		err = verifyCmd(os.Args[2:])
	case "tokenize":
		err = tokenizeCmd(os.Args[2:])
	case "convert":
		err = convertCmd(os.Args[2:])
	case "library":
		err = libraryCmd(os.Args[2:])
	case "embed":
		err = embedCmd(os.Args[2:])
	case "decide":
		err = decideCmd(os.Args[2:])
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `usage:
  jitllm version
  jitllm hardware
  jitllm capabilities [-json]   what this build converts, runs and serves (docs/capabilities.json)
  jitllm doctor [-json] [model.jlm]   can this machine and build run; exit 0 ok, 2 missing, 1 error
  jitllm info <model.gguf>
  jitllm tokenize [-tokenizer FILE] <model.jlm> <text...>
  jitllm run [flags] <model.jlm> <prompt...>
  jitllm embed [-ids] [-lines] <model.jlm> <text...>   an embedding model's normalised vector
  jitllm decide [-lines] <model.jlm> <request.json|->   a decision model's /v1/systemone answers
  jitllm bench [-tokenizer FILE] <model.jlm>
  jitllm speed [-devices SPEC] [-p N] [-n N] [-r N] <model.jlm>   llama-bench's pp/tg, warmed
  jitllm batch [-devices SPEC] [-batch N] [-n N] <model.jlm> <prompt...>
  jitllm verify [flags] <model.jlm>
  jitllm convert [-o DIR] [-chat-template FILE] <model.gguf|model-dir|URL|NAME> [mmproj.gguf|URL] [out.jlm]
  jitllm library [-refs]   the models convert fetches by NAME
  jitllm asm <Q4_0|Q8_0|Q4_K>[x4]

run flags:
  -devices SPEC          where to run the blocks (default auto). SPEC is a comma-separated
                         list of:
                           auto       one device, chosen by measuring each backend; a machine
                                      with no GPU is a CPU machine, not an error (the default)
                           all        every distinct device on this host, fastest first
                           cpu        no device; the host runs the whole model
                           gpu        every GPU, as all
                           gpu:N      the Nth of the GPUs jitllm hardware lists
                           cuda       every CUDA device, fastest first
                           cuda:ORD   one CUDA device by driver ordinal
                           vulkan     every Vulkan GPU, fastest first
                           vulkan:S   one Vulkan device by index or name substring
                           metal      the system default Metal device
                         a bare backend name is every device of it, and jitllm places the
                         blocks fastest first, using as many as the model needs; a
                         selector (:N) pins one. Any entry may carry =BYTES (3G, 512M,
                         800000000) to budget each device it names: -devices cuda:0=3G,vulkan:1=8G
  -gpu-layers N          cap the blocks offered to the device; -1 fits as many as the card holds
  -gpu-grow              start on the CPU and migrate blocks to the device while serving
  -vram BYTES            device weight budget; 0 (the default) asks each device what it has
                         free and keeps an eighth of it, at least 256 MiB, as headroom
                         a block that does not fit runs on the host unless -placement streams
                         it (N=DEV~): a page-in crosses the link at 3.19 GB/s where the host
                         reads DRAM at ~37, so streaming a decode pays only when the model does
                         not fit host memory either. JITLLM_ARENA=<bytes> bounds the
                         host-resident packed copy a page-in is a DMA from; unset, a streaming
                         placement sizes it itself.
  -maxmem BYTES          host weight residency budget (default: the cgroup limit
                         or MemAvailable, whichever is smaller, less headroom)
  -n N                   tokens to generate; 0 (the default) runs until the model ends its reply
  -depth N               pad the prompt to N tokens, so decode is measured at that context length
  -tune-seam             measure whether fewer device blocks are faster, and relocate
  -no-gpu-fallback       fail instead of finishing on the host when the device breaks
  -relocate              when the device cannot grow the context, hand its last block to
                         the host and keep the rest, rather than failing or demoting every
                         block. On by default wherever -devices admits the host (auto, or
                         a list with cpu); -relocate=false turns it off
  -image FILE            image to place in the prompt; the container must carry a
                         vision tower (jitllm convert model.gguf mmproj.gguf out.jlm)
  -tokenizer FILE        a HuggingFace tokenizer.json whose pre-tokenizer replaces the one
                         tokenizer.ggml.pre names -- the GGUF still supplies the vocabulary
  -kv-cache DIR          keep the prompt's KV cache in DIR and reuse it when a later prompt
                         shares a prefix, matched in 16-position chunks (a shared prefix
                         under 16 tokens misses). A hybrid model restores only a prompt it
                         saw whole, at its end
  -kv-cache-max BYTES    what that directory may occupy before the least recently used pages
                         are evicted (default 8 GiB; 0 is unbounded)
  -placement MAP         where blocks and the head run: 0=host,1-7=cuda:0,8=host!,
                         9-15=vulkan:1~,head=vulkan:1 -- a block, a range, head or * (every block
                         not named), then host or a device -devices names; a trailing ! pins it
                         (seam moves, relocation and a device budget shrink leave it alone) and a
                         trailing ~ streams it: its weights swap through the device's slots, so a
                         card smaller than the blocks it is given runs all of them, moving a
                         block's weights per token. Unnamed blocks are the engine's: resident on
                         the device while they fit, the host after
  -placement-strict      fail when an entry cannot be met, instead of placing that block elsewhere
  -kv-budget BYTES       KV history this session may hold in memory; older pages spill to
                         the -kv-cache directory and come back when attention reads them
                         (default 0: unbounded)

convert reads TWO source formats and writes one container. A GGUF is one file; a
safetensors model is a DIRECTORY, because config.json and tokenizer.json beside the
weights are where the rest of the model is, and sharded models come through
model.safetensors.index.json:
  jitllm convert models/SmolLM2-360M-Instruct out.jlm     a HuggingFace model directory
  jitllm convert model.safetensors                        a single-shard model
An mmproj is GGUF's two-file contract and is refused for a safetensors model, whose
vision tower lives in the same directory.

A GGUF's chat template is whatever its converter copied, and a publisher's own is
often newer (Mistral-7B-Instruct-v0.3's GGUF has no system role; Mistral's current
template does). -chat-template stores another in its place:
  -chat-template FILE    a tokenizer_config.json, a chat_template.json, a .jinja file or
                         a model directory; named templates (tool_use, rag) come along.
                         A file with no template, or one that does not compile, is refused
  jitllm convert -chat-template Mistral-7B-Instruct-v0.3/tokenizer_config.json \
      Mistral-7B-Instruct-v0.3-Q4_K_M.gguf

It also takes a HuggingFace reference, which is downloaded first:
  hf://OWNER/REPO/FILE.gguf            the short form; @REV pins a branch, tag or sha
  hf://OWNER/REPO                      the repo's only GGUF, or a refusal listing them
  https://huggingface.co/OWNER/REPO/resolve/REV/FILE.gguf   the download link
  https://huggingface.co/OWNER/REPO/blob/REV/FILE.gguf      the address bar
A download resumes, is checked against the Hub's sha256, and is skipped when the file
is already here and complete. It lands beside the models you already have -- see -o:
  -o DIR                 where a downloaded GGUF lands (JITLLM_MODELS sets the default)
  HF_TOKEN               a token for a private or gated repo; HUGGING_FACE_HUB_TOKEN
                         and ~/.cache/huggingface/token are read too

sampling (names match llama.cpp, so a prompt moves between engines):
  -temp F                sampling temperature; 0 = greedy (default)
  -top-k N               keep only the k most likely tokens; 0 = off
  -top-p F               nucleus sampling; 0 or 1 = off
  -min-p F               keep tokens above min-p * p(best); 0 = off
  -repeat-penalty F      penalise recently seen tokens; 1 = off
  -repeat-last-n N       how far back repeat-penalty looks (default 64)
  -seed N                RNG seed for sampling

verify flags:
  -devices SPEC          which device(s) to verify, in run's grammar (default auto)
  -n N                   tokens to compare (default 32)
  -vram BYTES            device weight budget; 0 asks the device
  -ab split|submit|head  run a paired A/B instead of verifying
  -verify                recompute every served matvec on the host`)
	os.Exit(2)
}

// dumpAsm disassembles a generated kernel, so the machine code jitllm emits can be
// read rather than taken on faith. Needs objdump; the bytes are real either way.
func dumpAsm(name string) error {
	rows := int8(1)
	if strings.HasSuffix(name, "x4") {
		rows, name = cpu.Interleave, strings.TrimSuffix(name, "x4")
	}
	var t quant.Type
	switch name {
	case "Q4_0":
		t = quant.Q4_0
	case "Q8_0":
		t = quant.Q8_0
	case "Q4_K":
		t = quant.Q4_K
	default:
		return fmt.Errorf("no kernel for %q (Q4_0, Q8_0, Q4_K, optionally +x4)", name)
	}
	// Print the kernel this host would run: the host tier's packed kernel,
	// which is what a container decodes through, not the GGUF row-major one.
	e := cpu.EmittersFor(cpu.HostTier())
	kind := "packed fused matvec (what a container decodes through)"
	emit := func() ([]byte, error) {
		if e.PackedSupported == nil || !e.PackedSupported(t) || e.PackedFused == nil {
			return nil, fmt.Errorf("the %s tier has no packed kernel for %s", cpu.HostTier(), t)
		}
		return e.PackedFused(t)
	}
	if rows == cpu.Interleave {
		kind = "GGUF row-major matvec, interleaved"
		emit = func() ([]byte, error) {
			if !e.RowMajorGGUF || e.RowMajorSupported == nil || !e.RowMajorSupported(t) {
				return nil, fmt.Errorf("the %s tier has no GGUF row-major kernel for %s "+
					"(a container does not use one)", cpu.HostTier(), t)
			}
			return e.RowMajor(cpu.Spec{W: t, Rows: rows, Accs: 1, Cols: 1})
		}
	}
	code, err := emit()
	if err != nil {
		return err
	}
	f, err := os.CreateTemp("", "jitllmasm")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if _, err := f.Write(code); err != nil {
		return err
	}
	f.Close()
	fmt.Printf("// %s %s, tier %s, %d bytes of x86-64\n", t, kind, cpu.HostTier(), len(code))
	out, err := exec.Command("objdump", "-D", "-b", "binary", "-m", "i386:x86-64",
		"-M", "intel", f.Name()).Output()
	if err != nil {
		fmt.Printf("// objdump unavailable; raw bytes:\n%x\n", code)
		return nil
	}
	fmt.Print(string(out))
	return nil
}

// benchmark measures one-core decode on the generated kernels as a fraction of
// a memory wall measured in this same process (AGENTS.md RULE 1).
func benchmark(path string, topts []tok.Option) error {
	if err := bench.Guard(os.Getenv("JITLLM_NO_GUARD") != ""); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n(set JITLLM_NO_GUARD=1 to measure anyway; the numbers will be wrong)\n", err)
		return err
	}
	m, err := model.Open(path, loadOpts(topts, 0)...)
	if err != nil {
		return err
	}
	defer m.Close()

	fmt.Printf("memory wall  1 thread %.1f GB/s   %d threads %.1f GB/s   (Go-side, scalar, unpinned)\n",
		bench.MemWall(1)/1e9, bench.Cores(), bench.MemWall(bench.Cores())/1e9)

	// Decode, one token at a time.
	ids := m.Vocab.Encode("The capital of France is", true)
	st := m.NewState(len(ids) + 8)
	defer st.Close()
	var logits []float32
	for _, id := range ids {
		if logits, err = st.Forward(id); err != nil {
			return err
		}
	}
	threads := st.Workers()
	tier := fmt.Sprintf("generated %v, %d shape-specialized", st.Accelerated(), st.Shapes())
	const nTok = 3
	start := time.Now()
	for i := 0; i < nTok; i++ {
		if logits, err = st.Forward(model.Greedy(logits)); err != nil {
			return err
		}
	}
	el := time.Since(start)

	perTok := m.BytesPerToken()
	fmt.Println(bench.Report(fmt.Sprintf("decode, %d core(s), %s", threads, tier), perTok*nTok, el, threads))
	fmt.Printf("             %.3f tok/s   %d B/token\n", nTok/el.Seconds(), perTok)
	if names, ns := model.OpProfile(); ns[0] > 0 {
		var tot int64
		for _, v := range ns {
			tot += v
		}
		fmt.Printf("             per-op:")
		for i, n := range names {
			if ns[i] > 0 {
				fmt.Printf("  %s %.1f%%", n, 100*float64(ns[i])/float64(tot))
			}
		}
		fmt.Println()
	}
	if q, kn, o := nn.ProfileNanos(); q+kn+o > 0 {
		tot := float64(q + kn + o)
		fmt.Printf("             serial %.0f%% (quantize %.0f%%, widen %.0f%%)  parallel %.0f%% (kernel)\n",
			100*float64(q+o)/tot, 100*float64(q)/tot, 100*float64(o)/tot, 100*float64(kn)/tot)
	}
	return nil
}

// chatSpec is -chat and -system together; nil is raw completion, which is the
// default and what every board row measures.
//
// repl is the interactive conversation `jitllm run model.jlm` holds when no
// prompt is given and stdin is a terminal (repl.go).
type chatSpec struct {
	system string
	repl   bool
}

func run(path, prompt string, n, depth int, devSpec string, gpuLayers int, vram, maxmem uint64, grow, tuneSeam, noFallback, relocate bool, sm *model.Sampler, imgPath string, topts []tok.Option, chat *chatSpec, kvCacheDir string, kvCacheMaxBytes, kvBudget uint64, place *model.Placement, specOpts []model.SpecOption) error {
	if v := os.Getenv("JITLLM_N"); v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			n = p
		}
	}
	// The budget is resolved once, before anything is allocated.
	// sched.MemBudget reads MemAvailable, so asking again after the weights are
	// resident counts the engine's own footprint and shrinks the page pool of a
	// model that fits.
	asked := maxmem
	if maxmem == 0 {
		maxmem = sched.MemBudget()
	}
	goheap.Cap()
	opts := loadOpts(topts, maxmem)
	if place != nil {
		opts = append(opts, model.WithPlacement(*place))
	}
	m, err := model.Open(path, opts...)
	// The meter starts at Open, not after placement: on a streamed model
	// admission is the largest read, and starting later leaves it uncounted.
	// See model.PhaseRow.
	var ph *model.PhaseMeter
	if m != nil {
		ph = m.NewPhaseMeter()
	}
	if err == nil && asked == 0 && m.DirectIO() {
		// Direct reads leave no page-cache shadow copy, so the budget may
		// claim more than sched.MemBudget's eight tenths. Nine tenths, not all:
		// the KV cache and the pool's scratch are anonymous, and goheap.Cap
		// keeps the collector a gigabyte under the same ceiling.
		//
		// The raise is also capped by sched.GCBudgetCap: page frames are
		// permanently live, and a budget near the heap goal makes the
		// collector run back to back (AGENTS.md RULE 2f).
		if lim := sched.MemLimit(); lim > 0 {
			raised := lim / 10 * 9
			if cap := sched.GCBudgetCap(); cap > 0 && cap < raised {
				raised = cap
			}
			if raised > maxmem {
				maxmem = raised
			}
		}
	}
	if err != nil {
		return err
	}
	defer m.Close()
	if err := preloadWait(m); err != nil {
		return err
	}
	goheap.OffHeap(maxmem)
	c := m.Cfg
	fmt.Fprintf(os.Stderr, "%s L=%d d=%d heads=%d/%d head_dim=%d ffn=%d vocab=%d\n",
		c.Arch, c.NLayer, c.NEmbd, c.NHead, c.NKVHead, c.HeadDim, c.NFFN, c.NVocab)
	// The pager's chunk size is stored in the container (jlm.chunkFor). It is
	// printed at load because the pages line below only appears when frames
	// are short.
	if n := m.ChunkBytes(); n > 0 {
		fmt.Fprintf(os.Stderr, "pager     %d KiB chunks, %d stream group(s)\n",
			n>>10, m.StreamGroups())
	}

	if m.Vocab == nil {
		return fmt.Errorf("%s: %w", path, m.TokErr)
	}
	// The host budget is settled before the devices open: a device whose
	// memory is the host's is carved out of it (tier.WithHostBudget).
	dev, closeDev, devName, err := openDevices(devSpec, vram, maxmem, m.StreamGroups())
	if err != nil {
		return err
	}
	defer closeDev()
	// A device a block streams through needs the packed arena, or every
	// page-in repacks, which is slower than the upload it schedules. The
	// arena gets what the host budget has left after the weights; a model too
	// big for the host lands on zero and repacks. JITLLM_ARENA, when set, wins.
	if g, ok := dev.(*tier.GPU); ok && place.Streams() && os.Getenv("JITLLM_ARENA") == "" {
		if maxmem > m.WeightBytes() {
			g.SetArena(maxmem - m.WeightBytes())
		}
	}

	// -chat is opt-in. The default stays raw completion because every board
	// row and the llama.cpp comparisons are completions of a bare prompt.
	ids := m.Vocab.Encode(prompt, true)
	replying := chat != nil && chat.repl
	if replying {
		if imgPath != "" || specOpts != nil || depth > 0 {
			return fmt.Errorf("the interactive chat takes neither -image, -spec nor -depth; give a prompt to use them")
		}
		ids = nil
	} else if chat != nil {
		if !m.HasChatTemplate() {
			return fmt.Errorf("-chat: %s carries no chat template; it is a base model.\n"+
				"  Run it without -chat and it completes the prompt instead.", path)
		}
		var msgs []model.ChatMessage
		if chat.system != "" {
			msgs = append(msgs, model.ChatMessage{Role: "system", Content: chat.system})
		}
		msgs = append(msgs, model.ChatMessage{Role: "user", Content: prompt})
		// ChatIDs renders AND tokenizes, because the two cannot be paired
		// wrongly: the rendered text carries its own specials and must be
		// encoded with addSpecial=false. See model.ChatPrompt.
		if ids, err = m.ChatIDs(msgs, true); err != nil {
			return err
		}
	}
	// -depth measures decode at a context length, where attention grows with
	// position and weight traffic does not. The padding repeats the prompt's
	// own tokens; only the length matters, as with llama-bench's -d.
	if depth > len(ids) {
		if len(ids) == 0 {
			return fmt.Errorf("cannot pad an empty prompt to depth %d", depth)
		}
		base := len(ids)
		for i := 0; len(ids) < depth; i++ {
			ids = append(ids, ids[i%base])
		}
	}
	// An image costs positions that are not in ids (it arrives as embeddings),
	// so the state must be sized for them before it exists. The tower lives in
	// the same container and shares this model's pager.
	tw := m.Tower()
	var spans []model.Span
	// pic is the picture as the prompt's spans carry it (model.Picture): the
	// prompt's prefill encodes it, so there is no encode stage here.
	var pic *picture
	extra := 0
	if imgPath != "" {
		if tw == nil {
			return fmt.Errorf("-image needs a container with a vision tower in it; "+
				"re-run `jitllm convert %s <mmproj.gguf> %s`", path, path)
		}
		// An image always goes through the model's own chat template, -chat or
		// not; -system composes with it.
		msgs := []model.ChatMessage{{Role: "user", Content: prompt, Images: 1}}
		if chat != nil && chat.system != "" {
			msgs = append([]model.ChatMessage{{Role: "system", Content: chat.system}}, msgs...)
		}
		if pic, err = openPicture(tw, imgPath); err != nil {
			return err
		}
		if pic.tiles != nil {
			ps, err := m.Llama4Spans(pic.tiles[0], pic.tiles[1], pic.pieces)
			if err != nil {
				return err
			}
			if spans, err = m.ChatSpansParts(msgs, [][]model.Span{ps}, true); err != nil {
				return err
			}
		} else if pic.lfm2 != nil {
			ps, err := m.LFM2Spans(pic.lfm2[0], pic.lfm2[1], pic.pieces)
			if err != nil {
				return err
			}
			if spans, err = m.ChatSpansParts(msgs, [][]model.Span{ps}, true); err != nil {
				return err
			}
		} else if pic.lay != nil {
			ps, err := m.PictureSpans(*pic.lay, 0, true, pic.pieces)
			if err != nil {
				return err
			}
			if spans, err = m.ChatSpansParts(msgs, [][]model.Span{ps}, true); err != nil {
				return err
			}
		} else if spans, err = m.ChatSpansImages(msgs, []model.Image{{Picture: pic.pieces[0]}}, true); err != nil {
			return err
		}
		extra = model.SpanPositions(spans, m.Cfg.NEmbd) - len(ids)
	}
	// Whether the pool spins or parks between regions is decided per token by
	// the engine, by whether the token paged anything in (model.State.forward);
	// JITLLM_SPIN_US still fixes it.
	// No -n: generate until the model ends its reply, in a session as long as
	// the model's context, as the server's sessions are by default. KV pages
	// are committed as the context grows, so the length costs nothing up front.
	replyCap := n
	if replying {
		// The conversation grows into the model's whole context.
		n = m.Cfg.NCtx - 1
	} else if n <= 0 {
		if n = m.Cfg.NCtx - len(ids) - extra - 1; n <= 0 {
			return fmt.Errorf("the prompt is %d positions and fills the model's context of %d", len(ids)+extra, m.Cfg.NCtx)
		}
	}
	st := m.NewState(len(ids) + extra + n + 1)
	defer st.Close()
	var kvStore *model.FileStore
	// The KV store is attached before the device is chosen. The namespace
	// names the model; everything else that must match (geometry, layout,
	// element width) is hashed into every page key by the engine.
	if kvCacheDir != "" {
		// Bounded by default, because nothing in a KV cache expires;
		// -kv-cache-max 0 makes it unbounded.
		fsStore, err := model.NewFileStoreLimit(kvCacheDir, kvCacheMaxBytes)
		if err != nil {
			return err
		}
		st.SetKVStore(fsStore)
		kvStore = fsStore
		// Not the basename alone: many containers are named model.jlm, and the
		// geometry hash cannot separate two fine-tunes of one architecture.
		// Size and mtime separate them without reading the file.
		ns := "jlm/" + filepath.Base(path)
		if fi, err := os.Stat(path); err == nil {
			ns = fmt.Sprintf("jlm/%s/%d-%d", filepath.Base(path), fi.Size(), fi.ModTime().UnixNano())
		}
		if err := st.SetCacheKey(ns); err != nil {
			return err
		}
	}
	// A resident KV cap spills the oldest pages into the store and fetches them
	// back on use; without a store it would drop history, so it is refused.
	if kvBudget > 0 {
		if err := st.SetKVBudget(kvBudget); err != nil {
			return fmt.Errorf("-kv-budget needs -kv-cache: %w", err)
		}
	}
	// newState clamps a request to the model's trained context. Say so
	// rather than letting generation stop early without explanation.
	if asked, got, clamped := st.ContextClamped(); clamped {
		fmt.Fprintf(os.Stderr,
			"context  %d positions requested, %d granted -- the model's trained context is %d\n",
			asked, got, m.Cfg.NCtx)
	}

	// The host budget is always reported: a model several times its budget
	// looks stuck, and the numbers say why.
	if maxmem > 0 {
		w := m.BytesPerToken()
		// The KV cache is reserved, not resident: Go takes a large span from
		// mmap without zeroing it, so pages commit only as the context reaches
		// them.
		kv := uint64(st.MaxSeq()) * uint64(m.Cfg.KVDim()) * 4 * 2 * uint64(m.Cfg.NLayer)
		fmt.Fprintf(os.Stderr, "memory   budget %.2f GiB   weights %.2f GiB   kv %.2f GiB reserved (-n %d, commits as the context grows)   touches %.2f GiB of weights a token",
			float64(maxmem)/(1<<30), float64(m.WeightBytes())/(1<<30),
			float64(kv)/(1<<30), n, float64(w)/(1<<30))
		if m.WeightBytes() > maxmem {
			fmt.Fprintf(os.Stderr, "   ★ OVER BUDGET by %.2f GiB",
				float64(m.WeightBytes()-maxmem)/(1<<30))
		}
		fmt.Fprintln(os.Stderr)
		// Over budget on a mixture of experts is the case jitllm can do something
		// about: keep the experts the router keeps asking for, drop the rest.
		// The full budget: State.SetMemBudget takes what a device on the host's
		// memory holds off it itself, once the device has placed its blocks and
		// whenever they move -- so an embedder that never reads HostReserved
		// still cannot double-spend the machine. See State.hostBudget.
		st.SetMemBudget(maxmem)
		if _, bud := st.Resident(); bud > 0 {
			fmt.Fprintf(os.Stderr, "         managing expert residency: %.2f GiB of expert budget\n",
				float64(bud)/(1<<30))
		}
	}

	if dev != nil {
		if grow {
			// Start on the CPU and migrate in while serving: preparing the
			// device up front costs more than the first tokens are worth, so
			// adopt one block per token instead.
			if err := st.SetDeviceLayers(dev, 0); err != nil {
				return err
			}
			target := c.NLayer
			if gpuLayers >= 0 {
				target = gpuLayers
			}
			st.PrewarmGPU(target)
			st.SetGPUTarget(target)
		} else if err := st.SetDeviceLayers(dev, gpuLayers); err != nil {
			return err
		}
		st.SetSeamTuning(tuneSeam)
		st.SetDeviceFallback(!noFallback)
		st.SetRelocate(relocate)
		fmt.Fprintf(os.Stderr, "device %s: %d/%d blocks", devName, st.GPULayers(), c.NLayer)
		// With no block placed, print the tier's own reason, so a capability
		// gap does not read as a memory gap.
		if st.GPULayers() == 0 {
			if g, ok := dev.(*tier.GPU); ok {
				if why := g.Err(); why != "" {
					fmt.Fprintf(os.Stderr, " -- %s", why)
				}
			}
		}
		if st.HeadOnDevice() {
			fmt.Fprint(os.Stderr, " + the output projection")
		}
		if grow {
			fmt.Fprint(os.Stderr, " (growing toward the card's limit while serving)")
		}
		if s := deviceStats(dev); s != "" {
			fmt.Fprintf(os.Stderr, ", %s", s)
		}
		fmt.Fprint(os.Stderr, deviceBudgets(dev))
		fmt.Fprintln(os.Stderr)
		// Account for every block the device declined, by reason.
		for _, d := range st.DeviceDeclines() {
			fmt.Fprintf(os.Stderr, "         %3d block(s) declined: %s\n", d.Blocks, d.Why)
		}
	} else {
		fmt.Fprintf(os.Stderr, "device %s\n", devName)
	}
	// After placement, because the host's page count is NLayer minus what the
	// devices took. Outside the device branch, so a CPU-only run gets a budget.
	setPageBudget(m, st, dev, maxmem)
	// With the device attached, HostBudget has taken off what an integrated GPU
	// holds; say what the host works to when that moved it.
	if maxmem > 0 && st.HostBudget() != maxmem {
		fmt.Fprintf(os.Stderr, "memory   host budget in force %.2f GiB: devices on the HOST's own "+
			"memory hold %.2f GiB of it (an integrated GPU spends the same bytes)\n",
			float64(st.HostBudget())/(1<<30), float64(hostReserved(dev))/(1<<30))
	}

	phRead(ph, dev, m)
	ph.Mark("load")
	if replying {
		if err := repl(m, st, sm, chat.system, replyCap, kvCacheDir != "", os.Stdin, os.Stdout); err != nil {
			return withDeviceErr(err, dev)
		}
		return nil
	}

	// -spec drafts with the model's own prediction block and verifies the
	// drafts in one pass. The draft session attaches to the device the trunk
	// is on. An image arrives as embeddings and a cached prefix skips the
	// rows the draft must also see, so neither combines with it.
	var sp *model.Speculator
	if specOpts != nil {
		if imgPath != "" || kvCacheDir != "" {
			return fmt.Errorf("-spec combines with neither -image nor -kv-cache: the prediction block " +
				"reads the trunk's hidden state at every prompt position, which an image's embeddings " +
				"and a restored prefix do not produce")
		}
		if sp, err = st.Speculate(specOpts...); err != nil {
			return err
		}
		defer sp.Close()
	}

	// JITLLM_WARMUP=k runs the prompt and -n greedy tokens k times on this
	// State, untimed, and resets it before the run that is measured: the
	// expert caches fill and the tuners settle as a server's would. The
	// device's counters printed at the end include the warm-ups.
	if k, _ := strconv.Atoi(os.Getenv("JITLLM_WARMUP")); k > 0 && sp == nil && imgPath == "" && kvCacheDir == "" {
		t0 := time.Now()
		for i := 0; i < k; i++ {
			lg, err := st.Prefill(ids)
			if err != nil {
				return withDeviceErr(err, dev)
			}
			tok := int32(0)
			for j := range lg {
				if lg[j] > lg[tok] {
					tok = int32(j)
				}
			}
			for j := 1; j < n; j++ {
				if tok, err = st.ForwardGreedy(tok); err != nil {
					return withDeviceErr(err, dev)
				}
			}
			st.Reset()
		}
		fmt.Fprintf(os.Stderr, "warm-up  %d untimed run(s) of the prompt and %d tokens in %.1fs, then reset\n",
			k, n, time.Since(t0).Seconds())
	}
	start := time.Now()
	// Batched: the prompt is processed PrefillChunk tokens at a time, so the
	// weights are read once per chunk instead of once per token. Prefill falls
	// back to a Forward loop by itself when the batch cannot pay.
	var logits []float32
	restored := 0
	specFirst := int32(-1)
	switch {
	case imgPath != "":
		logits, err = prefillImage(st, m, tw, pic, spans, kvCacheDir != "")
		restored = st.KVRestored()
	case kvCacheDir != "":
		logits, err = st.PrefillCached(ids)
		// A hybrid's recurrent summaries are stored beside its KV pages under
		// the same key, so every architecture reuses a prefix.
		restored = st.KVRestored()
	case sp != nil:
		// The prompt through the trunk and the prediction block; the first
		// token comes back decided, so the loop below starts from it.
		var first int32
		if first, err = sp.Start(ids, sm); err == nil {
			specFirst = first
		}
	default:
		logits, err = st.Prefill(ids)
	}
	if err != nil {
		return withDeviceErr(err, dev)
	}
	prefill := time.Since(start)
	phRead(ph, dev, m)
	ph.Mark("prefill")
	// So the profile below describes DECODE, not the prompt that preceded it.
	model.ResetProfile()

	top := 0
	if v := os.Getenv("JITLLM_TOP"); v != "" {
		top, _ = strconv.Atoi(v)
	}
	out := make([]int32, 0, min(n, 4096))
	// Stream the completion as it decodes. The suffix is recomputed from the
	// whole id sequence each step, because neither tokenizer is per-token
	// decodable: a BPE token can split a rune, and SPM's leading space depends
	// on position. The quadratic cost is negligible against a token.
	//
	// The prompt is echoed only for a completion; under -chat the model saw
	// the rendered template, and the reply is a separate turn.
	if chat == nil {
		fmt.Print(prompt)
	}
	// Each new id is decoded once (tok.ChatStream); a rune a token leaves
	// unfinished waits in held for the token that completes it.
	cs, decoded, held := m.Vocab.NewChatStream(), 0, ""
	emitOf := func(out []int32) {
		for _, id := range out[decoded:] {
			held += cs.Next(id)
		}
		decoded = len(out)
		if k := tok.Complete(held); k > 0 {
			fmt.Print(held[:k])
			held = held[k:]
		}
	}
	// Printing runs beside the next token, off the GPU's critical path. One
	// print at a time, in order; out is passed by value and only ever appended
	// to, so the goroutine reads a prefix that nothing writes.
	var printing chan struct{}
	emit := func() {
		if printing != nil {
			<-printing
		}
		done, o := make(chan struct{}), out
		printing = done
		go func() {
			emitOf(o)
			close(done)
		}()
	}
	// JITLLM_PAGE_SHRINK=<bytes> retargets every device's weight budget after
	// the first decoded token. It is a test hook: a demotion must change where
	// a block's weights are, never the ids.
	shrink := uint64(0)
	if v := os.Getenv("JITLLM_PAGE_SHRINK"); v != "" {
		if b, e := tier.ParseBytes(v); e == nil {
			shrink = b
		}
	}
	// A greedy decode asks for the token, not the logits: with the head on a
	// device the argmax runs there and four bytes come home. -top needs the
	// logits, so it keeps the other path.
	greedy := sm.Temp <= 0 && top == 0
	pending := int32(-1)
	start = time.Now()
	if sp != nil {
		// A round decides several tokens; the Speculator observes each into
		// the sampler itself.
		queue := []int32{specFirst}
		for len(out) < n {
			if len(queue) == 0 {
				if queue, err = sp.Next(sm); err != nil {
					return withDeviceErr(err, dev)
				}
			}
			next := queue[0]
			queue = queue[1:]
			if m.Vocab.IsEOG(next) {
				break
			}
			out = append(out, next)
			if len(queue) == 0 {
				emit()
			}
		}
		emit()
		n = 0 // the plain loop below has nothing left to do
	}
	for i := 0; i < n; i++ {
		if shrink > 0 && i == 1 {
			if g, ok := dev.(*tier.GPU); ok {
				kept, err := g.SetBudget(shrink)
				if err != nil {
					return err
				}
				fmt.Fprintf(os.Stderr, "\n★ shrinking every device budget to %d bytes mid-generation; %d block(s) keep their weights\n",
					shrink, kept)
			}
			shrink = 0
		}
		next := pending
		if next < 0 {
			next = sm.Sample(logits)
		}
		sm.Observe(next)
		if top > 0 {
			// The margin at each step is the thing to look at when jitllm and
			// llama.cpp disagree: a near-tie is float reassociation, a wide gap
			// is a bug.
			fmt.Fprintf(os.Stderr, "step %2d:", i)
			for _, c := range model.TopK(logits, top) {
				fmt.Fprintf(os.Stderr, "  %q %.4f", m.Vocab.Text(c.ID), c.Logit)
			}
			fmt.Fprintln(os.Stderr)
		}
		if m.Vocab.IsEOG(next) {
			break
		}
		out = append(out, next)
		emit()
		if greedy {
			if pending, err = st.ForwardGreedy(next); err != nil {
				return withDeviceErr(err, dev)
			}
		} else if top == 0 {
			// No logits are printed, so the token alone is wanted: a device
			// holding the head selects the sampler's candidates.
			if pending, err = st.ForwardSample(next, sm); err != nil {
				return withDeviceErr(err, dev)
			}
		} else if logits, err = st.Forward(next); err != nil {
			return withDeviceErr(err, dev)
		}
	}
	if printing != nil {
		<-printing
	}
	decode := time.Since(start)
	phRead(ph, dev, m)
	ph.Mark("decode")
	fmt.Println()
	if sp != nil {
		printSpec(os.Stderr, sp)
	}
	if grow && dev != nil {
		fmt.Fprintf(os.Stderr, "seam grew to %d/%d blocks\n", st.GPULayers(), c.NLayer)
	}
	// Say when the per-matvec seam was declined: the non-resident blocks ran
	// wholly on the host, on purpose. The verdict is measured on the model's
	// shapes, so it is only known after tokens have run.
	if g, ok := dev.(*tier.GPU); ok && g.Stats().TooSmall > 0 {
		fmt.Fprintf(os.Stderr,
			"%d matvecs stayed on the host: this model does not carry enough weight bytes "+
				"per host/device crossing to pay for one\n", g.Stats().TooSmall)
	}
	if n := 0; n > 0 {
		fmt.Fprintf(os.Stderr, "%d block(s) moved onto a later device so the context fits there\n", n)
	}
	if r := st.Relocations(); r > 0 {
		fmt.Fprintf(os.Stderr, "-relocate moved %d block(s) to the host so the context fits; the device "+
			"ends with %d of %d\n", r, st.GPULayers(), m.Cfg.NLayer)
	}
	if d := st.DeviceDemotions(); d > 0 {
		// Include the tier's reason for the demotion.
		why := ""
		if g, ok := dev.(*tier.GPU); ok && g.Err() != "" {
			why = ": " + g.Err()
		}
		fmt.Fprintf(os.Stderr,
			"★ the device failed %d time(s) and its blocks moved to the host%s; this rate is NOT comparable with a run that kept the card\n", d, why)
	}
	// Placement is not execution. The load banner says what each device holds;
	// this says what each device ran and read back (tier.Stats.Blocks counts
	// only submissions whose output came back). It is printed for one device
	// too, because a pager can place every block and still decline to compute.
	if g, ok := dev.(*tier.GPU); ok {
		per := g.DevStats()
		for i, n := range g.Placed() {
			fmt.Fprintf(os.Stderr, "device %-44s %d block(s) placed, %d run, %d matvec(s)\n",
				per[i].Device, n, per[i].Blocks, per[i].Served)
		}
		fmt.Fprintf(os.Stderr, "device %-44s %d block(s) of %d, whatever no device took\n",
			"host", c.NLayer-st.GPULayers(), c.NLayer)
		// The four terms of a page-in, summed over the whole run. TLayer is the
		// device's own compute: page-in happens in submit(), outside
		// layersOnce, so TLayer bounds what overlapping a page-in with compute
		// could buy.
		if ds := g.Stats(); ds.TPack > 0 || ds.TUpload > 0 || ds.TUnpack > 0 || ds.TLayer > 0 {
			sec := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1e6 }
			fmt.Fprintf(os.Stderr,
				"device cost: pack %.2f s, upload %.2f s, device unpack %.2f s, block compute %.2f s"+
					"  (%.2f GiB uploaded raw, %.2f GiB landed on the card)\n",
				sec(ds.TPack), sec(ds.TUpload), sec(ds.TUnpack), sec(ds.TLayer),
				float64(ds.UnpackBytes)/(1<<30), float64(ds.PageBytes)/(1<<30))
			// The stream line splits a streamed mixture's suspension by wall
			// (drain, host read, upload); a CPU profile cannot, because the
			// reads overlap. The rotary line says which arm built the table.
			if ds.Submits > 0 {
				us := func(d time.Duration) float64 { return float64(d.Microseconds()) / float64(ds.Submits) }
				fmt.Fprintf(os.Stderr, "host per submission: stage %.1f us, launch %.1f us (%.1f of it %d capture(s)), read %.1f us over %d\n",
					us(ds.TSubStage), us(ds.TSubLaunch), us(ds.TCapture), ds.Captures, us(ds.TSubRead), ds.Submits)
			}
			if ds.RopeTables > 0 || ds.RopeTableUploads > 0 {
				fmt.Fprintf(os.Stderr, "rotary table: %d row(s) built on the device, %d uploaded\n",
					ds.RopeTables, ds.RopeTableUploads)
			}
			if ds.TStreamRead > 0 || ds.TStreamPut > 0 || ds.TStreamWait > 0 {
				fmt.Fprintf(os.Stderr,
					"stream cost: drain %.2f s, host read %.2f s, upload %.2f s "+
						"over %d fill(s), %d overlapped read(s)\n",
					sec(ds.TStreamWait), sec(ds.TStreamRead), sec(ds.TStreamPut),
					ds.StreamFills, ds.StreamOverlaps)
				fmt.Fprintf(os.Stderr,
					"stream upload: sheet lookup %.2f s, gather %.2f s, host-to-device %.2f s for %.2f GiB (%.2f GiB/s); "+
						"%d plane(s) direct (%d page-locked), %d gathered; groups %d\n",
					sec(ds.TStreamSheet), sec(ds.TStreamCopy), sec(ds.TStreamH2D),
					float64(ds.StreamBytes)/(1<<30), float64(ds.StreamBytes)/(1<<30)/max(sec(ds.TStreamH2D), 1e-9),
					ds.StreamDirect, ds.StreamPinned, ds.StreamGathered, ds.StreamGroupsTuned)
			}
			if ds.StreamCacheHits+ds.StreamCacheMisses > 0 || ds.StreamCacheShort > 0 {
				fmt.Fprintf(os.Stderr, "expert cache: %d sheet(s) a block, %d hit(s), %d miss(es) (%.1f%% hit), %d block(s) without room for one (mean base %d B)\n",
					ds.StreamCacheSize, ds.StreamCacheHits, ds.StreamCacheMisses,
					100*float64(ds.StreamCacheHits)/max(float64(ds.StreamCacheHits+ds.StreamCacheMisses), 1), ds.StreamCacheShort, ds.AutoMeanBase)
			}
			if ds.HybridRuns > 0 {
				fmt.Fprintf(os.Stderr, "hybrid experts: %d block-step(s) on the host, %.2f s\n", ds.HybridRuns, sec(ds.THybrid))
				if ds.SplitRuns > 0 || ds.SplitAdmits > 0 || ds.TLateWrite > 0 {
					fmt.Fprintf(os.Stderr, "split experts: %d block-step(s) split, %d expert(s) run on the card, host %.2f s, %d taken into the caches in %.2f s; host sums written back in %.2f s\n",
						ds.SplitRuns, ds.SplitHits, sec(ds.TSplitHost), ds.SplitAdmits, sec(ds.TAdmit), sec(ds.TLateWrite))
				}
			}
			if ds.StreamPrefetched > 0 {
				fmt.Fprintf(os.Stderr, "cross-layer prefetch: %d expert(s) read ahead, %.2f s joining them\n",
					ds.StreamPrefetched, sec(ds.TStreamPrefetchWait))
			}
			if ds.ProbeExperts > 0 || ds.ProbeFused > 0 {
				fmt.Fprintf(os.Stderr, "cross-layer probe: %d of %d routed experts predicted (%.1f%%), %d probe(s) skipped on a fused route\n",
					ds.ProbeHits, ds.ProbeExperts, 100*float64(ds.ProbeHits)/max(float64(ds.ProbeExperts), 1), ds.ProbeFused)
			}
		}
	}
	// Say whether blocks paged, even when they did not: the slot count and
	// page-in count separate a model that fits from one streaming through a
	// few slots, which otherwise look identical from outside.
	if g, ok := dev.(*tier.GPU); ok {
		if st := g.Stats(); st.Slots > 0 && (st.PageIns > 0 || st.PageOuts > 0 || st.Blocks > 0) {
			// Zero page-ins with a streamed block means either the model fits
			// or the budget is below the slot floor and blocks were declined;
			// the two are told apart by st.Declined.
			how := " (no block streams; -placement N=DEV~ streams one through the device)"
			if place.Streams() {
				how = " -- nothing had to move: every block fits"
				if st.Declined > 0 {
					// DeclineWhy rather than LastErr: LastErr is last-writer-wins
					// across subsystems.
					why := st.DeclineWhy
					if why == "" {
						why = st.LastErr
					}
					how = fmt.Sprintf(" -- and %d tensor(s) were DECLINED rather than paged: %s",
						st.Declined, why)
				}
			}
			if st.PageIns > 0 || st.PageOuts > 0 {
				// Where the page-ins came from: arena hits are a DMA, Packs are
				// repacks outside the arena, which are slower.
				as := g.ArenaStats()
				how = fmt.Sprintf(", %d served from the packed arena and %d repacked "+
					"(JITLLM_ARENA=<bytes> retains packs; %.2f MiB of %.2f used)",
					as.Hits, st.Packs, float64(as.Bytes)/(1<<20), float64(as.Limit)/(1<<20))
			}
			// Tensors unpacked on the device appear in neither number above.
			// Reported outside the page-in branch so a model that fits shows
			// the load-time saving too. The bytes are the raw bytes that crossed.
			if st.Unpacks > 0 || st.NoUnpackKernel > 0 || st.NoUnpackRoom > 0 {
				why := ""
				if st.NoUnpackKernel+st.NoUnpackRoom > 0 {
					why = fmt.Sprintf(" (%d have no device unpack, %d had no room: %s)",
						st.NoUnpackKernel, st.NoUnpackRoom, st.UnpackWhy)
				}
				how += fmt.Sprintf("; %d tensor(s) uploaded RAW and unpacked on the "+
					"device, %.2f GiB in %s, staging %.2f MiB%s",
					st.Unpacks, float64(st.UnpackBytes)/(1<<30),
					st.TUnpack.Round(time.Millisecond),
					float64(st.StageVRAM)/(1<<20), why)
			}
			fmt.Fprintf(os.Stderr,
				"paging %d block slot(s), %d page-in(s), %d page-out(s), %.2f GiB moved%s\n",
				st.Slots, st.PageIns, st.PageOuts, float64(st.PageBytes)/(1<<30), how)
		}
	}
	if g, ok := dev.(*tier.GPU); ok {
		if as := g.ArenaStats(); as.Limit > 0 || as.Packs > 0 {
			fmt.Fprintf(os.Stderr,
				"arena %d packs, %d page-ins served, %.2f MiB retained of %.2f\n",
				as.Packs, as.Hits, float64(as.Bytes)/(1<<20), float64(as.Limit)/(1<<20))
		}
		// On unified memory the tier imports the arena's own pointer, so the
		// weights are never copied or charged; only this counter shows it.
		if st := g.Stats(); st.Imports > 0 {
			fmt.Fprintf(os.Stderr,
				"imports %d tensor(s) wrapped in place, %.2f GiB not copied and not charged "+
					"(unified memory: the device reads the arena's own bytes)\n",
				st.Imports, float64(st.ImportBytes)/(1<<30))
		}
	}
	if tuneSeam && dev != nil {
		if b, settled := st.SeamTuned(); settled {
			fmt.Fprintf(os.Stderr, "seam tuned to %d/%d blocks\n", b, c.NLayer)
		} else {
			fmt.Fprintf(os.Stderr, "seam tuning did not settle (needs about %d tokens); left at %d/%d\n",
				32+3*4*24, st.GPULayers(), c.NLayer)
		}
	}
	// The prompt is the text ids plus whatever an image spliced in (marker
	// tokens and one embedding per projected patch group).
	ptok := len(ids) + extra
	fmt.Fprintf(os.Stderr, "\nprompt %d tok in %v (%s)   decode %d tok in %v (%s)\n",
		ptok, prefill.Round(time.Millisecond), rateOf(ptok, prefill),
		len(out), decode.Round(time.Millisecond), rateOf(len(out), decode))
	if nodes, aff := st.JIT().NUMA(); nodes > 1 {
		fmt.Fprintf(os.Stderr, "numa     %d nodes, %d region(s) read node-affine\n", nodes, aff)
	}
	// Report what the KV cache did, including zero: a working cache and a
	// silently dead one look identical on a first run.
	if kvCacheDir != "" {
		// A prompt holding a picture is its spans' positions, not its ids.
		positions := len(ids)
		if spans != nil {
			positions = model.SpanPositions(spans, m.Cfg.NEmbd)
		}
		pct := 0.0
		if positions > 0 {
			pct = 100 * float64(restored) / float64(positions)
		}
		fmt.Fprintf(os.Stderr, "kv cache %d of %d prompt position(s) reused (%.0f%%)",
			restored, positions, pct)
		if f := st.KVStoreFailures(); f > 0 {
			fmt.Fprintf(os.Stderr, "   %d STORE WRITE FAILURE(S)", f)
		}
		if m := st.KVStoreMismatches(); m > 0 {
			fmt.Fprintf(os.Stderr, "   %d stale page(s) dropped", m)
		}
		if d := st.KVDeviceSyncFailures(); d > 0 {
			fmt.Fprintf(os.Stderr, "   %d placed block(s) would not hand their history back, "+
				"so nothing was cached for them", d)
		}
		if u := st.KVPrefixUnmigratable(); u > 0 {
			// Neutral wording: the counter covers both a tier that refused the
			// migration and a hybrid whose recurrent summary does not cover
			// the fill the pages were found at.
			fmt.Fprintf(os.Stderr, "   prefix given up (%d): the history could not be "+
				"restored at the position the pages were found, so the prompt was "+
				"prefilled in full", u)
		}
		// model.sealTail stores the partial page a prompt ends inside, so a
		// miss here means the prefix genuinely differs.
		fmt.Fprintln(os.Stderr)
		stored, fetched, evicted := st.KVPageStats()
		fmt.Fprintf(os.Stderr, "         %d page(s) stored, %d fetched back, %d evicted from memory",
			stored, fetched, evicted)
		if kvStore != nil {
			if total, err := kvStore.Total(); err == nil {
				fmt.Fprintf(os.Stderr, "   the store holds %d page file(s), %d deleted to stay under -kv-cache-max",
					total, kvStore.Evicted())
			}
		}
		fmt.Fprintln(os.Stderr)
	}
	// Say whether pages faulted: a run fully resident and one a page short
	// are different measurements.
	// evict, not out: `out` is the generated token slice in this scope.
	if fr, in, evict := m.PageStats(); fr > 0 {
		// Read requests, not page-ins, are what a partial page-in costs. The
		// chunk size is the File's own, so this reports what the pager did.
		fmt.Fprintf(os.Stderr, "pages    %d frame(s), %d page-in(s), %d eviction(s), %d fresh frame(s), %.2f GiB read in %d request(s) at %d KiB chunks\n",
			fr, in, evict, m.FreshFrames(), float64(m.BytesRead())/(1<<30), m.PagerReads(), m.ChunkBytes()>>10)
		// The same total split by phase: the whole-run figure is dominated by
		// admission on a streamed model. Each phase's rate is over its own
		// wall, which includes compute, so it is a floor.
		for _, r := range ph.Rows() {
			per := ""
			switch r.Name {
			case "decode":
				if len(out) > 0 {
					per = fmt.Sprintf(", %s B/token", commas(r.PerToken(len(out))))
				}
			case "prefill":
				if len(ids) > 0 {
					per = fmt.Sprintf(", %s B/prompt token", commas(r.PerToken(len(ids))))
				}
			}
			unit := "of wall"
			if r.OverRead() {
				unit = "reading"
			}
			fmt.Fprintf(os.Stderr, "  %-8s %6.2f GiB in %6d request(s), %4d fault(s), %5.2f GiB/s %s%s\n",
				r.Name, float64(r.Bytes)/(1<<30), r.Reads, r.Faults,
				r.Rate()/(1<<30), unit, per)
		}
		// The routed expert reads on their own line, whole-run: they are the
		// only part of the read that cannot be predicted a token ahead.
		if w, io, rd, by, calls := m.ExpertIO(); calls > 0 {
			fmt.Fprintf(os.Stderr, "  experts  %6.2f GiB in %6d request(s) over %d layer-step(s), %s blocked of which %s is I/O\n",
				float64(by)/(1<<30), rd, calls, w.Round(time.Millisecond), io.Round(time.Millisecond))
		}
	}
	fmt.Fprintf(os.Stderr, "ids %v\n", append(append([]int32{}, ids...), out...))
	// The op split, when asked for; useful at long context, where attention is
	// the one O(S) op.
	if names, ns := model.OpProfile(); ns[0] > 0 {
		var tot int64
		for _, v := range ns {
			tot += v
		}
		fmt.Fprintf(os.Stderr, "per-op:")
		for i, n := range names {
			if ns[i] > 0 {
				fmt.Fprintf(os.Stderr, "  %s %.1f%%", n, 100*float64(ns[i])/float64(tot))
			}
		}
		fmt.Fprintln(os.Stderr)
	}
	// The serial/parallel split, in absolute milliseconds rather than shares:
	// on a container the kernel and widen buckets are filled only by the GGUF
	// row-major path, so a share would read "quantize 100%".
	if q, kn, o := nn.ProfileNanos(); q+kn+o > 0 {
		ms := func(v int64) float64 { return float64(v) / 1e6 }
		fmt.Fprintf(os.Stderr, "quantize %.1f ms (activations, host)", ms(q))
		if kn+o > 0 {
			fmt.Fprintf(os.Stderr, "   kernel %.1f ms   widen %.1f ms", ms(kn), ms(o))
		}
		fmt.Fprintln(os.Stderr)
	}
	// The matvec bucket split into packing, kernel and transpose
	// (JITLLM_MMPROF).
	if pk, kn, tr, tot, calls, regions := nn.MatMulProfile(); calls > 0 {
		ms := func(v int64) float64 { return float64(v) / 1e6 }
		other := tot - pk - kn - tr
		fmt.Fprintf(os.Stderr,
			"MatMul: %d calls, %d pool regions, %.0f ms total"+
				"  (pack %.0f ms %.1f%%, kernel %.0f ms %.1f%%, transpose %.0f ms %.1f%%, other %.0f ms %.1f%%)\n",
			calls, regions, ms(tot),
			ms(pk), 100*float64(pk)/float64(tot),
			ms(kn), 100*float64(kn)/float64(tot),
			ms(tr), 100*float64(tr)/float64(tot),
			ms(other), 100*float64(other)/float64(tot))
	}
	return nil
}

// tokenize runs a model's tokenizer over some text. It reads the container,
// because that holds the tokenizer as conversion resolved it, which is what the
// engine runs.
func tokenize(path, text string, topts []tok.Option) error {
	c, err := jlm.Open(path)
	if err != nil {
		return err
	}
	defer c.Close()
	v, err := tok.New(c.Vocab(), topts...)
	if err != nil {
		return err
	}
	ids := v.Encode(text, true)
	fmt.Printf("%d tokens: %v\n", len(ids), ids)
	for _, id := range ids {
		fmt.Printf("  %6d -> %q\n", id, strings.ReplaceAll(v.Text(id), "\u2581", " "))
	}
	if got := v.Decode(ids); strings.TrimPrefix(got, " ") != text {
		fmt.Printf("round trip differs: %q\n", got)
	}
	return nil
}

// memWall is the DRAM read bandwidth of the development machine's P-cores.
// Every throughput number is reported as a fraction of a wall, never on its
// own: "85 tok/s" is unfalsifiable, "85 tok/s = 45.3 GB/s = 81% of the wall"
// can be checked with a calculator (AGENTS.md RULE 1).
const memWall = 55.71e9

func info(path string, showKV bool) error {
	f, err := gguf.Open(path, gguf.WithoutWarm(os.Getenv("JITLLM_NO_WARM") != ""))
	if err != nil {
		return err
	}
	defer f.Close()

	tb := f.TensorBytes()
	fmt.Printf("file      %s\n", path)
	fmt.Printf("gguf      v%d   %d tensors   %d kv   alignment %d\n",
		f.Version, len(f.Tensors), len(f.KV), f.Alignment)
	fmt.Printf("layout    data_start %d (%#x)   tensor bytes %s   trailing %d\n",
		f.DataStart, f.DataStart, commas(tb), f.Size()-f.DataStart-tb)

	// -kv prints the header verbatim, for keys the summary below does not know.
	if showKV {
		keys := make([]string, 0, len(f.KV))
		for k := range f.KV {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		fmt.Println("kv")
		for _, k := range keys {
			fmt.Printf("          %-46s %s\n", k, f.KV[k].Short())
		}
	}

	arch := f.Arch()
	name, _ := f.KV["general.name"].String()

	// An mmproj keeps its geometry under clip.vision.*, while meta.File.Key
	// only prefixes the arch, so add the "vision." segment.
	kp := ""
	if arch == "clip" {
		kp = "vision."
	}
	nLayer := f.UintKey(kp+"block_count", 0)
	embd := f.UintKey(kp+"embedding_length", 0)
	heads := f.UintKey(kp+"attention.head_count", 0)
	kvHeads := f.UintKey(kp+"attention.head_count_kv", heads) // absent => MHA
	headDim := f.UintKey(kp+"attention.key_length", 0)
	if headDim == 0 && heads != 0 {
		headDim = embd / heads
	}
	fmt.Printf("arch      %s %q\n", arch, name)
	fmt.Printf("          L=%d  d=%d  ffn=%d  heads=%d/%d  head_dim=%d  n_rot=%d\n",
		nLayer, embd, f.UintKey(kp+"feed_forward_length", 0), heads, kvHeads, headDim,
		f.UintKey("rope.dimension_count", headDim))
	if arch == "clip" {
		// A vision tower has no rope and no context; the projector decides
		// whether jitllm can run it.
		proj, _ := f.KV["clip.projector_type"].String()
		img := f.UintKey(kp+"image_size", 0)
		patch := f.UintKey(kp+"patch_size", 0)
		side := uint64(0)
		if patch != 0 {
			side = img / patch
		}
		fmt.Printf("          projector=%q  scale_factor=%d  image=%d  patch=%d  -> %d patches\n",
			proj, f.UintKey(kp+"projector.scale_factor", 0), img, patch, side*side)
		fmt.Printf("          layer_norm_eps=%g\n",
			f.FloatKey(kp+"attention.layer_norm_epsilon", 0))
	} else {
		fmt.Printf("          ctx=%d  rms_eps=%g  rope_base=%g\n",
			f.UintKey("context_length", 0),
			f.FloatKey("attention.layer_norm_rms_epsilon", 0),
			f.FloatKey("rope.freq_base", 10000))
	}

	fmt.Println("types")
	for _, c := range f.TypeHistogram() {
		pct := 100 * float64(c.Bytes) / float64(tb)
		fmt.Printf("          %-8s %4d  %14s  %6.2f%%\n", c.Type, c.Count, commas(c.Bytes), pct)
	}

	// Intra-block order is converter-dependent and differs between every model
	// family seen so far. Printing it is how you notice before a kernel does.
	if roles := blockRoles(f); len(roles) > 0 {
		fmt.Printf("block 0   %s\n", strings.Join(roles, " "))
	}
	if nLayer > 0 {
		fmt.Println("layers    (offsets relative to data_start; each block is one contiguous run)")
		for l := uint64(0); l < nLayer && l < 3; l++ {
			off, size, ok := f.LayerRange(int(l))
			if !ok {
				continue
			}
			fmt.Printf("          blk.%-3d [%12d, %12d)  %12s\n", l, off, off+size, commas(size))
		}
		if nLayer > 3 {
			fmt.Printf("          ... %d more\n", nLayer-3)
		}
	}

	// Decode traffic: every weight is read once per token except the embedding
	// table, of which exactly one row is read. A tied token_embd is also the
	// output projection and is read in full, so it is subtracted only when
	// output.weight exists. This mirrors model.BytesPerToken.
	perTok := tb
	if e, ok := f.Get("token_embd.weight"); ok && len(e.Dims) > 0 {
		row := e.Dims[0] / e.Type.BlockElems() * e.Type.BlockBytes()
		if _, untied := f.Get("output.weight"); untied {
			perTok = tb - e.NBytes + row
		} else {
			perTok = tb + row // counted once as lm_head, plus the row lookup
		}
	}
	// A mixture reads NExpertUsed of NExpert: a 3-D tensor is an expert bank,
	// scaled as model.BytesPerToken scales it.
	if nUsed := int(f.UintKey("expert_used_count", 0)); nUsed > 0 {
		if nExp := int(f.UintKey("expert_count", 0)); nExp > nUsed {
			var bank uint64
			for _, t := range f.Tensors {
				if len(t.Dims) >= 3 {
					bank += t.NBytes
				}
			}
			perTok -= bank - bank/uint64(nExp)*uint64(nUsed)
		}
	}
	// Sized at f32, the widest KV cache either tier allocates.
	kvPerPos := 2 * nLayer * kvHeads * headDim * 4
	fmt.Printf("decode    %s B/token weights   %s B/position kv\n", commas(perTok), commas(kvPerPos))
	fmt.Printf("roofline  %.1f tok/s at the %.1f GB/s 6P wall (upper bound, weights only)\n",
		memWall/float64(perTok), memWall/1e9)
	return nil
}

func blockRoles(f *meta.File) []string {
	var out []string
	for i := range f.Tensors {
		if n := f.Tensors[i].Name; strings.HasPrefix(n, "blk.0.") {
			out = append(out, strings.TrimSuffix(n[6:], ".weight"))
		}
	}
	return out
}

func commas(v uint64) string {
	s := fmt.Sprint(v)
	if len(s) <= 3 {
		return s
	}
	var b strings.Builder
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(c)
	}
	return b.String()
}

// runCmd parses run's flags and hands off. Device selection lives here rather
// than in a second binary; see the package comment.
func runCmd(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	dev := fs.String("devices", "auto", "auto|all|cpu|gpu[:N]|(cuda|vulkan|metal)[:DEVICE][=BYTES], comma-separated")
	layers := fs.Int("gpu-layers", -1, "cap the blocks offered to the device; -1 fits as many as the card holds")
	grow := fs.Bool("gpu-grow", false, "start on the CPU and migrate blocks to the device while serving")
	tuneSeam := fs.Bool("tune-seam", false, "measure whether fewer device blocks are faster, and relocate")
	noFallback := fs.Bool("no-gpu-fallback", false, "fail instead of finishing on the host when the device breaks")
	relocate := fs.Bool("relocate", true, "move blocks to the host when the device cannot grow the context, rather than demoting every block; on wherever -devices admits the host (auto, or a list with cpu)")
	// 0 means "ask the device": free memory less headroom, per device, rather
	// than one constant for every card on every machine. See tier.budgetFor.
	vram := bytesFlag(fs, "vram", 0, "device weight budget, bytes or 3G/512M; 0 asks each device what it has free")
	// 0 means "work it out": the cgroup limit or MemAvailable, whichever binds.
	maxmem := bytesFlag(fs, "maxmem", 0, "host weight residency budget, bytes or 3G/512M")
	n := fs.Int("n", 0, "tokens to generate; 0 runs until the model ends its reply or fills its context")
	chat := fs.Bool("chat", false, "wrap the prompt in the model's own chat template (instruct models)")
	system := fs.String("system", "", "a system message, with -chat")
	depth := fs.Int("depth", 0, "pad the prompt to this many tokens before generating, so decode is measured at that context length")
	// Sampling, named to match llama.cpp so a prompt moves between engines.
	// temp 0 is greedy and is the default: every correctness test in the tree
	// compares argmax, and a sampler on by default would make them fail
	// randomly rather than loudly.
	temp := fs.Float64("temp", 0, "sampling temperature; 0 = greedy (default)")
	topK := fs.Int("top-k", 0, "keep only the k most likely tokens; 0 = off")
	topP := fs.Float64("top-p", 0, "nucleus sampling; 0 or 1 = off")
	minP := fs.Float64("min-p", 0, "keep tokens above min-p * p(best); 0 = off")
	repPen := fs.Float64("repeat-penalty", 1, "penalise recently seen tokens; 1 = off")
	repN := fs.Int("repeat-last-n", 64, "how far back repeat-penalty looks")
	seed := fs.Int64("seed", 0, "RNG seed for sampling")
	// No -mmproj: the vision tower is in the container.
	imgPath := fs.String("image", "", "image to place in the prompt; the container must carry a vision tower")
	tokJSON := fs.String("tokenizer", "", "a HuggingFace tokenizer.json whose pre-tokenizer replaces the one tokenizer.ggml.pre names")
	// The cache namespace is not a flag: the engine hashes the geometry into
	// every page key and names the model from the container path.
	kvCache := fs.String("kv-cache", "",
		"directory holding the KV cache of earlier prompts; a prompt sharing a prefix of 16 or more positions with one skips it (a hybrid model restores only a whole prompt it saw before)")
	kvCacheMax := bytesFlag(fs, "kv-cache-max", 8<<30,
		"bytes the KV page cache may occupy before the least recently used pages are evicted; 0 is unbounded")
	placement := fs.String("placement", "",
		"where blocks and the head run, e.g. 0=host,1-7=cuda:0,8=host!,9-15=vulkan:1,head=vulkan:1; "+
			"a trailing ! pins the entry, and a device must be one -devices names")
	placeStrict := fs.Bool("placement-strict", false,
		"fail when a -placement entry cannot be met, instead of letting the engine place that block")
	kvBudget := bytesFlag(fs, "kv-budget", 0,
		"bytes (or 3G/512M) of KV history this session may hold in memory; older pages spill to -kv-cache and come back on use. 0 is unbounded")
	// A profile separates generated kernels, pool spin and pager reads, which
	// top shows as one number.
	cpuProf := fs.String("cpuprofile", "",
		"write a CPU profile here; `go tool pprof` then separates the generated "+
			"kernels from the pool's spin and from the pager's reads")
	spec := addSpecFlags(fs)
	addr := fs.String("addr", localserver.DefaultAddr,
		"the jitllmd a model NAME is run through when one answers there (see -embedded)")
	apiKey := fs.String("api-key", "", "the API key the jitllmd wants, if it has one (default: $JITLLM_API_KEY)")
	embedded := fs.Bool("embedded", false,
		"run in this process even when a jitllmd answers; a path, or any flag about this process's engine, does too")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := misplacedFlag(fs); err != nil {
		return err
	}
	specOpts, err := spec.options()
	if err != nil {
		return err
	}
	topts, closeTok, err := tokenizerOption(*tokJSON)
	if err != nil {
		return err
	}
	defer closeTok()
	prompt := strings.Join(fs.Args()[1:], " ")
	interactive := false
	switch {
	case fs.NArg() == 0:
		usage()
	case fs.NArg() == 1 && stdinIsTerminal():
		// No prompt and a person at the keyboard: a conversation, through the
		// model's own template (repl.go).
		interactive = true
	case fs.NArg() == 1:
		// No prompt on the command line and stdin a pipe or file: that is it.
		b, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}
		if prompt = strings.TrimRight(string(b), "\r\n"); prompt == "" {
			usage()
		}
	}
	sm := &model.Sampler{Temp: *temp, TopK: *topK, TopP: *topP, MinP: *minP,
		RepeatPen: *repPen, RepeatLastN: *repN, Seed: *seed}
	var cs *chatSpec
	if *chat || interactive {
		cs = &chatSpec{system: *system, repl: interactive}
	} else if *system != "" {
		return fmt.Errorf("-system needs -chat: without a template there is no system turn to put it in")
	}
	svc, path, err := runTarget(context.Background(), fs.Arg(0), *addr, keyOf(*apiKey), *embedded, engineFlags(fs, remoteFlags), os.Stderr)
	if err != nil {
		return err
	}
	if svc != nil {
		r := newRemoteRun(svc.Addr, keyOf(*apiKey), fs.Arg(0), *n, *system, fs, samplingFlags{temp: *temp, topP: *topP,
			minP: *minP, repPen: *repPen, topK: *topK, seed: *seed})
		if interactive {
			return r.repl(os.Stdin, os.Stdout)
		}
		return r.once(*chat, prompt, os.Stdout)
	}
	// -relocate is on by default: without it, a card that cannot grow its
	// history demotes every block to the host rather than one. A -devices list
	// without the host keeps it off, and asking for it there is an error.
	explicit := false
	fs.Visit(func(f *flag.Flag) { explicit = explicit || f.Name == "relocate" })
	if err := relocatable(*dev); err != nil {
		if explicit && *relocate {
			return err
		}
		*relocate = false
	}
	if *cpuProf != "" {
		f, err := os.Create(*cpuProf)
		if err != nil {
			return err
		}
		defer f.Close()
		sched.SetRegionLabels(true) // attribution for generated code; see sched.regionLabels
		if err := pprof.StartCPUProfile(f); err != nil {
			return err
		}
		defer pprof.StopCPUProfile()
	}
	var place *model.Placement
	if *placement != "" {
		p, err := model.ParsePlacement(*placement, *placeStrict)
		if err != nil {
			return err
		}
		place = &p
	} else if *placeStrict {
		return fmt.Errorf("-placement-strict needs -placement")
	}
	return run(path, prompt, *n, *depth, *dev, *layers, *vram, *maxmem, *grow, *tuneSeam, *noFallback, *relocate, sm, *imgPath, topts, cs, *kvCache, *kvCacheMax, *kvBudget, place, specOpts)
}

// tokenizerOption turns a -tokenizer path into the tok.Option it names, plus the
// closer for the file it opened. An empty path is no option and a no-op closer,
// so every caller is one shape.
//
// The file stays open across the load: tok.WithTokenizer takes a reader and
// parses it inside tok.New.
func tokenizerOption(path string) ([]tok.Option, func(), error) {
	if path == "" {
		return nil, func() {}, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, func() {}, err
	}
	return []tok.Option{tok.WithTokenizer(f)}, func() { f.Close() }, nil
}

// loadOpts turns the tokenizer options into load options and adds the page
// budget, which has to be known at Open because it decides how many block pages
// the container may hold. Zero means "ask the process".
// preloadWait holds the run until the background preload has read every page
// when JITLLM_PRELOAD_WAIT=1: a warm measurement whose model fits the budget
// then faults nothing in, as an engine that copies its weights to RAM before
// its first token does. Unset, the prompt starts beside the reads.
func preloadWait(m *model.Model) error {
	if os.Getenv("JITLLM_PRELOAD_WAIT") != "1" {
		return nil
	}
	t0 := time.Now()
	err := m.WaitPreload()
	fmt.Fprintf(os.Stderr, "preload   waited %.1fs for every page\n", time.Since(t0).Seconds())
	return err
}

func loadOpts(t []tok.Option, budget uint64) []model.Option {
	o := make([]model.Option, 0, len(t)+2)
	for _, x := range t {
		o = append(o, model.WithTokenizerOption(x))
	}
	if budget > 0 {
		o = append(o, model.WithPageBudget(budget))
	}
	if j := jitOptions(); len(j) > 0 {
		o = append(o, model.WithJITOptions(j...))
	}
	// JITLLM_HUGEPAGES=0 leaves the frames on the kernel's default page size.
	if os.Getenv("JITLLM_HUGEPAGES") == "0" {
		o = append(o, model.WithHugePages(false))
	}
	// JITLLM_SH_OVERLAP=0 runs a mixture's shared experts after its routed
	// read instead of behind it.
	// JITLLM_EXPERTS=host|card places every mixture block's routed experts
	// off the card (model.WithExperts); -placement's %host / %card per block.
	if e := os.Getenv("JITLLM_EXPERTS"); e != "" {
		o = append(o, model.WithExperts(e))
	}
	// JITLLM_STREAM_TRIAL=0 keeps an auto-streamed placement unmeasured.
	if os.Getenv("JITLLM_STREAM_TRIAL") == "aa" {
		o = append(o, model.WithStreamTrialAA(true))
	}
	if os.Getenv("JITLLM_STREAM_TRIAL") == "0" {
		o = append(o, model.WithStreamTrial(false))
	}
	// JITLLM_FUSE=0 runs decode one pool region per op (model.WithBlockFusion).
	if os.Getenv("JITLLM_FUSE") == "0" {
		o = append(o, model.WithBlockFusion(false))
	}
	// JITLLM_FUSE_GROUPS=0 keeps a fused run's matvecs a phase each
	// (model.WithMatVecGroups).
	if os.Getenv("JITLLM_FUSE_GROUPS") == "0" {
		o = append(o, model.WithMatVecGroups(false))
	}
	// JITLLM_LOOKAHEAD=0 turns off the fused runs' weight lookahead
	// (model.WithLookahead).
	if os.Getenv("JITLLM_LOOKAHEAD") == "0" {
		o = append(o, model.WithLookahead(false))
	}
	// JITLLM_FUSE_RESID=1 folds the residual add into the o and down groups
	// (model.WithResidualFold).
	if os.Getenv("JITLLM_FUSE_RESID") == "1" {
		o = append(o, model.WithResidualFold(true))
	}
	if os.Getenv("JITLLM_SH_OVERLAP") == "0" {
		o = append(o, model.WithSharedOverlap(false))
	}
	// JITLLM_TAIL_ROWS=0 runs a prompt's few-row chunk batched rather than as
	// a ragged step of its rows (model.WithTailRows).
	if os.Getenv("JITLLM_TAIL_ROWS") == "0" {
		o = append(o, model.WithTailRows(false))
	}
	if n := numaNodes(); n != nil {
		o = append(o, model.WithInterleave(n))
	}
	if c := pagerChunk(); c > 0 {
		o = append(o, model.WithChunk(c))
	}
	// JITLLM_PRELOAD=n keeps n pages in flight behind Open (model.WithPreload),
	// 0 reads each page when a token faults it in. 4 when unset.
	preload := 4
	if v := os.Getenv("JITLLM_PRELOAD"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			preload = n
		}
	}
	o = append(o, model.WithPreload(preload))
	// JITLLM_DEVICE_SAMPLE=on|off forces where a sampled token's candidates
	// are selected (model.WithDeviceSample); unset or auto measures.
	switch os.Getenv("JITLLM_DEVICE_SAMPLE") {
	case "on":
		o = append(o, model.WithDeviceSample(model.DeviceSampleOn))
	case "off":
		o = append(o, model.WithDeviceSample(model.DeviceSampleOff))
	}
	return append(o, modelOptions()...)
}

// modelOptions turns the graph tier's environment variables into model.Options.
// They are read here and nowhere else; see jitOptions.
func modelOptions() []model.Option {
	var o []model.Option
	if os.Getenv("JITLLM_PROFILE") != "" {
		o = append(o, model.WithProfile(true))
	}
	// JITLLM_HYBRID_SPIN_US sets how long the workers spin between a
	// device's host expert calls (model.WithHybridSpin); 0 keeps the pool's.
	if v := os.Getenv("JITLLM_HYBRID_SPIN_US"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			o = append(o, model.WithHybridSpin(time.Duration(n)*time.Microsecond))
		}
	}
	if os.Getenv("JITLLM_NO_HEAD_PLACE") != "" {
		o = append(o, model.WithHeadPlacement(false))
	}
	if os.Getenv("JITLLM_KV_HEADMAJOR") != "" {
		o = append(o, model.WithKVHeadMajor(true))
	}
	if v := os.Getenv("JITLLM_KV_F16"); v != "" {
		o = append(o, model.WithKVF16(v == "1"))
	}
	// The cache's format by name (f32, f16, q8_0); it overrides JITLLM_KV_F16.
	if v := os.Getenv("JITLLM_KV_TYPE"); v != "" {
		t, err := model.ParseKVType(v)
		if err != nil {
			fmt.Fprintf(os.Stderr, "jitllm: JITLLM_KV_TYPE: %v\n", err)
			os.Exit(2)
		}
		o = append(o, model.WithKVType(t))
	}
	if v := os.Getenv("JITLLM_ATTN_PAIR"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			o = append(o, model.WithAttnPair(n))
		}
	}
	if os.Getenv("JITLLM_ELEM_POOL") == "0" {
		o = append(o, model.WithTowerTiling(false))
	}
	if v := os.Getenv("JITLLM_TOWER_GPU_LAYERS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			o = append(o, model.WithTowerGPULayers(n))
		}
	}
	if os.Getenv("JITLLM_GROW_DEBUG") != "" {
		o = append(o, model.WithGrowDebug(true))
	}
	if os.Getenv("JITLLM_PREFILL_STREAM_DEBUG") != "" {
		o = append(o, model.WithPrefillStreamDebug(true))
	}
	n := func(k string) int {
		v, _ := strconv.Atoi(os.Getenv(k))
		return v
	}
	if w, r, q := n("JITLLM_SEAM_WARMUP"), n("JITLLM_SEAM_RUN"), n("JITLLM_SEAM_ROUNDS"); w > 0 || r > 0 || q > 0 ||
		os.Getenv("JITLLM_SEAM_DEBUG") != "" {
		o = append(o, model.WithSeamTuneSchedule(w, r, q, os.Getenv("JITLLM_SEAM_DEBUG") != ""))
	}
	return o
}

// jitOptions turns the generated tier's environment variables into nn.Options.
// They are read here and nowhere else: the libraries read no environment, so an
// embedded caller can set every knob through options.
func jitOptions() []nn.Option {
	var o []nn.Option
	if s := schedOptions(); len(s) > 0 {
		o = append(o, nn.WithSched(s...))
	}
	if os.Getenv("JITLLM_PROFILE") != "" {
		o = append(o, nn.WithProfile(true))
	}
	// The activation quantizer's Go-loop arm (cpu.QuantizeQ8Window), so both
	// arms of that comparison are reachable from one binary.
	if os.Getenv("JITLLM_QUANT_GO") != "" {
		o = append(o, nn.WithQuantActGo(true))
	}
	// The rotary table's Go-loop arm, for the same comparison. It only exists
	// in a jitllmtest build (engine/nn/ropego.go).
	if os.Getenv("JITLLM_ROPE_GO") != "" {
		o = append(o, nn.WithRopeGo(true))
	}
	if os.Getenv("JITLLM_MMPROF") != "" {
		o = append(o, nn.WithMatMulProfile(true))
	}
	if os.Getenv("JITLLM_TUNE_QUIET") != "" {
		o = append(o, nn.WithQuietTuner(true))
	}
	switch os.Getenv("JITLLM_TUNE") {
	case "0":
		o = append(o, nn.WithTune(nn.TuneOff))
	case "1":
		o = append(o, nn.WithTune(nn.TuneForce))
	}
	if v := os.Getenv("JITLLM_PACK"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			o = append(o, nn.WithPackWidth(n))
		}
	}
	if v := os.Getenv("JITLLM_FPF"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			o = append(o, nn.WithFusedPrefetch(n))
		}
	}
	if v := os.Getenv("JITLLM_PREFETCH"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			o = append(o, nn.WithPrefetch(n))
		}
	}
	if v := os.Getenv("JITLLM_A64_PF"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			o = append(o, nn.WithA64Prefetch(n))
		}
	}
	if v := os.Getenv("JITLLM_TILE"); v != "" {
		o = append(o, nn.WithGEMMTile(v))
	}
	if v := os.Getenv("JITLLM_CHUNK"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			o = append(o, nn.WithPrefillChunk(n))
		}
	}
	if v := os.Getenv("JITLLM_PREFILL_CORESET"); v != "" {
		o = append(o, nn.WithPrefillCoreSet(v))
	}
	if v := os.Getenv("JITLLM_PART"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			o = append(o, nn.WithParticipants(n))
		}
	}
	if v := os.Getenv("JITLLM_ACCS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			o = append(o, nn.WithAccs(n))
		}
	}
	if v := os.Getenv("JITLLM_FUSED_KSLICES"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			o = append(o, nn.WithFusedKSlices(n))
		}
	}
	// JITLLM_MATVEC_PICK: -1 keeps the fixed rule, n > 0 pins the picker's arm n.
	if v := os.Getenv("JITLLM_MATVEC_PICK"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			o = append(o, nn.WithMatVecPick(n))
		}
	}
	if v := os.Getenv("JITLLM_MOE_BATCH"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			o = append(o, nn.WithMixtureBatch(n))
		}
	}
	if v := os.Getenv("JITLLM_MVPICK"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			o = append(o, nn.WithMatVecPick(n))
		}
	}
	if v := os.Getenv("JITLLM_FUSED_KSPLIT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			o = append(o, nn.WithFusedKSplit(n))
		}
	}
	if os.Getenv("JITLLM_GEMM_EXACT") == "1" {
		o = append(o, nn.WithGEMMExact(true))
	}
	if v := os.Getenv("JITLLM_GEMM_ROWS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			o = append(o, nn.WithGEMMRows(n))
		}
	}
	if v := os.Getenv("JITLLM_GEMM_TOK"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			o = append(o, nn.WithGEMMTokens(n))
		}
	}
	if v := os.Getenv("JITLLM_TILE_TOK"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			o = append(o, nn.WithTileTokens(n))
		}
	}
	if v := os.Getenv("JITLLM_CHUNK_BYTES"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			o = append(o, nn.WithChunkBytes(n))
		}
	}
	if os.Getenv("JITLLM_NO_GEMM") != "" {
		o = append(o, nn.WithoutGEMM(true))
	}
	cpuConfigure()
	return o
}

// cpuConfigure hands jit/cpu the process-wide knobs that are not per-JIT: a
// perf map for the profiler, and the probes that deliberately slow a kernel so
// one class of operation can be priced against another.
func cpuConfigure() {
	n := func(k string) int {
		v, _ := strconv.Atoi(os.Getenv(k))
		return v
	}
	cpu.Configure(cpu.Config{
		PerfMap:     os.Getenv("JITLLM_PERFMAP") != "",
		ScalarProbe: n("JITLLM_SCALAR_PROBE"),
		LoadProbe:   n("JITLLM_LOAD_PROBE"),
		VectorProbe: n("JITLLM_VECTOR_PROBE"),
	})
}

// benchCmd parses bench's flags. It takes -tokenizer for the same reason run
// does: the prompt has to become ids before anything can be measured.
func benchCmd(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	tokJSON := fs.String("tokenizer", "", "a HuggingFace tokenizer.json whose pre-tokenizer replaces the one tokenizer.ggml.pre names")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := misplacedFlag(fs); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: jitllm bench [-tokenizer FILE] <model.jlm>")
	}
	topts, closeTok, err := tokenizerOption(*tokJSON)
	if err != nil {
		return err
	}
	defer closeTok()
	return benchmark(fs.Arg(0), topts)
}

// tokenizeCmd parses tokenize's flags. With -tokenizer it shows how a caller's
// pre-tokenizer diverges from the container's own.
func tokenizeCmd(args []string) error {
	fs := flag.NewFlagSet("tokenize", flag.ExitOnError)
	tokJSON := fs.String("tokenizer", "", "a HuggingFace tokenizer.json whose pre-tokenizer replaces the one tokenizer.ggml.pre names")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := misplacedFlag(fs); err != nil {
		return err
	}
	if fs.NArg() < 2 {
		usage()
	}
	topts, closeTok, err := tokenizerOption(*tokJSON)
	if err != nil {
		return err
	}
	defer closeTok()
	return tokenize(fs.Arg(0), strings.Join(fs.Args()[1:], " "), topts)
}

// prefillImage prefills a prompt that holds a picture. The picture is rows of
// the prompt (model.Picture): the prefill runs the State's vision segment over
// it -- the tower's blocks through the one runner and placement -- or the
// model's image cache answers, and with cached set the prefix cache restores
// past it, since its rows are named by the picture itself.
func prefillImage(st *model.State, m *model.Model, tw *model.Tower, pic *picture, spans []model.Span,
	cached bool) ([]float32, error) {
	if err := pic.preprocess(st); err != nil {
		return nil, err
	}
	c := tw.Cfg
	if pic.tiles != nil {
		fmt.Fprintf(os.Stderr, "vision   %s: a %dx%d canvas of tiles and %d piece(s) -> %d x %d embeddings of %d\n",
			c.Projector, pic.tiles[1], pic.tiles[0], len(pic.pieces), len(pic.pieces), c.Tokens(), c.ProjDim)
	} else if pic.lfm2 != nil {
		rows := 0
		for _, pc := range pic.pieces {
			rows += pc.Rows
		}
		fmt.Fprintf(os.Stderr, "vision   %s: %dx%d tiles and %d piece(s) -> %d embeddings of %d\n",
			c.Projector, pic.lfm2[0], pic.lfm2[1], len(pic.pieces), rows, c.ProjDim)
	} else if pic.lay != nil {
		fmt.Fprintf(os.Stderr, "vision   %s: %dx%d picture, an overview and %d slice(s) (%dx%d) -> "+
			"%d x %d embeddings of %d\n", c.Projector, pic.lay.W, pic.lay.H, len(pic.lay.Pieces)-1,
			pic.lay.Cols, pic.lay.Rows, len(pic.lay.Pieces), c.Tokens(), c.ProjDim)
	} else {
		p := pic.pieces[0]
		fmt.Fprintf(os.Stderr, "vision   %s: a %dx%d patch grid -> %d embeddings of %d\n",
			c.Projector, p.GW, p.GH, p.Rows, c.ProjDim)
	}
	t0 := time.Now()
	h0, _ := m.ImageCacheStats()
	var lg []float32
	var err error
	if cached {
		lg, err = st.PrefillCachedMixed(spans...)
	} else {
		lg, err = st.PrefillMixed(spans...)
	}
	if h1, _ := m.ImageCacheStats(); err == nil && h1 > h0 {
		fmt.Fprintf(os.Stderr, "vision   the image cache held the picture: no tower block ran\n")
	}
	if err == nil {
		fmt.Fprintf(os.Stderr, "vision   prompt with the picture prefilled in %.0f ms\n",
			float64(time.Since(t0).Microseconds())/1000)
	}
	return lg, err
}

// setPageBudget derives how many block pages fit in HOST memory and hands the
// container that number.
//
// Pages are derived from the memory budget, not dialled: the count depends on
// the page size, which devices are in play, how many blocks they took, and what
// an integrated device carved out of the same RAM. A budget that holds every
// host block is the fully resident configuration; paging simply never fires.
func setPageBudget(m *model.Model, st *model.State, dev nn.Device, maxmem uint64) {
	page := m.PageSize()
	if page == 0 {
		return // not a container; nothing is paged
	}
	// What the host may spend on WEIGHTS: the budget, less the KV cache this
	// run will commit, less the dense weights, which never page. The vision
	// tower needs no term of its own: its pages are pages of this container and
	// come out of this budget like any other. What a device living in host
	// memory holds is not subtracted here: the model takes it off itself, and
	// again whenever placement moves it (Model.SetPageBudget).
	avail := maxmem
	kv := uint64(st.MaxSeq()) * uint64(m.Cfg.KVDim()) * 4 * 2 * uint64(m.Cfg.NLayer)
	if kv < avail {
		avail -= kv
	} else {
		avail = 0
	}
	if d := m.DenseBytes(); d < avail {
		avail -= d
	} else {
		avail = 0
	}
	host := m.Cfg.NLayer - st.GPULayers()
	if host <= 0 {
		return // every block is on a device; nothing faults on the host
	}
	// The pager reads a zero budget as UNLIMITED, so a budget the dense
	// weights and the KV have already used up is one byte -- one page at a
	// time -- and not zero, which would hold every block of a model that is
	// over budget.
	avail = max(avail, 1)
	// A pool an earlier call narrowed is widened again; only a model that was
	// never capped can return early. The decision is taken on what the pager
	// would be handed, the device's holding off.
	net := max(avail-min(avail, hostReserved(dev)), 1)
	frames := int(net / page)
	pool, _, _ := m.PageStats()
	if frames >= host && pool == 0 {
		return // never budgeted, and the budget holds every host block
	}
	m.SetPageBudget(avail)
	// What the budget holds, not what is resident now: SetPageBudget's count
	// is the pages already read, which right after placement is the few the
	// devices left behind, and printed as a capacity it reported page-ins on
	// a model whose host blocks all fit.
	net = m.PageBudget()
	fit := int(net / page)
	if fit >= host {
		return // widened back to holding every host block: nothing will evict
	}
	fmt.Fprintf(os.Stderr, "pages    %d of %d host block(s) resident at once (%.1f MiB each, "+
		"%.2f GiB of host weight budget) -- %d page-in(s) per token",
		fit, host, float64(page)/(1<<20), float64(net)/(1<<30), host-fit)
	// Print what the collector is left with: a live heap at its goal looks
	// exactly like work from outside.
	if u := sched.GCUsable(); u > net {
		fmt.Fprintf(os.Stderr, ", %.2f GiB left for the Go heap", float64(u-net)/(1<<30))
	}
	fmt.Fprintln(os.Stderr)
}

// misplacedFlag refuses a flag that arrives after the first positional. Go's
// flag package stops parsing at the first non-flag argument, so
// `run model.jlm -chat "prompt"` would put "-chat" into the prompt and still
// produce a plausible answer.
//
// It matches against the flags this SET defines rather than against a leading
// dash, so a prompt that genuinely starts with "-" still runs; only a name the
// user plainly meant as a flag is refused.
func misplacedFlag(fs *flag.FlagSet) error {
	rest := fs.Args()
	if len(rest) < 2 {
		return nil
	}
	for _, a := range rest[1:] {
		name := strings.TrimLeft(a, "-")
		if name == a {
			continue
		}
		if i := strings.IndexByte(name, '='); i >= 0 {
			name = name[:i]
		}
		if fs.Lookup(name) == nil {
			continue
		}
		return fmt.Errorf("-%s came after the file, where Go's flag parser stops looking:\n"+
			"  it became part of the prompt rather than a flag, and the run would have\n"+
			"  produced a plausible answer from a configuration you did not select.\n"+
			"  Put every flag BEFORE the file: %s -%s ... <file> \"prompt\"", name, fs.Name(), name)
	}
	return nil
}

// withDeviceErr adds the tier's own reason (for example a CUDA error code) to an
// error a device run returned.
func withDeviceErr(err error, dev nn.Device) error {
	if g, ok := dev.(*tier.GPU); ok {
		if why := g.Err(); why != "" {
			return fmt.Errorf("%w (the device said: %s)", err, why)
		}
	}
	return err
}

// rateOf is n tokens over d as the run line prints it. A clock that moves in
// ticks (Windows) can read 0 for a short run, and n/0 printed +Inf tok/s: a
// rate no harness should read as a measurement. Such a run says so instead.
func rateOf(n int, d time.Duration) string {
	if d <= 0 {
		return "rate under the clock's resolution"
	}
	return fmt.Sprintf("%.2f tok/s", float64(n)/d.Seconds())
}
