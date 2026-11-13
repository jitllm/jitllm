package srcgate

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestNoRefusalButInputsAndLimits holds the inference packages -- engine,
// model, nn, composer, graph, format, sched, backend -- to the refusals
// below, each an input check, a limit of a device's instructions, or a
// reference's own refusal matched: a model the engine runs is composed and
// run, never refused for a feature (the goal's second check). A refusal
// added fails the gate; one gone leaves the list with it. The violation, a
// refusal the list does not hold, is seen.
func TestNoRefusalButInputsAndLimits(t *testing.T) {
	testmodels.SourceTree(t)
	allowed := map[string]string{
		"backend/view.go:views are not implemented on its backend": "a buffer type no backend makes: CUDA, Vulkan and Metal bind views",
		"composer/amdgpu/lower.go:are not built":                   "the AMD lowering states what its instructions express; the composer forms no tile op it refuses",
		"composer/msl/tile.go:a half accumulator is refused":       "Metal's lowering states its limit; the composer forms no half accumulator there",
		"composer/ptx/tile.go:is not built":                        "sm's instructions: a tile form a device does not lower is not formed for it",
		"composer/swrmix.go:ffmpeg's rematrix refuses it":          "the reference's refusal, matched (the user's decision: the resampler is ffmpeg's)",
		"format/quant/dequant.go:not implemented yet":              "a type off quant.Dequantable, which the converter refuses",
		"model/model.go:which the converter refuses":               "a container the converter does not write",

		"engine/joint.go:states no joint head":                       "a joint readout asked of a container that has none: the input's check",
		"engine/decide.go:states no prompt":                          "a container written before marker prompts were stated: convert it again",
		"engine/decide.go:states no label readout":                   "a decision asked of a container with no label readout: the input's check",
		"model/toolgrammar.go:states no tool-call form":              "a forced tool call asked of a model with no tool-call form: the input's check",
		"model/toolgrammar.go:states no parameter":                   "a tool form written wrong: the container's check",
		"model/toolsyntax.go:states no tool-call form":               "a container written before tool forms were stated: convert it again",
		"model/video.go:reads no video":                              "a video given a model whose reference processor takes none (convert.TestEveryVideoProcessorIsStated holds every one that does stated): the input's check",
		"model/tower.go:states no picture geometry":                  "a container written before a picture's geometry was stated: convert it again",
		"model/picture.go:states no picture geometry":                "a picture's squares asked of a tower that states no geometry, which Open refuses first: the input's check",
		"model/video.go:states no geometry of its frames":            "a video's frames sized from a record that states none, which the container's check refuses first: the input's check",
		"model/video.go:which the video's resize refuses":            "the reference processor's own refusal (frames past an aspect of 200, fewer than its temporal patch), stated as frames of no square: the input's check",
		"format/jlm/video.go:states no geometry of its frames":       "a video record written wrong: the container's check",
		"format/jlm/toolform.go:tool form":                           "a tool form the container states wrong: the container's check",
		"format/jlm/tower.go:states no expression or no width":       "a tower record written wrong: the container's check",
		"graph/verify.go:states no rows":                             "a graph that does not verify: the builder's check",
		"model/front.go:the front end states no graph":               "a block of a container the converter did not write: the census runs every one it writes",
		"engine/cpuserve.go:the front end states no graph":           "a block of a container the converter did not write: the census runs every one it writes",
		"composer/front.go:the model states no head or input rows":   "a model the front end states nothing of: the census runs every one the converter writes",
		"composer/decompose_history.go:the model states no table of": "a rotary no stated rotary holds: the front end states each one its blocks read",
		"engine/device/gpu/served.go:the plan states no graph":       "the plan states a graph for each rotary the model states (nn.PartsPlan)",
		"engine/device/gpu/rows.go:the composer states no sample":    "the composer states a sample for each vocabulary head it forms (composer.Front)",
		"engine/device/gpu/persist.go:states no co-residency":        "a device's limit: only a CUDA cooperative launch guarantees co-residency (RULE 13)",
		"backend/grid.go:states no co-residency limit":               "a device's limit: no grid barrier without one (RULE 13)",
		"backend/metal.go:states no gpu-core-count":                  "a device's limit: no core count, no co-residency (RULE 13)",
		"backend/vulkan.go:states no compute-unit count":             "a device's limit: no compute-unit count, no co-residency (RULE 13)",

		"engine/decide.go:too many options":                           "a question past its label set's own bound (the converter states each set's MaxOptions): the input's check",
		"engine/forward.go:cannot place it by name":                   "a placement naming a card on an instance whose device names none (the card tier names every device it holds): the caller's check",
		"engine/forward.go:cannot place a block by name":              "a placement or a move naming a card on an instance whose device names none: the caller's check",
		"engine/forward.go:cannot be continued on the CPU":            "a device that failed mid-sequence took the history it held: a hardware fault, reported by name",
		"engine/seamtune.go:too wide to decide":                       "the tuner's measurement gate (RULE 2): a split too noisy to decide keeps the placement it has",
		"engine/listed.go:this device cannot page a block":            "cards listed without the CPU on a device that names none (the card tier pages on itself, TestBlocksPageOnTheListedCard): the caller's check",
		"engine/device/cpu/baseline_amd64.go:cannot run the engine":   "the amd64 floor, SSE4.1 and SSSE3 (RULE 8, nn.HostRefusal): the host's instructions",
		"engine/device/cpu/baseline.go:cannot run here":               "a tier asked of a host whose instructions lack it (cpu:sse and its kin): the host's instructions",
		"engine/device/cpudev/tower.go:has no form for class":         "the tower forms an entry for every class its geometry states and every sample format at placement: an index past them is the caller's check",
		"engine/device/gpu/served.go:a pass cannot run again over":    "RULE 13: a kernel reading what it writes in one phase is the composer's contract broken, refused before it runs wrong",
		"composer/pass_liverows.go:is refused: it combines across":    "a kernel held to a call's rows by its launch alone whose surplus row would read the history: the composer's contract broken, refused before it runs wrong (no pass that joins kernels makes one: Fuse and Epilogue ask it first; the census on CUDA and Vulkan forms none)",
		"engine/device/gpu/open.go:cannot be listed beside":           "a device list naming a set and a member of it: the input's check",
		"engine/device/gpu/layer.go:which has no device kernel":       "a weight type no container carries (off quant.Dequantable, or TQ1_0, which the converter repacks): the container's check",
		"nn/device.go:cannot move between the devices directly":       "a mover's answer for a move it does not make itself: the caller carries the history through the host",
		"composer/form.go:cannot form":                                "names the part whose composition failed; the failure is the wrapped error's, under its own phrase",
		"composer/compose.go:jit: no lowering for":                    "a device kind the registry holds no lowering for: every device the backends open has one",
		"composer/msl/msl.go:an ldmatrix load":                        "NVIDIA's ldmatrix lane contract, formed only where the profile lowers it",
		"composer/msl/msl.go:float only":                              "Metal's simdgroup_matrix types are half, bfloat and float: the composer forms no integer matrix op for Metal",
		"composer/msl/msl.go:no lowering for %s":                      "an op past the lowering's switch: every op the composer forms for Metal is lowered (the census on the M4)",
		"composer/spirv/lower.go:an ldmatrix load":                    "NVIDIA's ldmatrix lane contract, formed only where the profile lowers it",
		"composer/spirv/lower.go:no lowering for %s":                  "an op past the lowering's switch: every op the composer forms for Vulkan is lowered (the census on Vulkan)",
		"composer/ptx/lower.go:no lowering for %s":                    "an op past the lowering's switch: every op the composer forms for CUDA is lowered (the census on CUDA)",
		"composer/amdgpu/lower.go:an ldmatrix load":                   "NVIDIA's ldmatrix lane contract, formed only where the profile lowers it",
		"composer/amdgpu/lower.go:32-lane mma.sync":                   "NVIDIA's mma.sync fragment layout, which neither MFMA nor WMMA has: formed only for sm_70 and up",
		"composer/amdgpu/lower.go:yet: the matrix units":              "the AMD lowering states what its instructions express; the composer forms no tile op it refuses",
		"composer/amdgpu/lower.go:amdgpu: no lowering for %s":         "an op past the lowering's switch: the AMD profile forms only what it lowers",
		"composer/lowera64_node.go:has no form of without i8mm":       "the composer forms an unsigned byte dot only under Int8Form.UnsignedDot's instruction sets, which no arm64 profile states",
		"format/layout/pack.go:cannot hold as a normal scale":         "an MXFP4 scale below f32's normal range on nonzero codes, which a device flushes to zero: the container's check",
		"format/jlm/write.go:cannot state":                            "a container version with no field for it: the writer's check",
		"format/jlm/migrate.go:which v31 cannot recover":              "a container older than the migration reads: convert it again",
		"backend/backend.go:cannot be assumed":                        "a device that promises no subgroup width: its own statement, and the kernel asked has a scalar twin",
		"backend/cuda/api.go:cannot map host memory":                  "a device without host mapping (CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY): the device's limit",
		"backend/cuda.go:cannot be asked for":                         "a CUDA warp is 32 lanes: the hardware's width",
		"backend/hip.go:cannot be asked for":                          "amdgcn's wave: the hardware's width",
		"backend/metal.go:cannot be asked for":                        "Metal's SIMD-group width: the hardware's",
		"backend/vulkan.go:seam tuner cannot place blocks":            "a default, not a refusal: a software Vulkan device runs on the host's own cores, and naming it takes it",
		"backend/metal/metal_darwin.go:is not available (macOS 15+)":  "an OS API before macOS 15: the platform's",
		"backend/metal/metal_other.go:not available on this platform": "Metal outside Apple's platforms",
		"backend/vulkan/device.go:does not support":                   "a driver without VK_EXT_memory_budget: free memory unknown and reported, nothing refused",
		"backend/cuda/cuda.go:cannot record an event node":            "a driver without cuEventRecordWithFlags: the driver's",
		"backend/grid.go:cannot bound its residency":                  "a device's limit: no grid barrier without a co-residency bound (RULE 13)",
		"backend/cuda/funcattr.go:cannot ask for":                     "a driver without cuFuncSetAttribute: no dynamic shared memory past the static 48 KB (the driver's)",
		"backend/vulkan/run.go:cannot guarantee a":                    "a device that cannot pin a subgroup width (VK_EXT_subgroup_size_control): the device's limit",
		"backend/vulkan/run.go:host memory cannot be imported":        "a driver without VK_EXT_external_memory_host: the driver's",
		"backend/vulkan/run.go:cannot import %d bytes":                "the driver's answer to a host import",
		"backend/vulkan/subgroup.go:subgroup shuffles":                "a device without subgroup shuffles: the device's limit",
		"backend/vulkan/subgroup.go:lanes and cannot be":              "a device whose subgroup width varies: the device's limit",
		"backend/vulkan/subgroup.go:cannot be promised":               "a device whose subgroup width varies: the device's limit",
		"backend/vulkan/subgroup.go:cannot be pinned":                 "a device whose subgroup width varies: the device's limit",

		"engine/audiorun.go:past the sound op's":                              "a clip whose resampler runs past the taps the sound op is formed for (a rate past 10 MHz, or 64 channels past 1.5 MHz): the input's check",
		"model/tower.go:past the sound op's":                                  "a clip whose resampler runs past the taps the sound op is formed for (a rate past 10 MHz, or 64 channels past 1.5 MHz): the input's check",
		"engine/device/cpudev/form.go:past the argument block":                "a code's word past its own argument block: the lowering broken, an invariant",
		"engine/device/gpu/picrows.go:live rows past the":                     "a picture's live rows past its form's, which the tower forms for every class at placement: an invariant",
		"engine/device/gpu/served.go:more than the block's":                   "a call's experts past the block's slots, which pagedCall holds the call to: an invariant",
		"engine/device/gpu/layer.go:wanted rows past the":                     "rows past the logits the step's staging was built for: an invariant",
		"engine/device/gpu/rows.go:more than the batch width":                 "a step's rows past the batch width, which the router chunks it to: an invariant",
		"engine/device/gpu/rows.go:more than the reserved width":              "a step's rows past the reserved width, which its caller chunks it to: an invariant",
		"model/ngram.go:reads past the table's":                               "an n-gram head past its table's rows: the container's check",
		"model/picture.go:past its %d pixels":                                 "a resize stage past the pixels its processor states: the input's check",
		"model/tower.go:its samples' U32 offsets reach":                       "a picture past 4 GiB, the reach of its samples' 32-bit offsets: the index width",
		"model/video.go:past its table of":                                    "a video value past its stated table: the container's check",
		"model/video.go:past the %d the picture table holds":                  "a video pair past 4 GiB, the reach of the picture table's 32-bit offsets: the index width",
		"composer/generate_scan.go:row store reads its state past its update": "a scan read out of order: the lowering broken, an invariant",
		"composer/generate_scan.go:sum reads its state past its update":       "a scan read out of order: the lowering broken, an invariant",
		"composer/lowergpu.go:past the device's":                              "a kernel's workgroup memory past the device's: the device's limit",
		"format/jlm/format.go:runs past the":                                  "a container span past its file: the container's check",
		"format/jlm/format.go:run past the":                                   "a container's pages past its file: the container's check",
		"format/jlm/tower.go:past its pictures'":                              "a tower whose video frames outgrow its pictures: the container's check",
		"format/jlm/tower.go:past its last":                                   "a tower step's value past its last: the container's check",
		"format/jlm/read.go:past the %d-byte file":                            "a container span past its file: the container's check",
		"format/jlm/read.go:run past the %d pages":                            "an expert page's sheets past the container's pages: the container's check",
		"format/jlm/codec.go:is past the":                                     "a container span past its file: the container's check",

		"engine/embedrun.go:this model embeds at most":                      "an embedding's input past the positions the model reaches (its context): the input's check, as llama-server refuses one past its batch",
		"engine/towerrun.go:past what the tower's largest class holds":      "a resized picture past the classes the tower forms from its stated geometry, which the resize is held to: an invariant",
		"engine/towerrun.go:on a canvas of":                                 "a picture's canvas past its class, which the same resize holds it to: an invariant",
		"engine/devfail.go:has no room for the call's history now":          "a device's memory: a call's history past the card's free pages after the instance moved blocks to the CPU until it fit (reserveMoving), so only where nothing can move; the server parks the row while other rows hold pages (parkForRoom)",
		"engine/session.go:has no room for this session's history":          "a device's memory: a session's history of a block coming back to a full card, which roomFor meets by moving blocks to the CPU until it settles; it reaches a caller only where the instance may not relocate (SetRelocate) or nothing can move",
		"engine/device/gpu/served.go:have no room on the card":              "a device's memory at placement: the head's input-row tables stay on the host with the head, the one decline a placement makes (RULE 8)",
		"engine/device/gpu/layer.go:have no room beside the head":           "a device's memory at placement: the head and its input-row tables stay on the host, the one decline a placement makes (RULE 8)",
		"engine/device/gpu/recpool.go:no room for the kept zeros":           "fault injection's own refusal (roomFaulted, under the jitllmfault tag): never in a release build",
		"engine/device/gpu/recpool.go:no room for a copy of a marked state": "a device's memory: the mark keeps no copy, and a rewind that needs it says it cannot rewind (rewindable)",
		"engine/device/gpu/kvpool.go:no room for another page":              "a device's memory: a layer's attention pool past the card's budget, the backend's buffer limit or the index width, after eviction and streaming (docs/design/device-kv-paging.md)",
		"model/model.go:wide over at most":                                  "a per-layer embedding table written wrong: the container's check",
		"composer/a64/a64.go:at most %d":                                    "an offset past what its instruction encodes: the ISA's limit",
		"graph/verify.go:at most a bias":                                    "a graph that does not verify: the builder's check",
		"graph/verify.go:a picture of at most":                              "a graph that does not verify: the builder's check",
		"format/jlm/video.go:at most %d frames":                             "a video record written wrong: the container's check",
		"backend/vulkan/run.go:so a launch holds at most":                   "a dispatch past vkCmdDispatch's 32-bit workgroup count: the API's limit",
		"backend/metal/groups.go:so a dispatch holds at most":               "a dispatch past the 32-bit threadgroup count: the API's limit",
		"backend/cuda.go:launches at most":                                  "a block past the threads its registers allow (CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK): the device's limit",
		"backend/cuda/api.go:takes at most %d along x":                      "a grid past the device's x dimension (CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X): the device's limit",
	}
	// phrase is a string literal that says the code refuses: something
	// refused, not implemented, not supported, not built or not available,
	// a model, a container or a device that reads or states no such thing,
	// cannot or has no form or lowering of it, or has too many of it, or
	// more than, past or at most a count it holds, or has no room for it.
	phrase := regexp.MustCompile(`"[^"\n]*(?i:not implemented|is not supported|are not supported|not built|unsupported|is refused|refuses|\b(?:reads|states) no\b|cannot|can not|too many|too wide|no device kernel|no lowering|does not support|not pre-packed|has no form|no form of|not available|past the|past its|more than the|at most|no room)[^"\n]*"`)
	root := filepath.Join("..", "..")
	found := map[string]bool{}
	var files int
	for _, dir := range []string{"engine", "model", "nn", "composer", "graph", "format", "sched", "backend"} {
		err := filepath.WalkDir(filepath.Join(root, dir), func(p string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
				return nil
			}
			b, err := os.ReadFile(p)
			if err != nil {
				return err
			}
			files++
			rel, _ := filepath.Rel(root, p)
			rel = filepath.ToSlash(rel)
			for _, line := range strings.Split(string(b), "\n") {
				if strings.HasPrefix(strings.TrimSpace(line), "//") {
					continue
				}
				for _, m := range phrase.FindAllString(line, -1) {
					key, ok := allowedKey(allowed, rel, m)
					if !ok {
						t.Errorf("%s refuses: %s -- implement it, or name the input check or device limit it is", rel, m)
						continue
					}
					found[key] = true
				}
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if files < 100 {
		t.Fatalf("read %d files: the gate read nothing", files)
	}
	for k := range allowed {
		if !found[k] {
			t.Errorf("%s no longer refuses: take it off the list", k)
		}
	}
	t.Logf("%d files, %d refusals, each an input check or a limit", files, len(found))
	for _, v := range []string{`"model: a sliding window is not supported"`, `"model: the block reads no window"`, `"model: the tower states no grid"`,
		`"tier: block 3 cannot move to cuda:1"`, `"tier: 70 kernel arguments is too many"`, `"tier: its bank is not pre-packed"`,
		`"a graph of 9 inputs and 1 outputs, past the 8 places the CPU's call site binds"`,
		`"model: a call of 9 experts, and a block binds at most 8"`, `"tier: no room for block 3's prompt scratch"`} {
		if _, ok := allowedKey(allowed, "engine/forward.go", v); ok || !phrase.MatchString(v) {
			t.Errorf("the violation %s, a refusal the list does not hold, was not seen", v)
		}
	}
}

// allowedKey is the list's entry for refusal m in file rel: its file and a
// fragment the literal holds.
func allowedKey(allowed map[string]string, rel, m string) (string, bool) {
	for k := range allowed {
		f, frag, _ := strings.Cut(k, ":")
		if f == rel && strings.Contains(m, frag) {
			return k, true
		}
	}
	return "", false
}
