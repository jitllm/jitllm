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
		"model/video.go:reads no video":                              "a video given a model whose processor takes none, or one of the gaps convert.TestEveryVideoProcessorIsStated names (SmolVLM, Gemma 4): work, not a limit",
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
	}
	// phrase is a string literal that says the code refuses: something
	// refused, not implemented, not supported or not built, or a model, a
	// container or a device that reads or states no such thing.
	phrase := regexp.MustCompile(`"[^"\n]*(?i:not implemented|is not supported|are not supported|not built|unsupported|is refused|refuses|\b(?:reads|states) no\b)[^"\n]*"`)
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
	for _, v := range []string{`"model: a sliding window is not supported"`, `"model: the block reads no window"`, `"model: the tower states no grid"`} {
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
