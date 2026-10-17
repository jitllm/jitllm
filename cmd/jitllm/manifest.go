package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// runManifest is what `jitllm speed -manifest` records about a run: the
// binary, the artefact, the configuration and workload, where the blocks ran,
// and which recipe each site ran and why (tier.RecipeChoice). It is the first
// stage of docs/design/optimization-agent-harness.md: a row that names the
// variant it measured, and a run that can be repeated from the file with
// `jitllm speed -from` and checked to have selected the same variant.
type runManifest struct {
	Version  int                 `json:"version"`
	Binary   manifestBinary      `json:"binary"`
	Host     manifestHost        `json:"host"`
	Model    manifestModel       `json:"model"`
	Args     []string            `json:"args"` // speed's arguments, the model path last
	Env      map[string]string   `json:"env"`  // every JITLLM_* variable set for the run
	Workload manifestWorkload    `json:"workload"`
	Placed   manifestPlacement   `json:"placement"`
	Recipes  []tier.RecipeChoice `json:"recipes,omitempty"`
	// Ran counts the recipes' launches over the run (tier.Stats): a recipe
	// chosen and never run is a RULE 10 failure a reader can see.
	Ran map[string]int `json:"ran,omitempty"`
	// Blocks is each text block's path in the timed run: through the block
	// graph or the hand-written body (with its named reason), on the host or
	// the device, and every rewrite matched on its graph, legal or refused
	// with its reason. A block that fell back is visible here by name.
	Blocks  []manifestBlock `json:"blocks,omitempty"`
	Results manifestResults `json:"results"`
}

type manifestBinary struct {
	Revision string `json:"revision"`
	Modified bool   `json:"modified"`
	Go       string `json:"go"`
}

type manifestHost struct {
	OS     string `json:"os"`
	Arch   string `json:"arch"`
	CPUs   int    `json:"cpus"`
	Device string `json:"device,omitempty"`
}

type manifestModel struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
	// HeadSHA256 hashes the container's first 4 MiB (its header, config and
	// tensor table): an identity for the file without reading its weights.
	HeadSHA256 string `json:"head_sha256"`
	Arch       string `json:"arch"`
	Layers     int    `json:"layers"`
	Embd       int    `json:"embd"`
	FFN        int    `json:"ffn"`
	Heads      int    `json:"heads"`
	KVHeads    int    `json:"kv_heads"`
}

type manifestWorkload struct {
	Prompt  int    `json:"prompt"`
	Gen     int    `json:"gen"`
	Reps    int    `json:"reps"`
	Devices string `json:"devices"`
	VRAM    uint64 `json:"vram"`
}

type manifestPlacement struct {
	PromptBlocks int      `json:"prompt_blocks"`
	DecodeBlocks int      `json:"decode_blocks"`
	Runs         [][2]int `json:"runs"`
}

type manifestBlock struct {
	Block    int               `json:"block"`
	Path     map[string]string `json:"path"` // phase ("prompt", "decode") -> "graph", "hand", "device-graph", "device-hand", mixed with "+"
	Why      string            `json:"uncovered,omitempty"`
	Rewrites []string          `json:"rewrites,omitempty"`
}

// paths records the State's per-block path for phase.
func (r *runManifest) paths(m *model.Model, phase string, ps []model.BlockPath) {
	if r.Blocks == nil {
		r.Blocks = make([]manifestBlock, len(ps))
		for li := range r.Blocks {
			b := &r.Blocks[li]
			b.Block, b.Path, b.Why = li, map[string]string{}, ps[li].Why
			for _, d := range m.BlockRewrites(li) {
				b.Rewrites = append(b.Rewrites, d.String())
			}
		}
	}
	for li, p := range ps {
		var on []string
		if p.Graph > 0 {
			on = append(on, "graph")
		}
		if p.Hand > 0 {
			on = append(on, "hand")
		}
		if p.Device > 0 && p.DeviceGraphed {
			on = append(on, "device-graph")
		} else if p.Device > 0 {
			on = append(on, "device-hand")
		}
		r.Blocks[li].Path[phase] = strings.Join(on, "+")
	}
}

type manifestResults struct {
	PP []float64 `json:"pp_tok_s"`
	TG []float64 `json:"tg_tok_s"`
}

func newManifest(args []string, m *model.Model, path string) (*runManifest, error) {
	r := &runManifest{Version: 1, Args: args, Env: map[string]string{}}
	r.Binary.Go = runtime.Version()
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			switch s.Key {
			case "vcs.revision":
				r.Binary.Revision = s.Value
			case "vcs.modified":
				r.Binary.Modified = s.Value == "true"
			}
		}
	}
	r.Host = manifestHost{OS: runtime.GOOS, Arch: runtime.GOARCH, CPUs: runtime.NumCPU()}
	for _, kv := range os.Environ() {
		if k, v, ok := strings.Cut(kv, "="); ok && strings.HasPrefix(k, "JITLLM_") {
			r.Env[k] = v
		}
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	h := sha256.New()
	if _, err := io.Copy(h, io.LimitReader(f, 4<<20)); err != nil {
		return nil, err
	}
	c := m.Cfg
	r.Model = manifestModel{Path: path, Size: st.Size(), HeadSHA256: hex.EncodeToString(h.Sum(nil)),
		Arch: c.Arch, Layers: c.NLayer, Embd: c.NEmbd, FFN: c.NFFN, Heads: c.NHead, KVHeads: c.NKVHead}
	return r, nil
}

// device records the plan explanation and the recipes' run counts of d.
func (r *runManifest) device(d nn.Device, name string) {
	r.Host.Device = name
	g, ok := d.(*tier.GPU)
	if !ok {
		return
	}
	r.Recipes = g.Explain()
	st := g.Stats()
	r.Ran = map[string]int{"gate2in": st.Gate2InRuns, "gate2": st.GateUp2Runs, "gu": st.GateUpFused,
		"ffn_keys_timed": st.FFNTimed, "ffn_keys_kept": st.FFNKept,
		"actin_keys_timed": st.ActInTimed, "actin_keys_kept": st.ActInKept}
}

func (r *runManifest) write(path string) error {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

func readManifest(path string) (*runManifest, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r runManifest
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, fmt.Errorf("manifest %s: %w", path, err)
	}
	if r.Version != 1 || len(r.Args) == 0 {
		return nil, fmt.Errorf("manifest %s: version %d with %d arguments is not one this binary reads", path, r.Version, len(r.Args))
	}
	return &r, nil
}

// replayEnv sets the recorded JITLLM_* variables and clears those the
// recorded run did not have, so the replay's options are the recorded ones.
func (r *runManifest) replayEnv() error {
	for _, kv := range os.Environ() {
		if k, _, ok := strings.Cut(kv, "="); ok && strings.HasPrefix(k, "JITLLM_") {
			if _, rec := r.Env[k]; !rec {
				if err := os.Unsetenv(k); err != nil {
					return err
				}
			}
		}
	}
	for k, v := range r.Env {
		if err := os.Setenv(k, v); err != nil {
			return err
		}
	}
	return nil
}

// variantDiff lists where run b selected a different variant from a: the
// artefact, the placement, and each site's recipe by device and key. Rates
// are not compared; they are the measurement, not the variant.
func variantDiff(a, b *runManifest) []string {
	var d []string
	if a.Model.HeadSHA256 != b.Model.HeadSHA256 || a.Model.Size != b.Model.Size {
		d = append(d, fmt.Sprintf("model: %s (%d bytes) against %s (%d bytes)", a.Model.HeadSHA256, a.Model.Size, b.Model.HeadSHA256, b.Model.Size))
	}
	if a.Binary.Revision != b.Binary.Revision || a.Binary.Modified != b.Binary.Modified {
		d = append(d, fmt.Sprintf("binary: %s (modified %v) against %s (modified %v)", a.Binary.Revision, a.Binary.Modified, b.Binary.Revision, b.Binary.Modified))
	}
	if a.Host.Device != b.Host.Device {
		d = append(d, fmt.Sprintf("device: %q against %q", a.Host.Device, b.Host.Device))
	}
	if a.Placed.PromptBlocks != b.Placed.PromptBlocks || a.Placed.DecodeBlocks != b.Placed.DecodeBlocks {
		d = append(d, fmt.Sprintf("placement: prompt %d decode %d blocks against prompt %d decode %d",
			a.Placed.PromptBlocks, a.Placed.DecodeBlocks, b.Placed.PromptBlocks, b.Placed.DecodeBlocks))
	}
	ra, rb := recipeMap(a), recipeMap(b)
	var keys []string
	for k := range ra {
		keys = append(keys, k)
	}
	for k := range rb {
		if _, ok := ra[k]; !ok {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	for _, k := range keys {
		x, okx := ra[k]
		y, oky := rb[k]
		switch {
		case !okx:
			d = append(d, fmt.Sprintf("recipe %s: absent against %s", k, y))
		case !oky:
			d = append(d, fmt.Sprintf("recipe %s: %s against absent", k, x))
		case x != y:
			d = append(d, fmt.Sprintf("recipe %s: %s against %s", k, x, y))
		}
	}
	return d
}

func recipeMap(r *runManifest) map[string]string {
	m := map[string]string{}
	for _, c := range r.Recipes {
		m[c.Device+" "+c.Site+" "+c.Key] = c.Recipe
	}
	return m
}
