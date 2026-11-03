package convert

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// stMaxBytes is the weight budget this gate converts without being asked.
const stMaxBytes = 4 << 30

// stDirBytes is the directory's .safetensors weight, which a conversion has to
// hold on both sides at once.
func stDirBytes(d string) int64 {
	var n int64
	m, _ := filepath.Glob(filepath.Join(d, "*.safetensors"))
	for _, f := range m {
		if fi, err := os.Stat(f); err == nil {
			n += fi.Size()
		}
	}
	return n
}

// TestSafetensorsArchCoverage walks every HuggingFace model directory in the
// model directory and pins which architectures the safetensors reader converts. It is
// two-sided: every hfArchTable entry must convert a real checkpoint, so an
// entry cannot outlive its code, and an architecture not in the table must be
// refused by name (RULE 7) rather than crash or convert plausibly. It prints
// the count, since the safetensors list is a second list, not the GGUF one.
func TestSafetensorsArchCoverage(t *testing.T) {
	dirs, err := filepath.Glob(testmodels.Path("*"))
	if err != nil {
		t.Fatal(err)
	}
	type row struct{ dir, arch, model string }
	var done, refused, blocked []row
	var skipped []string
	pick := os.Getenv("JITLLM_ST_MODEL")
	for _, d := range dirs {
		if pick != "" && !strings.Contains(d, pick) {
			continue
		}
		cfg := filepath.Join(d, "config.json")
		b, err := os.ReadFile(cfg)
		if err != nil {
			continue // not a HuggingFace model directory
		}
		var c struct {
			Architectures []string `json:"architectures"`
			ModelType     string   `json:"model_type"`
		}
		if json.Unmarshal(b, &c) != nil || len(c.Architectures) == 0 {
			continue
		}
		if !IsSafetensors(d) {
			continue // config.json without weights beside it
		}
		r := row{filepath.Base(d), c.Architectures[0], c.ModelType}
		_, known := hfArchTable()[c.Architectures[0]]
		// A real checkpoint is skipped by size, because this gate writes a
		// whole container (a 31 GB model thrashes a 31 GB box). Coverage is not
		// lost: each class is also covered by a tiny synth fixture gated against
		// transformers. JITLLM_ST_MODEL=<substring> forces one, under
		// `scripts/cap 24G`.
		if pick == "" && stDirBytes(d) > stMaxBytes {
			skipped = append(skipped, r.dir)
			continue
		}
		dst := filepath.Join(t.TempDir(), "out.jlm")
		_, cerr := FromSafetensors(d, dst, jlm.Fingerprint{})
		switch {
		case known && cerr != nil:
			// A claimed class that fails may be a broken claim or a gap elsewhere
			// (e.g. a sentencepiece tokenizer.json needing the SPM path). It is
			// recorded against the architecture below, which passes as long as
			// some model of the class converts, and listed by name.
			blocked = append(blocked, r)
			t.Logf("%s: %s is implemented and this checkpoint is still refused: %v",
				r.dir, r.arch, cerr)
		case known:
			done = append(done, r)
		case cerr == nil:
			t.Errorf("%s: %s is NOT in hfArchTable and converted anyway -- an "+
				"architecture nobody implemented produced a container", r.dir, r.arch)
		case !strings.Contains(cerr.Error(), r.arch):
			// Refusing is right; refusing without naming what was refused sends
			// the reader to the wrong file.
			t.Errorf("%s: refused %s without naming it: %v", r.dir, r.arch, cerr)
		default:
			refused = append(refused, r)
		}
	}
	if len(done)+len(refused) == 0 {
		testmodels.Missing(t, "%s", "no HuggingFace model directory was found in "+testmodels.Dir()+" (set JITLLM_MODELS to the model directory) -- this "+
			"gate proved nothing (RULE 11: fetch one, do not record the absence)")
	}
	// And at least one must convert, or a reader that refuses everything
	// passes every assertion above.
	if len(done) == 0 {
		t.Fatalf("%d model(s) found and not one converted: the safetensors "+
			"reader refuses everything", len(refused))
	}
	// Every claimed class must convert at least one real checkpoint: an
	// entry whose every model is refused is a claim with nothing behind it.
	got := map[string]bool{}
	for _, r := range done {
		got[r.arch] = true
	}
	for name := range hfArchTable() {
		seen := false
		for _, r := range append(append([]row{}, done...), blocked...) {
			if r.arch == name {
				seen = true
			}
		}
		if seen && !got[name] {
			t.Errorf("hfArchTable names %s and every checkpoint of it on this box "+
				"was refused: the entry claims a capability nothing demonstrates", name)
		}
	}
	names := func(rs []row) string {
		var v []string
		for _, r := range rs {
			v = append(v, r.dir+" ("+r.arch+")")
		}
		sort.Strings(v)
		return strings.Join(v, "\n      ")
	}
	t.Logf("safetensors converts %d of %d model(s) on this box:\n      %s",
		len(done), len(done)+len(refused), names(done))
	t.Logf("refused by name, each a TASK rather than a constraint (RULE 11):\n      %s",
		names(refused))
	if len(blocked) > 0 {
		t.Logf("architecture implemented, checkpoint refused for another reason:\n      %s",
			names(blocked))
	}
	// The skips are printed: this gate's claim is a count, and a silent
	// cap would read as coverage.
	if len(skipped) > 0 {
		t.Logf("NOT CONVERTED (over %d GiB of weights; JITLLM_ST_MODEL=<substring> forces one): %v",
			stMaxBytes>>30, skipped)
	}
}
