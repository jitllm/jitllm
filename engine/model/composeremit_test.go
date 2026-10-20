package model

import (
	"crypto/sha256"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/backend/recorder"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

var updateComposer = flag.Bool("update-composer", false, "rewrite testdata/composer_ir.golden")

// TestComposerEmitsEveryModel opens every fixture's blocks on a device that
// runs nothing and records what is compiled for it (recorder), prefills and
// steps once, and reads, per model, what the composer emitted: how many of the
// programs' ops ran kernels generated from themselves, how many did not, and a
// digest of every emitted kernel's IR. Against testdata/composer_ir.golden it
// fails where a model's emitted code changed (rewrite with -update-composer
// once the change is checked) or its generated ops fell. No GPU, no weights
// read for arithmetic: the composer's first check, in seconds.
func TestComposerEmitsEveryModel(t *testing.T) {
	paths, _ := filepath.Glob(filepath.Join(testmodels.Dir(), "synth-*.jlm"))
	if len(paths) == 0 {
		t.Fatalf("no synth-*.jlm under %s (set JITLLM_MODELS)", testmodels.Dir())
	}
	type row struct {
		gen, not int
		digest   string
	}
	got := map[string]row{}
	notBy := map[string]int{}
	for _, p := range paths {
		name := strings.TrimSuffix(filepath.Base(p), ".jlm")
		if strings.HasSuffix(name, "-vlm") || strings.Contains(name, "mmproj") {
			continue
		}
		m, err := Open(p, noTune)
		if err != nil {
			continue
		}
		func() {
			defer m.Close()
			if m.IsEncoder() {
				return
			}
			rec := recorder.New("ptx", 1<<40)
			g, err := tier.New([]tier.Slot{{Dev: rec, Bytes: 1 << 40}}, append(testTierOpts(t),
				tier.WithConfig(func(c *tier.Config) { c.NoGraph = true }))...)
			if err != nil {
				t.Logf("%s: %v", name, err)
				return
			}
			defer g.Close()
			s := m.NewState(32)
			defer s.Close()
			if err := s.SetDevice(g); err != nil || s.GPULayers() == 0 {
				return
			}
			lg, err := s.Prefill(m.Vocab.Encode("Hello", true))
			if err == nil {
				_, err = s.Forward(Greedy(lg))
			}
			r := row{}
			for _, br := range s.DeviceBlocks() {
				for li := br[0]; li < br[1]; li++ {
					prog, _ := g.BlockProgram(li)
					for _, line := range strings.Split(prog, "\n") {
						switch {
						case strings.Contains(line, "generated: "):
							r.gen++
						case strings.Contains(line, " = ") && strings.Contains(line, "; graph ops"):
							r.not++
							if k := opKindOf(line); k != "" {
								notBy[k]++
							}
						}
					}
				}
			}
			var hs []string
			for _, k := range rec.IR() {
				hs = append(hs, recorder.Hash(k))
			}
			slices.Sort(hs)
			h := sha256.Sum256([]byte(strings.Join(hs, ",")))
			r.digest = fmt.Sprintf("%x", h[:8])
			if err != nil {
				r.digest = "error"
			}
			got[name] = r
		}()
	}
	var names []string
	for n := range got {
		names = append(names, n)
	}
	slices.Sort(names)
	var b strings.Builder
	gen, not := 0, 0
	for _, n := range names {
		r := got[n]
		gen, not = gen+r.gen, not+r.not
		fmt.Fprintf(&b, "%s %s %d %d\n", n, r.digest, r.gen, r.not)
	}
	t.Logf("%d models: %d op-steps on generated kernels, %d not", len(names), gen, not)
	var kinds []string
	for k := range notBy {
		kinds = append(kinds, k)
	}
	slices.SortFunc(kinds, func(a, b string) int { return notBy[b] - notBy[a] })
	for _, k := range kinds {
		t.Logf("  not generated: %-20s %d", k, notBy[k])
	}
	path := filepath.Join("testdata", "composer_ir.golden")
	if *updateComposer {
		if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (rerun with -update-composer to record it)", err)
	}
	for _, line := range strings.Split(strings.TrimSpace(string(want)), "\n") {
		f := strings.Fields(line)
		if len(f) != 4 {
			continue
		}
		r, ok := got[f[0]]
		if !ok {
			t.Errorf("%s: emitted nothing; the golden has %s", f[0], f[1])
			continue
		}
		wantGen, _ := strconv.Atoi(f[2])
		if r.digest != f[1] {
			t.Errorf("%s: emitted code changed (%s, the golden's %s)", f[0], r.digest, f[1])
		}
		if r.gen < wantGen {
			t.Errorf("%s: %d ops generated, the golden's %d", f[0], r.gen, wantGen)
		}
	}
}

// opKindOf is the kind of a program dump's op line: the word before its "(".
func opKindOf(line string) string {
	_, rhs, ok := strings.Cut(line, " = ")
	if !ok {
		return ""
	}
	k, _, _ := strings.Cut(rhs, "(")
	return strings.TrimSpace(k)
}
