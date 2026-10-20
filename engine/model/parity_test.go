package model

import (
	"cmp"
	"crypto/sha256"
	"encoding/binary"
	"flag"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/backend/recorder"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/tier"
	"github.com/jitllm/jitllm/paritytest/maintier"
)

var parityProfiles = flag.String("parity-profiles", "cuda-sm70",
	"the recorder profiles TestComposerParity compares on")

// TestComposerParity holds what the composer emits to what main emits, with
// no GPU: every fixture opened twice on a device that runs nothing and
// records what is compiled for it (recorder) -- once on main's tier
// (paritytest/maintier, main's own code), once on the composer's -- a prompt
// chunk and a decode step asked of each, and every instruction of main's
// kernels looked for among the composer's formed kernels by its dataflow:
// the instruction and, recursively, everything it reads, so the order of
// independent instructions does not count. It reports per profile the share
// of main's instructions matched, the models whose every instruction is, and
// main's kernels by instructions missing. A measurement, not a gate.
func TestComposerParity(t *testing.T) {
	paths, _ := filepath.Glob(filepath.Join(testmodels.Dir(), "synth-*.jlm"))
	if len(paths) == 0 {
		t.Fatalf("no synth-*.jlm under %s (set JITLLM_MODELS)", testmodels.Dir())
	}
	for _, pn := range strings.Split(*parityProfiles, ",") {
		prof, ok := recorder.Profiles[pn]
		if !ok {
			t.Fatalf("no recorder profile %q", pn)
		}
		var total, same, models, identical int
		missing := map[string]int{}
		for _, p := range paths {
			name := strings.TrimSuffix(filepath.Base(p), ".jlm")
			if strings.HasSuffix(name, "-vlm") || strings.Contains(name, "mmproj") {
				continue
			}
			mk, err := parityKernels(t, prof, p, true)
			if err != nil {
				t.Logf("%s on %s, main: %v", name, pn, err)
			}
			if len(mk) == 0 {
				continue
			}
			ck, err := parityKernels(t, prof, p, false)
			if err != nil {
				t.Logf("%s on %s, composer: %v", name, pn, err)
			}
			have := map[[16]byte]int{}
			for _, k := range ck {
				if strings.HasPrefix(k.Comment, "composed") {
					for _, h := range dataflow(k) {
						have[h]++
					}
				}
			}
			mt, ms := 0, 0
			for _, k := range mk {
				prefix, _, _ := strings.Cut(k.Name, "_")
				for i, h := range dataflow(k) {
					if k.Ops[i].Kind == ir.OpParam {
						continue
					}
					mt++
					if have[h] > 0 {
						have[h]--
						ms++
						continue
					}
					missing[prefix]++
				}
			}
			total, same, models = total+mt, same+ms, models+1
			if ms == mt {
				identical++
			}
		}
		t.Logf("%-16s main's instructions %d, the composer's dataflow-identical %d (%.1f%%), models identical %d/%d",
			pn, total, same, 100*float64(same)/float64(max(total, 1)), identical, models)
		var names []string
		for n := range missing {
			names = append(names, n)
		}
		slices.SortFunc(names, func(a, b string) int { return cmp.Compare(missing[b], missing[a]) })
		for _, n := range names[:min(10, len(names))] {
			t.Logf("    %-22s %9d missing", n, missing[n])
		}
	}
}

// parityKernels is every kernel compiled for the model at path on a
// recording device standing in for prof, on main's tier or the composer's,
// over a prompt chunk and a decode step: main's launched, the composer's
// formed. Launches run nothing.
func parityKernels(t *testing.T, prof recorder.Profile, path string, main bool) ([]*ir.Kernel, error) {
	m, err := Open(path, noTune)
	if err != nil {
		return nil, err
	}
	defer m.Close()
	if m.IsEncoder() {
		return nil, nil
	}
	rec := recorder.New(prof, 1<<40)
	var g nn.Device
	if main {
		mg, err := maintier.New([]maintier.Slot{{Dev: rec, Bytes: 1 << 40}},
			maintier.WithConfig(func(c *maintier.Config) { c.NoGraph = true }))
		if err != nil {
			return nil, err
		}
		defer mg.Close()
		g = mg
	} else {
		cg, err := tier.New([]tier.Slot{{Dev: rec, Bytes: 1 << 40}}, append(testTierOpts(t),
			tier.WithConfig(func(c *tier.Config) { c.NoGraph = true }))...)
		if err != nil {
			return nil, err
		}
		defer cg.Close()
		g = cg
	}
	s := m.NewState(32)
	defer s.Close()
	if err := s.SetDevice(g); err != nil {
		return nil, err
	}
	lg, err := s.Prefill(m.Vocab.Encode("Hello there", true))
	tok := int32(1)
	if err == nil {
		tok = Greedy(lg)
	}
	if _, ferr := s.Forward(tok); err == nil {
		err = ferr
	}
	// Main's kernels are what it launched (its tuners compile alternatives
	// they never run); the composer's what it formed, launched or not (a
	// step stops at the first op it cannot form).
	var ks []*ir.Kernel
	if main {
		for _, e := range rec.Launched() {
			ks = append(ks, e.K)
		}
		return ks, err
	}
	for _, e := range rec.Compiled(func(k *ir.Kernel) bool { return strings.HasPrefix(k.Comment, "composed") }) {
		ks = append(ks, e.K)
	}
	return ks, err
}

// dataflow is a digest of each of k's instructions: its kind, type,
// immediate, tile and loop depth, and the digests of what it reads, so two
// instructions agree when they compute the same thing from the same inputs
// whatever order independent instructions were emitted in. A loop-carried
// value (a phi's back edge, a read of a later instruction) enters by its
// kind alone, which ends the recursion. A parameter is its place and element
// type.
func dataflow(k *ir.Kernel) [][16]byte {
	depth := make([]int, len(k.Ops))
	d := 0
	for i, o := range k.Ops {
		if o.Kind == ir.OpEndLoop {
			d--
		}
		depth[i] = d
		if o.Kind == ir.OpLoop {
			d++
		}
	}
	out := make([][16]byte, len(k.Ops))
	done := make([]bool, len(k.Ops))
	var h func(i int) [16]byte
	h = func(i int) [16]byte {
		if done[i] {
			return out[i]
		}
		o := k.Ops[i]
		s := sha256.New()
		var w [8]byte
		put := func(x uint64) { binary.LittleEndian.PutUint64(w[:], x); s.Write(w[:]) }
		put(uint64(o.Kind))
		put(uint64(o.Type))
		put(uint64(o.Imm))
		put(uint64(depth[i]))
		if o.Tile != nil {
			fmt.Fprintf(s, "%+v", *o.Tile)
		}
		if o.Kind == ir.OpParam && int(o.Imm) < len(k.Params) {
			p := k.Params[o.Imm]
			put(uint64(p.Elem))
			put(uint64(p.Tile))
		}
		args := append(o.Args[:], o.Vargs...)
		for j, a := range args {
			switch {
			case a == 0:
				put(0)
			case int(a)-1 >= i || o.Kind == ir.OpPhi && j == 1:
				put(1<<32 | uint64(k.Ops[int(a)-1].Kind))
			default:
				x := h(int(a) - 1)
				s.Write(x[:])
			}
		}
		copy(out[i][:], s.Sum(nil))
		done[i] = true
		return out[i]
	}
	for i := range k.Ops {
		h(i)
	}
	return out
}
