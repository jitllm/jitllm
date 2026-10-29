package model

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
)

func withStructures(r map[uint64]nn.Backends) Option {
	return func(l *loadOpts) { l.opt.structures = r }
}

// TestComposedDefaultFollowsTheRecord: a model whose every op structure is on
// the record for the CPU opens with its blocks on the CPU composed and serves
// them so; one with a structure off the record, or on it for a card only,
// opens on main's kernels; and WithComposedCPU decides either way over the
// record.
func TestComposedDefaultFollowsTheRecord(t *testing.T) {
	path := jlmOf(t, testmodels.Path("stories15M-q8_0.gguf"))
	probe, err := Open(path, noTune)
	if err != nil {
		t.Fatal(err)
	}
	set, ok := probe.composedStructuresOf()
	probe.Close()
	if !ok || len(set) == 0 {
		t.Fatal("the front end states no graphs for the model: the gate has nothing to read")
	}
	record := func(on nn.Backends, skip int) map[uint64]nn.Backends {
		r := map[uint64]nn.Backends{}
		for s := range set {
			if skip--; skip != 0 {
				r[s] = on
			}
		}
		return r
	}
	for _, c := range []struct {
		name  string
		opts  []Option
		on    bool
		ready nn.Backends
	}{
		{"on the record", []Option{withStructures(record(nn.BackendCPU, 0))}, true, nn.BackendCPU},
		{"one structure off the record", []Option{withStructures(record(nn.BackendCPU, 1))}, false, 0},
		{"on the record for a card only", []Option{withStructures(record(nn.BackendPTX, 0))}, false, nn.BackendPTX},
		{"empty record", []Option{withStructures(map[uint64]nn.Backends{})}, false, 0},
		{"on the record, switched off", []Option{withStructures(record(nn.BackendCPU, 0)), WithComposedCPU(false)}, false, nn.BackendCPU},
		{"off the record, switched on", []Option{withStructures(map[uint64]nn.Backends{}), WithComposedCPU(true)}, true, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			m, err := Open(path, append([]Option{noTune}, c.opts...)...)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if got := m.composedReady(); got != c.ready {
				t.Fatalf("ready on %b, want %b", got, c.ready)
			}
			if m.opt.composedCPU != c.on {
				t.Fatalf("composed CPU %v, want %v", m.opt.composedCPU, c.on)
			}
			st := m.NewState(64)
			defer st.Close()
			if _, err := st.Prefill(m.Vocab.Encode("The river rose every spring until", true)); err != nil {
				t.Fatal(err)
			}
			served, _ := st.CPUBlockSteps()
			if c.on != (served > 0) {
				t.Fatalf("composed CPU block steps served %d, with composed CPU %v", served, c.on)
			}
		})
	}
}

// TestComposedStructuresOfModels prints each listed model's op structures
// (JITLLM_COMPOSED_STRUCTURES, names separated by commas): what the record
// is built from after the board's measurements.
func TestComposedStructuresOfModels(t *testing.T) {
	names := os.Getenv("JITLLM_COMPOSED_STRUCTURES")
	if names == "" {
		t.Skip("a listing: set JITLLM_COMPOSED_STRUCTURES")
	}
	for _, name := range strings.Split(names, ",") {
		m, err := Open(jlmOf(t, testmodels.Path(name)), noTune)
		if err != nil {
			t.Fatal(err)
		}
		set, ok := m.composedStructuresOf()
		m.Close()
		if !ok {
			t.Logf("STRUCTURES %s uncovered", name)
			continue
		}
		var hs []string
		for s := range set {
			hs = append(hs, fmt.Sprintf("%016x", s))
		}
		t.Logf("STRUCTURES %s %s", name, strings.Join(hs, " "))
	}
}
