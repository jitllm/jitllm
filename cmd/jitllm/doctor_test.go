package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// fakeEnv is a healthy machine: a code generator, no GPU, known memory.
func fakeEnv() doctorEnv {
	return doctorEnv{version: "test", cpuName: "fake", tier: "avx2", jit: true, totalRAM: 16 << 30,
		budget: 8 << 30, hipAbsent: "no ROCm here", archKnown: jlm.Arch.Valid}
}

// smallContainer converts the committed stories260K into a temporary
// container.
func smallContainer(t *testing.T) string {
	t.Helper()
	testmodels.SourceTree(t)
	// The one model the repository commits, so this never misses.
	src := filepath.Join("..", "..", "testdata", "models", "stories260K.gguf")
	dst := filepath.Join(t.TempDir(), "stories260K.jlm")
	if _, err := convert.FromGGUF(src, dst, jlm.Fingerprint{Host: "test"}); err != nil {
		t.Fatal(err)
	}
	return dst
}

// codeOf runs the printer and returns the exit code and the JSON it wrote.
func codeOf(t *testing.T, r doctorReport) (int, doctorReport) {
	t.Helper()
	var b bytes.Buffer
	code := doctorPrint(&b, true, r)
	var back doctorReport
	if err := json.Unmarshal(b.Bytes(), &back); err != nil {
		t.Fatalf("the -json output does not parse: %v\n%s", err, b.String())
	}
	return code, back
}

func problemCodes(r doctorReport) string {
	var cs []string
	for _, p := range r.Problems {
		cs = append(cs, p.Code)
		if p.Next == "" {
			cs = append(cs, "(no next action)")
		}
	}
	return strings.Join(cs, ",")
}

// TestDoctorOnAHealthyCPUMachine: no GPU is a note, not a problem, and the
// exit code is 0.
func TestDoctorOnAHealthyCPUMachine(t *testing.T) {
	r, err := diagnose(fakeEnv(), smallContainer(t))
	if err != nil {
		t.Fatal(err)
	}
	code, back := codeOf(t, r)
	if code != doctorOK || back.Status != "ok" || len(back.Problems) != 0 {
		t.Fatalf("exit %d status %s problems %s", code, back.Status, problemCodes(back))
	}
	if back.Container == nil || !back.Container.Runs || back.Container.Version != jlm.Version || back.Container.Arch == "" {
		t.Fatalf("container %+v", back.Container)
	}
	for _, b := range back.Backends {
		if b.Usable {
			t.Errorf("backend %s usable with no device", b.Name)
		}
		if b.Name == "hip" && b.Why != "no ROCm here" {
			t.Errorf("hip's reason is %q, not the probe's", b.Why)
		}
	}
	if len(back.Notes) == 0 {
		t.Error("no note says the machine is CPU-only")
	}
}

// TestDoctorReportsFakeDevices: a device list names its backends usable.
func TestDoctorReportsFakeDevices(t *testing.T) {
	env := fakeEnv()
	env.devices = []doctorDevice{{Backend: backendName("ptx"), Name: "card", FreeBytes: 1 << 30, TotalBytes: 2 << 30},
		{Backend: backendName("spirv"), Name: "card"}}
	r, err := diagnose(env, "")
	if err != nil {
		t.Fatal(err)
	}
	usable := map[string]bool{}
	for _, b := range r.Backends {
		usable[b.Name] = b.Usable
	}
	if !usable["cuda"] || !usable["vulkan"] || usable["hip"] || usable["metal"] {
		t.Fatalf("usable %v", usable)
	}
	if code, _ := codeOf(t, r); code != doctorOK {
		t.Fatalf("exit %d", code)
	}
}

// writeOldContainer writes a file whose first twelve bytes say it is a
// container of version v.
func writeOldContainer(t *testing.T, v uint32) string {
	t.Helper()
	b := make([]byte, jlm.HeaderBytes)
	copy(b, jlm.Magic[:])
	binary.LittleEndian.PutUint32(b[8:], v)
	p := filepath.Join(t.TempDir(), "old.jlm")
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestDoctorNamesWhatIsMissing runs each missing requirement and holds it to
// its problem code, a next action and exit 2.
func TestDoctorNamesWhatIsMissing(t *testing.T) {
	small := smallContainer(t)
	notJLM := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(notJLM, []byte("GGUF and more bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name      string
		env       func(*doctorEnv)
		container string
		want      string
	}{
		{"older container", nil, writeOldContainer(t, jlm.Version-2), "container-version"},
		{"newer container", nil, writeOldContainer(t, jlm.Version+1), "container-version"},
		{"no container", nil, filepath.Join(t.TempDir(), "absent.jlm"), "container-missing"},
		{"a GGUF", nil, notJLM, "not-a-container"},
		{"unknown architecture", func(e *doctorEnv) { e.archKnown = func(jlm.Arch) bool { return false } }, small, "architecture-unknown"},
		{"no code generator", func(e *doctorEnv) { e.jit = false }, "", "no-code-generator"},
		{"cpu below baseline", func(e *doctorEnv) { e.hostErr = errors.New("missing SSE4.1") }, "", "cpu-below-baseline"},
	}
	for _, c := range cases {
		env := fakeEnv()
		if c.env != nil {
			c.env(&env)
		}
		r, err := diagnose(env, c.container)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		code, back := codeOf(t, r)
		if code != doctorMissing || back.Status != "missing" || problemCodes(back) != c.want {
			t.Errorf("%s: exit %d status %s problems %q, want exit %d and %s", c.name, code, back.Status,
				problemCodes(back), doctorMissing, c.want)
		}
	}
}

// TestDoctorErrorIsStructured: an inspection that cannot run is exit 1 with a
// JSON error on stdout.
func TestDoctorErrorIsStructured(t *testing.T) {
	var b bytes.Buffer
	code := doctorFail(&b, true, "usage", errors.New("bad flag"))
	var r doctorReport
	if err := json.Unmarshal(b.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	if code != doctorError || r.Status != "error" || r.Error == nil || r.Error.Code != "usage" {
		t.Fatalf("exit %d report %+v", code, r)
	}
	if doctorError == doctorMissing || doctorError == doctorOK || doctorMissing == doctorOK {
		t.Fatal("the exit codes are not distinct")
	}
}
