package engine

import (
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/format/jlm"
)

// An engine error must be explained in words, with the fix that applies.
//
// The stale case matters most: the raw error tells a desktop user to run a
// command, and a two-line dialog cut it before the instruction.
func TestEngineErrorsAreExplainedWithTheirFix(t *testing.T) {
	header := func(name string, v uint32) string {
		p := filepath.Join(t.TempDir(), name)
		b := make([]byte, 64)
		copy(b, jlm.Magic)
		binary.LittleEndian.PutUint32(b[8:], v)
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	stale := header("old.jlm", jlm.Version-4)
	newer := header("new.jlm", jlm.Version+1)

	cases := []struct {
		name, stage, path string
		err               error
		fix               fix
		title             string
	}{
		{"gguf", stageOpen, "/m/gemma.gguf", &model.NotConvertedError{Path: "/m/gemma.gguf"}, fixConvert, "converting"},
		{"stale", stageOpen, stale, errors.New("jlm: container is version 17 and this build reads 21"), fixReconvert, "older jitllm"},
		// jlm refuses a newer file too; calling it "older" would offer a Reconvert
		// that downgrades it.
		{"newer", stageOpen, newer, errors.New("jlm: container is version 22 and this build reads 21"), fixNone, "newer jitllm"},
		{"missing", stageOpen, "/m/gone.jlm", errors.New("open /m/gone.jlm: no such file or directory"), fixNone, "Couldn't open gone.jlm"},
		{"device", stageDevice, "/m/x.jlm", errors.New(`no device matched "cuda:3"`), fixDevices, "device setting"},
		{"reply", stageReply, "/m/x.jlm", errors.New("cuda: out of memory"), fixNone, "reply stopped"},
	}
	for _, c := range cases {
		title, detail, f := explain(c.stage, c.path, c.err)
		if f != c.fix {
			t.Errorf("%s: fix %d, want %d (%q)", c.name, f, c.fix, title)
		}
		if !strings.Contains(title, c.title) {
			t.Errorf("%s: title %q does not say %q", c.name, title, c.title)
		}
		if strings.Contains(title+detail, "`jitllm") || strings.Contains(title+detail, "re-run") {
			t.Errorf("%s: a command-line instruction reached the user: %q / %q", c.name, title, detail)
		}
		if c.fix == fixNone && c.name != "newer" && !strings.Contains(detail, c.err.Error()) {
			t.Errorf("%s: an unexplained error lost its text: %q", c.name, detail)
		}
	}
}
