package convert

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestHeadJoinsAsIfTheTrunkCarriedIt converts scripts/qwen4expgold.py's
// synth-qwen4exp-mtp three ways, all written by llama.cpp's own converter
// from one checkpoint: the model with its prediction block in one GGUF, and
// the trunk alone (--no-mtp) joined to the block alone (--mtp, the shape the
// release's head is published in). The two containers must be the same bytes.
//
// The violations: a head joined to a trunk it was not exported beside (the
// plain synth-qwen4exp, another checkpoint), a head whose hyperparameter
// differs from the trunk's, and a trunk that already carries its block.
func TestHeadJoinsAsIfTheTrunkCarriedIt(t *testing.T) {
	full := testmodels.Path("synth-qwen4exp-mtp.gguf")
	trunk := testmodels.Path("synth-qwen4exp-mtp-trunk.gguf")
	head := testmodels.Path("synth-qwen4exp-mtp-head.gguf")
	for _, p := range []string{full, trunk, head} {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "%s (scripts/qwen4expgold.py synth-qwen4exp-mtp)", p)
		}
	}
	dir := t.TempDir()
	fp := jlm.Fingerprint{Writer: "test"}
	a, b := filepath.Join(dir, "full.jlm"), filepath.Join(dir, "joined.jlm")
	if _, err := FromGGUF(full, a, fp); err != nil {
		t.Fatal(err)
	}
	if _, err := FromGGUF(trunk, b, fp, WithHead(head)); err != nil {
		t.Fatal(err)
	}
	ab, err := os.ReadFile(a)
	if err != nil {
		t.Fatal(err)
	}
	bb, err := os.ReadFile(b)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(ab, bb) {
		t.Fatalf("the joined container (%d bytes) is not the one-file conversion (%d bytes)", len(bb), len(ab))
	}
	f, err := jlm.Open(a)
	if err != nil {
		t.Fatal(err)
	}
	c := f.Config()
	nmtp, nlayer := c.NMTP, c.NLayer
	_, own := f.Find(jlm.RoleHCHeadNorm, int32(nlayer), -1)
	f.Close()
	if nmtp != 1 || !own {
		t.Fatalf("the container has %d prediction blocks and the block's own head mixer: %v", nmtp, own)
	}
	t.Logf("joined = one file: %d bytes, %d trunk blocks and %d prediction block", len(ab), nlayer, nmtp)

	// Another checkpoint's trunk: the shapes agree, the weights do not, and
	// the head's copy of the embedding says so.
	other := testmodels.Path("synth-qwen4exp.gguf")
	if _, err := os.Stat(other); err == nil {
		if _, err := FromGGUF(other, filepath.Join(dir, "other.jlm"), fp, WithHead(head)); err == nil {
			t.Errorf("a head joined to a trunk it was not exported beside")
		} else {
			t.Logf("another trunk: %v -- refused", err)
		}
	}
	if _, err := FromGGUF(full, filepath.Join(dir, "twice.jlm"), fp, WithHead(head)); err == nil {
		t.Errorf("a head joined to a file that carries its block already")
	}
	tf, err := gguf.Open(trunk)
	if err != nil {
		t.Fatal(err)
	}
	defer tf.Close()
	hf, err := gguf.Open(head)
	if err != nil {
		t.Fatal(err)
	}
	defer hf.Close()
	k := hf.Arch() + ".attention.head_count"
	v := hf.KV[k]
	raw := append([]byte(nil), v.Raw...)
	raw[0]++
	v.Raw = raw
	hf.KV[k] = v
	if _, err := withHead(tf, hf, trunk, head); err == nil {
		t.Errorf("a head stating another head count joined")
	}
}
