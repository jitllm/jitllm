package convert

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

func testNames(t testing.TB) tensorNames {
	t.Helper()
	v, err := readTensorNames()
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// repeatedNames are the names b's object holds more than once, which
// json.Unmarshal would read as the last one's role without a word.
func repeatedNames(b []byte) ([]string, error) {
	d := json.NewDecoder(bytes.NewReader(b))
	if tok, err := d.Token(); err != nil || tok != json.Delim('{') {
		return nil, fmt.Errorf("not an object: %v %v", tok, err)
	}
	seen := map[string]bool{}
	var out []string
	for d.More() {
		tok, err := d.Token()
		if err != nil {
			return nil, err
		}
		k := tok.(string)
		if seen[k] {
			out = append(out, k)
		}
		seen[k] = true
		var v string
		if err := d.Decode(&v); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// TestTensorNamesAreOneVocabulary holds tensornames.json to one role per
// name: no name twice, every role one the container knows, and no name the
// fused-projection and decision-head rules read, which the file would shadow.
func TestTensorNamesAreOneVocabulary(t *testing.T) {
	b, err := tensorNamesFS.ReadFile("tensornames.json")
	if err != nil {
		t.Fatal(err)
	}
	if rep, err := repeatedNames([]byte(`{"a.weight": "attn_q", "a.weight": "attn_k"}`)); err != nil || len(rep) != 1 {
		t.Fatalf("the repeat check read %v (%v) of a file naming a.weight twice", rep, err)
	}
	rep, err := repeatedNames(b)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range rep {
		t.Errorf("%q is named more than once", n)
	}
	v := testNames(t)
	if len(v.vocab) == 0 {
		t.Fatal("the vocabulary holds no name: this gate proved nothing")
	}
	ruled := []string{"ssm_in.weight", "shortconv.in_proj.weight"}
	for r := jlm.RoleDecHiddenNorm; r <= jlm.RoleDecFFNDownBias; r++ {
		if r.Valid() {
			ruled = append(ruled, r.String(), r.String()+".weight")
		}
	}
	for _, n := range ruled {
		if r, ok := v.vocab[n]; ok {
			t.Errorf("%q is %v in tensornames.json, which a rule reads", n, r)
		}
	}
	t.Logf("%d names, %d read by rule", len(v.vocab), len(ruled))
}
