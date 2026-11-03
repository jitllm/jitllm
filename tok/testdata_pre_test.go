package tok

import (
	"fmt"
	"github.com/jitllm/jitllm/tok/pretok"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// TestTestdataPreTokenizersMatchTheGeneratedTable reads every real pre_tokenizer
// in testdata/pre (dumped from the repos) and requires the runtime parse to
// equal the generated table row. Unlike the hand-written JSON of
// TestParsePreTokenizerAgreesWithTheGeneratedTable, these are what the models
// actually serve.
//
// A file that disagrees with its table row is a finding (the model moved, or
// the table is wrong), not a fixture to refresh until green.
func TestTestdataPreTokenizersMatchTheGeneratedTable(t *testing.T) {
	// The names are written down: a directory listing as the expectation
	// would pass over an empty directory. gpt-4o adds no new stage sequence;
	// its difference (o200k) is inside the Split, so it is listed explicitly,
	// and so is seed-coder (a punctuation run with no newline tail).
	// deepseek-v3 shares hunyuan-dense's three whole-pattern Splits, and
	// spark2_5 is the same three with a variant third pattern
	// (pretok.OpSparkSplit); neither may read as three llama3 splits, which
	// would tokenize "12345" one digit at a time.
	want := []string{"deepseek-v3", "falcon-h1", "falcon3", "gpt-2", "gpt-4o", "hunyuan-dense", "llama-bpe", "qwen2",
		"seed-coder", "smollm", "spark2_5"}

	found, err := filepath.Glob("testdata/pre/*.json")
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(found))
	for _, p := range found {
		names = append(names, strings.TrimSuffix(filepath.Base(p), ".json"))
	}
	sort.Strings(names)
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("testdata/pre holds %v, want %v -- see its README for what each covers", names, want)
	}

	shapes := map[string]string{}
	for _, name := range want {
		b, err := os.ReadFile(filepath.Join("testdata", "pre", name+".json"))
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		ops, err := pretok.ParsePreTokenizer(strings.NewReader(string(b)))
		if err != nil {
			t.Errorf("%s: the dumped object does not parse: %v", name, err)
			continue
		}
		row, ok := pretok.Table()[name]
		if !ok {
			t.Errorf("%s: not in the generated table; this file proved nothing", name)
			continue
		}
		if !reflect.DeepEqual(ops, row) {
			t.Errorf("%s: the model's own pre_tokenizer and the generated table disagree\n  file  %+v\n  table %+v",
				name, ops, row)
			continue
		}
		// The pipeline must also build -- parsing to the right ops and then
		// refusing to compose them would be a green row and a dead loader.
		if _, err := buildPipeline(ops); err != nil {
			t.Errorf("%s: parsed but does not build: %v", name, err)
			continue
		}
		var kinds []string
		for _, op := range ops {
			kinds = append(kinds, op.Kind.String())
		}
		shapes[strings.Join(kinds, "->")] = name
	}

	// Seven shapes is the claim pipeline.go's header makes; asserting it
	// catches a sample set that shrinks and a new shape upstream.
	if len(shapes) != 7 {
		keys := make([]string, 0, len(shapes))
		for k := range shapes {
			keys = append(keys, fmt.Sprintf("%s (%s)", k, shapes[k]))
		}
		sort.Strings(keys)
		t.Errorf("testdata covers %d distinct pipeline shapes, want the 7 in pipeline.go's header:\n  %s",
			len(shapes), strings.Join(keys, "\n  "))
	}
}
