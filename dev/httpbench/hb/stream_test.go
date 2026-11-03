package hb

import (
	"strings"
	"testing"
	"time"
)

// TestReadStreamTimesTextChunksAndTakesUsage: a chat stream with a role-only
// first chunk, two text chunks, an empty one and a usage chunk. Only text
// chunks are timed; the token count is usage's, not the chunk count.
func TestReadStreamTimesTextChunksAndTakesUsage(t *testing.T) {
	body := `data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}

data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}

data: {"choices":[{"index":0,"delta":{}}]}

data: {"choices":[{"index":0,"delta":{"content":"lo"}}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}

data: [DONE]

`
	// The request started a millisecond ago: a stream read from memory ends
	// inside one tick of a coarse clock (Windows), and TTFT must still count
	// from the request's start, not read 0.
	var s Sample
	if err := readStream(strings.NewReader(body), time.Now().Add(-time.Millisecond), &s, nil); err != nil {
		t.Fatal(err)
	}
	if s.Chunks != 2 || len(s.ITL) != 1 || s.CompletionTokens != 3 || s.PromptTokens != 5 ||
		!s.UsageSeen || s.FinishReason != "length" || s.TTFT <= 0 {
		t.Fatalf("%+v", s)
	}
}

// TestReadStreamRefusesABrokenStream: no [DONE], an error event, an error
// object and an unparseable chunk are each an error, never a short sample.
func TestReadStreamRefusesABrokenStream(t *testing.T) {
	for _, body := range []string{
		"data: {\"choices\":[{\"text\":\"a\"}]}\n\n",
		"event: error\ndata: {\"message\":\"boom\"}\n\ndata: [DONE]\n\n",
		"data: {\"error\":{\"message\":\"boom\"}}\n\ndata: [DONE]\n\n",
		"data: {not json\n\ndata: [DONE]\n\n",
	} {
		var s Sample
		if err := readStream(strings.NewReader(body), time.Now(), &s, nil); err == nil {
			t.Errorf("%q parsed as %+v", body, s)
		}
	}
	// Without usage the count falls back to chunks, and the sample says so.
	var s Sample
	two := "data: {\"choices\":[{\"text\":\"a\"}]}\n\ndata: {\"choices\":[{\"text\":\"b\"}]}\n\ndata: [DONE]\n\n"
	if err := readStream(strings.NewReader(two), time.Now(), &s, nil); err != nil {
		t.Fatal(err)
	}
	if s.UsageSeen || s.CompletionTokens != 2 {
		t.Fatalf("%+v", s)
	}
}

// TestJobsAreTheSameForEveryArmOfARound: the arms of one round are sent the
// same prompts and the same model mix; another round gets another list; the
// weights are honoured.
func TestJobsAreTheSameForEveryArmOfARound(t *testing.T) {
	c := Config{Seed: 3, PromptLens: []int{4, 16}, Models: []Model{{"a", 3}, {"b", 1}}}
	x, y, z := c.jobs(nil, 8, 1, 400), c.jobs(nil, 8, 1, 400), c.jobs(nil, 8, 2, 400)
	same, na := true, 0
	for i := range x {
		if x[i] != y[i] {
			t.Fatal("one round's list differs between two calls")
		}
		if x[i] != z[i] {
			same = false
		}
		if x[i].model == "a" {
			na++
		}
		if n := len(strings.Fields(x[i].prompt)); n != 4 && n != 16 {
			t.Fatalf("a prompt of %d words()", n)
		}
	}
	if same {
		t.Fatal("two rounds sent the same list")
	}
	if na < 260 || na > 340 {
		t.Fatalf("model a got %d of 400 at weight 3:1", na)
	}
}
