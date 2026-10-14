package model

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/tok/jinja"
)

// TestDecisionMediaReadsLlamaCppsShapes holds the request's pictures to
// llama.cpp's server's reading (server-decision.cpp parse_state): "files"
// then "images", then the image_url parts of a chat-message state, taken out
// of their messages; a null state with pictures; and the refusals.
func TestDecisionMediaReadsLlamaCppsShapes(t *testing.T) {
	const a, b, c = "data:image/png;base64,QQ==", "data:image/png;base64,Qg==", "data:image/png;base64,Qw=="
	parse := func(body string) (jinja.Value, []string, error) {
		t.Helper()
		_, state, _, urls, err := ParseDecisionRequest([]byte(body))
		return state, urls, err
	}
	q := `"questions":{"x":{"type":"noul","instructions":"?"}}`
	state, urls, err := parse(`{"state":null,"images":["` + a + `"],"files":["` + b + `"],` + q + `}`)
	if err != nil {
		t.Fatal(err)
	}
	if !state.IsNone() || strings.Join(urls, " ") != b+" "+a {
		t.Fatalf("state %v, urls %v: want null and files before images", state, urls)
	}
	state, urls, err = parse(`{"state":{"messages":[{"role":"user","content":[{"type":"text","text":"hi"},` +
		`{"type":"image_url","image_url":{"url":"` + c + `"}}]}],"tag":1},"images":["` + a + `"],` + q + `}`)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(urls, " ") != a+" "+c {
		t.Fatalf("urls %v: want the images array, then the message's part", urls)
	}
	if got := state.JSON(); got != `{"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "tag": 1}` {
		t.Fatalf("the state kept its picture or lost its order: %s", got)
	}
	for _, bad := range []struct{ body, want string }{
		{`{"state":"s","videos":["` + a + `"],` + q + `}`, `"videos" is not supported`},
		{`{"state":"s","images":"` + a + `",` + q + `}`, `must be an array`},
		{`{"state":"s","images":["https://example.com/x.png"],` + q + `}`, `data URLs`},
		{`{"state":"s","images":["` + strings.Repeat(a+`","`, 8) + a + `"],` + q + `}`, `too many images`},
	} {
		if _, _, err := parse(bad.body); err == nil || !strings.Contains(err.Error(), bad.want) {
			t.Errorf("%s: %v, want %q", bad.body[:40], err, bad.want)
		}
	}
}
