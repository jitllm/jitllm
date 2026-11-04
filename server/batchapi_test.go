package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func uploadBatchFile(t *testing.T, url, purpose, content string) (int, oaFile) {
	t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	mw.WriteField("purpose", purpose)
	fw, err := mw.CreateFormFile("file", "in.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	fw.Write([]byte(content))
	mw.Close()
	resp, err := http.Post(url+"/v1/files", mw.FormDataContentType(), &buf)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var f oaFile
	json.NewDecoder(resp.Body).Decode(&f)
	return resp.StatusCode, f
}

func getBody(t *testing.T, url string) (int, []byte) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, b
}

func waitBatch(t *testing.T, url, id string, done func(oaBatch) bool) oaBatch {
	t.Helper()
	deadline := time.Now().Add(2 * time.Minute)
	for {
		code, b := getBody(t, url+"/v1/batches/"+id)
		var x oaBatch
		if code != 200 || json.Unmarshal(b, &x) != nil {
			t.Fatalf("GET batch %s: %d %s", id, code, b)
		}
		if done(x) {
			return x
		}
		if time.Now().After(deadline) {
			t.Fatalf("batch %s still %q after two minutes", id, x.Status)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// TestBatchAPIRunsEachLineThroughTheLiveRoute: on stories260K, a jsonl of
// completion requests uploaded through /v1/files and run as a /v1/batches
// batch answers each line with exactly the body the live route gives the same
// request; a line the route refuses lands in the error file; a cancelled
// batch ends cancelled; and the records survive a new store over the same
// directory.
func TestBatchAPIRunsEachLineThroughTheLiveRoute(t *testing.T) {
	path := modelPath(t, smallModel)
	dir := t.TempDir()
	e := New(admitConfig(Config{ModelDir: filepath.Dir(path), BatchDir: dir, Probe: oneCardProbe, Version: "test"}))
	t.Cleanup(e.Close)
	if _, err := e.LoadModel(LoadOptions{Path: path, ModelID: "small"}); err != nil {
		t.Fatal(err)
	}
	c := serveEngine(t, e)

	reqs := []string{
		`{"model":"small","prompt":"Once upon a time","max_tokens":12,"temperature":0.9,"seed":4}`,
		`{"model":"small","prompt":"The cat","max_tokens":8,"temperature":0}`,
		`{"model":"small","prompt":"x","logprobs":99}`,
	}
	var in strings.Builder
	for i, r := range reqs {
		fmt.Fprintf(&in, `{"custom_id":"r%d","method":"POST","url":"/v1/completions","body":%s}`+"\n", i, r)
	}
	if code, _ := uploadBatchFile(t, c.url, "fine-tune", in.String()); code != 400 {
		t.Fatalf("an upload for another purpose: status %d, want 400", code)
	}
	code, f := uploadBatchFile(t, c.url, "batch", in.String())
	if code != 200 || f.ID == "" || f.Bytes != int64(in.Len()) {
		t.Fatalf("upload: %d %+v", code, f)
	}
	var x oaBatch
	if code := postJSON(t, c.url, "/v1/batches", fmt.Sprintf(
		`{"input_file_id":%q,"endpoint":"/v1/completions","completion_window":"24h"}`, f.ID), &x); code != 200 {
		t.Fatalf("create: %d", code)
	}
	x = waitBatch(t, c.url, x.ID, func(b oaBatch) bool { return b.Status == "completed" || b.Status == "failed" })
	if x.Status != "completed" || x.RequestCounts != (oaBatchCounts{Total: 3, Completed: 2, Failed: 1}) ||
		x.OutputFileID == nil || x.ErrorFileID == nil {
		t.Fatalf("batch ended %+v", x)
	}

	_, outRaw := getBody(t, c.url+"/v1/files/"+*x.OutputFileID+"/content")
	lines := strings.Split(strings.TrimSpace(string(outRaw)), "\n")
	if len(lines) != 2 {
		t.Fatalf("output file: %d lines:\n%s", len(lines), outRaw)
	}
	for i, l := range lines {
		var res struct {
			CustomID string `json:"custom_id"`
			Response struct {
				StatusCode int       `json:"status_code"`
				Body       legacyOut `json:"body"`
			} `json:"response"`
		}
		if err := json.Unmarshal([]byte(l), &res); err != nil {
			t.Fatal(err)
		}
		var live legacyOut
		if code := postJSON(t, c.url, "/v1/completions", reqs[i], &live); code != 200 {
			t.Fatalf("live %d: %d", i, code)
		}
		if res.CustomID != fmt.Sprintf("r%d", i) || res.Response.StatusCode != 200 ||
			len(res.Response.Body.Choices) != 1 || res.Response.Body.Choices[0].Text != live.Choices[0].Text {
			t.Fatalf("line %d: %s; the live route says %q", i, l, live.Choices[0].Text)
		}
	}
	_, errRaw := getBody(t, c.url+"/v1/files/"+*x.ErrorFileID+"/content")
	if !strings.Contains(string(errRaw), `"custom_id":"r2"`) || !strings.Contains(string(errRaw), `"status_code":400`) {
		t.Fatalf("error file: %s", errRaw)
	}

	// A batch refused at validation fails whole, naming the line.
	_, bad := uploadBatchFile(t, c.url, "batch", `{"custom_id":"a","method":"POST","url":"/v1/embeddings","body":{}}`+"\n")
	var y oaBatch
	postJSON(t, c.url, "/v1/batches", fmt.Sprintf(`{"input_file_id":%q,"endpoint":"/v1/completions"}`, bad.ID), &y)
	y = waitBatch(t, c.url, y.ID, func(b oaBatch) bool { return b.Status == "failed" || b.Status == "completed" })
	if y.Status != "failed" || y.Errors == nil || y.Errors.Data[0].Line == nil || *y.Errors.Data[0].Line != 1 {
		t.Fatalf("a line naming another endpoint: %+v", y)
	}

	// Cancel: a long batch stops before its last line.
	var long strings.Builder
	for i := 0; i < 200; i++ {
		fmt.Fprintf(&long, `{"custom_id":"c%d","method":"POST","url":"/v1/completions","body":{"model":"small","prompt":"Once","max_tokens":32}}`+"\n", i)
	}
	_, lf := uploadBatchFile(t, c.url, "batch", long.String())
	var z oaBatch
	postJSON(t, c.url, "/v1/batches", fmt.Sprintf(`{"input_file_id":%q,"endpoint":"/v1/completions"}`, lf.ID), &z)
	waitBatch(t, c.url, z.ID, func(b oaBatch) bool { return b.RequestCounts.Completed > 0 })
	if code := postJSON(t, c.url, "/v1/batches/"+z.ID+"/cancel", "", nil); code != 200 {
		t.Fatalf("cancel: %d", code)
	}
	z = waitBatch(t, c.url, z.ID, func(b oaBatch) bool { return b.Status == "cancelled" })
	if z.RequestCounts.Completed >= 200 || z.CancelledAt == nil {
		t.Fatalf("cancelled batch %+v", z)
	}

	// The list, newest first.
	var list struct {
		Data []oaBatch `json:"data"`
	}
	_, lb := getBody(t, c.url+"/v1/batches?limit=10")
	json.Unmarshal(lb, &list)
	if len(list.Data) != 3 || list.Data[0].ID != z.ID {
		t.Fatalf("list: %s", lb)
	}

	// A second store over the same directory finds every record.
	b2, err := newBatches(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := b2.loadBatch(x.ID); err != nil || got.Status != "completed" {
		t.Fatalf("after a restart: %+v %v", got, err)
	}
}
