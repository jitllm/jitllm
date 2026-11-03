package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// OpenAI's Files and Batch APIs: /v1/files (upload a jsonl with purpose
// "batch", list, retrieve, read its content, delete) and /v1/batches (create,
// retrieve, list, cancel).
//
// A batch runs each line of its input through the same handlers the live
// routes use -- /v1/chat/completions, /v1/completions, /v1/embeddings -- one
// line after another, in the background, and writes OpenAI's output and error
// files. Nothing is a second inference path: a line is an in-process request
// to the mux.
//
// Files and batch records live in a directory under the server's model
// directory (BatchStore.BatchDir), one data file and one JSON record each, so
// they survive a restart; a batch that was running when the process stopped
// comes back failed, since its run did not.

// BatchStore is a Backend that keeps the Files and Batch APIs' state on disk.
// The Engine is one; a Backend that is not serves those routes as 501.
type BatchStore interface {
	BatchDir() string
}

// BatchDir is where the Files and Batch APIs keep their files.
func (e *Engine) BatchDir() string {
	if e.cfg.BatchDir != "" {
		return e.cfg.BatchDir
	}
	return filepath.Join(e.ModelDir(), ".jitllm-batch")
}

// maxBatchFile bounds an upload; OpenAI's own limit is 200 MB.
const maxBatchFile = 200 << 20

type oaFile struct {
	ID        string `json:"id"`
	Object    string `json:"object"`
	Bytes     int64  `json:"bytes"`
	CreatedAt int64  `json:"created_at"`
	Filename  string `json:"filename"`
	Purpose   string `json:"purpose"`
}

type oaBatchCounts struct {
	Total     int `json:"total"`
	Completed int `json:"completed"`
	Failed    int `json:"failed"`
}

type oaBatch struct {
	ID               string            `json:"id"`
	Object           string            `json:"object"`
	Endpoint         string            `json:"endpoint"`
	InputFileID      string            `json:"input_file_id"`
	CompletionWindow string            `json:"completion_window"`
	Status           string            `json:"status"`
	OutputFileID     *string           `json:"output_file_id"`
	ErrorFileID      *string           `json:"error_file_id"`
	CreatedAt        int64             `json:"created_at"`
	InProgressAt     *int64            `json:"in_progress_at"`
	CompletedAt      *int64            `json:"completed_at"`
	FailedAt         *int64            `json:"failed_at"`
	CancellingAt     *int64            `json:"cancelling_at"`
	CancelledAt      *int64            `json:"cancelled_at"`
	RequestCounts    oaBatchCounts     `json:"request_counts"`
	Metadata         map[string]string `json:"metadata"`
	Errors           *oaBatchErrors    `json:"errors"`
}

// oaBatchCreate is POST /v1/batches's body.
type oaBatchCreate struct {
	InputFileID      string            `json:"input_file_id"`
	Endpoint         string            `json:"endpoint"`
	CompletionWindow string            `json:"completion_window"`
	Metadata         map[string]string `json:"metadata"`
}

type oaBatchErrors struct {
	Object string         `json:"object"`
	Data   []oaBatchError `json:"data"`
}

type oaBatchError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Line    *int   `json:"line,omitempty"`
}

// batches is the Files and Batch APIs' state: the records on disk, and the
// cancel of every batch this process is running.
type batches struct {
	dir    string
	mux    http.Handler // the live routes a line runs through
	mu     sync.Mutex
	cancel map[string]context.CancelFunc
	wg     sync.WaitGroup
}

func newBatches(dir string, mux http.Handler) (*batches, error) {
	for _, d := range []string{"files", "batches"} {
		if err := os.MkdirAll(filepath.Join(dir, d), 0o755); err != nil {
			return nil, err
		}
	}
	b := &batches{dir: dir, mux: mux, cancel: map[string]context.CancelFunc{}}
	// A batch left running by an earlier process did not finish.
	for _, x := range b.listBatches() {
		if x.Status == "validating" || x.Status == "in_progress" || x.Status == "finalizing" || x.Status == "cancelling" {
			now := time.Now().Unix()
			if x.Status == "cancelling" {
				x.Status, x.CancelledAt = "cancelled", &now
			} else {
				x.Status, x.FailedAt = "failed", &now
				x.Errors = &oaBatchErrors{Object: "list", Data: []oaBatchError{{Code: "server_restarted",
					Message: "the server stopped while the batch ran"}}}
			}
			b.saveBatch(x)
		}
	}
	return b, nil
}

func (b *batches) fileData(id string) string { return filepath.Join(b.dir, "files", id+".data") }
func (b *batches) fileMeta(id string) string { return filepath.Join(b.dir, "files", id+".json") }
func (b *batches) batchMeta(id string) string {
	return filepath.Join(b.dir, "batches", id+".json")
}

// validID keeps an id from the URL inside the store.
func validID(id string) bool {
	return id != "" && !strings.ContainsAny(id, `/\.`) && len(id) < 128
}

func writeRecord(path string, v any) error {
	raw, err := json.Marshal(v)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func readRecord(path string, v any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, v)
}

func (b *batches) saveBatch(x *oaBatch) {
	b.mu.Lock()
	defer b.mu.Unlock()
	// A record that cannot be written leaves the last one standing; the
	// batch's own run still finishes.
	writeRecord(b.batchMeta(x.ID), x)
}

func (b *batches) loadBatch(id string) (*oaBatch, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	var x oaBatch
	return &x, readRecord(b.batchMeta(id), &x)
}

func (b *batches) listBatches() []*oaBatch {
	ents, err := os.ReadDir(filepath.Join(b.dir, "batches"))
	if err != nil {
		return nil
	}
	var out []*oaBatch
	for _, en := range ents {
		id, ok := strings.CutSuffix(en.Name(), ".json")
		if !ok {
			continue
		}
		if x, err := b.loadBatch(id); err == nil {
			out = append(out, x)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].CreatedAt != out[j].CreatedAt {
			return out[i].CreatedAt > out[j].CreatedAt
		}
		return out[i].ID > out[j].ID
	})
	return out
}

func (b *batches) putFile(id, name, purpose string, data []byte) (*oaFile, error) {
	if err := os.WriteFile(b.fileData(id), data, 0o644); err != nil {
		return nil, err
	}
	f := &oaFile{ID: id, Object: "file", Bytes: int64(len(data)), CreatedAt: time.Now().Unix(),
		Filename: name, Purpose: purpose}
	return f, writeRecord(b.fileMeta(id), f)
}

func (b *batches) getFile(id string) (*oaFile, error) {
	var f oaFile
	return &f, readRecord(b.fileMeta(id), &f)
}

// ---------------------------------------------------------------- routes

func (c *compat) batchesFor(w http.ResponseWriter) *batches {
	c.batchOnce.Do(func() {
		bs, ok := c.b.(BatchStore)
		if !ok {
			c.batchErr = errors.New("this backend keeps no files")
			return
		}
		c.batch, c.batchErr = newBatches(bs.BatchDir(), c.mux)
	})
	if c.batchErr != nil {
		oaFail(w, http.StatusNotImplemented, "files and batches: "+c.batchErr.Error(), "server_error")
		return nil
	}
	return c.batch
}

// openAIFiles is /v1/files and /v1/files/{id}[/content].
func (c *compat) openAIFiles(w http.ResponseWriter, r *http.Request) {
	b := c.batchesFor(w)
	if b == nil {
		return
	}
	rest := strings.Trim(strings.TrimPrefix(r.URL.Path, "/v1/files"), "/")
	id, sub, _ := strings.Cut(rest, "/")
	switch {
	case rest == "" && r.Method == http.MethodPost:
		r.Body = http.MaxBytesReader(w, r.Body, maxBatchFile+1<<20)
		f, hdr, err := func() (io.ReadCloser, string, error) {
			if err := r.ParseMultipartForm(32 << 20); err != nil {
				return nil, "", err
			}
			f, h, err := r.FormFile("file")
			if err != nil {
				return nil, "", err
			}
			return f, h.Filename, nil
		}()
		if err != nil {
			oaFail(w, http.StatusBadRequest, "a multipart upload with a `file` part: "+err.Error(), "invalid_request_error")
			return
		}
		defer f.Close()
		purpose := r.FormValue("purpose")
		if purpose != "batch" {
			oaFail(w, http.StatusBadRequest, `purpose must be "batch": it is the only one this server runs`, "invalid_request_error")
			return
		}
		data, err := io.ReadAll(io.LimitReader(f, maxBatchFile+1))
		if err != nil || len(data) > maxBatchFile {
			oaFail(w, http.StatusBadRequest, "the file is unreadable or larger than 200 MB", "invalid_request_error")
			return
		}
		out, err := b.putFile("file-"+c.b.NextID("f"), hdr, purpose, data)
		if err != nil {
			oaFail(w, http.StatusInternalServerError, err.Error(), "server_error")
			return
		}
		writeJSON(w, http.StatusOK, out)
	case rest == "" && r.Method == http.MethodGet:
		out := struct {
			Object string    `json:"object"`
			Data   []*oaFile `json:"data"`
		}{Object: "list", Data: []*oaFile{}}
		ents, err := os.ReadDir(filepath.Join(b.dir, "files"))
		if err == nil {
			for _, en := range ents {
				if fid, ok := strings.CutSuffix(en.Name(), ".json"); ok {
					if f, err := b.getFile(fid); err == nil {
						out.Data = append(out.Data, f)
					}
				}
			}
		}
		sort.Slice(out.Data, func(i, j int) bool { return out.Data[i].CreatedAt > out.Data[j].CreatedAt })
		writeJSON(w, http.StatusOK, out)
	case !validID(id):
		oaFail(w, http.StatusNotFound, "no such file", "invalid_request_error")
	default:
		f, err := b.getFile(id)
		if err != nil {
			oaFail(w, http.StatusNotFound, fmt.Sprintf("no file %q", id), "invalid_request_error")
			return
		}
		switch {
		case sub == "" && r.Method == http.MethodGet:
			writeJSON(w, http.StatusOK, f)
		case sub == "content" && r.Method == http.MethodGet:
			w.Header().Set("Content-Type", "application/octet-stream")
			http.ServeFile(w, r, b.fileData(id))
		case sub == "" && r.Method == http.MethodDelete:
			os.Remove(b.fileData(id))
			os.Remove(b.fileMeta(id))
			writeJSON(w, http.StatusOK, map[string]any{"id": id, "object": "file", "deleted": true})
		default:
			oaFail(w, http.StatusMethodNotAllowed, "no such operation on a file", "invalid_request_error")
		}
	}
}

// batchEndpoints are the routes a batch line may name.
func batchEndpoints() map[string]bool {
	return map[string]bool{
		"/v1/chat/completions": true,
		"/v1/completions":      true,
		"/v1/embeddings":       true,
	}
}

// openAIBatches is /v1/batches, /v1/batches/{id} and /v1/batches/{id}/cancel.
func (c *compat) openAIBatches(w http.ResponseWriter, r *http.Request) {
	b := c.batchesFor(w)
	if b == nil {
		return
	}
	rest := strings.Trim(strings.TrimPrefix(r.URL.Path, "/v1/batches"), "/")
	id, sub, _ := strings.Cut(rest, "/")
	switch {
	case rest == "" && r.Method == http.MethodPost:
		var req oaBatchCreate
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			oaFail(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
			return
		}
		if !batchEndpoints()[req.Endpoint] {
			oaFail(w, http.StatusBadRequest, "endpoint must be /v1/chat/completions, /v1/completions or /v1/embeddings", "invalid_request_error")
			return
		}
		if !validID(req.InputFileID) {
			oaFail(w, http.StatusBadRequest, "input_file_id names no file", "invalid_request_error")
			return
		}
		f, err := b.getFile(req.InputFileID)
		if err != nil || f.Purpose != "batch" {
			oaFail(w, http.StatusBadRequest, fmt.Sprintf("input_file_id %q is not an uploaded batch file", req.InputFileID), "invalid_request_error")
			return
		}
		if req.CompletionWindow == "" {
			req.CompletionWindow = "24h"
		}
		x := &oaBatch{ID: "batch_" + c.b.NextID("b"), Object: "batch", Endpoint: req.Endpoint,
			InputFileID: req.InputFileID, CompletionWindow: req.CompletionWindow, Status: "validating",
			CreatedAt: time.Now().Unix(), Metadata: req.Metadata}
		ctx, cancel := context.WithCancel(context.Background())
		b.mu.Lock()
		b.cancel[x.ID] = cancel
		b.mu.Unlock()
		b.saveBatch(x)
		created := *x // the run owns x from here on
		b.wg.Add(1)
		go b.run(ctx, x, c.b)
		writeJSON(w, http.StatusOK, created)
	case rest == "" && r.Method == http.MethodGet:
		all := b.listBatches()
		limit := 20
		fmt.Sscan(r.URL.Query().Get("limit"), &limit)
		if after := r.URL.Query().Get("after"); after != "" {
			for i, x := range all {
				if x.ID == after {
					all = all[i+1:]
					break
				}
			}
		}
		more := len(all) > limit
		if more {
			all = all[:limit]
		}
		out := map[string]any{"object": "list", "data": all, "has_more": more}
		if len(all) > 0 {
			out["first_id"], out["last_id"] = all[0].ID, all[len(all)-1].ID
		} else {
			out["data"] = []*oaBatch{}
		}
		writeJSON(w, http.StatusOK, out)
	case !validID(id):
		oaFail(w, http.StatusNotFound, "no such batch", "invalid_request_error")
	case sub == "" && r.Method == http.MethodGet:
		x, err := b.loadBatch(id)
		if err != nil {
			oaFail(w, http.StatusNotFound, fmt.Sprintf("no batch %q", id), "invalid_request_error")
			return
		}
		writeJSON(w, http.StatusOK, x)
	case sub == "cancel" && r.Method == http.MethodPost:
		x, err := b.loadBatch(id)
		if err != nil {
			oaFail(w, http.StatusNotFound, fmt.Sprintf("no batch %q", id), "invalid_request_error")
			return
		}
		b.mu.Lock()
		cancel := b.cancel[id]
		b.mu.Unlock()
		if cancel == nil {
			// Already finished: cancelling it changes nothing, as the API
			// answers.
			writeJSON(w, http.StatusOK, x)
			return
		}
		now := time.Now().Unix()
		x.Status, x.CancellingAt = "cancelling", &now
		b.saveBatch(x)
		cancel()
		writeJSON(w, http.StatusOK, x)
	default:
		oaFail(w, http.StatusMethodNotAllowed, "no such operation on a batch", "invalid_request_error")
	}
}

// batchLine is one line of an input file.
type batchLine struct {
	CustomID string          `json:"custom_id"`
	Method   string          `json:"method"`
	URL      string          `json:"url"`
	Body     json.RawMessage `json:"body"`
}

// run is one batch, start to end, on its own goroutine.
func (b *batches) run(ctx context.Context, x *oaBatch, be Backend) {
	defer b.wg.Done()
	defer func() {
		b.mu.Lock()
		if c := b.cancel[x.ID]; c != nil {
			c()
		}
		delete(b.cancel, x.ID)
		b.mu.Unlock()
	}()
	fail := func(code, msg string, line *int) {
		now := time.Now().Unix()
		x.Status, x.FailedAt = "failed", &now
		x.Errors = &oaBatchErrors{Object: "list", Data: []oaBatchError{{Code: code, Message: msg, Line: line}}}
		b.saveBatch(x)
	}
	raw, err := os.ReadFile(b.fileData(x.InputFileID))
	if err != nil {
		fail("invalid_file", err.Error(), nil)
		return
	}
	// Validation: every line parses and names this batch's endpoint, before
	// any of them runs.
	var lines []batchLine
	seen := map[string]bool{}
	sc := bufio.NewScanner(bytes.NewReader(raw))
	sc.Buffer(make([]byte, 1<<20), maxBatchFile)
	for n := 1; sc.Scan(); n++ {
		t := strings.TrimSpace(sc.Text())
		if t == "" {
			continue
		}
		var l batchLine
		if err := json.Unmarshal([]byte(t), &l); err != nil {
			fail("invalid_json_line", err.Error(), &n)
			return
		}
		if l.CustomID == "" || seen[l.CustomID] {
			fail("invalid_custom_id", "every line needs a custom_id of its own", &n)
			return
		}
		seen[l.CustomID] = true
		if l.Method != http.MethodPost || l.URL != x.Endpoint {
			fail("invalid_url", fmt.Sprintf("a line must be POST %s, the batch's endpoint", x.Endpoint), &n)
			return
		}
		lines = append(lines, l)
	}
	if err := sc.Err(); err != nil {
		fail("invalid_file", err.Error(), nil)
		return
	}
	if len(lines) == 0 {
		fail("empty_file", "the input file has no requests", nil)
		return
	}
	now := time.Now().Unix()
	x.Status, x.InProgressAt, x.RequestCounts.Total = "in_progress", &now, len(lines)
	b.saveBatch(x)

	var out, errs bytes.Buffer
	for _, l := range lines {
		if ctx.Err() != nil {
			break
		}
		// A streamed line would answer in SSE frames; a batch line's answer is
		// one body.
		body := l.Body
		var m map[string]json.RawMessage
		if json.Unmarshal(body, &m) == nil {
			delete(m, "stream")
			body, _ = json.Marshal(m)
		}
		req := httptest.NewRequestWithContext(ctx, http.MethodPost, l.URL, bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		b.mux.ServeHTTP(rec, req)
		reqID := "req_" + be.NextID("r")
		res := map[string]any{"id": "batch_req_" + be.NextID("br"), "custom_id": l.CustomID}
		if rec.Code == http.StatusOK {
			res["response"] = map[string]any{"status_code": rec.Code, "request_id": reqID,
				"body": json.RawMessage(rec.Body.Bytes())}
			res["error"] = nil
			line, _ := json.Marshal(res)
			out.Write(append(line, '\n'))
			x.RequestCounts.Completed++
		} else {
			var e oaError
			json.Unmarshal(rec.Body.Bytes(), &e)
			res["response"] = map[string]any{"status_code": rec.Code, "request_id": reqID,
				"body": json.RawMessage(rec.Body.Bytes())}
			res["error"] = map[string]any{"code": e.Error.Type, "message": e.Error.Message}
			line, _ := json.Marshal(res)
			errs.Write(append(line, '\n'))
			x.RequestCounts.Failed++
		}
		b.saveBatch(x)
	}

	x.Status = "finalizing"
	b.saveBatch(x)
	if out.Len() > 0 {
		f, err := b.putFile("file-"+be.NextID("f"), x.ID+"_output.jsonl", "batch_output", out.Bytes())
		if err != nil {
			fail("output_file", err.Error(), nil)
			return
		}
		x.OutputFileID = &f.ID
	}
	if errs.Len() > 0 {
		f, err := b.putFile("file-"+be.NextID("f"), x.ID+"_error.jsonl", "batch_output", errs.Bytes())
		if err != nil {
			fail("error_file", err.Error(), nil)
			return
		}
		x.ErrorFileID = &f.ID
	}
	now = time.Now().Unix()
	if ctx.Err() != nil {
		x.Status, x.CancelledAt = "cancelled", &now
	} else {
		x.Status, x.CompletedAt = "completed", &now
	}
	b.saveBatch(x)
}
