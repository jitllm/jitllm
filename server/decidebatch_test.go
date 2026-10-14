package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
	"github.com/jitllm/jitllm/tok/jinja"
)

// decisionServerModels are the decision models the batching gate serves:
// each readout the engine has, under $JITLLM_MODELS as the engine's decision
// gates find them.
var decisionServerModels = []struct {
	name, path string
	// tol is how far a batched probability may sit from the request's alone:
	// the engine's batch session against its single prefill
	// (model.TestDecisionBatchMatchesAlone has the measurements).
	tol float64
}{
	{"d1-3B", "d1/d1-3B-Q8_0.jlm", 0.03},
	{"lev", "lev/lev-Q8_0.jlm", 0.05},
	{"laya", "laya/gguf/Laya-BF16.jlm", 0},
	// Clef answers a request in one pass of its own: batched is alone.
	{"clef", "clef/Clef-Flash-Q8_0.jlm", 0},
	{"clef-fixture", "clef/fixture/clef-fixture.jlm", 0},
}

// TestSystemOneBatchesConcurrentRequests sends concurrent /v1/systemone
// requests to a loaded decision model and holds each answer to the same
// request answered alone by a model.Decider on the same container. The
// requests are held at the model's Decider until every one has queued, so
// they run as one batch, which the gate asserts (the batch's size) rather
// than infers; one request that fails its checks fails alone.
func TestSystemOneBatchesConcurrentRequests(t *testing.T) {
	bodies := map[string][]byte{}
	for _, r := range []string{"support", "ambiguous"} {
		b, err := os.ReadFile(filepath.Join("..", "engine", "model", "testdata", "decision", r+".json"))
		if err != nil {
			t.Fatal(err)
		}
		bodies[r] = b
	}
	for _, c := range decisionServerModels {
		t.Run(c.name, func(t *testing.T) {
			path := testmodels.Path(c.path)
			fresh.Must(t, path)
			if _, err := os.Stat(path); err != nil {
				t.Skipf("MODEL MISSING: %s (set JITLLM_MODELS to the model directory) -- this gate proved nothing", path)
			}
			e := New(Config{ModelDir: filepath.Dir(path), Probe: noProbe})
			t.Cleanup(e.Close)
			lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "dec"})
			if err != nil {
				t.Fatalf("LoadModel(%s): %v", path, err)
			}
			srv := httptest.NewServer(e.Handler())
			t.Cleanup(srv.Close)

			// The requests: each body twice, and one with more options than
			// the model has labels for, refused by the Decider, not the wire.
			var reqs [][]byte
			for range 2 {
				for _, r := range []string{"support", "ambiguous"} {
					reqs = append(reqs, withModel(bodies[r]))
				}
			}
			bad := len(reqs)
			var many bytes.Buffer
			many.WriteString(`{"model": "dec", "state": "x", "questions": {"a": {"type": "choice", "criteria": {`)
			for i := range 300 {
				if i > 0 {
					many.WriteString(", ")
				}
				fmt.Fprintf(&many, `"o%d": null`, i)
			}
			many.WriteString("}}}}")
			reqs = append(reqs, many.Bytes())

			// Each alone, on a Decider of its own over the same model.
			d, err := lm.m.NewDecider(e.defaultMaxSeq(lm))
			if err != nil {
				t.Fatal(err)
			}
			want := make([][]byte, len(reqs))
			for i, b := range reqs[:bad] {
				v, err := jinja.FromJSON(b)
				if err != nil {
					t.Fatal(err)
				}
				state, _ := v.AsDict().Get("state")
				qv, _ := v.AsDict().Get("questions")
				qs, err := model.ParseDecisionQuestions(qv)
				if err != nil {
					t.Fatal(err)
				}
				ans, err := d.Decide(state, qs, nil)
				if err != nil {
					t.Fatal(err)
				}
				want[i] = model.DecisionResponse("dec", qs, ans, d.InputTokens())
			}
			d.Close()

			// Held at the Decider until every request has queued.
			lm.decMu.Lock()
			got := make([][]byte, len(reqs))
			status := make([]int, len(reqs))
			var wg sync.WaitGroup
			for i, b := range reqs {
				wg.Go(func() {
					resp, err := http.Post(srv.URL+"/v1/systemone", "application/json", bytes.NewReader(b))
					if err != nil {
						t.Error(err)
						return
					}
					defer resp.Body.Close()
					status[i] = resp.StatusCode
					got[i], _ = io.ReadAll(resp.Body)
				})
			}
			for deadline := time.Now().Add(time.Minute); ; {
				lm.decQMu.Lock()
				n := len(lm.decQ)
				lm.decQMu.Unlock()
				if n == len(reqs) {
					break
				}
				if time.Now().After(deadline) {
					lm.decMu.Unlock()
					t.Fatalf("%d of %d requests queued", n, len(reqs))
				}
				time.Sleep(5 * time.Millisecond)
			}
			lm.decMu.Unlock()
			wg.Wait()

			if n := lm.decBatchMax.Load(); n != int64(len(reqs)) {
				t.Fatalf("the largest batch answered %d requests of the %d queued", n, len(reqs))
			}
			if status[bad] != http.StatusUnprocessableEntity {
				t.Errorf("the request the model cannot ask: status %d: %s", status[bad], got[bad])
			}
			worst := 0.0
			for i := range reqs[:bad] {
				if status[i] != http.StatusOK {
					t.Fatalf("request %d: status %d: %s", i, status[i], got[i])
				}
				w, err := jsonDiff(want[i], got[i])
				if err != nil {
					t.Fatalf("request %d: %v\n got %s\nwant %s", i, err, got[i], want[i])
				}
				worst = max(worst, w)
			}
			t.Logf("%d requests answered as one batch: worst difference from each alone %.6f (bound %g)",
				len(reqs), worst, c.tol)
			if !(worst <= c.tol) {
				t.Errorf("batched, an answer moved %.6f from the request's alone", worst)
			}
		})
	}
}

// withModel is a request body naming the model "dec", the state and
// questions in the order the body wrote them.
func withModel(b []byte) []byte {
	b = bytes.TrimSpace(b)
	return append([]byte(`{"model": "dec", `), b[1:]...)
}

// jsonDiff is the largest difference between the numbers of two JSON
// documents of the same shape, and an error where the shapes differ.
func jsonDiff(a, b []byte) (float64, error) {
	var x, y any
	if err := json.Unmarshal(a, &x); err != nil {
		return 0, err
	}
	if err := json.Unmarshal(b, &y); err != nil {
		return 0, err
	}
	return valueDiff(x, y, "")
}

func valueDiff(x, y any, at string) (float64, error) {
	switch xv := x.(type) {
	case float64:
		yv, ok := y.(float64)
		if !ok {
			return 0, fmt.Errorf("%s: %v against %v", at, x, y)
		}
		d := math.Abs(xv - yv)
		if math.IsNaN(d) {
			return math.Inf(1), nil
		}
		return d, nil
	case map[string]any:
		yv, ok := y.(map[string]any)
		if !ok || len(xv) != len(yv) {
			return 0, fmt.Errorf("%s: the objects differ", at)
		}
		w := 0.0
		for k, v := range xv {
			d, err := valueDiff(v, yv[k], at+"."+k)
			if err != nil {
				return 0, err
			}
			w = max(w, d)
		}
		return w, nil
	case []any:
		yv, ok := y.([]any)
		if !ok || len(xv) != len(yv) {
			return 0, fmt.Errorf("%s: the lists differ", at)
		}
		w := 0.0
		for i := range xv {
			d, err := valueDiff(xv[i], yv[i], fmt.Sprintf("%s[%d]", at, i))
			if err != nil {
				return 0, err
			}
			w = max(w, d)
		}
		return w, nil
	default:
		if x != y {
			return 0, fmt.Errorf("%s: %v against %v", at, x, y)
		}
		return 0, nil
	}
}
