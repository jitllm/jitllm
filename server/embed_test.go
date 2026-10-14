package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/internal/testmodels/fresh"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
	"github.com/jitllm/jitllm/server/gen/jitllm/v1/jitllmv1connect"
)

// The embedding gates: a real embedding container on every path the server
// has -- the Connect RPC, /v1/embeddings as floats and as base64, text and
// token ids, one request against many, many requests at once -- and every
// vector is held BIT FOR BIT to a model.Embedder run on the same container,
// which is what `jitllm embed` prints.
//
// The models are the engine's own embedding gate's, under
// $JITLLM_MODELS/embed/: one per readout the engine has. A missing one skips
// naming the file; the default set of internal/testmodels/fetch.sh carries
// all four GGUFs and `jitllm convert` makes the containers.
var embedGateModels = []string{
	"MiniLM-L6-v2.Q8_0.jlm",    // BERT, mean pooling
	"nomic-embed-text.jlm",     // the rotary encoder, no position table
	"qwen3-embedding-0.6b.jlm", // a causal decoder pooled at its EOS
	"embeddinggemma-300m.jlm",  // a bidirectional decoder with dense heads
}

var embedTexts = []string{
	"The capital of France is Paris.",
	"search_query: what does a pager do",
	"a",
	"Embeddings are vectors; the server must return the ones the engine computes, in the order asked.",
}

// embedServer loads one container into a fresh engine behind its full mux.
func embedServer(t *testing.T, name string, cfg Config) (*Engine, *LoadedModel, string) {
	t.Helper()
	path := testmodels.Path(filepath.Join("embed", name))
	fresh.Must(t, path)
	if _, err := os.Stat(path); err != nil {
		t.Skipf("MODEL MISSING: %s (set JITLLM_MODELS to the model directory and convert "+
			"embed/%s.gguf) -- this gate proved nothing", path, strings.TrimSuffix(name, ".jlm"))
	}
	cfg.ModelDir, cfg.Probe = filepath.Dir(path), noProbe
	e := New(cfg)
	t.Cleanup(e.Close)
	lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "emb"})
	if err != nil {
		t.Fatalf("LoadModel(%s): %v", path, err)
	}
	s := httptest.NewServer(e.Handler())
	t.Cleanup(s.Close)
	return e, lm, s.URL
}

// embedReference is what `jitllm embed` computes: a model.Embedder of its
// own, one text at a time.
func embedReference(t *testing.T, m *model.Model, texts []string) (vecs [][]float32, ids [][]int32, limit int) {
	t.Helper()
	emb, err := m.NewEmbedder()
	if err != nil {
		t.Fatal(err)
	}
	defer emb.Close()
	for _, s := range texts {
		x := m.EmbedIDs(s)
		v, err := emb.Embed(x)
		if err != nil {
			t.Fatal(err)
		}
		vecs = append(vecs, append([]float32(nil), v...))
		ids = append(ids, x)
	}
	// The embedder against itself: a gate demanding bit equality must first
	// see that the reference is bit-stable, or a red line would mean nothing.
	again, err := emb.Embed(ids[0])
	if err != nil {
		t.Fatal(err)
	}
	if d := firstDiff(again, vecs[0]); d >= 0 {
		t.Fatalf("the embedder is not bit-stable against itself at element %d", d)
	}
	return vecs, ids, emb.MaxTokens()
}

func firstDiff(a, b []float32) int {
	if len(a) != len(b) {
		return 0
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return i
		}
	}
	return -1
}

func requireSame(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d dimensions, want %d", what, len(got), len(want))
	}
	if d := firstDiff(got, want); d >= 0 {
		t.Fatalf("%s: element %d is %v, the embedder computes %v", what, d, got[d], want[d])
	}
}

type oaEmbedReply struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string          `json:"object"`
		Index     int             `json:"index"`
		Embedding json.RawMessage `json:"embedding"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
	Error *oaErrorBody `json:"error"`
}

func postEmbeddings(t *testing.T, url string, body any) (int, oaEmbedReply) {
	t.Helper()
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(url+"/v1/embeddings", "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	var out oaEmbedReply
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("%d %s: %v", resp.StatusCode, raw, err)
	}
	return resp.StatusCode, out
}

// vectorOf decodes one data entry, a float array or the base64 of its
// little-endian float32 bytes.
func vectorOf(t *testing.T, raw json.RawMessage) []float32 {
	t.Helper()
	var f []float32
	if json.Unmarshal(raw, &f) == nil {
		return f
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("embedding is neither floats nor base64: %s", raw)
	}
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		t.Fatal(err)
	}
	if len(b)%4 != 0 {
		t.Fatalf("base64 embedding is %d bytes, not a whole number of float32s", len(b))
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func TestEmbedMatchesTheEmbedder(t *testing.T) {
	for _, name := range embedGateModels {
		t.Run(name, func(t *testing.T) {
			_, lm, url := embedServer(t, name, Config{})
			m := lm.m
			want, wantIDs, limit := embedReference(t, m, embedTexts)
			total := 0
			for _, x := range wantIDs {
				total += len(x)
			}
			t.Logf("%s: %s, %d dims, %v pooling, at most %d tokens", name, m.Cfg.Arch, m.Cfg.NEmbd, m.Pooling(), limit)

			cl := jitllmv1connect.NewInferenceServiceClient(http.DefaultClient, url)
			t.Run("rpc, one request of every input", func(t *testing.T) {
				req := &v1.EmbedRequest{ModelId: "emb"}
				for _, s := range embedTexts {
					req.Inputs = append(req.Inputs, &v1.EmbedInput{Input: &v1.EmbedInput_Text{Text: s}})
				}
				resp, err := cl.Embed(context.Background(), connect.NewRequest(req))
				if err != nil {
					t.Fatal(err)
				}
				got := resp.Msg
				if len(got.Embeddings) != len(embedTexts) {
					t.Fatalf("%d embeddings for %d inputs", len(got.Embeddings), len(embedTexts))
				}
				for i, g := range got.Embeddings {
					requireSame(t, fmt.Sprintf("input %d", i), g.Vector, want[i])
					if int(g.Tokens) != len(wantIDs[i]) {
						t.Errorf("input %d: %d tokens, the embedder ran %d", i, g.Tokens, len(wantIDs[i]))
					}
				}
				if int(got.TotalTokens) != total || int(got.Dimensions) != m.Cfg.NEmbd ||
					got.Pooling != m.Pooling().String() || got.ModelId != "emb" {
					t.Errorf("total %d dims %d pooling %q model %q; want %d %d %q emb",
						got.TotalTokens, got.Dimensions, got.Pooling, got.ModelId, total, m.Cfg.NEmbd, m.Pooling())
				}
			})

			t.Run("rpc, token ids embed as their text does", func(t *testing.T) {
				req := &v1.EmbedRequest{ModelId: lm.Name()} // the file name resolves too
				for _, x := range wantIDs {
					req.Inputs = append(req.Inputs, &v1.EmbedInput{Input: &v1.EmbedInput_TokenIds{TokenIds: &v1.TokenIDs{Ids: x}}})
				}
				resp, err := cl.Embed(context.Background(), connect.NewRequest(req))
				if err != nil {
					t.Fatal(err)
				}
				for i, g := range resp.Msg.Embeddings {
					requireSame(t, fmt.Sprintf("ids %d", i), g.Vector, want[i])
				}
			})

			t.Run("/v1/embeddings, a batch in order, floats and base64", func(t *testing.T) {
				for _, format := range []string{"", "float", "base64"} {
					body := map[string]any{"model": "emb", "input": embedTexts}
					if format != "" {
						body["encoding_format"] = format
					}
					code, out := postEmbeddings(t, url, body)
					if code != http.StatusOK {
						t.Fatalf("%q: %d %+v", format, code, out.Error)
					}
					if out.Object != "list" || out.Model != "emb" || len(out.Data) != len(embedTexts) {
						t.Fatalf("%q: object %q model %q %d entries", format, out.Object, out.Model, len(out.Data))
					}
					for i, d := range out.Data {
						if d.Object != "embedding" || d.Index != i {
							t.Fatalf("%q: entry %d is object %q index %d", format, i, d.Object, d.Index)
						}
						requireSame(t, fmt.Sprintf("%q entry %d", format, i), vectorOf(t, d.Embedding), want[i])
					}
					if out.Usage.PromptTokens != total || out.Usage.TotalTokens != total {
						t.Errorf("%q: usage %+v, the embedder ran %d tokens", format, out.Usage, total)
					}
				}
			})

			t.Run("/v1/embeddings, one at a time and as ids", func(t *testing.T) {
				for i, s := range embedTexts {
					code, out := postEmbeddings(t, url, map[string]any{"model": "emb", "input": s})
					if code != http.StatusOK || len(out.Data) != 1 {
						t.Fatalf("text %d: %d %+v", i, code, out.Error)
					}
					requireSame(t, fmt.Sprintf("alone %d", i), vectorOf(t, out.Data[0].Embedding), want[i])
					if out.Usage.PromptTokens != len(wantIDs[i]) {
						t.Errorf("text %d: %d prompt tokens, want %d", i, out.Usage.PromptTokens, len(wantIDs[i]))
					}
					code, out = postEmbeddings(t, url, map[string]any{"model": "emb", "input": wantIDs[i]})
					if code != http.StatusOK || len(out.Data) != 1 {
						t.Fatalf("ids %d: %d %+v", i, code, out.Error)
					}
					requireSame(t, fmt.Sprintf("ids alone %d", i), vectorOf(t, out.Data[0].Embedding), want[i])
				}
				code, out := postEmbeddings(t, url, map[string]any{"model": "emb", "input": wantIDs})
				if code != http.StatusOK || len(out.Data) != len(wantIDs) {
					t.Fatalf("[][]int: %d %+v", code, out.Error)
				}
				for i, d := range out.Data {
					requireSame(t, fmt.Sprintf("[][]int %d", i), vectorOf(t, d.Embedding), want[i])
				}
			})

			t.Run("dimensions: the model's own width or a refusal", func(t *testing.T) {
				code, out := postEmbeddings(t, url, map[string]any{"model": "emb", "input": "a", "dimensions": m.Cfg.NEmbd})
				if code != http.StatusOK {
					t.Fatalf("dimensions = the model's width: %d %+v", code, out.Error)
				}
				code, out = postEmbeddings(t, url, map[string]any{"model": "emb", "input": "a", "dimensions": m.Cfg.NEmbd / 2})
				if code != http.StatusBadRequest || out.Error == nil || !strings.Contains(out.Error.Message, "Matryoshka") {
					t.Fatalf("dimensions = half: %d %+v", code, out.Error)
				}
				_, err := cl.Embed(context.Background(), connect.NewRequest(&v1.EmbedRequest{ModelId: "emb",
					Dimensions: int32(m.Cfg.NEmbd / 2),
					Inputs:     []*v1.EmbedInput{{Input: &v1.EmbedInput_Text{Text: "a"}}}}))
				if connect.CodeOf(err) != connect.CodeInvalidArgument {
					t.Fatalf("rpc dimensions = half: %v", err)
				}
			})

			t.Run("an input past the limit is refused, not truncated", func(t *testing.T) {
				long := make([]int32, limit+1)
				for i := range long {
					long[i] = wantIDs[0][i%len(wantIDs[0])]
				}
				code, out := postEmbeddings(t, url, map[string]any{"model": "emb", "input": long})
				if code != http.StatusBadRequest || out.Error == nil || !strings.Contains(out.Error.Message, "at most") {
					t.Fatalf("%d tokens: %d %+v", len(long), code, out.Error)
				}
			})
		})
	}
}

// TestEmbedConcurrentRequestsAreCorrect: many requests at once -- through
// Engine.Embed, the RPC and /v1/embeddings -- with the host wide enough for
// two embedders on one model to run together. Every vector must still be the
// embedder's, at its own index.
func TestEmbedConcurrentRequestsAreCorrect(t *testing.T) {
	e, lm, url := embedServer(t, embedGateModels[0], Config{})
	// Long inputs and several per request, so two requests' embeds overlap
	// on the host rather than slipping between each other's round trips.
	var long []string
	for k := 0; k < 2; k++ {
		for _, s := range embedTexts {
			long = append(long, strings.Repeat(fmt.Sprintf("%d %s ", k, s), 150/(len(s)/4+2)+1))
		}
	}
	want, _, _ := embedReference(t, lm.m, long)
	cl := jitllmv1connect.NewInferenceServiceClient(http.DefaultClient, url)

	// each runs one request of texts by one path and returns its vectors.
	each := func(path int, texts []string) ([][]float32, error) {
		var got [][]float32
		switch path {
		case 0:
			o := EmbedOptions{Model: "emb"}
			for _, s := range texts {
				o.Inputs = append(o.Inputs, Prompt{Kind: PromptText, Text: s})
			}
			res, err := e.Embed(context.Background(), o)
			if err != nil {
				return nil, err
			}
			got = res.Vectors
		case 1:
			req := &v1.EmbedRequest{ModelId: "emb"}
			for _, s := range texts {
				req.Inputs = append(req.Inputs, &v1.EmbedInput{Input: &v1.EmbedInput_Text{Text: s}})
			}
			resp, err := cl.Embed(context.Background(), connect.NewRequest(req))
			if err != nil {
				return nil, err
			}
			for _, g := range resp.Msg.Embeddings {
				got = append(got, g.Vector)
			}
		default:
			b, err := json.Marshal(map[string]any{"model": "emb", "input": texts, "encoding_format": "base64"})
			if err != nil {
				return nil, err
			}
			resp, err := http.Post(url+"/v1/embeddings", "application/json", bytes.NewReader(b))
			if err != nil {
				return nil, err
			}
			defer resp.Body.Close()
			var out struct {
				Data []struct {
					Index     int    `json:"index"`
					Embedding string `json:"embedding"`
				} `json:"data"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
				return nil, err
			}
			for i, d := range out.Data {
				if d.Index != i {
					return nil, fmt.Errorf("entry %d carries index %d", i, d.Index)
				}
				raw, err := base64.StdEncoding.DecodeString(d.Embedding)
				if err != nil {
					return nil, err
				}
				v := make([]float32, len(raw)/4)
				for k := range v {
					v[k] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*k:]))
				}
				got = append(got, v)
			}
		}
		return got, nil
	}

	var wg sync.WaitGroup
	errs := make(chan error, 64)
	for w := 0; w < 6; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for r := 0; r < 3; r++ {
				// Each request a different rotation, so a vector handed to the
				// wrong request or the wrong index cannot pass.
				order := make([]int, len(long))
				texts := make([]string, len(long))
				for i := range order {
					order[i] = (i + 5*w + r) % len(long)
					texts[i] = long[order[i]]
				}
				got, err := each((w+r)%3, texts)
				if err != nil {
					errs <- fmt.Errorf("worker %d round %d: %v", w, r, err)
					return
				}
				if len(got) != len(order) {
					errs <- fmt.Errorf("worker %d round %d: %d vectors for %d inputs", w, r, len(got), len(order))
					return
				}
				for i, j := range order {
					if d := firstDiff(got[i], want[j]); d >= 0 {
						errs <- fmt.Errorf("worker %d round %d input %d (text %d): element %d differs", w, r, i, j, d)
						return
					}
				}
			}
		}(w)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

// TestEmbedRefusesWhatCannotEmbed: a generative container is refused by name
// on both paths, and an encoder refuses a decoder session rather than taking
// the server down with NewState's panic.
func TestEmbedRefusesWhatCannotEmbed(t *testing.T) {
	gen := testmodels.Path("stories260K.jlm")
	fresh.Must(t, gen)
	if _, err := os.Stat(gen); err != nil {
		t.Skipf("MODEL MISSING: %s (internal/testmodels/fetch.sh copies stories260K and "+
			"`jitllm convert` makes the container) -- this gate proved nothing", gen)
	}
	e, enc, url := embedServer(t, embedGateModels[0], Config{})
	if _, err := e.LoadModel(LoadOptions{Path: gen, ModelID: "gen"}); err != nil {
		t.Fatal(err)
	}
	genLM, err := e.Model("gen")
	if err != nil {
		t.Fatal(err)
	}
	if gi, ei := e.pbModelInfo(genLM), e.pbModelInfo(enc); gi.Pooling != "" || gi.Encoder ||
		ei.Pooling != enc.m.Pooling().String() || !ei.Encoder {
		t.Fatalf("ModelInfo: generative pooling %q encoder %v; encoder pooling %q encoder %v",
			gi.Pooling, gi.Encoder, ei.Pooling, ei.Encoder)
	}

	code, out := postEmbeddings(t, url, map[string]any{"model": "gen", "input": "hello"})
	if code != http.StatusBadRequest || out.Error == nil || out.Error.Type != "invalid_request_error" ||
		!strings.Contains(out.Error.Message, "not an embedding model") {
		t.Fatalf("/v1/embeddings on a generative model: %d %+v", code, out.Error)
	}
	cl := jitllmv1connect.NewInferenceServiceClient(http.DefaultClient, url)
	_, err = cl.Embed(context.Background(), connect.NewRequest(&v1.EmbedRequest{ModelId: "gen",
		Inputs: []*v1.EmbedInput{{Input: &v1.EmbedInput_Text{Text: "hello"}}}}))
	if connect.CodeOf(err) != connect.CodeInvalidArgument || !strings.Contains(err.Error(), "not an embedding model") {
		t.Fatalf("rpc Embed on a generative model: %v", err)
	}

	code, out = postEmbeddings(t, url, map[string]any{"model": "absent", "input": "hello"})
	if code != http.StatusNotFound {
		t.Fatalf("an unknown model: %d %+v", code, out.Error)
	}
	for _, bad := range []any{"", []string{}, []string{"a", ""}, map[string]int{"x": 1}} {
		if code, out := postEmbeddings(t, url, map[string]any{"model": "emb", "input": bad}); code != http.StatusBadRequest {
			t.Errorf("input %v: %d %+v", bad, code, out.Error)
		}
	}
	if code, out := postEmbeddings(t, url, map[string]any{"model": "emb", "input": []int{enc.m.Cfg.NVocab}}); code != http.StatusBadRequest {
		t.Errorf("a token id past the vocabulary: %d %+v", code, out.Error)
	}
	if code, out := postEmbeddings(t, url, map[string]any{"model": "emb", "input": "a", "encoding_format": "int8"}); code != http.StatusBadRequest {
		t.Errorf("encoding_format int8: %d %+v", code, out.Error)
	}

	if _, err := e.CreateSession(SessionOptions{ModelID: "emb"}); !errors.Is(err, ErrInvalid) ||
		!strings.Contains(err.Error(), "encoder") {
		t.Fatalf("a session on an encoder: %v", err)
	}
	b, _ := json.Marshal(map[string]any{"model": "emb", "prompt": "hello", "max_tokens": 2})
	resp, err := http.Post(url+"/v1/completions", "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("/v1/completions on an encoder: %d", resp.StatusCode)
	}
}

// TestUnloadWaitsForEmbedsInFlight: an unload with an embedder out waits for
// it, and closes it, before the model; embeds racing the unload either finish
// with the right vector or are refused as not found.
func TestUnloadWaitsForEmbedsInFlight(t *testing.T) {
	e, lm, _ := embedServer(t, embedGateModels[0], Config{})
	want, _, _ := embedReference(t, lm.m, embedTexts[:1])

	held, err := lm.takeEmbedder()
	if err != nil {
		t.Fatal(err)
	}
	unloaded := make(chan error, 1)
	go func() {
		_, err := e.UnloadModel("emb", false)
		unloaded <- err
	}()

	var wg sync.WaitGroup
	var mu sync.Mutex
	finished, refused := 0, 0
	errs := make(chan error, 16)
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res, err := e.Embed(context.Background(), EmbedOptions{Model: "emb",
				Inputs: []Prompt{{Kind: PromptText, Text: embedTexts[0]}}})
			mu.Lock()
			defer mu.Unlock()
			switch {
			case errors.Is(err, ErrNotFound):
				refused++
			case err != nil:
				errs <- err
			case firstDiff(res.Vectors[0], want[0]) >= 0:
				errs <- fmt.Errorf("an embed beside the unload returned a different vector")
			default:
				finished++
			}
		}()
	}
	wg.Wait()
	// The unload has reached the embedders once they are marked closed; from
	// there it may only wait.
	for deadline := time.Now().Add(10 * time.Second); ; time.Sleep(time.Millisecond) {
		lm.embMu.Lock()
		closed := lm.emb.closed
		lm.embMu.Unlock()
		if closed {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the unload never reached the embedders")
		}
	}
	time.Sleep(20 * time.Millisecond)
	select {
	case err := <-unloaded:
		t.Fatalf("the unload returned (%v) with an embedder still out", err)
	default:
	}
	// The held embedder still works: the model under it is open.
	if v, err := held.Embed(lm.m.EmbedIDs(embedTexts[0])); err != nil || firstDiff(v, want[0]) >= 0 {
		t.Fatalf("the held embedder after the unload began: %v", err)
	}
	lm.giveEmbedder(held)
	if err := <-unloaded; err != nil {
		t.Fatal(err)
	}
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	t.Logf("%d embeds finished beside the unload, %d were refused", finished, refused)
	if lm.emb.busy != 0 || len(lm.emb.free) != 0 || !lm.emb.closed {
		t.Fatalf("after unload: %d busy, %d idle embedders, closed %v", lm.emb.busy, len(lm.emb.free), lm.emb.closed)
	}
}
