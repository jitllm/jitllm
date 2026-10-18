package server

import (
	"fmt"
	"net/http"
	"reflect"
	"sort"
	"strings"

	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/internal/capabilities"
	"github.com/jitllm/jitllm/internal/localserver"
)

// The server's half of the capability manifest (internal/capabilities). The
// routes below are the ones CompatHandler and Handler mount -- the same
// tables, not a description of them -- and each request's fields are read off
// its struct by reflection. What reflection cannot know, what a field does
// and its limits, is fieldNotes, which TestAPIManifestCoversEveryField holds
// to the structs in both directions; the refusals are held to the handlers by
// TestRefusalsAreRefused.

// compatRoute is one route of the two compatibility APIs.
type compatRoute struct {
	path    string
	methods []string
	surface string
	note    string
	// req is the JSON body's type, nil for a body that is not one struct
	// (multipart, or none).
	req    any
	handle func(*compat, http.ResponseWriter, *http.Request)
}

var compatRoutes = []compatRoute{
	{path: "/v1/chat/completions", methods: []string{http.MethodPost}, surface: "openai", req: oaChatRequest{},
		handle: (*compat).openAIChatCompletions,
		note:   "streams with stream:true (SSE); the model's own chat template renders the messages and tools"},
	{path: "/v1/completions", methods: []string{http.MethodPost}, surface: "openai", req: oaCompletionRequest{},
		handle: (*compat).openAICompletions,
		note:   "a bare prompt, no template; prompt is a string, an array of strings, or token ids"},
	{path: "/v1/models", methods: []string{http.MethodGet}, surface: "openai", handle: (*compat).openAIModels,
		note: "the loaded models, with max_model_len"},
	{path: "/v1/embeddings", methods: []string{http.MethodPost}, surface: "openai", req: oaEmbeddingRequest{},
		handle: (*compat).openAIEmbeddings, note: "an embedding model only; a text model is refused"},
	{path: "/v1/files", methods: []string{http.MethodGet, http.MethodPost}, surface: "openai",
		handle: (*compat).openAIFiles,
		note:   fmt.Sprintf("multipart upload (field file, purpose batch), at most %d MiB", maxBatchFile>>20)},
	{path: "/v1/files/", methods: []string{http.MethodGet, http.MethodDelete}, surface: "openai",
		handle: (*compat).openAIFiles, note: "/v1/files/{id} and /v1/files/{id}/content"},
	{path: "/v1/batches", methods: []string{http.MethodGet, http.MethodPost}, surface: "openai", req: oaBatchCreate{},
		handle: (*compat).openAIBatches, note: "the Batch API over an uploaded file"},
	{path: "/v1/batches/", methods: []string{http.MethodGet, http.MethodPost}, surface: "openai",
		handle: (*compat).openAIBatches, note: "/v1/batches/{id} and POST /v1/batches/{id}/cancel"},
	{path: "/v1/systemone", methods: []string{http.MethodPost}, surface: "jitllm", handle: (*compat).systemOne,
		note: fmt.Sprintf("a decision model's answers: a JSON object with state and questions, at most %d MiB",
			maxSystemOneBody>>20)},
	{path: "/completion", methods: []string{http.MethodPost}, surface: "llamacpp", handle: (*compat).llamaCompletion,
		note: "llama.cpp's /completion: prompt as a string, token ids or both mixed; streams with stream:true; fields the sampler lacks are refused by name"},
	{path: "/infill", methods: []string{http.MethodPost}, surface: "llamacpp", handle: (*compat).llamaInfill,
		note: "llama.cpp's /infill, for a model with fill-in-the-middle tokens"},
	{path: "/tokenize", methods: []string{http.MethodPost}, surface: "llamacpp", handle: (*compat).llamaTokenize,
		note: "content, add_special, parse_special, with_pieces"},
	{path: "/detokenize", methods: []string{http.MethodPost}, surface: "llamacpp", handle: (*compat).llamaDetokenize,
		note: "tokens to content"},
	{path: "/apply-template", methods: []string{http.MethodPost}, surface: "llamacpp", handle: (*compat).llamaApplyTemplate,
		note: "messages through the model's chat template, to a prompt string"},
	{path: "/embedding", methods: []string{http.MethodPost}, surface: "llamacpp", handle: (*compat).llamaEmbedding,
		note: "an embedding model only; llama.cpp's reply shape, L2-normalised"},
	{path: "/props", methods: []string{http.MethodGet, http.MethodPost}, surface: "llamacpp", handle: (*compat).llamaProps,
		note: "read only; POST answers 501 as llama-server does without --props"},
	{path: "/slots", methods: []string{http.MethodGet}, surface: "llamacpp", handle: (*compat).llamaSlots,
		note: "the model's sessions, read only"},
	{path: "/v1/messages", methods: []string{http.MethodPost}, surface: "anthropic", req: anRequest{},
		handle: (*compat).anthropicMessages,
		note:   "streams with stream:true (SSE events as the Messages API sends them)"},
}

// engineRoute is one route Handler mounts beside the compatibility APIs.
type engineRoute struct {
	path    string
	methods []string
	note    string
	handle  func(*Engine, http.ResponseWriter, *http.Request)
}

// engineRoutes is a function rather than a table because one of its routes
// serves the manifest, which lists the table.
func engineRoutes() []engineRoute {
	return []engineRoute{
		{path: "/metrics", methods: []string{http.MethodGet}, handle: (*Engine).serveMetrics,
			note: "Prometheus text format"},
		{path: "/healthz", methods: []string{http.MethodGet}, handle: serveHealth, note: "ok once listening"},
		// /health is the path vLLM and llama-server answer, and so the one their
		// benchmark harnesses poll before sending load.
		{path: "/health", methods: []string{http.MethodGet}, handle: serveLlamaHealth, note: `llama-server's shape: {"status": "ok"}`},
		{path: "/v1/health", methods: []string{http.MethodGet}, handle: serveLlamaHealth, note: "as /health"},
		{path: "/v1/capabilities", methods: []string{http.MethodGet}, handle: (*Engine).serveCapabilities,
			note: "this manifest, for the running build"},
	}
}

func serveHealth(_ *Engine, w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Write([]byte("ok\n"))
}

// serveCapabilities is GET /v1/capabilities: the manifest with this binary's
// identity and the route table it is actually serving.
func (e *Engine) serveCapabilities(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		oaFail(w, http.StatusMethodNotAllowed, "GET only", "invalid_request_error")
		return
	}
	m, err := capabilities.WithAPI(e.cfg.Version, API())
	if err != nil {
		oaFail(w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}
	m.ServerAPI = localserver.APIVersion
	writeJSON(w, http.StatusOK, m)
}

// fieldNotes says what each request field does and its limits, by route and
// wire name. Every field reflection finds must have one, and every note a
// field (TestAPIManifestCoversEveryField).
var fieldNotes = map[string]map[string]string{
	"/v1/chat/completions": withSampling(map[string]string{
		"model":                 "a loaded model's id or name",
		"messages":              fmt.Sprintf("role and content; content is a string or content parts: text, and image_url (a data: URL in base64, or an https URL when the server lists hosts it may fetch from); PNG, JPEG, WebP or GIF, at most %d MiB each, %d MiB and %d pictures a request, %d pixels each, on a model with a vision tower; audio and file parts are refused. tool_calls and tool_call_id carry a tool turn", DefaultImageMaxBytes>>20, DefaultImageMaxRequestBytes>>20, DefaultImageMaxImages, DefaultImageMaxPixels),
		"max_tokens":            "the completion's token limit; absent is the session's KV capacity",
		"max_completion_tokens": "max_tokens's newer spelling; max_tokens wins when both are set",
		"stream":                "SSE frames as the API sends them",
		"stop":                  "a string or an array of strings",
		"user":                  "accepted and ignored",
		"tools":                 "an array of {type: function, function}; handed to the model's own chat template",
		"tool_choice":           "none, auto, required, or {type: function, function: {name}}; required and a named function hold the reply to a call in the model's own syntax",
		"parallel_tool_calls":   "false holds a forced reply to one call",
		"ignore_eos":            "generate past an end-of-generation token up to max_tokens; refused with a forced tool call or a grammar",
		"jitllm_session":        "a server session id: the request continues its KV history",
		"jitllm_priority":       "high or normal",
		"jitllm_speculate":      "speculative decoding on or off for this request; absent takes the session's",
		"response_format":       "text, json_object, or json_schema (json_schema.schema compiled to a grammar); a forced tool call takes precedence",
	}, map[string]string{
		"logprobs":     fmt.Sprintf("a bool; the raw distribution's log-probabilities. top_logprobs is 0..%d", model.MaxTopLogprobs),
		"top_logprobs": fmt.Sprintf("0..%d, only with logprobs", model.MaxTopLogprobs),
	}),
	"/v1/completions": withSampling(map[string]string{
		"model":            "a loaded model's id or name",
		"prompt":           "a string, an array of strings, token ids, or arrays of token ids; a list cannot stream",
		"max_tokens":       "the completion's token limit; absent is the session's KV capacity",
		"stream":           "SSE frames as the API sends them; one prompt only",
		"stop":             "a string or an array of strings",
		"echo":             "prepend the prompt to the text; refused with logprobs",
		"ignore_eos":       "generate past an end-of-generation token up to max_tokens; refused with a grammar",
		"jitllm_session":   "a server session id: the request continues its KV history",
		"jitllm_priority":  "high or normal",
		"jitllm_speculate": "speculative decoding on or off for this request; absent takes the session's",
		"response_format":  "text, json_object, or json_schema (json_schema.schema compiled to a grammar)",
	}, map[string]string{
		"logprobs":     fmt.Sprintf("a count, 0..%d: the top alternatives per token", model.MaxTopLogprobs),
		"top_logprobs": "not read on this route; logprobs is the count",
	}),
	"/v1/embeddings": {
		"model":           "a loaded embedding model's id or name",
		"input":           "a string, an array of strings, token ids, or arrays of token ids; none may be empty, and an input longer than the model's limit is refused, never truncated",
		"encoding_format": "float (default) or base64",
		"dimensions":      "at least 1: a Matryoshka prefix of the vector, renormalised",
		"user":            "accepted and ignored",
	},
	"/v1/batches": {
		"input_file_id":     "an uploaded file with purpose batch",
		"endpoint":          "/v1/chat/completions, /v1/completions or /v1/embeddings",
		"completion_window": "accepted; default 24h",
		"metadata":          "stored and returned",
	},
	"/v1/messages": {
		"model":           "a loaded model's id or name",
		"messages":        fmt.Sprintf("role and content; content is a string or blocks: text, image (source base64 with media_type image/png, image/jpeg, image/webp or image/gif, or url when the server lists hosts it may fetch from; at most %d MiB each and %d pictures a request, on a model with a vision tower), tool_use and tool_result; a document block is refused", DefaultImageMaxBytes>>20, DefaultImageMaxImages),
		"system":          "a string or text blocks",
		"max_tokens":      "required, positive",
		"temperature":     "0 is greedy; default 1",
		"top_p":           "nucleus, [0, 1]",
		"top_k":           "0 disables",
		"stop_sequences":  "strings",
		"stream":          "SSE events as the Messages API sends them",
		"tools":           "name, description, input_schema; cache_control marks a prompt-cache breakpoint",
		"tool_choice":     "auto, any, tool (with name) or none; disable_parallel_tool_use",
		"jitllm_session":  "a server session id: the request continues its KV history",
		"jitllm_priority": "high or normal",
	},
}

// withSampling adds oaSampling's fields, which both OpenAI generate routes
// embed, then the route's own logprobs notes.
func withSampling(m map[string]string, logprobs map[string]string) map[string]string {
	for k, v := range map[string]string{
		"temperature":        "0 is greedy; default 1; not negative",
		"top_p":              "nucleus, [0, 1]; default 1",
		"top_k":              "0 or -1 disables",
		"min_p":              "[0, 1]; 0 disables",
		"repetition_penalty": "positive; 1 disables; over the last 64 tokens",
		"presence_penalty":   "only 0: a non-zero value is refused (repetition_penalty is the engine's)",
		"frequency_penalty":  "only 0: a non-zero value is refused (repetition_penalty is the engine's)",
		"seed":               "the draw's seed; seed+i for choice i",
		"n":                  fmt.Sprintf("1..%d continuations of one prompt, prefilled once; each on a fresh session", maxChoices),
	} {
		m[k] = v
	}
	for k, v := range logprobs {
		m[k] = v
	}
	return m
}

// refusal is one combination the server refuses rather than ignores. probe,
// when set, is a body that must draw a 400 carrying message from the route
// alone (no engine); an engine-level refusal has no probe and its message
// must be one the engine's source returns.
type refusal struct {
	capabilities.Refusal
	probe string
}

var refusals = []refusal{
	{Refusal: capabilities.Refusal{ID: "presence-penalty", Endpoints: []string{"/v1/chat/completions", "/v1/completions"},
		When: "presence_penalty is non-zero", Message: "presence_penalty is not supported"},
		probe: `{"model":"m","messages":[{"role":"user","content":"hi"}],"presence_penalty":0.5}`},
	{Refusal: capabilities.Refusal{ID: "frequency-penalty", Endpoints: []string{"/v1/chat/completions", "/v1/completions"},
		When: "frequency_penalty is non-zero", Message: "frequency_penalty is not supported"},
		probe: `{"model":"m","messages":[{"role":"user","content":"hi"}],"frequency_penalty":0.5}`},
	{Refusal: capabilities.Refusal{ID: "n-range", Endpoints: []string{"/v1/chat/completions", "/v1/completions"},
		When: fmt.Sprintf("n is outside 1..%d", maxChoices), Message: fmt.Sprintf("n must be between 1 and %d", maxChoices)},
		probe: fmt.Sprintf(`{"model":"m","messages":[{"role":"user","content":"hi"}],"n":%d}`, maxChoices+1)},
	{Refusal: capabilities.Refusal{ID: "top-logprobs-range", Endpoints: []string{"/v1/chat/completions"},
		When:    fmt.Sprintf("top_logprobs is outside 0..%d", model.MaxTopLogprobs),
		Message: fmt.Sprintf("top_logprobs must be between 0 and %d", model.MaxTopLogprobs)},
		probe: fmt.Sprintf(`{"model":"m","messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":%d}`, model.MaxTopLogprobs+1)},
	{Refusal: capabilities.Refusal{ID: "tool-type", Endpoints: []string{"/v1/chat/completions"},
		When: "a tool is not {type: function} with a name", Message: `only {"type": "function"} with a name is supported`},
		probe: `{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"retrieval"}]}`},
	{Refusal: capabilities.Refusal{ID: "tool-choice-undeclared", Endpoints: []string{"/v1/chat/completions", "/v1/messages"},
		When: "tool_choice names a tool the request does not declare", Message: "which is not among the request's tools"},
		probe: `{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"a"}}],"tool_choice":{"type":"function","function":{"name":"b"}}}`},
	{Refusal: capabilities.Refusal{ID: "completions-list-stream", Endpoints: []string{"/v1/completions"},
		When: "a list of prompts with stream", Message: "a list of prompts cannot be streamed"},
		probe: `{"model":"m","prompt":["a","b"],"stream":true}`},
	{Refusal: capabilities.Refusal{ID: "completions-logprobs-echo", Endpoints: []string{"/v1/completions"},
		When: "logprobs with echo", Message: "logprobs with echo would need the prompt's own"},
		probe: `{"model":"m","prompt":"a","logprobs":1,"echo":true}`},
	{Refusal: capabilities.Refusal{ID: "completions-list-logprobs-n", Endpoints: []string{"/v1/completions"},
		When: "logprobs or n with a list of prompts", Message: "logprobs and n with a list of prompts are not built"},
		probe: `{"model":"m","prompt":["a","b"],"logprobs":1}`},
	{Refusal: capabilities.Refusal{ID: "embedding-format", Endpoints: []string{"/v1/embeddings"},
		When: "encoding_format is neither float nor base64", Message: "is not one of"},
		probe: `{"model":"m","input":"a","encoding_format":"int8"}`},
	{Refusal: capabilities.Refusal{ID: "embedding-dimensions", Endpoints: []string{"/v1/embeddings"},
		When: "dimensions is below 1", Message: "dimensions must be at least 1"},
		probe: `{"model":"m","input":"a","dimensions":0}`},
	{Refusal: capabilities.Refusal{ID: "anthropic-max-tokens", Endpoints: []string{"/v1/messages"},
		When: "max_tokens is absent or not positive", Message: "max_tokens is required and must be positive"},
		probe: `{"model":"m","messages":[{"role":"user","content":"hi"}]}`},
	{Refusal: capabilities.Refusal{ID: "speculation-logprobs", Endpoints: []string{"/v1/chat/completions", "/v1/completions"},
		When: "speculative decoding with logprobs", Message: "speculation with logprobs is not built"}},
	{Refusal: capabilities.Refusal{ID: "speculation-n", Endpoints: []string{"/v1/chat/completions", "/v1/completions"},
		When: "speculative decoding with n above 1", Message: "speculation with several choices is not built"}},
	{Refusal: capabilities.Refusal{ID: "n-session", Endpoints: []string{"/v1/chat/completions", "/v1/completions"},
		When: "n above 1 with jitllm_session", Message: "several choices run on fresh sessions of a model"}},
	{Refusal: capabilities.Refusal{ID: "forced-tool-ignore-eos", Endpoints: []string{"/v1/chat/completions", "/v1/messages"},
		When:    "a forced tool call (tool_choice required, any or named) with ignore_eos",
		Message: "a forced tool call ends its reply with an end-of-generation token"}},
	{Refusal: capabilities.Refusal{ID: "grammar-ignore-eos", Endpoints: []string{"/v1/chat/completions", "/v1/completions"},
		When:    "response_format json_object or json_schema with ignore_eos",
		Message: "a grammar ends its reply with an end-of-generation token"}},
	{Refusal: capabilities.Refusal{ID: "embedding-non-embedder", Endpoints: []string{"/v1/embeddings"},
		When: "the model is not an embedding model", Message: "is not an embedding model"}},
	{Refusal: capabilities.Refusal{ID: "unsupported-part", Endpoints: []string{"/v1/chat/completions"},
		When: "a content part of type input_audio, file or input_file", Message: "which this server does not take"},
		probe: `{"model":"m","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{}}]}]}`},
	{Refusal: capabilities.Refusal{ID: "document-block", Endpoints: []string{"/v1/messages"},
		When: "a document content block", Message: "is a document block, which this server does not take"}},
	{Refusal: capabilities.Refusal{ID: "image-non-vision", Endpoints: []string{"/v1/chat/completions", "/v1/messages"},
		When: "a picture for a model whose container carries no vision tower", Message: "does not accept images"}},
	{Refusal: capabilities.Refusal{ID: "image-fetch-off", Endpoints: []string{"/v1/chat/completions", "/v1/messages"},
		When:    "a picture named by a remote URL on a server that lists no host to fetch from",
		Message: "names a picture by a remote URL, and this server does not fetch"}},
	{Refusal: capabilities.Refusal{ID: "image-too-many", Endpoints: []string{"/v1/chat/completions", "/v1/messages"},
		When: fmt.Sprintf("more pictures than the server's limit (default %d)", DefaultImageMaxImages), Message: "a request carries at most"}},
	{Refusal: capabilities.Refusal{ID: "image-oversize", Endpoints: []string{"/v1/chat/completions", "/v1/messages"},
		When: "a picture over the byte or pixel limit, or a request's pictures over the request byte limit", Message: "over the limit of"}},
	{Refusal: capabilities.Refusal{ID: "image-format", Endpoints: []string{"/v1/chat/completions", "/v1/messages"},
		When: "a picture that is not PNG, JPEG, WebP or GIF", Message: "the pictures accepted are"}},
	{Refusal: capabilities.Refusal{ID: "decision-non-decider", Endpoints: []string{"/v1/systemone"},
		When: "the model is not a decision model", Message: "is not a decision model"}},
}

// API is the server's half of the manifest, from the tables above and the
// generated ConnectRPC descriptors.
func API() capabilities.API {
	var a capabilities.API
	for _, r := range compatRoutes {
		a.Endpoints = append(a.Endpoints, capabilities.Endpoint{Path: r.path, Methods: r.methods,
			Surface: r.surface, Note: r.note, Fields: fieldsOf(r.path, r.req)})
	}
	for _, r := range engineRoutes() {
		a.Endpoints = append(a.Endpoints, capabilities.Endpoint{Path: r.path, Methods: r.methods,
			Surface: "jitllm", Note: r.note})
	}
	for _, r := range refusals {
		a.Refusals = append(a.Refusals, r.Refusal)
	}
	a.Control = controlPlane()
	return a
}

// fieldsOf is a request struct's JSON fields, embedded structs flattened, in
// declaration order, each with its note.
func fieldsOf(path string, req any) []capabilities.Field {
	if req == nil {
		return nil
	}
	var out []capabilities.Field
	var walk func(t reflect.Type)
	walk = func(t reflect.Type) {
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if f.Anonymous && f.Type.Kind() == reflect.Struct {
				walk(f.Type)
				continue
			}
			name, _, _ := strings.Cut(f.Tag.Get("json"), ",")
			if name == "" || name == "-" || !f.IsExported() {
				continue
			}
			out = append(out, capabilities.Field{Name: name, Type: wireType(f.Type), Note: fieldNotes[path][name]})
		}
	}
	walk(reflect.TypeOf(req))
	return out
}

// wireType names a Go field's JSON shape.
func wireType(t reflect.Type) string {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.String() == "json.RawMessage" {
		return "json"
	}
	switch t.Kind() {
	case reflect.String:
		return "string"
	case reflect.Bool:
		return "bool"
	case reflect.Int, reflect.Int32, reflect.Int64:
		return "integer"
	case reflect.Float32, reflect.Float64:
		return "number"
	case reflect.Slice:
		return "array"
	case reflect.Map, reflect.Struct:
		return "object"
	}
	return t.Kind().String()
}

// controlPlane lists the jitllm.v1 services and their methods from the
// descriptors the generated code registers (linked through the
// jitllmv1connect handlers Handler mounts).
func controlPlane() []capabilities.Service {
	var out []capabilities.Service
	protoregistry.GlobalFiles.RangeFilesByPackage("jitllm.v1", func(fd protoreflect.FileDescriptor) bool {
		ss := fd.Services()
		for i := 0; i < ss.Len(); i++ {
			s := ss.Get(i)
			svc := capabilities.Service{Name: string(s.FullName())}
			ms := s.Methods()
			for j := 0; j < ms.Len(); j++ {
				svc.Methods = append(svc.Methods, string(ms.Get(j).Name()))
			}
			out = append(out, svc)
		}
		return true
	})
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}
