package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"net/http"
	"strings"
	"unicode/utf8"

	"github.com/jitllm/jitllm/engine/grammar"
	"github.com/jitllm/jitllm/engine/model"
)

// llama.cpp's native HTTP API (tools/server): /completion, /infill,
// /tokenize, /detokenize, /apply-template, /embedding, /props, /slots and a
// /health in its shape. These are adapters like the OpenAI and Anthropic
// shims: /completion and /infill build a GenerateOptions and call
// Backend.Generate, /embedding calls Backend.Embed, so there is no second
// generation path. What the tokenizer and the template need beyond the
// Backend is LlamaBackend.
//
// llama-server serves one model, so a request names none. Here the model is
// the one loaded; with several, a `model` field (an extension llama.cpp
// ignores) picks one, and without it the request is refused naming the field.
// Every request field is either honoured, accepted because its value is the
// default and changes nothing, or refused by name (llamaFields).

// LlamaBackend is what the llama.cpp routes read beyond Backend: the model's
// tokenizer, its chat template and its props. *Engine implements it; a Backend
// that does not has the routes answer 501.
type LlamaBackend interface {
	// LlamaModel resolves a model name onto its props. The name is one
	// ListLoaded or ListAvailable gave.
	LlamaModel(name string) (LlamaModelInfo, error)
	// Tokenize encodes text. addSpecial puts the BOS the vocabulary adds;
	// parseSpecial matches control tokens in the text as themselves.
	Tokenize(model, text string, addSpecial, parseSpecial bool) ([]int32, error)
	// Piece is the text one token stands for, as bytes.
	Piece(model string, id int32) ([]byte, error)
	Detokenize(model string, ids []int32) (string, error)
	// ApplyTemplate renders messages (and tools, a JSON array) through the
	// model's chat template with the generation prompt, as /apply-template.
	ApplyTemplate(model string, msgs []model.ChatMessage, tools []byte) (string, error)
	// Slots is the sessions of a model, read only.
	Slots(model string) []LlamaSlot
}

// LlamaModelInfo is a model's props.
type LlamaModelInfo struct {
	ID, Name, Path string
	NCtx           int
	ChatTemplate   string
	Vision         bool
	Embedding      bool
	Version        string
	FIM            *FIMTokens
}

// FIMTokens are the fill-in-the-middle control tokens a vocabulary has. Repo
// and Sep are -1 when absent.
type FIMTokens struct {
	Pre, Suf, Mid, Repo, Sep int32
}

// LlamaSlot is one session as /slots reports it.
type LlamaSlot struct {
	ID         string
	NCtx       int
	Processing bool
}

type llamaField struct {
	ok  func(json.RawMessage) bool // llamaDefault: is this the neutral value
	why string                     // the refusal's reason
}

func numIs(want float64) func(json.RawMessage) bool {
	return func(r json.RawMessage) bool {
		var f float64
		return json.Unmarshal(r, &f) == nil && f == want
	}
}

func numLE(want float64) func(json.RawMessage) bool {
	return func(r json.RawMessage) bool {
		var f float64
		return json.Unmarshal(r, &f) == nil && f <= want
	}
}

func boolIs(want bool) func(json.RawMessage) bool {
	return func(r json.RawMessage) bool {
		var b bool
		return json.Unmarshal(r, &b) == nil && b == want
	}
}

func emptyJSON(r json.RawMessage) bool {
	t := strings.TrimSpace(string(r))
	return t == "null" || t == "[]" || t == "{}" || t == `""`
}

func anyValue(json.RawMessage) bool { return true }

// samplerSubset is the samplers this engine's sampler chain has.
func samplerSubset(r json.RawMessage) bool {
	var names []string
	if json.Unmarshal(r, &names) != nil {
		return false
	}
	for _, n := range names {
		switch n {
		case "penalties", "top_k", "top_p", "min_p", "temperature":
		default:
			return false
		}
	}
	return true
}

// llamaFields is every /completion field (llama.cpp's server README) that is
// not read: each is accepted at its neutral value and refused at any other.
// A field in neither this table nor llamaRead is refused as unknown.
var llamaFields = map[string]llamaField{
	"dynatemp_range":        {numIs(0), "dynamic temperature is not built"},
	"dynatemp_exponent":     {anyValue, ""},
	"typical_p":             {numIs(1), "locally typical sampling is not built"},
	"presence_penalty":      {numIs(0), "the sampler has no presence penalty; repeat_penalty is built"},
	"frequency_penalty":     {numIs(0), "the sampler has no frequency penalty; repeat_penalty is built"},
	"dry_multiplier":        {numIs(0), "DRY sampling is not built"},
	"dry_base":              {anyValue, ""},
	"dry_allowed_length":    {anyValue, ""},
	"dry_penalty_last_n":    {anyValue, ""},
	"dry_sequence_breakers": {anyValue, ""},
	"xtc_probability":       {numIs(0), "XTC sampling is not built"},
	"xtc_threshold":         {anyValue, ""},
	"mirostat":              {numIs(0), "Mirostat is not built"},
	"mirostat_tau":          {anyValue, ""},
	"mirostat_eta":          {anyValue, ""},
	"top_n_sigma":           {numLE(-1), "top-n-sigma sampling is not built"},
	"adaptive_target":       {numLE(-1), "adaptive-p sampling is not built"},
	"adaptive_decay":        {anyValue, ""},
	"min_keep":              {numIs(0), "min_keep is not built"},
	"logit_bias":            {emptyJSON, "logit bias is not built"},
	"samplers":              {samplerSubset, "only the samplers penalties, top_k, top_p, min_p and temperature exist"},
	"n_indent":              {numIs(0), "n_indent is not built"},
	"n_keep":                {numIs(0), "there is no context shift, so n_keep has nothing to keep"},
	"n_discard":             {numIs(0), "there is no context shift"},
	"n_cmpl":                {numIs(1), "send one request per completion"},
	"n_cache_reuse":         {numIs(0), "KV shifting reuse is not built; the prefix cache is automatic"},
	"t_max_predict_ms":      {numIs(0), "a prediction time limit is not built; use n_predict"},
	"t_max_prompt_ms":       {numIs(0), "a prompt time limit is not built"},
	"timings_per_token":     {boolIs(false), "per-token timings are not built; the final object carries timings"},
	"return_progress":       {boolIs(false), "prompt progress is not built"},
	"post_sampling_probs":   {boolIs(false), "probabilities after sampling are not built; n_probs reports the raw distribution"},
	"response_fields":       {emptyJSON, "response_fields is not built"},
	"lora":                  {emptyJSON, "LoRA adapters are not built"},
	"sse_ping_interval":     {numIs(0), "SSE pings are not built"},
	"multimodal_data":       {emptyJSON, "send pictures through /v1/chat/completions"},
	"image_data":            {emptyJSON, "send pictures through /v1/chat/completions"},
}

// llamaRead are the fields the handler reads.
var llamaRead = map[string]bool{
	"prompt": true, "temperature": true, "top_k": true, "top_p": true, "min_p": true,
	"n_predict": true, "max_tokens": true, "stop": true, "seed": true, "stream": true,
	"cache_prompt": true, "n_probs": true, "grammar": true, "json_schema": true,
	"id_slot": true, "ignore_eos": true, "return_tokens": true, "repeat_penalty": true,
	"repeat_last_n": true, "model": true,
	// /infill's own.
	"input_prefix": true, "input_suffix": true, "input_extra": true,
}

// llamaCheck refuses a field the handler does not honour, naming it. Fields
// are checked in name order so the message is stable.
func llamaCheck(body map[string]json.RawMessage, infill bool) error {
	names := make([]string, 0, len(body))
	for k := range body {
		names = append(names, k)
	}
	sortStrings(names)
	for _, k := range names {
		if llamaRead[k] {
			if !infill && strings.HasPrefix(k, "input_") {
				return fmt.Errorf("%w: %s belongs to /infill", ErrInvalid, k)
			}
			continue
		}
		f, ok := llamaFields[k]
		if !ok {
			return fmt.Errorf("%w: %s is not a field this server reads", ErrInvalid, k)
		}
		if !f.ok(body[k]) {
			return fmt.Errorf("%w: %s=%s is not supported: %s", ErrInvalid, k, strings.TrimSpace(string(body[k])), f.why)
		}
	}
	return nil
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

// llFail is llama-server's error body: {"error": {code, message, type}}.
func llFail(w http.ResponseWriter, status int, msg, typ string) {
	writeJSON(w, status, map[string]any{"error": map[string]any{"code": status, "message": msg, "type": typ}})
}

func llFailErr(w http.ResponseWriter, err error) {
	status, typ := errClass(w, err)
	llFail(w, status, err.Error(), typ)
}

// llamaBackend is the LlamaBackend behind the compat shim, or a 501.
func (c *compat) llamaBackend(w http.ResponseWriter) (LlamaBackend, bool) {
	lb, ok := c.b.(LlamaBackend)
	if !ok {
		llFail(w, http.StatusNotImplemented, "this backend serves no llama.cpp routes", "not_supported_error")
	}
	return lb, ok
}

// llamaModelName is the model a llama.cpp request addresses: the `model`
// field when given, otherwise the only model there is.
func (c *compat) llamaModelName(requested string) (string, error) {
	if requested != "" {
		return requested, nil
	}
	loaded := c.b.ListLoaded()
	if len(loaded) == 1 {
		return loaded[0].ID, nil
	}
	if len(loaded) == 0 {
		if avail := c.b.ListAvailable(); len(avail) == 1 {
			return avail[0].ID, nil
		}
	}
	return "", fmt.Errorf("%w: %d models are loaded and llama.cpp's API names none; "+
		"add a \"model\" field (an id from /v1/models)", ErrInvalid, len(loaded))
}

// llamaPre is the start of every llama.cpp handler: method, backend, model.
func (c *compat) llamaPre(w http.ResponseWriter, r *http.Request, method, requested string) (LlamaBackend, LlamaModelInfo, bool) {
	var zero LlamaModelInfo
	if r.Method != method {
		llFail(w, http.StatusMethodNotAllowed, method+" only", "invalid_request_error")
		return nil, zero, false
	}
	lb, ok := c.llamaBackend(w)
	if !ok {
		return nil, zero, false
	}
	name, err := c.llamaModelName(requested)
	if err != nil {
		llFailErr(w, err)
		return nil, zero, false
	}
	info, err := lb.LlamaModel(name)
	if err != nil {
		llFailErr(w, err)
		return nil, zero, false
	}
	return lb, info, true
}

// readLlamaBody decodes a JSON object body into its raw fields.
func readLlamaBody(w http.ResponseWriter, r *http.Request) (map[string]json.RawMessage, bool) {
	var body map[string]json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		llFail(w, http.StatusBadRequest, "invalid JSON body: "+err.Error(), "invalid_request_error")
		return nil, false
	}
	return body, true
}

func modelField(body map[string]json.RawMessage) string {
	var s string
	json.Unmarshal(body["model"], &s)
	return s
}

// ---- /tokenize, /detokenize, /apply-template

func (c *compat) llamaTokenize(w http.ResponseWriter, r *http.Request) {
	body, ok := c.llamaJSON(w, r, http.MethodPost)
	if !ok {
		return
	}
	lb, info, ok := c.llamaPre(w, r, http.MethodPost, modelField(body))
	if !ok {
		return
	}
	var content string
	if raw, has := body["content"]; !has || json.Unmarshal(raw, &content) != nil {
		llFail(w, http.StatusBadRequest, "content (a string) is required", "invalid_request_error")
		return
	}
	addSpecial, parseSpecial, pieces := false, true, false
	for k, dst := range map[string]*bool{"add_special": &addSpecial, "parse_special": &parseSpecial, "with_pieces": &pieces} {
		if raw, has := body[k]; has && json.Unmarshal(raw, dst) != nil {
			llFail(w, http.StatusBadRequest, k+" must be a boolean", "invalid_request_error")
			return
		}
	}
	for k := range body {
		switch k {
		case "content", "add_special", "parse_special", "with_pieces", "model":
		default:
			llFailErr(w, fmt.Errorf("%w: %s is not a field /tokenize reads", ErrInvalid, k))
			return
		}
	}
	ids, err := lb.Tokenize(info.ID, content, addSpecial, parseSpecial)
	if err != nil {
		llFailErr(w, err)
		return
	}
	if !pieces {
		writeJSON(w, http.StatusOK, map[string]any{"tokens": append([]int32{}, ids...)})
		return
	}
	out := make([]any, len(ids))
	for i, id := range ids {
		p, err := lb.Piece(info.ID, id)
		if err != nil {
			llFailErr(w, err)
			return
		}
		if utf8.Valid(p) {
			out[i] = map[string]any{"id": id, "piece": string(p)}
		} else {
			out[i] = map[string]any{"id": id, "piece": oaBytes(string(p))}
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"tokens": out})
}

// llamaJSON reads the body of a POST route before the model is resolved, so
// the model field can name it. A body is put back for nothing: handlers take
// the decoded map.
func (c *compat) llamaJSON(w http.ResponseWriter, r *http.Request, method string) (map[string]json.RawMessage, bool) {
	if r.Method != method {
		llFail(w, http.StatusMethodNotAllowed, method+" only", "invalid_request_error")
		return nil, false
	}
	return readLlamaBody(w, r)
}

func (c *compat) llamaDetokenize(w http.ResponseWriter, r *http.Request) {
	body, ok := c.llamaJSON(w, r, http.MethodPost)
	if !ok {
		return
	}
	lb, info, ok := c.llamaPre(w, r, http.MethodPost, modelField(body))
	if !ok {
		return
	}
	var ids []int32
	if raw, has := body["tokens"]; !has || json.Unmarshal(raw, &ids) != nil {
		llFail(w, http.StatusBadRequest, "tokens (an array of token ids) is required", "invalid_request_error")
		return
	}
	for k := range body {
		if k != "tokens" && k != "model" {
			llFailErr(w, fmt.Errorf("%w: %s is not a field /detokenize reads", ErrInvalid, k))
			return
		}
	}
	text, err := lb.Detokenize(info.ID, ids)
	if err != nil {
		llFailErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"content": text})
}

func (c *compat) llamaApplyTemplate(w http.ResponseWriter, r *http.Request) {
	body, ok := c.llamaJSON(w, r, http.MethodPost)
	if !ok {
		return
	}
	lb, info, ok := c.llamaPre(w, r, http.MethodPost, modelField(body))
	if !ok {
		return
	}
	for k := range body {
		switch k {
		case "messages", "tools", "model":
		default:
			llFailErr(w, fmt.Errorf("%w: %s is not a field /apply-template reads", ErrInvalid, k))
			return
		}
	}
	var msgs []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if raw, has := body["messages"]; !has || json.Unmarshal(raw, &msgs) != nil || len(msgs) == 0 {
		llFail(w, http.StatusBadRequest, "messages (an array of {role, content}) is required", "invalid_request_error")
		return
	}
	var cms []model.ChatMessage
	for i, m := range msgs {
		text, imgs, err := oaContent(m.Content)
		if err != nil || len(imgs) > 0 {
			llFail(w, http.StatusBadRequest, fmt.Sprintf("messages[%d]: content must be text", i), "invalid_request_error")
			return
		}
		cms = append(cms, chatMessage(m.Role, text))
	}
	tools, err := oaTools(body["tools"], nil)
	if err != nil {
		llFailErr(w, fmt.Errorf("%w: %v", ErrInvalid, err))
		return
	}
	prompt, err := lb.ApplyTemplate(info.ID, cms, tools)
	if err != nil {
		llFailErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"prompt": prompt})
}

// ---- /embedding

func (c *compat) llamaEmbedding(w http.ResponseWriter, r *http.Request) {
	body, ok := c.llamaJSON(w, r, http.MethodPost)
	if !ok {
		return
	}
	_, info, ok := c.llamaPre(w, r, http.MethodPost, modelField(body))
	if !ok {
		return
	}
	for k, raw := range body {
		switch k {
		case "content", "model":
		case "embd_normalize":
			if !numIs(2)(raw) {
				llFailErr(w, fmt.Errorf("%w: embd_normalize=%s is not supported: vectors are L2-normalised (2)", ErrInvalid, raw))
				return
			}
		default:
			llFailErr(w, fmt.Errorf("%w: %s is not a field /embedding reads", ErrInvalid, k))
			return
		}
	}
	inputs, err := oaEmbedInputs(body["content"])
	if err != nil {
		llFail(w, http.StatusBadRequest, "content: "+err.Error(), "invalid_request_error")
		return
	}
	res, err := c.b.Embed(r.Context(), EmbedOptions{Model: info.ID, Inputs: inputs})
	if err != nil {
		llFailErr(w, err)
		return
	}
	// llama-server's reply is an array with one entry per input, its
	// embedding an array of vectors (one per input row, one when pooled).
	out := make([]any, len(res.Vectors))
	for i, v := range res.Vectors {
		out[i] = map[string]any{"index": i, "embedding": [][]float32{v}}
	}
	writeJSON(w, http.StatusOK, out)
}

// ---- /health, /props, /slots

func (c *compat) llamaProps(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		llFail(w, http.StatusNotImplemented, "This server does not support changing global properties. "+
			"Start it with `--props`", "not_supported_error")
		return
	}
	var requested string
	requested = r.URL.Query().Get("model")
	lb, info, ok := c.llamaPre(w, r, http.MethodGet, requested)
	if !ok {
		return
	}
	slots := lb.Slots(info.ID)
	busy := false
	for _, s := range slots {
		busy = busy || s.Processing
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"default_generation_settings": map[string]any{
			"id": 0, "id_task": -1, "n_ctx": info.NCtx, "speculative": false, "is_processing": busy,
			"params": llamaParams(llamaDefaults(), 0, nil, info.NCtx),
			"prompt": "",
			"next_token": map[string]any{"has_next_token": true, "has_new_line": false, "n_remain": -1,
				"n_decoded": 0, "stopping_word": ""},
		},
		"total_slots":        max(len(slots), 1),
		"model_path":         info.Path,
		"model_alias":        info.Name,
		"chat_template":      info.ChatTemplate,
		"chat_template_caps": map[string]any{},
		"modalities":         map[string]any{"vision": info.Vision},
		"build_info":         info.Version,
		"is_sleeping":        false,
	})
}

func (c *compat) llamaSlots(w http.ResponseWriter, r *http.Request) {
	lb, info, ok := c.llamaPre(w, r, http.MethodGet, r.URL.Query().Get("model"))
	if !ok {
		return
	}
	slots := lb.Slots(info.ID)
	if len(slots) == 0 {
		// A server with no session yet has the slot a request would open.
		slots = []LlamaSlot{{ID: "0", NCtx: info.NCtx}}
	}
	out := make([]any, len(slots))
	for i, s := range slots {
		out[i] = map[string]any{
			"id": i, "id_task": -1, "n_ctx": s.NCtx, "speculative": false, "is_processing": s.Processing,
			"session": s.ID,
			"params":  llamaParams(llamaDefaults(), 0, nil, s.NCtx),
			"next_token": map[string]any{"has_next_token": s.Processing, "has_new_line": false, "n_remain": -1,
				"n_decoded": 0, "stopping_word": ""},
		}
	}
	writeJSON(w, http.StatusOK, out)
}

// serveLlamaHealth is llama-server's /health: {"status": "ok"}. /healthz stays
// the plain-text probe.
func serveLlamaHealth(_ *Engine, w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// ---- /completion and /infill

// llamaSettings is a request's resolved generation settings, as
// generation_settings reports them.
type llamaSettings struct {
	Temperature, TopP, MinP, RepeatPenalty float64
	TopK, RepeatLastN, NPredict, NProbs    int
	Seed                                   int64
	Stop                                   []string
	IgnoreEOS, Stream                      bool
	Grammar, JSONSchema                    string
}

// llamaDefaults are llama-server's defaults for an absent field.
func llamaDefaults() llamaSettings {
	return llamaSettings{Temperature: 0.8, TopP: 0.95, MinP: 0.05, TopK: 40, RepeatPenalty: 1,
		RepeatLastN: 64, NPredict: -1, Seed: -1, Stop: []string{}}
}

func llamaParams(s llamaSettings, _ int, _ []int32, _ int) map[string]any {
	return map[string]any{
		"n_predict": s.NPredict, "max_tokens": s.NPredict, "seed": s.Seed, "temperature": s.Temperature,
		"top_k": s.TopK, "top_p": s.TopP, "min_p": s.MinP, "repeat_last_n": s.RepeatLastN,
		"repeat_penalty": s.RepeatPenalty, "stop": s.Stop, "ignore_eos": s.IgnoreEOS, "stream": s.Stream,
		"n_probs": s.NProbs, "grammar": s.Grammar, "json_schema": s.JSONSchema,
		"samplers": []string{"penalties", "top_k", "top_p", "min_p", "temperature"},
	}
}

func decodeField[T any](body map[string]json.RawMessage, name string, dst *T) error {
	raw, ok := body[name]
	if !ok || string(raw) == "null" {
		return nil
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		return fmt.Errorf("%w: %s: %v", ErrInvalid, name, err)
	}
	return nil
}

// llamaPrompt reads `prompt`: a string, token ids, or strings and ids mixed,
// as one token sequence. Strings are tokenized with special tokens parsed, and
// only a leading string gets the BOS, as llama-server does.
func llamaPrompt(lb LlamaBackend, modelID string, raw json.RawMessage) (ids []int32, text string, err error) {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		ids, err = lb.Tokenize(modelID, s, true, true)
		return ids, s, err
	}
	var parts []json.RawMessage
	if json.Unmarshal(raw, &parts) != nil {
		var obj struct {
			PromptString string          `json:"prompt_string"`
			Multimodal   json.RawMessage `json:"multimodal_data"`
		}
		if json.Unmarshal(raw, &obj) == nil && obj.PromptString != "" {
			if !emptyJSON(obj.Multimodal) && len(obj.Multimodal) > 0 {
				return nil, "", fmt.Errorf("%w: prompt.multimodal_data is not supported: send pictures through /v1/chat/completions", ErrInvalid)
			}
			ids, err = lb.Tokenize(modelID, obj.PromptString, true, true)
			return ids, obj.PromptString, err
		}
		return nil, "", fmt.Errorf("%w: prompt must be a string, token ids, or strings and ids mixed", ErrInvalid)
	}
	for i, p := range parts {
		var n int32
		var str string
		var many []json.RawMessage
		switch {
		case json.Unmarshal(p, &n) == nil:
			ids = append(ids, n)
		case json.Unmarshal(p, &str) == nil:
			t, err := lb.Tokenize(modelID, str, i == 0, true)
			if err != nil {
				return nil, "", err
			}
			ids = append(ids, t...)
			text += str
		case json.Unmarshal(p, &many) == nil:
			return nil, "", fmt.Errorf("%w: a list of prompts is not supported: send one request per prompt", ErrInvalid)
		default:
			return nil, "", fmt.Errorf("%w: prompt[%d] is neither a token id nor a string", ErrInvalid, i)
		}
	}
	if len(ids) == 0 {
		return nil, "", fmt.Errorf("%w: prompt is empty", ErrInvalid)
	}
	return ids, text, nil
}

func (c *compat) llamaCompletion(w http.ResponseWriter, r *http.Request) {
	c.llamaGenerate(w, r, false)
}

func (c *compat) llamaInfill(w http.ResponseWriter, r *http.Request) {
	c.llamaGenerate(w, r, true)
}

func (c *compat) llamaGenerate(w http.ResponseWriter, r *http.Request, infill bool) {
	body, ok := c.llamaJSON(w, r, http.MethodPost)
	if !ok {
		return
	}
	lb, info, ok := c.llamaPre(w, r, http.MethodPost, modelField(body))
	if !ok {
		return
	}
	if err := llamaCheck(body, infill); err != nil {
		llFailErr(w, err)
		return
	}
	set := llamaDefaults()
	var temp, topP, minP, repeat *float64
	var topK, lastN, nPredict, maxTokens, nProbs, slot *int
	var seed *int64
	var stop json.RawMessage
	var grammarSrc, schema json.RawMessage
	var ignore, stream, returnTokens bool
	var firstErr error
	rd := func(err error) {
		if firstErr == nil {
			firstErr = err
		}
	}
	rd(decodeField(body, "temperature", &temp))
	rd(decodeField(body, "top_p", &topP))
	rd(decodeField(body, "min_p", &minP))
	rd(decodeField(body, "repeat_penalty", &repeat))
	rd(decodeField(body, "top_k", &topK))
	rd(decodeField(body, "repeat_last_n", &lastN))
	rd(decodeField(body, "n_predict", &nPredict))
	rd(decodeField(body, "max_tokens", &maxTokens))
	rd(decodeField(body, "n_probs", &nProbs))
	rd(decodeField(body, "id_slot", &slot))
	rd(decodeField(body, "seed", &seed))
	rd(decodeField(body, "ignore_eos", &ignore))
	rd(decodeField(body, "stream", &stream))
	rd(decodeField(body, "return_tokens", &returnTokens))
	stop, grammarSrc, schema = body["stop"], body["grammar"], body["json_schema"]
	if firstErr != nil {
		llFailErr(w, firstErr)
		return
	}
	if temp != nil {
		set.Temperature = *temp
	}
	if topP != nil {
		set.TopP = *topP
	}
	if minP != nil {
		set.MinP = *minP
	}
	if repeat != nil {
		set.RepeatPenalty = *repeat
	}
	if topK != nil {
		set.TopK = *topK
	}
	if lastN != nil {
		set.RepeatLastN = *lastN
	}
	if seed != nil {
		set.Seed = *seed
	}
	set.IgnoreEOS, set.Stream = ignore, stream
	// n_predict wins over the max_tokens alias; -1 and -2 are "until done".
	switch {
	case nPredict != nil:
		set.NPredict = *nPredict
	case maxTokens != nil:
		set.NPredict = *maxTokens
	}
	if set.NPredict == 0 {
		llFailErr(w, fmt.Errorf("%w: n_predict=0 is not supported: it asks for no tokens", ErrInvalid))
		return
	}
	if slot != nil && *slot != -1 && *slot != 0 {
		llFailErr(w, fmt.Errorf("%w: id_slot=%d is not supported: there is one slot, 0", ErrInvalid, *slot))
		return
	}
	if nProbs != nil {
		set.NProbs = *nProbs
	}
	if set.NProbs < 0 || set.NProbs > model.MaxTopLogprobs {
		llFailErr(w, fmt.Errorf("%w: n_probs=%d is outside [0, %d]", ErrInvalid, set.NProbs, model.MaxTopLogprobs))
		return
	}
	if set.RepeatLastN != 64 && set.RepeatLastN != 0 {
		llFailErr(w, fmt.Errorf("%w: repeat_last_n=%d is not supported: the repeat penalty looks back 64 tokens (0 disables it)",
			ErrInvalid, set.RepeatLastN))
		return
	}
	if len(stop) > 0 && string(stop) != "null" {
		var many []string
		var one string
		switch {
		case json.Unmarshal(stop, &many) == nil:
			set.Stop = many
		case json.Unmarshal(stop, &one) == nil:
			set.Stop = []string{one}
		default:
			llFailErr(w, fmt.Errorf("%w: stop must be an array of strings", ErrInvalid))
			return
		}
	}
	fs := samplingFields{Temperature: &set.Temperature, TopP: &set.TopP, MinP: &set.MinP,
		RepetitionPenalty: &set.RepeatPenalty, TopK: &set.TopK}
	if set.TopK == 0 {
		fs.TopK = nil // llama.cpp's 0 is "disabled"
	}
	if err := fs.validate(); err != nil {
		llFailErr(w, fmt.Errorf("%w: %v", ErrInvalid, err))
		return
	}

	// The prompt.
	var ids []int32
	var promptText string
	var err error
	if infill {
		ids, promptText, err = llamaInfillIDs(lb, info, body)
	} else {
		if _, has := body["prompt"]; !has {
			llFail(w, http.StatusBadRequest, "prompt is required", "invalid_request_error")
			return
		}
		ids, promptText, err = llamaPrompt(lb, info.ID, body["prompt"])
	}
	if err != nil {
		llFailErr(w, err)
		return
	}

	o := GenerateOptions{
		Prompt:    Prompt{Kind: PromptIDs, IDs: ids},
		Stop:      set.Stop,
		IgnoreEOS: set.IgnoreEOS,
		Sampling:  llamaSampler(set),
		Logprobs:  set.NProbs > 0,
	}
	o.TopLogprobs = set.NProbs
	if set.NPredict > 0 {
		o.MaxTokens = set.NPredict
	}
	if len(grammarSrc) > 0 && !emptyJSON(grammarSrc) {
		if err := json.Unmarshal(grammarSrc, &set.Grammar); err != nil {
			llFailErr(w, fmt.Errorf("%w: grammar must be a string", ErrInvalid))
			return
		}
	}
	if len(schema) > 0 && !emptyJSON(schema) {
		if set.Grammar != "" {
			llFailErr(w, fmt.Errorf("%w: grammar and json_schema together: send one", ErrInvalid))
			return
		}
		set.JSONSchema = string(schema)
		if set.Grammar, err = grammar.FromJSONSchema(schema); err != nil {
			llFailErr(w, fmt.Errorf("%w: json_schema: %v", ErrInvalid, err))
			return
		}
	}
	o.Grammar = set.Grammar
	if o.IgnoreEOS && o.Grammar != "" {
		llFailErr(w, fmt.Errorf("%w: ignore_eos with a grammar", ErrInvalid))
		return
	}
	if err := c.b.BindTarget(&o, "", info.ID); err != nil {
		llFailErr(w, err)
		return
	}
	run := &llamaRun{c: c, w: w, r: r, o: o, info: info, set: set, promptText: promptText,
		promptIDs: ids, returnTokens: returnTokens}
	if stream {
		run.stream()
		return
	}
	run.whole()
}

// llamaSampler is the engine's sampler for a request's settings. Temperature 0
// is greedy, as in llama.cpp; an unset seed (-1) draws its own.
func llamaSampler(s llamaSettings) *model.Sampler {
	sm := &model.Sampler{Temp: s.Temperature, TopP: s.TopP, MinP: s.MinP, TopK: max(s.TopK, 0),
		RepeatPen: s.RepeatPenalty, Seed: s.Seed}
	if s.RepeatLastN == 0 {
		sm.RepeatPen = 1
	}
	if s.Seed < 0 {
		sm.Seed = rand.Int64()
	}
	return sm
}

// llamaInfillIDs builds /infill's prompt from the model's fill-in-the-middle
// tokens, as the README lays it out. A model without them is refused.
func llamaInfillIDs(lb LlamaBackend, info LlamaModelInfo, body map[string]json.RawMessage) ([]int32, string, error) {
	fim := info.FIM
	if fim == nil {
		return nil, "", fmt.Errorf("%w: model %q has no fill-in-the-middle tokens, so /infill cannot run", ErrInvalid, info.Name)
	}
	var pre, suf, prompt string
	var extra []struct {
		Filename string `json:"filename"`
		Text     string `json:"text"`
	}
	for name, dst := range map[string]*string{"input_prefix": &pre, "input_suffix": &suf, "prompt": &prompt} {
		if err := decodeField(body, name, dst); err != nil {
			return nil, "", err
		}
	}
	if err := decodeField(body, "input_extra", &extra); err != nil {
		return nil, "", err
	}
	if _, has := body["input_prefix"]; !has {
		return nil, "", fmt.Errorf("%w: input_prefix is required", ErrInvalid)
	}
	if _, has := body["input_suffix"]; !has {
		return nil, "", fmt.Errorf("%w: input_suffix is required", ErrInvalid)
	}
	tk := func(s string) ([]int32, error) { return lb.Tokenize(info.ID, s, false, false) }
	ids, err := lb.Tokenize(info.ID, "", true, false) // the BOS, when the vocabulary adds one
	if err != nil {
		return nil, "", err
	}
	add := func(s string) error {
		t, err := tk(s)
		ids = append(ids, t...)
		return err
	}
	var text strings.Builder
	if len(extra) > 0 {
		if fim.Repo >= 0 && fim.Sep >= 0 {
			ids = append(ids, fim.Repo)
			if err := add("myproject\n"); err != nil {
				return nil, "", err
			}
			for _, x := range extra {
				ids = append(ids, fim.Sep)
				if err := add(x.Filename + "\n" + x.Text + "\n"); err != nil {
					return nil, "", err
				}
			}
			ids = append(ids, fim.Sep)
			if err := add("filename\n"); err != nil {
				return nil, "", err
			}
		} else {
			for _, x := range extra {
				if err := add(x.Text + "\n"); err != nil {
					return nil, "", err
				}
			}
		}
	}
	ids = append(ids, fim.Pre)
	if err := add(pre); err != nil {
		return nil, "", err
	}
	ids = append(ids, fim.Suf)
	if err := add(suf); err != nil {
		return nil, "", err
	}
	ids = append(ids, fim.Mid)
	if err := add(prompt); err != nil {
		return nil, "", err
	}
	text.WriteString(pre + suf + prompt)
	return ids, text.String(), nil
}

// llamaRun is one /completion or /infill response being built.
type llamaRun struct {
	c            *compat
	w            http.ResponseWriter
	r            *http.Request
	o            GenerateOptions
	info         LlamaModelInfo
	set          llamaSettings
	promptText   string
	promptIDs    []int32
	returnTokens bool

	started *Started
	fin     *Finished
	text    strings.Builder
	ids     []int32
	probs   []any
	n       int
}

func (g *llamaRun) take(t *Token) {
	if t.ID >= 0 {
		g.n++
		g.ids = append(g.ids, t.ID)
		if t.Logprob != nil {
			g.probs = append(g.probs, llamaProb(t))
		}
	}
	g.text.WriteString(t.Text)
}

func llamaProb(t *Token) map[string]any {
	lp := t.Logprob
	top := make([]any, len(lp.Top))
	for i, a := range lp.Top {
		top[i] = map[string]any{"id": a.ID, "token": a.Text, "logprob": a.Logprob, "bytes": oaBytes(a.Text)}
	}
	return map[string]any{"id": t.ID, "token": lp.Text, "logprob": lp.Logprob, "bytes": oaBytes(lp.Text),
		"top_logprobs": top}
}

func llamaStopType(r FinishReason) string {
	switch r {
	case FinishEOS:
		return "eos"
	case FinishStop:
		return "word"
	case FinishMaxTokens:
		return "limit"
	}
	return "none"
}

// final is the object a finished completion ends with, whole or as the last
// stream frame (whose content and tokens are only what has not been sent).
func (g *llamaRun) final(content string, ids []int32, probs []any) map[string]any {
	f := g.fin
	cache := f.Restored
	var pms, dms float64
	pms = float64(f.Prefill.Microseconds()) / 1000
	dms = float64(f.Decode.Microseconds()) / 1000
	perTok := func(ms float64, n int) float64 {
		if n == 0 {
			return 0
		}
		return ms / float64(n)
	}
	perSec := func(ms float64, n int) float64 {
		if ms <= 0 {
			return 0
		}
		return float64(n) / ms * 1000
	}
	promptN := max(f.PromptTokens-cache, 0)
	out := map[string]any{
		"index": 0, "content": content, "tokens": ids, "id_slot": 0, "stop": true,
		"model": g.info.Name, "tokens_predicted": f.CompletionTokens, "tokens_evaluated": f.PromptTokens,
		"generation_settings": g.settings(),
		"prompt":              g.promptText, "has_new_line": strings.Contains(g.text.String(), "\n"),
		"truncated": false, "stop_type": llamaStopType(f.Reason), "stopping_word": f.StopMatched,
		"tokens_cached": f.Restored,
		"timings": map[string]any{
			"cache_n": cache, "prompt_n": promptN, "prompt_ms": pms,
			"prompt_per_token_ms": perTok(pms, promptN), "prompt_per_second": perSec(pms, promptN),
			"predicted_n": f.CompletionTokens, "predicted_ms": dms,
			"predicted_per_token_ms": perTok(dms, f.CompletionTokens), "predicted_per_second": perSec(dms, f.CompletionTokens),
		},
	}
	if len(probs) > 0 {
		out["completion_probabilities"] = probs
	}
	return out
}

func (g *llamaRun) settings() map[string]any {
	p := llamaParams(g.set, 0, nil, g.info.NCtx)
	p["n_ctx"], p["model"] = g.info.NCtx, g.info.Name
	return p
}

func (g *llamaRun) events(stream func(ev Event) error) error {
	return g.c.b.Generate(g.r.Context(), g.o, func(ev Event) error {
		switch ev.Kind {
		case EventStarted:
			g.started = ev.Started
		case EventFinished:
			g.fin = ev.Finished
		}
		return stream(ev)
	})
}

func (g *llamaRun) whole() {
	err := g.events(func(ev Event) error {
		if ev.Kind == EventToken {
			g.take(ev.Token)
		}
		return nil
	})
	if err != nil {
		llFailErr(g.w, err)
		return
	}
	if g.fin == nil {
		llFail(g.w, http.StatusInternalServerError, "the generate finished without a finished event", "server_error")
		return
	}
	ids := []int32{}
	if g.returnTokens {
		ids = g.ids
	}
	writeJSON(g.w, http.StatusOK, g.final(g.text.String(), ids, g.probs))
}

// stream sends the text as `data: {json}` frames with no terminator, as
// llama-server does: each carries content, tokens and stop:false, and the last
// carries everything with stop:true.
func (g *llamaRun) stream() {
	if err := admits(g.c.b, g.o); err != nil {
		llFailErr(g.w, err)
		return
	}
	sse, err := newSSE(g.w)
	if err != nil {
		llFail(g.w, http.StatusInternalServerError, err.Error(), "server_error")
		return
	}
	var pendIDs []int32
	var pendProbs []any
	gerr := g.events(func(ev Event) error {
		switch ev.Kind {
		case EventToken:
			before := len(g.probs)
			g.take(ev.Token)
			if ev.Token.ID >= 0 {
				pendIDs = append(pendIDs, ev.Token.ID)
			}
			pendProbs = append(pendProbs, g.probs[before:]...)
			// A token whose text is held back (a partial rune, a possible
			// stop string) sends with the next one that has text.
			if ev.Token.Text == "" {
				return nil
			}
			frame := map[string]any{"index": 0, "content": ev.Token.Text, "tokens": pendIDs, "stop": false,
				"id_slot": 0, "tokens_predicted": g.n}
			if g.started != nil {
				frame["tokens_evaluated"] = g.started.PromptTokens
			}
			if len(pendProbs) > 0 {
				frame["completion_probabilities"] = pendProbs
			}
			pendIDs, pendProbs = nil, nil
			return sse.send(frame)
		case EventFinished:
			return sse.send(g.final("", append([]int32{}, pendIDs...), pendProbs))
		}
		return nil
	})
	if gerr != nil {
		status, typ := http.StatusInternalServerError, "server_error"
		if errors.Is(gerr, ErrInvalid) {
			status, typ = http.StatusBadRequest, "invalid_request_error"
		}
		b, _ := json.Marshal(map[string]any{"error": map[string]any{"code": status, "message": gerr.Error(), "type": typ}})
		sse.w.Write(append(append([]byte("error: "), b...), '\n', '\n'))
		sse.rc.Flush()
	}
}
