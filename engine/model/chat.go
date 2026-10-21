package model

import (
	"fmt"
	"maps"
	"strings"
	"time"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/tok/jinja"
)

// A ChatMessage is one turn. Role is the template's vocabulary, not this
// package's -- "system", "user", "assistant" are conventional and a template is
// free to read others.
type ChatMessage struct {
	Role    string
	Content string
	// Images is how many pictures this turn carries, placed before Content --
	// the order every VLM template here was trained with. Only ChatSpans can
	// render a message with Images > 0; ChatIDs refuses one, because the
	// placeholder text it would tokenize is not the image.
	Images int
	// ToolCalls are the calls an assistant turn made (see tools.go).
	ToolCalls []ToolCall
	// ToolCallID and Name identify a "tool" turn's result: which call it
	// answers and which function produced it.
	ToolCallID, Name string
}

// ChatTemplate returns the model's own Jinja chat template. "" is the model's
// default: its single template, or the one named "default" of several
// (jlm.Vocab.Template). A request picks between named ones with
// chatTemplateFor.
func (m *Model) ChatTemplate(name string) (string, bool) {
	if m == nil || m.container == nil {
		return "", false
	}
	v := m.container.Vocab()
	if v == nil {
		return "", false
	}
	return v.Template(name)
}

// HasChatTemplate reports whether this model carries one at all. A base model
// does not, and asking it to chat is a caller error rather than a default.
func (m *Model) HasChatTemplate() bool {
	_, ok := m.ChatTemplate("")
	return ok
}

// ChatCapable reports whether this container carries a chat template, which is
// the difference between an instruct model and a base one, so a caller can ask
// instead of finding out by failing.
func (m *Model) ChatCapable() bool {
	_, ok := m.ChatTemplate("")
	return ok
}

// ChatPrompt renders the model's template into the text to tokenize.
//
// The result is a complete prompt carrying its own special tokens (Llama-3.2
// renders bos_token, Qwen3 emits none), so it must be encoded with
// addSpecial=false; ChatIDs does that pairing. bos_token and eos_token are
// bound from the container's own ids.
func (m *Model) ChatPrompt(msgs []ChatMessage, addGenerationPrompt bool) (string, error) {
	return m.ChatPromptTools(msgs, nil, addGenerationPrompt)
}

// ChatPromptTools is ChatPrompt with tool definitions (see ChatIDsTools).
func (m *Model) ChatPromptTools(msgs []ChatMessage, tools []byte, addGenerationPrompt bool) (string, error) {
	return m.ChatPromptToolChoice(msgs, tools, ToolChoice{}, addGenerationPrompt)
}

// ChatPromptToolChoice is ChatPromptTools with the request's tool_choice,
// handed to a template that reads it (Kimi-K3's tells the model it MUST
// call). The zero ToolChoice passes none.
func (m *Model) ChatPromptToolChoice(msgs []ChatMessage, tools []byte, choice ToolChoice, addGenerationPrompt bool) (string, error) {
	var mk imageMarkers
	if tw := m.Tower(); tw != nil {
		mk = imageMarkersOf[tw.Cfg.Projector]
	}
	return m.renderChat(msgs, tools, choice, addGenerationPrompt, mk)
}

// chatTemplateFor picks the template a request renders with, as transformers'
// get_chat_template does (tokenization_utils_base.py): a model with one
// template uses it whatever the request; of several named ones, a request
// carrying tools takes "tool_use" when the model has one, and anything else
// takes "default". Several with no default is refused by name rather than
// guessed at, as transformers refuses it. tools is transformers' `tools is not
// None`: an empty list still asks for tool_use.
func (m *Model) chatTemplateFor(tools bool) (string, error) {
	var ts []jlm.ChatTemplate
	if m != nil && m.container != nil {
		if v := m.container.Vocab(); v != nil {
			ts = v.Templates
		}
	}
	return pickChatTemplate(ts, tools)
}

// pickChatTemplate is chatTemplateFor over a template list.
func pickChatTemplate(ts []jlm.ChatTemplate, tools bool) (string, error) {
	if len(ts) == 0 {
		return "", fmt.Errorf("model: this container carries no chat template; it is a base model, not an instruct one")
	}
	if len(ts) == 1 && ts[0].Name == "" {
		return ts[0].Body, nil
	}
	find := func(name string) (string, bool) {
		for _, t := range ts {
			if t.Name == name {
				return t.Body, true
			}
		}
		return "", false
	}
	if tools {
		if b, ok := find("tool_use"); ok {
			return b, nil
		}
	}
	if b, ok := find("default"); ok {
		return b, nil
	}
	names := make([]string, len(ts))
	for i, t := range ts {
		names[i] = t.Name
	}
	return "", fmt.Errorf("model: this container carries several chat templates (%s) and "+
		"none named default", strings.Join(names, ", "))
}

// renderChat is ChatPrompt with the text a string-content template gets in
// place of each image.
func (m *Model) renderChat(msgs []ChatMessage, tools []byte, choice ToolChoice, addGenerationPrompt bool, mk imageMarkers) (string, error) {
	src, err := m.chatTemplateFor(len(tools) > 0)
	if err != nil {
		return "", err
	}
	tpl, err := m.compiledTemplate(src)
	if err != nil {
		return "", err
	}
	bos, eos := m.specialText()
	return renderCompiled(tpl, src, bos, eos, m.opt.chatClock, msgs, tools, choice, addGenerationPrompt, mk)
}

// compiledTemplate is src compiled, once per model: a compiled template is
// read-only while it renders, so every request and session shares it.
func (m *Model) compiledTemplate(src string) (*jinja.Template, error) {
	if t, ok := m.chatTemplates.Load(src); ok {
		return t.(*jinja.Template), nil
	}
	tpl, err := jinja.Compile(src)
	if err != nil {
		return nil, fmt.Errorf("model: chat template (%d bytes): %w", len(src), err)
	}
	t, _ := m.chatTemplates.LoadOrStore(src, tpl)
	return t.(*jinja.Template), nil
}

// renderChatTemplate renders msgs through the template src. The context is
// apply_chat_template's: messages, tools and documents (None when the request
// carries none -- Command-R's tool_use and rag templates read the difference),
// add_generation_prompt, the special tokens, and strftime_now when clock is set.
func renderChatTemplate(src, bos, eos string, clock func() time.Time, msgs []ChatMessage, tools []byte,
	addGenerationPrompt bool, mk imageMarkers) (string, error) {
	tpl, err := jinja.Compile(src)
	if err != nil {
		return "", fmt.Errorf("model: chat template (%d bytes): %w", len(src), err)
	}
	return renderCompiled(tpl, src, bos, eos, clock, msgs, tools, ToolChoice{}, addGenerationPrompt, mk)
}

// renderCompiled is renderChatTemplate over src already compiled, with the
// request's tool_choice in the context when it gives one.
func renderCompiled(tpl *jinja.Template, src, bos, eos string, clock func() time.Time, msgs []ChatMessage,
	tools []byte, choice ToolChoice, addGenerationPrompt bool, mk imageMarkers) (string, error) {
	ph := mk.placeholder
	ctx := map[string]any{
		"tools":                 nil,
		"documents":             nil,
		"add_generation_prompt": addGenerationPrompt,
		"bos_token":             bos,
		"eos_token":             eos,
	}
	if clock != nil {
		ctx["strftime_now"] = jinja.StrftimeNow(clock)
	}
	if len(tools) > 0 {
		// A template that reads neither the tools nor a tool call was not
		// trained to call them. One that renders tool calls was, even when
		// its definitions travel outside the template -- DeepSeek-V3's and
		// V3.2's write calls and results and leave the tool list to the
		// system prompt -- and transformers renders it with tools passed.
		if !strings.Contains(src, "tools") && !strings.Contains(src, "tool_calls") {
			return "", fmt.Errorf("model: this model's chat template never reads `tools`, so it was " +
				"not trained to call them; send the request without tools")
		}
		tv, err := jinja.FromJSON(tools)
		if err != nil {
			return "", fmt.Errorf("model: tools: %w", err)
		}
		ctx["tools"] = tv
		if v := choice.templateValue(); v != nil {
			ctx["tool_choice"] = v
		}
	}
	parts := wantsContentParts(tpl)
	msgs = foldSystem(tpl, ctx, parts, msgs)
	messages := func(stringArgs bool) ([]any, error) {
		ms := make([]any, len(msgs))
		for i, x := range msgs {
			if x.Images > 0 && ph == "" {
				return nil, fmt.Errorf("model: message %d carries %d image(s) and this model has "+
					"no vision tower", i, x.Images)
			}
			mm := map[string]any{"role": x.Role, "content": contentValue(x.Content, x.Images, ph, mk.sep(), parts)}
			if err := toolFields(mm, x, stringArgs); err != nil {
				return nil, fmt.Errorf("model: message %d: %w", i, err)
			}
			ms[i] = mm
		}
		return ms, nil
	}
	ms, err := messages(false)
	if err != nil {
		return "", err
	}
	ctx["messages"] = ms
	out, err := tpl.Render(ctx)
	if err != nil && carriesToolCalls(msgs) {
		// A template that concatenates a call's arguments as text --
		// DeepSeek-V3's and V3.2's write them between ```json fences -- fails
		// on the mapping transformers recommends, and renders only from the
		// JSON string the OpenAI API carries. Hand it that string.
		if sm, serr := messages(true); serr == nil {
			ctx["messages"] = sm
			if o, rerr := tpl.Render(ctx); rerr == nil {
				out, err = o, nil
			}
		}
	}
	if err != nil {
		return "", fmt.Errorf("model: rendering the chat template: %w", err)
	}
	if out == "" {
		// A template that renders nothing is not a prompt, and the model would
		// answer from an empty context rather than fail.
		return "", fmt.Errorf("model: the chat template rendered empty")
	}
	return out, nil
}

// carriesToolCalls reports whether any message calls a tool.
func carriesToolCalls(msgs []ChatMessage) bool {
	for _, x := range msgs {
		if len(x.ToolCalls) > 0 {
			return true
		}
	}
	return false
}

// foldSystem merges a leading system message into the first message after it
// when the template has no system role, and drops it when nothing follows.
//
// Mistral's own templates and others of their age raise on any role but user
// and assistant, so a system prompt makes every turn fail to render; some
// others render the turn and silently leave the system text out. Both are told
// apart from a template that takes it by one probe render, and the merge is
// llama.cpp's (system_message_not_supported in common/chat.cpp: the system
// text, a newline, then the message) -- RULE 7m: the reference most people run, chosen
// rather than inherited. transformers raises instead, which would leave a
// system prompt unusable with these models.
//
// ctx is the render's context without its messages and parts its content
// shape (wantsContentParts); the probe renders with both, so a template that
// reads strftime_now, the special tokens or typed parts is probed as it will
// run -- SmolVLM's renders "System: ..." only from parts.
func foldSystem(tpl *jinja.Template, ctx map[string]any, parts bool, msgs []ChatMessage) []ChatMessage {
	if len(msgs) == 0 || msgs[0].Role != "system" || systemRendered(tpl, ctx, parts) {
		return msgs
	}
	if len(msgs) == 1 {
		return nil
	}
	out := append([]ChatMessage(nil), msgs[1:]...)
	out[0].Content = msgs[0].Content + "\n" + out[0].Content
	return out
}

// systemRendered reports whether the template puts a system message's text
// in its output: a probe render of a system and a user message, as llama.cpp's
// supports_system_role capability check does.
func systemRendered(tpl *jinja.Template, ctx map[string]any, parts bool) bool {
	const probe = "jitllm-system-probe"
	pc := maps.Clone(ctx)
	pc["messages"] = []any{
		map[string]any{"role": "system", "content": contentValue(probe, 0, "", "", parts)},
		map[string]any{"role": "user", "content": contentValue("User message", 0, "", "", parts)},
	}
	pc["add_generation_prompt"] = true
	out, err := tpl.Render(pc)
	return err == nil && strings.Contains(out, probe)
}

// ChatIDs is ChatPrompt plus the tokenization, and it exists so the two cannot
// be paired wrongly: the rendered text carries its own specials, so it is
// encoded with addSpecial=false. Callers that want the text render it
// themselves; callers that want ids should use this.
func (m *Model) ChatIDs(msgs []ChatMessage, addGenerationPrompt bool) ([]int32, error) {
	return m.ChatIDsTools(msgs, nil, addGenerationPrompt)
}

// ChatIDsTools is ChatIDs with tool definitions: tools is a JSON array in the
// OpenAI shape ([{"type": "function", "function": {name, description,
// parameters}}]), handed to the template as `tools` with its key order kept.
// nil or empty renders exactly what ChatIDs does.
func (m *Model) ChatIDsTools(msgs []ChatMessage, tools []byte, addGenerationPrompt bool) ([]int32, error) {
	return m.ChatIDsToolChoice(msgs, tools, ToolChoice{}, addGenerationPrompt)
}

// ChatIDsToolChoice is ChatIDsTools with the request's tool_choice
// (ChatPromptToolChoice).
func (m *Model) ChatIDsToolChoice(msgs []ChatMessage, tools []byte, choice ToolChoice, addGenerationPrompt bool) ([]int32, error) {
	if m.Vocab == nil {
		return nil, fmt.Errorf("model: no tokenizer: %v", m.TokErr)
	}
	for i, x := range msgs {
		if x.Images > 0 {
			return nil, fmt.Errorf("model: message %d carries an image; ChatSpans is the "+
				"path for that, since an image is embeddings and not ids", i)
		}
	}
	text, err := m.ChatPromptToolChoice(msgs, tools, choice, addGenerationPrompt)
	if err != nil {
		return nil, err
	}
	return m.Vocab.EncodeSpecial(text, false), nil
}

// specialText resolves BOS and EOS to their literal piece text, which is what a
// template interpolates. Empty when the container names no such token, which a
// template referring to it will then render as the empty string rather than
// failing -- the same thing llama.cpp does.
func (m *Model) specialText() (bos, eos string) {
	v := m.container.Vocab()
	if v == nil {
		return "", ""
	}
	at := func(id int32) string {
		if id < 0 || int(id) >= v.Tokens.Len() {
			return ""
		}
		return v.Tokens.At(int(id))
	}
	return at(v.BOS), at(v.EOS)
}

// wantsContentParts asks the template which shape of `content` it reads, by
// rendering a sentinel through it, rather than guessing from the model's name.
//
// A VLM template (SmolVLM's) reads content as a list of typed parts and a text
// one reads a string; the wrong shape fails silently, e.g. a loop over a Go
// string iterates characters and the message renders empty. Probing asks the
// template itself, as llama.cpp's minja does, for one extra render per turn.
func wantsContentParts(tpl *jinja.Template) bool {
	const probe = "\x00jitllm-probe\x00"
	out, err := tpl.Render(map[string]any{
		"messages":              []any{map[string]any{"role": "user", "content": probe}},
		"add_generation_prompt": false,
		"bos_token":             "",
		"eos_token":             "",
	})
	return err != nil || !strings.Contains(out, probe)
}

// contentValue presents a message in the shape the template reads: images
// first, as typed parts, or -- for a template that reads a string -- as the
// placeholder text followed by sep, which is llava's "<image>\n{prompt}".
func contentValue(text string, images int, ph, sep string, parts bool) any {
	if !parts {
		return strings.Repeat(ph+sep, images) + text
	}
	out := make([]any, 0, images+1)
	for range images {
		out = append(out, map[string]any{"type": "image"})
	}
	return append(out, map[string]any{"type": "text", "text": text})
}

// imageMarkers is how one projector family wraps an image in the token stream.
type imageMarkers struct {
	// placeholder is what the model's chat template writes for one image part
	// -- and what a string-content template is handed in its place.
	placeholder string
	// pre and post are vocabulary pieces around the embeddings, looked up by
	// piece and never encoded.
	pre, post []string
	// preText and postText are TEXT around the embeddings, encoded with the
	// prompt segment beside them -- the reference processor expands its
	// placeholder into a string and tokenizes the whole, so a newline it adds
	// may merge with the template's own (gemma3's "\n\n" after "user\n").
	preText, postText string
	// bidir says the image's rows attend to each other in both directions in
	// the text model (Span.Bidir): gemma3's reference masks them so.
	bidir bool
	// bidirSWA says the rows are such a run only on a text model whose
	// sliding layers see an image both ways (Config.BidirSWA): Gemma 4.
	bidirSWA bool
	// inline says a string-content template gets the placeholder directly
	// before the text, with no newline between: gemma3's template trims the
	// text of a typed part and writes the placeholder flush against it, and
	// it reads a string as readily as parts.
	inline bool
}

// sep is what follows each placeholder in a string-content template.
func (mk imageMarkers) sep() string {
	if mk.inline {
		return ""
	}
	return "\n"
}

// imageMarkersOf is keyed by TowerConfig.Projector.
//
// The text model was trained with the embeddings wrapped in specific tokens,
// and getting them wrong does not fault, it just answers wrongly. Each entry
// is read off the artefact:
//
//   - idefics3 (SmolVLM): the template writes "<image>"; mtmd's
//     tok_ov_img_start is "\n\n" <fake_token_around_image> <global-img> and its
//     end is <fake_token_around_image>. The pieces are looked up, as mtmd's
//     lookup_token does: BPE-encoding "\n\n" after "User:" need not reach the
//     same split.
//   - qwen2vl_merger: the template writes
//     <|vision_start|><|image_pad|><|vision_end|>, and <|image_pad|> is where
//     the embeddings go -- the span IS the padding, so it is not emitted.
//   - mlp (llava): no wrapping at all; the embeddings stand where "<image>"
//     stood, and llava-phi-3's template reads a string.
//   - gemma3: the template writes "<start_of_image>", and Gemma3Processor
//     expands it to "\n\n<start_of_image>" + 256 soft tokens +
//     "<end_of_image>\n\n"; the soft tokens are where the embeddings go, and
//     they attend to each other bidirectionally (its token_type_ids mask).
//   - internvl: the template (InternVL's own, and a string template given the
//     placeholder) writes "<IMG_CONTEXT>"; InternVLProcessor expands it to
//     "<img>" + 256 per tile + "</img>", every tile in one span.
//
// A missing piece is an error, never skipped: a model given no markers still
// answers plausibly, so nothing else would report it.
var imageMarkersOf = map[string]imageMarkers{
	jlm.ProjIdefics3.String(): {placeholder: "<image>",
		pre:  []string{"\n\n", "<fake_token_around_image>", "<global-img>"},
		post: []string{"<fake_token_around_image>"}},
	jlm.ProjQwen2VL.String(): {placeholder: "<|vision_start|><|image_pad|><|vision_end|>",
		pre: []string{"<|vision_start|>"}, post: []string{"<|vision_end|>"}},
	jlm.ProjQwen25VL.String(): {placeholder: "<|vision_start|><|image_pad|><|vision_end|>",
		pre: []string{"<|vision_start|>"}, post: []string{"<|vision_end|>"}},
	jlm.ProjQwen3VL.String(): {placeholder: "<|vision_start|><|image_pad|><|vision_end|>",
		pre: []string{"<|vision_start|>"}, post: []string{"<|vision_end|>"}},
	// GLM-4.xV: the template writes <|begin_of_image|><|image|><|end_of_image|>
	// and the processor expands <|image|> into the picture's rows.
	jlm.ProjGLM4V.String(): {placeholder: "<|begin_of_image|><|image|><|end_of_image|>",
		pre: []string{"<|begin_of_image|>"}, post: []string{"<|end_of_image|>"}},
	// Kimi-VL: the template writes
	// <|media_start|>image<|media_content|><|media_pad|><|media_end|> and the
	// processor expands <|media_pad|> into the picture's rows. Its typed
	// parts write the image flush against the text (inline).
	jlm.ProjKimiVL.String(): {placeholder: "<|media_start|>image<|media_content|><|media_pad|><|media_end|>",
		preText: "<|media_start|>image<|media_content|>", postText: "<|media_end|>", inline: true},
	// HunyuanVL: the template writes start, image and end place-holders flush
	// against the text, and the processor expands the image one into the
	// picture's rows (its begin, grid, newline and end rows) between the
	// other two.
	jlm.ProjHunyuanVL.String(): {placeholder: "<｜hy_place▁holder▁no▁100｜><｜hy_place▁holder▁no▁102｜><｜hy_place▁holder▁no▁101｜>",
		pre: []string{"<｜hy_place▁holder▁no▁100｜>"}, post: []string{"<｜hy_place▁holder▁no▁101｜>"}, inline: true},
	jlm.ProjMLP.String(): {placeholder: "<image>"},
	jlm.ProjGemma3.String(): {placeholder: "<start_of_image>",
		preText: "\n\n<start_of_image>", postText: "<end_of_image>\n\n", bidir: true, inline: true},
	jlm.ProjInternVL.String(): {placeholder: "<IMG_CONTEXT>", preText: "<img>", postText: "</img>"},
	// Gemma 3n: gemma3's sequence (Gemma3nProcessor's full_image_sequence),
	// the template's placeholder its own, and the picture's rows causal.
	jlm.ProjGemma3nV.String(): {placeholder: "<image_soft_token>",
		preText: "\n\n<start_of_image>", postText: "<end_of_image>\n\n", inline: true},
	// MiniCPM-V: its chat() writes "(<image>./</image>)" then a newline ahead
	// of the question, and the processor replaces that placeholder with the
	// picture's whole run -- an overview and its slices, each in markers of its
	// own -- which PictureSpans builds, so there is no single pre and post.
	jlm.ProjResampler.String(): {placeholder: "(<image>./</image>)"},
	// Janus-Pro: the GGUF's template writes "<image_placeholder>", and the
	// processor (processing_janus.py) expands it to <begin_of_image>, the
	// image tokens and <end_of_image>. mtmd wraps nothing (a divergence,
	// RULE 7m); the processor is what the weights saw.
	jlm.ProjJanus.String(): {placeholder: "<image_placeholder>",
		pre: []string{"<begin_of_image>"}, post: []string{"<end_of_image>"}},
	// Pixtral / Mistral 3: the template writes "[IMG]", and PixtralProcessor
	// replaces it with each merged grid row's [IMG] tokens followed by
	// [IMG_BREAK], the last [IMG_BREAK] made [IMG_END]. The head writes the
	// rows and the breaks between them (pixtral.go); [IMG_END] is the marker.
	jlm.ProjPixtral.String(): {placeholder: "[IMG]", post: []string{"[IMG_END]"}},
	// Llama 4: the template writes "<|image|>", and Llama4Processor expands
	// it to the tiles and their separators, which Llama4Spans lays out
	// (ChatSpansParts), so there is no single pre and post.
	jlm.ProjLlama4.String(): {placeholder: "<|image|>"},
	// Gemma 4: the template writes "<|image|>", and Gemma4Processor expands it
	// to <|image>, the soft tokens and <image|>. On the 26B and 31B the soft
	// tokens see each other on the sliding layers (Config.BidirSWA).
	// Phi-4-reasoning-vision: its own processor splits the prompt at
	// "<image>" (processing_phi4_visionr.py's tokenizer_image_token) and puts
	// the rows there, with no markers; mtmd's PHI4 keeps the boundary text
	// empty too.
	jlm.ProjPhi4.String(): {placeholder: "<image>"},
	// LFM2-VL: the template writes "<image>", and Lfm2VlProcessor expands it
	// to the picture's tiles and thumbnail in their markers, which LFM2Spans
	// lays out (ChatSpansParts), so there is no single pre and post.
	jlm.ProjLFM2.String(): {placeholder: "<image>"},
	jlm.ProjGemma4V.String(): {placeholder: "<|image|>",
		pre: []string{"<|image>"}, post: []string{"<image|>"}, bidirSWA: true},
}

// ChatSpans renders msgs through the model's own chat template and returns the
// prompt as spans for PrefillMixed, embeds[i] standing where the i-th image of
// the conversation goes (in message order). An embeds entry is the tower's
// output for that image (State.EncodeImage); ChatSpansImages takes a Picture
// instead, which the prompt's prefill encodes.
//
// The template renders the whole prompt, turn text included; only the image
// placeholders are replaced.
//
// Every image is taken to be the tower's fixed grid (TowerConfig.Grid). A
// dynamic-resolution tower's grid follows each picture, so it takes
// ChatSpansImages, and an embedding whose row count is not the fixed grid's
// is refused here rather than laid out on the wrong grid.
func (m *Model) ChatSpans(msgs []ChatMessage, embeds [][]float32, addGenerationPrompt bool) ([]Span, error) {
	imgs := make([]Image, len(embeds))
	if tw := m.Tower(); tw != nil {
		g := tw.Cfg.Grid()
		w := tw.Cfg.RowWidth()
		for i, e := range embeds {
			if tw.Cfg.Dynamic() && e != nil && len(e) != g.Rows()*w {
				return nil, fmt.Errorf("model: image %d is %d rows and this tower's grid follows the "+
					"picture: ChatSpansImages carries each image's grid", i, len(e)/w)
			}
			// A deepstack tower's rows carry their taps' behind them (Encode).
			embd, deep := splitDeep(e, len(e)/w, tw.Cfg.ProjDim)
			imgs[i] = Image{Embd: embd, Grid: g, Deep: deep}
		}
	}
	return m.ChatSpansImages(msgs, imgs, addGenerationPrompt)
}

// Image is one picture of a chat: a Picture the prompt's prefill encodes
// (State.Picture), or rows already encoded (State.EncodeImage) and the grid
// they cover.
type Image struct {
	Picture *Picture
	Embd    []float32
	Grid    ImageGrid
	// Key names the picture the rows came from (State.EncodeImage), which is
	// what lets the prefix cache name the pages after it. nil leaves the rows
	// unnamed: nothing after them is cached.
	Key *ImageKey
	// Deep is the picture's deepstack rows beside Embd (Span.Deep).
	Deep []float32
}

// ChatSpansImages is ChatSpans with each image's grid, which is where its rows
// turn on an M-RoPE text model.
func (m *Model) ChatSpansImages(msgs []ChatMessage, imgs []Image, addGenerationPrompt bool) ([]Span, error) {
	return m.chatSpansImages(msgs, imgs, nil, ToolChoice{}, addGenerationPrompt)
}

// chatSpansImages is ChatSpansImages with a tool list and choice rendered as
// ChatIDsToolChoice renders them.
func (m *Model) chatSpansImages(msgs []ChatMessage, imgs []Image, tools []byte, choice ToolChoice,
	addGenerationPrompt bool) ([]Span, error) {
	n := 0
	for _, x := range msgs {
		n += x.Images
	}
	if n != len(imgs) {
		return nil, fmt.Errorf("model: %d image(s) in the messages and %d embedding(s)", n, len(imgs))
	}
	if n == 0 {
		ids, err := m.ChatIDsToolChoice(msgs, tools, choice, addGenerationPrompt)
		return []Span{{Tokens: ids}}, err
	}
	tw := m.Tower()
	if tw == nil {
		return nil, fmt.Errorf("model: %d image(s) and this model has no vision tower", n)
	}
	mk, ok := imageMarkersOf[tw.Cfg.Projector]
	if !ok {
		return nil, fmt.Errorf("model: no image markers for projector %q", tw.Cfg.Projector)
	}
	piece := func(names []string) ([]int32, error) {
		var ids []int32
		for _, s := range names {
			id, ok := m.Vocab.ID(s)
			if !ok {
				return nil, fmt.Errorf("model: the vocabulary has no %q, so the image markers cannot be built", s)
			}
			ids = append(ids, id)
		}
		return ids, nil
	}
	pre, err := piece(mk.pre)
	if err != nil {
		return nil, err
	}
	post, err := piece(mk.post)
	if err != nil {
		return nil, err
	}
	images := make([][]Span, n)
	for i := range images {
		var sp []Span
		if len(pre) > 0 {
			sp = append(sp, Span{Tokens: pre})
		}
		sp = append(sp, Span{Embd: imgs[i].Embd, Picture: imgs[i].Picture, Bidir: mk.bidir || mk.bidirSWA && m.Cfg.BidirSWA,
			Grid: imgs[i].Grid, Key: imgs[i].Key, Deep: imgs[i].Deep})
		if len(post) > 0 {
			sp = append(sp, Span{Tokens: post})
		}
		images[i] = sp
	}
	return m.chatSpansParts(msgs, images, tools, choice, addGenerationPrompt)
}

// ChatSpansParts is ChatSpans for pictures whose runs the caller has already
// laid out, markers included (PictureSpans): images[i] stands where the i-th
// image of the conversation goes. A projector that cuts a picture into an
// overview and slices has no single embedding per image to hand ChatSpans.
func (m *Model) ChatSpansParts(msgs []ChatMessage, images [][]Span, addGenerationPrompt bool) ([]Span, error) {
	return m.chatSpansParts(msgs, images, nil, ToolChoice{}, addGenerationPrompt)
}

// chatSpansParts is ChatSpansParts with a tool list and choice rendered as
// ChatIDsToolChoice renders them.
func (m *Model) chatSpansParts(msgs []ChatMessage, images [][]Span, tools []byte, choice ToolChoice,
	addGenerationPrompt bool) ([]Span, error) {
	n := 0
	for _, x := range msgs {
		n += x.Images
	}
	if n != len(images) {
		return nil, fmt.Errorf("model: %d image(s) in the messages and %d picture(s)", n, len(images))
	}
	tw := m.Tower()
	if tw == nil {
		return nil, fmt.Errorf("model: %d image(s) and this model has no vision tower", n)
	}
	mk, ok := imageMarkersOf[tw.Cfg.Projector]
	if !ok {
		return nil, fmt.Errorf("model: no image markers for projector %q", tw.Cfg.Projector)
	}
	text, err := m.renderChat(msgs, tools, choice, addGenerationPrompt, mk)
	if err != nil {
		return nil, err
	}
	// The count is checked, because a template that drops an image part, or
	// user text that happens to contain the placeholder, would otherwise shift
	// every picture after it onto the wrong turn.
	segs := strings.Split(text, mk.placeholder)
	if len(segs)-1 != n {
		return nil, fmt.Errorf("model: the chat template wrote %d image placeholder(s) %q for %d image(s)",
			len(segs)-1, mk.placeholder, n)
	}
	var out []Span
	add := func(ids []int32) {
		if len(ids) > 0 {
			out = append(out, Span{Tokens: ids})
		}
	}
	for i, seg := range segs {
		if i > 0 {
			seg = mk.postText + seg
		}
		if i < n {
			seg += mk.preText
		}
		add(m.Vocab.EncodeSpecial(seg, false))
		if i < n {
			for _, sp := range images[i] {
				if sp.rows() {
					out = append(out, sp)
				} else {
					add(sp.Tokens)
				}
			}
		}
	}
	return out, nil
}

// SpanPositions is how many positions spans occupy, which is what a State
// must be sized for.
func SpanPositions(spans []Span, nEmbd int) int {
	t := 0
	for _, sp := range spans {
		t += sp.n(nEmbd)
	}
	return t
}
