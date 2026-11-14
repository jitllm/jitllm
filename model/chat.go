package model

import (
	"fmt"
	"maps"
	"slices"
	"strconv"
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
	// Images is how many pictures this turn carries, placed before Content.
	// No model renders one until a tower runs: the render refuses it.
	Images int
	// Videos is how many videos it carries, after its pictures.
	Videos int
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
	if m == nil || m.Container == nil {
		return "", false
	}
	v := m.Container.Vocab()
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
	return m.renderChat(msgs, tools, choice, addGenerationPrompt)
}

// chatTemplateFor picks the template a request renders with, as transformers'
// get_chat_template does (tokenization_utils_base.py): a model with one
// template uses it whatever the request; of several named ones, a request
// carrying tools takes "tool_use" when the model has one, and anything else
// takes "default". Several with no default is refused by name rather than
// guessed at, as transformers refuses it. tools is transformers' `tools is not
// None`: an empty list still asks for tool_use. A conversation carrying a
// video takes "video" where the model has one: the template a processor
// keeps for one, which the converter states beside the model's own.
func (m *Model) ChatTemplateFor(tools, video bool) (string, error) {
	var ts []jlm.ChatTemplate
	if m != nil && m.Container != nil {
		if v := m.Container.Vocab(); v != nil {
			ts = v.Templates
		}
	}
	return PickChatTemplate(ts, tools, video)
}

// pickChatTemplate is chatTemplateFor over a template list.
func PickChatTemplate(ts []jlm.ChatTemplate, tools, video bool) (string, error) {
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
	if video {
		if b, ok := find("video"); ok {
			return b, nil
		}
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

// renderChat is ChatPrompt through the template a request picks.
func (m *Model) renderChat(msgs []ChatMessage, tools []byte, choice ToolChoice, addGenerationPrompt bool) (string, error) {
	src, err := m.ChatTemplateFor(len(tools) > 0, carriesVideo(msgs))
	if err != nil {
		return "", err
	}
	tpl, err := m.CompiledTemplate(src)
	if err != nil {
		return "", err
	}
	bos, eos := m.SpecialText()
	return RenderCompiled(tpl, src, bos, eos, m.chatClock, msgs, tools, choice, addGenerationPrompt, false, "")
}

// compiledTemplate is src compiled, once per model: a compiled template is
// read-only while it renders, so every request and session shares it.
func (m *Model) CompiledTemplate(src string) (*jinja.Template, error) {
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
func RenderChatTemplate(src, bos, eos string, clock func() time.Time, msgs []ChatMessage, tools []byte,
	addGenerationPrompt bool) (string, error) {
	tpl, err := jinja.Compile(src)
	if err != nil {
		return "", fmt.Errorf("model: chat template (%d bytes): %w", len(src), err)
	}
	return RenderCompiled(tpl, src, bos, eos, clock, msgs, tools, ToolChoice{}, addGenerationPrompt, false, "")
}

// renderCompiled is renderChatTemplate over src already compiled, with the
// request's tool_choice in the context when it gives one. pics says a
// message's images are written as the template writes an image part, which
// only ChatSpans asks for: everywhere else a picture is refused, since its
// placeholder's text is not the picture. A template that reads no typed
// parts is handed mark before the message's text for each image, where the
// tower states one (jlm.TowerSpan.Mark).
func RenderCompiled(tpl *jinja.Template, src, bos, eos string, clock func() time.Time, msgs []ChatMessage,
	tools []byte, choice ToolChoice, addGenerationPrompt, pics bool, mark string) (string, error) {
	if !readsTools(src) && (len(tools) > 0 && choice.Mode != ToolChoiceNone || carriesToolTurns(msgs)) {
		// A template that reads no tools is served the generic form
		// (toolgeneric.go): the tools and how to answer in the system
		// message, a call and its result as the text of their turns. A
		// request that forbids calls is told nothing of its tools.
		render := func(ms []ChatMessage, gen bool) (string, error) {
			return RenderCompiled(tpl, src, bos, eos, clock, ms, nil, ToolChoice{}, gen, pics, mark)
		}
		var example string
		if choice.Mode == ToolChoiceNone {
			tools = nil
		} else if len(tools) > 0 {
			example = genericExample(render, eos)
		}
		return render(genericMessages(msgs, tools, example), addGenerationPrompt)
	}
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
		// A template that renders tool calls was trained to call them, even
		// when its definitions travel outside the template -- DeepSeek-V3's
		// and V3.2's write calls and results and leave the tool list to the
		// system prompt -- and transformers renders it with tools passed.
		tv, err := jinja.FromJSON(tools)
		if err != nil {
			return "", fmt.Errorf("model: tools: %w", err)
		}
		ctx["tools"] = tv
		if v := choice.templateValue(); v != nil {
			ctx["tool_choice"] = v
		}
	}
	parts := WantsContentParts(tpl)
	typed := parts || readsContentParts(tpl)
	msgs = FoldSystem(tpl, ctx, parts, msgs)
	messages := func(stringArgs bool) ([]any, error) {
		ms := make([]any, len(msgs))
		for i, x := range msgs {
			if x.Images+x.Videos > 0 && !pics {
				return nil, fmt.Errorf("model: message %d carries %d image(s) and %d video(s); ChatSpans is the "+
					"path for them", i, x.Images, x.Videos)
			}
			if x.Images > 0 && !typed && mark == "" {
				return nil, fmt.Errorf("model: message %d carries %d image(s) and this chat "+
					"template reads a message as a string, so it writes no image part", i, x.Images)
			}
			if x.Videos > 0 && !typed {
				return nil, fmt.Errorf("model: message %d carries %d video(s) and this chat "+
					"template reads a message as a string, so it writes no video part", i, x.Videos)
			}
			content := contentValue(x.Content, x.Images, x.Videos, parts || typed && x.Images+x.Videos > 0)
			if x.Images > 0 && !typed {
				content = strings.Repeat(mark, x.Images) + x.Content
			}
			mm := map[string]any{"role": x.Role, "content": content}
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
	if err != nil && CarriesToolCalls(msgs) {
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
func CarriesToolCalls(msgs []ChatMessage) bool {
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
func FoldSystem(tpl *jinja.Template, ctx map[string]any, parts bool, msgs []ChatMessage) []ChatMessage {
	if len(msgs) == 0 || msgs[0].Role != "system" || SystemRendered(tpl, ctx, parts) {
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
func SystemRendered(tpl *jinja.Template, ctx map[string]any, parts bool) bool {
	const probe = "jitllm-system-probe"
	pc := maps.Clone(ctx)
	pc["messages"] = []any{
		map[string]any{"role": "system", "content": contentValue(probe, 0, 0, parts)},
		map[string]any{"role": "user", "content": contentValue("User message", 0, 0, parts)},
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
		if x.Images > 0 || x.Videos > 0 {
			return nil, fmt.Errorf("model: message %d carries a picture or a video; ChatSpans is the "+
				"path for that, since its rows are embeddings and not ids", i)
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
func (m *Model) SpecialText() (bos, eos string) {
	v := m.Container.Vocab()
	if v == nil {
		return "", ""
	}
	at := func(id int32) string {
		if id < 0 || int(id) >= len(v.Tokens) {
			return ""
		}
		return v.Tokens[id]
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
func WantsContentParts(tpl *jinja.Template) bool {
	const probe = "\x00jitllm-probe\x00"
	out, err := tpl.Render(map[string]any{
		"messages":              []any{map[string]any{"role": "user", "content": probe}},
		"add_generation_prompt": false,
		"bos_token":             "",
		"eos_token":             "",
	})
	return err != nil || !strings.Contains(out, probe)
}

// readsContentParts says the template reads `content` as typed parts too: a
// text part's text reaches its output and the part itself does not -- a
// template printing the list whole writes a key nothing reads. A template
// that reads both shapes (Gemma 3's) is handed a message carrying an image
// as parts.
func readsContentParts(tpl *jinja.Template) bool {
	const probe, unread = "\x00jitllm-probe\x00", "\x00jitllm-unread\x00"
	out, err := tpl.Render(map[string]any{
		"messages": []any{map[string]any{"role": "user",
			"content": []any{map[string]any{"type": "text", "text": probe, "unread": unread}}}},
		"add_generation_prompt": false,
		"bos_token":             "",
		"eos_token":             "",
	})
	return err == nil && strings.Contains(out, probe) && !strings.Contains(out, unread)
}

// carriesVideo reports a conversation with a video in one of its messages.
func carriesVideo(msgs []ChatMessage) bool {
	for _, x := range msgs {
		if x.Videos > 0 {
			return true
		}
	}
	return false
}

// contentValue presents a message in the shape the template reads: its text,
// or its images first as typed parts, then its videos, then its text.
func contentValue(text string, images, videos int, parts bool) any {
	if !parts {
		return text
	}
	out := make([]any, 0, images+videos+1)
	for range images {
		out = append(out, map[string]any{"type": "image"})
	}
	for range videos {
		out = append(out, map[string]any{"type": "video"})
	}
	return append(out, map[string]any{"type": "text", "text": text})
}

// ChatSpans is a conversation's prompt as spans, for a model with a tower: the
// chat template's text, and in place of each image of the messages, in order,
// pics[i] -- a picture the prefill encodes (Model.PictureSpan), or rows
// already encoded (Instance.EncodePicture), named by their Key where the
// caller knows it -- between the markers the container states the
// rows are wrapped in (jlm.TowerSpan). The template renders the whole prompt;
// only what it writes for an image part is replaced, and that text is read
// off the template (picturePlaceholder), never named per family.
func (m *Model) ChatSpans(msgs []ChatMessage, pics []Span, addGenerationPrompt bool) ([]Span, error) {
	return m.ChatSpansToolChoice(msgs, pics, nil, ToolChoice{}, addGenerationPrompt)
}

// ChatSpansToolChoice is ChatSpans with a tool list and choice rendered as
// ChatIDsToolChoice renders them.
func (m *Model) ChatSpansToolChoice(msgs []ChatMessage, pics []Span, tools []byte, choice ToolChoice,
	addGenerationPrompt bool) ([]Span, error) {
	return m.ChatSpansMedia(msgs, pics, nil, tools, choice, addGenerationPrompt)
}

// ChatSpansMedia is ChatSpansToolChoice with videos: in place of each video
// of the messages, in order, videos[i] (Model.VideoParts), its markers its
// own, its text at either end encoded with the prompt's beside it -- where
// the template writes a video part.
func (m *Model) ChatSpansMedia(msgs []ChatMessage, pics []Span, videos []VideoParts, tools []byte, choice ToolChoice,
	addGenerationPrompt bool) ([]Span, error) {
	n, nv := 0, 0
	for _, x := range msgs {
		n, nv = n+x.Images, nv+x.Videos
	}
	if n != len(pics) {
		return nil, fmt.Errorf("model: %d image(s) in the messages and %d picture(s)", n, len(pics))
	}
	if nv != len(videos) {
		return nil, fmt.Errorf("model: %d video(s) in the messages and %d given", nv, len(videos))
	}
	if n+nv == 0 {
		ids, err := m.ChatIDsToolChoice(msgs, tools, choice, addGenerationPrompt)
		return []Span{{Tokens: ids}}, err
	}
	if m.Tower == nil {
		return nil, fmt.Errorf("model: %d image(s), %d video(s) and this model has no vision tower", n, nv)
	}
	if m.Vocab == nil {
		return nil, fmt.Errorf("model: no tokenizer: %v", m.TokErr)
	}
	for i, p := range pics {
		switch {
		case p.Embd == nil && p.Picture != nil:
			b := p.Picture.Bounds()
			want, err := m.Tower.PicRows(b.Dx(), b.Dy())
			if err != nil {
				return nil, fmt.Errorf("model: picture %d: %w", i, err)
			}
			if p.PicRows != want {
				return nil, fmt.Errorf("model: picture %d takes %d rows and the tower writes %d",
					i, p.PicRows, want)
			}
		case len(p.Embd) == 0 || len(p.Embd)%m.Cfg.NEmbd != 0:
			return nil, fmt.Errorf("model: picture %d is %d values, not rows of the model's %d",
				i, len(p.Embd), m.Cfg.NEmbd)
		}
	}
	src, err := m.ChatTemplateFor(len(tools) > 0, carriesVideo(msgs))
	if err != nil {
		return nil, err
	}
	tpl, err := m.CompiledTemplate(src)
	if err != nil {
		return nil, err
	}
	bos, eos := m.SpecialText()
	mark := m.Container.Vision().Tower.Span.Mark
	var ph, vph string
	if n > 0 {
		if ph, err = m.Tower.placeholder(tpl, src, bos, eos, mark, false); err != nil {
			return nil, err
		}
	}
	if nv > 0 {
		if vph, err = m.Tower.placeholder(tpl, src, bos, eos, mark, true); err != nil {
			return nil, err
		}
	}
	text, err := RenderCompiled(tpl, src, bos, eos, m.chatClock, msgs, tools, choice, addGenerationPrompt, true, mark)
	if err != nil {
		return nil, err
	}
	return m.PromptSpansMedia(text, ph, vph, pics, videos)
}

// PromptSpans is a rendered prompt as spans: text, in which ph stands for
// each of pics in order, the text around each encoded with the markers the
// container wraps a picture's rows in (jlm.TowerSpan) and each picture the
// span pics holds for it.
func (m *Model) PromptSpans(text, ph string, pics []Span) ([]Span, error) {
	return m.PromptSpansMedia(text, ph, "", pics, nil)
}

// PromptSpansMedia is PromptSpans with videos: vph stands for each of
// videos in order, the placeholder text and the video's markers both
// replaced by the video's spans, which are its markers and its groups, and
// its lead and tail joined to the text either side, as the processor
// expands its placeholder into a string before tokenizing it.
func (m *Model) PromptSpansMedia(text, ph, vph string, pics []Span, videos []VideoParts) ([]Span, error) {
	if m.Tower == nil {
		return nil, fmt.Errorf("model: %d picture(s), %d video(s) and this model has no vision tower", len(pics), len(videos))
	}
	if m.Vocab == nil {
		return nil, fmt.Errorf("model: no tokenizer: %v", m.TokErr)
	}
	if ph != "" && vph != "" && (strings.Contains(ph, vph) || strings.Contains(vph, ph)) {
		return nil, fmt.Errorf("model: the template writes %q for an image and %q for a video, one inside the other", ph, vph)
	}
	// Where each placeholder stands, in the order the text holds them.
	type hold struct {
		at, end int
		video   bool
	}
	var holds []hold
	for _, k := range []struct {
		ph    string
		video bool
	}{{ph, false}, {vph, true}} {
		for i := 0; k.ph != ""; {
			j := strings.Index(text[i:], k.ph)
			if j < 0 {
				break
			}
			holds = append(holds, hold{i + j, i + j + len(k.ph), k.video})
			i += j + len(k.ph)
		}
	}
	slices.SortFunc(holds, func(a, b hold) int { return a.at - b.at })
	// The counts are checked: a template that drops an image part, or text
	// that happens to hold the placeholder, would put every later picture on
	// the wrong turn.
	np := 0
	for _, h := range holds {
		if !h.video {
			np++
		}
	}
	if np != len(pics) || len(holds)-np != len(videos) {
		return nil, fmt.Errorf("model: the template wrote %d image placeholder(s) %q and %d video placeholder(s) %q for %d image(s) and %d video(s)",
			np, ph, len(holds)-np, vph, len(pics), len(videos))
	}
	mk := m.Container.Vision().Tower.Span
	var out []Span
	add := func(ids []int32) {
		if len(ids) > 0 {
			out = append(out, Span{Tokens: ids})
		}
	}
	// Text around a picture is encoded with the prompt beside it, as the
	// processor expands its placeholder into a string before tokenizing it.
	prev, after, pi, vi := 0, "", 0, 0
	for k := 0; k <= len(holds); k++ {
		end := len(text)
		if k < len(holds) {
			end = holds[k].at
		}
		seg := after + text[prev:end]
		after = ""
		pic := k < len(holds) && !holds[k].video
		switch {
		case pic:
			if mk.IndexBefore != "" || mk.IndexAfter != "" {
				seg += mk.IndexBefore + strconv.Itoa(pi) + mk.IndexAfter
			}
			seg += mk.BeforeText
			after = mk.AfterText
		case k < len(holds):
			seg += videos[vi].Lead
			after = videos[vi].Tail
		}
		add(m.Vocab.EncodeSpecial(seg, false))
		switch {
		case pic:
			add(mk.Before)
			p := pics[pi]
			p.Bidir = p.Bidir || mk.Bidir
			out = append(out, p)
			add(mk.After)
			pi++
		case k < len(holds):
			out = append(out, videos[vi].Spans...)
			vi++
		}
		if k < len(holds) {
			prev = holds[k].end
		}
	}
	return out, nil
}

// mediaPlaceholder is what the template writes for one image part, or for
// one video part: a user turn rendered with one before its text and without
// differ by that text, which is what is left of the first between the two's
// common prefix and suffix.
func mediaPlaceholder(tpl *jinja.Template, src, bos, eos, mark string, video bool) (string, error) {
	const probe = "\u2042probe\u2042"
	turn := func(n int) (string, error) {
		msg := ChatMessage{Role: "user", Content: probe, Images: n}
		if video {
			msg.Images, msg.Videos = 0, n
		}
		return RenderCompiled(tpl, src, bos, eos, nil, []ChatMessage{msg}, nil, ToolChoice{}, false, true, mark)
	}
	with, err := turn(1)
	if err != nil {
		return "", err
	}
	without, err := turn(0)
	if err != nil {
		return "", err
	}
	p := 0
	for p < len(with) && p < len(without) && with[p] == without[p] {
		p++
	}
	s := 0
	for s < len(with)-p && s < len(without)-p && with[len(with)-1-s] == without[len(without)-1-s] {
		s++
	}
	ph := with[p : len(with)-s]
	if ph == "" || strings.Contains(ph, probe) || strings.Count(with, ph) != 1 {
		what := "an image"
		if video {
			what = "a video"
		}
		return "", fmt.Errorf("model: the chat template writes no text of its own for %s part (%q)", what, ph)
	}
	return ph, nil
}

// SpanPositions is how many positions spans occupy, which is what an instance
// must be sized for.
func SpanPositions(spans []Span, nEmbd int) int {
	t := 0
	for _, sp := range spans {
		t += sp.N(nEmbd)
	}
	return t
}
