package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"strings"
	"testing"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/engine/model"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// Pictures beside the request features the text path has: tools and a forced
// call, several choices, and a continued session.

// toolVisionModel is the smallest test model with a vision tower whose chat
// template renders tools.
const toolVisionModel = "Qwen3.5-0.8B-f16-vlm.jlm"

func decodePic(t *testing.T, b []byte) image.Image {
	t.Helper()
	img, _, err := image.Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	return img
}

// spanText is the text of spans' token runs.
func spanText(lm *LoadedModel, spans []model.Span) string {
	var b strings.Builder
	for _, sp := range spans {
		for _, id := range sp.Tokens {
			b.WriteString(lm.m.Vocab.Literal(id))
		}
	}
	return b.String()
}

// TestImageWithToolsRendersAndForces: a picture beside tools renders the tool
// list as the text path does, and a named tool_choice forces the call under
// the tool grammar, through the server.
func TestImageWithToolsRendersAndForces(t *testing.T) {
	_, lm, c := loadedEngine(t, toolVisionModel, "m", LoadOptions{})
	if lm.m.Tower() == nil {
		t.Fatalf("%s carries no vision tower", toolVisionModel)
	}
	quad := readTestdata(t, "quad-and-disc.png")
	st := lm.m.NewInstance(64)
	defer st.Close()
	msgs := []model.ChatMessage{{Role: "user", Content: "Describe the picture.", Images: 1}}
	with, err := st.ChatPictureSpans(msgs, []image.Image{decodePic(t, quad)}, []byte(forcedTools), model.ToolChoice{}, true)
	if err != nil {
		t.Fatal(err)
	}
	without, err := st.ChatPictureSpans(msgs, []image.Image{decodePic(t, quad)}, nil, model.ToolChoice{}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(spanText(lm, with), "get_time") {
		t.Fatalf("the tools were not rendered into the picture's prompt: %q", spanText(lm, with))
	}
	if strings.Contains(spanText(lm, without), "get_time") {
		t.Fatal("the prompt without tools names a tool: the comparison above proves nothing")
	}

	body := func(choice string) string {
		return fmt.Sprintf(`{"model":"m","temperature":0,"max_tokens":256,"tools":%s%s,"messages":[
			{"role":"system","content":"You are a poet. Call a tool only when the user asks for the weather or the time."},
			{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}},
			{"type":"text","text":"Write one short sentence about this picture."}]}]}`,
			forcedTools, choice, base64.StdEncoding.EncodeToString(quad))
	}
	call := func(choice string) ([]oaToolCall, string) {
		var out oaChatResponse
		status, b := postRaw(t, c.url+"/v1/chat/completions", body(choice))
		if status != 200 || json.Unmarshal([]byte(b), &out) != nil {
			t.Fatalf("%d %s", status, b)
		}
		return out.Choices[0].Message.ToolCalls, *out.Choices[0].FinishReason
	}
	// The violation arm: unforced, the model does not call get_time.
	calls, finish := call(`,"tool_choice":"auto"`)
	for _, tc := range calls {
		if tc.Function.Name == "get_time" {
			t.Fatalf("unforced, the model called %+v: the forced arm would prove nothing", calls)
		}
	}
	t.Logf("auto: finish %q calls %+v", finish, calls)
	before := lm.grammars.toolForced.Load()
	calls, finish = call(`,"tool_choice":{"type":"function","function":{"name":"get_time"}}`)
	t.Logf("forced generates %d -> %d", before, lm.grammars.toolForced.Load())
	if finish != "tool_calls" || len(calls) != 1 || calls[0].Function.Name != "get_time" {
		t.Fatalf("forced get_time: finish %q calls %+v", finish, calls)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(calls[0].Function.Arguments), &args); err != nil || args["zone"] == nil {
		t.Fatalf("the forced call's arguments %q do not fit the schema: %v", calls[0].Function.Arguments, err)
	}
	if lm.grammars.toolForced.Load() == before {
		t.Fatal("the forced call ran without the tool grammar")
	}
}

// TestImageChoicesPrefillOnce: n choices of a picture prompt each equal the
// same request sent alone with that choice's seed, and every choice after the
// first restores the prompt -- the picture included -- rather than running it.
func TestImageChoicesPrefillOnce(t *testing.T) {
	_, _, c := loadedEngine(t, visionModel, "vlm", LoadOptions{})
	quad := readTestdata(t, "quad-and-disc.png")
	body := func(n, seed int) string {
		return fmt.Sprintf(`{"model":"vlm","temperature":0.9,"seed":%d,"n":%d,"max_tokens":12,"messages":[{"role":"user","content":[
			{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}},{"type":"text","text":%q}]}]}`,
			seed, n, base64.StdEncoding.EncodeToString(quad), askPicture)
	}
	var both oaChatResponse
	status, b := postRaw(t, c.url+"/v1/chat/completions", body(3, 7))
	if status != 200 || json.Unmarshal([]byte(b), &both) != nil || len(both.Choices) != 3 {
		t.Fatalf("%d %s", status, b)
	}
	for i := range 3 {
		var one oaChatResponse
		status, b := postRaw(t, c.url+"/v1/chat/completions", body(1, 7+i))
		if status != 200 || json.Unmarshal([]byte(b), &one) != nil {
			t.Fatalf("%d %s", status, b)
		}
		if got, want := both.Choices[i].Message.Content, one.Choices[0].Message.Content; got != want {
			t.Errorf("choice %d: %q, want the seed-%d request's %q", i, got, 7+i, want)
		}
	}
	r := both.Jitllm.PromptRestored
	t.Logf("contents %q %q %q; restored %v", both.Choices[0].Message.Content, both.Choices[1].Message.Content,
		both.Choices[2].Message.Content, r)
	if len(r) != 3 || r[1] == 0 || r[2] == 0 {
		t.Fatalf("restored %v: the later choices ran the picture's prompt again", r)
	}
}

// TestImageContinuesASession: a continued session's next turn appends its
// picture to a history that already holds the first one. The continued turn
// answers differently from the same turn on a fresh session (the history kept
// the first picture), its own picture changes its answer (the new one was
// appended), and the session's position grows past the first turn's.
func TestImageContinuesASession(t *testing.T) {
	e, lm, _ := loadedEngine(t, visionModel, "vlm", LoadOptions{})
	quad, pink := readTestdata(t, "quad-and-disc.png"), readTestdata(t, "blue-purple-pink.lossy.webp")
	turn := func(sid string, cont bool, pic []byte, text string) string {
		t.Helper()
		var out strings.Builder
		err := e.Generate(context.Background(), GenerateOptions{SessionID: sid, Continue: cont, MaxTokens: 12,
			Sampling: &model.Sampler{}, Logprobs: true,
			Prompt: Prompt{Kind: PromptChat, Chat: &ChatInput{AddGenerationPrompt: true,
				Messages: []model.ChatMessage{{Role: "user", Content: text, Images: 1}},
				Images:   []image.Image{decodePic(t, pic)}}}},
			func(ev Event) error {
				if ev.Kind == EventToken {
					// The text and each token's log-probability: a lost
					// history can give the same words, not the same values.
					fmt.Fprintf(&out, "%s|%.6f ", ev.Token.Text, ev.Token.Logprob.Logprob)
				}
				return nil
			})
		if err != nil {
			t.Fatal(err)
		}
		return out.String()
	}
	const second = "And this one? Describe it in one sentence."
	two := func(first, next []byte) (string, int, int) {
		s, err := e.CreateSession(SessionOptions{ModelID: lm.id})
		if err != nil {
			t.Fatal(err)
		}
		defer e.CloseSession(s.id)
		turn(s.id, false, first, askPicture)
		p1 := s.st.Pos()
		got := turn(s.id, true, next, second)
		return got, p1, s.st.Pos()
	}
	fresh := func(pic []byte) string {
		s, err := e.CreateSession(SessionOptions{ModelID: lm.id})
		if err != nil {
			t.Fatal(err)
		}
		defer e.CloseSession(s.id)
		return turn(s.id, false, pic, second)
	}
	qp, p1, p2 := two(quad, pink)
	qq, _, _ := two(quad, quad)
	alone := fresh(pink)
	t.Logf("quad then pink: %q; quad then quad: %q; pink alone: %q; positions %d -> %d", qp, qq, alone, p1, p2)
	if qp == alone {
		t.Error("the continued turn answers as a fresh session does: the history lost the first picture")
	}
	if qp == qq {
		t.Error("the second turn's picture does not change its answer: it was not appended")
	}
	if p1 == 0 || p2 <= p1 {
		t.Fatalf("the session went from %d to %d positions: the second turn did not continue it", p1, p2)
	}
}

// TestConnectGenerateTakesPictures: a chat's pictures sent through the Connect
// API give the engine's own image run's tokens; a model with no tower, a bad
// picture and ApplyChatTemplate with a picture are each InvalidArgument.
func TestConnectGenerateTakesPictures(t *testing.T) {
	e, lm, c := loadedEngine(t, visionModel, "vlm", LoadOptions{})
	quad := readTestdata(t, "quad-and-disc.png")
	want, _ := engineRun(t, e, lm, quad)
	chat := func(model string, img *v1.ChatImage) *v1.GenerateRequest {
		return &v1.GenerateRequest{ModelId: model, MaxTokens: 12, Prompt: &v1.PromptInput{
			Input: &v1.PromptInput_Chat{Chat: &v1.ChatPrompt{AddGenerationPrompt: true,
				Messages: []*v1.ChatMessage{{Role: "user", Content: askPicture, Images: []*v1.ChatImage{img}}}}}}}
	}
	res, err := c.inference.Complete(context.Background(), req(chat("vlm", &v1.ChatImage{Data: quad, MediaType: "image/png"})))
	if err != nil {
		t.Fatal(err)
	}
	if res.Msg.Text != want {
		t.Fatalf("Connect: %q, want the engine's %q", res.Msg.Text, want)
	}
	_, err = c.inference.Complete(context.Background(), req(chat("vlm", &v1.ChatImage{Data: quad, MediaType: "image/webp"})))
	if ce := wantCode(t, "a mismatched media type", err, connect.CodeInvalidArgument); !strings.Contains(ce.Message(),
		"prompt.chat.messages[0].images[0] is declared image/webp") {
		t.Fatalf("the refusal does not name the part: %s", ce.Message())
	}
	_, err = c.model.ApplyChatTemplate(context.Background(), req(&v1.ApplyChatTemplateRequest{ModelId: "vlm",
		Messages: []*v1.ChatMessage{{Role: "user", Content: "x", Images: []*v1.ChatImage{{Data: quad}}}}}))
	wantCode(t, "ApplyChatTemplate with a picture", err, connect.CodeInvalidArgument)

	_, _, tc := loadedEngine(t, chatModel, "txt", LoadOptions{})
	_, err = tc.inference.Complete(context.Background(), req(chat("txt", &v1.ChatImage{Data: quad})))
	if ce := wantCode(t, "a picture to a model with no tower", err, connect.CodeInvalidArgument); !strings.Contains(ce.Message(),
		`model "txt" does not accept images`) {
		t.Fatalf("%s", ce.Message())
	}
}
