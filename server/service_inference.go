package server

import (
	"context"
	"strings"
	"unicode/utf8"

	"connectrpc.com/connect"

	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// InferenceService implements jitllm.v1.InferenceService. It holds a Backend
// so its streaming path can be gated without a loaded model.
type InferenceService struct {
	B Backend
	// Images bounds a chat's pictures, as it does on the HTTP shims.
	Images ImagePolicy
}

// generateOptions turns the wire request into the engine's one request shape,
// the same struct the HTTP shims build, with the default picture limits.
// The request's fields are read in service_generate.go.
func generateOptions(m *v1.GenerateRequest) (GenerateOptions, error) {
	return generateOptionsWith(context.Background(), m, ImagePolicy{})
}

func (s *InferenceService) Generate(ctx context.Context, req *connect.Request[v1.GenerateRequest], st *connect.ServerStream[v1.GenerateResponse]) error {
	o, err := generateOptionsWith(ctx, req.Msg, s.Images)
	if err != nil {
		return err
	}
	run := newGenRun(s.B, o)
	err = s.B.Generate(ctx, o, func(ev Event) error {
		return run.handle(ev, st.Send)
	})
	if err != nil {
		return connectErr(err)
	}
	if req.Msg.IncludeUsage {
		if err := run.done(); err != nil {
			return connect.NewError(connect.CodeInternal, err)
		}
		return st.Send(&v1.GenerateResponse{Event: &v1.GenerateResponse_Usage{
			Usage: &v1.GenerateUsage{Usage: run.usage()}}})
	}
	return nil
}

// Complete is the streaming path with the events collected: the same call
// with a different sink.
func (s *InferenceService) Complete(ctx context.Context, req *connect.Request[v1.GenerateRequest]) (*connect.Response[v1.CompleteResponse], error) {
	o, err := generateOptionsWith(ctx, req.Msg, s.Images)
	if err != nil {
		return nil, err
	}
	run := newGenRun(s.B, o)
	cs := make([]*v1.CompleteChoice, len(run.cs))
	for i := range cs {
		cs[i] = &v1.CompleteChoice{Index: int32(i)}
	}
	err = s.B.Generate(ctx, o, func(ev Event) error {
		return run.handle(ev, func(r *v1.GenerateResponse) error {
			c := cs[r.Choice]
			switch e := r.Event.(type) {
			case *v1.GenerateResponse_Started:
				c.Started = e.Started
			case *v1.GenerateResponse_Token:
				if e.Token.TokenId >= 0 {
					c.TokenIds = append(c.TokenIds, e.Token.TokenId)
				}
				if e.Token.Logprob != nil {
					c.Logprobs = append(c.Logprobs, e.Token.Logprob)
				}
			case *v1.GenerateResponse_ToolCalls:
				c.ToolCalls = append(c.ToolCalls, e.ToolCalls.Calls...)
			case *v1.GenerateResponse_Finished:
				c.Finished = e.Finished
			}
			return nil
		})
	})
	if err != nil {
		return nil, connectErr(err)
	}
	if err := run.done(); err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}
	out := &v1.CompleteResponse{Usage: run.usage()}
	for i, c := range cs {
		// The text is what the engine sent, or when the reply made tool
		// calls, the reply without them: the content the OpenAI surface
		// answers with.
		c.Text = run.cs[i].raw.String()
		if len(run.cs[i].calls) > 0 {
			c.Text = run.cs[i].content
		}
		c.Text = validText(c.Text)
		out.Choices = append(out.Choices, c)
	}
	first := out.Choices[0]
	out.Started, out.Text, out.TokenIds, out.Finished = first.Started, first.Text, first.TokenIds, first.Finished
	return connect.NewResponse(out), nil
}

func (s *InferenceService) Cancel(ctx context.Context, req *connect.Request[v1.CancelRequest]) (*connect.Response[v1.CancelResponse], error) {
	was, err := s.B.CancelSession(req.Msg.SessionId)
	if err != nil {
		return nil, connectErr(err)
	}
	return connect.NewResponse(&v1.CancelResponse{WasGenerating: was}), nil
}

// validText is s as a protobuf string may carry it: a completion's bytes are
// the tokens', and a reply cut at its token limit -- or a model emitting a
// lone byte token -- ends inside a UTF-8 sequence, which the wire refuses to
// marshal. Each invalid byte becomes U+FFFD, as the JSON surfaces' encoder
// writes it.
func validText(s string) string {
	if utf8.ValidString(s) {
		return s
	}
	return strings.ToValidUTF8(s, "\uFFFD")
}

func pbEvent(ev Event) *v1.GenerateResponse {
	switch ev.Kind {
	case EventStarted:
		return &v1.GenerateResponse{Event: &v1.GenerateResponse_Started{Started: pbStarted(ev.Started)}}
	case EventToken:
		return &v1.GenerateResponse{Event: &v1.GenerateResponse_Token{Token: &v1.GenerateToken{
			TokenId: ev.Token.ID, Text: validText(ev.Token.Text), Index: int32(ev.Token.Index),
		}}}
	default:
		return &v1.GenerateResponse{Event: &v1.GenerateResponse_Finished{Finished: pbFinished(ev.Finished)}}
	}
}

func pbStarted(s *Started) *v1.GenerateStarted {
	return &v1.GenerateStarted{
		SessionId:         s.SessionID,
		ModelId:           s.ModelID,
		EphemeralSession:  s.Ephemeral,
		PromptTokens:      int32(s.PromptTokens),
		QueuedMillis:      millis(s.QueuedFor),
		QueueDepthOnEntry: s.QueueDepth,
		DeviceBlocks:      int32(s.DeviceBlocks),
		HostBlocks:        int32(s.HostBlocks),
		DeviceIds:         s.DeviceIDs,
		PrefillMillis:     millis(s.Prefill),
		Batched:           s.Batched,
	}
}

func pbFinished(f *Finished) *v1.GenerateFinished {
	return &v1.GenerateFinished{
		Reason:                pbFinish(f.Reason),
		StopMatched:           f.StopMatched,
		PromptTokens:          int32(f.PromptTokens),
		CompletionTokens:      int32(f.CompletionTokens),
		PrefillMillis:         millis(f.Prefill),
		DecodeMillis:          millis(f.Decode),
		DecodeTokensPerSecond: f.TokensPerSecond,
		BytesPerToken:         f.BytesPerToken,
		Position:              int32(f.Position),
		CachedTokens:          int32(f.Restored),
	}
}

// Embed is Engine.Embed with the wire request turned into EmbedOptions, the
// same struct /v1/embeddings builds, and refused for the same inputs in the
// same words.
func (s *InferenceService) Embed(ctx context.Context, req *connect.Request[v1.EmbedRequest]) (*connect.Response[v1.EmbedResponse], error) {
	m := req.Msg
	if m.Dimensions < 0 {
		return nil, invalid("dimensions must be at least 1")
	}
	if len(m.Inputs) == 0 {
		return nil, invalid("input cannot be an empty array")
	}
	o := EmbedOptions{
		Model:      m.ModelId,
		Dimensions: int(m.Dimensions),
	}
	for i, in := range m.Inputs {
		switch p := in.GetInput().(type) {
		case *v1.EmbedInput_Text:
			if p.Text == "" {
				return nil, invalid("input[%d] cannot be an empty string", i)
			}
			o.Inputs = append(o.Inputs, Prompt{Kind: PromptText, Text: p.Text})
		case *v1.EmbedInput_TokenIds:
			o.Inputs = append(o.Inputs, Prompt{Kind: PromptIDs, IDs: p.TokenIds.GetIds()})
		default:
			return nil, invalid("embed: input %d carries neither text nor token ids", i)
		}
	}
	res, err := s.B.Embed(ctx, o)
	if err != nil {
		return nil, connectErr(err)
	}
	out := &v1.EmbedResponse{
		ModelId:           res.ModelID,
		Pooling:           res.Pooling,
		TotalTokens:       int32(res.TotalTokens()),
		QueuedMillis:      millis(res.QueuedFor),
		QueueDepthOnEntry: res.QueueDepth,
		EmbedMillis:       millis(res.Took),
	}
	for i, v := range res.Vectors {
		out.Embeddings = append(out.Embeddings, &v1.Embedding{Vector: v, Tokens: int32(res.Tokens[i])})
		out.Dimensions = int32(len(v))
	}
	return connect.NewResponse(out), nil
}
