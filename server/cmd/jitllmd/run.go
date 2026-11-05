package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"connectrpc.com/connect"

	"github.com/jitllm/jitllm/nn"
	v1 "github.com/jitllm/jitllm/server/gen/jitllm/v1"
)

// cmdRun streams a generation from a running jitllmd. Each token is written
// to stdout and flushed as it arrives, since a slow model buffered to the end
// looks like a hang. The started/finished envelopes go to stderr, so stdout is
// the completion and nothing else.
func cmdRun(ctx context.Context, c cli, args []string) error {
	fs := flag.NewFlagSet("jitllmd run", flag.ContinueOnError)
	addr := addrFlag(fs)

	// Session or model, not both: one continues a sequence, the other makes
	// a throwaway, and the server refuses the pair.
	modelID := fs.String("model", "",
		"model id (or file name) to generate against, in an ephemeral session")
	sessionID := fs.String("session", "",
		"an existing session to generate in, keeping its KV cache and position")

	n := fs.Int("n", 0, "tokens to generate; 0 runs until the model ends its reply or fills the session's context")
	chat := fs.Bool("chat", false, "wrap the prompt in the model's own chat template (instruct models)")
	system := fs.String("system", "", "a system message, with -chat")
	echo := fs.Bool("echo", false, "have the server emit the prompt's tokens back before generating")
	cont := fs.Bool("continue", false,
		"continue from the session's position instead of prefilling the prompt from scratch")
	stop := fs.String("stop", "", "comma-separated stop strings")
	ignoreEOS := fs.Bool("ignore-eos", false,
		"keep generating past an end-of-generation token, up to -n or the end of the context (stop strings still stop)")
	queueMS := fs.Int("queue-timeout", 0,
		"milliseconds to wait for a device another session is holding; 0 waits indefinitely")

	temp := fs.Float64("temp", 0, "sampling temperature; 0 = greedy (default)")
	topK := fs.Int("top-k", 0, "keep only the k most likely tokens; 0 = off")
	topP := fs.Float64("top-p", 0, "nucleus sampling; 0 or 1 = off")
	minP := fs.Float64("min-p", 0, "keep tokens above min-p * p(best); 0 = off")
	repPen := fs.Float64("repeat-penalty", 1, "penalise recently seen tokens; 1 = off")
	repN := fs.Int("repeat-last-n", 64, "how far back repeat-penalty looks")
	seed := fs.Int64("seed", 0, "RNG seed for sampling")

	ids := fs.Bool("ids", false, "print the generated token ids to stderr when finished")
	quiet := fs.Bool("quiet", false, "suppress the started/finished lines on stderr")

	if err := parse(c, fs, args); err != nil {
		return err
	}
	prompt := strings.Join(fs.Args(), " ")

	if *sessionID != "" && *modelID != "" {
		return usagef("give -session or -model, not both: -session continues a sequence " +
			"and -model makes a throwaway one, and picking one silently would make a stateful " +
			"client stateless without telling it")
	}
	if prompt == "" && !*cont {
		return usagef("run: no prompt. Give one as the last argument, or -continue to carry " +
			"on from a session's position")
	}
	if *system != "" && !*chat {
		return usagef("-system needs -chat: without a template there is no system turn to " +
			"render, and a raw completion of a system message is fluent and meaningless")
	}

	cn, err := target(ctx, c, fs, *addr)
	if err != nil {
		return err
	}

	// The control plane looks models up by id only (unlike the compat shims'
	// BindTarget), so a bare name is resolved here from ListModels, and the
	// choice is printed.
	if *sessionID == "" {
		resolved, err := resolveModel(ctx, cn, *modelID)
		if err != nil {
			return err
		}
		if resolved != *modelID && !*quiet {
			fmt.Fprintf(c.err, "model    %s\n", resolved)
		}
		*modelID = resolved
	}

	req := &v1.GenerateRequest{
		SessionId:          *sessionID,
		ModelId:            *modelID,
		MaxTokens:          int32(*n),
		Stop:               csv(*stop),
		ContinueSession:    *cont,
		EchoPrompt:         *echo,
		QueueTimeoutMillis: int32(*queueMS),
		IgnoreEos:          *ignoreEOS,
		Sampling:           samplingFrom(fs, temp, topP, topK, minP, repPen, repN, seed),
	}
	if prompt != "" {
		req.Prompt = promptFrom(prompt, *chat, *system)
	}

	st, err := cn.Inference.Generate(ctx, connect.NewRequest(req))
	if err != nil {
		return clientError("generate", cn.base, err)
	}
	defer st.Close()

	var (
		started  *v1.GenerateStarted
		finished *v1.GenerateFinished
		tokIDs   []int32
	)
	for st.Receive() {
		switch ev := st.Msg().GetEvent().(type) {
		case *v1.GenerateResponse_Started:
			started = ev.Started
			if !*quiet {
				printStarted(c.err, started)
			}
		case *v1.GenerateResponse_Token:
			tokIDs = append(tokIDs, ev.Token.GetTokenId())
			// An empty text is not a dropped token: the server holds back a
			// partial UTF-8 sequence, so it means "nothing printable yet".
			if t := ev.Token.GetText(); t != "" {
				if err := emit(c.out, t); err != nil {
					return fmt.Errorf("writing the completion: %w", err)
				}
			}
		case *v1.GenerateResponse_Finished:
			finished = ev.Finished
		}
	}
	if err := st.Err(); err != nil && ctx.Err() == nil {
		return clientError("generate", cn.base, err)
	}

	if !*quiet {
		printFinished(c.err, started, finished, ctx.Err() != nil)
	}
	if *ids {
		fmt.Fprintf(c.err, "ids      %v\n", tokIDs)
	}
	if finished == nil && ctx.Err() == nil {
		// A stream that ends without a finished event is a truncation,
		// unless the caller stopped it with ^C, which is not a fault.
		return fmt.Errorf("the stream ended without a finished event: the generation was cut " +
			"short rather than completed")
	}
	return nil
}

// resolveModel turns what the caller typed into a model id the server knows.
func resolveModel(ctx context.Context, cn *conn, want string) (string, error) {
	resp, err := cn.Model.ListModels(ctx, connect.NewRequest(&v1.ListModelsRequest{LoadedOnly: true}))
	if err != nil {
		// When the caller named a model, a failed lookup must not pre-empt
		// the generate, whose own error is the one worth showing.
		if want != "" {
			return want, nil
		}
		return "", clientError("listing loaded models", cn.base, err)
	}
	loaded := resp.Msg.GetLoaded()
	if want != "" {
		for _, m := range loaded {
			if m.GetModelId() == want {
				return want, nil
			}
		}
		for _, m := range loaded {
			if m.GetName() == want {
				return m.GetModelId(), nil
			}
		}
		// Not ours to decide: hand it to the server and let its own refusal
		// arrive. It knows things this list does not.
		return want, nil
	}
	switch len(loaded) {
	case 0:
		return "", fmt.Errorf("no model is loaded: start the daemon with `jitllmd serve -load ...`, " +
			"or load one with `jitllmd models -load <path>`")
	case 1:
		return loaded[0].GetModelId(), nil
	}
	var names []string
	for _, m := range loaded {
		names = append(names, m.GetModelId())
	}
	return "", usagef("%d models are loaded -- name one with -model: %s",
		len(loaded), strings.Join(names, ", "))
}

func promptFrom(text string, chat bool, system string) *v1.PromptInput {
	if !chat {
		return &v1.PromptInput{Input: &v1.PromptInput_Text{Text: text}}
	}
	// The server renders the template (model.ChatIDs pairs it with the
	// tokenizer so BOS is not doubled). The system turn travels in its own
	// field, as the proto keeps it.
	cp := &v1.ChatPrompt{
		Messages:            []*v1.ChatMessage{{Role: "user", Content: text}},
		AddGenerationPrompt: true,
	}
	if system != "" {
		s := system
		cp.System = &s
	}
	return &v1.PromptInput{Input: &v1.PromptInput_Chat{Chat: cp}}
}

// samplingFrom sends only the knobs the caller actually typed. Every
// SamplingParams field is optional and unset means "the session's own
// sampler", so a client default must not override it.
func samplingFrom(fs *flag.FlagSet, temp, topP *float64, topK *int, minP, repPen *float64, repN *int, seed *int64) *v1.SamplingParams {
	p := &v1.SamplingParams{}
	any := false
	if wasSet(fs, "temp") {
		v := float32(*temp)
		p.Temperature, any = &v, true
	}
	if wasSet(fs, "top-p") {
		v := float32(*topP)
		p.TopP, any = &v, true
	}
	if wasSet(fs, "top-k") {
		v := int32(*topK)
		p.TopK, any = &v, true
	}
	if wasSet(fs, "min-p") {
		v := float32(*minP)
		p.MinP, any = &v, true
	}
	if wasSet(fs, "repeat-penalty") {
		v := float32(*repPen)
		p.RepeatPenalty, any = &v, true
	}
	if wasSet(fs, "repeat-last-n") {
		v := int32(*repN)
		p.RepeatLastN, any = &v, true
	}
	if wasSet(fs, "seed") {
		v := uint64(*seed)
		p.Seed, any = &v, true
	}
	if !any {
		return nil
	}
	return p
}

// printStarted reports what a token stream alone cannot say. The queue line is
// printed even when zero, so an empty queue is distinguishable from an
// unmeasured one.
func printStarted(w io.Writer, s *v1.GenerateStarted) {
	if s == nil {
		return
	}
	where := nn.CPU
	if s.GetDeviceBlocks() > 0 {
		where = strings.Join(s.GetDeviceIds(), ",")
		if where == "" {
			where = "device"
		}
	}
	kind := "session"
	if s.GetEphemeralSession() {
		kind = "ephemeral session"
	}
	fmt.Fprintf(w, "started  model %s   %s %s   prompt %d tok   prefill %s\n",
		s.GetModelId(), kind, s.GetSessionId(), s.GetPromptTokens(),
		durMS(s.GetPrefillMillis()))
	fmt.Fprintf(w, "queued   %s behind %d request(s)\n",
		durMS(s.GetQueuedMillis()), s.GetQueueDepthOnEntry())
	fmt.Fprintf(w, "blocks   %d on %s, %d on cpu\n",
		s.GetDeviceBlocks(), where, s.GetHostBlocks())
}

// printFinished reports the rate with the bytes behind it, so it can be
// checked against the machine's read wall. Every number is the server's own;
// the client takes no stopwatch.
func printFinished(w io.Writer, s *v1.GenerateStarted, f *v1.GenerateFinished, cancelled bool) {
	if f == nil {
		// "Stopped" (^C) and "cut short" (a dead connection) leave the same
		// partial text; this line tells them apart.
		if cancelled {
			fmt.Fprintf(w, "\nstopped by the caller; the tokens above are real and the "+
				"generation did not finish\n")
			return
		}
		fmt.Fprintf(w, "\nfinished (no finished event: the stream was cut short)\n")
		return
	}
	stop := ""
	if m := f.GetStopMatched(); m != "" {
		stop = fmt.Sprintf(" on %q", m)
	}
	fmt.Fprintf(w, "\nfinished %s%s   %d prompt + %d completion tok   position %d\n",
		finishName(f.GetReason()), stop,
		f.GetPromptTokens(), f.GetCompletionTokens(), f.GetPosition())

	where := nn.CPU
	if s != nil && s.GetDeviceBlocks() > 0 {
		if ids := strings.Join(s.GetDeviceIds(), ","); ids != "" {
			where = fmt.Sprintf("%s + host", ids)
		}
	}
	rate := f.GetDecodeTokensPerSecond()
	bpt := f.GetBytesPerToken()
	line := fmt.Sprintf("decode   %d tok in %s = %.2f tok/s",
		f.GetCompletionTokens(), durMS(f.GetDecodeMillis()), rate)
	if bpt == 0 {
		// An unreported byte count is said to be missing, not printed as
		// 0.00 GB/s.
		line += "   (bytes/token not reported: this rate cannot be checked against a roofline)"
	} else {
		line += fmt.Sprintf(" = %s B/token = %.2f GB/s", commas(bpt), rate*float64(bpt)/1e9)
	}
	fmt.Fprintf(w, "%s   [%s]\n", line, where)
	fmt.Fprintf(w, "prefill  %d tok in %s\n", f.GetPromptTokens(), durMS(f.GetPrefillMillis()))
}

func durMS(ms int64) string {
	return (time.Duration(ms) * time.Millisecond).String()
}
