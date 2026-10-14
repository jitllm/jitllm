package server

import (
	"errors"
	"fmt"
	"image"
	"io"
	"net/http"
	"strings"

	"github.com/jitllm/jitllm/engine/model"
	"github.com/jitllm/jitllm/tok/jinja"
)

// TypeSafe-compatible POST /v1/systemone: the System One request, as the
// TypeSafe SDK sends it (typesafe-sdk's OpenAPI models) and as `lev serve`,
// `laya-serve` and llama-server answer it. An adapter like the rest of the
// shim: it builds DecideOptions and calls Backend.Decide.
//
// The request is read with every object's keys in the order the client wrote
// them, since the state is rendered into the prompt and a reordered object is
// a different prompt. A malformed request is a 422 in FastAPI's shape
// ({"detail": [{"loc", "msg", "type"}]}), which is what TypeSafe's API and the
// reference servers return.

// maxSystemOneBody bounds a request body; a state is text, not a payload.
const maxSystemOneBody = 16 << 20

type s1Detail struct {
	Loc  []any  `json:"loc"`
	Msg  string `json:"msg"`
	Type string `json:"type"`
}

func s1Fail(w http.ResponseWriter, status int, msg string, loc ...any) {
	writeJSON(w, status, map[string][]s1Detail{"detail": {{Loc: append([]any{"body"}, loc...), Msg: msg, Type: "value_error"}}})
}

func (c *compat) systemOne(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s1Fail(w, http.StatusMethodNotAllowed, "POST only")
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, maxSystemOneBody+1))
	if err != nil {
		s1Fail(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(body) > maxSystemOneBody {
		s1Fail(w, http.StatusRequestEntityTooLarge, "the request is larger than 16 MiB")
		return
	}
	v, err := jinja.FromJSON(body)
	if err != nil || !v.IsDict() {
		s1Fail(w, http.StatusUnprocessableEntity, "the body must be a JSON object")
		return
	}
	req := v.AsDict()
	state, ok := req.Get("state")
	if !ok {
		s1Fail(w, http.StatusUnprocessableEntity, "Field required", "state")
		return
	}
	qv, ok := req.Get("questions")
	if !ok {
		s1Fail(w, http.StatusUnprocessableEntity, "Field required", "questions")
		return
	}
	qs, err := model.ParseDecisionQuestions(qv)
	if err != nil {
		s1Fail(w, http.StatusUnprocessableEntity, err.Error(), "questions")
		return
	}
	// The pictures: "files" and "images" data URLs, and the image_url parts
	// of a chat-message state, as llama.cpp's server reads them.
	state, urls, err := model.DecisionMedia(req, state)
	if err != nil {
		s1Fail(w, http.StatusUnprocessableEntity, err.Error())
		return
	}
	pics := newPictures(c.img)
	for i, u := range urls {
		err := pics.add(fmt.Sprintf("images[%d]", i), func() (image.Image, error) { return pics.p.decodeDataURL(u) })
		if err != nil {
			s1Fail(w, http.StatusUnprocessableEntity, err.Error())
			return
		}
	}
	name := ""
	if mv, ok := req.Get("model"); ok && mv.IsString() {
		name = mv.AsString()
	}
	if name == "" {
		// TypeSafe's request names a model; a server with one model loaded
		// answers with it, as lev serve and llama-server do.
		if ls := c.b.ListLoaded(); len(ls) == 1 {
			name = ls[0].ID
		}
	}
	res, err := c.b.Decide(r.Context(), DecideOptions{Model: name, State: state, Questions: qs, Images: pics.imgs})
	if err != nil {
		switch {
		case errors.Is(err, ErrInvalid):
			s1Fail(w, http.StatusUnprocessableEntity, strings.TrimPrefix(err.Error(), ErrInvalid.Error()+": "))
		case errors.Is(err, ErrNotFound):
			s1Fail(w, http.StatusNotFound, err.Error(), "model")
		default:
			oaFailErr(w, err)
		}
		return
	}
	writeRawJSON(w, http.StatusOK, model.DecisionResponse(res.ModelID, qs, res.Answers, res.InputTokens))
}

func writeRawJSON(w http.ResponseWriter, status int, b []byte) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(b)
}
