package server

import (
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"io"
	"net/http"
	"strings"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/tok/jinja"
)

// TypeSafe-compatible POST /v1/systemone: the System One request, as the
// TypeSafe SDK sends it and as llama-server answers it. An adapter like the
// rest of the shim: it builds DecideOptions and calls Backend.Decide.
//
// The request is read with every object's keys in the order the client wrote
// them, since the state is rendered into the prompt and a reordered object is
// a different prompt. A malformed request is a 422 in FastAPI's shape
// ({"detail": [{"loc", "msg", "type"}]}), which is what TypeSafe's API
// returns.

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
	qs, err := engine.ParseDecisionQuestions(qv)
	if err != nil {
		s1Fail(w, http.StatusUnprocessableEntity, err.Error(), "questions")
		return
	}
	name := ""
	if mv, ok := req.Get("model"); ok && mv.IsString() {
		name = mv.AsString()
	}
	if name == "" {
		// A server with one model loaded answers with it.
		if ls := c.b.ListLoaded(); len(ls) == 1 {
			name = ls[0].ID
		}
	}
	pics := newPictures(c.img)
	if iv, ok := req.Get("images"); ok && !iv.IsNone() {
		if !iv.IsList() {
			s1Fail(w, http.StatusUnprocessableEntity, "must be a list of image URLs", "images")
			return
		}
		for i, u := range iv.AsList().Items {
			if !u.IsString() {
				s1Fail(w, http.StatusUnprocessableEntity, "must be an image URL", "images", i)
				return
			}
			url := u.AsString()
			if err := pics.add(fmt.Sprintf("images[%d]", i), func() (image.Image, error) {
				return pics.p.imageURL(r.Context(), url)
			}); err != nil {
				s1Fail(w, http.StatusUnprocessableEntity, err.Error(), "images", i)
				return
			}
		}
	}
	// "files" are voice clips, each a data URL of an audio type over a WAV
	// the engine reads (engine.DecodeWAV).
	var clips [][]float32
	if fv, ok := req.Get("files"); ok && !fv.IsNone() {
		if !fv.IsList() {
			s1Fail(w, http.StatusUnprocessableEntity, "must be a list of audio data URLs", "files")
			return
		}
		for i, u := range fv.AsList().Items {
			clip, err := clipURL(u)
			if err != nil {
				s1Fail(w, http.StatusUnprocessableEntity, err.Error(), "files", i)
				return
			}
			clips = append(clips, clip)
		}
	}
	res, err := c.b.Decide(r.Context(), DecideOptions{Model: name, State: state, Questions: qs, Images: pics.imgs, Clips: clips})
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
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(res.Body)
}

// clipURL is a request file's samples: a data URL of an audio type, base64,
// over a WAV the engine reads.
func clipURL(u jinja.Value) ([]float32, error) {
	if !u.IsString() {
		return nil, fmt.Errorf("must be an audio data URL")
	}
	head, data, ok := strings.Cut(u.AsString(), ",")
	if !ok || !strings.HasPrefix(head, "data:audio/") || !strings.HasSuffix(head, ";base64") {
		return nil, fmt.Errorf("must be a base64 data URL of an audio type")
	}
	raw, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return nil, fmt.Errorf("is not base64: %v", err)
	}
	return engine.DecodeWAV(raw)
}
