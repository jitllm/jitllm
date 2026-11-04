//go:build jitllmbench

// This file is a measurement instrument, not a gate, so it is opt-in by build
// tag rather than by a skip in every default run:
//
//	go test -tags jitllmbench -run TestTeacherForcedTrace ./engine/model
//
// JITLLM_TRACE_* variables select its parameters.

package model_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/model"
)

// TestTeacherForcedTrace prints the top logits along a teacher-forced chain and
// writes the logits and every traced intermediate (Model.Trace) at one
// position as raw little-endian f32, for diffing against a reference layer by
// layer.
//
//	JITLLM_TRACE_MODEL   a GGUF or container
//	JITLLM_TRACE_PROMPT  the prompt, encoded with the model's specials
//	JITLLM_TRACE_IDS     the ids fed after it, comma-separated
//	JITLLM_TRACE_POS     the position to write (0 is the prompt's last row)
//	JITLLM_TRACE_OUT     the directory written: logits.bin, <name>.<layer>.bin
//	JITLLM_TRACE_KVF16   1 for the f16 KV cache, else f32
//	JITLLM_TRACE_DECODE  1 to feed the prompt one token at a time
func TestTeacherForcedTrace(t *testing.T) {
	path := os.Getenv("JITLLM_TRACE_MODEL")
	if path == "" {
		t.Skip("set JITLLM_TRACE_MODEL")
	}
	prompt := os.Getenv("JITLLM_TRACE_PROMPT")
	var forced []int32
	for _, f := range strings.Split(os.Getenv("JITLLM_TRACE_IDS"), ",") {
		if f == "" {
			continue
		}
		v, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil {
			t.Fatal(err)
		}
		forced = append(forced, int32(v))
	}
	dump, _ := strconv.Atoi(os.Getenv("JITLLM_TRACE_POS"))
	out := os.Getenv("JITLLM_TRACE_OUT")
	f16 := os.Getenv("JITLLM_TRACE_KVF16") == "1"
	c := jlmOf(t, path)
	m, err := model.Open(c, model.WithKVF16(f16))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode(prompt, true)
	t.Logf("prompt ids %v", ids)
	st := m.NewInstance(len(ids) + len(forced) + 2)
	defer st.Close()
	var logits []float32
	if os.Getenv("JITLLM_TRACE_DECODE") == "1" {
		for _, id := range ids {
			if logits, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
	} else if logits, err = st.Prefill(ids); err != nil {
		t.Fatal(err)
	}
	writeF := func(name string, v []float32) {
		b := make([]byte, 4*len(v))
		for i, x := range v {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
		}
		if err := os.WriteFile(filepath.Join(out, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i <= len(forced); i++ {
		idx := make([]int, len(logits))
		for j := range idx {
			idx[j] = j
		}
		sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
		var sb strings.Builder
		for _, j := range idx[:8] {
			fmt.Fprintf(&sb, " %d %q %.4f |", j, m.Vocab.Text(int32(j)), logits[j])
		}
		t.Logf("pos %d:%s", i, sb.String())
		if i == dump && out != "" {
			writeF("logits.bin", logits)
		}
		if i == len(forced) {
			break
		}
		if i+1 == dump && out != "" {
			m.Trace(func(l int, name string, v []float32) {
				writeF(fmt.Sprintf("%s.%d.bin", strings.ReplaceAll(name, "/", "_"), l), v)
			})
		}
		if logits, err = st.Forward(forced[i]); err != nil {
			t.Fatal(err)
		}
		m.Trace(nil)
	}
}
