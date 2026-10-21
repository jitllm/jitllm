//go:build linux

package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestChatTemplateIsAppliedAndSpecialsAreNotDoubled: the chat template must be
// rendered, and its specials paired correctly. A rendered template is a
// complete prompt (Llama-3.2's opens with `{{- bos_token }}`, Qwen's emits no
// BOS), so encoding it with addSpecial=true doubles the first, and hardcoding
// specials gives the second one it never saw in training. Both produce text.
func TestChatTemplateIsAppliedAndSpecialsAreNotDoubled(t *testing.T) {
	paths := testmodels.Glob("*.gguf")
	checked, addsBOS := 0, 0
	for _, p := range paths {
		fi, err := os.Stat(p)
		if err != nil || fi.Size() > 2<<30 {
			continue // a metadata+tokenizer test wants breadth, not the big files
		}
		m, err := Open(jlmOf(t, p))
		if err != nil {
			continue
		}
		name := filepath.Base(p)
		func() {
			defer m.Close()
			if m.Vocab == nil || !m.HasChatTemplate() {
				return // a base model, which is a legitimate answer
			}
			checked++

			text, err := m.ChatPrompt([]ChatMessage{{Role: "user", Content: "hello"}}, true)
			if err != nil {
				t.Errorf("%s: %v", name, err)
				return
			}
			if !strings.Contains(text, "hello") {
				t.Errorf("%s: the rendered prompt does not contain the user's message:\n%q",
					name, text)
			}

			ids, err := m.ChatIDs([]ChatMessage{{Role: "user", Content: "hello"}}, true)
			if err != nil {
				t.Errorf("%s: %v", name, err)
				return
			}
			if len(ids) == 0 {
				t.Errorf("%s: chat prompt tokenized to nothing", name)
				return
			}

			// Counting BOS tokens is not the assertion: SmolLM2's BOS is the turn
			// marker, legitimately present once per turn. The property is that
			// the tokenizer adds no special the template did not render, so
			// compare against the naive pairing (the rendered text encoded with
			// addSpecial=true) and require strictly fewer specials wherever the
			// vocabulary would add one.
			cv := m.container.Vocab()
			// The naive pairing must parse specials too (EncodeSpecial), or on a
			// SentencePiece vocabulary both sides encode "<bos>" as prose.
			naive := m.Vocab.EncodeSpecial(text, true)
			if cv.BOS >= 0 && int(cv.BOS) < cv.Tokens.Len() && strings.HasPrefix(text, cv.Tokens.At(int(cv.BOS))) &&
				ids[0] != cv.BOS {
				t.Errorf("%s: the template renders %q first and the chat ids start %d, not its id %d",
					name, cv.Tokens.At(int(cv.BOS)), ids[0], cv.BOS)
			}
			count := func(ids []int32, id int32) int {
				n := 0
				for _, x := range ids {
					if x == id {
						n++
					}
				}
				return n
			}
			if cv.AddBOS && cv.BOS >= 0 {
				addsBOS++
				if got, want := count(ids, cv.BOS), count(naive, cv.BOS); got >= want {
					t.Errorf("%s: this vocabulary adds BOS, and the chat prompt has %d of "+
						"token %d against the naive pairing's %d -- the rendered template "+
						"must be encoded with addSpecial=false or its own bos_token is "+
						"doubled", name, got, cv.BOS, want)
				}
				if ids[0] != naive[1] {
					t.Errorf("%s: chat ids start %d, the naive pairing starts %d,%d -- the "+
						"two should differ by exactly the added BOS", name, ids[0], naive[0], naive[1])
				}
			}

			// The chat prompt must not be the bare text, or the feature is a
			// no-op this test would call a pass.
			raw := m.Vocab.Encode("hello", true)
			if len(ids) <= len(raw) {
				t.Errorf("%s: chat prompt is %d tokens against a raw %d -- the template "+
					"added no turn structure", name, len(ids), len(raw))
			}
			t.Logf("%-46s %3d tok (naive pairing %3d), AddBOS=%v", name, len(ids), len(naive), cv.AddBOS)
		}()
	}
	if checked < 3 {
		testmodels.Missing(t, "only %d models with a chat template were reached; this gate is vacuous below 3", checked)
	}
	// At least one model must be one the bug could happen to (AddBOS set, e.g.
	// Llama-3.2-1B-Instruct), or the pairing is never exercised.
	if addsBOS == 0 {
		t.Errorf("no model reached has AddBOS set, so the doubling this gate exists for " +
			"was never reachable in it")
	}
}

// TestBaseModelRefusesAChatTemplate: a model without one says so rather than
// rendering an empty prompt and answering from nothing.
func TestBaseModelRefusesAChatTemplate(t *testing.T) {
	m, err := Open(jlmOf(t, testmodels.Path("stories15M-q8_0.gguf")))
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()
	if m.HasChatTemplate() {
		t.Skip("stories15M has a chat template now; pick another base model")
	}
	if _, err := m.ChatPrompt([]ChatMessage{{Role: "user", Content: "hi"}}, true); err == nil {
		t.Fatal("a base model rendered a chat prompt; it must refuse by name")
	}
}
