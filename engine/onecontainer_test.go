package engine

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestOneContainerCarriesTextAndVision converts a text model and its vision
// projector into one container: both block arrays in one file, the tower's
// blocks past the text's in one block space, each array its own page size,
// and one budget binding across both.
func TestOneContainerCarriesTextAndVision(t *testing.T) {
	src := jlmOfPair(t, testmodels.Path("SmolVLM-256M-Instruct-Q8_0.gguf"), testmodels.Path("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"))
	c, err := jlm.Open(src)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	nText, nVis := int(c.H.NBlocks), c.VisionBlocks()
	if nText == 0 || nVis == 0 {
		t.Fatalf("the container has %d text and %d vision blocks; one file must carry both", nText, nVis)
	}
	if got := c.NPages(); got != nText+nVis {
		t.Fatalf("the block space is %d pages for %d text and %d vision blocks", got, nText, nVis)
	}
	text, vis := c.PageBytes(0), c.PageBytes(nText)
	if text == 0 || vis == 0 {
		t.Fatalf("the text and vision pages are %d and %d bytes", text, vis)
	}
	if b := c.Budget(); b != 0 && b < text+vis {
		t.Fatalf("a budget of %d holds no page of each array (%d and %d)", b, text, vis)
	}
	t.Logf("%d text blocks of %d-byte pages, %d vision blocks of %d-byte pages, in one block space", nText, text, nVis, vis)
}
