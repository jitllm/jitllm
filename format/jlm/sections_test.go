package jlm

import (
	"reflect"
	"strings"
	"testing"
)

// Every field round-trips, and the gate fills them all rather than sampling:
// a field added to the encoder and not the decoder shifts every field after
// it and produces a plausible wrong model. reflect.DeepEqual on a
// fully-populated struct is the check that sees it.
func TestSectionsRoundTrip(t *testing.T) {
	cfg := &Config{
		Arch: ArchQwen3MoE, NLayer: 28, NEmbd: 2048, NHead: 16, NKVHead: 8, HeadDim: 128,
		NRot: 128, NFFN: 6144, NVocab: 151936, NCtx: 40960, RMSEps: 1e-6, RopeBase: 1e6,
		AttnFactor: 1.5, EmbdScale: 45.25, NExpert: 128, NExpertUsed: 8, NFFNExp: 768,
		SWAWindow: 1024, SWAPeriod: 6, RopeBaseSWA: 10000,
		Flags: FlagGELU | FlagTiedEmbd | FlagQKNorm | FlagRopeNeox,
		// The hybrid fields, filled so the round trip covers them: a short
		// array here would shift every field after it and produce a plausible
		// wrong model rather than an error.
		NFFNShExp:  512,
		SSM:        SSMConfig{ConvKernel: 4, Groups: 16, Inner: 4096, StateSize: 128, NHeadV: 32},
		LayerKinds: []LayerKind{LayerLinearAttn, LayerLinearAttn, LayerLinearAttn, LayerFullAttn},
	}
	cfg.NLayer = uint32(len(cfg.LayerKinds))
	got, err := decodeConfig(encodeConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, cfg) {
		t.Errorf("config round-trip:\n got %+v\nwant %+v", got, cfg)
	}

	vis := &Vision{
		NLayer: 12, NEmbd: 768, NHead: 12, NFFN: 3072, ImageSize: 512, PatchSize: 16,
		ProjDim: 576, Scale: 4, Eps: 1e-6, Projector: ProjIdefics3,
		Mean: [3]float32{0.5, 0.5, 0.5}, Std: [3]float32{0.5, 0.5, 0.5}, Flags: FlagGELU,
	}
	gv, err := decodeVision(encodeVision(vis))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gv, vis) {
		t.Errorf("vision round-trip:\n got %+v\nwant %+v", gv, vis)
	}
	// The optional tail, alone and with the fields before it set.
	for _, tail := range []Vision{{MinPixels: 65536, MaxPixels: 921600},
		{MinTiles: 1, MaxTiles: 12, WinPattern: 8, WinSize: 112, MinPixels: 3136, MaxPixels: 313600},
		{TileSize: 512, Tolerance: 2},
		{MinTiles: 2, MaxTiles: 10, MinPixels: 65536, MaxPixels: 262144, TileSize: 512, Tolerance: 2}} {
		v := *vis
		v.Projector = ProjPhi4
		v.MinTiles, v.MaxTiles, v.WinPattern, v.WinSize = tail.MinTiles, tail.MaxTiles, tail.WinPattern, tail.WinSize
		v.MinPixels, v.MaxPixels = tail.MinPixels, tail.MaxPixels
		v.TileSize, v.Tolerance = tail.TileSize, tail.Tolerance
		gv, err := decodeVision(encodeVision(&v))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(gv, &v) {
			t.Errorf("vision round-trip with its tail:\n got %+v\nwant %+v", gv, &v)
		}
	}

	voc := &Vocab{
		Kind:   VocabBPE,
		Tokens: []string{"<unk>", "Ġthe", "🙂", "with space"},
		Scores: []float32{0, -1, -2, -3},
		Kinds:  []TokenKind{TokenUnknown, TokenNormal, TokenNormal, TokenUserDefined},
		// A merge whose side contains a space, which the joined "left right"
		// form cannot represent.
		Merges: []Merge{{"Ġt", "he"}, {"with", " space"}},
		Added:  []AddedToken{{ID: 3, Content: "<|im_start|>", Special: true, LStrip: true, SingleWord: true}},
		BOS:    1, EOS: 2, Unk: 0, Pad: -1, Sep: -1, Mask: -1, Stop: []int32{3, 1},
		AddBOS: true, AddEOS: false, AddSpacePrefix: true, ByteFallback: true, IgnoreMerges: true,
		Pre: []PreOp{
			{Kind: PreSplit, Contractions: true, CaseFold: true, LetterMarks: true, DigitGroup: 3},
			{Kind: PreByteLevel, UseRegex: true, AddPrefixSpace: false},
			{Kind: PreDigits, Individual: true},
			{Kind: PrePunctuation, Behavior: "Isolated"},
		},
		PreName:   "llama-bpe",
		Templates: []ChatTemplate{{"", "{{ bos }}"}, {"tool_use", "{{ tools }}"}},
	}
	gvo, err := decodeVocab(encodeVocab(voc))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gvo, voc) {
		t.Errorf("vocab round-trip:\n got %+v\nwant %+v", gvo, voc)
	}

	// The stop ids ride a tail behind the per-op flags, which are then
	// written even when no op sets one so a reader can find where it starts;
	// here with no op at all.
	sv := *voc
	sv.Pre = nil
	if got, err := decodeVocab(encodeVocab(&sv)); err != nil {
		t.Fatal(err)
	} else if !reflect.DeepEqual(got, &sv) {
		t.Errorf("vocab round-trip with stop ids and no ops:\n got %+v\nwant %+v", got, &sv)
	}
	// An id the token table does not hold is refused, not carried to a
	// generation loop that compares against it forever.
	bad := *voc
	bad.Stop = []int32{int32(len(voc.Tokens))}
	if _, err := decodeVocab(encodeVocab(&bad)); err == nil {
		t.Error("a stop id past the token table was accepted")
	}
}

// A section is untrusted input and must error, never panic, for any bytes
// (RULE 9).
func TestSectionsRefuseGarbage(t *testing.T) {
	full := encodeVocab(&Vocab{
		Kind: VocabSPM, Tokens: []string{"a", "b"}, Scores: []float32{0, 1},
		Kinds: []TokenKind{TokenNormal, TokenNormal}, PreName: "x",
	})
	// Every truncation of a valid section.
	for i := 0; i < len(full); i++ {
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("decodeVocab panicked on a %d-byte prefix: %v", i, p)
				}
			}()
			decodeVocab(full[:i])
		}()
	}
	// A count field that claims more than the file holds.
	b := append([]byte(nil), full...)
	b[1], b[2], b[3], b[4] = 0xFF, 0xFF, 0xFF, 0x7F // token count
	if _, err := decodeVocab(b); err == nil {
		t.Error("a vocab claiming 2 billion tokens was accepted")
	} else if !strings.Contains(err.Error(), "elements") && !strings.Contains(err.Error(), "truncated") {
		t.Errorf("refused with %q, which does not say what was wrong", err)
	}
	if _, err := decodeConfig([]byte{1, 2, 3}); err == nil {
		t.Error("a 3-byte config was accepted")
	}
	// A config whose architecture code this format does not define.
	c := encodeConfig(&Config{Arch: Arch(9999), NLayer: 1})
	if _, err := decodeConfig(c); err == nil {
		t.Error("an unknown architecture code was accepted")
	}
}
