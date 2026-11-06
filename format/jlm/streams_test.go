package jlm

import (
	"reflect"
	"strings"
	"testing"
)

// TestStreamsRoundTrip holds the residual's stated streams to the config's
// tails' contract: they round-trip, a config without them writes no tail of
// them, and a statement that names a stream, a block or an input it does
// not have is refused rather than read.
func TestStreamsRoundTrip(t *testing.T) {
	base := func() *Config {
		return &Config{Arch: ArchLlama, NLayer: 2, NEmbd: 64, NHead: 4, NKVHead: 2,
			HeadDim: 16, NRot: 16, NFFN: 128, NVocab: 1000, RMSEps: 1e-5, RopeBase: 1e4}
	}
	read := func(from ...int8) SubStreams {
		p := &Program{NIn: uint16(len(from))}
		for j := range from {
			p.Ops = append(p.Ops, TowerOp{Code: TOpRMSNorm, In: []uint16{uint16(j)}, Eps: 1e-5})
		}
		ins := make([]uint16, len(from))
		for j := range ins {
			ins[j] = uint16(len(from) + j)
		}
		p.Ops = append(p.Ops, TowerOp{Code: TOpConcat, In: ins}, TowerOp{Code: TOpSlice, In: []uint16{uint16(2 * len(from))}, Width: 64})
		p.Out = uint16(2*len(from) + 1)
		return SubStreams{From: from, Read: p}
	}
	stated := func() *Config {
		c := base()
		c.Streams = Streams{N: 3, Blocks: []BlockStreams{
			{Bank: 1, Attn: SubStreams{Join: &Program{NIn: 2, Out: 1}}, FFN: read(StreamIn, StreamRun)},
			{Attn: read(1, StreamRun), FFN: read(1, 2, StreamRun)},
		}, Head: read(1, 2, StreamRun)}
		return c
	}
	got, err := decodeConfig(encodeConfig(stated()))
	if err != nil {
		t.Fatal(err)
	}
	if want := stated().Streams; !reflect.DeepEqual(got.Streams, want) {
		t.Fatalf("round trip:\n got %+v\nwant %+v", got.Streams, want)
	}
	if plain, _ := decodeConfig(encodeConfig(base())); plain.Streams.N != 0 || len(plain.Streams.Blocks) != 0 {
		t.Fatalf("a config without streams read as %+v", plain.Streams)
	}
	for _, v := range []struct {
		name, want string
		bad        func(c *Config)
	}{
		{"a block too few", "state 1 blocks of 2", func(c *Config) { c.Streams.Blocks = c.Streams.Blocks[:1] }},
		{"a bank past the streams", "banks into stream 3", func(c *Config) { c.Streams.Blocks[0].Bank = 3 }},
		{"a read past the streams", "reads stream 3", func(c *Config) { c.Streams.Blocks[1].Attn.From[0] = 3 }},
		{"a read of more inputs than the program", "through a program of", func(c *Config) {
			c.Streams.Head.From = append(c.Streams.Head.From, 1)
		}},
		{"a join of one input", "joins through a program of 1", func(c *Config) { c.Streams.Blocks[0].Attn.Join.NIn = 1 }},
		{"an empty join past its inputs", "returns value 2", func(c *Config) { c.Streams.Blocks[0].Attn.Join.Out = 2 }},
		{"a slice of nothing", "slices 0", func(c *Config) { c.Streams.Head.Read.Ops[len(c.Streams.Head.Read.Ops)-1].Width = 0 }},
	} {
		c := stated()
		v.bad(c)
		if _, err := decodeConfig(encodeConfig(c)); err == nil || !strings.Contains(err.Error(), v.want) {
			t.Errorf("%s: %v, want an error naming %q", v.name, err, v.want)
		}
	}
}
