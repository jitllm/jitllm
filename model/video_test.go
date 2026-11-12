package model

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// operand is the expression whose value is operand i.
func operand(i int) jlm.Expr {
	var e jlm.ExprBuilder
	e.Fn(jlm.FnAdd, jlm.Operand(uint32(i)), jlm.Num(0))
	return jlm.Expr{Terms: e.T}
}

// pictureVideo reads each frame as a picture, at most seven of a video,
// every other frame's index halved and each taken once; its prompt writes
// the count from a table, the length as a clock with days, and each frame
// numbered from 1 with its time cut to two places, a line between one and
// the next.
func pictureVideo() *jlm.Video {
	var c, ix, j1 jlm.ExprBuilder
	c.Fn(jlm.FnMin, jlm.Operand(jlm.VideoN), jlm.Num(7))
	ix.Fn(jlm.FnFloor, ix.Fn(jlm.FnMul, jlm.Operand(jlm.VideoAt), jlm.Num(0.5)))
	j1.Fn(jlm.FnAdd, jlm.Operand(jlm.VideoGroup), jlm.Num(1))
	text := func(s string) jlm.VideoPiece { return jlm.VideoPiece{Kind: jlm.PieceText, Text: s} }
	ids := func(i int32) jlm.VideoPiece { return jlm.VideoPiece{Kind: jlm.PieceIDs, IDs: []int32{i}} }
	return &jlm.Video{Count: jlm.Expr{Terms: c.T}, Index: jlm.Expr{Terms: ix.T}, Once: true, Group: 1, Sized: operand(0),
		Prompt: []jlm.VideoPiece{
			text("A video of "),
			{Kind: jlm.PieceValue, Value: operand(jlm.VideoK), Format: jlm.ValueFormat{Kind: jlm.FormatTable,
				Strings: []string{"no", "one", "two", "three", "four"}}},
			text(" frames, "),
			{Kind: jlm.PieceValue, Value: operand(jlm.VideoSeconds), Format: jlm.ValueFormat{Kind: jlm.FormatClock,
				Clock: [3]int32{1, 2, 2}, Sep: ":", Days: [2]string{" day, ", " days, "}}},
			ids(7),
			{Kind: jlm.PieceGroups, Between: []jlm.VideoPiece{text("\n")}, Body: []jlm.VideoPiece{
				text("Frame"),
				{Kind: jlm.PieceValue, Value: jlm.Expr{Terms: j1.T}, Format: jlm.ValueFormat{Kind: jlm.FormatDecimal}},
				text(" at "),
				{Kind: jlm.PieceValue, Value: operand(jlm.VideoT0), Format: jlm.ValueFormat{Kind: jlm.FormatDecimal, Digits: 2, Trunc: true}},
				text(": "), ids(8), {Kind: jlm.PieceSpan}, ids(9),
			}},
			text("\n\n"),
		}}
}

// written is a laid-out prompt written out: its text as it is, each id in
// brackets, each group's rows its number in braces.
func written(items []VideoItem) string {
	var b strings.Builder
	for _, it := range items {
		switch {
		case it.Group >= 0:
			fmt.Fprintf(&b, "{%d}", it.Group)
		case it.IDs != nil:
			for _, id := range it.IDs {
				fmt.Fprintf(&b, "[%d]", id)
			}
		default:
			b.WriteString(it.Text)
		}
	}
	return b.String()
}

// TestAStatedVideoIsEvaluated holds the evaluation of a video's statement
// (jlm.Video) to what each of its pieces states, on a statement no family
// has: the count and indices evaluated, a frame taken twice kept once, the
// text between two ids or rows one item, each group's operands its own and
// the values written as their formats say.
func TestAStatedVideoIsEvaluated(t *testing.T) {
	v := pictureVideo()
	if err := v.Valid(); err != nil {
		t.Fatal(err)
	}
	s, err := SampleVideo(v, 10, 2)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(s.Frames) != "[0 1 2 3]" || s.K != 4 || s.Rate != 2 {
		t.Errorf("ten frames at 2 a second sample %v (%d), at %v a second", s.Frames, s.K, s.Rate)
	}
	items, err := VideoLayout(v, s)
	if err != nil {
		t.Fatal(err)
	}
	want := "A video of four frames, 0:00:05[7]Frame1 at 0.00: [8]{0}[9]\nFrame2 at 0.50: [8]{1}[9]\n" +
		"Frame3 at 1.00: [8]{2}[9]\nFrame4 at 1.50: [8]{3}[9]\n\n"
	if got := written(items); got != want {
		t.Errorf("the prompt reads\n%q\nwant\n%q", got, want)
	}
	if len(items) != 19 || items[0].Text == "" || items[len(items)-1].Text != "\n\n" {
		t.Errorf("the prompt is %d items, the first %+v and the last %+v", len(items), items[0], items[len(items)-1])
	}
	// No rate stated: the statement's own, here none, so the times are not
	// numbers, and writing one is refused.
	if s, err := SampleVideo(v, 10, 0); err != nil {
		t.Fatal(err)
	} else if _, err := VideoLayout(v, s); err == nil {
		t.Error("a video of no rate wrote its length")
	}
	// Six frames sample three, past the table's words.
	v.Prompt[1].Format.Strings = v.Prompt[1].Format.Strings[:3]
	if s, err := SampleVideo(v, 6, 2); err != nil {
		t.Fatal(err)
	} else if _, err := VideoLayout(v, s); err == nil {
		t.Errorf("a count of %d was written from a table of 3", s.K)
	}
}

// TestAVideoOfPairsFillsItsLastPair holds a statement read in pairs to its
// bounds: the last frame repeated to fill a pair, a count past its bound
// refused, an index that is not a number refused, and its stride evaluated
// over the rate.
func TestAVideoOfPairsFillsItsLastPair(t *testing.T) {
	var stride, nan jlm.ExprBuilder
	stride.Fn(jlm.FnMul, jlm.Num(2), stride.Fn(jlm.FnTrunc, stride.Fn(jlm.FnDiv, jlm.Num(2), jlm.Operand(jlm.VideoRate))))
	st := jlm.Expr{Terms: stride.T}
	v := &jlm.Video{Count: operand(jlm.VideoN), Index: operand(jlm.VideoAt), MaxFrames: 4, Group: 2, Sized: operand(1), Stride: &st}
	for _, c := range []struct {
		n      int
		fps    float64
		frames string
		k      int
		stride int
	}{{3, 0.5, "[0 1 2 2]", 3, 8}, {4, 30, "[0 1 2 3]", 4, 0}, {1, 1, "[0 0]", 1, 4}} {
		s, err := SampleVideo(v, c.n, c.fps)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprint(s.Frames) != c.frames || s.K != c.k || s.Stride != c.stride {
			t.Errorf("%d frames at %v: %v (%d), stride %d; want %s (%d), %d", c.n, c.fps, s.Frames, s.K, s.Stride, c.frames, c.k, c.stride)
		}
	}
	if _, err := SampleVideo(v, 5, 1); err == nil {
		t.Error("five frames were sampled past a bound of four")
	}
	nan.Fn(jlm.FnDiv, jlm.Operand(jlm.VideoAt), nan.Fn(jlm.FnSub, jlm.Operand(jlm.VideoK), jlm.Num(1)))
	v.Index = jlm.Expr{Terms: nan.T}
	if _, err := SampleVideo(v, 1, 1); err == nil {
		t.Error("a frame at 0/0 was sampled")
	}
}

// TestVideoValuesAreWrittenAsStated holds each format to the text it
// writes.
func TestVideoValuesAreWrittenAsStated(t *testing.T) {
	clock := jlm.ValueFormat{Kind: jlm.FormatClock, Clock: [3]int32{1, 2, 2}, Sep: ":", Days: [2]string{" day, ", " days, "}}
	minutes := jlm.ValueFormat{Kind: jlm.FormatClock, Clock: [3]int32{0, 2, 2}, Sep: ":"}
	for _, c := range []struct {
		f    jlm.ValueFormat
		v    float64
		want string
	}{
		{jlm.ValueFormat{Kind: jlm.FormatDecimal, Digits: 1}, 0.25, "0.2"},
		{jlm.ValueFormat{Kind: jlm.FormatDecimal, Digits: 1}, 0.35, "0.3"},
		{jlm.ValueFormat{Kind: jlm.FormatDecimal, Digits: 1}, 25.25, "25.2"},
		{jlm.ValueFormat{Kind: jlm.FormatDecimal, Digits: 2, Trunc: true}, 0.999, "0.99"},
		{jlm.ValueFormat{Kind: jlm.FormatDecimal, Trunc: true}, 7.9, "7"},
		{jlm.ValueFormat{Kind: jlm.FormatDecimal}, 7.5, "8"},
		{clock, 59.9, "0:00:59"},
		{clock, 90061, "1 day, 1:01:01"},
		{clock, 176461, "2 days, 1:01:01"},
		{minutes, 3725, "62:05"},
		{jlm.ValueFormat{Kind: jlm.FormatTable, Strings: []string{"a", "b"}}, 1.5, "b"},
	} {
		got, err := formatValue(&c.f, c.v)
		if err != nil || got != c.want {
			t.Errorf("%v as %+v: %q, %v; want %q", c.v, c.f, got, err, c.want)
		}
	}
	for _, c := range []struct {
		f jlm.ValueFormat
		v float64
	}{
		{clock, -1}, {clock, math.NaN()},
		{jlm.ValueFormat{Kind: jlm.FormatTable, Strings: []string{"a", "b"}}, 2},
		{jlm.ValueFormat{Kind: jlm.FormatTable, Strings: []string{"a"}}, -0.5},
		{jlm.ValueFormat{Kind: 9}, 1},
	} {
		if got, err := formatValue(&c.f, c.v); err == nil {
			t.Errorf("%v as %+v was written %q", c.v, c.f, got)
		}
	}
}

// TestVideoSizesAreEvaluatedAsStated holds each function a video's sizes
// state to its value at halves and either sign, where the roundings part.
func TestVideoSizesAreEvaluatedAsStated(t *testing.T) {
	for _, c := range []struct {
		fn   jlm.Fn
		args []float64
		want float64
	}{
		{jlm.FnFloor, []float64{-1.5}, -2}, {jlm.FnCeil, []float64{-1.5}, -1}, {jlm.FnTrunc, []float64{-1.5}, -1},
		{jlm.FnRound, []float64{2.5}, 3}, {jlm.FnRoundEven, []float64{2.5}, 2}, {jlm.FnRoundEven, []float64{3.5}, 4},
		{jlm.FnRoundEven, []float64{-2.5}, -2}, {jlm.FnF32, []float64{0.1}, float64(float32(0.1))},
		{jlm.FnWhere, []float64{1, 2, 3}, 2}, {jlm.FnWhere, []float64{0, 2, 3}, 3}, {jlm.FnLess, []float64{1, 2}, 1},
		{jlm.FnLess, []float64{2, 2}, 0}, {jlm.FnEqual, []float64{2, 2}, 1}, {jlm.FnFma, []float64{2, 3, 4}, 10},
		{jlm.FnMax, []float64{2, 3}, 3}, {jlm.FnMin, []float64{2, 3}, 2}, {jlm.FnDiv, []float64{1, 4}, 0.25},
		{jlm.FnSqrt, []float64{16}, 4}, {jlm.FnAbs, []float64{-3}, 3}, {jlm.FnRecip, []float64{4}, 0.25},
	} {
		var args [3]jlm.Arg
		for i := range c.args {
			args[i] = jlm.Operand(uint32(i))
		}
		e := jlm.Expr{Terms: []jlm.Term{{Fn: c.fn, Args: args}}}
		got, err := evalSize(&e, c.args)
		if err != nil || got != c.want {
			t.Errorf("%v%v: %v, %v; want %v", c.fn, c.args, got, err, c.want)
		}
	}
	e := jlm.Expr{Terms: []jlm.Term{{Fn: jlm.FnTanh, Args: [3]jlm.Arg{jlm.Operand(0)}}}}
	if _, err := evalSize(&e, []float64{1}); err == nil {
		t.Error("a size was evaluated through tanh")
	}
}
