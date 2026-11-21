package jlm

import (
	"fmt"
	"strings"
)

// ToolForm is how a model writes a tool call, as text with holes: the
// converter states it from the chat template the weights were trained to
// emit (convert.ToolFormOf), and one reader in the engine matches a reply
// against these templates, and one grammar forces a call into them. Nothing
// past the converter knows which model a form is.
//
// A template is text and holes. Text matches as written, a run of
// whitespace in it matching any run (or none); a hole reads:
//
//	$NAME   a tool's name           $QNAME  the name as a quoted string
//	$XNAME  the name as an attribute's escaped text
//	$ARGS   the arguments, one object in the form's notation
//	$CALL   one call written as an object, its name and arguments by key
//	$CALLS  one such call, or a list of them
//	$PARAMS the arguments one by one, each a Params template
//	$KEY    an argument's key       $VALUE  its value
//	$TYPE   a word naming the value's type (Types)
//	$ID     an id the model numbers its calls by (IDDigits)
//
// and (?...?) is the text inside, written or not.
type ToolForm struct {
	// Name is the form's name, for a message.
	Name string
	// Calls are the templates a call is written in, tried in order.
	Calls []string
	// Open, Sep and Close are written around a reply's calls and between
	// them; Section is written around each call. Bare says a reply may be
	// the calls alone, Open and Close left out.
	Open, Sep, Close string
	Section          [2]string
	Bare             bool
	// Leads says the reply must open with its calls: Open (or the first
	// call) counts only at the reply's start.
	Leads bool
	// Params are the templates an argument of $PARAMS is written in, tried
	// in order, ParamSep between two. A string argument is written in the
	// first, any other in the last; Sorted writes them by key.
	Params   []string
	ParamSep string
	Sorted   bool
	// TextValues says a $VALUE is the text up to the template's next mark:
	// a string, or JSON where its type ($TYPE, else the tool's schema) says
	// it is not one, or when it reads as JSON and nothing types it. Without
	// it a $VALUE is one value in the notation.
	TextValues bool
	// GuessJSON reads an untyped text value that is valid JSON as JSON
	// rather than as a string.
	GuessJSON bool
	// Types maps a $TYPE word to the value's type: "string" or "json".
	// A word it does not hold is a schema type name.
	Types map[string]string
	// NameKey and ArgsKey are a $CALL's keys, AltArgsKey read where ArgsKey
	// is absent; Keyed writes a call as {NAME: {...}}.
	NameKey, ArgsKey, AltArgsKey string
	Keyed                        bool
	Notation                     ToolNotation
	// IDDigits is how many characters an $ID has at most, IDMin its least
	// first digit when it is a number (0: any), IDAlnum letters and digits
	// in exactly IDDigits.
	IDDigits, IDMin int
	IDAlnum         bool
	// Several says a reply may hold more than one call.
	Several bool
	// ThinkOpen and ThinkClose are the marks a turn's reasoning is written
	// between (<think> and </think> when empty); ThinkSeen is the text a
	// template that writes such blocks holds (ThinkClose when empty),
	// ThinkAlways says a reply may reason whatever the template, and
	// ThinkThen is what follows a closed reasoning block before the call.
	ThinkOpen, ThinkClose, ThinkSeen, ThinkThen string
	ThinkAlways                                 bool
	// Markers are the texts a forced call writes that a vocabulary may hold
	// as control tokens.
	Markers []string
	// Write is what a forced call writes: one of the shapes the form reads.
	Write ToolWrite
}

// ToolWrite is the text a forced call is held to, in ToolForm's templates:
// several calls are Open, the calls with Sep between, and Close, each in
// Section. A string argument of $PARAMS is written in Params[0], its text
// up to the template's next mark; any other in the last of Params, as the
// form's notation writes values.
type ToolWrite struct {
	Call             string
	Open, Sep, Close string
	Section          [2]string
	Params           []string
	ParamSep         string
	Sorted           bool
}

// ToolNotation is how a value is written: JSON's notation is Quotes {"\""},
// Escapes, and the words true, false and null.
type ToolNotation struct {
	// Quotes open and close a string, any one of them.
	Quotes []string
	// Escapes says a backslash in a string escapes the next character.
	Escapes bool
	// BareKeys says an object's key may be written without quotes.
	BareKeys bool
	// True, False and None are the words those values are written as.
	True, False, None []string
}

// Valid is whether every template of f is written in holes it states how to
// read.
func (f *ToolForm) Valid() error {
	if f.Write.Call == "" {
		return fmt.Errorf("jlm: the %s tool form states no forced call", f.Name)
	}
	if len(f.Calls) == 0 {
		return fmt.Errorf("jlm: the %s tool form states no call", f.Name)
	}
	if len(f.Notation.Quotes) == 0 {
		return fmt.Errorf("jlm: the %s tool form states no string quote", f.Name)
	}
	for _, c := range f.Calls {
		switch {
		case strings.Contains(c, "$PARAMS") && len(f.Params) == 0:
			return fmt.Errorf("jlm: the %s tool form writes $PARAMS and states no parameter", f.Name)
		case (strings.Contains(c, "$CALL")) && !f.Keyed && (f.NameKey == "" || f.ArgsKey == ""):
			return fmt.Errorf("jlm: the %s tool form writes $CALL and states no keys", f.Name)
		case strings.Contains(c, "$ID") && f.IDDigits == 0:
			return fmt.Errorf("jlm: the %s tool form writes $ID and states no width", f.Name)
		}
	}
	return nil
}
