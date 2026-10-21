package srcgate

import (
	"bufio"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode"
)

// The compiler's packages name what a block does, never which model it came
// from (AGENTS.md, "The compiler names mechanisms, not models"): a field,
// type, function or variable named after a model family -- or after one
// family's mechanism (k3, q4e, ds4, altRouter, mla, msa, qsa) -- is a family
// switch in disguise, and the device program decides by the graph's
// structure instead.
//
// TestNoFamilyNamedMechanisms parses those packages and fails on any such
// identifier that is not on the ratchet list (testdata/family_names.txt):
// the names the block encoder still carries, each to go as its feature
// moves into the device program. A listed name that no longer exists fails
// too, so the list only shrinks.
//
// VIOLATION SIGNATURE. Add a field `q4eGate` to tier's layer struct, or a
// func `emitGemmaNorm`, and this names it.

func TestNoFamilyNamedMechanisms(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	allowed := readAllowed(t, filepath.Join("testdata", "family_names.txt"))
	found := map[string]bool{}
	// The compiler's packages, relative to the module root.
	for _, dir := range [...]string{"jit_old/gpu/tier", "engine/graph", "engine/nn"} {
		for _, name := range familyIdents(t, filepath.Join(root, dir)) {
			key := dir + "." + name
			found[key] = true
			if !allowed[key] {
				t.Errorf("%s: %q names a model family; name the mechanism by its structure (AGENTS.md)", dir, name)
			}
		}
	}
	for key := range allowed {
		if !found[key] {
			t.Errorf("%s is on the ratchet list and no longer exists: remove it from testdata/family_names.txt", key)
		}
	}
	t.Logf("%d family-named identifiers left in the compiler's packages", len(found))
}

// familyIdents is every identifier declared at package level in dir's
// non-test files -- types, funcs and methods, vars, consts and struct
// fields -- that names a model family, sorted and unique.
func familyIdents(t *testing.T, dir string) []string {
	fs := token.NewFileSet()
	files, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	add := func(id *ast.Ident) {
		if id != nil && familyNamed(id.Name) {
			out = append(out, id.Name)
		}
	}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		af, err := parser.ParseFile(fs, f, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range af.Decls {
			switch d := d.(type) {
			case *ast.FuncDecl:
				add(d.Name)
			case *ast.GenDecl:
				for _, s := range d.Specs {
					switch s := s.(type) {
					case *ast.TypeSpec:
						add(s.Name)
						if st, ok := s.Type.(*ast.StructType); ok {
							for _, fl := range st.Fields.List {
								for _, n := range fl.Names {
									add(n)
								}
							}
						}
					case *ast.ValueSpec:
						for _, n := range s.Names {
							add(n)
						}
					}
				}
			}
		}
	}
	slices.Sort(out)
	return slices.Compact(out)
}

// familyNamed reports name's camel or snake tokens include a family or a
// family's mechanism; it is matched by token, so mv.alt (an alternative) is
// not altRouter (Gemma 3n's AltUp).
func familyNamed(name string) bool {
	low := strings.ToLower(name)
	// Names that begin a family mechanism's (AltUp's router, prediction and
	// correction).
	if strings.HasPrefix(low, "altrouter") || strings.HasPrefix(low, "altpred") || strings.HasPrefix(low, "altcorr") {
		return true
	}
	for _, tok := range identTokens(name) {
		if familyToken(tok) {
			return true
		}
	}
	return false
}

// familyToken reports tok names a model family or one family's mechanism.
func familyToken(tok string) bool {
	switch tok {
	case "k3", "q4e", "ds4", "mla", "msa", "qsa", "altup", "laurel",
		"kimi", "qwen", "gemma", "llama", "deepseek", "mistral", "mixtral", "olmo", "olmoe", "granite", "hunyuan",
		"glm", "falcon", "jamba", "nemotron", "bitnet", "apertus", "ernie", "minimax", "gptoss", "starcoder",
		"stablelm", "smollm", "exaone", "cohere", "dbrx", "arcee", "bailing", "pixtral", "internvl", "minicpm",
		"janus", "idefics", "llava", "phimoe", "clef", "laya", "kev":
		return true
	}
	return false
}

// identTokens splits an identifier at underscores, lower-to-upper changes
// and the end of an upper-case run, lower-cased: "emitMLARows" is emit, mla,
// rows; "q4eAttnIn" is q4e, attn, in.
func identTokens(name string) []string {
	var toks []string
	var cur []rune
	rs := []rune(name)
	flush := func() {
		if len(cur) > 0 {
			toks = append(toks, strings.ToLower(string(cur)))
			cur = cur[:0]
		}
	}
	for i, r := range rs {
		switch {
		case r == '_':
			flush()
			continue
		case unicode.IsUpper(r) && i > 0 && (unicode.IsLower(rs[i-1]) ||
			unicode.IsUpper(rs[i-1]) && i+1 < len(rs) && unicode.IsLower(rs[i+1])):
			flush()
		}
		cur = append(cur, r)
	}
	flush()
	return toks
}

func readAllowed(t *testing.T, path string) map[string]bool {
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	out := map[string]bool{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if line := strings.TrimSpace(sc.Text()); line != "" && !strings.HasPrefix(line, "#") {
			out[line] = true
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}
