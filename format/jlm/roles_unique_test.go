package jlm

import (
	_ "embed"
	"regexp"
	"strconv"
	"testing"
)

// Embedded rather than read from the working directory, so the check runs on
// hosts that get only the cross-compiled test binary (an arm64 Mac, say):
// a gate that cannot run on a host does not cover it (RULE 11d).
//
//go:embed schema.go
var schemaSrc string

// TestRoleCodesAreUnique is the gate the compiler only half provides.
//
// A role code is an on-disk identifier, so two roles sharing one resolve
// tensors to the wrong graph node, silently: both names exist and both
// lookups succeed. A duplicate map key catches it only for named roles.
func TestRoleCodesAreUnique(t *testing.T) {
	seen := map[Role]string{}
	for c := 0; c <= 0xffff; c++ {
		r := Role(c)
		var names []string
		if n, ok := coreRoleName(r); ok {
			names = append(names, n)
		}
		for _, group := range roleGroups() {
			if n, ok := group(r); ok {
				names = append(names, n)
			}
		}
		if len(names) > 1 {
			t.Errorf("role code %d is named by two groups: %q", r, names)
		}
		if len(names) == 0 {
			continue
		}
		seen[r] = names[0]
		if r == RoleNone {
			t.Errorf("%q uses the zero code, which means \"no role\"", names[0])
		}
	}
	if len(seen) < 40 {
		t.Fatalf("only %d roles named -- this gate proved nothing", len(seen))
	}
	t.Logf("%d role codes, all distinct", len(seen))
}

// reservedRoles are role codes the schema declares and deliberately does not
// name, so Valid() refuses them and no container can carry one.
//
// They are reserved rather than deleted because a role code is an on-disk
// identifier: the linear block reads RoleAttnQKV and RoleAttnGate (see
// convert/roles.go), these two never got a name, and reusing either code
// later would make an old container resolve to the wrong graph node.
func reservedRole(r Role) (string, bool) {
	switch r {
	case RoleSSMInProj:
		return "the linear block reads RoleAttnQKV", true
	case RoleSSMOutGate:
		return "the linear block reads RoleAttnGate", true
	}
	return "", false
}

// TestEveryDeclaredRoleIsNamedOrReserved reads the schema's own constants and
// requires each one to be either named -- and therefore writable -- or listed
// above as deliberately not.
//
// It parses the source rather than taking a hand-kept bound, which goes
// stale as the list grows; and it must not skip unnamed roles, or it only
// restates Valid()'s definition.
func TestEveryDeclaredRoleIsNamedOrReserved(t *testing.T) {
	re := regexp.MustCompile(`(?m)^\s*(Role[A-Za-z0-9]*)\s+Role\s*=\s*(\d+)`)
	ms := re.FindAllStringSubmatch(schemaSrc, -1)
	if len(ms) < 40 {
		t.Fatalf("found %d role declarations in schema.go -- this gate parsed nothing", len(ms))
	}
	var unnamed int
	for _, m := range ms {
		code, err := strconv.Atoi(m[2])
		if err != nil {
			t.Fatalf("%s = %q: %v", m[1], m[2], err)
		}
		r := Role(code)
		if r == RoleNone {
			continue
		}
		_, named := roleName(r)
		why, reserved := reservedRole(r)
		switch {
		case named && reserved:
			t.Errorf("%s (%d) is both named %q and reserved (%s) -- pick one",
				m[1], code, r, why)
		case named:
			if !r.Valid() {
				t.Errorf("%s (%d) is named %q and Valid() says otherwise", m[1], code, r)
			}
		case reserved:
			unnamed++
			if r.Valid() {
				t.Errorf("%s (%d) is reserved (%s) and Valid() accepts it -- a container could carry one",
					m[1], code, why)
			}
		default:
			t.Errorf("%s (%d) is declared, unnamed and not in reservedRole -- Valid() refuses it, "+
				"so jlm.write and jlm.codec would reject any container carrying one. Name it or "+
				"reserve it by name.", m[1], code)
		}
	}
	t.Logf("%d roles declared, %d reserved", len(ms), unnamed)
}
