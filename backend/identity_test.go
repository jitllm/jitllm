package backend

import "testing"

// The two rules a device identity has to obey, both about saying nothing. An
// identity merges two enumerated devices into one, so a wrong "yes" deletes
// hardware, while a wrong "no" only double-counts memory. The rules keep every
// uncertain case on the recoverable side.

// TestAnAllZeroUUIDIsNotAnIdentity: a driver that returns success and writes
// nothing hands back sixteen zero bytes, which would compare equal between two
// unrelated devices. The guard cannot live in the backends' error handling.
//
// Violation signature: delete the zero check in UUIDIdentity and this reads
//
//	an all-zero UUID produced the key "uuid:00000000000000000000000000000000",
//	which every device that could not be asked would also produce
func TestAnAllZeroUUIDIsNotAnIdentity(t *testing.T) {
	zero := make([]byte, 16)
	id := UUIDIdentity(zero, true, "a driver that said yes and wrote nothing")
	if id.Known() {
		t.Fatalf("an all-zero UUID produced the key %q, which every device that could not be "+
			"asked would also produce", id.Key)
	}
	if id.Source == "" {
		t.Error("the refusal does not say why there is no identity")
	}
	// Two of them must not be equal to each other either, which is the failure
	// the key above would have caused.
	other := UUIDIdentity(make([]byte, 16), true, "another such driver")
	if id.Same(other) || other.Same(id) {
		t.Fatal("two all-zero UUIDs compared EQUAL: two unrelated devices would merge")
	}

	// A non-zero UUID is an identity, and it is the hex of the bytes: a gate
	// on a host reads the same string off two different backends.
	card := []byte{0x5f, 0x3c, 0x1a, 0x2e, 0x9b, 0x7d, 0x4e, 0x6f,
		0x8a, 0x0b, 0x1c, 0x2d, 0x3e, 0x4f, 0x5a, 0x6b}
	got := UUIDIdentity(card, true, "cuDeviceGetUuid_v2")
	if got.Key != "uuid:5f3c1a2e9b7d4e6f8a0b1c2d3e4f5a6b" {
		t.Fatalf("key %q, want the hex of the sixteen bytes", got.Key)
	}
	if !got.Same(UUIDIdentity(card, true, "VkPhysicalDeviceIDProperties.deviceUUID")) {
		t.Fatal("the same sixteen bytes from two backends did not match, which is the whole " +
			"mechanism")
	}
}

// TestAnUnknownIdentityMatchesNothing.
//
// Violation signature: make Same `i.Key == o.Key` without the Known() guard and
// this reads
//
//	two devices that could not be asked compared EQUAL: every unidentifiable
//	device on a host would collapse into one
func TestAnUnknownIdentityMatchesNothing(t *testing.T) {
	a := UUIDIdentity(nil, false, "this backend reports no device identity")
	b := UUIDIdentity(nil, false, "nor does this one")
	if a.Known() || b.Known() {
		t.Fatal("an absent UUID produced a usable key")
	}
	if a.Same(b) || b.Same(a) || a.Same(a) {
		t.Fatal("two devices that could not be asked compared EQUAL: every unidentifiable " +
			"device on a host would collapse into one")
	}
	known := UUIDIdentity([]byte{1}, true, "a backend that answers")
	if a.Same(known) || known.Same(a) {
		t.Fatal("an unknown identity matched a known one")
	}
	if s := a.String(); s == "" || s[0] != 'u' {
		t.Errorf("String() = %q, want it to lead with \"unknown\" and carry the reason", s)
	}
}

// TestTheBackendPreferenceIsCUDAThenMetalThenHIPThenVulkan. It orders backends
// for one card and nothing else; see APIRank.
func TestTheBackendPreferenceIsCUDAThenMetalThenHIPThenVulkan(t *testing.T) {
	if !(APIRank("ptx") > APIRank("msl") && APIRank("msl") > APIRank("amdgcn") &&
		APIRank("amdgcn") > APIRank("spirv")) {
		t.Fatalf("rank ptx=%d msl=%d amdgcn=%d spirv=%d, want CUDA over Metal over HIP over Vulkan",
			APIRank("ptx"), APIRank("msl"), APIRank("amdgcn"), APIRank("spirv"))
	}
	// The grammar's own spellings rank the same as the code generators' names,
	// because one vocabulary reaches the API and the CLI.
	for _, p := range [][2]string{{"ptx", "cuda"}, {"msl", "metal"}, {"amdgcn", "hip"}, {"spirv", "vulkan"}} {
		if APIRank(p[0]) != APIRank(p[1]) {
			t.Errorf("%q and %q are the same backend and rank %d against %d",
				p[0], p[1], APIRank(p[0]), APIRank(p[1]))
		}
	}
	if APIRank("something new") >= APIRank("spirv") {
		t.Error("an unknown backend outranks a known one: a new backend must not silently " +
			"take a card from the one that was measured")
	}
}
