package main

import (
	"bytes"
	"debug/pe"
	"encoding/binary"
	"image/png"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestSysoIsAnObjectTheLinkerReads opens the object with debug/pe, as the
// linker's loader does: one .rsrc section, a relocation per resource, and the
// three resource types at its root.
func TestSysoIsAnObjectTheLinkerReads(t *testing.T) {
	for _, arch := range []string{"amd64", "arm64"} {
		b, err := syso(arch, "1.2.3-next")
		if err != nil {
			t.Fatal(err)
		}
		f, err := pe.NewFile(bytes.NewReader(b))
		if err != nil {
			t.Fatalf("%s: %v", arch, err)
		}
		s := f.Section(".rsrc")
		if s == nil {
			t.Fatalf("%s: no .rsrc section", arch)
		}
		if want := len(icoSizes()) + 2; len(s.Relocs) != want {
			t.Errorf("%s: %d relocations, want %d", arch, len(s.Relocs), want)
		}
		types, err := resourceTypes(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, want := range []uint32{rtIcon, rtGroupIcon, rtVersion} {
			if !types[want] {
				t.Errorf("%s: resource type %d missing: %v", arch, want, types)
			}
		}
	}
	if _, err := syso("386", "1.0.0"); err == nil {
		t.Error("an unsupported GOARCH was accepted")
	}
}

func TestQuad(t *testing.T) {
	for in, want := range map[string][4]uint16{
		"1.2.3": {1, 2, 3, 0}, "v0.9.1-rc1": {0, 9, 1, 0}, "1.2.3.4000": {1, 2, 3, 4000},
	} {
		got, err := quad(in)
		if err != nil || got != want {
			t.Errorf("quad(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
	for _, bad := range []string{"1.2.3.4.5", "x.y", "70000.0.0"} {
		if _, err := quad(bad); err == nil {
			t.Errorf("quad(%q) accepted", bad)
		}
	}
}

// TestIcnsElementsAreTheirSize walks the .icns as macOS reads it: every
// element's length lands on the next, and each PNG is the size its type says.
func TestIcnsElementsAreTheirSize(t *testing.T) {
	b, err := icns()
	if err != nil {
		t.Fatal(err)
	}
	if string(b[:4]) != "icns" || int(binary.BigEndian.Uint32(b[4:])) != len(b) {
		t.Fatalf("header %q %d, file %d bytes", b[:4], binary.BigEndian.Uint32(b[4:]), len(b))
	}
	at, n := 8, 0
	for at < len(b) {
		tag, l := string(b[at:at+4]), int(binary.BigEndian.Uint32(b[at+4:]))
		img, err := png.DecodeConfig(bytes.NewReader(b[at+8 : at+l]))
		if err != nil {
			t.Fatalf("%s: %v", tag, err)
		}
		if want := icnsTypes()[n].size; img.Width != want || img.Height != want || tag != icnsTypes()[n].tag {
			t.Errorf("%s is %dx%d, want %s at %d", tag, img.Width, img.Height, icnsTypes()[n].tag, want)
		}
		at += l
		n++
	}
	if at != len(b) || n != len(icnsTypes()) {
		t.Errorf("walked %d of %d bytes, %d of %d elements", at, len(b), n, len(icnsTypes()))
	}
}

// TestIcoOffsetsLandOnTheirImages reads the .ico as Windows does: every
// entry's offset and length frame a PNG of the entry's size, and the images
// tile the file after the directory with nothing left over.
func TestIcoOffsetsLandOnTheirImages(t *testing.T) {
	b, err := ico()
	if err != nil {
		t.Fatal(err)
	}
	if r, typ, n := binary.LittleEndian.Uint16(b), binary.LittleEndian.Uint16(b[2:]), int(binary.LittleEndian.Uint16(b[4:])); r != 0 || typ != 1 || n != len(icoSizes()) {
		t.Fatalf("ICONDIR %d %d %d", r, typ, n)
	}
	end := 6 + 16*len(icoSizes())
	for i, s := range icoSizes() {
		e := b[6+16*i:]
		l, off := int(binary.LittleEndian.Uint32(e[8:])), int(binary.LittleEndian.Uint32(e[12:]))
		if off != end {
			t.Fatalf("image %d at %d, want %d", i, off, end)
		}
		img, err := png.DecodeConfig(bytes.NewReader(b[off : off+l]))
		if err != nil {
			t.Fatalf("image %d: %v", i, err)
		}
		if img.Width != s || int(e[0]) != s%256 {
			t.Errorf("image %d is %d wide, entry says %d, want %d", i, img.Width, e[0], s)
		}
		end = off + l
	}
	if end != len(b) {
		t.Errorf("images end at %d of %d bytes", end, len(b))
	}
}

// TestBundleCarriesTheCLIs builds a bundle around stand-in binaries and
// finds each program where the app and the cask look for it. On macOS the
// bundle would be signed, which a shell-script stand-in does not survive; the
// release workflow builds the real bundle there.
func TestBundleCarriesTheCLIs(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("codesign signs what it bundles; see the release workflow's macos-app job")
	}
	dir := t.TempDir()
	var bins []string
	for _, n := range []string{"jitllm-desktop", "jitllm", "jitllmd"} {
		p := filepath.Join(dir, n)
		if err := os.WriteFile(p, []byte("#!/bin/sh\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		bins = append(bins, p)
	}
	app := filepath.Join(dir, "jitllm.app")
	if err := bundle(bins[0], bins[1:], "1.2.3", app, ""); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{"MacOS/jitllm-desktop", "Resources/bin/jitllm", "Resources/bin/jitllmd", "Resources/jitllm.icns", "Info.plist"} {
		if _, err := os.Stat(filepath.Join(app, "Contents", p)); err != nil {
			t.Error(err)
		}
	}
	if runtime.GOOS == "windows" {
		return // no execute bits to read; the bundle is checked on macOS and Linux
	}
	for _, p := range []string{"MacOS/jitllm-desktop", "Resources/bin/jitllm", "Resources/bin/jitllmd"} {
		if fi, err := os.Stat(filepath.Join(app, "Contents", p)); err == nil && fi.Mode()&0o111 == 0 {
			t.Errorf("%s is not executable", p)
		}
	}
}
