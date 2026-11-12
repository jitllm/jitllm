package model

import (
	"crypto/sha256"
	"image"
	"slices"
	"testing"
)

// TestPictureKeyCoversTheLayout: a picture's rows are named by everything the
// tower reads of it, its layout as well as its pixels, so one pixel buffer
// laid out 16x4 and 4x16 names two pictures, as do two pictures a byte apart,
// and one picture named twice names one. The violation: a name over the
// pixel bytes alone aliases the two layouts.
func TestPictureKeyCoversTheLayout(t *testing.T) {
	pix := make([]byte, 16*4*4)
	for i := range pix {
		pix[i] = byte(i*7 + 3)
	}
	wide := &image.NRGBA{Pix: pix, Stride: 16 * 4, Rect: image.Rect(0, 0, 16, 4)}
	tall := &image.NRGBA{Pix: slices.Clone(pix), Stride: 4 * 4, Rect: image.Rect(0, 0, 4, 16)}
	apart := &image.NRGBA{Pix: slices.Clone(pix), Stride: 16 * 4, Rect: image.Rect(0, 0, 16, 4)}
	apart.Pix[5] ^= 1
	key := func(img image.Image) RowsKey {
		t.Helper()
		k, err := pictureKey(img)
		if err != nil {
			t.Fatal(err)
		}
		return *k
	}
	if key(wide) != key(wide) {
		t.Fatal("one picture named twice names two")
	}
	if key(wide) == key(tall) {
		t.Fatal("one pixel buffer laid out 16x4 and 4x16 names one picture")
	}
	if key(apart) == key(wide) {
		t.Fatal("two pictures a byte apart name one")
	}
	planes := func(img image.Image) [32]byte {
		var ent [4 * 256]byte
		_, ps, err := picturePlanes(img, &ent)
		if err != nil {
			t.Fatal(err)
		}
		hs := sha256.New()
		for _, p := range ps {
			hs.Write(p)
		}
		var s [32]byte
		hs.Sum(s[:0])
		return s
	}
	if planes(wide) != planes(tall) {
		t.Fatal("VIOLATION: the two layouts' pixel bytes differ, so this gate cannot see a name over the pixels alone")
	}
}
