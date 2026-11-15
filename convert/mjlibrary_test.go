package convert

import "testing"

func libraryFile(t *testing.T, class string) *modelJSON {
	t.Helper()
	lib, err := modelLibrary()
	if err != nil {
		t.Fatal(err)
	}
	mj := lib[class]
	if mj == nil {
		t.Fatalf("no %s in the library", class)
	}
	return mj
}

func iota32(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(i)
	}
	return v
}
