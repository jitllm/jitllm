package library

import "testing"

func TestTheCatalogueReads(t *testing.T) {
	ms, err := catalogue()
	if err != nil {
		t.Fatal(err)
	}
	if len(ms) < 80 {
		t.Fatalf("read %d models", len(ms))
	}
	if _, err := catalogueFS.ReadFile("models.json"); err != nil {
		t.Fatal(err)
	}
	if len(Models()) != len(ms) {
		t.Fatalf("Models has %d of the file's %d", len(Models()), len(ms))
	}
}
