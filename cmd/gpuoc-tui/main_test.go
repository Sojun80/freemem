package main

import "testing"

func TestMemoryLockChoices(t *testing.T) {
	if got := nextMemLock("RTX 5070", 0, 1); got != 810 {
		t.Fatalf("50-series: got %d", got)
	}
	if got := nextMemLock("RTX 4090", 810, 1); got != 5000 {
		t.Fatalf("40-series: got %d", got)
	}
}

func TestThemePreferenceDefaults(t *testing.T) {
	if loadPreferences().Theme != matrix && loadPreferences().Theme != amber {
		t.Fatal("invalid default theme")
	}
}
