package main

import "testing"

func TestCSVFields(t *testing.T) {
	got := csvFields(" 0, NVIDIA Test, 0000:01:00.0 ", 3)
	if got == nil || got[1] != "NVIDIA Test" {
		t.Fatalf("unexpected fields: %#v", got)
	}
}

func TestValidationBoundaries(t *testing.T) {
	if abs(-500) != 500 || abs(0) != 0 {
		t.Fatal("absolute value is wrong")
	}
	if n, err := nonNegativeInt("0"); err != nil || n != 0 {
		t.Fatal("zero GPU index must be valid")
	}
	if _, err := nonNegativeInt("-1"); err == nil {
		t.Fatal("negative GPU index must fail")
	}
}

func TestPCIBusNumber(t *testing.T) {
	got, ok := pciBusNumber("00000000:2d:00.0")
	if !ok || got != 0x2d {
		t.Fatalf("got %d, %v", got, ok)
	}
}
