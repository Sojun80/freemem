.PHONY: build test

build:
	go build -o gpuoc ./cmd/gpuoc
	mkdir -p build
	go build -o build/gpuoc-tui-go-preview ./cmd/gpuoc-tui

test:
	go test ./...
