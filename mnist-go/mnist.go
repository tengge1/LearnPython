package main

import "github.com/petar/GoMNIST"

func main() {
	train, test, err := GoMNIST.Load("./data")
	if err != nil {
		panic(err)
	}
	_, _ = train, test
}
