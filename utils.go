package learn

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func addBias(X *mat.Dense) *mat.Dense {
	m, n := X.Dims()
	data := X.RawMatrix().Data
	dataLen := len(data)
	newDataLen := dataLen + m
	newData := make([]float64, newDataLen)
	j := 0
	newCols := n + 1
	for i := 0; i < newDataLen; i++ {
		if (i % newCols) == 0 {
			newData[i] = 1
		} else {
			newData[i] = data[j]
			j++
		}
	}
	return mat.NewDense(m, newCols, newData)
}

func vecToSlice(v *mat.VecDense) []float64 {
	l := v.Len()
	data := make([]float64, l)
	for i := 0; i < l; i++ {
		data[i] = v.AtVec(i)
	}
	return data
}
