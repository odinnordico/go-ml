package learn

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func Test_matPrint(t *testing.T) {
	type args struct {
		X mat.Matrix
	}
	tests := []struct {
		name string
		args args
	}{
		{
			"Just print a matrix",
			args{mat.NewDense(16, 5, getDefaultXsamplesSlice())},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			matPrint(tt.args.X)
		})
	}
}

func Test_addBias(t *testing.T) {
	type args struct {
		X *mat.Dense
	}
	tests := []struct {
		name string
		args args
		want *mat.Dense
	}{
		{
			"Bias added",
			args{mat.NewDense(16, 4, getDefaultXsamplesNoBiasSlice())},
			mat.NewDense(16, 5, getDefaultXsamplesSlice()),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := addBias(tt.args.X); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("addBias() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_deleteme(t *testing.T) {
	t.Run("Deleteme", func(t *testing.T) {
		tmpOrg := mat.NewDense(16, 4, getDefaultXsamplesNoBiasSlice())
		matPrint(tmpOrg)
		println("  --  ")

		m, n := tmpOrg.Dims()
		data := tmpOrg.RawMatrix().Data
		dataLen := len(data)
		newDataLen := dataLen + m
		newData := make([]float64, newDataLen)
		println(n)
		println(newDataLen)
		j := 0
		newCols := n + 1
		for i := 0; i < newDataLen; i++ {
			if (i % newCols) == 0 {
				newData[i] = 1
				newData[i+1] = data[j]

			} else {
				newData[i] = data[j]
				j++
			}
			// fmt.Printf("%v\n", newData)
		}
		matPrint(mat.NewDense(16, 5, newData))
	})
}
