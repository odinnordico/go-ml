package learn

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// featureNormalizationScaling updates the X matrix of samples
// applying feature scaling technique with formula x' = ( x - min(X) ) / ( max(X) - min(X) )
// where
//  - x' is the normalized value of the feature
//  - x is the actual value of the feaure
//  - min(X) is the minimun value of the feature
//  - max(X) is the maximun value of the feature
// return X normalized, max, min
func featureNormalizationScaling(X *mat.Dense) (*mat.Dense, []float64, []float64) {
	cols := X.RawMatrix().Cols
	Xnorm := mat.NewDense(X.RawMatrix().Rows, cols, nil)
	max := make([]float64, cols)
	min := make([]float64, cols)
	for c := 0; c < cols; c++ {
		var colVec mat.VecDense
		colVec.CloneFromVec(X.ColView(c))
		colData := colVec.RawVector().Data
		colMin := floats.Min(colData)
		colMax := floats.Max(colData)
		colDeviation := colMax - colMin
		for r, val := range colData {
			colData[r] = (val - colMin) / colDeviation
		}
		max[c] = colMax
		min[c] = colMin
		Xnorm.SetCol(c, colData)
	}
	return Xnorm, max, min
}

// featureNormalizationMeanWithMaxMin updates the X matrix of samples
// applying mean normalization technique with formula x' = ( x - μ  ) / s
// where
//  - x' is the normalized value of the feature
//  - x is the actual value of the feaure
//  - μ is the average of the feature values
//  - s is the deviation or maximun value of the feature minus minimu value of the feature of the feature values
// return X normalized
func featureNormalizationMeanWithMaxMin(X *mat.Dense, max []float64, min []float64) *mat.Dense {
	cols := X.RawMatrix().Cols
	Xnorm := mat.NewDense(X.RawMatrix().Rows, cols, nil)
	for c := 0; c < cols; c++ {
		var colVec mat.VecDense
		colVec.CloneFromVec(X.ColView(c))
		colData := colVec.RawVector().Data
		colMin := min[c]
		colMax := max[c]
		colDeviation := colMax - colMin
		for r, val := range colData {
			colData[r] = (val - colMin) / colDeviation
		}
		max[c] = colMax
		min[c] = colMin
		Xnorm.SetCol(c, colData)
	}
	return Xnorm
}

// featureNormalizationMean updates the X matrix of samples
// applying mean normalization technique with formula x' = ( x - μ  ) / s
// where
//  - x' is the normalized value of the feature
//  - x is the actual value of the feaure
//  - μ is the average of the feature values
//  - s is the deviation or maximun value of the feature minus minimu value of the feature of the feature values
// return X normalized, mu, sigma
func featureNormalizationMean(X *mat.Dense) (*mat.Dense, []float64, []float64) {
	cols := X.RawMatrix().Cols
	Xnorm := mat.NewDense(X.RawMatrix().Rows, cols, nil)
	mu := make([]float64, cols)
	sigma := make([]float64, cols)
	for c := 0; c < cols; c++ {
		var colVec mat.VecDense
		colVec.CloneFromVec(X.ColView(c))
		colData := colVec.RawVector().Data
		colSigma := stat.StdDev(colData, nil)
		colMu := stat.Mean(colData, nil)
		for r, val := range colData {
			colData[r] = (val - colMu) / colSigma
		}
		mu[c] = colMu
		sigma[c] = colSigma
		Xnorm.SetCol(c, colData)
	}
	return Xnorm, mu, sigma
}

// featureNormalizationMeanWithMuSig updates the X matrix of samples
// applying mean normalization technique with formula x' = ( x - μ  ) / s
// where
//  - x' is the normalized value of the feature
//  - x is the actual value of the feaure
//  - μ is the average of the feature values
//  - s is the deviation or maximun value of the feature minus minimu value of the feature of the feature values
// return X normalized
func featureNormalizationMeanWithMuSig(X *mat.Dense, mu []float64, sigma []float64) *mat.Dense {
	cols := X.RawMatrix().Cols
	Xnorm := mat.NewDense(X.RawMatrix().Rows, cols, nil)
	for c := 0; c < cols; c++ {
		var colVec mat.VecDense
		colVec.CloneFromVec(X.ColView(c))
		colData := colVec.RawVector().Data
		colSigma := sigma[c]
		colMu := mu[c]
		for r, val := range colData {
			colData[r] = (val - colMu) / colSigma
		}
		mu[c] = colMu
		sigma[c] = colSigma
		Xnorm.SetCol(c, colData)
	}
	return Xnorm
}
