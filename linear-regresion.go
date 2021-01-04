package learn

import (
	"fmt"
	"log"

	"gonum.org/v1/gonum/mat"
)

const (
	errMsg                string = "Columns on Features (%v) must match with leng of theta (%v)"
	errMsgFeaturesSamples string = "The number of training examples (%v) must be greater than the number of features (%v)"
	errMsgLablesLength    string = "The number of training examples (%v) must be the same as the number of labes (%v)"
)

// linearPredict predict the linear value using linear regresion y = b + mx where
// 'X': Matrix of samples to predict
// 'theta': theta values used to predict
// return the coresponded vector of predictions
// throw error if 'X' columns and 'theta' lengt are not the same
func linearPredict(X *mat.Dense, theta *mat.VecDense) (*mat.VecDense, error) {
	n := X.RawMatrix().Cols
	yLen := theta.Len()
	if yLen != n {
		log.Printf(errMsg, n, yLen)
		return nil, fmt.Errorf(errMsg, n, yLen)
	}

	var predicition mat.VecDense
	predicition.MulVec(X, theta)
	return &predicition, nil
}

// linearCostFunction calculates the cost using Mean Square Error
// 'X': Matrix of samples to predict
// 'y' Vector of actual results
// 'theta': theta values used to predict
// return the cost value for given parameters
func linearCostFunction(X *mat.Dense, y *mat.VecDense, theta *mat.VecDense) (float64, error) {
	m, _ := X.Dims()
	predicition, err := linearPredict(X, theta)
	if err != nil {
		log.Fatal(err)
		return 0, err
	}
	ratio := (1.0 / (2.0 * float64(m)))
	var costErr mat.VecDense
	costErr.SubVec(predicition, y)
	costErrT := costErr.T()
	// Mean Square Error
	var mse mat.VecDense
	mse.MulVec(costErrT, &costErr)
	return ratio * mse.At(0, 0), nil
}

// linearGradientDescentIter calculates theta with the Gradient Descent of a given samples and theta
// using Mean Square Error
// 'X': Matrix of samples to predict
// 'y': Vector of actual results or sample label
// 'theta': theta values used to predict
// 'alpha': learnig rate
// return the new 'theta' values
func linearGradientDescentIter(X *mat.Dense, y *mat.VecDense, theta *mat.VecDense, alpha float64) (*mat.VecDense, error) {
	m := y.Len()
	predicition, err := linearPredict(X, theta)
	if err != nil {
		log.Fatal(err)
		return nil, err
	}
	var costErr mat.VecDense
	costErr.SubVec(predicition, y)
	XT := X.T()
	var costErrAndT mat.VecDense
	costErrAndT.MulVec(XT, &costErr)
	rate := alpha / float64(m)
	var rateErr mat.VecDense
	rateErr.ScaleVec(rate, &costErrAndT)
	theta.SubVec(theta, &rateErr)
	return theta, nil
}

// linearGradientDescen calculates theta with the Gradient Descent of a given samples and theta
// using Mean Square Error
// 'X': Matrix of samples to predict
// 'y': Vector of actual results or sample label
// 'theta': theta values used to predict
// 'alpha': learnig rate
// 'iterations': amount of iterations to run gradient decent
// return the new 'theta' values
func linearGradientDescent(X *mat.Dense, y *mat.VecDense, theta *mat.VecDense, alpha float64, iterations int) (*mat.VecDense, []float64, error) {
	baseTheta := theta
	var err error = nil
	costHistory := make([]float64, iterations)
	for i := 0; i < iterations; i++ {
		baseTheta, err = linearGradientDescentIter(X, y, baseTheta, alpha)
		if nil != err {
			return nil, nil, err
		}
		cost, _ := linearCostFunction(X, y, baseTheta)
		costHistory[i] = cost
	}
	return baseTheta, costHistory, nil
}

// linearNormalEq calculates theta using normal equation
// 'feature scaling' not required
// when 'X.T() * X' is noninvertible check:
//  - Two or more features are redundant features or linear dependent
//  - There are more features than trainig examples so 'regularizarion' is required by deleting some features
// 'X': Matrix of samples to predict
// 'y': Vector of actual results or sample label
// return the new 'theta' values
func linearNormalEq(X *mat.Dense, y *mat.VecDense) (*mat.VecDense, error) {
	m, n := X.Dims()
	if m <= n {
		log.Printf(errMsgFeaturesSamples, m, n)
		return nil, fmt.Errorf(errMsgFeaturesSamples, m, n)
	}
	var xxt mat.Dense
	xxt.Mul(X.T(), X)
	var xxtInv mat.Dense
	xxtInv.Inverse(&xxt)
	var xxtit mat.Dense
	xxtit.Mul(&xxtInv, X.T())
	var theta mat.VecDense
	theta.MulVec(&xxtit, y)
	return &theta, nil
}
