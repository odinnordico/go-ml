package learn

import (
	"log"
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func Test_linearPredict(t *testing.T) {
	const rows = 16
	const cols = 5
	featuresData := getDefaultXsamplesSlice()
	thetaData := getDefaultTheta()
	resultData := []float64{
		6.656640299999998, 4.3540618, 1.5180953999999947, 3.279853000000001,
		4.083190999999999, 3.6411014000000006, 3.2146500000000002, 3.8002449999999994,
		1.9188479999999983, 2.221580999999997, 2.8353029999999997, 3.1365839999999996,
		2.9528980000000007, 2.9690599999999994, 2.679024, 1.9201407000000001,
	}
	type args struct {
		Xsamples *mat.Dense
		theta    *mat.VecDense
	}
	tests := []struct {
		name    string
		args    args
		want    *mat.VecDense
		wantErr bool
	}{
		{
			"Successful calculation",
			args{
				mat.NewDense(rows, cols, featuresData),
				mat.NewVecDense(cols, thetaData),
			},
			mat.NewVecDense(rows, resultData),
			false,
		},
		{
			"Fails by factures mismatch with theta rows",
			args{
				mat.NewDense(rows, cols, featuresData),
				mat.NewVecDense(rows, nil),
			},
			nil,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := linearPredict(tt.args.Xsamples, tt.args.theta)
			if (err != nil) != tt.wantErr {
				t.Errorf("linearPredict() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("linearPredict() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_linearCostFunction(t *testing.T) {
	const rows = 16
	const cols = 5
	featuresData := getDefaultXsamplesSlice()
	yLabels := getDefaultYLabels()
	thetaData := getDefaultTheta()
	thetaDataWithError := []float64{
		-1.233758177597947, -0.12634751070237293, -0.5209945711531503, 2.28571911769958, 0.3228052526115799,
	}
	type args struct {
		Xsamples *mat.Dense
		y        *mat.VecDense
		theta    *mat.VecDense
	}
	tests := []struct {
		name    string
		args    args
		want    float64
		wantErr bool
	}{
		{
			"Low Error",
			args{
				mat.NewDense(rows, cols, featuresData),
				mat.NewVecDense(rows, yLabels),
				mat.NewVecDense(cols, thetaData),
			},
			0.6639815354658731,
			false,
		},
		{
			"Random Error",
			args{
				mat.NewDense(rows, cols, featuresData),
				mat.NewVecDense(rows, yLabels),
				mat.NewVecDense(cols, thetaDataWithError),
			},
			11.652965500291712,
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := linearCostFunction(tt.args.Xsamples, tt.args.y, tt.args.theta)
			if (err != nil) != tt.wantErr {
				t.Errorf("linearCostFunction() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("linearCostFunction() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_linearGradientDescentIter(t *testing.T) {
	const rows = 16
	const cols = 5
	featuresData := getDefaultXsamplesSlice()
	yLabels := getDefaultYLabels()
	thetaData := getDefaultTheta()
	thetaDataWithError := []float64{
		-1.233758177597947, -0.12634751070237293, -0.5209945711531503, 2.28571911769958, 0.3228052526115799,
	}
	type args struct {
		Xsamples *mat.Dense
		y        *mat.VecDense
		theta    *mat.VecDense
		alpha    float64
	}
	tests := []struct {
		name    string
		args    args
		want    *mat.VecDense
		wantErr bool
	}{
		{
			"Low Error",
			args{
				mat.NewDense(rows, cols, featuresData),
				mat.NewVecDense(rows, yLabels),
				mat.NewVecDense(cols, thetaData),
				0.05,
			},
			mat.NewVecDense(cols, []float64{3.116996010625, -2.518006888815625, 7.18399340567975, -6.5380065598, 2.0269931504748127}),
			false,
		},
		{
			"Random Error",
			args{
				mat.NewDense(rows, cols, featuresData),
				mat.NewVecDense(rows, yLabels),
				mat.NewVecDense(cols, thetaDataWithError),
				0.05,
			},
			mat.NewVecDense(cols, []float64{-1.0631428558040166, -0.3444630059815565, -0.7477349794098651, 2.0639915916875764, 0.10306135778082307}),
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := linearGradientDescentIter(tt.args.Xsamples, tt.args.y, tt.args.theta, tt.args.alpha)
			if (err != nil) != tt.wantErr {
				t.Errorf("linearGradientDescentIter() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("linearGradientDescentIter() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_linearNormalEq(t *testing.T) {
	const rows = 16
	const cols = 5
	featuresData := getDefaultXsamplesSlice()
	yLabels := getDefaultYLabels()
	thetaData := []float64{3.11694451886693, -2.518137542594296, 7.184396423418232, -6.53804297664471, 2.026752707820876}
	type args struct {
		Xsamples *mat.Dense
		y        *mat.VecDense
	}
	tests := []struct {
		name    string
		args    args
		want    *mat.VecDense
		wantErr bool
	}{
		{
			"'X.T() * X' Invertible",
			args{
				mat.NewDense(rows, cols, featuresData),
				mat.NewVecDense(rows, yLabels),
			},
			mat.NewVecDense(cols, thetaData),
			false,
		},
		{
			"'X.T() * X' with m = n",
			args{
				mat.NewDense(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}),
				nil,
			},
			nil,
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := linearNormalEq(tt.args.Xsamples, tt.args.y)
			if (err != nil) != tt.wantErr {
				t.Errorf("linearNormalEq() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("linearNormalEq() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_linearRegresion(t *testing.T) {
	t.Run("Linear Regresion", func(t *testing.T) {

		m := 47
		data := mat.NewDense(m, 3, getTwoVariableDataForLinear())
		var y mat.VecDense
		y.ColViewOf(data, 2)
		// TODO: fix the sub-matrix extraction
		Xsamples := mat.NewDense(m, 2, make([]float64, m*2))
		var first mat.VecDense
		first.ColViewOf(data, 0)
		var second mat.VecDense
		second.ColViewOf(data, 1)

		Xsamples.SetCol(0, vecToSlice(&first))
		Xsamples.SetCol(1, vecToSlice(&second))
		// Normalizing
		Xnorm, mu, sigma := featureNormalizationMean(Xsamples)
		// Xnorm, max, min := featureNormalizationScaling(Xsamples)
		X := addBias(Xnorm)
		// Running Gradient decent
		alpha := 0.01
		numIters := 5000
		// Initial theta with zeros
		theta := mat.NewVecDense(3, make([]float64, 3))
		var err error = nil
		// matPrint(X)
		var costHist []float64
		theta, costHist, err = linearGradientDescent(X, &y, theta, alpha, numIters)
		if nil != err {
			t.Error(err)
			return
		}
		log.Printf("ERROR HISTORY")
		for i, val := range costHist {
			log.Printf(", %v, %v", i+1, val)
		}
		log.Printf("")

		log.Printf("Theta computed from gradient descent:\n %v\n", theta)
		dataToPRedict := addBias(featureNormalizationMeanWithMuSig(mat.NewDense(1, 2, []float64{1650.0, 3.0}), mu, sigma))
		// dataToPRedict := addBias(featureNormalizationMeanWithMaxMin(mat.NewDense(1, 2, []float64{1650.0, 3.0}), max, min))

		var price *mat.VecDense
		price, err = linearPredict(dataToPRedict, theta)
		if nil != err {
			t.Error(err)
			return
		}
		theta = nil
		log.Printf("Predicted of 1650 and 3, (using gradient descent):\n %v\n", price)
		log.Print("\n\n\nSolving with normal equations...\n\n")
		X.SetCol(1, vecToSlice(&first))
		X.SetCol(2, vecToSlice(&second))
		theta, err = linearNormalEq(X, &y)
		if nil != err {
			t.Error(err)
			return
		}
		log.Printf("Theta computed from normal equation:\n %v\n", theta)
		dataToPRedict = addBias(mat.NewDense(1, 2, []float64{1650.0, 3.0}))
		price, err = linearPredict(dataToPRedict, theta)
		log.Printf("Predicted of 1650 and 3, (using normal equations):\n %v\n", price)
	})
}
