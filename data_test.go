package learn

func getDefaultXsamplesSlice() []float64 {
	return []float64{
		1, 4.52, 4.725, 4.333, 4.5909,
		1, 2.60, 2.50, 2.345, 2.5434,
		1, -2.34, -2.555, -2.3543, -2.234,
		1, 4.08, 4.001, 4.043, 4.009,
		1, 1.76, 1.79, 1.699, 1.799,
		1, 1.59, 1.5757, 1.5343, 1.598,
		1, 0.63, 0.63, 0.63, 0.63,
		1, -0.29, -0.212, -0.298, -0.233,
		1, -1.41, -1.77, -1.67, -1.456,
		1, -2.78, -2.798, -2.734, -2.797,
		1, -0.78, -0.777, -0.758, -0.799,
		1, -1.28, -1.234, -1.3, -1.4,
		1, 0.59, 0.577, 0.59, 0.51,
		1, 0.40, 0.49, 0.5, 0.3,
		1, -0.28, -0.287, -0.223, -0.266,
		1, 1.95, 1.8, 1.999, 1.9001,
	}
}

func getDefaultXsamplesNoBiasSlice() []float64 {
	return []float64{
		4.52, 4.725, 4.333, 4.5909,
		2.60, 2.50, 2.345, 2.5434,
		-2.34, -2.555, -2.3543, -2.234,
		4.08, 4.001, 4.043, 4.009,
		1.76, 1.79, 1.699, 1.799,
		1.59, 1.5757, 1.5343, 1.598,
		0.63, 0.63, 0.63, 0.63,
		-0.29, -0.212, -0.298, -0.233,
		-1.41, -1.77, -1.67, -1.456,
		-2.78, -2.798, -2.734, -2.797,
		-0.78, -0.777, -0.758, -0.799,
		-1.28, -1.234, -1.3, -1.4,
		0.59, 0.577, 0.59, 0.51,
		0.40, 0.49, 0.5, 0.3,
		-0.28, -0.287, -0.223, -0.266,
		1.95, 1.8, 1.999, 1.9001,
	}
}

func getDefaultTheta() []float64 {
	return []float64{3.117, -2.518, 7.184, -6.538, 2.027}
}

func getDefaultYLabels() []float64 {
	return []float64{
		6.15,
		4.34,
		0.97,
		0.97,
		6.32,
		4.98,
		3.44,
		3.38,
		2.30,
		2.00,
		0.87,
		3.40,
		2.18,
		4.21,
		2.35,
		3.32,
	}
}

func getTwoVariableDataForLinear() []float64 {
	return []float64{
		2104, 3, 399900,
		1600, 3, 329900,
		2400, 3, 369000,
		1416, 2, 232000,
		3000, 4, 539900,
		1985, 4, 299900,
		1534, 3, 314900,
		1427, 3, 198999,
		1380, 3, 212000,
		1494, 3, 242500,
		1940, 4, 239999,
		2000, 3, 347000,
		1890, 3, 329999,
		4478, 5, 699900,
		1268, 3, 259900,
		2300, 4, 449900,
		1320, 2, 299900,
		1236, 3, 199900,
		2609, 4, 499998,
		3031, 4, 599000,
		1767, 3, 252900,
		1888, 2, 255000,
		1604, 3, 242900,
		1962, 4, 259900,
		3890, 3, 573900,
		1100, 3, 249900,
		1458, 3, 464500,
		2526, 3, 469000,
		2200, 3, 475000,
		2637, 3, 299900,
		1839, 2, 349900,
		1000, 1, 169900,
		2040, 4, 314900,
		3137, 3, 579900,
		1811, 4, 285900,
		1437, 3, 249900,
		1239, 3, 229900,
		2132, 4, 345000,
		4215, 4, 549000,
		2162, 4, 287000,
		1664, 2, 368500,
		2238, 3, 329900,
		2567, 4, 314000,
		1200, 3, 299000,
		852, 2, 179900,
		1852, 4, 299900,
		1203, 3, 239500,
	}
}