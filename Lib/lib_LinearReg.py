
def LinearReg(train_input, test_input, train_target, test_target, degree):

    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly.fit(train_input)
    train_poly = poly.transform(train_input)
    print('PolyShape: '+ str(train_poly.shape))
    poly.get_feature_names()
    test_poly = poly.transform(test_input)


    from sklearn.linear_model import LinearRegression
    lr = LinearRegression
    lr.fit(train_poly, train_target)

    pred = lr.predict(test_poly)
    print('LinearReg Train Score: %.4f' % lr.score(train_poly, train_target))
    print('LinearReg Test Score: %.4f' % lr.score(test_poly, test_target))

    return lr, pred



