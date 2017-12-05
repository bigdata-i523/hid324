mkdir -p data
curl https://www.quandl.com/api/v3/datasets/BCHARTS/COINBASEUSD.csv?api_key=3ny-RLKDcGxjRDxRco8J >>data/Bitcoin_USD_History.csv
curl https://www.quandl.com/api/v3/datasets/BCHARTS/COINBASEGBP.csv?api_key=3ny-RLKDcGxjRDxRco8J >>data/Bitcoin_GBP_History.csv
curl https://www.quandl.com/api/v3/datasets/BCHARTS/COINBASEEUR.csv?api_key=3ny-RLKDcGxjRDxRco8J >>data/Bitcoin_EUR_History.csv
