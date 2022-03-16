# Robust Linear Regression

There are some `.py` files that are named as `cluster_version_*.py` and those are the ones parallelized and stripped down of the progress bars using thus minimal dependancies.

To run optimal lambda simulations the files to run are the ones `cluster_version_huber_lambda_opt.py` and `cluster_version_L2_lambda_opt.py`. To run them one should use 3 system arguments as follows:

```
python cluster_version_*_lambda_opt epsilon delta_small delta_large
```

this should do the job.