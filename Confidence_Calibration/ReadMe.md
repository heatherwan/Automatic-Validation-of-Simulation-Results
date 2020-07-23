### 1. Calibration model's tuning, training and evaluation

- Temperature Scaling (TempS):

```python main.py --file_val xxx_val_logit.txt --file_test xxx_test_logit.txt --method 'TS' ```

- Matrix Scaling with Off-diagonal and Intercept regularisation (MS-ODIR):

```python main.py --file_val xxx_val_logit.txt --file_test xxx_test_logit.txt --method 'MS-ODIR' ```

- Dirichlet with Off-diagonal and Intercept regularisation (Dir-ODIR):

```python main.py --file_val xxx_val_logit.txt --file_test xxx_test_logit.txt --method 'DIR-ODIR' ```


### 2. output results

Output calibration score is in `result` under name like exp604_method_result
Output un-calibrated and calibrated probabilities are in `result` under name like exp604_test_0.01_0.01.txt

