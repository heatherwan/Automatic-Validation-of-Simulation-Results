# Functions for Dirichlet parameter tuning main class for CIFAR-100

import numpy as np
import argparse
from calibration.calibration_functions import tune_dir_nn_heather, cal_TS_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_val', type=str, default='exp000', help='input val logit name')
    parser.add_argument('--file_test', type=str, default='exp000', help='input val logit name')
    parser.add_argument('--method', type=str, default='TS', help='calibration method: MS-ODIR, DIR-ODIR, TS')
    args = parser.parse_args()
    files = [f'Input_logit/{args.file_val}', f'Input_logit/{args.file_test}']
    name = args.file_val.split('_')[0]
    print(name)
    if args.method == "TS":
        df_guo = cal_TS_results(name, args.method, files, approach="all")
        df_guo.to_csv(f'result/{name}_{args.method}_result')
    else:
        if 'DIR-ODIR' in name:
            use_logits = True
        else:
            use_logits = False

        # set parameters
        model_dir = 'model_weights'
        loss_fn = 'sparse_categorical_crossentropy'
        k_folds = 5
        random_state = 15
        use_scipy = False
        comp_l2 = True
        double = True

        # Set regularisation parameters to check through
        lambdas = np.array([10 ** i for i in np.arange(-2.0, -1.5)])
        lambdas = sorted(np.concatenate([lambdas, lambdas * 0.25, lambdas * 0.5]))
        mus = np.array([10 ** i for i in np.arange(-2.0, -1.5)])

        # print out parameters
        print("Lambdas:", len(lambdas))
        print("Mus:", str(mus))
        print("Double learning:", double)
        print("Complementary L2:", comp_l2)
        print("Using logits for Dirichlet:", use_logits)
        print("Using Scipy model instead of Keras:", use_scipy)

        df_res, df_res_ensemble = tune_dir_nn_heather(name, args.method, files, lambdas=lambdas, mus=mus, verbose=False,
                                                      k_folds=k_folds,
                                                      random_state=random_state, double_learning=double,
                                                      model_dir=model_dir,
                                                      loss_fn=loss_fn, comp_l2=comp_l2,
                                                      use_logits=use_logits,
                                                      use_scipy=use_scipy)

        df_res.to_csv(f'result/{name}_{args.method}_result')
