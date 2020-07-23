import datetime
import gc
import os
import pickle
import time
from os.path import join

import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.model_selection import KFold

from calibration.cal_methods import Dirichlet_NN, softmax, LogisticCalibration, TemperatureScaling
from calibration.evaluation import evaluate_rip, evaluate


def kf_model(input_val, y_val, fn, fn_kwargs={}, k_folds=5, random_state=15, verbose=False):
    """
    K-fold task, mean and std of results are calculated over K folds
    
    Params:    
        input_val: (np.array) 2-D array holding instances (features) of validation set.
        y_val: (np.array) 1-D array holding y-values for validation set.
        fn: (class) a method used for calibration
        l2: (float) L2 regulariation value.
        k_folds: (int) how many crossvalidation folds are used.
        comp_l2: (bool) use reversed L2 matrix for regulariation (default = False)
    
    returns: 
        mean_error, mean_ece, mean_mce, mean_loss, mean_brier, std_loss, std_brier
    """

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    kf_results = []
    models = []

    for train_index, test_index in kf.split(input_val):
        X_train_c, X_val_c = input_val[train_index], input_val[test_index]
        y_train_c, y_val_c = y_val[train_index], y_val[test_index]

        t1 = time.time()

        model = fn(**fn_kwargs)
        model.fit(X_train_c, y_train_c)
        print("Model trained:", time.time() - t1)

        probs_holdout = model.predict_proba(X_val_c)
        error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier = evaluate_rip(probs_holdout,
                                                                                                      y_val_c,
                                                                                                      verbose=False)
        kf_results.append([error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier])
        models.append(model)

    return models, (np.mean(kf_results, axis=0))


def one_model(input_val, y_val, fn, fn_kwargs={}, k_folds=1, random_state=15, verbose=False):
    """
    1-fold task, mean and std of results are calculated over 1 folds
    
    Params:    
        input_val: (np.array) 2-D array holding instances (features) of validation set.
        y_val: (np.array) 1-D array holding y-values for validation set.
        fn: (class) a method used for calibration
        l2: (float) L2 regulariation value.
        k_folds: (int) how many crossvalidation folds are used.
        comp_l2: (bool) use reversed L2 matrix for regulariation (default = False)
    
    returns: 
        mean_error, mean_ece, mean_mce, mean_loss, mean_brier, std_loss, std_brier
    """

    kf_results = []
    models = []

    t1 = time.time()

    model = fn(**fn_kwargs)
    model.fit(input_val, y_val)

    print("Model trained:", time.time() - t1)

    probs_holdout = model.predict_proba(input_val)
    error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier = evaluate_rip(probs_holdout, y_val,
                                                                                                  verbose=False)
    kf_results.append([error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier])
    models.append(model)

    return models, ((np.mean(kf_results, axis=0)), (np.std(np.array(kf_results)[:, -2:], axis=0)))


def get_cal_prob(models, probs):
    all_pred = []
    for mod in models:
        all_pred.append(mod.predict_proba(probs))
    average_pred = np.mean(all_pred, axis=0)
    return average_pred


def get_test_scores(models, probs, true):
    scores = []

    for mod in models:
        preds = mod.predict_proba(probs)
        scores.append(evaluate_rip(probs=preds, y_true=true, verbose=False))

    return np.mean(scores, axis=0)


def get_test_scores2(models, probs, true):
    preds = []

    for mod in models:
        preds.append(mod.predict_proba(probs))

    return evaluate_rip(np.mean(preds, axis=0), y_true=true, verbose=False)


def tune_dir_nn_heather(name, method, files, lambdas, mus, k_folds=5, random_state=15, verbose=True,
                        double_learning=False,
                        model_dir="models_dump", loss_fn="sparse_categorical_crossentropy",
                        comp_l2=True, use_logits=False, use_scipy=False):
    """

    Params:
        fn (class): class of the calibration method used. It must contain methods "fit" and "predict",
                    where first fits the models and second outputs calibrated probabilities.
        path (string): path to the folder with logits files
        files (list of strings): pickled logits files ((logits_val, y_val), (logits_test, y_test))
        comp_l2 (bool): use reversed L2 matrix for regulariation (default = False)

    Returns:
        df (pandas.DataFrame): dataframe with calibrated and uncalibrated results for all the input files.

    """
    df_columns = ["Name", "L2", "mu", "Error", "ECE", "ECE2", "ECE_CW", "ECE_CW2", "ECE_FULL", "ECE_FULL2", "MCE",
                  "MCE2", "Loss", "Brier"]

    results = []
    results2 = []

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    t1 = time.time()

    # Read in the data

    val_df = pd.read_csv(files[0], sep='\t')
    test_df = pd.read_csv(files[1], sep='\t')

    logits_val = val_df.iloc[:, 2:6].to_numpy()
    y_val = val_df.iloc[:, 1:2].to_numpy().ravel()

    logits_test = test_df.iloc[:, 2:6].to_numpy()
    y_test = test_df.iloc[:, 1:2].to_numpy().ravel()

    # Convert into probabilities
    if use_logits:
        input_val = logits_val
        input_test = logits_test
    else:
        input_val = softmax(logits_val)  # Softmax logits
        input_test = softmax(logits_test)

    error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier = evaluate_rip(
        softmax(logits_val), y_val, verbose=False)  # Uncalibrated results
    print(
        "Uncal Val: Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; "
        "loss %f; brier %f" % (
            error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier))
    results.append(
        [(name + 'val_uncal'), error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier])

    error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier = evaluate_rip(
        softmax(logits_test), y_test, verbose=False)  # Uncalibrated results
    print(
        "Uncal Test: Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; "
        "loss %f; brier %f" % (
            error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier))
    results.append(
        [(name + 'test_uncal'), error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier])

    # Loop over lambdas to test
    for l2 in lambdas:
        for mu in mus:
            # Cross-validation
            if mu is None:
                mu = l2

            if use_scipy:
                temp_res = kf_model(input_val, y_val, LogisticCalibration, {"C": np.true_divide(1, l2), },
                                    k_folds=k_folds, random_state=random_state, verbose=verbose)
            else:
                if k_folds > 1:
                    temp_res = kf_model(input_val, y_val, Dirichlet_NN,
                                        {"l2": l2, "mu": mu, "patience": 15, "loss": loss_fn,
                                         "double_fit": double_learning, "comp": comp_l2, "use_logits": use_logits},
                                        k_folds=k_folds, random_state=random_state, verbose=verbose)
                else:
                    temp_res = one_model(input_val, y_val, Dirichlet_NN,
                                         {"l2": l2, "mu": mu, "patience": 15, "loss": loss_fn,
                                          "double_fit": double_learning, "comp": comp_l2, "use_logits": use_logits},
                                         k_folds=k_folds, random_state=random_state, verbose=verbose)

            (models, (avg_error, avg_ece, avg_ece2, avg_ece_cw, avg_ece_cw2, avg_ece_full, avg_ece_full2, avg_mce,
                      avg_mce2, avg_loss, avg_brier)) = temp_res
            results.append(
                [(name + 'val_cal'), l2, mu, avg_error, avg_ece, avg_ece2, avg_ece_cw, avg_ece_cw2, avg_ece_full,
                 avg_ece_full2, avg_mce, avg_mce2, avg_loss, avg_brier])

            # TODO separate function for pickling models and results
            fname = f"model_{method}_{name}_l2={l2}_mu={mu}.p"

            model_weights = []
            for mod in models:
                if not use_scipy:
                    model_weights.append(mod.model.get_weights())
                else:
                    model_weights.append([mod.coef_, mod.intercept_])

            with open(join(model_dir, fname), "wb") as f:
                pickle.dump((model_weights, temp_res[1], (name, l2, mu)), f)

            print(f"L2 = {l2}, Mu= {mu}, Validation Error {avg_error}; "
                  f"ece {avg_ece}; ece2 {avg_ece2}; "
                  f"ece_cw {avg_ece_cw}; ece_cw2 {avg_ece_cw2}; "
                  f"ece_full {avg_ece_full}; ece_full2 {avg_ece_full2}; "
                  f"mce {avg_mce}; mce2 {avg_mce2}; loss {avg_loss}; brier {avg_brier}")

            with open(f'result/{name}_{method}_val_{l2}_{mu}.txt', "wb") as f:
                np.savetxt(f, input_val)
                np.savetxt(f, get_cal_prob(models, input_val))

            with open(f'result/{name}_{method}_test_{l2}_{mu}.txt', "wb") as f2:
                np.savetxt(f2, input_test)
                np.savetxt(f2, get_cal_prob(models, input_test))

            error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier = get_test_scores(models,
                                                                                                             input_test,
                                                                                                             y_test)
            print(f"L2 = {l2}, Mu= {mu}, Test Error {error}; "
                  f"ece {ece}; ece2 {ece2}; "
                  f"ece_cw {ece_cw}; ece_cw2 {ece_cw2}; "
                  f"ece_full {ece_full}; ece_full2 {ece_full2}; "
                  f"mce {mce}; mce2 {mce2}; loss {loss}; brier {brier}")

            results.append(
                [(name + '_cal_test'), l2, mu, error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2,
                 loss, brier])

            print("Ensembled results:")
            error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier = get_test_scores2(
                models, input_test, y_test)
            print(f"L2 = {l2}, Mu= {mu}, Test Error {error}; "
                  f"ece {ece}; ece2 {ece2}; "
                  f"ece_cw {ece_cw}; ece_cw2 {ece_cw2}; "
                  f"ece_full {ece_full}; ece_full2 {ece_full2}; "
                  f"mce {mce}; mce2 {mce2}; loss {loss}; brier {brier}")

            results2.append(
                [name, l2, mu, error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier])

            # Garbage collection, I had some issues with newer version of Keras.
            K.clear_session()
            for mod in models:  # Delete old models and close class
                del mod
            del models
            del temp_res
            K.clear_session()
            gc.collect()

    t2 = time.time()
    print("Time taken:", (t2 - t1), "\n")

    df = pd.DataFrame(results, columns=df_columns)
    df2 = pd.DataFrame(results2, columns=df_columns)

    return df, df2


def cal_TS_results(name, method, files, m_kwargs={}, approach="all"):
    """
    Calibrate models scores, using output from logits files and given function (fn).
    There are implemented to different approaches "all" and "1-vs-K" for calibration,
    the approach of calibration should match with function used for calibration.

    Params:
        fn (class): class of the calibration method used. It must contain methods "fit" and "predict",
                    where first fits the models and second outputs calibrated probabilities.
        path (string): path to the folder with logits files
        files (list of strings): pickled logits files ((logits_val, y_val), (logits_test, y_test))
        m_kwargs (dictionary): keyword arguments for the calibration class initialization
        approach (string): "all" for multiclass calibration and "1-vs-K" for 1-vs-K approach.
        input (string): "probabilities" or "logits", specific to calibration method

    Returns:
        df (pandas.DataFrame): dataframe with calibrated and uncalibrated results for all the input files.

    """

    df = pd.DataFrame(
        columns=["Name", "Error", "ECE", "ECE2", "ECE_CW", "ECE_CW2", "ECE_FULL", "ECE_FULL2", "MCE", "MCE2", "Loss",
                 "Brier"])

    t1 = time.time()

    val_df = pd.read_csv(files[0], sep='\t')
    test_df = pd.read_csv(files[1], sep='\t')

    logits_val = val_df.iloc[:, 2:6].to_numpy()
    y_val = val_df.iloc[:, 1:2].to_numpy()

    logits_test = test_df.iloc[:, 2:6].to_numpy()
    y_test = test_df.iloc[:, 1:2].to_numpy()

    input_val = logits_val
    input_test = logits_test

    # Train and test model based on the approach "all" or "1-vs-K"
    if approach == "all":

        y_val_flat = y_val.flatten()

        model = TemperatureScaling(**m_kwargs)

        opt = model.fit(input_val, y_val_flat)
        print(f'the optimal temperature is {opt.x[0]}')
        file1 = open(f"model_weights/model_TS_{name}.txt", "w")
        file1.write(str(opt.x[0]))
        probs_val = model.predict(input_val)
        probs_test = model.predict(input_test)

        error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier = evaluate(
            softmax(logits_val), y_val, verbose=False)  # Uncalibrated results
        error1, ece1, ece1_2, ece_cw1_1, ece_cw1_2, ece_full1_1, ece_full1_2, mce1_1, mce1_2, loss1, brier1 = evaluate(
            softmax(logits_test), y_test, verbose=False)  # Uncalibrated results
        error2, ece2_1, ece2_2, ece_cw2_1, ece_cw2_2, ece_full2_1, ece_full2_2, mce2_1, mce2_2, loss2, brier2 = evaluate(
            probs_test, y_test, verbose=False)
        error3, ece3_1, ece3_2, ece_cw3_1, ece_cw3_2, ece_full3_1, ece_full3_2, mce3_1, mce3_2, loss3, brier3 = evaluate(
            probs_val, y_val, verbose=False)

        print(
            "Uncal Valid Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; "
            "loss %f, brier %f" % (
                error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier))
        print(
            "Uncal Test Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; "
            "loss %f, brier %f" % (
                error1, ece1, ece1_2, ece_cw1_1, ece_cw1_2, ece_full1_1, ece_full1_2, mce1_1, mce1_2, loss1, brier1))
        print(
            "Test Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; "
            "loss %f, brier %f" % (
                error2, ece2_1, ece2_2, ece_cw2_1, ece_cw2_2, ece_full2_1, ece_full2_2, mce2_1, mce2_2, loss2,
                brier2))
        print(
            "Validation Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 "
            "%f; loss %f, brier %f" % (
                error3, ece3_1, ece3_2, ece_cw3_1, ece_cw3_2, ece_full3_1, ece_full3_2, mce3_1, mce3_2, loss3,
                brier3))

    else:  # 1-vs-k models

        K = input_test.shape[1]

        probs_val = np.zeros_like(input_val)
        probs_test = np.zeros_like(input_test)

        # Go through all the classes
        for k in range(K):
            # Prep class labels (1 fixed true class, 0 other classes)
            y_cal = np.array(y_val == k, dtype="int")[:, 0]

            # Train model
            model = TemperatureScaling(**m_kwargs)
            model.fit(input_val[:, k], y_cal)  # Get only one column with probs for given class "k"

            probs_val[:, k] = model.predict(input_val[:, k])  # Predict new values based on the fittting
            probs_test[:, k] = model.predict(input_test[:, k])

        error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier = evaluate(
            softmax(logits_val), y_val, verbose=False)  # Uncalibrated results
        error1, ece1, ece1_2, ece_cw1_1, ece_cw1_2, ece_full1_1, ece_full1_2, mce1_1, mce1_2, loss1, brier1 = evaluate(
            softmax(logits_test), y_test, verbose=False)  # Uncalibrated results
        error2, ece2_1, ece2_2, ece_cw2_1, ece_cw2_2, ece_full2_1, ece_full2_2, mce2_1, mce2_2, loss2, brier2 = evaluate(
            probs_test, y_test, verbose=False)
        error3, ece3_1, ece3_2, ece_cw3_1, ece_cw3_2, ece_full3_1, ece_full3_2, mce3_1, mce3_2, loss3, brier3 = evaluate(
            probs_val, y_val, verbose=False)

        print(
            "Uncal Valid Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; "
            "loss %f, brier %f" % (
                error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier))
        print(
            "Uncal Test Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; "
            "loss %f, brier %f" % (
                error1, ece1, ece1_2, ece_cw1_1, ece_cw1_2, ece_full1_1, ece_full1_2, mce1_1, mce1_2, loss1, brier1))
        print(
            "Test Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; "
            "loss %f, brier %f" % (
                error2, ece2_1, ece2_2, ece_cw2_1, ece_cw2_2, ece_full2_1, ece_full2_2, mce2_1, mce2_2, loss2,
                brier2))
        print(
            "Validation Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 "
            "%f; loss %f, brier %f" % (
                error3, ece3_1, ece3_2, ece_cw3_1, ece_cw3_2, ece_full3_1, ece_full3_2, mce3_1, mce3_2, loss3,
                brier3))

    with open(f'result/{name}_{method}_val.txt', "wb") as f:
        np.savetxt(f, softmax(logits_val))
        np.savetxt(f, probs_val)

    with open(f'result/{name}_{method}_test.txt', "wb") as f2:
        np.savetxt(f2, softmax(logits_test))
        np.savetxt(f2, probs_test)

    df.loc[0] = [(name + "_val_uncalib"), error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss,
                 brier]
    df.loc[1] = [(name + "_test_uncalib"), error1, ece1, ece1_2, ece_cw1_1, ece_cw1_2, ece_full1_1, ece_full1_2,
                 mce1_1, mce1_2, loss1, brier1]
    df.loc[2] = [(name + "_test_calib"), error2, ece2_1, ece2_2, ece_cw2_1, ece_cw2_2, ece_full2_1, ece_full2_2,
                 mce2_1, mce2_2, loss2, brier2]
    df.loc[3] = [(name + "_val_calib"), error3, ece3_1, ece3_2, ece_cw3_1, ece_cw3_2, ece_full3_1, ece_full3_2,
                 mce3_1, mce3_2, loss3, brier3]

    t2 = time.time()
    print("Time taken:", (t2 - t1), "\n")

    return df
