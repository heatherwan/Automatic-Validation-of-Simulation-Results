import pickle


filename = ['log/exp00_point1024_nn40_cheby_4_3_confusion_mat',
            'log/exp00_point1024_nn40_cheby_4_3_mean_class_acc_record',
            'log/exp00_point1024_nn40_cheby_4_3_overall_acc_record']

for file in filename:
    with open(file, 'rb') as pickle_file:
        content = pickle.load(pickle_file)
        print(content)