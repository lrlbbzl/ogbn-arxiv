import torch
import numpy as np

file_name = "cassette.txt"

class Logger():
    def __init__(self, runs):
        self.runs = runs
        self.result = [[] for i in range(runs)]
    
    def update(self, run, res):
        self.result[run].append(res)

    def print(self, run, args):
        if run is not None:
            res = np.array(self.result[run])
            _finalist = np.argmax(res[:, 1]) # find the best one based on validate set
            print("run %d" % (run + 1))
            print("Highest accuracy in train set: %.2f" % (100 * res[:, 0].max()))
            print("Highest accuracy in validate set: %.2f" % (100 * res[:, 1].max()))
            print("Final accuracy in validate set: %.2f" % (100 * res[_finalist][1]))
            print("Final accuracy in test set: %.2f" % (100 * res[_finalist][2]))
        else:
            # print the final record
            best_res = []
            for res in self.result:
                res = np.array(res)
                res = res * 100
                _finalist = np.argmax(res[:, 1]) # the best place in validate set
                max_train, max_validate = np.max(res[:, 0]), res[_finalist][1]
                final_train, final_test = res[_finalist][0], res[_finalist][2]
                best_res.append((max_train, max_validate, final_train, final_test))
            best_res = np.array(best_res)
            print("Final record:")
            temp = np.array(best_res[:, 0])
            print("Highest accuracy in train set: %.2f ± %.2f" % (np.mean(temp), np.std(temp)))
            temp = np.array(best_res[:, 1])
            print("Highest accuracy in validate set: %.2f ± %.2f" % (np.mean(temp), np.std(temp)))
            temp = np.array(best_res[:, 2])
            print("Final accuracy in train set: %.2f ± %.2f" % (np.mean(temp), np.std(temp)))
            temp = np.array(best_res[:, 3])
            print("Final accuracy in test set: %.2f ± %.2f" % (np.mean(temp), np.std(temp)))

            with open(file_name, "a+") as fp:
                fp.write("Model: %s    hidden_channel: %d    num_layers: %d    \n" % (args.model, args.hidden_channel, args.num_layers))
                fp.write("dropout_rate: %.2f    learning_rate: %.4f    epochs: %d    \n" % (args.dropout_rate, args.learning_rate, args.epochs))
                temp = np.array(best_res[:, 0])
                fp.write("Highest accuracy in train set: %.2f ± %.2f\n" % (np.mean(temp), np.std(temp)))
                temp = np.array(best_res[:, 1])
                fp.write("Highest accuracy in validate set: %.2f ± %.2f\n" % (np.mean(temp), np.std(temp)))
                temp = np.array(best_res[:, 2])
                fp.write("Final accuracy in train set: %.2f ± %.2f\n" % (np.mean(temp), np.std(temp)))
                temp = np.array(best_res[:, 3])
                fp.write("Final accuracy in test set: %.2f ± %.2f\n" % (np.mean(temp), np.std(temp)))
                fp.write("\n\n")
            fp.close()