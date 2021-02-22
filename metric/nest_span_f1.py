#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: xiaoyli  
# description:
# 



class Tag(object):
    def __init__(self, tag, begin, end):
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.tag, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})


def nested_calculate_f1(pred_span_tag_lst, gold_span_tag_lst, dims=2):
    if dims == 2:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        count = 0
        for pred_span_tags, gold_span_tags in zip(pred_span_tag_lst, gold_span_tag_lst):
            pred_set = set((tag.begin, tag.end, tag.tag) for tag in pred_span_tags)
            gold_set = set((tag.begin, tag.end, tag.tag) for tag in gold_span_tags)
            #print(pred_set)
            #print(gold_set)
            count += 1
            print(count)
            for pred in pred_set:
                if pred in gold_set:
                    print('These are true examples')
                    print(pred)
                    true_positives += 1
                else:
                    false_positives += 1

            for pred in gold_set:
                if pred not in pred_set:
                    false_negatives += 1

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return precision, recall, f1 
    
    else:
        raise ValueError("Can not be other number except 2 !")

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
def nested_calculate_f1_macros(pred_span_tag_lst, gold_span_tag_lst, label_lst, dims=2):
    if dims == 2:
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred_span_tags, gold_span_tags in zip(pred_span_tag_lst, gold_span_tag_lst):
            pred_set = set((tag.begin, tag.end, tag.tag) for tag in pred_span_tags)
            gold_set = set((tag.begin, tag.end, tag.tag) for tag in gold_span_tags)
            js = len(pred_set)
            np = len(gold_set)
            print(js)
            print(np)
            if js == np:
                cnf_matrix = confusion_matrix(pred_set, gold_set)
                FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
                FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
                TP = np.diag(cnf_matrix)
                TN = cnf_matrix.sum() - (FP + FN + TP)
                FP = FP.astype(float)
                FN = FN.astype(float)
                TP = TP.astype(float)
                TN = TN.astype(float)
                # Sensitivity, hit rate, recall, or true positive rate
                TPR = TP/(TP+FN)
                print(TPR)
                print('this was recall-micro')
                # Specificity or true negative rate
                TNR = TN/(TN+FP) 
                # Precision or positive predictive value
                PPV = TP/(TP+FP)
                print(PPV)
                print('this was Precision-micro')
                # Negative predictive value
                NPV = TN/(TN+FN)
                # Fall out or false positive rate
                FPR = FP/(FP+TN)
                # False negative rate
                FNR = FN/(TP+FN)
                # False discovery rate
                FDR = FP/(TP+FP)
                # Overall accuracy for each class
                ACC = (TP+TN)/(TP+FP+FN+TN)
                print(ACC)
                print('this was accuracy- micro')
                matrix = confusion_matrix(pred_set,gold_set, labels=label_lst)
                print('Confusion matrix : \n',matrix)
                # outcome values order in sklearn
                tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=label_lst).reshape(-1)
                print('Outcome values : \n', tp, fn, fp, tn)
                # classification report for precision, recall f1-score and accuracy
                matrix = classification_report(actual,predicted,labels=label_lst)
                print('Classification report : \n',matrix)


                return matrix 
            return 0
    
    else:
        raise ValueError("Can not be other number except 2 !")
