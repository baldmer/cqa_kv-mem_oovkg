import sys

'''
p = tp/(tp + fp)

r = tp/(tp + fn)
r = tp/(tp ) #there is no fn

acc = tp + tn/(tp + tn + fp + fn)

acc = tp + 0/tota_labels
'''

NKB = 1
#KB_PAD_IDX = 0
NUMBER_OF_INSTANCES = 81994


def prec_recall_f1(file_name):

    sum_prec = 0
    sum_rec = 0
    sum_acc = 0
    counter = 0

    total_oov = 0

    lines = open(file_name).read().strip().split('\n')

    total_multi_label = 0                                   
    total_single_label = 0

    for line in lines:
        line = line.split('\t')
        
        if len(line) != 7:
            continue
        
        _, _, _, pred_ent, all_gold_ent, _, _ = line
        
        # count oov entities in gold target.
        total_oov += len([oov for oov in all_gold_ent.split('|') if oov == str(NKB)])
        
        # extract golt entities and filter out oov.
        all_gold_ent = [ent for ent in all_gold_ent.split('|') if ent not in [str(NKB)]]
        pred_ent = [pred_ent] #it's a single str./ent., NKB not filtered, but it is in all_gold_ent, might affect counter.
        pred_ent = [ent for ent in pred_ent if ent not in [str(NKB)]]
        
        if len(all_gold_ent) > 1:
            total_multi_label +=1
        
        if len(all_gold_ent) == 1:
            total_single_label += 1
        
        if len(all_gold_ent) >= 1:
            #ids of ents. are treated as str.
            correct_pred = len(set(all_gold_ent).intersection(set(pred_ent)))
        
            #acc or predicting a single entity or all the entities (version support only first case)
            
            if correct_pred == len(all_gold_ent):
                sum_acc += 1
            
            # if one of the multiple calsses is correct, count as correct pred. for acc. (would be like precision)
            # if multilabel-suppored measure each correct pred. 
            '''
            if correct_pred > 0:
                sum_acc += 1
            '''
            
            sum_rec += correct_pred*1.0/float(len(all_gold_ent))
            counter +=1
            
            if len(pred_ent) > 0: # len pred_ent_id should be always 1 (pred. only one ent.)
                sum_prec += correct_pred*1.0/float(len(pred_ent))

    print ("Total instances (no oov): %d" % counter)
    print ("Total instances with multi-labels (no oov): %d" % total_multi_label)
    print ("Total instances with single-label (no oov): %d" % total_single_label)
    print ("Total oov gold entities used (one entity could appear multiple times): %d" % total_oov)

    #per baseline.
    avg_prec= sum_prec/float(counter)
    avg_rec = sum_rec/float(counter)

    avg_acc = sum_acc/float(counter)

    #use all instances
    #avg_prec= sum_prec/float(NUMBER_OF_INSTANCES)
    #avg_rec = sum_rec/float(NUMBER_OF_INSTANCES)

    print("Avg. Precision: {:.1%}".format(avg_prec))
    print("Avg. Recall: {:.1%}".format(avg_rec))

    # will be afected by the multi-label problem and oov.
    print("Avg. Accuracy of predicting a single entity: {:.1%}".format(avg_acc))


if __name__ == "__main__":
    #file_name = "transe_v1_4_epoch_model_out.txt"
    file_name = sys.argv[1]
    
    prec_recall_f1(file_name)

