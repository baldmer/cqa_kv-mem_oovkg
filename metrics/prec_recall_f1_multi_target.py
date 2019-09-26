import sys

'''
p = tp/(tp + fp)
r = tp/(tp + fn)

acc = tp + tn/(tp + tn + fp + fn)
acc = tp + 0/tota_labels

# In our problem the Precision and Recall are measured as:

$recall       =  target entities \cap predicted entities / target entities$
$precision    =  target entities \cap predicted entities / predicted entities$

'''

NKB = 1
KB_PAD_IDX = 0
NUMBER_OF_INSTANCES = 81994

def prec_recall_f1(file_name):

    sum_prec = 0
    sum_rec = 0
    sum_acc = 0
    counter = 0
    total_oov_all_gold = 0

    lines = open(file_name).read().strip().split('\n')

    total_multi_label = 0                                   
    total_single_label = 0

    for line in lines:
        line = line.split('\t')
        
        if len(line) != 4:
            continue
        
        pred_ent, all_gold_ent, _, _ = line
        
        # count oov entities in gold target.
        total_oov_all_gold += len([oov for oov in all_gold_ent.split('|') if oov == str(NKB)])
        
        # extract gold entities and filter out oov.
        all_gold_ent = [ent for ent in all_gold_ent.split('|') if ent not in [str(NKB)]]
        pred_ent = [ent for ent in pred_ent.split('|') if ent not in [str(NKB), str(KB_PAD_IDX)]]
        
        if len(all_gold_ent) > 1:
            total_multi_label +=1
        
        if len(all_gold_ent) == 1:
            total_single_label += 1
        
        if len(all_gold_ent) >= 1:
            
            correct_pred = len(set(all_gold_ent).intersection(set(pred_ent)))
            
            # acc or predicting exactly the target
            if correct_pred == len(all_gold_ent):
                sum_acc += 1
            
            sum_rec += correct_pred*1.0/float(len(all_gold_ent))
            counter +=1
            
            if len(pred_ent) > 0: # len pred_ent_id should be always 1 (pred. only one ent.)
                sum_prec += correct_pred*1.0/float(len(set(pred_ent)))

    print ("Total instances (no oov): %d" % counter)
    print ("Total instances with multi-labels (no oov): %d" % total_multi_label)
    print ("Total instances with single-label (no oov): %d" % total_single_label)
    print ("Total oov entities present in gold target (one entity could appear multiple times): %d" % total_oov_all_gold)
    
    # per baseline.
    #avg_prec= sum_prec/float(counter)
    #avg_rec = sum_rec/float(counter)
    #avg_acc = sum_acc/float(counter)

    # use all instances
    avg_prec= sum_prec/float(NUMBER_OF_INSTANCES)
    avg_rec = sum_rec/float(NUMBER_OF_INSTANCES)
    avg_acc = sum_acc/float(NUMBER_OF_INSTANCES)

    print("Avg. Precision: {:.1%}".format(avg_prec))
    print("Avg. Recall: {:.1%}".format(avg_rec))

    # will be afected by the multi-label problem and oov.
    print("Avg. Accuracy of predicting exactly all the targets: {:.1%}".format(avg_acc))


if __name__ == "__main__":
    #file_name = "transe_v1_4_epoch_model_out.txt"
    file_name = sys.argv[1]
    
    prec_recall_f1(file_name)

