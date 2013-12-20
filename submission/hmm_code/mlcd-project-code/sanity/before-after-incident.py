__author__ = 'arenduchintala'

cleaned = '.cleaned'
root = "../../lowres_features/"
train_map = root + "trainset.recs.updated.lowres" + cleaned
test_map = root + "testset.recs.updated.lowres" + cleaned
dev_map = root + "devset.recs.updated.lowres" + cleaned
width = int(10)
h_width = int(width / 2)
types = ['sanity_before_', 'sanity_after_']
for type in types:
    all_lists = [train_map]
    for list_file in all_lists:
        lowres_file_list = open(list_file, 'r').readlines()
        for l in lowres_file_list:
            [file_name, time_of_clinical_obs] = l.strip().split('\t')
            lowres_feats = open(file_name, 'r').readlines()

            time_of_clinical_event = int(time_of_clinical_obs)
            if type == 'onset_':
                start_time_of_clinical_event = time_of_clinical_event - h_width if time_of_clinical_event > h_width else 0
                end_time_of_clinical_event = time_of_clinical_event + h_width if time_of_clinical_event < len(lowres_feats) else len(
                    lowres_feats)
            elif type == 'pre_':
                start_time_of_clinical_event = 0
                end_time_of_clinical_event = width
            elif type == 'post_':
                start_time_of_clinical_event = time_of_clinical_event
                end_time_of_clinical_event = time_of_clinical_event + width if (time_of_clinical_event + width) < len(
                    lowres_feats) else len(lowres_feats)
            elif type == 'sanity_before_':
                start_time_of_clinical_event = 0
                end_time_of_clinical_event = time_of_clinical_event
            elif type == 'sanity_after_':
                start_time_of_clinical_event = time_of_clinical_event
                end_time_of_clinical_event = len(lowres_feats) - 1

            examples = lowres_feats[start_time_of_clinical_event:end_time_of_clinical_event]
            example_file_name = file_name.replace('lowres_features', 'lowres_features/' + type + 'examples').replace('.feats', '-' + str(
                width) + '.' + type + 'examples')
            writer = open(example_file_name, 'w')
            writer.write(''.join(examples))
            writer.flush()
            writer.close()