__author__ = 'arenduchintala'
"""
This script goes through each file in the training set and collects samples from the first 60 min of the patient log.

e.g. for the following patient record:
../../lowres_features/sepshock/s00292-3050-10-10-19-46.feats    116

the incident of sepshock was reported at time stamp 116, thus samples from 116-176 could serve as examples
for sepshock.
These samples are then placed in initial_examples folder.

../../lowres_features/pre_examples/sepshock/s00292-3050-10-10-19-46.pre_examples

"""

cleaned = '.cleaned'
root = "../../lowres_features/"
train_map = root + "trainset.recs.updated.lowres" + cleaned
test_map = root + "testset.recs.updated.lowres" + cleaned
dev_map = root + "devset.recs.updated.lowres" + cleaned
width = int(10)
h_width = int(width / 2)
types = ["onset_", "pre_", "post_"]
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

            examples = lowres_feats[start_time_of_clinical_event:end_time_of_clinical_event]
            example_file_name = file_name.replace('lowres_features', 'lowres_features/' + type + 'examples').replace('.feats', '-'+str(
                width) + '.' + type + 'examples')
            writer = open(example_file_name, 'w')
            writer.write(''.join(examples))
            writer.flush()
            writer.close()