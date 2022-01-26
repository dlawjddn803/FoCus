#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.

#data format
#{"data": [{"utterance": ["It's the Museum of History and Industry, you love museum."], "persona_grounding": [false, false, false, true, false], "knowledge_answer_index": 8}]}

import json

dial_id_list = list()
with open('test_focus.json','r') as testfile:
    test_data = json.load(testfile)
    for item in test_data['data']:
        dial_id = item['dialogID']
        dial_id_list.append(dial_id)
utterance=["It's the Museum of History and Industry, you love museum."]
content_list = list()
for i in range(1445):
    for j in range(6):
        elem = {"persona_pred": [False, False, True, True, False], "knowledge_pred": 8}
        utt_key = 'machine_utt_' + str(j)
        elem[utt_key] = utterance
        elem['dialog_ID'] = dial_id_list[i]
        content_list.append(elem)
dictionary = dict()
dictionary['data'] = content_list

with open('pred.json','w') as file:
    json.dump(dictionary, file, indent="\t")
