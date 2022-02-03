# Call for Customized Conversation: Customized Conversation Grounding Persona and Knowledge
* * *  

Source code for the baseline models of **Call for Customized Conversation: Customized Conversation Grounding Persona and Knowledge**, accepted at AAAI-22 [[paper](https://arxiv.org/abs/2112.08619)].



### Environment Setting
We trained the models under the setting of python==3.7, torch==1.5.0, transformers==4.5.0, tensorboardX==2.1, etc.

Please see **requirements.txt** file for more information.


### Dataset
We put train, dev, test files of the dataset in the **data** folder.

As we propose new dataset in our paper, the entire dataset is not opened yet.


Instead of the full version, we attach a toy dataset of ours.


### Training the models
Uncomment the command to start training the model in the **train.sh** file.

    sh train.sh 


### Evaluation
Uncomment the command in the **test.sh** file, to evaluate the model on the test-set.

    sh test.sh


### Inference
Uncomment the command of the **inference.sh** file, to generate utterances with the trained models.

    sh inference.sh


### Evaluate the submitted results from the leaderboard. Please specify the file name as an argument 'file_name'. We made a fake result file and used it.

    CUDA_VISIBLE_DEVICES=? python evaluation_leaderboard.py

The submitted files should include 6 generated machine's utterance per one dialog, and persona_pred with bool expression, knowledge_pred with integer (0-9) for each utterance. 
The result files should follow the format below:

    {"data": [{"persona_pred": [false, false, true, true, false], "knowledge_pred": 8, "machine_utt_0": ["It's the Museum of History and Industry, you love museum."], "dialog_ID": "JLG63YRTYNF3"}, {"persona_pred": [false, false, true, true, false], "knowledge_pred": 8, "machine_utt_1": ["It's the Museum of History and Industry, you love museum."], "dialog_ID": "JLG63YRTYNF3"}, {"persona_pred": [false, false, true, true, false], "knowledge_pred": 8, "machine_utt_2": ["It's the Museum of History and Industry, you love museum."], "dialog_ID": "JLG63YRTYNF3"}, {"persona_pred": [false, false, true, true, false], "knowledge_pred": 8, "machine_utt_3": ["It's the Museum of History and Industry, you love museum."], "dialog_ID": "JLG63YRTYNF3"}, {"persona_pred": [false, false, true, true, false], "knowledge_pred": 8, "machine_utt_4": ["It's the Museum of History and Industry, you love museum."], "dialog_ID": "JLG63YRTYNF3"}, {"persona_pred": [false, false, true, true, false], "knowledge_pred": 8, "machine_utt_5": ["It's the Museum of History and Industry, you love museum."], "dialog_ID": "JLG63YRTYNF3"}, ... ]}


(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.