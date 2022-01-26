#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
import logging
import random
import json
from argparse import ArgumentParser
from pprint import pformat
from tqdm import tqdm
import torch
from data_utils import get_testdata_loaders, add_special_tokens_
from ignite.metrics import Bleu, RougeL, RougeN, Accuracy
from ignite.metrics.precision import CharPrecision
from ignite.metrics.recall import CharRecall
logger = logging.getLogger(__file__)

SPECIAL_TOKENS = ["<machine>", "<human>", "<persona>", "<knowledge>"]


def get_pred_data_loaders(filename, tokenizer):
    #load submitted json file
    with open(filename,'r') as file:
        loaded_file = json.load(file)
    pred_list = loaded_file['data']
    logger.info("Tokenize the predicted utterances")
    for index, item in enumerate(pred_list):
        utt_i = index%6
        utt_key = 'machine_utt_' + str(utt_i)
        item[utt_key] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item[utt_key][0]))
    return pred_list

def run():
    parser = ArgumentParser()
    parser.add_argument("--test_dataset_path", type=str, default="data/test_focus.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--test_dataset_cache", type=str, default='data/focus_cache.tar.gz', help="Path or url of the dataset cache")
    parser.add_argument("--model_name", type=str, default="GPT2", help="{GPT2, BART, transformer-decoder, transformer-encdec}")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--file_name", type=str, default="data/pred.json", help="submitted json file name")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous utterances to keep in history")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--inference", action='store_true', help="If true, inference with gold knowledge")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))
    args.distributed = (args.local_rank != -1)


    logger.info("Get model and tokenizer")

    if args.model_name == 'GPT2':
        from transformers import GPT2Tokenizer
        from classification_modules import GPT2PK_ctxt
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
        model = GPT2PK_ctxt.from_pretrained(args.model_checkpoint)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)

    elif args.model_name == 'BART':
        from transformers import BartTokenizer
        from classification_modules import BARTPK_ctxt
        tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint)
        model = BARTPK_ctxt.from_pretrained(args.model_checkpoint)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)

    else:
        raise NotImplementedError

    #dataset = get_dataset_only_train(tokenizer, args.dataset_path, args.dataset_cache)
    logger.info("Prepare datasets")
    test_loader, test_sampler = get_testdata_loaders(args, tokenizer, generation=True)

    pred_loader = get_pred_data_loaders(args.file_name, tokenizer)


    with torch.no_grad():
        r1 = RougeN(ngram=1)
        r2 = RougeN(ngram=2)
        rl = RougeL()
        b1 = Bleu(ngram=1)
        b2 = Bleu(ngram=2)
        b3 = Bleu(ngram=3)
        b4 = Bleu(ngram=4)
        pre = CharPrecision()
        rec = CharRecall()
        pg = Accuracy()
        kg = Accuracy()

        for index, test_data in enumerate(tqdm(test_loader)):
            pred_item = pred_loader[index]
            if model.config.model_type == 'gpt2':
                input_ids, input_eos, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
                knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_token_ids, tot_knowledge_eos, reply, dialog, dialog_tti = test_data

            elif model.config.model_type == 'bart':
                input_ids, input_eos, decoder_input_ids, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
                knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_eos, reply, dialog = test_data
            else:
                raise NotImplementedError
            mask = (reply != tokenizer.pad_token_id)
            reply = reply[mask]

            #machine, human, persona, knowledge, padding, bos = 50257, 50258, 50259, 50260, 50261, 50256
            device = input_ids.get_device()

            #machine_tensor = torch.tensor([machine]).cuda(device)
            #persona_tensor = torch.tensor([persona]).cuda(device)
            #knowledge_tensor = torch.tensor([knowledge]).cuda(device)
            #bos_tensor = torch.tensor([bos]).cuda(device)
            #machine, human, persona, knowledge = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
            #special_tokens_list = [machine, human, persona, knowledge, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]


            gold_reply = reply

            utt_i = index%6
            utt_key = 'machine_utt_' + str(utt_i)
            gold_reply = gold_reply.tolist()
            pred_reply = pred_item[utt_key]
            persona_pred = [1 if elem==True else 0 for elem in pred_item['persona_pred']]
            #print('persona_pred: ', persona_pred)
            p_pred_cvtd = torch.tensor(persona_pred)
            knowledge_pred = [1 if pred_item['knowledge_pred']==elem else 0 for elem in range(10)]
            #print('knowledge pred: ', knowledge_pred)
            k_pred_cvtd = torch.tensor(knowledge_pred)

            #ROUGE
            r1.update((pred_reply, [gold_reply]))
            r2.update((pred_reply, [gold_reply]))
            rl.update((pred_reply, [gold_reply]))
            r1_res = r1.compute()
            r2_res = r2.compute()
            rl_res = rl.compute()

            #BLEU1,2,3,4 / BLEU avg
            b1.update((pred_reply, [gold_reply]))
            b2.update((pred_reply, [gold_reply]))
            b3.update((pred_reply, [gold_reply]))
            b4.update((pred_reply, [gold_reply]))
            b1_res = b1.compute()
            b2_res = b2.compute()
            b3_res = b3.compute()
            b4_res = b4.compute()

            #CharF1
            tensor_pred = torch.tensor(pred_reply).type(torch.cuda.FloatTensor)
            tensor_gold = torch.tensor(gold_reply).type(torch.cuda.FloatTensor)
            pre.update((tensor_pred, tensor_gold))
            rec.update((tensor_pred, tensor_gold))
            pre_res = pre.compute()
            rec_res = rec.compute()

            # PG
            p_label_cvtd = torch.tensor([1 if num in persona_grounding else 0 for num in range(5)], device=args.device)
            pg.update((p_pred_cvtd.squeeze(), p_label_cvtd))
            pg_res = pg.compute()

            # KG
            k_label_cvtd = torch.tensor([1 if num in knowledge_grounding else 0 for num in range(10)], device=args.device)
            kg.update((k_pred_cvtd, k_label_cvtd))
            kg_res = kg.compute()

            bleu_res = (b1_res.item() + b2_res.item() + b3_res.item() + b4_res.item())/4

            precision = pre_res.item()
            recall = rec_res.item()
            f1_res = (1.0 + 1 ** 2) * precision * recall / (1 ** 2 * precision + recall + 1e-15)

        print("F1: ", round(f1_res,3)*100)
        print("ROUGE1", round(r1_res['Rouge-1-F']*100,3))
        print("ROUGE2", round(r2_res['Rouge-2-F']*100,3))
        print("ROUGEL", round(rl_res['Rouge-L-F']*100,3))
        print("avg BLEU: ", round(bleu_res*100,3))
        print("PG: ", round(pg_res*100,3))
        print("KG: ", round(kg_res*100,3))


if __name__ == "__main__":
    run()
