import torch
import random
import os
import re

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, LlamaModel, get_linear_schedule_with_warmup

from dataset import Dataset as CoverageDataset
from instruction_constructor import InstructionConstructor
from configs import Configs as ProjectConfigs
from tqdm import tqdm
random.seed(42)


class Predictor(torch.nn.Module):
    def __init__(self, encoder, cover_label_idx):
        super(Predictor, self).__init__()
        self.encoder = encoder
        self.cover_label_idx = cover_label_idx
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, 2, bias=False, dtype=torch.bfloat16)

    def forward(self, input_ids, labels, valid_lens):
        outputs = self.encoder(
            input_ids,
            output_hidden_states=True,
        )
        
        input_tokens_rep = outputs.hidden_states[-1]  # (bs, seq_len, hidden_size)

        cover_label_representation = []
        for idx, each_input_tokens_rep in enumerate(input_tokens_rep):
            each_n_cov = valid_lens[idx]
            each_input_ids = input_ids[idx]
            # get the representation of cover label <c?>
            cover_label_position = (each_input_ids == self.cover_label_idx)  # (seq_len)
            cover_label_rep = each_input_tokens_rep[cover_label_position, :]
            cover_label_rep = cover_label_rep[-each_n_cov:, :]  # only consider the cov tags in the focal method
            cover_label_representation.append(cover_label_rep)
        cover_label_representation = torch.cat(cover_label_representation, dim=0)

        outputs_for_cover_tag = self.classifier(cover_label_representation)
        return outputs_for_cover_tag


class FmTc2CovDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels, valid_lens):
        self.inputs = inputs
        self.labels = labels
        self.valid_lens = valid_lens

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.valid_lens[idx]


def load_fm_tc2cov_data():
    dataset = CoverageDataset(configs)
    coverage_data = dataset.load_coverage_data(label_method='human') # format: [<Coverage, Context, Test case>]

    # convert format by adding instruction: 
    # list({"messages": [{"role":"system", "content": system_instruction,},{"role":"user", "content": user_instruction}]})
    fm_tc2cov_inputs = []
    fm_tc2cov_labels = []
    instruction_constructor = InstructionConstructor()
    for each_coverage in coverage_data:
        # add instructions of system and usesr
        coverage, context, test_case = each_coverage
        parsed_lines = []
        labels = []  # 0: uncover, 1: cover
        enter_focal_method_body = False
        for each_line in coverage.split("\n"):
            if "<COVER>" in each_line:
                enter_focal_method_body = True

                each_line_parsed = each_line.replace("<COVER>", '')
                each_line_parsed = f"{each_line_parsed} <c?>"
                labels.append(1)
            elif enter_focal_method_body and len(each_line.strip()) > 0 and any(c.isalpha() for c in each_line): 
                each_line_parsed = each_line + " <c?>"
                labels.append(0)
            else:
                each_line_parsed = each_line
            parsed_lines.append(each_line_parsed)

        focal_method = '\n'.join(parsed_lines)

        system_user_instruct = instruction_constructor.instruct_for_coverage_predict_given_tc_for_classifier(focal_method, test_case)

        fm_tc2cov_inputs.append(system_user_instruct)
        fm_tc2cov_labels.append(labels)
    return fm_tc2cov_inputs, fm_tc2cov_labels


def prepare_encoder_tokenizer():
    model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
    token='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'

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens({'additional_special_tokens': ['<c?>']})
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("INFO: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def prepare_dataloader(dataset_inputs, dataset_labels, tokenizer):
    input_ids = tokenizer.apply_chat_template(
            dataset_inputs,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=False,  # cannot truncate the input_ids as the focal method will be truncated.
            padding=True,
        )
    assert input_ids.shape[1] < 8000, f"The input_ids is too long ({input_ids.shape[1]}), please check the input_ids length."

    # padding the labels
    valid_lens = [len(each_label) for each_label in dataset_labels]
    max_len = max(valid_lens)
    for idx, each_label in enumerate(dataset_labels):
        dataset_labels[idx] = each_label + [-1] * (max_len - len(each_label))
    input_labels = torch.stack([torch.tensor(each_label) for each_label in dataset_labels], dim=0)
    valid_lens = torch.tensor(valid_lens)

    shuffle_indices = list(range(len(input_ids)))
    random.shuffle(shuffle_indices)
    train_indices = shuffle_indices[:int(len(input_ids) * 0.8)]
    test_indices = shuffle_indices[int(len(input_ids) * 0.8):]

    # datast_labels is a list()
    train_dataset = FmTc2CovDataset(input_ids[train_indices], 
                                    input_labels[train_indices],
                                    valid_lens[train_indices])
    test_dataset = FmTc2CovDataset(input_ids[test_indices], 
                                   input_labels[test_indices],
                                   valid_lens[test_indices])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader, train_indices, test_indices


def train_predictor(model, train_dataloader, test_dataloader):
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.module.classifier.parameters() if hasattr(model, 'module') else model.classifier.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    t_total = len(train_dataloader) // gradient_accumulation_steps * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(t_total*0.1),
                                                num_training_steps=t_total)

    for epoch in range(n_epochs):
        if hasattr(model, 'module'):
            model.module.encoder.eval()
            model.module.classifier.train()
        else:
            model.encoder.eval()
            model.classifier.train()

        tqdm_bar = tqdm(enumerate(train_dataloader), ncols=100, total=len(train_dataloader))
        for step, batch in tqdm_bar:
            input_ids, labels, valid_lens = batch
            input_ids, labels, valid_lens = input_ids.to(device), labels.to(device), valid_lens.to(device)
            
            outputs = model(input_ids, labels, valid_lens)
            labels = labels[labels != -1]
            loss = criterion(outputs, labels)

            if n_gpu > 1:
                loss = loss.mean()
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            tqdm_bar.set_description(f'Epoch {epoch+1} loss: {loss.item():.4f}')

            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        
        accuracy, precision, recall, f1 = evaluate_predictor(model, test_dataloader)
        print(f"Epoch {epoch+1}, accuracy: {accuracy:.2%} precision: {precision:.2%}, recall: {recall:.2%}, f1: {f1:.2%}\n")
        
    return model


@torch.no_grad()
def evaluate_predictor(model, test_dataloader):
    model.eval()

    total_labels, total_preds = [], []
    for batch in test_dataloader:
        input_ids, labels, valid_lens = batch
        input_ids, labels, valid_lens = input_ids.to(device), labels.to(device), valid_lens.to(device)

        outputs = model(input_ids, labels, valid_lens)
        pred = torch.argmax(outputs, dim=1)
        labels = labels[labels != -1]

        total_labels.extend(labels.cpu().numpy())
        total_preds.extend(pred.cpu().numpy())
    # calculate accuracy
    correct = sum([1 for i in range(len(total_labels)) if total_labels[i] == total_preds[i]])
    total = len(total_labels)

    # calculate precision, recall, f1
    tp = sum([1 for i in range(len(total_labels)) if total_labels[i] == 1 and total_preds[i] == 1])
    fp = sum([1 for i in range(len(total_labels)) if total_labels[i] == 0 and total_preds[i] == 1])
    fn = sum([1 for i in range(len(total_labels)) if total_labels[i] == 1 and total_preds[i] == 0])
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    return correct/total, precision, recall, f1
    


@torch.no_grad()
def predict_coverage(model, tokenizer, fm_tc2cov_inputs, fm_tc2cov_labels, test_indices):
    model.to(device)

    for each_idx in test_indices:
        prompt = fm_tc2cov_inputs[each_idx]
        input_ids = tokenizer.apply_chat_template(
            [prompt],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = input_ids.to(device)
        
        label = torch.tensor(fm_tc2cov_labels[each_idx]).to(device)
        valid_len = torch.tensor([len(label[label != -1])]).to(device)

        outputs = model(input_ids, label, valid_len)
        pred = torch.argmax(outputs, dim=1)
        
        # link the prediction back to the original focal method
        tc_fm = re.findall(r'```\n(.*?)\n```', prompt[1]['content'], re.DOTALL)
        fm = tc_fm[1].strip()
        cov_pred = fm
        for each_pred_label in pred:
            if each_pred_label == 1:
                cov_pred = cov_pred.replace("<c?>", "<COVER>", 1)
            else:
                cov_pred = cov_pred.replace("<c?>", "<UN-COVER>", 1)
        
        cov_true = fm
        for each_true_label in label[:valid_len[0]]:
            if each_true_label == 1:
                cov_true = cov_true.replace("<c?>", "<COVER>", 1)
            else:
                cov_true = cov_true.replace("<c?>", "<UN-COVER>", 1)
        print(f"Predicted coverage:\n{cov_pred}\n")
        print('---' * 10)
        print(f"True coverage:\n{cov_true}\n\n")
        print(f'pred labels: {pred.cpu().numpy()}')
        print(f'true labels: {label[:valid_len[0]].cpu().numpy()}')
        print('===' * 10)


def statistic_coverage(fm_tc2cov_labels):
    stat_total_tag = [0, 0, 0, 0]  # 1, 2, 3, >3
    stat_full_cov = 0
    stat_partial_cov = 0
    stat_partial_cov_for_greater_than_3 = 0

    for each_labels in fm_tc2cov_labels:
        if len(each_labels) > 3:
            stat_total_tag[3] += 1
        else:
            stat_total_tag[len(each_labels) - 1] += 1
        
        if 0 in each_labels:
            stat_partial_cov += 1
            if len(each_labels) > 3:
                stat_partial_cov_for_greater_than_3 += 1
        else:
            stat_full_cov += 1
    print(f'total samples: {len(fm_tc2cov_labels)}')
    print(f'full/partial coverage samples: {stat_full_cov}:{stat_partial_cov}')
    print(f'num of cov&uncov lines:\n  1 : {stat_total_tag[0]}\n  2 : {stat_total_tag[1]}\n  3 : {stat_total_tag[2]}\n  >3: {stat_total_tag[3]}')
    print(f'num of partial coverage samples with >3 cov&uncov lines: {stat_partial_cov_for_greater_than_3}')


def main():
    fm_tc2cov_inputs, fm_tc2cov_labels = load_fm_tc2cov_data()
    statistic_coverage(fm_tc2cov_labels)

    encoder, tokenizer = prepare_encoder_tokenizer()

    cover_label_idx = tokenizer.convert_tokens_to_ids("<c?>")

    train_dataloader, test_dataloader, train_indices, test_indices = prepare_dataloader(fm_tc2cov_inputs, fm_tc2cov_labels, tokenizer)

    predictor = Predictor(encoder, cover_label_idx)

    if os.path.exists(f'{configs.predictor_model_dir}/predictor_classifier.pth'):
        print(f"Loading the predictor model from {configs.predictor_model_dir}/predictor_classifier.pth")
        predictor.classifier.load_state_dict(torch.load(f'{configs.predictor_model_dir}/predictor_classifier.pth', map_location=device))
    else:
        predictor = train_predictor(predictor, train_dataloader, test_dataloader)

    # only save the classifier of predictor
    if not os.path.exists(configs.predictor_model_dir):
        os.makedirs(configs.predictor_model_dir)
    model = predictor.module if hasattr(predictor, 'module') else predictor
    torch.save(model.classifier.state_dict(), f'{configs.predictor_model_dir}/predictor_classifier.pth')

    predict_coverage(predictor, tokenizer, fm_tc2cov_inputs, fm_tc2cov_labels, test_indices)

    accuracy, precision, recall, f1 = evaluate_predictor(model, test_dataloader)
    print(f"\n\naccuracy: {accuracy:.2%} precision: {precision:.2%}, recall: {recall:.2%}, f1: {f1:.2%}\n")

    
if __name__ == "__main__":
    configs = ProjectConfigs()

    n_epochs = 5
    batch_size = 2
    gradient_accumulation_steps = 8
    n_gpu = torch.cuda.device_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main()