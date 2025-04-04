#!/usr/bin/env python3
# 在第一个版本基础上，加入了说话人识别，作为辅助任务，识别效果有提升，最好为0.83
import copy
import logging
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

import librosa
from lang_trans import arabic

import soundfile as sf
from transformers.integrations import TensorBoardCallback  # tb

from model import Wav2Vec2ForCTCnCLS
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.tensorboard.writer import SummaryWriter  # tb
import os

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    is_apex_available,
    trainer_utils,
    EarlyStoppingCallback,  # tb
)

# if is_apex_available():
    # from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)

writer = SummaryWriter(log_dir="logging_dir")   # tb

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    alpha: Optional[float] = field(
        default=0.01,
        metadata={"help": "loss_cls + alpha * loss_ctc"},
    )
    beta: Optional[float] = field(
        default=0.05,
        metadata={"help": "loss_cls + beta * loss_spc"},
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer"}
    )


def configure_logger(model_args: ModelArguments, training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if model_args.verbose_logging:
        logging_level = logging.DEBUG
    elif trainer_utils.is_main_process(training_args.local_rank):
        logging_level = logging.INFO
    logger.setLevel(logging_level)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        default='emotion', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: Optional[str] = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    validation_split_name: Optional[str] = field(
        default="validation",
        metadata={
            "help": "The name of the validation data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    target_text_column: Optional[str] = field(
        default="text",
        metadata={"help": "Column in the dataset that contains label (target text). Defaults to 'text'"},
    )
    speech_file_column: Optional[str] = field(
        default="file",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'file'"},
    )
    target_feature_extractor_sampling_rate: Optional[bool] = field(
        default=False,
        metadata={"help": "Resample loaded audio to target feature extractor's sampling rate or not."},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=None,
        metadata={"help": "Filters out examples longer than specified. Defaults to no filtering."},
    )
    orthography: Optional[str] = field(
        default="librispeech",
        metadata={
            "help": "Orthography used for normalization and tokenization: 'librispeech' (default), 'timit', or 'buckwalter'."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # select which split as test
    split_id: str = field(
        default='01F', metadata={"help": "iemocap_ + splitid (e.g. 01M, 02F, etc) + train/test.csv"}
    )

    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Output file."},
    )


@dataclass
class Orthography:
    """
    Orthography scheme used for text normalization and tokenization.

    Args:
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to accept lowercase input and lowercase the output when decoding.
        vocab_file (:obj:`str`, `optional`, defaults to :obj:`None`):
            File containing the vocabulary.
        word_delimiter_token (:obj:`str`, `optional`, defaults to :obj:`"|"`):
            The token used for delimiting words; it needs to be in the vocabulary.
        translation_table (:obj:`Dict[str, str]`, `optional`, defaults to :obj:`{}`):
            Table to use with `str.translate()` when preprocessing text (e.g., "-" -> " ").
        words_to_remove (:obj:`Set[str]`, `optional`, defaults to :obj:`set()`):
            Words to remove when preprocessing text (e.g., "sil").
        untransliterator (:obj:`Callable[[str], str]`, `optional`, defaults to :obj:`None`):
            Function that untransliterates text back into native writing system.
    """

    do_lower_case: bool = False
    vocab_file: Optional[str] = None
    word_delimiter_token: Optional[str] = "|"
    translation_table: Optional[Dict[str, str]] = field(default_factory=dict)
    words_to_remove: Optional[Set[str]] = field(default_factory=set)
    untransliterator: Optional[Callable[[str], str]] = None
    tokenizer: Optional[str] = None

    @classmethod
    def from_name(cls, name: str):
        if name == "librispeech":
            return cls()
        if name == "timit":
            return cls(
                do_lower_case=True,
                # break compounds like "quarter-century-old" and replace pauses "--"
                translation_table=str.maketrans({"-": " "}),
            )
        if name == "buckwalter":
            translation_table = {
                "-": " ",  # sometimes used to represent pauses
                "^": "v",  # fixing "tha" in arabic_speech_corpus dataset
            }
            return cls(
                vocab_file=pathlib.Path(__file__).parent.joinpath("vocab/buckwalter.json"),
                word_delimiter_token="/",  # "|" is Arabic letter alef with madda above
                translation_table=str.maketrans(translation_table),
                words_to_remove={"sil"},  # fixing "sil" in arabic_speech_corpus dataset
                untransliterator=arabic.buckwalter.untransliterate,
            )
        raise ValueError(f"Unsupported orthography: '{name}'.")

    def preprocess_for_training(self, text: str) -> str:
        # TODO(elgeish) return a pipeline (e.g., from jiwer) instead? Or rely on branch predictor as is
        if len(self.translation_table) > 0:
            text = text.translate(self.translation_table)
        if len(self.words_to_remove) == 0:
            try:
                text = " ".join(text.split())  # clean up whitespaces
            except:
                text = "NULL"
        else:
            text = " ".join(w for w in text.split() if w not in self.words_to_remove)  # and clean up whilespaces
        return text

    def create_processor(self, model_args: ModelArguments) -> Wav2Vec2Processor:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
        if self.vocab_file:
            tokenizer = Wav2Vec2CTCTokenizer(
                self.vocab_file,
                cache_dir=model_args.cache_dir,
                do_lower_case=self.do_lower_case,
                word_delimiter_token=self.word_delimiter_token,
            )
        else:
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                self.tokenizer,
                cache_dir=model_args.cache_dir,
                do_lower_case=self.do_lower_case,
                word_delimiter_token=self.word_delimiter_token,
            )
        return Wav2Vec2Processor(feature_extractor, tokenizer)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None  # 表示不需要将填充后的序列长度调整为某个整数的倍数。
    audio_only = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        if self.audio_only is False:
            # label_features是标签特征，最后两位是说话人和情感标签
            label_features = [{"input_ids": feature["labels"][:-2]} for feature in features]  # 将-1改为了-2,feature["labels"]后两位是speaker和emotion标签
            cls_labels = [feature["labels"][-1] for feature in features]
            spc_labels = [feature["labels"][-2] for feature in features]  # 修改部分，添加了spe_labels, feature["labels"]的倒数第二位就是speaker标签，对应下方的batch["labels"][i].append(spc_lables[i])

            # # 测试
            # logger.info("cls_labels")
            # logger.info(cls_labels)
            # logger.info("spc_labels")
            # logger.info(spc_labels)

        # 这个方法将对输入特征进行填充，使它们具有相同的长度，并返回一个批处理张量。
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if self.audio_only is False:
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            ctc_labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            # batch["labels"] = (ctc_labels, torch.tensor(cls_labels))  # labels = (ctc_labels, cls_labels)

            # 修改部分,将batch['labels']重新赋值,增加了spc_labels
            # labels = (ctc_labels, cls_labels, spc_labels)
            batch["labels"] = (ctc_labels, torch.tensor(cls_labels), torch.tensor(spc_labels))
        return batch


# 训练器类
class CTCTrainer(Trainer):
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        for k, v in inputs.items():
            # logger.info("k = ")
            # logger.info(k)
            # logger.info("v = ")
            # logger.info(v)
            if isinstance(v, torch.Tensor):
                kwargs = dict(device=self.args.device)
                if self.deepspeed and inputs[k].dtype != torch.int64:
                    kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
                inputs[k] = v.to(**kwargs)

            if k == 'labels':  # labels are list of tensor, not tensor, special handle here
                for i in range(len(inputs[k])):
                    kwargs = dict(device=self.args.device)
                    if self.deepspeed and inputs[k][i].dtype != torch.int64:
                        kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
                    inputs[k][i] = inputs[k][i].to(**kwargs)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    configure_logger(model_args, training_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    orthography = Orthography.from_name(data_args.orthography.lower())
    orthography.tokenizer = model_args.tokenizer
    processor = orthography.create_processor(model_args)

    if data_args.dataset_name == 'emotion':
        # train_dataset = datasets.load_dataset('csv', data_files='iemocap3/iemocap_' + data_args.split_id + '.train3.csv',
        #                                       cache_dir=model_args.cache_dir)['train']
        # val_dataset = datasets.load_dataset('csv', data_files='iemocap3/iemocap_' + data_args.split_id + '.test3.csv',
        #                                    cache_dir=model_args.cache_dir)['train']
        train_dataset = datasets.load_dataset('csv', data_files='iemocap3_2/emotion_speaker_text' + '.train.csv',
                                              cache_dir=model_args.cache_dir)['train']
        val_dataset = datasets.load_dataset('csv', data_files='iemocap3_2/emotion_speaker_text' + '.test.csv',
                                            cache_dir=model_args.cache_dir)['train']
        # 四种情感（主任务），十个说话人(辅助任务)
        cls_label_map = {"e0": 0, "e1": 1, "e2": 2, "e3": 3}  # 情感标签
        spc_label_map = {"01F": 0, "01M": 1, "02F": 2, "02M": 3,
                         "03F": 4, "03M": 5, "04F": 6, "04M": 7,
                         "05F": 8, "05M": 9}  # 说话人标签

    model = Wav2Vec2ForCTCnCLS.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        gradient_checkpointing=True,  # training_args.gradient_checkpointing,
        vocab_size=len(processor.tokenizer),
        cls_len=len(cls_label_map),
        spc_len=len(spc_label_map),
        alpha=model_args.alpha,
        beta=model_args.beta,
    )

    logger.info("alpha = ")
    logger.info(model_args.alpha)
    logger.info("beta = ")
    logger.info(model_args.beta)

    wer_metric = datasets.load_metric("wer")
    target_sr = processor.feature_extractor.sampling_rate if data_args.target_feature_extractor_sampling_rate else None
    vocabulary_chars_str = "".join(t for t in processor.tokenizer.get_vocab().keys() if len(t) == 1)
    vocabulary_text_cleaner = re.compile(  # remove characters not in vocabulary
        f"[^\s{re.escape(vocabulary_chars_str)}]",  # allow space in addition to chars in vocabulary
        flags=re.IGNORECASE if processor.tokenizer.do_lower_case else 0,
    )
    text_updates = []

    def prepare_example(example, audio_only=False):  # TODO(elgeish) make use of multiprocessing?
        example["speech"], example["sampling_rate"] = librosa.load(example[data_args.speech_file_column],
                                                                   sr=target_sr)  # 读取example['file']

        # logger.info("example['file] = ")
        # logger.info(example[data_args.speech_file_column])

        if data_args.max_duration_in_seconds is not None:
            example["duration_in_seconds"] = len(example["speech"]) / example["sampling_rate"]
        if audio_only is False:
            # Normalize and clean up text; order matters!
            updated_text = orthography.preprocess_for_training(example[data_args.target_text_column])
            updated_text = vocabulary_text_cleaner.sub("", updated_text)
            if updated_text != example[data_args.target_text_column]:
                text_updates.append((example[data_args.target_text_column], updated_text))
                example[data_args.target_text_column] = updated_text
        return example

    if training_args.do_train:
        train_dataset = train_dataset.map(prepare_example, remove_columns=[data_args.speech_file_column])
    if training_args.do_predict:
        val_dataset = val_dataset.map(prepare_example, fn_kwargs={'audio_only': True})

    elif training_args.do_eval:
        val_dataset = val_dataset.map(prepare_example, remove_columns=[data_args.speech_file_column])

    if data_args.max_duration_in_seconds is not None:
        logger.info("data_args.max_duration_in_seconds is not None")

        def filter_by_max_duration(example):
            return example["duration_in_seconds"] <= data_args.max_duration_in_seconds

        if training_args.do_train:
            old_train_size = len(train_dataset)
            train_dataset = train_dataset.filter(filter_by_max_duration, remove_columns=["duration_in_seconds"])
            if len(train_dataset) > old_train_size:
                logger.warning(
                    f"Filtered out {len(train_dataset) - old_train_size} train example(s) longer than {data_args.max_duration_in_seconds} second(s)."
                )
        if training_args.do_predict or training_args.do_eval:
            old_val_size = len(val_dataset)
            val_dataset = val_dataset.filter(filter_by_max_duration, remove_columns=["duration_in_seconds"])

            logger.info("after filter, val_dataset: ")
            logger.info(val_dataset)

            if len(val_dataset) > old_val_size:
                logger.warning(
                    f"Filtered out {len(val_dataset) - old_val_size} validation example(s) longer than {data_args.max_duration_in_seconds} second(s)."
                )
    # logger.info(f"Split sizes: {len(train_dataset)} train and {len(val_dataset)} validation.")

    logger.warning(f"Updated {len(text_updates)} transcript(s) using '{data_args.orthography}' orthography rules.")
    if logger.isEnabledFor(logging.DEBUG):
        for original_text, updated_text in text_updates:
            logger.debug(f'Updated text: "{original_text}" -> "{updated_text}"')
    text_updates = None

    def prepare_dataset(batch, audio_only=False):
        # check that all files have the correct sampling rate
        assert (
                len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
        if audio_only is False:  # 这里修改batch["labels"],依次将csv文件中的text，speaker，emotion信息转换为数据读入
            # 改这里（）
            cls_labels = list(map(lambda e: cls_label_map[e], batch["emotion"]))  # 拿出emotion标签对应的数字
            spc_labels = list(map(lambda s: spc_label_map[s], batch["speaker"]))  # 拿出speaker标签对应的数字
            with processor.as_target_processor():
                batch["labels"] = processor(batch[data_args.target_text_column]).input_ids  # batch["labels"]是预处理后的文本数据,可以是多维的
            # logger.info('batch["labels"]')
            # logger.info(batch["labels"])
            for i in range(len(spc_labels)):  # 将speaker标签加到batch["labels"]的倒数第二位, 对应上方的__call__()函数
                batch["labels"][i].append(spc_labels[i])  # batch["labels"] element has to be a single list
            for i in range(len(cls_labels)):  # 将emotion标签加到batch["labels"][i]的最后一位
                batch["labels"][i].append(cls_labels[i])  # batch["labels"] element has to be a single list

        # batch就是处理后的每条音频数据，
        # 其标签为'emotion': ['e0'], 'sampling_rate': [16000], 'speech': [巨量数据], 'text': ['YEAH '], 'input_values', 'labels': [[22, 5, 7, 11, 4, 0]]
        return batch

    if training_args.do_train:
        train_dataset = train_dataset.map(
            prepare_dataset,
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
    if training_args.do_predict:
        val_dataset = val_dataset.map(
            prepare_dataset,
            fn_kwargs={'audio_only': True},
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )

    elif training_args.do_eval:
        logger.info("do_eval")
        val_dataset = val_dataset.map(
            prepare_dataset,
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # 计算评价指标（未改）
    def compute_metrics(pred):
        # logger.info("pred.predictions: ")
        # logger.info(pred.predictions)

        # 计算cls的正确率
        cls_pred_logits = pred.predictions[1]  # predictions[1]是cls，ctc是[0],那spc是[2]
        cls_pred_ids = np.argmax(cls_pred_logits, axis=-1)  # 沿着列，取出cls_pred_logits的每行最大值
        total = len(pred.label_ids[1])
        cls_correct = (cls_pred_ids == pred.label_ids[1]).sum().item()  # label = (ctc_label, cls_label), label_id是真实值

        # 修改(添加spc的评估部分)
        spc_pred_logits = pred.predictions[2]  # predictions[1]是cls，ctc是[0],那spc是[2]
        spc_pred_ids = np.argmax(spc_pred_logits, axis=-1)  # 沿着列，取出spc_pred_logits的每行最大值
        spc_correct = (spc_pred_ids == pred.label_ids[2]).sum().item()  # label = (ctc_label, cls_label), label_id是真实值

        # 计算ctc的错词率和字符串长度
        ctc_pred_logits = pred.predictions[0]
        ctc_pred_ids = np.argmax(ctc_pred_logits, axis=-1)
        pred.label_ids[0][pred.label_ids[0] == -100] = processor.tokenizer.pad_token_id
        ctc_pred_str = processor.batch_decode(ctc_pred_ids)
        # we do not want to group tokens when computing the metrics
        ctc_label_str = processor.batch_decode(pred.label_ids[0], group_tokens=False)
        if logger.isEnabledFor(logging.DEBUG):
            for reference, predicted in zip(label_str, pred_str):
                logger.debug(f'reference: "{reference}"')
                logger.debug(f'predicted: "{predicted}"')
                if orthography.untransliterator is not None:
                    logger.debug(f'reference (untransliterated): "{orthography.untransliterator(reference)}"')
                    logger.debug(f'predicted (untransliterated): "{orthography.untransliterator(predicted)}"')

        # wer由ctc计算得来，后续可以考虑删除wer
        wer = wer_metric.compute(predictions=ctc_pred_str, references=ctc_label_str)

        # tensorboard
        # writer.add_scalar("eval/accuracy", correct / total, trainer.global_step)

        return {"acc": cls_correct / total, "wer": wer, "spc_acc": spc_correct / total, "correct": cls_correct, "total": total, "strlen": len(ctc_label_str)}

    if model_args.freeze_feature_extractor:
        # logger.info("model_args.freeze_feature_extractor = True")
        model.freeze_feature_extractor()
        # logger.info("after model.freeze_feature_extractor(), val_dataset[0]['file']")
        # logger.info(val_dataset[0]['file'])

    # 在predict的时候下面会出现val_dataset的KeyError
    # 深拷贝备份val_dataset可以解决（但在do_eval时要注释掉）
    # temp_val_dataset = copy.deepcopy(val_dataset)
    logger.info("val_dataset")
    logger.info(val_dataset)

    # 问题在这里。。。定义了一个trainer，val_dataset的值就改变了，只剩下一个键值对{’input_value':[...]}
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,  # 修改部分，暂时不用评估指标
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
        # tb
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=200, early_stopping_threshold=0.01),
            TensorBoardCallback(),
        ],
    )

    # train_loss = trainer.state.metrics['loss']
    # writer.add_scalar("eval/loss", train_loss, trainer.global_step)

    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

    if training_args.do_predict:
        logger.info('******* Predict ********')

        data_collator.audio_only = True

        logger.info("before predict , val_dataset:")
        logger.info(val_dataset)

        # 这里也有问题，trainer.predict()之后，也会改变val_dataset的值, 下面需要重新给val_dataset赋值
        predictions, labels, metrics = trainer.predict(val_dataset, metric_key_prefix="predict")
        logits_ctc, logits_cls, logits_spc = predictions
        pred_ids = np.argmax(logits_cls, axis=-1)
        pred_probs = F.softmax(torch.from_numpy(logits_cls).float(), dim=-1)

        # 深拷贝重新给val_dataset赋值，程序正常运行
        # val_dataset = copy.deepcopy(temp_val_dataset)
        # logger.info("after predict , val_dataset:")
        # logger.info(val_dataset)
        # logger.info("after predict , val_dataset[0]['file']:")
        # logger.info(val_dataset[0]['file'])
        with open(data_args.output_file, 'w') as f:
            for i in range(len(pred_ids)):
                f.write(val_dataset[i]['file'].split("/")[-1] + " " + str(
                    len(val_dataset[i]['input_values']) / 16000) + " ")
                pred = pred_ids[i]
                f.write(str(pred) + ' ')
                for j in range(4):
                    f.write(' ' + str(pred_probs[i][j].item()))
                f.write('\n')
        f.close()

    elif training_args.do_eval:
        predictions, labels, metrics = trainer.predict(val_dataset, metric_key_prefix="eval")
        logits_ctc, logits_cls, logits_spc = predictions  # 预测结果应该有3个

        pred_ids = np.argmax(logits_cls, axis=-1)
        correct = np.sum(pred_ids == labels[1])
        acc = correct / len(pred_ids)
        print('correct:', correct, ', acc:', acc)

    writer.close()


if __name__ == "__main__":
    main()
