import os
import random
import pandas as pd
from typing import List
from .flair.data import Sentence
from .flair.datasets import CSVClassificationCorpus
from time import time


def read_csv(file_name):
    print(file_name)
    data = pd.read_csv(file_name, header=None)
    for line in data.values:
        if len(line) == 2:
            yield line[0], line[1]
        elif len(line) == 1:
            yield line[0]


class Oracle:
    def get_label(self, sample: Sentence):
        raise NotImplementedError


class MemoryOracle(Oracle):
    def __init__(self, memory=None):
        if memory is None:
            self.memory = {}
        else:
            self.memory = memory
        self.read_all_labelled_data()

    def read_all_labelled_data(self):
        csv_file_list = []
        for root, _, files in os.walk("."):
            for file in files:
                if file.split('.')[-1] == 'csv' and file.find('labelled_') >= 0:
                    if os.path.basename(root) != '.':
                        csv_file_list.append(os.path.basename(root) + file)
                    else:
                        csv_file_list.append(file)
        for file in csv_file_list:
            for sample, label in read_csv(file):
                self.memory[sample] = label

    def get_label(self, sample):
        if sample.to_plain_string() in self.memory:
            return self.memory[sample.to_plain_string()]


class RuledOracle(Oracle):
    def __init__(self, function):
        self.function = function

    def get_label(self, sample):
        return self.function(sample)


class HumanOracle(RuledOracle):
    def __init__(self):
        def human_labeler(sample):
            print(sample)
            print('Y/N/S - Y for correct N for wrong S for Skip')
            choice = input()
            if choice.lower() == 'y':
                return 1
            if choice.lower() == 's':
                return None
            return 0

        super().__init__(human_labeler)


class HybridOracle(Oracle):
    def __init__(self,
                 experiment_name,
                 all_data_file,
                 valid_file='valid.txt',
                 test_file='test.txt',
                 use_memory=True,
                 use_rules=False,
                 memory=None,
                 rule_func=lambda x: 1,
                 generate_valid_file=False,
                 generate_test_file=False,
                 valid_size=None,
                 test_size=None,
                 overwrite=False):
        self.experiment_name = experiment_name
        if not os.path.exists(self.experiment_name):
            os.mkdir(self.experiment_name)
        self.all_data = list(read_csv(all_data_file))
        self.valid_file = self.experiment_name + '/' + valid_file
        self.test_file = self.experiment_name + '/' + test_file
        self.use_memory = use_memory
        self.use_rules = use_rules
        if self.use_memory:
            self.memory_oracle = MemoryOracle(memory)
        if self.use_rules:
            self.ruled_oracle = RuledOracle(rule_func)
        self.human_oracle = HumanOracle()
        if generate_valid_file:
            if overwrite or not os.path.exists(self.valid_file):
                if valid_size is None:
                    print("'valid_size' must be provided when generating validation file")
                    raise AttributeError
                else:
                    assert len(self.all_data) > valid_size
                    valid_data = random.sample(self.all_data, valid_size)
                    self.all_data = list(set(self.all_data) - set(valid_data))
                    self.save_labelled_csv(self._sentencify(valid_data), self.valid_file)
        if generate_test_file:
            if overwrite or not os.path.exists(self.test_file):
                if test_size is None:
                    print("'test_size' must be provided when generating test file")
                    raise AttributeError
                else:
                    assert len(self.all_data) > test_size
                    test_data = random.sample(self.all_data, test_size)
                    self.all_data = list(set(self.all_data) - set(test_data))
                    self.save_labelled_csv(self._sentencify(test_data), self.test_file)

    @staticmethod
    def _sentencify(l):
        return [Sentence(s) for s in l]

    def get_all_sentences(self):
        return self._sentencify(self.all_data)

    def get_label(self, sample):
        label = None
        if self.use_memory:
            label = self.memory_oracle.get_label(sample)
        if self.use_rules:
            if label is None:
                label = self.ruled_oracle.get_label(sample)
        if label is None:
            label = self.human_oracle.get_label(sample)
        return label

    def get_labelled_corpus(self, samples: List[Sentence]):
        train_file_name = f'labelled_{len(samples)}_{int(time())}.csv'
        self.save_labelled_csv(samples, train_file_name)
        corpus = CSVClassificationCorpus(data_folder=self.experiment_name,
                                         column_name_map={0: 'text', 1: 'label'},
                                         train_file=train_file_name,
                                         valid_file=self.valid_file,
                                         test_file=self.test_file,
                                         skip_header=False)
        return corpus

    def _get_labels_for_samples(self, samples: List[Sentence]):
        labels = []
        total_len = len(samples)
        for idx, sample in enumerate(samples):
            print(f'DONE {idx + 1}/{total_len} LABELS')
            label = self.get_label(sample)
            labels.append(label)
            if self.use_memory and label is not None:
                self.memory_oracle.memory[sample.to_plain_string()] = label
        return labels

    def save_labelled_csv(self, samples: List[Sentence], file_name):
        print('STARTING LABELLING')
        labels = self._get_labels_for_samples(samples)

        while None in labels:
            print('Re-Label skipped samples? Y/N')
            choice = input()
            if choice.lower() != 'y':
                break
            labels = self._get_labels_for_samples(samples)

        with open(file_name, 'w') as f:
            for idx in range(len(samples)):
                f.write(f'"{samples[idx].to_plain_string()}", {labels[idx]}\n')

