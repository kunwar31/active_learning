from typing import List
from .flair.data import Sentence
from time import time


class Oracle:
    def get_label(self, sample: Sentence):
        raise NotImplementedError


class MemoryOracle(Oracle):
    def __init__(self, memory):
        if memory is None:
            self.memory = {}
        else:
            self.memory = memory

    def _read_all_labelled_data(self):
        pass

    def get_label(self, sample):
        if sample in self.memory:
            return self.memory[sample]


class RuledOracle(Oracle):
    def __init__(self, function):
        self.function = function

    def get_label(self, sample):
        return self.function(sample)


class HumanOracle(Oracle):
    def get_label(self, sample):
        print(sample)
        print('Y/N - Y for correct N for wrong')
        choice = input()
        if choice.lower() == 'y':
            return 1
        else:
            return 0


class MultiSourceOracle(Oracle):
    def __init__(self, all_data_path,
                 valid_path,
                 test_path,
                 use_memory=True,
                 use_rules=False,
                 memory=None,
                 rule_func=lambda x: 1):
        self.all_data_path = all_data_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.use_memory = use_memory
        self.use_rules = use_rules
        if self.use_memory:
            self.memory_oracle = MemoryOracle(memory)
        if self.use_rules:
            self.ruled_oracle = RuledOracle(rule_func)
        self.human_oracle = HumanOracle()

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

    def get_labelled_corpus(self, samples: List[Sentence], experiment_name):
        labels = []
        for sample in samples:
            labels.append(self.get_label(sample))
        with open(f'{experiment_name}/labelled_{len(samples)}_{int(time())}.csv', 'w') as f:
            for idx in range(len(samples)):
                f.write(f'{samples[idx].to_plain_string()}, {labels[idx]}')



