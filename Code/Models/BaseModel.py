import tensorflow as tf


class BaseModel:
    def __init__(self, name, save_dir):
        self.NAME = name
        self.SAVE_DIR = save_dir

    def build_graph(self, summary):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def train(self, code_summary=True):
        pass
