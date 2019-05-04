import torch
import os


def _make_model_file_dict(record_dir, id, network_list):
    model_file_dict = {}
    for network_name in network_list:
        model_file_dict[network_name] = os.path.join(record_dir, "{}-{}".format(network_name, id))
    return model_file_dict


class SaveRestoreService():
    def __init__(self, record_dir=None, network_dict=None):
        self.do_nothing = record_dir is None or network_dict is None
        if self.do_nothing:
            return
        self.step_id = 0
        self.model_file_dict = None
        self.network_dict = network_dict
        record_dir = os.path.abspath(record_dir)
        if os.path.exists(record_dir):
            max_step_id = -1
            network_list = list(network_dict.keys())
            for f in os.listdir(record_dir):
                if f.startswith(network_list[0]):
                    step_id = int(f.split("-")[1])
                    if step_id > max_step_id:
                        max_step_id = step_id
            if self.step_id <= max_step_id:
                self.step_id = max_step_id
                self.model_file_dict = _make_model_file_dict(record_dir, max_step_id, network_list)
        else:
            os.makedirs(record_dir)
        self.record_dir = record_dir

    def restore(self):
        if self.do_nothing or self.model_file_dict is None:
            return
        print("restored from", self.model_file_dict)
        for k, network in self.network_dict.items():
            network.load_state_dict(torch.load(self.model_file_dict[k]))

    def save(self):
        if self.do_nothing:
            return
        self.model_file_dict = _make_model_file_dict(self.record_dir, self.step_id, self.network_dict.keys())
        for k, network in self.network_dict.items():
            torch.save(network.state_dict(), self.model_file_dict[k])
        self.step_id += 1