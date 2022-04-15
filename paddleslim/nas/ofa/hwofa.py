import random
import logging
import numpy as np

from collections import OrderedDict

from paddle import DataParallel


from .ofa import OFA
from .layers_base import BaseBlock
from ...core import GraphWrapper, dygraph2program
from .get_sub_model import get_prune_params_config, prune_params, check_search_space
from ...common import get_logger

_logger = get_logger(__name__, level=logging.INFO)


class HWOFA(OFA):
    def __init__(self,
                 model,
                 run_config=None,
                 distill_config=None,
                 elastic_order=None,
                 train_full=False,
                 candidate_config=None,
                 backbone='resnet34'
                 ):
        super().__init__(model, run_config, distill_config, elastic_order, train_full)
        self.model.eval()
        self._clear_search_space()
        self.cand_cfg = candidate_config
        # self.cand_cfg = {
        #     'i': [224],  # image size
        #     'd': [(3, 3), (4, 4), (6, 6), (3, 3)],  # depth
        #     'k': [3],  # kernel size
        #     'c': [1.0, 0.95, 0.9, 0.85]  # channel ratio
        # }
        self.im_size_dict = {x: i for i, x in enumerate(self.cand_cfg['i'], 1)}
        self.depth_dict = {k: k-1 for s, e in self.cand_cfg['d'] for k in range(s, e+1)}
        self.kernel_dict = {x: i for i, x in enumerate(self.cand_cfg['k'], 1)}
        self.channel_dict = {x: i for i, x in enumerate(self.cand_cfg['c'], 1)}
        self.subnet_code = ''
        self.layer_factor = 6
        if backbone in ['resnet34']:
            self.block_conv_num = 2  # res18,res34: 2, >res50: 3
        elif backbone in ['resnet50']:
            self.block_conv_num = 3
        else:
            raise ValueError
    
    def gen_subnet_code(self):
        submodel_code = [self.im_size_dict[self.act_im_size]]
        submodel_code += [self.depth_dict[d] for d in self.act_depth_list]
        submodel_code_str = ''.join([str(x) for x in submodel_code])
        # k_code = ['', '', '', '', '']
        c_code = ['', '', '', '', '']
        for k, v in self.current_config.items():
            if 'layer' in k and 'downsample' not in k:
                if 'layer1' in k:
                    idx = 1
                elif 'layer2' in k:
                    idx = 2
                elif 'layer3' in k:
                    idx = 3
                else:
                    idx = 4
                # k_code[idx] += str(self.kernel_dict[v['kernel_size']])
                c_code[idx] += str(self.channel_dict[v['expand_ratio']])
            elif 'conv1' == k:
                # k_code[0] += str(self.kernel_dict[v['kernel_size']])
                c_code[0] += str(self.channel_dict[v['expand_ratio']])
        c_code = [x.ljust(self.layer_factor*self.block_conv_num, '0') for x in c_code[1:]]
        # k_code = [x.ljust(self.layer_factor*3,'0') for x in k_code[1:]]
        for x in c_code:
            submodel_code_str += x
        # for x in k_code:
        #     submodel_code_str += x
        return submodel_code_str

    def active_subnet(self, img_size=None):
        if img_size is None:
            self.act_im_size = random.choice(self.cand_cfg['i'])
        else:
            self.act_im_size = img_size
        self.act_depth_list = [random.randint(s, e) for s, e in self.cand_cfg['d']]

        self.current_config = OrderedDict()
        for key in self.universe:
            if key in self._ofa_layers:
                if key == 'conv1' or 'layer1' in key or 'layer2' in key:
                    self.current_config[key] = {'expand_ratio': random.choice(self.cand_cfg['c'])}
                elif 'layer3' in key or 'layer4' in key:
                    self.current_config[key] = {'expand_ratio': random.choice(self.cand_cfg['c'])}
                else:
                    raise ValueError
        self.current_config['fc'] = {}
        self._broadcast_ss()

    def _clear_search_space(self):
        """ find shortcut in model, and clear up the search space """
        _st_prog = dygraph2program(self.model, inputs=[2, 3, 224, 224], dtypes=[np.float32])
        self._same_ss = check_search_space(GraphWrapper(_st_prog))

        self._same_ss = sorted(self._same_ss)
        self._param2key = {}
        self._broadcast = True

        self.universe = []
        ### the name of sublayer is the key in search space
        ### param.name is the name in self._same_ss
        model_to_traverse = self.model._layers if isinstance(self.model, DataParallel) else self.model
        for name, sublayer in model_to_traverse.named_sublayers():
            if isinstance(sublayer, BaseBlock):
                for param in sublayer.parameters():
                    if self._find_ele(param.name, self._same_ss):
                        self._param2key[param.name] = name
                    if 'conv' in name:
                        self.universe.append(name)
        self.universe.sort()
        ### double clear same search space to avoid outputs weights in same ss.
        tmp_same_ss = []
        for ss in self._same_ss:
            per_ss = []
            for key in ss:
                if key not in self._param2key.keys():
                    continue

                if self._param2key[key] in self._ofa_layers.keys() and (
                    'expand_ratio' in self._ofa_layers[self._param2key[key]] or \
                        'channel' in self._ofa_layers[self._param2key[key]]):
                    per_ss.append(key)
                else:
                    _logger.info("{} not in ss".format(key))
            if len(per_ss) != 0:
                tmp_same_ss.append(per_ss)
        self._same_ss = tmp_same_ss

        for per_ss in self._same_ss:
            for ss in per_ss[1:]:
                if 'expand_ratio' in self._ofa_layers[self._param2key[ss]]:
                    self._ofa_layers[self._param2key[ss]].pop('expand_ratio')
                elif 'channel' in self._ofa_layers[self._param2key[ss]]:
                    self._ofa_layers[self._param2key[ss]].pop('channel')
                if len(self._ofa_layers[self._param2key[ss]]) == 0:
                    self._ofa_layers.pop(self._param2key[ss])

    def forward(self, x):
        teacher_output = None
        if self._add_teacher:
            self._reset_hook_before_forward()
            teacher_output = self.ofa_teacher_model.model.forward(x)
            teacher_output.stop_gradient = True

        # self.active_subnet()
        # print(self.gen_subnet_code())

        if teacher_output is not None and self.training:
            stu_out = self.model.forward(x, self.act_depth_list)
            return stu_out, teacher_output
        else:
            return self.model.forward(x, self.act_depth_list)
