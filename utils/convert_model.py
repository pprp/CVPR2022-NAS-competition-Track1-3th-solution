import paddle 
import os 

'''
student subnet:
    # model._layers.blocks.13.bn2.weight [256]
    # ValueError: model.blocks.0.conv.fn.weight is not found in the providing file.

teacher network:
    # ofa_teacher_model.model.blocks.23.bn2._variance [512]
    # ValueError: ofa_teacher_model.model.blocks.0.conv.weight is not found in the providing file.
'''

final_path = "checkpoints/res48-depth/final_multigpu.pdparams"
save_path = "checkpoints/res48-depth/final.pdparams"


old = paddle.load(final_path)
new = dict()

for k, v in old.items():
    # delete _layers in multi gpu mode 
    if "_layers" in k:
        new[k.replace("._layers", "")] = v 
    else:
        new[k] = v

paddle.save(new, save_path)