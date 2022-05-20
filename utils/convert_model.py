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

def convert_multi2single(final_path, save_path):
    old = paddle.load(final_path)
    new = dict()

    for k, v in old.items():
        # delete _layers in multi gpu mode 
        if "_layers" in k:
            new[k.replace("._layers", "")] = v 
        else:
            new[k] = v

    paddle.save(new, save_path)

def convert_single2multi(final_path, save_path):
    old = paddle.load(final_path)
    new = dict()

    for k, v in old.items():
        # delete _layers in multi gpu mode 
        if "_layers" in k:
            new[k.replace("model.blocks", "model._layers.blocks")] = v 
        else:
            new[k] = v

    paddle.save(new, save_path)

if __name__ == "__main__":
    final_path = "checkpoints/res48_prelu_rankloss_run5/final_multi.pdopt"
    save_path = "checkpoints/res48_prelu_rankloss_run5/final.pdopt"
    convert_multi2single(final_path, save_path)
