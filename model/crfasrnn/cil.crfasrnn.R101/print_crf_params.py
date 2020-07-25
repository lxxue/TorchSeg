import torch
from network import CrfRnnNet

if __name__ == '__main__':
	model = CrfRnnNet(2,n_iter=0)
	ptr_model_pth = "log/snapshot/epoch-last.pth"
	ptr_dict = torch.load(ptr_model_pth, map_location='cpu')['model']
	model.load_state_dict(ptr_dict)

	for name, param in model.crfrnn.named_parameters():
	    if param.requires_grad:
	        print(name)
	        print(param.data)