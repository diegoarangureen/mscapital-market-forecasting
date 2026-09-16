import torch, os, glob
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))
print('input:', glob.glob('/kaggle/input/*'))
print('dataset files:', glob.glob('/kaggle/input/mscapital-matrices/*')[:10])
open('/kaggle/working/smoke_out.txt','w').write('gpu='+str(torch.cuda.is_available()))
print('SMOKE_DONE')
